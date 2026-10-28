//! SDP*: an interior-point solver for semidefinite programs on a chordal star graph.
//!
//! The problem is
//!
//! ```text
//!   min  Σ_i <C_i, X_i>
//!   s.t. <A_ij, X_i> = b_ij       for every clique i and constraint j   (local)
//!        X_i[..h, ..h] = U        for every clique i                    (hub consensus)
//!        <F_k, U> = f_k           for every hub constraint k
//!        X_i ⪰ 0,  U free
//! ```
//!
//! Every block (a *clique*) has its own size and constraints, and its leading `h × h`
//! submatrix (the *hub*) must equal a shared matrix `U`: the cliques of a chordal star graph,
//! which overlap only on the hub.
//!
//! The solver is a primal-dual interior-point method with Nesterov-Todd scaling and
//! Mehrotra's predictor-corrector. Its linear algebra never couples the cliques beyond the
//! hub: each clique factors its own Schur complement (in parallel), and the cliques meet
//! only in one small system of size `h(h+1)/2` plus the number of hub constraints.
//!
//! # Conventions
//!
//! * Symmetric matrices are given as lower-triangle triplets `(row, col, value)` with
//!   `row >= col`; each triplet sets both `A[row, col]` and `A[col, row]`. Repeated positions
//!   are summed. Upper-triangle triplets are rejected, so a full symmetric list cannot be
//!   counted twice by mistake.
//! * Indices are 0-based. The hub is indices `0..h` of every clique.
//! * `U` is returned through its lower triangle, column by column: `(0,0), (1,0), …,
//!   (h-1,0), (1,1), …, (h-1,h-1)`, see [`hub_pairs`].
//!
//! # Example
//!
//! Three cliques `[[1, t_i], [t_i, c_i]]` share the hub entry `U = 1`; minimizing `Σ c_i`
//! gives `Σ t_i²`, at the rank-one blocks `(1, t_i)(1, t_i)^T`.
//!
//! ```
//! use sdp_star::{solve, Clique, Constraint, Options, StarSdp};
//!
//! let ts = [0.5, -1.0, 2.0];
//! let problem = StarSdp {
//!     hub: 1,
//!     cliques: ts
//!         .iter()
//!         .map(|&t| Clique {
//!             dim: 2,
//!             objective: vec![(1, 1, 1.0)], // c_i
//!             // <A, X> = X[1,0] needs A[1,0] = A[0,1] = 1/2
//!             constraints: vec![Constraint { matrix: vec![(1, 0, 0.5)], rhs: t }],
//!         })
//!         .collect(),
//!     hub_constraints: vec![Constraint { matrix: vec![(0, 0, 1.0)], rhs: 1.0 }], // U = 1
//! };
//! let sol = solve(&problem, &Options::default()).unwrap();
//! assert!(sol.converged());
//! assert!((sol.dual_objective - 5.25).abs() < 1e-6);
//! ```
//!
//! # Accuracy and certificates
//!
//! The method is tuned for relaxations whose solution has rank one, solved to certify a
//! nonconvex problem. Such SDPs are typically degenerate (many more constraints than a
//! rank-one solution has degrees of freedom), and the primal side of any Schur-complement
//! method loses accuracy near the end: expect primal infeasibility around `1e-6`. The dual
//! side stays accurate (about `1e-12`), so [`Solution::dual_objective`] is the value to use
//! as a lower bound: by weak duality it bounds the optimum for every exactly dual-feasible
//! point, and the returned dual variables let a user check that.
//!
//! Linearly dependent local constraints are detected at the first iteration and removed;
//! they still count in the reported primal infeasibility, so an inconsistent right-hand
//! side shows up there.

mod data;
mod ipm;
mod linalg;

pub use nalgebra;
use nalgebra::{DMatrix, DVector};

/// A symmetric matrix as lower-triangle triplets `(row, col, value)`, `row >= col`.
pub type SymTriplets = Vec<(usize, usize, f64)>;

/// One linear equality `<A, X> = rhs`.
#[derive(Clone, Debug, Default)]
pub struct Constraint {
    pub matrix: SymTriplets,
    pub rhs: f64,
}

/// One PSD block `X_i` of side `dim`; its leading `hub × hub` submatrix is the hub.
#[derive(Clone, Debug, Default)]
pub struct Clique {
    pub dim: usize,
    pub objective: SymTriplets,
    pub constraints: Vec<Constraint>,
}

/// The whole problem: the hub size, the cliques, and the constraints on the shared hub
/// `U` (their matrices are `hub × hub`).
#[derive(Clone, Debug, Default)]
pub struct StarSdp {
    pub hub: usize,
    pub cliques: Vec<Clique>,
    pub hub_constraints: Vec<Constraint>,
}

/// Solver parameters. The defaults are what the method reliably reaches on degenerate,
/// rank-one relaxations; see the crate documentation.
#[derive(Clone, Debug)]
pub struct Options {
    pub max_iter: usize,
    /// relative primal infeasibility
    pub tol_primal: f64,
    /// relative dual infeasibility; also required for [`Status::Converged`]
    pub tol_dual: f64,
    /// relative duality gap, plus `tol_gap_abs` absolute (an optimum of 0 has no relative gap)
    pub tol_gap: f64,
    pub tol_gap_abs: f64,
    /// the best iterate counts as converged within this factor of the primal and gap tolerances
    pub accept: f64,
    /// stop after this many iterations without a better iterate, once near the end
    pub stall: usize,
    /// worker threads for the per-clique work; `None` uses all available cores
    pub threads: Option<usize>,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            max_iter: 40,
            tol_primal: 1e-6,
            tol_dual: 1e-8,
            tol_gap: 1e-6,
            tol_gap_abs: 1e-9,
            accept: 10.0,
            stall: 2,
            threads: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// The dual tolerance holds and the primal and gap tolerances hold within `accept`.
    Converged,
    /// Stopped (stall or iteration limit) before converging; the best iterate is returned.
    NotConverged,
    /// A block left the cone or the hub system could not be solved; the best iterate is returned.
    NumericalFailure,
}

/// The best iterate found, primal and dual.
#[derive(Clone, Debug)]
pub struct Solution {
    pub status: Status,
    /// primal blocks `X_i`
    pub x: Vec<DMatrix<f64>>,
    /// dual slacks `S_i = C_i - Σ_j y_ij A_ij - Σ_k z_ik E_k`
    pub s: Vec<DMatrix<f64>>,
    /// multipliers of each clique's local constraints, in input order (0 for removed ones)
    pub y: Vec<DVector<f64>>,
    /// multipliers of each clique's hub consensus, in [`hub_pairs`] order
    pub z: Vec<DVector<f64>>,
    /// the shared hub `U`, lower triangle in [`hub_pairs`] order
    pub u: DVector<f64>,
    /// multipliers of the hub constraints
    pub v: DVector<f64>,
    pub primal_objective: f64,
    /// `Σ_ij b_ij y_ij + Σ_k f_k v_k`: a lower bound on the optimum (weak duality)
    pub dual_objective: f64,
    pub primal_infeasibility: f64,
    pub dual_infeasibility: f64,
    pub iterations: usize,
    /// per clique, the local constraints removed as linearly dependent on the others
    pub removed: Vec<Vec<usize>>,
}

impl Solution {
    pub fn converged(&self) -> bool {
        self.status == Status::Converged
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    InvalidInput(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
        }
    }
}

impl std::error::Error for Error {}

/// The hub pairs `(a, b)`, `a >= b`, in the order used for `u`, `z` and hub selectors:
/// column by column of the lower triangle.
pub fn hub_pairs(hub: usize) -> Vec<(usize, usize)> {
    (0..hub)
        .flat_map(|b| (b..hub).map(move |a| (a, b)))
        .collect()
}

/// Solve the problem. Returns an error only for malformed input; numerical trouble is
/// reported through [`Solution::status`].
pub fn solve(problem: &StarSdp, options: &Options) -> Result<Solution, Error> {
    data::validate(problem)?;
    Ok(ipm::solve(problem, options))
}
