An interior-point method for semidefinite programs (SDP) on a **chordal star graph**. Consider the SDP:

$$
\begin{aligned}
\min_{X_1,\dots,X_N,\,U}\quad & \sum_{i=1}^{N} \langle C_i, X_i \rangle \\
\text{s.t.}\quad & \langle A_{ij}, X_i \rangle = b_{ij}, && i = 1,\dots,N,\ \ j = 1,\dots,m_i \\
& [X_i]_{\mathrm{hub}} = U, && i = 1,\dots,N \\
& \langle F_k, U \rangle = f_k, && k = 1,\dots,q \\
& X_i \succeq 0, && i = 1,\dots,N
\end{aligned}
$$

Here $\langle A, X \rangle = \operatorname{tr}(AX)$, each clique $X_i \in \mathbb{S}^{d_i}$ has its own size and local constraints, $[X_i]_{\mathrm{hub}}$ is its leading $h \times h$ submatrix, and the shared hub $U \in \mathbb{S}^{h}$ is a free variable with its own constraints. Certifiable estimation problems have this structure when every measurement brings its own auxiliary variables around one shared unknown (a pose, for instance). 

> Experimental. Written for the Chordal-SDP and Reduced-SDP relaxations of [PnP*](https://github.com/doggydoggy0101/pnp-star).


## Usage

```rust
use sdp_star::{solve, Clique, Constraint, Options, StarSdp};

let problem = StarSdp {
    hub: 1,
    cliques: vec![Clique {
        dim: 2,
        objective: vec![(1, 1, 1.0)],
        constraints: vec![Constraint { matrix: vec![(1, 0, 0.5)], rhs: 2.0 }], // X[1,0] = 2
    }],
    hub_constraints: vec![Constraint { matrix: vec![(0, 0, 1.0)], rhs: 1.0 }], // U = 1
};
let sol = solve(&problem, &Options::default())?;
println!("{:?}, lower bound {}", sol.status, sol.dual_objective); // Converged, 4
```

Conventions:

- Symmetric matrices are lower-triangle triplets `(row, col, value)` with `row >= col`; each triplet sets both `A[row, col]` and `A[col, row]`, so `<A, X> = X[1,0]` is `(1, 0, 0.5)`. Upper-triangle triplets are rejected.
- Indices are 0-based, and the hub is indices `0..h` of every clique.
- `U` and the hub multipliers are ordered column by column of the lower triangle (`sdp_star::hub_pairs`).

The `Solution` holds the primal blocks, the dual slacks and all multipliers, the primal and dual objectives and infeasibilities, the iteration count, and the local constraints that were removed as linearly dependent.

## Accuracy

The method targets relaxations with redundant constraints (LICQ doesn't hold) such that the dual solutions are not unique. Near the optimum the primal side of any Schur-complement method then loses accuracy, and the default tolerances reflect what is reliably reachable:

| quantity | default tolerance |
|---|---|
| relative dual infeasibility | `1e-8` (required) |
| relative primal infeasibility | `1e-6` (accepted up to 10×) |
| duality gap | `1e-6` relative + `1e-9` absolute (accepted up to 10×) |

The dual side stays accurate (around `1e-12`), so use `dual_objective` as the lower bound of a certificate: by weak duality it bounds the optimum for every exactly dual-feasible point, and the returned dual variables let you verify feasibility yourself.

## Method

Primal-dual interior-point method with Nesterov-Todd scaling, Mehrotra's predictor-corrector and an infeasible start. Specific to this structure:

- each clique's Schur complement is factored on its own, and the hub coupling is eliminated exactly, leaving one small saddle-point system per step;
- the scaling matrix is split into its dominant rank-one part and the rest, and the hub rows are rotated so that the part of size `1/μ` sits in a single row, factored last, which keeps the factorization from cancelling it near the optimum;
- linearly dependent local constraints are detected at the first iteration and removed.
