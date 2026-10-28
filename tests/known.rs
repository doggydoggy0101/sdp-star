//! End-to-end tests on SDPs with known optima, plus input validation.

use sdp_star::nalgebra::{DMatrix, DVector, Matrix2, Vector2};
use sdp_star::{hub_pairs, solve, Clique, Constraint, Error, Options, StarSdp, Status};

fn con(matrix: Vec<(usize, usize, f64)>, rhs: f64) -> Constraint {
    Constraint { matrix, rhs }
}

/// The dual point is feasible: S_i = C_i - Σ y A - Σ z E is PSD, matches the returned S_i,
/// and the hub multipliers balance, Σ_i z_i = F^T v.
fn check_dual(p: &StarSdp, sol: &sdp_star::Solution) {
    let pairs = hub_pairs(p.hub);
    let mut zsum = DVector::zeros(pairs.len());
    for (i, cl) in p.cliques.iter().enumerate() {
        let d = cl.dim;
        let mut s = DMatrix::<f64>::zeros(d, d);
        let mut add = |t: &[(usize, usize, f64)], w: f64| {
            for &(a, b, v) in t {
                s[(a, b)] += w * v;
                if a != b {
                    s[(b, a)] += w * v;
                }
            }
        };
        add(&cl.objective, 1.0);
        for (j, c) in cl.constraints.iter().enumerate() {
            add(&c.matrix, -sol.y[i][j]);
        }
        for (k, &(a, b)) in pairs.iter().enumerate() {
            add(&[(a, b, if a == b { 1.0 } else { 0.5 })], -sol.z[i][k]);
        }
        assert!((&s - &sol.s[i]).norm() < 1e-7, "clique {i}: S mismatch");
        assert!(
            s.symmetric_eigenvalues().min() > -1e-8,
            "clique {i}: S not PSD"
        );
        zsum += &sol.z[i];
    }
    let mut ftv = DVector::zeros(pairs.len());
    for (k, c) in p.hub_constraints.iter().enumerate() {
        for &(a, b, v) in &c.matrix {
            let l = pairs.iter().position(|&q| q == (a, b)).unwrap();
            ftv[l] += sol.v[k] * if a == b { v } else { 2.0 * v };
        }
    }
    assert!((zsum - ftv).norm() < 1e-7, "hub multipliers do not balance");
}

#[test]
fn tiny_rank_one() {
    // min X00 + X11 s.t. X10 = 1, X ⪰ 0: optimum 2 at X = (1,1)(1,1)^T
    let p = StarSdp {
        hub: 1,
        cliques: vec![Clique {
            dim: 2,
            objective: vec![(0, 0, 1.0), (1, 1, 1.0)],
            constraints: vec![con(vec![(1, 0, 0.5)], 1.0)],
        }],
        hub_constraints: vec![],
    };
    let sol = solve(&p, &Options::default()).unwrap();
    assert_eq!(sol.status, Status::Converged);
    assert!(
        (sol.dual_objective - 2.0).abs() < 1e-6,
        "{}",
        sol.dual_objective
    );
    assert!((&sol.x[0] - DMatrix::from_element(2, 2, 1.0)).norm() < 1e-4);
    check_dual(&p, &sol);
}

/// N cliques [[1, t_i], [t_i, c_i]] sharing the hub entry U = 1: min Σ c_i is Σ t_i², at
/// rank-one blocks (1, t_i)(1, t_i)^T.
fn scalar_hub(ts: &[f64]) -> StarSdp {
    StarSdp {
        hub: 1,
        cliques: ts
            .iter()
            .map(|&t| Clique {
                dim: 2,
                objective: vec![(1, 1, 1.0)],
                constraints: vec![con(vec![(1, 0, 0.5)], t)],
            })
            .collect(),
        hub_constraints: vec![con(vec![(0, 0, 1.0)], 1.0)],
    }
}

#[test]
fn shared_hub_rank_one() {
    let ts = [0.5, -1.0, 2.0, 0.25, 1.5, -0.75];
    let p = scalar_hub(&ts);
    let sol = solve(&p, &Options::default()).unwrap();
    let opt: f64 = ts.iter().map(|t| t * t).sum();
    assert_eq!(sol.status, Status::Converged);
    assert!(
        (sol.dual_objective - opt).abs() < 1e-6 * opt,
        "{} vs {opt}",
        sol.dual_objective
    );
    assert!((sol.u[0] - 1.0).abs() < 1e-5);
    for (x, &t) in sol.x.iter().zip(&ts) {
        assert!((x[(1, 0)] - t).abs() < 1e-5 && (x[(1, 1)] - t * t).abs() < 1e-4);
    }
    check_dual(&p, &sol);
}

#[test]
fn full_rank_hub() {
    // hub U = [[1, .5], [.5, 1]] fixed by hub constraints; clique i is [[U, w_i], [w_i^T, c_i]]
    // with w_i fixed: min Σ c_i is Σ w_i^T U^{-1} w_i (not rank one)
    let ws = [
        Vector2::new(1.0, 0.0),
        Vector2::new(0.3, -0.8),
        Vector2::new(-1.0, 1.0),
    ];
    let p = StarSdp {
        hub: 2,
        cliques: ws
            .iter()
            .map(|w| Clique {
                dim: 3,
                objective: vec![(2, 2, 1.0)],
                constraints: vec![con(vec![(2, 0, 0.5)], w[0]), con(vec![(2, 1, 0.5)], w[1])],
            })
            .collect(),
        hub_constraints: vec![
            con(vec![(0, 0, 1.0)], 1.0),
            con(vec![(1, 1, 1.0)], 1.0),
            con(vec![(1, 0, 0.5)], 0.5),
        ],
    };
    let sol = solve(&p, &Options::default()).unwrap();
    let ui = Matrix2::new(1.0, 0.5, 0.5, 1.0).try_inverse().unwrap();
    let opt: f64 = ws.iter().map(|w| w.dot(&(ui * w))).sum();
    assert_eq!(sol.status, Status::Converged);
    assert!(
        (sol.dual_objective - opt).abs() < 1e-6 * opt,
        "{} vs {opt}",
        sol.dual_objective
    );
    check_dual(&p, &sol);
}

#[test]
fn dependent_constraints_are_removed() {
    let ts = [0.5, -1.0, 2.0];
    let mut p = scalar_hub(&ts);
    // a consistent duplicate (twice the constraint, twice the rhs) in clique 1
    p.cliques[1]
        .constraints
        .push(con(vec![(1, 0, 1.0)], 2.0 * ts[1]));
    let sol = solve(&p, &Options::default()).unwrap();
    let opt: f64 = ts.iter().map(|t| t * t).sum();
    assert_eq!(sol.status, Status::Converged);
    assert_eq!(sol.removed, vec![vec![], vec![1], vec![]]);
    assert_eq!(sol.y[1][1], 0.0);
    assert!((sol.dual_objective - opt).abs() < 1e-6 * opt);
    check_dual(&p, &sol);
}

#[test]
fn inconsistent_duplicate_is_not_converged() {
    let mut p = scalar_hub(&[0.5, -1.0, 2.0]);
    // the same constraint again with a different rhs: infeasible
    p.cliques[0].constraints.push(con(vec![(1, 0, 0.5)], 0.9));
    let sol = solve(&p, &Options::default()).unwrap();
    assert_ne!(sol.status, Status::Converged);
    assert!(sol.primal_infeasibility > 1e-3);
}

#[test]
fn rejects_malformed_input() {
    let mut p = scalar_hub(&[1.0]);
    p.cliques[0].constraints[0].matrix = vec![(0, 1, 0.5)];
    assert!(matches!(
        solve(&p, &Options::default()),
        Err(Error::InvalidInput(_))
    ));
    let mut p = scalar_hub(&[1.0]);
    p.cliques[0].objective = vec![(2, 2, 1.0)];
    assert!(matches!(
        solve(&p, &Options::default()),
        Err(Error::InvalidInput(_))
    ));
    let mut p = scalar_hub(&[1.0]);
    p.hub = 3;
    assert!(matches!(
        solve(&p, &Options::default()),
        Err(Error::InvalidInput(_))
    ));
}

#[test]
fn single_thread_matches_parallel() {
    let p = scalar_hub(&[0.5, -1.0, 2.0, 0.25, 1.5, -0.75, 0.1, 0.9]);
    let par = solve(&p, &Options::default()).unwrap();
    let one = solve(
        &p,
        &Options {
            threads: Some(1),
            ..Options::default()
        },
    )
    .unwrap();
    assert_eq!(par.dual_objective, one.dual_objective);
}
