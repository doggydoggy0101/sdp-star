//! Dense kernels: packed indexing, Householder reflection, blocked Cholesky, step bounds.

use nalgebra::{DMatrix, DVector};

/// Packed lower index of (a, b), a ≥ b, column-major, side d.
pub(crate) fn pk(d: usize, a: usize, b: usize) -> usize {
    b * (2 * d - b + 1) / 2 + a - b
}

/// Householder reflection H = I - 2 v v^T / (v^T v), applied in place.
pub(crate) fn reflect(hv: &DVector<f64>, x: &mut [f64]) {
    let c = 2.0 * hv.iter().zip(x.iter()).map(|(a, b)| a * b).sum::<f64>() / hv.norm_squared();
    for (xi, &vi) in x.iter_mut().zip(hv.iter()) {
        *xi -= c * vi;
    }
}

/// In-place Cholesky of the lower triangle of `a`, blocked: each panel of NB columns is
/// factored left-looking, then the trailing lower part is updated through gemm.
/// ! A pivot below `drop_tol` of its original diagonal is a dependent row: it is set huge,
/// so solves return ~0 in that component. Returns the dropped rows.
pub(crate) fn cholesky(a: &mut DMatrix<f64>, drop_tol: f64) -> Vec<usize> {
    const NB: usize = 48;
    let n = a.nrows();
    let diag: Vec<f64> = (0..n).map(|i| a[(i, i)]).collect();
    let mut dropped = vec![];
    let mut k0 = 0;
    while k0 < n {
        let k1 = (k0 + NB).min(n);
        // panel A[k0.., k0..k1]: earlier panels are already applied
        {
            let s = a.as_mut_slice();
            for j in k0..k1 {
                let (prev, cur) = s.split_at_mut(j * n);
                let colj = &mut cur[j..n];
                for k in k0..j {
                    let ljk = prev[k * n + j];
                    if ljk != 0.0 {
                        for (x, &y) in colj.iter_mut().zip(&prev[k * n + j..(k + 1) * n]) {
                            *x -= ljk * y;
                        }
                    }
                }
                let mut piv = colj[0];
                // NaN counts as dropped too
                if piv.is_nan() || piv <= drop_tol * diag[j] {
                    piv = 1e128;
                    dropped.push(j);
                }
                let ljj = piv.sqrt();
                colj[0] = ljj;
                let inv = 1.0 / ljj;
                for x in &mut colj[1..] {
                    *x *= inv;
                }
            }
        }
        if k1 < n {
            // A[k1.., k1..] -= L21 L21^T, lower column blocks only
            let l21 = a.view((k1, k0), (n - k1, k1 - k0)).into_owned();
            let l21t = l21.transpose();
            let mut j0 = k1;
            while j0 < n {
                let j1 = (j0 + NB).min(n);
                a.view_mut((j0, j0), (n - j0, j1 - j0)).gemm(
                    -1.0,
                    &l21.rows(j0 - k1, n - j0),
                    &l21t.columns(j0 - k1, j1 - j0),
                    1.0,
                );
                j0 = j1;
            }
        }
        k0 = k1;
    }
    dropped
}

/// M^{-1} r from the factor in the lower triangle of l.
pub(crate) fn chol_solve(l: &DMatrix<f64>, r: &DVector<f64>) -> DVector<f64> {
    let mut x = r.clone();
    l.solve_lower_triangular_mut(&mut x);
    l.tr_solve_lower_triangular_mut(&mut x);
    x
}

/// Largest α with Λ + α D ⪰ 0 (∞ if D ⪰ 0).
pub(crate) fn max_step(lam: &DVector<f64>, dmat: &DMatrix<f64>) -> f64 {
    let d = lam.len();
    let t = DMatrix::from_fn(d, d, |a, b| dmat[(a, b)] / (lam[a] * lam[b]).sqrt());
    let ev_min = t.symmetric_eigenvalues().min();
    if ev_min >= 0.0 {
        f64::INFINITY
    } else {
        -1.0 / ev_min
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spd(n: usize, seed: u64) -> DMatrix<f64> {
        let mut s = seed;
        let a = DMatrix::from_fn(n, n, |_, _| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        });
        &a * a.transpose() + DMatrix::identity(n, n)
    }

    #[test]
    fn blocked_cholesky_matches_definition() {
        for n in [5, 48, 49, 130] {
            let a = spd(n, n as u64);
            let mut l = a.clone();
            assert!(cholesky(&mut l, 0.0).is_empty());
            let low = l.lower_triangle();
            assert!((&low * low.transpose() - &a).norm() < 1e-10 * a.norm());
        }
    }

    #[test]
    fn cholesky_drops_dependent_rows() {
        // rows 0,1 independent, row 2 = row 0 + row 1 in a Gram matrix
        let b = DMatrix::from_row_slice(3, 4, &[1., 0., 2., 0., 0., 1., 0., 3., 1., 1., 2., 3.]);
        let mut g = &b * b.transpose();
        assert_eq!(cholesky(&mut g, 1e-14), vec![2]);
    }

    #[test]
    fn householder_maps_to_last_axis() {
        let h = DVector::from_vec(vec![0.3, -0.4, 0.5, 0.7]).normalize();
        let mut v = h.clone();
        v[3] += 1.0; // sign of h[3] is positive
        let mut x = h.as_slice().to_vec();
        reflect(&v, &mut x);
        assert!(x[..3].iter().all(|t| t.abs() < 1e-14) && (x[3] + 1.0).abs() < 1e-14);
    }
}
