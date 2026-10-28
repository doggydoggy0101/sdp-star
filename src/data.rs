//! Input validation and the per-clique constraint storage.

use std::collections::BTreeMap;

use nalgebra::{DMatrix, DVector};

use crate::linalg::pk;
use crate::{Clique, Error, StarSdp};

pub(crate) fn validate(p: &StarSdp) -> Result<(), Error> {
    let bad = |msg: String| Err(Error::InvalidInput(msg));
    if p.hub == 0 {
        return bad("hub size must be at least 1".into());
    }
    if p.cliques.is_empty() {
        return bad("no cliques".into());
    }
    let check = |what: &str, t: &[(usize, usize, f64)], dim: usize| -> Result<(), Error> {
        for &(r, c, v) in t {
            if r < c {
                return bad(format!(
                    "{what}: ({r}, {c}) is above the diagonal; give the lower triangle"
                ));
            }
            if r >= dim {
                return bad(format!("{what}: index {r} out of range (size {dim})"));
            }
            if !v.is_finite() {
                return bad(format!("{what}: non-finite value at ({r}, {c})"));
            }
        }
        Ok(())
    };
    for (i, cl) in p.cliques.iter().enumerate() {
        if cl.dim < p.hub {
            return bad(format!(
                "clique {i}: size {} smaller than the hub {}",
                cl.dim, p.hub
            ));
        }
        check(&format!("clique {i} objective"), &cl.objective, cl.dim)?;
        for (j, c) in cl.constraints.iter().enumerate() {
            check(&format!("clique {i} constraint {j}"), &c.matrix, cl.dim)?;
            if !c.rhs.is_finite() {
                return bad(format!(
                    "clique {i} constraint {j}: non-finite right-hand side"
                ));
            }
        }
    }
    for (k, c) in p.hub_constraints.iter().enumerate() {
        check(&format!("hub constraint {k}"), &c.matrix, p.hub)?;
        if !c.rhs.is_finite() {
            return bad(format!("hub constraint {k}: non-finite right-hand side"));
        }
    }
    Ok(())
}

/// Lower-triangle entry (a ≥ b) of a symmetric coefficient matrix.
#[derive(Clone, Copy)]
pub(crate) struct Entry {
    pub a: usize,
    pub b: usize,
    pub full: usize, // (a, b) in a column-major d × d
    pub tr: usize,   // (b, a)
    pub pk: usize,   // (a, b) in packed lower storage
    pub v: f64,      // A[a, b]
    pub w: f64,      // v doubled off the diagonal: <A, X> = Σ w X[a, b]
}

pub(crate) fn entries(d: usize, lower: &[(usize, usize, f64)]) -> Vec<Entry> {
    let mut acc: BTreeMap<(usize, usize), f64> = BTreeMap::new();
    for &(r, c, v) in lower {
        *acc.entry((c, r)).or_insert(0.0) += v; // key (b, a): column-major order
    }
    acc.into_iter()
        .filter(|&(_, v)| v != 0.0)
        .map(|((b, a), v)| Entry {
            a,
            b,
            full: b * d + a,
            tr: a * d + b,
            pk: pk(d, a, b),
            v,
            w: if a == b { v } else { 2.0 * v },
        })
        .collect()
}

/// Static data of one clique: rows are its local constraints, then the hub selectors
/// H_k(X) = X[a, b] for the hub pairs (a ≥ b).
pub(crate) struct Data {
    pub d: usize,
    pub ml: usize,
    pub c: DMatrix<f64>,
    pub b: DVector<f64>,
    pub rows: Vec<Vec<Entry>>,
    /// input index of each local row
    pub orig: Vec<usize>,
    /// local rows removed as dependent: input index, entries, rhs
    pub removed: Vec<(usize, Vec<Entry>, f64)>,
}

impl Data {
    pub fn new(cl: &Clique, hub_pairs: &[(usize, usize)]) -> Self {
        let d = cl.dim;
        let mut c = DMatrix::zeros(d, d);
        for &(r, k, v) in &cl.objective {
            c[(r, k)] += v;
            if r != k {
                c[(k, r)] += v;
            }
        }
        let mut rows: Vec<Vec<Entry>> = cl
            .constraints
            .iter()
            .map(|a| entries(d, &a.matrix))
            .collect();
        for &(a, b) in hub_pairs {
            // E_ab = (e_a e_b^T + e_b e_a^T) / 2, so <E_ab, X> = X[a, b]
            let v = if a == b { 1.0 } else { 0.5 };
            rows.push(entries(d, &[(a, b, v)]));
        }
        Data {
            d,
            ml: cl.constraints.len(),
            c,
            b: DVector::from_iterator(cl.constraints.len(), cl.constraints.iter().map(|a| a.rhs)),
            rows,
            orig: (0..cl.constraints.len()).collect(),
            removed: vec![],
        }
    }

    pub fn m(&self) -> usize {
        self.rows.len()
    }

    /// Remove the given local rows (positions in the current list), keeping them for residuals.
    pub fn remove_local(&mut self, drop: &[usize]) {
        let mut keep_rows = vec![];
        let mut keep_b = vec![];
        let mut keep_orig = vec![];
        for (k, row) in self.rows.drain(..).enumerate() {
            if k < self.ml && drop.contains(&k) {
                self.removed.push((self.orig[k], row, self.b[k]));
            } else {
                if k < self.ml {
                    keep_b.push(self.b[k]);
                    keep_orig.push(self.orig[k]);
                }
                keep_rows.push(row);
            }
        }
        self.rows = keep_rows;
        self.ml = keep_b.len();
        self.b = DVector::from_vec(keep_b);
        self.orig = keep_orig;
    }

    /// [<G_k, X>]_k, X symmetric.
    pub fn apply(&self, x: &DMatrix<f64>) -> DVector<f64> {
        let xs = x.as_slice();
        DVector::from_iterator(
            self.m(),
            self.rows
                .iter()
                .map(|row| row.iter().map(|e| e.w * xs[e.full]).sum::<f64>()),
        )
    }

    /// Σ ||<A_k, X> - b_k||² over the removed rows.
    pub fn removed_residual2(&self, x: &DMatrix<f64>) -> f64 {
        let xs = x.as_slice();
        self.removed
            .iter()
            .map(|(_, row, b)| (b - row.iter().map(|e| e.w * xs[e.full]).sum::<f64>()).powi(2))
            .sum()
    }

    /// Σ_k w_k G_k.
    pub fn adjoint(&self, w: &DVector<f64>) -> DMatrix<f64> {
        let mut out = DMatrix::zeros(self.d, self.d);
        let os = out.as_mut_slice();
        for (row, &wk) in self.rows.iter().zip(w.iter()) {
            for e in row {
                os[e.full] += wk * e.v;
                if e.full != e.tr {
                    os[e.tr] += wk * e.v;
                }
            }
        }
        out
    }

    /// [G_k p]_k as the columns of a d × m matrix.
    pub fn times_vec(&self, p: &DVector<f64>) -> DMatrix<f64> {
        let mut out = DMatrix::zeros(self.d, self.m());
        for (k, row) in self.rows.iter().enumerate() {
            let mut col = out.column_mut(k);
            for e in row {
                col[e.a] += e.v * p[e.b];
                if e.a != e.b {
                    col[e.b] += e.v * p[e.a];
                }
            }
        }
        out
    }

    /// Lower triangle of G (W ⊗ W) G^T, i.e. <G_j, W G_k W>, rows local then hub.
    /// z is packed scratch for one W G_j W.
    pub fn schur(&self, w: &DMatrix<f64>, m: &mut DMatrix<f64>, z: &mut [f64]) {
        let d = self.d;
        let ws = w.as_slice();
        let n = self.m();
        for j in 0..n {
            // z = W G_j W (packed lower): G_j = Σ v (e_a e_b^T + e_b e_a^T), halved on the diagonal
            z.fill(0.0);
            for e in &self.rows[j] {
                let wa = &ws[e.a * d..(e.a + 1) * d];
                let wb = &ws[e.b * d..(e.b + 1) * d];
                let mut off = 0;
                for c in 0..d {
                    let zc = &mut z[off..off + d - c];
                    if e.a == e.b {
                        let s = e.v * wa[c];
                        for (zi, &wi) in zc.iter_mut().zip(&wa[c..]) {
                            *zi += s * wi;
                        }
                    } else {
                        let (s1, s2) = (e.v * wb[c], e.v * wa[c]);
                        for ((zi, &wai), &wbi) in zc.iter_mut().zip(&wa[c..]).zip(&wb[c..]) {
                            *zi += s1 * wai + s2 * wbi;
                        }
                    }
                    off += d - c;
                }
            }
            for k in j..n {
                m[(k, j)] = self.rows[k].iter().map(|e| e.w * z[e.pk]).sum();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{hub_pairs, Constraint};

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5
    }

    fn random_clique(d: usize, m: usize, seed: &mut u64) -> Clique {
        let constraints = (0..m)
            .map(|_| {
                let matrix = (0..4)
                    .map(|_| {
                        let a = ((lcg(seed) + 0.5) * d as f64) as usize % d;
                        let b = ((lcg(seed) + 0.5) * d as f64) as usize % d;
                        (a.max(b), a.min(b), lcg(seed))
                    })
                    .collect();
                Constraint { matrix, rhs: 0.0 }
            })
            .collect();
        Clique {
            dim: d,
            objective: vec![],
            constraints,
        }
    }

    fn sym(d: usize, seed: &mut u64) -> DMatrix<f64> {
        let a = DMatrix::from_fn(d, d, |_, _| lcg(seed));
        &a + a.transpose()
    }

    #[test]
    fn adjoint_is_the_transpose_of_apply() {
        let mut seed = 7;
        let dt = Data::new(&random_clique(9, 12, &mut seed), &hub_pairs(4));
        let x = sym(9, &mut seed);
        let w = DVector::from_fn(dt.m(), |_, _| lcg(&mut seed));
        let lhs = dt.adjoint(&w).dot(&x);
        let rhs = w.dot(&dt.apply(&x));
        assert!((lhs - rhs).abs() < 1e-12 * (1.0 + lhs.abs()));
    }

    #[test]
    fn schur_matches_brute_force() {
        let mut seed = 11;
        let dt = Data::new(&random_clique(8, 10, &mut seed), &hub_pairs(3));
        let a = DMatrix::from_fn(8, 8, |_, _| lcg(&mut seed));
        let w = &a * a.transpose() + DMatrix::identity(8, 8);
        let n = dt.m();
        let mut m = DMatrix::zeros(n, n);
        let mut z = vec![0.0; 8 * 9 / 2];
        dt.schur(&w, &mut m, &mut z);
        for j in 0..n {
            let gj = dt.adjoint(&DVector::from_fn(n, |r, _| (r == j) as u8 as f64));
            let wgw = &w * gj * &w;
            for k in j..n {
                let gk = dt.adjoint(&DVector::from_fn(n, |r, _| (r == k) as u8 as f64));
                let brute = gk.dot(&wgw);
                assert!((m[(k, j)] - brute).abs() < 1e-10 * (1.0 + brute.abs()));
            }
        }
    }
}
