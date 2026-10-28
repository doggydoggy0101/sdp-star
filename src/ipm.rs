//! Primal-dual interior-point method, NT scaling, Mehrotra predictor-corrector, infeasible
//! start. Each clique factors its own Schur complement M_i = G_i (W_i ⊗ W_i) G_i^T over its
//! rows G_i = [A_i; H] (H selects the hub), and the cliques couple only through U: a step is
//! a parallel factor pass, one small solve on the hub, and a parallel back-substitution.
//!
//! ! Near a rank-1 optimum W has one eigenvalue ~1/√μ and the rest ~√μ, so the hub rows of
//! M carry a rank-1 term ~1/μ that Cholesky would cancel down to round-off. W is kept split
//! as σ p p^T + W_S, and the hub rows are rotated so that term lives in the last row only:
//! M is then assembled without cancellation, and the 1/μ pivot is taken last.

use std::thread;

use nalgebra::{DMatrix, DVector};

use crate::data::Data;
use crate::linalg::{chol_solve, cholesky, max_step, pk, reflect};
use crate::{hub_pairs, Options, Solution, StarSdp, Status};

/// ! A Cholesky pivot below this fraction of its diagonal marks a dependent row.
const DROP_TOL: f64 = 1e-14;

/// Iterate and per-iteration work of one clique.
struct Block {
    dt: Data,
    x: DMatrix<f64>,
    s: DMatrix<f64>,
    yz: DVector<f64>, // multipliers of [A; H]
    // residuals at the current point
    r: DVector<f64>,  // [b; u] - G(X)
    rd: DMatrix<f64>, // C - G^*(yz) - S
    cx: f64,
    by: f64,
    xs: f64,
    rp2: f64,
    rh2: f64,
    rd2: f64,
    // NT scaling: G^T S G = G^{-1} X G^{-T} = diag(lam), W = G G^T = sig p p^T + ws
    g: DMatrix<f64>,
    ginv: DMatrix<f64>,
    lam: DVector<f64>,
    sig: f64,
    p: DVector<f64>,
    ws: DMatrix<f64>,
    wrdw: DMatrix<f64>,
    // hub rotation (shared by all cliques) and the Schur factor in rotated rows
    hv: DVector<f64>,
    l: DMatrix<f64>,
    khub: DMatrix<f64>, // E^T M^{-1} E, rotated
    zbuf: Vec<f64>,
    first: bool,
    // right-hand sides (rotated): g = r - G(Q), ΔX = Q + W G^*(Δyz) W
    q: DMatrix<f64>,
    gv: DVector<f64>,
    t: DVector<f64>,
    // corrector, split as Q = q0 + σμ S^{-1}, g = g0 - σμ g1
    q0: DMatrix<f64>,
    sinv: DMatrix<f64>,
    g0: DVector<f64>,
    g1: DVector<f64>,
    t0: DVector<f64>,
    t1: DVector<f64>,
    // direction, its scaled pair G^{-1} ΔX G^{-T}, G^T ΔS G, and step bounds
    dx: DMatrix<f64>,
    ds: DMatrix<f64>,
    dyz: DVector<f64>,
    dx_sc: DMatrix<f64>,
    ds_sc: DMatrix<f64>,
    ax: f64,
    as_: f64,
    xds: f64,
    dxs: f64,
    dxds: f64,
    ok: bool,
}

impl Block {
    fn new(dt: Data, p: usize) -> Self {
        let (d, m) = (dt.d, dt.m());
        let zd = || DMatrix::zeros(d, d);
        let zv = |k| DVector::zeros(k);
        Block {
            x: DMatrix::identity(d, d),
            s: DMatrix::identity(d, d),
            yz: zv(m),
            r: zv(m),
            rd: zd(),
            cx: 0.0,
            by: 0.0,
            xs: 0.0,
            rp2: 0.0,
            rh2: 0.0,
            rd2: 0.0,
            g: zd(),
            ginv: zd(),
            lam: zv(d),
            sig: 0.0,
            p: zv(d),
            ws: zd(),
            wrdw: zd(),
            hv: zv(p),
            l: DMatrix::zeros(m, m),
            khub: DMatrix::zeros(p, p),
            zbuf: vec![0.0; d * (d + 1) / 2],
            first: true,
            q: zd(),
            gv: zv(m),
            t: zv(p),
            q0: zd(),
            sinv: zd(),
            g0: zv(m),
            g1: zv(m),
            t0: zv(p),
            t1: zv(p),
            dx: zd(),
            ds: zd(),
            dyz: zv(m),
            dx_sc: zd(),
            ds_sc: zd(),
            dt,
            ax: 0.0,
            as_: 0.0,
            xds: 0.0,
            dxs: 0.0,
            dxds: 0.0,
            ok: true,
        }
    }

    fn hub(&self, v: &DVector<f64>) -> DVector<f64> {
        v.rows(self.dt.ml, v.len() - self.dt.ml).into_owned()
    }

    /// Apply the hub rotation to the hub part of a row vector (it is its own inverse).
    fn rot(&self, mut v: DVector<f64>) -> DVector<f64> {
        let ml = self.dt.ml;
        reflect(&self.hv, &mut v.as_mut_slice()[ml..]);
        v
    }

    /// W Y W with W = sig p p^T + ws, term by term: no 1/μ part is cancelled.
    fn wyw(&self, y: &DMatrix<f64>) -> DMatrix<f64> {
        let yp = y * &self.p;
        let z = &self.ws * &yp;
        let pyp = self.p.dot(&yp);
        let mut out = &self.ws * y * &self.ws;
        out += &self.p * self.p.transpose() * (self.sig * self.sig * pyp);
        out += (&self.p * z.transpose() + &z * self.p.transpose()) * self.sig;
        out
    }

    /// Residuals and objective terms at the current point; removed dependent rows count
    /// in the primal residual.
    fn residuals(&mut self, u: &DVector<f64>) {
        let dt = &self.dt;
        let ml = dt.ml;
        let mut rhs = DVector::zeros(dt.m());
        rhs.rows_mut(0, ml).copy_from(&dt.b);
        rhs.rows_mut(ml, u.len()).copy_from(u);
        self.r = rhs - dt.apply(&self.x);
        self.rd = &dt.c - dt.adjoint(&self.yz) - &self.s;
        self.cx = dt.c.dot(&self.x);
        self.by = dt.b.dot(&self.yz.rows(0, ml));
        self.xs = self.x.dot(&self.s);
        self.rp2 = self.r.rows(0, ml).norm_squared() + dt.removed_residual2(&self.x);
        self.rh2 = self.r.rows(ml, u.len()).norm_squared();
        self.rd2 = self.rd.norm_squared();
    }

    /// M = σ² e e^T + 2σ C^T W_S C + <G_j, W_S G_k W_S> in rotated rows, e_k = p^T G_k p,
    /// C = [G_k p]; then its Cholesky factor. Returns the dropped pivots.
    fn factor(&mut self, drop_tol: f64) -> Vec<usize> {
        let dt = &self.dt;
        let (m, ml) = (dt.m(), dt.ml);
        if self.l.nrows() != m {
            self.l = DMatrix::zeros(m, m);
        }
        dt.schur(&self.ws, &mut self.l, &mut self.zbuf);
        for j in 0..m {
            for k in j + 1..m {
                self.l[(j, k)] = self.l[(k, j)];
            }
        }
        // rotate the hub rows and columns of the W_S term: L <- R L R, R = diag(I, H),
        // H = I - hn v v^T applied as rank-1 updates
        let (hv, hn) = (&self.hv, 2.0 / self.hv.norm_squared());
        let vr = (hv.transpose() * self.l.rows(ml, m - ml)).transpose();
        self.l.rows_mut(ml, m - ml).ger(-hn, hv, &vr, 1.0);
        let rv = self.l.columns(ml, m - ml) * hv;
        self.l.columns_mut(ml, m - ml).ger(-hn, &rv, hv, 1.0);
        let mut c = dt.times_vec(&self.p);
        let cv = c.columns(ml, m - ml) * hv;
        c.columns_mut(ml, m - ml).ger(-hn, &cv, hv, 1.0);
        let e = c.tr_mul(&self.p);
        let wsc = &self.ws * &c;
        self.l.gemm(2.0 * self.sig, &c.transpose(), &wsc, 1.0);
        self.l.ger(self.sig * self.sig, &e, &e, 1.0);
        cholesky(&mut self.l, drop_tol)
    }

    /// Take the previous step, then residuals, NT scaling, Schur factor, and the
    /// predictor right-hand side (R = -X).
    fn phase_a(&mut self, ap: f64, ad: f64, u: &DVector<f64>, hv: &DVector<f64>) {
        let d = self.dt.d;
        if ap > 0.0 {
            self.x += &self.dx * ap;
        }
        if ad > 0.0 {
            self.s += &self.ds * ad;
            self.yz += &self.dyz * ad;
        }
        self.hv.copy_from(hv);

        // NT scaling from X = Lx Lx^T, Lx^T S Lx = U D U^T: G = Lx U D^{-1/4}
        let Some(chol) = self.x.clone().cholesky() else {
            self.ok = false;
            return;
        };
        let lx = chol.unpack();
        let t = lx.tr_mul(&(&self.s * &lx));
        let se = t.symmetric_eigen();
        let dvals = se.eigenvalues.map(|v| v.max(1e-300));
        self.lam = dvals.map(f64::sqrt);
        let mut g = &lx * &se.eigenvectors;
        for (k, mut col) in g.column_iter_mut().enumerate() {
            col /= dvals[k].powf(0.25);
        }
        let Some(linv) = lx.solve_lower_triangular(&DMatrix::identity(d, d)) else {
            self.ok = false;
            return;
        };
        let mut ginv = se.eigenvectors.tr_mul(&linv);
        for (k, mut row) in ginv.row_iter_mut().enumerate() {
            row *= dvals[k].powf(0.25);
        }
        // W = G G^T = sig p p^T + ws: q the top right singular vector of G, a = G q,
        // ws = G' G'^T with G' = G - a q^T (exact for any unit q, accurate for the top one)
        let gtg = g.tr_mul(&g).symmetric_eigen();
        let top = gtg.eigenvalues.imax();
        let qv = gtg.eigenvectors.column(top).into_owned();
        let a = &g * &qv;
        self.sig = a.norm_squared();
        self.p = &a / a.norm();
        let gp = &g - &a * qv.transpose();
        self.ws = &gp * gp.transpose();
        self.g = g;
        self.ginv = ginv;

        loop {
            self.residuals(u);
            let dropped = self.factor(if self.first { DROP_TOL } else { 0.0 });
            // ! the first point has W ∝ I, so M is the Gram matrix of the rows: a dropped
            // local row is a combination of the others. Removed once, before any multiplier
            // is nonzero; it keeps counting in the primal residual.
            let local: Vec<usize> = dropped.into_iter().filter(|&k| k < self.dt.ml).collect();
            if !self.first || local.is_empty() {
                break;
            }
            self.dt.remove_local(&local);
            self.yz = DVector::zeros(self.dt.m());
        }
        self.first = false;
        let ml = self.dt.ml;
        let p = self.hv.len();
        let l22 = self.l.view((ml, ml), (p, p)).lower_triangle();
        let Some(l22inv) = l22.solve_lower_triangular(&DMatrix::identity(p, p)) else {
            self.ok = false;
            return;
        };
        self.khub = l22inv.tr_mul(&l22inv);

        self.wrdw = self.wyw(&self.rd);
        self.q = -&self.x - &self.wrdw;
        self.gv = self.rot(&self.r - self.dt.apply(&self.q));
        self.t = self.hub(&chol_solve(&self.l, &self.gv));
    }

    /// Direction for the rotated Δu given the current (q, gv): fills dx, ds, dyz and the
    /// step bounds.
    fn direction(&mut self, du: &DVector<f64>) {
        let ml = self.dt.ml;
        let mut rhs = self.gv.clone();
        let mut hub = rhs.rows_mut(ml, du.len());
        hub += du;
        self.dyz = self.rot(chol_solve(&self.l, &rhs));
        let gs = self.dt.adjoint(&self.dyz);
        self.ds = &self.rd - &gs;
        self.dx = &self.q + self.wyw(&gs);
        self.dx_sc = &self.ginv * &self.dx * self.ginv.transpose();
        self.ds_sc = self.g.tr_mul(&self.ds) * &self.g;
        self.ax = max_step(&self.lam, &self.dx_sc);
        self.as_ = max_step(&self.lam, &self.ds_sc);
    }

    /// Predictor direction, then the corrector's right-hand sides split in σμ.
    fn phase_b(&mut self, du: &DVector<f64>) {
        self.direction(du);
        self.xds = self.x.dot(&self.ds);
        self.dxs = self.dx.dot(&self.s);
        self.dxds = self.dx.dot(&self.ds);
        // scaled complementarity Λ∘Y = -Λ² - (DX DS + DS DX)/2, Y_ab = 2 Rc_ab / (λ_a + λ_b)
        let (dxs, dss) = (&self.dx_sc, &self.ds_sc);
        let mut rc = -(dxs * dss + dss * dxs) * 0.5;
        let lam = &self.lam;
        for k in 0..lam.len() {
            rc[(k, k)] -= lam[k] * lam[k];
        }
        let y0 = DMatrix::from_fn(lam.len(), lam.len(), |a, b| {
            2.0 * rc[(a, b)] / (lam[a] + lam[b])
        });
        let r0 = &self.g * y0 * self.g.transpose();
        self.q0 = r0 - &self.wrdw;
        let mut gl = self.g.clone();
        for (k, mut col) in gl.column_iter_mut().enumerate() {
            col /= lam[k];
        }
        self.sinv = &gl * self.g.transpose(); // S^{-1} = G Λ^{-1} G^T
        self.g0 = self.rot(&self.r - self.dt.apply(&self.q0));
        self.g1 = self.rot(self.dt.apply(&self.sinv));
        self.t0 = self.hub(&chol_solve(&self.l, &self.g0));
        self.t1 = self.hub(&chol_solve(&self.l, &self.g1));
    }

    /// Corrector direction for σμ and the rotated Δu, then this clique's steps: gamma times
    /// the distance to the boundary, shrunk until X + αΔX and S + αΔS factor. ! The
    /// eigenvalue bound alone can overshoot by round-off near the optimum; any smaller step
    /// stays inside, so the min over cliques does.
    fn phase_c(&mut self, du: &DVector<f64>, smu: f64, gamma: f64) {
        self.gv = &self.g0 - &self.g1 * smu;
        self.q = &self.q0 + &self.sinv * smu;
        self.direction(du);
        let safe = |m: &DMatrix<f64>, dm: &DMatrix<f64>, mut a: f64| {
            while a > 1e-12 && (m + dm * a).cholesky().is_none() {
                a *= 0.8;
            }
            a
        };
        self.ax = safe(&self.x, &self.dx, (gamma * self.ax).min(1.0));
        self.as_ = safe(&self.s, &self.ds, (gamma * self.as_).min(1.0));
    }
}

/// Run `f` on every block, the blocks split evenly over `threads` workers.
fn par<T: Send>(items: &mut [T], threads: usize, f: impl Fn(&mut T) + Sync) {
    let threads = threads.min(items.len()).max(1);
    if threads == 1 {
        items.iter_mut().for_each(&f);
        return;
    }
    let chunk = items.len().div_ceil(threads);
    thread::scope(|sc| {
        let mut chunks = items.chunks_mut(chunk);
        let first = chunks.next().unwrap();
        for c in chunks {
            let f = &f;
            sc.spawn(move || c.iter_mut().for_each(f));
        }
        first.iter_mut().for_each(&f);
    });
}

/// The best iterate seen so far.
struct Snapshot {
    merit: f64,
    accept: bool,
    pobj: f64,
    dobj: f64,
    pinf: f64,
    dinf: f64,
    iteration: usize,
    x: Vec<DMatrix<f64>>,
    s: Vec<DMatrix<f64>>,
    yz: Vec<DVector<f64>>,
    u: DVector<f64>,
    v: DVector<f64>,
}

pub(crate) fn solve(prob: &StarSdp, opt: &Options) -> Solution {
    let h = prob.hub;
    let pairs = hub_pairs(h);
    let p = pairs.len();
    let threads = opt
        .threads
        .unwrap_or_else(|| thread::available_parallelism().map_or(1, |n| n.get()));
    let data: Vec<Data> = prob.cliques.iter().map(|c| Data::new(c, &pairs)).collect();

    // F (q × p): <F_k, U> = Σ_l F[k, l] u_l; an off-diagonal entry of U appears twice
    let nq = prob.hub_constraints.len();
    let mut fm = DMatrix::zeros(nq, p);
    for (k, con) in prob.hub_constraints.iter().enumerate() {
        for &(a, b, v) in &con.matrix {
            fm[(k, pk(h, a, b))] += if a == b { v } else { 2.0 * v };
        }
    }
    let f = DVector::from_iterator(nq, prob.hub_constraints.iter().map(|c| c.rhs));
    let n_tot: f64 = data.iter().map(|dt| dt.d as f64).sum();
    let b2: f64 = f.norm_squared() + data.iter().map(|dt| dt.b.norm_squared()).sum::<f64>();
    let n_cl = data.len() as f64;
    let norm_c = 1.0 + data.iter().map(|dt| dt.c.norm()).sum::<f64>();

    // start at X_i = S_i = I, w = 0, u = vec(I), v = 0
    let mut blocks: Vec<Block> = data.into_iter().map(|dt| Block::new(dt, p)).collect();
    let mut u = DVector::from_iterator(p, pairs.iter().map(|&(a, b)| (a == b) as u8 as f64));
    let mut v = DVector::zeros(nq);

    let (mut ap, mut ad) = (0.0, 0.0);
    let mut best: Option<Snapshot> = None;
    let mut since_best = 0;
    let mut failed = false;
    let mut iterations = 0;
    for it in 0..opt.max_iter {
        iterations = it + 1;
        // hub rotation: maps the hub direction of U's top eigenvector, (x_a x_b)_k, onto
        // the last hub row
        let um = DMatrix::from_fn(h, h, |a, b| u[pk(h, a.max(b), a.min(b))]);
        let ue = um.symmetric_eigen();
        let xt = ue.eigenvectors.column(ue.eigenvalues.imax()).into_owned();
        let mut hv = DVector::from_iterator(p, pairs.iter().map(|&(a, b)| xt[a] * xt[b]));
        hv /= hv.norm();
        hv[p - 1] += if hv[p - 1] >= 0.0 { 1.0 } else { -1.0 };

        par(&mut blocks, threads, |b| b.phase_a(ap, ad, &u, &hv));
        if blocks.iter().any(|b| !b.ok) {
            failed = true;
            break;
        }
        let rf = &f - &fm * &u;
        let mut ru = fm.tr_mul(&v);
        for b in &blocks {
            ru -= b.hub(&b.yz);
        }
        let pobj = blocks.iter().map(|b| b.cx).sum::<f64>();
        let dobj = blocks.iter().map(|b| b.by).sum::<f64>() + f.dot(&v);
        let xs: f64 = blocks.iter().map(|b| b.xs).sum();
        let mu = xs / n_tot;
        // relative to the whole right-hand side [b_i; u]_i, f: u sits in every clique's
        // rows, so the residual and its scale both grow like √N
        let norm_b = 1.0 + (b2 + n_cl * u.norm_squared()).sqrt();
        let pinf =
            (blocks.iter().map(|b| b.rp2 + b.rh2).sum::<f64>() + rf.norm_squared()).sqrt() / norm_b;
        let dinf = (blocks.iter().map(|b| b.rd2).sum::<f64>() + ru.norm_squared()).sqrt() / norm_c;
        // gap ≤ tol_gap |obj| + tol_gap_abs, as one ratio
        let gap =
            (pobj - dobj).abs() / (pobj.abs().max(dobj.abs()) + opt.tol_gap_abs / opt.tol_gap);
        let merit = (pinf / opt.tol_primal)
            .max(dinf / opt.tol_dual)
            .max(gap / opt.tol_gap);
        let accept =
            (pinf / opt.tol_primal).max(gap / opt.tol_gap) <= opt.accept && dinf <= opt.tol_dual;
        if best.as_ref().is_none_or(|s| merit < s.merit) {
            best = Some(Snapshot {
                merit,
                accept,
                pobj,
                dobj,
                pinf,
                dinf,
                iteration: iterations,
                x: blocks.iter().map(|b| b.x.clone()).collect(),
                s: blocks.iter().map(|b| b.s.clone()).collect(),
                yz: blocks.iter().map(|b| b.yz.clone()).collect(),
                u: u.clone(),
                v: v.clone(),
            });
            since_best = 0;
        } else if best.as_ref().is_some_and(|s| s.merit < 1e3) {
            // only near the end: early merit is not monotone
            since_best += 1;
        }
        if merit <= 1.0 || since_best >= opt.stall {
            break;
        }

        // hub system in rotated coordinates (u = H u'):
        // K Δu' - F'^T Δv = H ru - Σ t_i,  F' Δu' = rf,  F' = F H
        let mut fr = fm.clone();
        let fv = &fm * &hv;
        fr.ger(-2.0 / hv.norm_squared(), &fv, &hv, 1.0);
        let mut rur = ru.clone();
        reflect(&hv, rur.as_mut_slice());
        // ! K spans ~μ to ~1/μ near the optimum: LU on the whole saddle system
        let k: DMatrix<f64> = blocks
            .iter()
            .fold(DMatrix::zeros(p, p), |acc, b| acc + &b.khub);
        let mut kkt = DMatrix::zeros(p + nq, p + nq);
        kkt.view_mut((0, 0), (p, p)).copy_from(&k);
        kkt.view_mut((0, p), (p, nq)).copy_from(&(-fr.transpose()));
        kkt.view_mut((p, 0), (nq, p)).copy_from(&fr);
        let lu = kkt.lu();
        let hub_solve = |t: &DVector<f64>| {
            let mut rhs = DVector::zeros(p + nq);
            rhs.rows_mut(0, p).copy_from(&(&rur - t));
            rhs.rows_mut(p, nq).copy_from(&rf);
            lu.solve(&rhs)
                .map(|x| (x.rows(0, p).into_owned(), x.rows(p, nq).into_owned()))
        };

        // predictor
        let t: DVector<f64> = blocks.iter().fold(DVector::zeros(p), |acc, b| acc + &b.t);
        let Some((du, _)) = hub_solve(&t) else {
            failed = true;
            break;
        };
        par(&mut blocks, threads, |b| b.phase_b(&du));
        let ap_aff = blocks.iter().map(|b| b.ax).fold(1.0, f64::min);
        let ad_aff = blocks.iter().map(|b| b.as_).fold(1.0, f64::min);
        let mu_aff: f64 = blocks
            .iter()
            .map(|b| b.xs + ad_aff * b.xds + ap_aff * b.dxs + ap_aff * ad_aff * b.dxds)
            .sum::<f64>()
            / n_tot;
        let sigma = (mu_aff / mu).max(0.0).powi(3).min(1.0);

        // corrector
        let t0: DVector<f64> = blocks.iter().fold(DVector::zeros(p), |acc, b| acc + &b.t0);
        let t1: DVector<f64> = blocks.iter().fold(DVector::zeros(p), |acc, b| acc + &b.t1);
        let Some((du, dv)) = hub_solve(&(t0 - t1 * (sigma * mu))) else {
            failed = true;
            break;
        };
        let gamma = 0.9 + 0.09 * ap_aff.min(ad_aff);
        par(&mut blocks, threads, |b| b.phase_c(&du, sigma * mu, gamma));
        ap = blocks.iter().map(|b| b.ax).fold(1.0, f64::min);
        ad = blocks.iter().map(|b| b.as_).fold(1.0, f64::min);
        let mut dur = du;
        reflect(&hv, dur.as_mut_slice());
        u += &dur * ap;
        v += &dv * ad;
    }

    let status = match &best {
        Some(s) if s.accept => Status::Converged,
        _ if failed => Status::NumericalFailure,
        _ => Status::NotConverged,
    };
    let removed: Vec<Vec<usize>> = blocks
        .iter()
        .map(|b| b.dt.removed.iter().map(|r| r.0).collect())
        .collect();
    let Some(best) = best else {
        // failed before the first iterate was evaluated
        let n = blocks.len();
        return Solution {
            status,
            x: blocks.iter().map(|b| b.x.clone()).collect(),
            s: blocks.iter().map(|b| b.s.clone()).collect(),
            y: prob
                .cliques
                .iter()
                .map(|c| DVector::zeros(c.constraints.len()))
                .collect(),
            z: vec![DVector::zeros(p); n],
            u,
            v,
            primal_objective: f64::NAN,
            dual_objective: f64::NAN,
            primal_infeasibility: f64::NAN,
            dual_infeasibility: f64::NAN,
            iterations,
            removed,
        };
    };
    // multipliers back to input order: removed rows get 0
    let y = blocks
        .iter()
        .zip(&best.yz)
        .zip(&prob.cliques)
        .map(|((b, yz), cl)| {
            let mut y = DVector::zeros(cl.constraints.len());
            for (k, &o) in b.dt.orig.iter().enumerate() {
                y[o] = yz[k];
            }
            y
        })
        .collect();
    let z = blocks
        .iter()
        .zip(&best.yz)
        .map(|(b, yz)| yz.rows(b.dt.ml, p).into_owned())
        .collect();
    Solution {
        status,
        x: best.x,
        s: best.s,
        y,
        z,
        u: best.u,
        v: best.v,
        primal_objective: best.pobj,
        dual_objective: best.dobj,
        primal_infeasibility: best.pinf,
        dual_infeasibility: best.dinf,
        iterations: best.iteration,
        removed,
    }
}
