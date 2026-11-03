#!/usr/bin/env -S uv run --no-project --with numpy --with scipy python
"""Numerical confirmation of the explicit kernel constants in paper Section 5.

CPU only; run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1.

Conventions: vertex x=(x1,x2) in {1..L}^2, x1 horizontal (weight 1), x2 vertical
(weight i).  N=L+1 (odd).  Torus side M=2N.  T = A(x)I + i I(x)A.
  g(z) = M^-2 sum_{j,k} e^{i pi (j z1 + k z2)/N} / (2cos(pi j/N) + 2i cos(pi k/N))
Proven (Lemma 3.2):  |g(z)| <= min(1/sqrt2, A/||z||_T),  A=(1+sqrt2)/(2 asinh 1).
Proven (Cor. 3.4):   |T_D^-1(x,y)| <= 4 min(1/sqrt2, A/||x-y||_inf)  (all x,y),
                     |T_D^-1(x,y) - g(x-y)| <= 3 min(1/sqrt2, A/max(2D, ||x-y||_inf)),
                     |P(e) - 1/4| <= 3 min(1/sqrt2, A/(2D(e))).
"""
import json
import math
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

KAPPA = math.asinh(1.0)
A_CONST = (1 + math.sqrt(2)) / (2 * KAPPA)
U0 = 1 / math.sqrt(2)


def torus_g(N):
    M = 2 * N
    th = np.pi * np.arange(M) / N
    S = 2 * np.cos(th)[:, None] + 2j * np.cos(th)[None, :]
    return np.fft.ifft2(1.0 / S)  # g[z1, z2]


def torus_norm(N):
    M = 2 * N
    d = np.minimum(np.arange(M), M - np.arange(M))
    return np.maximum(d[:, None], d[None, :]), d


def check_1d(rng):
    """Cycle resolvent (A_cyc - zeta)^-1, zeta = 2ic, vs closed form and bound."""
    worst_formula = 0.0
    worst_ratio = 0.0
    for N in (5, 11, 31, 101):
        M = 2 * N
        Acyc = np.zeros((M, M))
        for x in range(M):
            Acyc[x, (x + 1) % M] = Acyc[x, (x - 1) % M] = 1
        for j in range(M):
            c = math.cos(math.pi * j / N)
            zeta = 2j * c
            G = np.linalg.inv(Acyc - zeta * np.eye(M))[:, 0]
            r = math.sqrt(1 + c * c) - abs(c)
            assert abs(r - math.exp(-math.asinh(abs(c)))) < 1e-14
            mu = -1j * math.copysign(1, c) * r
            assert abs(mu + 1 / mu - zeta) < 1e-12
            nmax = int(60 / max(1e-9, -math.log(r) * M)) + 3
            closed = np.zeros(M, complex)
            for x in range(M):
                closed[x] = sum(mu ** (abs(x + 2 * n * N) + 1) / (mu * mu - 1)
                                for n in range(-nmax, nmax + 1))
            worst_formula = max(worst_formula, float(np.abs(closed - G).max()))
            d = np.minimum(np.arange(M), M - np.arange(M))
            bound = 0.5 * (r ** d + r ** (M - d)) / (1 - r ** M)
            worst_ratio = max(worst_ratio, float((np.abs(G) / bound).max()))
    return dict(max_abs_error_closed_form=worst_formula,
                max_ratio_to_bound=worst_ratio)


def check_sum_bound():
    """The intermediate 1D sum of Lemma 3.3 and its two analytic ingredients.
    S(d) = (1/M) sum_j cosh(s_j(N-d)) / (2 cosh(s_j) sinh(s_j N)),  s_j = asinh|cos(pi j/N)|.
    Claims: d*S(d) <= A for 1<=d<=N; the k-th smallest s_j (k=0,1,..) is >= kappa*(2*floor(k/4)+1)/N."""
    worst = 0.0
    worst_sorted = 0.0
    for N in list(range(3, 202, 2)) + [301, 513, 1001]:
        M = 2 * N
        s = np.arcsinh(np.abs(np.cos(np.pi * np.arange(M) / N)))
        ss = np.sort(s)
        lower = KAPPA * (2 * (np.arange(M) // 4) + 1) / N
        worst_sorted = max(worst_sorted, float((lower / ss).max()))
        d = np.arange(1, N + 1)[None, :]
        sc = s[:, None]
        ratio = (np.exp(-sc * d) + np.exp(-sc * (2 * N - d))) / (1 - np.exp(-2 * sc * N))
        S = (ratio / (2 * np.cosh(sc))).sum(axis=0) / M
        worst = max(worst, float((S * d[0]).max()))
    return dict(max_d_times_S=worst, proven_A=A_CONST, max_ratio_lower_to_sorted_s=worst_sorted)


def check_torus(L):
    N = L + 1
    g = torus_g(N)
    nT, d = torus_norm(N)
    M = 2 * N
    # residual of D g = delta
    Dg = (np.roll(g, -1, 0) + np.roll(g, 1, 0) + 1j * (np.roll(g, -1, 1) + np.roll(g, 1, 1)))
    delta = np.zeros_like(g)
    delta[0, 0] = 1
    res = float(np.abs(Dg - delta).max())
    bound = np.where(nT > 0, np.minimum(U0, A_CONST / np.maximum(nT, 1)), U0)
    ratio = float((np.abs(g) / bound).max())
    obsA = float((np.abs(g) * nT).max())
    return dict(L=L, residual=res, g_e1=complex(g[1, 0]), g_e2=complex(g[0, 1]),
                max_abs_g=float(np.abs(g).max()),
                observed_sup_norm_times_g=obsA, proven_A=A_CONST,
                max_ratio_to_proven_bound=ratio), g


def image_kernel(g, N, x1, x2, y1, y2):
    M = 2 * N
    return (g[(x1 - y1) % M, (x2 - y2) % M] - g[(x1 + y1) % M, (x2 - y2) % M]
            - g[(x1 - y1) % M, (x2 + y2) % M] + g[(x1 + y1) % M, (x2 + y2) % M])


def dirichlet_matrix(L):
    n = L * L
    idx = lambda x1, x2: (x1 - 1) + L * (x2 - 1)
    rows, cols, vals = [], [], []
    for x2 in range(1, L + 1):
        for x1 in range(1, L + 1):
            i = idx(x1, x2)
            for dx, dy, w in ((1, 0, 1), (-1, 0, 1), (0, 1, 1j), (0, -1, 1j)):
                u1, u2 = x1 + dx, x2 + dy
                if 1 <= u1 <= L and 1 <= u2 <= L:
                    rows.append(i); cols.append(idx(u1, u2)); vals.append(w)
    return sp.csc_matrix((vals, (rows, cols)), shape=(n, n), dtype=complex)


def check_dirichlet(L, g, nsrc, rng):
    N = L + 1
    t0 = time.time()
    T = dirichlet_matrix(L)
    lu = spla.splu(T)
    X1, X2 = np.meshgrid(np.arange(1, L + 1), np.arange(1, L + 1), indexing='xy')
    X1 = X1.ravel(); X2 = X2.ravel()  # index (x1-1) + L (x2-1)
    srcs = [(1, 1), (1, L), (L, 1), (L, L), (L // 2, L // 2), (1, L // 2), (L // 2, 2)]
    while len(srcs) < nsrc:
        srcs.append((int(rng.integers(1, L + 1)), int(rng.integers(1, L + 1))))
    worst_err = 0.0
    worst_ratio_global = 0.0
    obs_global = 0.0
    worst_ratio_image = 0.0
    for (y1, y2) in srcs:
        e = np.zeros(L * L, complex)
        e[(y1 - 1) + L * (y2 - 1)] = 1
        u = lu.solve(e)
        img = image_kernel(g, N, X1, X2, y1, y2)
        worst_err = max(worst_err, float(np.abs(u - img).max()))
        dinf = np.maximum(np.abs(X1 - y1), np.abs(X2 - y2))
        mask = dinf > 0
        gb = 4 * np.minimum(U0, A_CONST / np.maximum(dinf, 1))
        worst_ratio_global = max(worst_ratio_global, float((np.abs(u[mask]) / gb[mask]).max()))
        obs_global = max(obs_global, float((np.abs(u[mask]) * dinf[mask]).max()))
        # image remainder bound with boundary distance D
        D = np.minimum.reduce([X1, N - X1, X2, N - X2, np.full_like(X1, min(y1, N - y1, y2, N - y2))])
        direct = g[(X1 - y1) % (2 * N), (X2 - y2) % (2 * N)]
        rem = np.abs(u - direct)
        ib = 3 * np.minimum(U0, A_CONST / np.maximum(np.maximum(2 * D, dinf), 1))
        worst_ratio_image = max(worst_ratio_image, float((rem / ib).max()))
    return dict(L=L, sources=len(srcs), max_abs_error_image_vs_sparse_solve=worst_err,
                max_ratio_global_bound=worst_ratio_global,
                observed_sup_dist_times_Kinv=obs_global, proven_global_constant_4A=4 * A_CONST,
                max_ratio_image_remainder_bound=worst_ratio_image,
                seconds=round(time.time() - t0, 2))


def edge_marginals(L, g):
    """All edge marginals via the image formula. Returns arrays and checks."""
    N = L + 1
    out = []
    # horizontal edges (x1,x2)-(x1+1,x2); vertical (x1,x2)-(x1,x2+1)
    for kind in ('h', 'v'):
        if kind == 'h':
            a1, a2 = np.meshgrid(np.arange(1, L), np.arange(1, L + 1), indexing='ij')
            c1, c2 = a1 + 1, a2
            kap = 1
        else:
            a1, a2 = np.meshgrid(np.arange(1, L + 1), np.arange(1, L), indexing='ij')
            c1, c2 = a1, a2 + 1
            kap = 1j
        a1 = a1.ravel(); a2 = a2.ravel(); c1 = c1.ravel(); c2 = c2.ravel()
        black_a = (a1 + a2) % 2 == 0
        b1 = np.where(black_a, a1, c1); b2 = np.where(black_a, a2, c2)
        w1 = np.where(black_a, c1, a1); w2 = np.where(black_a, c2, a2)
        P = kap * image_kernel(g, N, w1, w2, b1, b2)
        D = np.minimum.reduce([a1, N - a1, a2, N - a2, c1, N - c1, c2, N - c2])
        out.append((a1, a2, c1, c2, P, D))
    return out


def check_marginals(L, g):
    parts = edge_marginals(L, g)
    cover = np.zeros((L + 2, L + 2))
    maxim = 0.0
    worst_ratio = 0.0
    obs = 0.0
    total = 0.0
    trimmed_min = 1.0
    a = L // 32
    for (a1, a2, c1, c2, P, D) in parts:
        maxim = max(maxim, float(np.abs(P.imag).max()))
        Pr = P.real
        total += Pr.sum()
        np.add.at(cover, (a1, a2), Pr)
        np.add.at(cover, (c1, c2), Pr)
        err = np.abs(Pr - 0.25)
        bnd = 3 * np.minimum(U0, A_CONST / (2 * D))
        worst_ratio = max(worst_ratio, float((err / bnd).max()))
        obs = max(obs, float((err * D).max()))
        if L >= 64:
            inQ = np.minimum.reduce([a1, a2, c1, c2]) >= a + 1
            inQ &= np.maximum.reduce([a1, a2, c1, c2]) <= L - a
            trimmed_min = min(trimmed_min, float(Pr[inQ].min()))
    cov = cover[1:L + 1, 1:L + 1]
    return dict(L=L, max_imag_part=maxim, sum_marginals_minus_L2_over_2=float(total - L * L / 2),
                max_vertex_cover_error=float(np.abs(cov - 1).max()),
                max_ratio_to_proven_error_bound=worst_ratio,
                observed_sup_D_times_err=obs, proven_3A_over_2=1.5 * A_CONST,
                min_marginal_in_trimmed_Q=trimmed_min if L >= 64 else None,
                proven_lower_bound_in_Q=0.25 - 48 * A_CONST / L)


def main():
    rng = np.random.default_rng(20261103)
    out = {'A': A_CONST, 'one_dimensional': check_1d(rng), 'sum_bound': check_sum_bound(),
           'torus': [], 'dirichlet': [], 'marginals': []}
    assert out['sum_bound']['max_d_times_S'] <= A_CONST
    assert out['sum_bound']['max_ratio_lower_to_sorted_s'] <= 1 + 1e-12
    gs = {}
    for L in (8, 16, 32, 64, 128, 256, 300, 400, 512):
        r, g = check_torus(L)
        out['torus'].append(r)
        gs[L] = g
        assert r['residual'] < 1e-10
        assert abs(r['g_e1'] - 0.25) < 1e-12 and abs(r['g_e2'] + 0.25j) < 1e-12
        assert r['max_ratio_to_proven_bound'] <= 1 + 1e-9
        m = check_marginals(L, g)
        out['marginals'].append(m)
        assert m['max_ratio_to_proven_error_bound'] <= 1 + 1e-9
        assert m['max_vertex_cover_error'] < 1e-9
    for L in (8, 16, 64, 128, 256, 300):
        r = check_dirichlet(L, gs[L], 24, rng)
        out['dirichlet'].append(r)
        assert r['max_abs_error_image_vs_sparse_solve'] < 1e-9
        assert r['max_ratio_global_bound'] <= 1 + 1e-9
        assert r['max_ratio_image_remainder_bound'] <= 1 + 1e-9
    for k in ('torus',):
        for r in out[k]:
            for key in ('g_e1', 'g_e2'):
                r[key] = [r[key].real, r[key].imag]
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
