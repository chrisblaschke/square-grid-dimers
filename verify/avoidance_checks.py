#!/usr/bin/env -S uv run --no-project --with numpy --with scipy python
"""End-to-end numerical sanity check of paper Section 6 (not part of the proof).

For several balanced edge partitions and profiles: run the selection lemma
literally (trim, mixed vertices, rows/columns, residue class mod 3, forbidden
edge per selected vertex), build H via the exact image formula, and compare
  - t with L/(3 sqrt6),
  - tr H / t with the proven p(L) and max row sum of |H|^2 with the proven R(L),
  - det(I - eta H) (must be real, > 0) with the Carleman bound,
  - the exact avoidance probability det(I - H) = Pr(M cap F = empty)
    with the proven bound exp(-c(L) L).
CPU only; OPENBLAS_NUM_THREADS=1.
"""
import json
import math

import numpy as np

from kernel_bounds import torus_g, image_kernel
from constants import bounds


def grid_edges(L):
    E = []
    for x2 in range(1, L + 1):
        for x1 in range(1, L + 1):
            if x1 < L:
                E.append(((x1, x2), (x1 + 1, x2)))
            if x2 < L:
                E.append(((x1, x2), (x1, x2 + 1)))
    return E


def partitions(L, rng):
    E = grid_edges(L)
    out = {}
    out['all-horizontal'] = {e for e in E if e[0][1] == e[1][1]}
    out['left-half'] = {e for e in E if min(e[0][0], e[1][0]) <= L // 2}
    out['diagonal'] = {e for e in E if e[0][0] + e[0][1] <= L}
    out['random-hash'] = {e for e in E if rng.random() < 0.5}
    b = 4
    out['checker-4'] = {e for e in E if ((e[0][0] - 1) // b + (e[0][1] - 1) // b) % 2 == 0}
    out['row-stripes'] = {e for e in E if e[0][1] % 2 == 1}
    # comb: vertical teeth of width 2 occupying the left 2/3, plus nothing else
    out['comb'] = {e for e in E if e[0][0] <= 2 * L // 3 and ((e[0][0] - 1) // 2) % 2 == 0}
    m = len(E)
    res = {}
    for k, Aset in out.items():
        frac = len(Aset) / m
        if 1 / 3 <= frac <= 2 / 3:
            res[k] = Aset
    return E, res


def profile_of(Aset, M):
    S = set()
    for e in M:
        if e in Aset:
            S.update(e)
    return S


def brick(L):
    return [((2 * k - 1, r), (2 * k, r)) for r in range(1, L + 1) for k in range(1, L // 2 + 1)]


def select(L, E, Aset, S, rng):
    a = L // 32
    lo, hi = a + 1, L - a
    inQ = lambda v: lo <= v[0] <= hi and lo <= v[1] <= hi
    QE = [e for e in E if inQ(e[0]) and inQ(e[1])]
    inc = {}
    for e in QE:
        for v in e:
            inc.setdefault(v, []).append(e)
    mixed = [v for v, es in inc.items() if any(e in Aset for e in es) and any(e not in Aset for e in es)]
    rows = {}
    cols = {}
    for v in mixed:
        rows.setdefault(v[1], []).append(v)
        cols.setdefault(v[0], []).append(v)
    use_rows = len(rows) >= len(cols)
    lines = rows if use_rows else cols
    span = max(len(rows), len(cols))
    classes = [sorted(k for k in lines if k % 3 == r) for r in range(3)]
    chosen = max(classes, key=len)
    F = []
    for k in chosen:
        v = lines[k][int(rng.integers(len(lines[k])))]
        want_B = v in S
        cands = [e for e in inc[v] if (e not in Aset) == want_B]
        F.append(cands[int(rng.integers(len(cands)))])
    return F, span, use_rows


def H_matrix(g, L, F):
    N = L + 1
    bs, ws, ks = [], [], []
    for (u, v) in F:
        b, w = (u, v) if (u[0] + u[1]) % 2 == 0 else (v, u)
        bs.append(b); ws.append(w); ks.append(1 if b[1] == w[1] else 1j)
    W1 = np.array([w[0] for w in ws])[:, None]; W2 = np.array([w[1] for w in ws])[:, None]
    B1 = np.array([b[0] for b in bs])[None, :]; B2 = np.array([b[1] for b in bs])[None, :]
    return image_kernel(g, N, W1, W2, B1, B2) * np.array(ks)[None, :], bs, ws


def main():
    rng = np.random.default_rng(7)
    out = []
    for L in (300, 600, 1000):
        g = torus_g(L + 1)
        bd = bounds(L)
        E, parts = partitions(L, rng)
        for name, Aset in parts.items():
            for prof in ('brick', 'random'):
                S = profile_of(Aset, brick(L)) if prof == 'brick' else \
                    {(x1, x2) for x1 in range(1, L + 1) for x2 in range(1, L + 1) if rng.random() < 0.5}
                F, span, use_rows = select(L, E, Aset, S, rng)
                t = len(F)
                # separation check
                coord = 1 if use_rows else 0
                ends = [sorted({u[coord], v[coord]}) for (u, v) in F]
                for i in range(t):
                    for j in range(i + 1, min(t, i + 4)):
                        gap = min(abs(x - y) for x in ends[i] for y in ends[j])
                        assert gap >= 3 * (j - i) - 2
                H, bs, ws = H_matrix(g, L, F)
                tr = np.trace(H)
                assert abs(tr.imag) < 1e-9
                diag = H.diagonal().real
                rows2 = (np.abs(H) ** 2).sum(axis=1)
                fro2 = float(rows2.sum())
                eta_star = min(1.0, tr.real / fro2)
                sgn, ld = np.linalg.slogdet(np.eye(t) - eta_star * H)
                assert abs(sgn - 1) < 1e-8, sgn
                carleman = -eta_star * tr.real + eta_star ** 2 * fro2 / 2
                assert ld <= carleman + 1e-9
                sgn1, ld1 = np.linalg.slogdet(np.eye(t) - H)  # exact avoidance prob
                assert abs(sgn1 - 1) < 1e-6
                rec = dict(L=L, partition=name, profile=prof, mixed_span=span,
                           span_lower_bound=L / math.sqrt(6), t=t, t_lower_bound=L / (3 * math.sqrt(6)),
                           min_marginal=float(diag.min()), proven_p=bd['p'],
                           max_row_sum_sq=float(rows2.max()), proven_R=bd['R'],
                           log_det_tilt=float(ld), carleman_bound=float(carleman),
                           log_avoid_exact=float(ld1),
                           proven_log_bound=-(bd['c'] or 0) * L)
                assert span >= L / math.sqrt(6)
                assert t >= L / (3 * math.sqrt(6))
                if bd['c']:
                    assert diag.min() >= bd['p'] - 1e-12
                    assert rows2.max() <= bd['R'] + 1e-12
                    assert ld1 <= rec['proven_log_bound'] + 1e-9
                out.append(rec)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
