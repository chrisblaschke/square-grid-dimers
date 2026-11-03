#!/usr/bin/env -S uv run --no-project python
"""Exact checks for paper Section 4 (Kasteleyn signs, generating circuit, tilted identity).

Coordinates: vertex (x1, x2), 1..L; x1 = column (horizontal), x2 = row.
Black = x1+x2 even. Complex Kasteleyn entries: horizontal 1, vertical i.
Real gauge K'(b,w) = (-i)^{x2(b)} i^{x2(w)} K(b,w):
  horizontal -> +1, vertical with white above black -> -1, white below -> +1.

Checks (pure Python integers, exact):
 1. every unit face of the real-gauged matrix has sign product -1;
 2. every perfect matching contributes the SAME sign to det K' (real gauge)
    and the SAME phase to det K (complex weights 1, i);
 3. det K'(z) = eps * sum_M prod_{e in M} z_e at random integer points z
    (including negative entries: it is a polynomial identity);
 4. tilted identity det(I - eta H) = E[(1-eta)^{|M cap F|}] for random F,
    eta = 2/5, exact rationals.
"""
import json
import random
from fractions import Fraction as Fr


def grid(L):
    V = [(x1, x2) for x2 in range(1, L + 1) for x1 in range(1, L + 1)]
    E = []
    for (x1, x2) in V:
        if x1 < L:
            E.append(((x1, x2), (x1 + 1, x2)))
        if x2 < L:
            E.append(((x1, x2), (x1, x2 + 1)))
    return V, E


def oriented(E):
    """Return list of (b, w, kind) with b black; kind 'h' or 'up'/'down' (white rel. black)."""
    out = []
    for u, v in E:
        b, w = (u, v) if (u[0] + u[1]) % 2 == 0 else (v, u)
        if b[1] == w[1]:
            kind = 'h'
        elif w[1] == b[1] + 1:
            kind = 'up'
        else:
            kind = 'down'
        out.append((b, w, kind))
    return out


def real_sign(kind):
    return {'h': 1, 'up': -1, 'down': 1}[kind]


def matchings(V, E):
    idx = {v: i for i, v in enumerate(V)}
    adj = [[] for _ in V]
    for k, (u, v) in enumerate(E):
        adj[idx[u]].append((idx[v], k))
        adj[idx[v]].append((idx[u], k))
    out = []

    def rec(left, chosen):
        if not left:
            out.append(tuple(chosen))
            return
        low = left & -left
        u = low.bit_length() - 1
        for v, k in adj[u]:
            if left >> v & 1:
                chosen.append(k)
                rec(left ^ low ^ (1 << v), chosen)
                chosen.pop()

    rec((1 << len(V)) - 1, [])
    return out


def perm_sign(p):
    n = len(p)
    seen = [False] * n
    s = 1
    for i in range(n):
        if not seen[i]:
            j, ln = i, 0
            while not seen[j]:
                seen[j] = True
                j = p[j]
                ln += 1
            if ln % 2 == 0:
                s = -s
    return s


def det_int(M):
    """Bareiss fraction-free determinant over Python ints."""
    A = [row[:] for row in M]
    n = len(A)
    sign, prev = 1, 1
    for k in range(n - 1):
        if A[k][k] == 0:
            sw = next((i for i in range(k + 1, n) if A[i][k] != 0), None)
            if sw is None:
                return 0
            A[k], A[sw] = A[sw], A[k]
            sign = -sign
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                A[i][j] = (A[i][j] * A[k][k] - A[i][k] * A[k][j]) // prev
        prev = A[k][k]
    return sign * A[n - 1][n - 1]


def det_frac(M):
    A = [[Fr(x) for x in row] for row in M]
    n = len(A)
    d = Fr(1)
    for c in range(n):
        p = next((r for r in range(c, n) if A[r][c] != 0), None)
        if p is None:
            return Fr(0)
        if p != c:
            A[c], A[p] = A[p], A[c]
            d = -d
        d *= A[c][c]
        for r in range(c + 1, n):
            f = A[r][c] / A[c][c]
            if f:
                for k in range(c, n):
                    A[r][k] -= f * A[c][k]
    return d


def inv_frac(M):
    n = len(M)
    R = [[Fr(x) for x in row] + [Fr(int(i == j)) for j in range(n)] for i, row in enumerate(M)]
    for c in range(n):
        p = next(r for r in range(c, n) if R[r][c] != 0)
        R[c], R[p] = R[p], R[c]
        piv = R[c][c]
        R[c] = [x / piv for x in R[c]]
        for r in range(n):
            if r != c and R[r][c] != 0:
                f = R[r][c]
                R[r] = [a - f * b for a, b in zip(R[r], R[c])]
    return [row[n:] for row in R]


def run(L, rng):
    V, E = grid(L)
    OE = oriented(E)
    B = [v for v in V if (v[0] + v[1]) % 2 == 0]
    W = [v for v in V if (v[0] + v[1]) % 2 == 1]
    bi = {v: i for i, v in enumerate(B)}
    wi = {v: i for i, v in enumerate(W)}
    sgn = [real_sign(k) for (_, _, k) in OE]
    eidx = {(b, w): k for k, (b, w, _) in enumerate(OE)}

    # 1. face condition
    faces_bad = 0
    nfaces = 0
    for x2 in range(1, L):
        for x1 in range(1, L):
            corners = [(x1, x2), (x1 + 1, x2), (x1, x2 + 1), (x1 + 1, x2 + 1)]
            fe = []
            for (b, w, k) in OE:
                if b in corners and w in corners:
                    fe.append(real_sign(k))
            assert len(fe) == 4
            prod = 1
            for s in fe:
                prod *= s
            nfaces += 1
            faces_bad += prod != -1

    # 2. same sign / same phase for every matching
    Ms = matchings(V, E)
    real_terms = set()
    complex_phases = set()
    for M in Ms:
        p = [None] * len(B)
        s_real = 1
        nvert = 0
        for k in M:
            b, w, kind = OE[k]
            p[bi[b]] = wi[w]
            s_real *= sgn[k]
            nvert += kind != 'h'
        ps = perm_sign(p)
        real_terms.add(ps * s_real)
        # complex term: ps * i^{nvert}; record as (ps * i^nvert) in {1,i,-1,-i}
        complex_phases.add((ps * [1, 1j, -1, -1j][nvert % 4]))
    assert len(real_terms) == 1, real_terms
    assert len(complex_phases) == 1, complex_phases
    eps = real_terms.pop()

    # 3. polynomial identity det K'(z) = eps * sum_M prod z_e
    poly_checks = 0
    for trial in range(6):
        lo = -50 if trial % 2 else 1
        z = [rng.randint(lo, 10 ** 6) for _ in E]
        K = [[0] * len(W) for _ in B]
        for k, (b, w, kind) in enumerate(OE):
            K[bi[b]][wi[w]] = sgn[k] * z[k]
        lhs = det_int(K)
        rhs = 0
        for M in Ms:
            t = 1
            for k in M:
                t *= z[k]
            rhs += t
        assert lhs == eps * rhs, (L, lhs, rhs)
        poly_checks += 1

    # 4. tilted identity with real-gauge H (|H| entries equal complex-gauge ones)
    K1 = [[0] * len(W) for _ in B]
    for k, (b, w, kind) in enumerate(OE):
        K1[bi[b]][wi[w]] = sgn[k]
    Kinv = inv_frac(K1)  # rows W, cols B
    Z = len(Ms)
    Msets = [set(M) for M in Ms]
    tilt_checks = 0
    eta = Fr(2, 5)
    for trial in range(25):
        t = rng.randint(1, min(8, len(E)))
        F = rng.sample(range(len(E)), t)
        H = [[Kinv[wi[OE[i][1]]][bi[OE[j][0]]] * sgn[j] for j in F] for i in F]
        lhs = det_frac([[Fr(int(a == c)) - eta * H[a][c] for c in range(t)] for a in range(t)])
        rhs = sum((1 - eta) ** len(Ms_ & set(F)) for Ms_ in Msets) / Z
        assert lhs == rhs and lhs >= 0
        # diagonal of H = edge marginals
        for a, e in enumerate(F):
            assert H[a][a] == Fr(sum(e in Ms_ for Ms_ in Msets), Z)
        tilt_checks += 1
    return dict(L=L, matchings=Z, faces=nfaces, faces_violating=faces_bad,
                real_gauge_common_sign=eps, complex_common_phase=str(complex_phases),
                polynomial_identity_checks=poly_checks, tilted_identity_checks=tilt_checks)


def main():
    rng = random.Random(20261103)
    out = [run(L, rng) for L in (2, 4, 6)]
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
