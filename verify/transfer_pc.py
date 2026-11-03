#!/usr/bin/env -S uv run --no-project python
"""Explicit column-transfer PC for U_L (paper Theorem C, tightness).

Builds the circuit as an explicit DAG with indicator leaves [x_e=0], [x_e=1],
checks smoothness, decomposability and determinism structurally, and checks
exactness: unnormalized value 1 at every perfect matching (enumerated for
L<=6) and total mass Z (all indicators = 1) equal to the matching count.
Since the circuit is nonnegative, value 1 on Z distinct matchings plus total
mass Z forces value 0 everywhere else.
"""
import json
from functools import lru_cache


def edges(L):
    H = {(c, r): ('h', c, r) for c in range(1, L) for r in range(1, L + 1)}   # (c,r)-(c+1,r)
    Vt = {(c, r): ('v', c, r) for c in range(1, L + 1) for r in range(1, L)}  # (c,r)-(c,r+1)
    return H, Vt


def vertical_fill(L, covered):
    """Rows 1..L in a column, `covered` = set of rows already covered by horizontals.
    Return the set of vertical edges (lower row r means edge r-(r+1)) or None."""
    out = set()
    r = 1
    while r <= L:
        if r in covered:
            r += 1
            continue
        if r + 1 <= L and (r + 1) not in covered:
            out.add(r)
            r += 2
        else:
            return None
    return out


def build(L):
    H, Vt = edges(L)
    nodes = []  # (type, payload, children)

    def leaf(var, val):
        nodes.append(('leaf', (var, val), ()))
        return len(nodes) - 1

    leaf_id = {}
    for var in list(H.values()) + list(Vt.values()):
        for val in (0, 1):
            leaf_id[var, val] = leaf(var, val)

    subsets = list(range(1 << L))

    def rows(mask):
        return {r + 1 for r in range(L) if mask >> r & 1}

    prev = {0: None}  # state -> node id (None = empty product)
    nprod = 0
    for c in range(1, L + 1):
        last = c == L
        cur = {}
        targets = [0] if last else subsets
        for s in targets:
            kids = []
            for sp, sp_node in prev.items():
                if sp & s:
                    continue
                vf = vertical_fill(L, rows(sp) | rows(s))
                if vf is None:
                    continue
                ch = [] if sp_node is None else [sp_node]
                ch += [leaf_id[Vt[c, r], int(r in vf)] for r in range(1, L)]
                if not last:
                    ch += [leaf_id[H[c, r], int(r in rows(s))] for r in range(1, L + 1)]
                nodes.append(('prod', (c, sp, s), tuple(ch)))
                nprod += 1
                kids.append(len(nodes) - 1)
            if kids:
                nodes.append(('sum', (c, s), tuple(kids)))
                cur[s] = len(nodes) - 1
        prev = cur
    root = prev[0]
    return nodes, root, H, Vt


def check(L, enumerate_pm):
    nodes, root, H, Vt = build(L)
    # scopes, smoothness, decomposability
    scope = [None] * len(nodes)
    for i, (typ, pay, ch) in enumerate(nodes):
        if typ == 'leaf':
            scope[i] = frozenset([pay[0]])
        elif typ == 'prod':
            tot = 0
            u = set()
            for c in ch:
                tot += len(scope[c])
                u |= scope[c]
            assert tot == len(u), 'not decomposable'
            scope[i] = frozenset(u)
        else:
            s0 = scope[ch[0]]
            assert all(scope[c] == s0 for c in ch), 'not smooth'
            scope[i] = s0
    allvars = set(H.values()) | set(Vt.values())
    assert scope[root] == allvars
    # determinism: children of each sum differ in the H_{c-1} state sp (distinct by construction)
    for typ, pay, ch in nodes:
        if typ == 'sum':
            sps = [nodes[c][1][1] for c in ch]
            assert len(set(sps)) == len(sps)

    def evaluate(assign):  # assign: var -> 0/1, or None for marginalize-all
        val = [0] * len(nodes)
        for i, (typ, pay, ch) in enumerate(nodes):
            if typ == 'leaf':
                val[i] = 1 if assign is None else int(assign[pay[0]] == pay[1])
            elif typ == 'prod':
                v = 1
                for c in ch:
                    v *= val[c]
                    if v == 0:
                        break
                val[i] = v
            else:
                val[i] = sum(val[c] for c in ch)
        return val[root]

    Z = evaluate(None)
    n_edges = sum(len(ch) for _, _, ch in nodes)
    res = dict(L=L, nodes=len(nodes), wires=n_edges, total_mass_Z=Z)
    if enumerate_pm:
        import sys
        sys.path.insert(0, '.')
        from kasteleyn_pgc_checks import grid, matchings
        V, E = grid(L)
        Ms = matchings(V, E)
        evar = []
        for (u, v) in E:
            if u[1] == v[1]:
                evar.append(H[min(u[0], v[0]), u[1]])
            else:
                evar.append(Vt[u[0], min(u[1], v[1])])
        ok = 0
        for M in Ms:
            a = {var: 0 for var in allvars}
            for k in M:
                a[evar[k]] = 1
            assert evaluate(a) == 1
            ok += 1
        assert Z == len(Ms)
        res['pm_value_one_checked'] = ok
    return res


def main():
    known = {2: 2, 4: 36, 6: 6728, 8: 12988816, 10: 258584046368}
    out = []
    for L in (2, 4, 6, 8, 10):
        r = check(L, enumerate_pm=L <= 6)
        assert r['total_mass_Z'] == known[L]
        out.append(r)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
