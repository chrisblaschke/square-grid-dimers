#!/usr/bin/env python3
"""Check Proposition prop:flux of the paper (what fails on the square lattice).

Patch {1..m}^2, m even, black iff x+y even.  Remove the white vertices of the
top row and the black vertices of the bottom row.  Claim: the remnant has a
perfect matching, and no perfect matching uses a vertical edge between rows
m/2 and m/2+1 whose upper endpoint is white.  Kuhn's augmenting-path matching;
standard library only.
"""
import json, sys
sys.setrecursionlimit(100000)

def has_pm(cells):
    cells = set(cells)
    black = [v for v in cells if sum(v) % 2 == 0]
    white = {v for v in cells if sum(v) % 2 == 1}
    if len(black) != len(white):
        return False
    match = {}
    def aug(b, seen):
        x, y = b
        for w in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if w in white and w not in seen:
                seen.add(w)
                if w not in match or aug(match[w], seen):
                    match[w] = b
                    return True
        return False
    return all(aug(b, set()) for b in black)

out = []
for m in range(4, 31, 2):
    patch = {(x, y) for x in range(1, m + 1) for y in range(1, m + 1)}
    drop = {(x, m) for x in range(1, m + 1) if (x + m) % 2 == 1}
    drop |= {(x, 1) for x in range(1, m + 1) if (x + 1) % 2 == 0}
    rem = patch - drop
    r = m // 2
    ok_pm = has_pm(rem)
    bad = []
    for x in range(1, m + 1):
        top = (x, r + 1)
        if sum(top) % 2 == 1:  # white upper endpoint
            if has_pm(rem - {top, (x, r)}):
                bad.append(x)
    out.append(dict(m=m, remnant_has_pm=ok_pm, white_top_middle_edges_in_some_pm=bad))
print(json.dumps(out))
assert all(o["remnant_has_pm"] and not o["white_top_middle_edges_in_some_pm"] for o in out)
print("OK", file=sys.stderr)
