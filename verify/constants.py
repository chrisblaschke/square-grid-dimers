#!/usr/bin/env -S uv run --no-project python
"""Certified bounds for eq:cL in the paper, in exact rational arithmetic.

Transcendental input inequalities (certified below by rational arithmetic):
  A  = (1+sqrt2)/(2 asinh 1) < A_UP = 1.3696
  U0 = 1/sqrt2              < U_UP = 0.70711
  1/(3 sqrt6)              > S_LO = 0.136
  trimmed square: boundary distance D >= a+1 > L/32, so 2D > L/16.
  |H_ii| = P(e_i) in [1/4 - 48A/L, 1/4 + 48A/L]
  |H_ij| <= min(U0, A/d_k) + 3 min(U0, A/max(L/16, d_k)),  d_k = 3|i-j|-2.
Row sum bound  R(L) = (1/4 + 48A/L)^2 + 2 sum_{k=1}^{infinity} h_k^2
  (sum the rational upper bounds to k=L exactly; for k>L use h_k <= 4A_UP/d_k, and
   sum_{k>L} 1/(3k-2)^2 <= 1/(3(3L-2))).  Every h_k decreases in L, so R(L)
   decreases and c(L) increases with L: c(L0) is valid for all L >= L0.
Certified exponent lower bound: S_LO * p_lo^2/(2R_up), p_lo = 1/4 - 48A_UP/L.
Output keys p, R, c, one_over_c are respectively lower, upper, lower, upper
bounds, displayed as floats (not directed decimal rounding). The infinity row
sums k <= 20000 with rational upper inputs and adds a rational integral tail;
it is a conservative certificate, not the exact limiting value in eq:cL.
The certificate is positive for every even L >= 264.
"""
import json
from fractions import Fraction as Fr

A_UP = Fr(13696, 10000)
U_UP = Fr(70711, 100000)
S_LO = Fr(136, 1000)

# a < sqrt(2) < b, and log(1+sqrt(2)) = 2*atanh(sqrt(2)-1).
# Its positive series gives a strict lower bound after eight terms.
a, b = Fr(1414213, 10**6), Fr(1414214, 10**6)
assert a*a < 2 < b*b
t = a - 1
log_lo = 2 * sum((t**(2*j + 1) / (2*j + 1) for j in range(8)), Fr(0))
assert (1 + b) / (2 * log_lo) < A_UP
assert U_UP > 0 and U_UP**2 > Fr(1, 2)
assert S_LO > 0 and 54 * S_LO**2 < 1


def bounds(L):
    L = Fr(L)
    v = 48 * A_UP / L
    p = Fr(1, 4) - v
    R = (Fr(1, 4) + v) ** 2
    s = Fr(0)
    for k in range(1, int(L) + 1):
        d = Fr(3 * k - 2)
        direct = min(U_UP, A_UP / d)
        image = min(U_UP, A_UP / max(L / 16, d))
        s += (direct + 3 * image) ** 2
    s += 16 * A_UP ** 2 / (3 * (3 * L - 2))  # tail k > L
    R += 2 * s
    if p <= 0:
        return dict(L=int(L), p=float(p), R=float(R), c=None)
    assert p / R <= 1
    c = S_LO * p * p / (2 * R)
    return dict(L=int(L), p=float(p), R=float(R), c=float(c), one_over_c=float(1 / c), c_exact=c)


def asymptotic():
    # Conservative lower certificate for lim c(L), not the defining limit.
    # L -> infinity: v -> 0, images -> 0.
    s = Fr(0)
    for k in range(1, 20001):
        d = Fr(3 * k - 2)
        s += min(U_UP, A_UP / d) ** 2
    tail = A_UP ** 2 * Fr(1, 9) * Fr(1, 19999)  # sum_{k>20000} 1/(3k-2)^2 <= (1/9) * 1/(20000-1)
    R = Fr(1, 16) + 2 * (s + tail)
    c = S_LO * Fr(1, 16) / (2 * R)
    return dict(L='inf', R=float(R), c=float(c), one_over_c=float(1 / c))


def main():
    out = []
    for L in (264, 300, 400, 600, 1000, 2000, 5000, 10000, 100000):
        b = bounds(L)
        if L == 264:
            assert b['c_exact'] >= Fr(1, 77600000)
        if L == 1000:
            assert b['c_exact'] >= Fr(1, 1129)
        b.pop('c_exact', None)
        out.append(b)
    out.append(asymptotic())
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
