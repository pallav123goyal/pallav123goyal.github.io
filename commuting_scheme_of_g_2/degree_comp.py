import sys, time, math, argparse
from collections import deque

import numpy as np

# ----------------------------------------------------------------------------
# Root systems (Bourbaki simple roots in integer ambient coordinates; E and F scaled by 2)
# ----------------------------------------------------------------------------
def simple_roots(typ, l):
    def e(i, n):
        v = [0] * n
        v[i] = 1
        return v
    if typ == 'A':
        n = l + 1
        return [[a - b for a, b in zip(e(i, n), e(i + 1, n))] for i in range(l)]
    if typ in 'BCD':
        S = [[a - b for a, b in zip(e(i, l), e(i + 1, l))] for i in range(l - 1)]
        if typ == 'B':
            S.append(e(l - 1, l))
        elif typ == 'C':
            S.append([2 * x for x in e(l - 1, l)])
        else:
            S.append([a + b for a, b in zip(e(l - 2, l), e(l - 1, l))])
        return S
    if typ == 'G':
        return [[1, -1, 0], [-2, 1, 1]]
    if typ == 'F':
        return [[0, 2, -2, 0], [0, 0, 2, -2], [0, 0, 0, 2], [1, -1, -1, -1]]
    if typ == 'E':
        a1 = [1, -1, -1, -1, -1, -1, -1, 1]
        a2 = [2, 2, 0, 0, 0, 0, 0, 0]
        a3 = [-2, 2, 0, 0, 0, 0, 0, 0]
        rest = []
        for k in range(5):                      # a4..a8 = e_{k+3} - e_{k+2}, scaled by 2
            v = [0] * 8
            v[k + 1], v[k + 2] = -2, 2
            rest.append(v)
        return ([a1, a2, a3] + rest)[:l]
    raise ValueError(typ)


WEYL_ORDER = {'A': lambda l: math.factorial(l + 1),
              'B': lambda l: 2 ** l * math.factorial(l),
              'C': lambda l: 2 ** l * math.factorial(l),
              'D': lambda l: 2 ** (l - 1) * math.factorial(l),
              'E': lambda l: {6: 51840, 7: 2903040, 8: 696729600}[l],
              'F': lambda l: 1152,
              'G': lambda l: 12}
NUM_ROOTS = {'A': lambda l: l * (l + 1), 'B': lambda l: 2 * l * l, 'C': lambda l: 2 * l * l,
             'D': lambda l: 2 * l * (l - 1), 'E': lambda l: {6: 72, 7: 126, 8: 240}[l],
             'F': lambda l: 48, 'G': lambda l: 12}
DIM_G = {'A': lambda l: l * (l + 2), 'B': lambda l: l * (2 * l + 1), 'C': lambda l: l * (2 * l + 1),
         'D': lambda l: l * (2 * l - 1), 'E': lambda l: {6: 78, 7: 133, 8: 248}[l],
         'F': lambda l: 52, 'G': lambda l: 14}


def dot(u, v):
    return sum(a * b for a, b in zip(u, v))


def reflect(v, alpha):
    """s_alpha(v) = v - <v, alpha^vee> alpha."""
    num, den = 2 * dot(v, alpha), dot(alpha, alpha)
    assert num % den == 0
    c = num // den
    return tuple(x - c * a for x, a in zip(v, alpha))


def positive_roots(typ, l):
    """All roots as the W-orbit of the simple roots; positive = nonnegative coefficients."""
    S = [tuple(a) for a in simple_roots(typ, l)]
    roots = {}
    for i, a in enumerate(S):
        c = [0] * l
        c[i] = 1
        roots[a] = tuple(c)
    queue = deque(roots.items())
    while queue:
        v, c = queue.popleft()
        for i, a in enumerate(S):
            k = 2 * dot(v, a) // dot(a, a)
            w = tuple(x - k * y for x, y in zip(v, a))
            if w not in roots:
                cw = list(c)
                cw[i] -= k
                roots[w] = tuple(cw)
                queue.append((w, tuple(cw)))
    assert len(roots) == NUM_ROOTS[typ](l)
    pos = [r for r, c in roots.items() if all(x >= 0 for x in c)]
    assert 2 * len(pos) == len(roots)
    return pos, S


def weyl_orbit(lam, S):
    seen, queue = {tuple(lam)}, deque([tuple(lam)])
    while queue:
        v = queue.popleft()
        for a in S:
            w = reflect(v, a)
            if w not in seen:
                seen.add(w)
                queue.append(w)
    return list(seen)


def setup(typ, l):
    pos, S = positive_roots(typ, l)
    N = len(pos)
    lam = tuple(sum(r[i] for r in pos) for i in range(len(S[0])))        # 2 rho
    assert all(dot(r, lam) > 0 for r in pos)
    orbit = weyl_orbit(lam, S)
    assert len(orbit) == WEYL_ORDER[typ](l)
    D = 1
    for r in pos:
        D *= dot(r, lam)
    return pos, N, orbit, D


# ----------------------------------------------------------------------------
# Exact evaluation (pure Python integers; only used for small W)
# ----------------------------------------------------------------------------
def exact_degree(typ, l):
    pos, N, orbit, D = setup(typ, l)
    total = 0
    for mu in orbit:
        avals = [dot(r, mu) for r in pos]
        sgn = -1 if sum(1 for a in avals if a < 0) % 2 else 1
        coeffs = [1] + [0] * N
        for a in avals:                                   # multiply by 2 - 3 a t + a^3 t^3
            a3 = a ** 3
            new = [0] * (N + 1)
            for k in range(N + 1):
                s = 2 * coeffs[k]
                if k >= 1:
                    s -= 3 * a * coeffs[k - 1]
                if k >= 3:
                    s += a3 * coeffs[k - 3]
                new[k] = s
            coeffs = new
        total += sgn * coeffs[N]
    num, den = (-1) ** N * total, len(orbit) * D
    assert num % den == 0, "the formula did not return an integer"
    return num // den


# ----------------------------------------------------------------------------
# Modular evaluation with numpy + Chinese remainder theorem
# ----------------------------------------------------------------------------
def is_prime(n):
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    if n < 2:
        return False
    for p in small:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in small:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def primes_below_2_30(k):
    ps, n = [], 2 ** 30 - 1
    while len(ps) < k:
        if is_prime(n):
            ps.append(n)
        n -= 2
    return ps


def orbit_sum_mod_p(A, sgn, N, p, chunk=20000):
    """sum_mu sgn(mu) P_N(mu) mod p.  A: int64 array (|W|, N) of pairings <gamma, mu>."""
    total = 0
    for start in range(0, A.shape[0], chunk):
        a = A[start:start + chunk] % p
        a3 = (a * a % p) * a % p
        m3a = (p - 3 * a % p) % p
        C = np.zeros((a.shape[0], N + 1), dtype=np.int64)
        C[:, 0] = 1
        for j in range(N):
            new = 2 * C
            new[:, 1:] += m3a[:, j:j + 1] * C[:, :-1]
            new[:, 3:] += a3[:, j:j + 1] * C[:, :-3]
            C = new % p
        total = (total + int(np.sum((C[:, N] * sgn[start:start + chunk]) % p) % p)) % p
    return total


def crt(residues, primes):
    x, mod = 0, 1
    for r, p in zip(residues, primes):
        x += mod * ((r - x) * pow(mod % p, p - 2, p) % p)
        mod *= p
    return x


def modular_degree(typ, l, nprimes=7):
    pos, N, orbit, D = setup(typ, l)
    W = len(orbit)
    A = np.dot(np.array(orbit, dtype=np.int64), np.array(pos, dtype=np.int64).T)
    assert np.all(A != 0)
    sgn = np.where((A < 0).sum(axis=1) % 2 == 0, 1, -1).astype(np.int64)
    primes = primes_below_2_30(nprimes)
    residues = []
    for p in primes:
        s = orbit_sum_mod_p(A, sgn, N, p)
        den = (W * D * (-1) ** N) % p
        residues.append(s * pow(den, p - 2, p) % p)
    full = crt(residues, primes)
    assert full == crt(residues[:-1], primes[:-1]), "CRT unstable: increase nprimes"
    return full


def degree(typ, l):
    d = modular_degree(typ, l)
    if WEYL_ORDER[typ](l) <= 5000:                 # independent exact cross-check when cheap
        assert d == exact_degree(typ, l)
    return d

LIE = {'A': lambda l: 'sl_%d' % (l + 1), 'B': lambda l: 'so_%d' % (2 * l + 1),
       'C': lambda l: 'sp_%d' % (2 * l), 'D': lambda l: 'so_%d' % (2 * l),
       'E': lambda l: 'e_%d' % l, 'F': lambda l: 'f_4', 'G': lambda l: 'g_2'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--amax', type=int, default=8)
    ap.add_argument('--extra', nargs='*', default=[])
    ap.add_argument('--tex', default=None)
    args = ap.parse_args()

    cases = [('A', l) for l in range(1, 16)] \
        + [('B', l) for l in range(2, 7)] + [('C', l) for l in range(3, 7)] \
        + [('D', l) for l in range(4, 8)] + [('E', 6), ('G', 2)] \
        + [(s[0].upper(), int(s[1:])) for s in args.extra]

    print("%-8s%-8s%7s%5s%12s   %s" % ("type", "g", "dim g", "N", "|W|", "deg C(g)"))
    results = []
    for typ, l in cases:
        t0 = time.time()
        d = degree(typ, l)
        check = ''
        print("%-8s%-8s%7d%5d%12d   %d%s   [%.0fs]" % (typ + str(l), LIE[typ](l), DIM_G[typ](l),
              NUM_ROOTS[typ](l) // 2, WEYL_ORDER[typ](l), d, check, time.time() - t0), flush=True)
        results.append((typ, l, d))

    if args.tex:
        with open(args.tex, 'w') as f:
            f.write("\\begin{tabular}{llrrrr}\n\\toprule\n")
            f.write("$\\mathfrak g$ & type & $\\dim\\mathfrak g$ & $N$ & $|W|$ & $\\deg\\mathcal C(\\mathfrak g)$ \\\\\n\\midrule\n")
            for typ, l, d in results:
                name, idx = LIE[typ](l).split('_')
                f.write("$\\mathfrak{%s}_{%s}$ & $%s_{%d}$ & %d & %d & %d & %d \\\\\n" % (
                    name, idx, typ, l, DIM_G[typ](l), NUM_ROOTS[typ](l) // 2, WEYL_ORDER[typ](l), d))
            f.write("\\bottomrule\n\\end{tabular}\n")
        print("LaTeX table written to", args.tex)


if __name__ == '__main__':
    main()
