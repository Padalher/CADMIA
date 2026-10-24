"""Shared statistics: ordinary least-squares linear regression and its significance.

Kept dependency-free (no numpy/scipy) so it can be reused anywhere in the core. The
two-tailed p-value of the slope comes from Student's t distribution, evaluated through the
regularized incomplete beta function (Numerical-Recipes ``betai``/``betacf``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass
class LinFit:
    """Result of an ordinary least-squares fit y = slope·x + intercept."""

    n: int
    slope: float
    intercept: float
    r: float
    r2: float
    stderr_slope: Optional[float] = None
    t_stat: Optional[float] = None
    p_value: Optional[float] = None


def pearson_r(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """Pearson correlation coefficient, or None when undefined (n<2 or a constant column)."""
    n = len(x)
    if n < 2 or len(y) != n:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx == 0 or syy == 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def linregress(x: Sequence[float], y: Sequence[float]) -> Optional[LinFit]:
    """Least-squares line fit. Returns None if n<2, lengths differ, or x is constant."""
    n = len(x)
    if n < 2 or len(y) != n:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    if sxx == 0:
        return None  # constant x → slope undefined
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    syy = sum((b - my) ** 2 for b in y)
    slope = sxy / sxx
    intercept = my - slope * mx
    r = sxy / math.sqrt(sxx * syy) if syy > 0 else 0.0
    r2 = r * r

    stderr = t = p = None
    if n > 2:
        ss_res = max(syy - slope * sxy, 0.0)   # residual sum of squares = syy·(1−r²)
        df = n - 2
        s2 = ss_res / df
        stderr = math.sqrt(s2 / sxx)
        if stderr > 0:
            t = slope / stderr
            p = _student_t_two_tailed(t, df)
        elif r2 >= 1.0:
            t, p = math.inf, 0.0                # perfect fit
    return LinFit(n=n, slope=slope, intercept=intercept, r=r, r2=r2,
                  stderr_slope=stderr, t_stat=t, p_value=p)


# ── Student's t two-tailed p-value via the regularized incomplete beta ──

def _betacf(a: float, b: float, x: float) -> float:
    MAXIT, EPS, FPMIN = 200, 3.0e-12, 1.0e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(ln_beta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _student_t_two_tailed(t: float, df: int) -> Optional[float]:
    """Two-tailed p-value P(|T| ≥ |t|) for Student's t with ``df`` degrees of freedom."""
    if df <= 0:
        return None
    if math.isinf(t):
        return 0.0
    return _betai(0.5 * df, 0.5, df / (df + t * t))
