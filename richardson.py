#!/usr/bin/env python3
"""Richardson-expansion fit of a convergence sweep, per Plan/CALIBRATION_METHODS.md
(§1-§4 only: points/errors, fit model, admissibility, asymptotic window, AIC k).
No n^req extraction and no error propagation here.

    L(n) = L_inf - sum_{i=1..k} C_i (n/n0)^(-p*i)

* p is DERIVED from the scan design (§3): alpha = dln(m)/dln(n) with
  m = n_m/(nx*ny*nz);  p = min(alpha*P_SHOT, P_CIC), p = P_CIC if alpha <= ALPHA_TOL.
* weighted LS with W = diag(1/sigma_i^2), sigma_i = SEM (§1.1, §2.1);
  constrained fallback (C_i >= 0) only if the unconstrained series never
  crosses the +/-tol band (§2.1).
* an order k is admissible only if |T_1| > |T_2| > ... > |T_k| at the smallest
  fitted n (§2.2); k in 1..min(K_MAX, N-2), selected by plain AIC (§4).
* optional asymptotic-window search: drop lowest points (floor 3) until some k
  is admissible (§2.3).
"""
import math
import numpy as np

P_SHOT, P_CIC, ALPHA_TOL = 1.0, 2.0, 1e-2
K_MAX, TOL, MIN_WINDOW = 3, 0.05, 3


def build_points(rows, key):
    """rows: iterable of (n, L) ; returns sorted arrays n, Lmean, std, N.
    std is NaN for single-seed points (caller decides how to fill it)."""
    from collections import defaultdict
    d = defaultdict(list)
    for n, L in rows:
        d[n].append(L)
    n = np.array(sorted(d))
    Lm = np.array([np.mean(d[v]) for v in n])
    N = np.array([len(d[v]) for v in n])
    s = np.array([np.std(d[v], ddof=1) if len(d[v]) > 1 else np.nan for v in n])
    return n, Lm, s, N


def scan_alpha(n, m):
    """alpha = d ln m / d ln n over the sampled points (LS slope)."""
    x, y = np.log(np.asarray(n, float)), np.log(np.asarray(m, float))
    if len(x) < 2 or np.ptp(x) == 0:
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def derive_p(alpha):
    if alpha <= ALPHA_TOL:
        return P_CIC
    return float(min(alpha * P_SHOT, P_CIC))


def design(n, L, sig, p, k, n0):
    X = np.empty((len(n), k + 1))
    X[:, 0] = 1.0
    for i in range(1, k + 1):
        X[:, i] = -(np.asarray(n, float) / n0) ** (-p * i)
    return X


def wls(n, L, sig, p, k, n0, nonneg=False):
    X = design(n, L, sig, p, k, n0)
    w = 1.0 / np.asarray(sig, float) ** 2
    if nonneg:
        from scipy.optimize import lsq_linear
        sw = np.sqrt(w)
        lb = np.r_[-np.inf, np.zeros(k)]
        beta = lsq_linear(X * sw[:, None], np.asarray(L) * sw, bounds=(lb, np.inf),
                          method="trf").x
    else:
        XtW = X.T * w
        beta = np.linalg.solve(XtW @ X, XtW @ np.asarray(L))
    XtW = X.T * w
    try:
        cov = np.linalg.inv(XtW @ X)
    except np.linalg.LinAlgError:
        cov = np.full((k + 1, k + 1), np.nan)
    resid = (np.asarray(L) - X @ beta) / np.asarray(sig)
    return beta, cov, float(resid @ resid)


def model(beta, n, p, n0):
    n = np.asarray(n, float)
    out = np.full_like(n, beta[0])
    for i, C in enumerate(beta[1:], start=1):
        out = out - C * (n / n0) ** (-p * i)
    return out


def terms(beta, n_min, p, n0):
    return np.array([abs(C) * (n_min / n0) ** (-p * i) for i, C in enumerate(beta[1:], 1)])


def admissible(beta, n_min, p, n0):
    T = terms(beta, n_min, p, n0)
    return bool(np.all(np.diff(T) < 0)) if len(T) > 1 else True


def has_crossing(beta, p, n0):
    """Does |L(n)-L_inf| cross tol*|L_inf| somewhere on a wide n/n0 grid?"""
    x = np.geomspace(1e-5, 1e5, 1201) * n0
    with np.errstate(over="ignore", invalid="ignore"):
        dev = np.abs(model(beta, x, p, n0) - beta[0]) - TOL * abs(beta[0])
    dev = dev[np.isfinite(dev)]
    return len(dev) > 1 and np.any(np.sign(dev[:-1]) != np.sign(dev[1:]))


def select_k(n, L, sig, p, n0):
    """AIC over admissible orders. Returns dict or None if none admissible."""
    N = len(n)
    cands = []
    for k in range(1, min(K_MAX, N - 2) + 1):
        beta, cov, chi2 = wls(n, L, sig, p, k, n0)
        mode = "unconstrained"
        if not has_crossing(beta, p, n0):
            try:
                beta, cov, chi2 = wls(n, L, sig, p, k, n0, nonneg=True)
                mode = "constrained (C_i>=0)"
            except Exception:
                pass
        adm = admissible(beta, n[0], p, n0)
        K = k + 1
        aic = chi2 + 2 * K
        aicc = aic + (2 * K * (K + 1) / (N - K - 1) if N - K - 1 > 0 else np.inf)
        cands.append(dict(k=k, beta=beta, cov=cov, chi2=chi2, dof=N - K, aic=aic,
                          aicc=aicc, admissible=adm, mode=mode))
    adm = [c for c in cands if c["admissible"]]
    if not adm:
        return None, cands
    best = min(adm, key=lambda c: c["aic"])
    amin = best["aic"]
    for c in cands:
        c["dAIC"] = c["aic"] - amin
    return best, cands


def fit(n, L, sig, p, window_search=True, n0=None):
    """Fit on the largest top-window with an admissible order.
    Returns dict(beta, k, n0, p, chi2, dof, aic table, window index i0, mode)
    or None if no fit is possible (fewer than 3 points / nothing admissible)."""
    n = np.asarray(n, float); L = np.asarray(L, float); sig = np.asarray(sig, float)
    N = len(n)
    if N < MIN_WINDOW:
        return None
    last_i0 = 0 if window_search else 0
    i0_range = range(0, N - MIN_WINDOW + 1) if window_search else [0]
    for i0 in i0_range:
        nn, LL, ss = n[i0:], L[i0:], sig[i0:]
        if len(nn) < MIN_WINDOW:
            break
        n0_ = n0 if n0 else float(np.exp(np.mean(np.log(nn))))
        best, cands = select_k(nn, LL, ss, p, n0_)
        if best is not None:
            return dict(beta=best["beta"], cov=best["cov"], k=best["k"], n0=n0_, p=p,
                        chi2=best["chi2"], dof=best["dof"], cands=cands, i0=i0,
                        mode=best["mode"], n=nn, L=LL, sig=ss)
    return None


def summary(res):
    if res is None:
        return "no admissible fit"
    b = res["beta"]
    C = ", ".join(f"C{i}={c:+.3g}" for i, c in enumerate(b[1:], 1))
    return (f"p={res['p']:g} k={res['k']} L_inf={b[0]:.4f} [{C}] "
            f"chi2/dof={res['chi2']:.2f}/{res['dof']} n0={res['n0']:.4g} "
            f"window i0={res['i0']} {res['mode']}")
