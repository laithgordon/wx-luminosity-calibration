#!/usr/bin/env python3
"""Every number quoted from the WarpX side of the paper, computed from data/ alone and printed to stdout.

Each quantity is taken from the same code path that draws the corresponding figure, so a number and its figure can
never disagree: luminosities and ratios from plot_comparison.curves() / plot_extension.series(), H_D from the same
luminosities over plot_comparison.L_geom, kappa from paper_figures.kappa_vs_Dy, the convergence fits from
paper_figures.WX_{nm,ny}_req_calibration, and the pinched-core values from plot_core_width.draw(). The figure-saving
calls are suppressed here, so running this script never touches plots/.

Uncertainties: std = sample standard deviation over independent seeds (ddof = 1); se = std/sqrt(n_seeds). Ratios carry
the quadrature sum of relative standard errors. Same-seed repeat runs are averaged before the seed statistics.

    python paper_numbers.py                      # printed summary
    python paper_numbers.py --json numbers.json  # also write every value, at full precision, to a JSON file
"""
import argparse, json, math, os
from pathlib import Path
import numpy as np

os.environ.setdefault("WX_CACHED_ONLY", "1")          # plot_core_width: cached slice widths only, never the dumps
import nmreq as Q
import paper_figures as PF
import plot_comparison as PC
import plot_extension as PX
import plot_core_width as PCW

ROOT = Path(__file__).resolve().parent
PS1_PUBLISHED = 1.35                                  # 1e34 cm^-2 s^-1, the published PS1 luminosity
EYS = [float(e) for e in Q.EYS]


def _no_save(*args, **kwargs):
    return None


PF.save_fig = _no_save; PC.save_fig = _no_save; PX.save_fig = _no_save; PCW.save_fig = _no_save


def point(curve, e):
    es, L, sd, n = curve
    i = [float(x) for x in es].index(float(e))
    return dict(value=float(L[i]), std=float(sd[i]), stderr=float(sd[i] / math.sqrt(n[i])), n_seeds=int(n[i]))


def ratio(a, b):
    r = a["value"] / b["value"]
    return dict(value=r, stderr=r * math.hypot(a["stderr"] / a["value"], b["stderr"] / b["value"]),
                std=r * math.hypot(a["std"] / a["value"], b["std"] / b["value"]))


def main(json_path=None):
    C = PC.curves()
    have = lambda k, e: float(e) in [float(x) for x in C[k][0]]
    out = {}

    # luminosity, tuned (frozen production locus) and untuned, and their ratio
    lum = {}
    for e in EYS:
        c, n = point(C["wx_cal"], e), point(C["wx_uncal"], e)
        lum[f"{e:g}"] = dict(conservative=c, nominal=n, conservative_over_nominal=ratio(c, n),
                             gp_conservative=point(C["gp_cal"], e) if have("gp_cal", e) else None,
                             gp_nominal=point(C["gp_uncal"], e) if have("gp_uncal", e) else None)
    out["luminosity_1e34"] = lum

    # WarpX/GP tuned ratio, the series of WX_code_ratio_vs_ey
    S = PX.series()
    wg = {}
    for e in sorted(e for e in S["wx_tuned"] if e in S["gp_tuned"]):
        a = dict(zip(("value", "std", "n_seeds"), S["wx_tuned"][e])); b = dict(zip(("value", "std", "n_seeds"), S["gp_tuned"][e]))
        for d in (a, b): d["stderr"] = d["std"] / math.sqrt(d["n_seeds"])
        wg[f"{e:g}"] = ratio(a, b)
    band = lambda lo, hi: [min(v["value"] for k, v in wg.items() if lo <= float(k) <= hi), max(v["value"] for k, v in wg.items() if lo <= float(k) <= hi)]
    out["wx_over_gp"] = dict(per_emittance=wg, range_8_to_100=band(8, 100), range_1_to_20=band(1, 20))

    # 40-100 nm frozen extension: two-simulator agreement, and extrapolated (PE) vs frozen (PF) with each seed scatter
    ext = {}
    for e in Q.EYS_EXT:
        pf, pe = point(C["wx_frz20"], e), point(C["wx_cal"], e)
        g = dict(zip(("value", "std", "n_seeds"), S["gp_frozen"][float(e)])); g["stderr"] = g["std"] / math.sqrt(g["n_seeds"])
        ext[f"{e:g}"] = dict(wx_frozen=pf, wx_extrapolated=pe, gp_frozen=g, wx_frozen_over_gp_frozen=ratio(pf, g),
                             extrapolated_minus_frozen_pct=100 * (pe["value"] / pf["value"] - 1),
                             scatter_frozen_pct=100 * pf["std"] / pf["value"], scatter_extrapolated_pct=100 * pe["std"] / pe["value"])
    out["frozen_extension"] = ext

    # PS1: 20 nm untuned
    ps = point(C["wx_uncal"], 20)
    out["ps1"] = dict(**ps, ratio_to_published=ps["value"] / PS1_PUBLISHED, ratio_stderr=ps["stderr"] / PS1_PUBLISHED)

    # H_D = L / L_geom
    hd = {}
    for e in EYS:
        lg = PC.L_geom(e); d = lum[f"{e:g}"]
        scale = lambda p: None if p is None else dict(value=p["value"] / lg, std=p["std"] / lg, stderr=p["stderr"] / lg, n_seeds=p["n_seeds"])
        hd[f"{e:g}"] = dict(L_geom=lg, wx_conservative=scale(d["conservative"]), wx_nominal=scale(d["nominal"]),
                            gp_conservative=scale(d["gp_conservative"]), gp_nominal=scale(d["gp_nominal"]))
    out["H_D"] = hd

    # tuning-ladder requirements, their fits, and kappa
    A = Q.requirements_stat(); B = Q.ny_requirements_stat()
    m = PF.WX_nm_req_calibration(A); y = PF.WX_ny_req_calibration(B)
    out["convergence"] = dict(
        s=dict(value=m["fit"]["s"], sigma=m["fit"]["sb"], chi2=m["fit"]["chi2"], ndf=m["fit"]["ndf"], excluded=m["excluded"]),
        q_n=dict(value=y["fit"]["b"], sigma=y["fit"]["sb"], chi2=y["fit"]["chi2"], ndf=y["fit"]["ndf"], C_y=y["C_y"], excluded=y["excluded"]),
        conservative_locus_fit=dict(C_per_nominal_cell=m["cons"]["C"], b_raw=m["cons"]["b_raw"], sigma_b_raw=m["cons"]["sb_raw"]),
        n_m_req={f"{e:g}": dict(value=r["nreq"], sig_log=r["sig_log"]) for e, _, r in A},
        n_y_req={f"{e:g}": dict(value=r["nreq"], sig_log=r["sig_log"]) for e, _, r in B})
    k = PF.kappa_vs_Dy(B)
    out["kappa"] = dict(WX={x: k["WX"][x] for x in ("mean", "sig_mean", "slope", "sig_slope", "chi2", "ndf")},
                        GP={x: k["GP"][x] for x in ("mean", "sig_mean", "slope", "sig_slope", "chi2", "ndf")},
                        agreement_nsigma=abs(k["WX"]["mean"] - k["GP"]["mean"]) / math.hypot(k["WX"]["sig_mean"], k["GP"]["sig_mean"]))

    # pinched core: minimum core width, compression against the first-waist prediction
    rows, fit = PCW.draw()
    Dt, R1t, _, _ = PCW.R1_table()
    core = {}
    for e, d, g, sd, fe, ge, rr, ns, th, n in rows:
        r1 = float(np.interp(d, Dt, R1t))
        core[f"{e:g}"] = dict(D_y=float(d), sigma_min_over_sigma0=float(g), err=float(ge), seed_std=float(sd), fit_err=float(fe),
                              n_seeds=int(n), sigma_min_nm=float(g) * Q.sigma_y(float(e)) * 1e9,
                              R_measured=1 / float(g), R1_predicted=r1, measured_over_theory=float(g) * r1)
    out["pinched_core"] = dict(points=core, free_fit=fit)

    out["locus_constants_frozen"] = dict(n_m=f"{Q.LOCUS_NM_PREFACTOR} D_y^{Q.LOCUS_NM_EXPONENT}",
                                         n_y=f"smallest of {Q.LOCUS_NY_RUNGS} >= {Q.LOCUS_NY_PREFACTOR} D_y^{Q.LOCUS_NY_EXPONENT}")
    if json_path:
        Path(json_path).write_text(json.dumps(out, indent=1, allow_nan=False))

    print(f"{'e_y':>5} {'L cons':>10} {'L nom':>10} {'cons/nom':>9} {'H_D cons':>9}")
    for e in EYS:
        d = lum[f"{e:g}"]
        print(f"{e:>5g} {d['conservative']['value']:10.6f} {d['nominal']['value']:10.6f} {d['conservative_over_nominal']['value']:9.5f} {hd[f'{e:g}']['wx_conservative']['value']:9.5f}")
    print("WarpX/GP:", ", ".join(f"{e} nm {v['value']:.4f}" for e, v in wg.items()))
    print(f"PS1 {ps['value']:.6f} (x{out['ps1']['ratio_to_published']:.4f} of published)  kappa agreement {out['kappa']['agreement_nsigma']:.3f} sigma")
    print(f"s = {m['fit']['s']:.4f} +- {m['fit']['sb']:.4f}   q_n = {y['fit']['b']:.4f} +- {y['fit']['sb']:.4f}")
    if json_path:
        print(f"wrote {json_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", metavar="PATH", help="also write every value, at full precision, to this JSON file")
    main(ap.parse_args().json)
