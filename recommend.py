"""Recommended beam-beam simulation configuration from beam parameters.

For a strongly disrupted flat-beam collision, this returns a configuration that resolves the
pinched core and populates it: the cell counts and the macroparticle count, for WarpX
and GUINEA-PIG++. The requirements and the study behind them are described in
"Evaluating beam-beam simulations in the pursuit of designing next generation colliders"
(arXiv link to be added on release); the recipe is set out under HOW THE RECOMMENDATION IS MADE
below, so this module can be read on its own.

    python3 recommend.py                      # self-test, then two worked examples

    from recommend import recommend
    print(recommend(E_GeV=125.0, N=6.24e9, sigma_z_m=100e-6, eps_y_nm=8.0,
                    beta_y_m=0.12e-3, eps_x_nm=900.0, beta_x_m=12e-3, code="GP"))

Results outside the ranges the study covered are flagged on the result and raised as Python
warnings. The `model` mode is experimental: it predicts requirements for a machine geometry the
study did not run, and nothing in the paper tests it.


HOW THE RECOMMENDATION IS MADE
==============================

1. Beam sizes at the waist.  sigma* = sqrt(beta* eps_norm / gamma), separately in x and y.

2. Vertical disruption.  D_y = 2 N r_e sigma_z / (gamma sigma_y* (sigma_x* + sigma_y*)), and
   D_x with sigma_x* in place of sigma_y*.  D_y is the single variable the laws are written in.

3. Vertical compression of the pinch, R(D_y) = sigma_y* / sigma_y^min.  The beam is focused to
   a minimum vertical size inside the crossing, and it is that size the grid must resolve, not
   the incoming one.  R follows from a linear envelope integration: the opposing bunch acts as a
   focusing channel of strength K_y(s) = [D_y / (sqrt(3) sigma_z^2)] g(s), with g the Gaussian
   longitudinal profile normalised so the flat-top equivalent reproduces K_y = D_y/(sqrt(3)
   sigma_z^2).  The Courant-Snyder parameters are propagated from s = -3 sigma_z, where the
   field is negligible and the incoming beam has drifted from its waist to
   beta = beta_y* + s^2/beta_y*.  R is taken at the FIRST local minimum of beta(s): the linear
   envelope conserves emittance and so makes every later waist tighter, while the real force is
   linear only near the axis, so later waists are in fact broader.

   R therefore depends on TWO dimensionless numbers, not one: D_y, and the hourglass ratio
   beta_y*/sigma_z, which fixes where the waist sits relative to the field region.  The study
   ran one value of it, beta_y*/sigma_z = 1.2 (HOURGLASS_REF).

4. Vertical cells.  n_y^req = C_y D_y^q_n with C_y = 2 c_y kappa Lambda: enough cells across the
   pinched core, kappa of them per core sigma, times D_y^(1/4) because the beam re-enters the
   core N_osc ~ sqrt(D_y) times and the deposition error accumulates coherently.  The phase that
   sets N_osc, Phi = int sqrt(K_y(s)) ds, contains no beta_y*, so that exponent of 1/4 is a
   property of D_y alone; only q_p, the depth of the first pinch, moves with the geometry.

5. Macroparticles.  n_m^req = C_m D_y^(s - q_p) n_x n_y n_z, from requiring the central cell of
   the pinched core to hold P(D_y) = P_0 D_y^s macroparticles, the occupancy of that cell being
   [8 c_x c_y c_z / (2 pi)^(3/2)] n_m/(n_x n_y n_z) multiplied by the compression R.  s is
   empirical; q_p is the envelope exponent of step 3.

   Both laws use the conservative locus: the fitted line with its normalisation raised until
   every tuning point sits beneath it.

6. Rounding.  n_y goes up to a power of two (WarpX to the smallest power of two its ladders
   sampled, 32 to 8192); n_m is the law value, which the tables of the paper quote to two
   significant figures.

6b. Box extents.  The cut multipliers set the box as +-c sigma, so the cell size is 2 c sigma/n
   and a different box needs proportionally many cells to keep the same resolution:
   n_x, n_y, n_z all scale with their own multiplier, as in n_x^req = 2 c_x N_0.  The occupancy
   of the pinched central cell, [8 c_x c_y c_z/(2 pi)^(3/2)] n_m/(n_x n_y n_z), then carries the
   cut multipliers in the denominator, so n_m is unchanged by a box that grows with its cell
   count, and moves only through the rounding of the cells.  The study ran one box per code; any
   other is outside it, and a smaller one can crop the collision.

7. Other machine geometries (mode='model', EXPERIMENTAL).  Steps 4 and 5 both carry R through
   their normalisations, C_y proportional to Lambda and C_m inversely proportional to it.  When
   beta_y*/sigma_z differs from HOURGLASS_REF, the recommendation is transported by the ratio of
   modelled compressions,

       n_y -> n_y R_here/R_ref,    n_m -> n_m (n_y_new/n_y_old) (R_ref/R_here),

   which keeps every quantity the study established (kappa, s, the conservative margins, the
   empirical exponents) and applies only what the envelope model says about the geometry change.
   The n_m factors are the cell height, which moves with n_y, and the compression, which
   concentrates charge into the central cell.

Nothing here is fitted.  `selftest()` reproduces both recommended-configuration tables of the
paper digit for digit and checks R(D_y) against the values in `nmreq.py`.
"""
from dataclasses import dataclass, field
import math
import warnings

R_E = 2.8179403262e-15          # classical electron radius [m]
M_E_GEV = 0.51099895000e-3      # electron rest mass [GeV]

# ── GUINEA-PIG++ conservative locus (gp-luminosity-calibration: constants.py) ──
GP = dict(code="GUINEA-PIG++", C_y=50.0, q_n=0.402, C_m=2.305e-7, nm_exponent=3.300,
          n_x=512, n_z=64, cuts=(20, 20, 3.5))
# ── WarpX production locus (nmreq.py) ──
WX = dict(code="WarpX", C_y=38.1, q_n=0.450260, C_m=0.440, nm_exponent=3.269,
          n_x=512, n_z=128, cuts=(16, 16, 8),
          ny_rungs=(32, 64, 128, 256, 512, 1024, 2048, 4096, 8192))

# ── what the study covered ──
D_Y_TESTED = (21.5, 97.4)            # vertical disruption of the tuned range
EPS_Y_TESTED_NM = (1.0, 20.0)
HOURGLASS_REF = 1.2                  # beta_y*/sigma_z of the reference machine
HOURGLASS_TOL = 0.02                 # relative departure treated as the same geometry
REFERENCE = dict(name="C3-250", E_GeV=125.0, N=6.24e9, sigma_z_m=100e-6,
                 eps_x_nm=900.0, beta_x_m=12e-3, beta_y_m=0.12e-3)
GP_NM_WALLCLOCK_LIMIT = 1e7          # largest count reached inside the three-day limit
WX_NM_MEMORY_LIMIT = 1.6e8           # largest count reached below the GPU memory ceiling
FLATNESS_REF = 0.009                 # sigma_y*/sigma_x* of the reference machine
# R(D_y) at the reference ratio, from the envelope integration (see nmreq.R1_of_D)
R_REFERENCE = {137.8: 5.305, 97.4: 4.968, 79.4: 4.773, 68.8: 4.636, 61.5: 4.530,
               48.5: 4.308, 34.2: 3.985, 27.9: 3.800, 24.1: 3.668, 21.5: 3.566}


class ExperimentalRecommendation(UserWarning):
    """The configuration was transported to a geometry the study did not run."""


class OutsideTestedRange(UserWarning):
    """The configuration lies outside the range the study covered."""


def R_pinch(D_y: float, hourglass: float = HOURGLASS_REF, n_steps: int = 6000) -> float:
    """Vertical compression sigma_y*/sigma_y^min from the linear envelope model (step 3).

    `hourglass` is beta_y*/sigma_z. Lengths are in units of sigma_z, so only the two
    dimensionless numbers enter. Fixed-step RK4 keeps this free of dependencies; the step is
    set so the first minimum is located to better than a part in 10^3 of beta.
    """
    s0, s1 = -3.0, 3.0
    h = (s1 - s0) / n_steps
    amp = (D_y / math.sqrt(3.0)) * (2.0 * math.sqrt(3.0) / math.sqrt(2.0 * math.pi))

    def deriv(s, y):
        k = amp * math.exp(-2.0 * s * s)                 # sigma_z = 1 in these units
        return (y[1], -k * y[0], y[3], -k * y[2])

    y = (1.0, 0.0, 0.0, 1.0)                             # principal solutions C, S at s0
    bi = hourglass + s0 * s0 / hourglass                 # drifted beta at the entrance
    ai = -s0 / hourglass                                 # drifted alpha (converging)
    gi = (1.0 + ai * ai) / bi
    beta_prev2 = beta_prev = None
    first_min = None
    for i in range(n_steps):
        s = s0 + i * h
        k1 = deriv(s, y)
        k2 = deriv(s + h / 2, tuple(y[j] + h / 2 * k1[j] for j in range(4)))
        k3 = deriv(s + h / 2, tuple(y[j] + h / 2 * k2[j] for j in range(4)))
        k4 = deriv(s + h, tuple(y[j] + h * k3[j] for j in range(4)))
        y = tuple(y[j] + h / 6 * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]) for j in range(4))
        beta = bi * y[0] ** 2 - 2.0 * ai * y[0] * y[2] + gi * y[2] ** 2
        if (first_min is None and beta_prev2 is not None
                and beta_prev < beta_prev2 and beta_prev < beta):
            first_min = beta_prev
        beta_prev2, beta_prev = beta_prev, beta
    return math.sqrt(hourglass / (first_min if first_min is not None else beta_prev))


@dataclass
class Recommendation:
    code: str
    mode: str
    n_x: int
    n_y: int
    n_z: int
    n_m: int
    D_y: float
    D_x: float
    hourglass: float
    sigma_x_m: float
    sigma_y_m: float
    cuts: tuple
    R_ref: float = float("nan")
    R_here: float = float("nan")
    warnings: list = field(default_factory=list)

    @property
    def is_experimental(self) -> bool:
        return self.mode == "model"

    def __str__(self):
        banner = ""
        if self.is_experimental:
            banner = ("    *** EXPERIMENTAL: transported to a machine geometry the study did not "
                      "run ***\n")
        elif any("outside" in w for w in self.warnings):
            banner = "    *** OUTSIDE THE RANGE THE STUDY COVERED ***\n"
        head = (f"{self.code} [{self.mode}]: (n_x, n_y, n_z) = ({self.n_x}, {self.n_y}, "
                f"{self.n_z}), n_m = {self.n_m:,}\n" + banner +
                f"    D_y = {self.D_y:.3f}, D_x = {self.D_x:.4f}, "
                f"beta_y*/sigma_z = {self.hourglass:.3f}, "
                f"sigma_x* = {self.sigma_x_m * 1e9:.2f} nm, "
                f"sigma_y* = {self.sigma_y_m * 1e9:.3f} nm\n"
                f"    cut multipliers (c_x, c_y, c_z) = {self.cuts}")
        if self.is_experimental:
            head += (f"\n    compression R: {self.R_ref:.3f} at beta_y*/sigma_z = "
                     f"{HOURGLASS_REF} -> {self.R_here:.3f} here")
        return head + ("\n" + "\n".join(f"    ! {w}" for w in self.warnings) if self.warnings else "")


def recommend(E_GeV: float, N: float, sigma_z_m: float, eps_y_nm: float, beta_y_m: float,
              eps_x_nm: float, beta_x_m: float, code: str = "GP",
              mode: str = "recommended", cuts: tuple = None) -> Recommendation:
    """Recommended configuration for one collision.

    code : 'GP' (GUINEA-PIG++) or 'WX' (WarpX).
    cuts : box half-extents (c_x, c_y, c_z) in units of (sigma_x*, sigma_y*, sigma_z);
           defaults to the box this study ran for that code, (20, 20, 3.5) for
           GUINEA-PIG++ and (16, 16, 8) for WarpX. Cell counts scale with them, so the resolution is
           held; a box the study did not run is reported, and a smaller one may crop the
           collision.
    mode : 'recommended' (default) applies the loci of the paper as they stand, and warns when
           this machine's geometry is not the one they were established on;
           'model' is EXPERIMENTAL: it transports them to this machine's beta_y*/sigma_z by the
           envelope model of step 7. Nothing in the paper tests that transport, so the result is
           untested.

    Raises ValueError for a beam that is not flat (sigma_y* >= sigma_x*): the laws describe the
    vertical plane of a flat beam, and applying them to a tall beam would put the requirement on
    the weakly disrupted plane. Swapping the roles of x and y would need the constants
    re-established, which this tool cannot do.
    """
    if mode not in ("recommended", "model"):
        raise ValueError(f"mode must be 'recommended' or 'model', not {mode!r}")
    cfg = GP if code.upper().startswith("G") else WX
    gamma = E_GeV / M_E_GEV
    sx = math.sqrt(beta_x_m * eps_x_nm * 1e-9 / gamma)
    sy = math.sqrt(beta_y_m * eps_y_nm * 1e-9 / gamma)
    if sy >= sx:
        raise ValueError(
            f"sigma_y* = {sy * 1e9:.3f} nm is not smaller than sigma_x* = {sx * 1e9:.3f} nm. "
            "These laws are derived for a flat beam, where the vertical plane is the strongly "
            "disrupted one; here the pinch would be horizontal, so the vertical requirement "
            "returned would apply to the wrong plane.")
    s_sum = sx + sy
    d_y = 2 * N * R_E * sigma_z_m / (gamma * sy * s_sum)
    d_x = 2 * N * R_E * sigma_z_m / (gamma * sx * s_sum)
    hourglass = beta_y_m / sigma_z_m

    # ── steps 4-6: the conservative loci, in the box the caller asked for ──
    c_ref = cfg["cuts"]
    c = tuple(float(v) for v in cuts) if cuts is not None else c_ref
    if len(c) != 3 or any(v <= 0 for v in c):
        raise ValueError(f"cuts must be three positive multipliers (c_x, c_y, c_z), not {cuts!r}")

    def _round_ny(raw):
        if cfg is GP:
            return 2 ** math.ceil(math.log2(raw))        # next power of two
        return min((p for p in cfg["ny_rungs"] if p >= raw), default=cfg["ny_rungs"][-1])

    def _pow2(v):
        return 2 ** math.ceil(math.log2(v))

    # a box of +-c sigma with n cells has cell size 2 c sigma/n, so holding the resolution of
    # the study means scaling each cell count with its own extent
    n_x = _pow2(cfg["n_x"] * c[0] / c_ref[0])
    n_z = _pow2(cfg["n_z"] * c[2] / c_ref[2])
    ny_law = cfg["C_y"] * d_y ** cfg["q_n"]
    n_y_ref = _round_ny(ny_law)                          # at the box the study ran
    ny_raw = ny_law * c[1] / c_ref[1]
    n_y = _round_ny(ny_raw)
    # GUINEA-PIG++ quotes the requirement per cell, WarpX as an absolute count; both are the
    # same occupancy condition, so both transport the same way below
    nm_ref = (cfg["C_m"] * d_y ** cfg["nm_exponent"] * cfg["n_x"] * n_y_ref * cfg["n_z"]
              if cfg is GP else cfg["C_m"] * d_y ** cfg["nm_exponent"])
    cut_ratio = (c[0] * c[1] * c[2]) / (c_ref[0] * c_ref[1] * c_ref[2])
    cells_ref = cfg["n_x"] * n_y_ref * cfg["n_z"]
    r_ref = r_here = float("nan")

    # ── step 7: transport to this machine's hourglass ratio (experimental) ──
    off_ratio = abs(hourglass / HOURGLASS_REF - 1.0) > HOURGLASS_TOL
    if mode == "model" and off_ratio:
        r_ref, r_here = R_pinch(d_y, HOURGLASS_REF), R_pinch(d_y, hourglass)
        n_y = _round_ny(ny_raw * r_here / r_ref)

    # occupancy of the pinched central cell is fixed by cells per sigma: n_m follows the cell
    # count, the inverse of the box volume, and the compression that concentrates the charge.
    # A box that grows together with its cell count therefore leaves n_m alone.
    nm_val = nm_ref * (n_x * n_y * n_z) / cells_ref / cut_ratio
    if mode == "model" and off_ratio:
        nm_val *= r_ref / r_here

    n_m = math.ceil(nm_val) if cfg is GP else _round_2sf(nm_val)

    r = Recommendation(code=cfg["code"], mode=mode, n_x=n_x, n_y=n_y, n_z=n_z,
                       n_m=n_m, D_y=d_y, D_x=d_x, hourglass=hourglass,
                       sigma_x_m=sx, sigma_y_m=sy, cuts=c, R_ref=r_ref, R_here=r_here)

    # ── what the study covered, and what it did not ──
    if off_ratio:
        if mode == "model":
            r.warnings.append(
                f"EXPERIMENTAL: beta_y*/sigma_z = {hourglass:.3f} differs from the "
                f"{HOURGLASS_REF} the study ran, and the configuration has been transported by "
                "the envelope model. That transport is untested.")
            warnings.warn(r.warnings[-1], ExperimentalRecommendation, stacklevel=2)
        else:
            r.warnings.append(
                f"beta_y*/sigma_z = {hourglass:.3f} differs from the {HOURGLASS_REF} the study "
                "ran: the compression R, and with it both normalisations, depends on this ratio "
                "as well as on D_y, so these constants are outside the geometry they were "
                "established on. Use mode='model' for the envelope-model estimate.")
            warnings.warn(r.warnings[-1], OutsideTestedRange, stacklevel=2)
    if c != c_ref:
        smaller = [ax for ax, v, vr in zip('xyz', c, c_ref) if v < vr]
        msg = (f"box extents (c_x, c_y, c_z) = {c} are not the {c_ref} this study ran; the "
               "cell counts have been scaled with them to hold the resolution.")
        if smaller:
            msg += (f" The box is smaller in {', '.join(smaller)}, so the collision may be "
                    "cropped; this study did not test that.")
        r.warnings.append(msg)
        warnings.warn(msg, OutsideTestedRange, stacklevel=2)
    lo, hi = D_Y_TESTED
    if not (lo <= d_y <= hi):
        r.warnings.append(f"D_y = {d_y:.1f} is outside the tested band [{lo}, {hi}].")
        warnings.warn(r.warnings[-1], OutsideTestedRange, stacklevel=2)
    if not (EPS_Y_TESTED_NM[0] <= eps_y_nm <= EPS_Y_TESTED_NM[1]):
        r.warnings.append(f"eps_y = {eps_y_nm:g} nm is outside the tuned range "
                          f"{EPS_Y_TESTED_NM[0]:g}-{EPS_Y_TESTED_NM[1]:g} nm.")
        warnings.warn(r.warnings[-1], OutsideTestedRange, stacklevel=2)
    if sy / sx > 10 * FLATNESS_REF:
        r.warnings.append(f"sigma_y*/sigma_x* = {sy / sx:.3f}: the beam is much less flat than "
                          f"the {REFERENCE['name']} ratio of {FLATNESS_REF}.")
        warnings.warn(r.warnings[-1], OutsideTestedRange, stacklevel=2)
    if d_x > 0.5:
        r.warnings.append(f"D_x = {d_x:.2f} is not small: the horizontal requirement is a "
                          "constant only while the horizontal plane is weakly disrupted.")
        warnings.warn(r.warnings[-1], OutsideTestedRange, stacklevel=2)
    if cfg is GP and n_m > GP_NM_WALLCLOCK_LIMIT:
        r.warnings.append(f"n_m = {n_m:,} exceeds the largest count reached within the "
                          "three-day wall-clock limit of this study.")
        warnings.warn(r.warnings[-1], OutsideTestedRange, stacklevel=2)
    if cfg is WX and n_m > WX_NM_MEMORY_LIMIT:
        r.warnings.append(f"n_m = {n_m:,} exceeds the GPU memory ceiling reached in this study.")
        warnings.warn(r.warnings[-1], OutsideTestedRange, stacklevel=2)
    return r


def _round_2sf(x: float) -> int:
    if x <= 0:
        return 0
    e = math.floor(math.log10(x))
    return int(round(x, -(e - 1)))


def selftest(verbose: bool = True) -> bool:
    """Reproduce both recommended-configuration tables, and R(D_y) against its reference."""
    worst = max(abs(R_pinch(d) - r) for d, r in R_REFERENCE.items())
    ok = worst < 5e-3
    if verbose:
        print(f"R(D_y) reproduces the reference values to {worst:.1e}")
    kw = {k: v for k, v in REFERENCE.items() if k != "name"}
    gp_table = {0.5: (512, 4.4e7), 1.0: (512, 1.4e7), 1.5: (512, 7.2e6), 2.0: (512, 4.5e6),
                2.5: (512, 3.1e6), 4.0: (256, 7.1e5), 8.0: (256, 2.2e5), 12: (256, 1.1e5),
                16: (256, 7.0e4), 20: (256, 4.8e4)}
    wx_table = {0.5: (512, 4.3e6), 1.0: (512, 1.4e6), 2.0: (256, 4.5e5), 4.0: (256, 1.4e5),
                8.0: (256, 4.6e4), 12: (256, 2.3e4), 16: (256, 1.4e4), 20: (256, 1.0e4)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                  # the 0.5 nm rows sit outside the band
        for code, table in (("GP", gp_table), ("WX", wx_table)):
            for eps, (n_y, n_m) in table.items():
                r = recommend(eps_y_nm=eps, code=code, **kw)
                good = (r.n_y == n_y) and math.isclose(_round_2sf(r.n_m), n_m, rel_tol=0.011)
                ok &= good
                if verbose and not good:
                    print(f"  MISMATCH {code} eps_y={eps}: n_y {r.n_y} vs {n_y}, "
                          f"n_m {_round_2sf(r.n_m):.2g} vs {n_m:.2g}")
    if verbose:
        print("both recommended-configuration tables reproduced" if ok else "SELFTEST FAILED")
    return ok


if __name__ == "__main__":
    selftest()
    kw = {k: v for k, v in REFERENCE.items() if k != "name"}
    print("\n" + str(recommend(eps_y_nm=8.0, code="GP", **kw)))
    print("\n" + str(recommend(eps_y_nm=8.0, code="WX", **kw)))
