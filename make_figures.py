#!/usr/bin/env python3
"""Regenerate every figure in plots/ from the datasets in data/.

    python make_figures.py            # figures that need only data/ (all but three)
    python make_figures.py --dumps    # also the three that read the per-step particle dumps (see README)

Figures from data/ only: WX_L_vs_nm, WX_nm_req_calibration, WX_L_vs_ny, WX_ny_req_calibration, kappa_vs_Dy,
WX_GP_comparison (paper_figures); WX_comparison_L_vs_ey, WX_comparison_HD_vs_ey, WX_comparison_ratio_vs_ey
(plot_comparison); WX_extension_L_vs_ey, WX_code_ratio_vs_ey (plot_extension); WX_deposition_L_vs_ey
(plot_deposition); WX_coarse_grid_L_vs_nm (plot_coarse); nx_nm_coupling (plot_nx_nm_coupling);
WX_core_width_vs_Dy (plot_core_width, from the cached slice widths); WX_depo_correlation (plot_depo_correlation).
Dump-based: WX_pinch_evolution, WX_pinch_histogram (plot_pinch_evolution), WX_depo_error_passes (deposition_error)."""
import os, sys, runpy
os.environ.setdefault("WX_CACHED_ONLY", "1")          # plot_core_width: use the cached slice widths, never the dumps
import paper_figures as PF
PF.make_all()
import plot_comparison as PC; PC.draw("L"); PC.draw("HD"); PC.draw_ratio()
import plot_extension as PX; PX.draw(); PX.draw_code_ratio()
import plot_deposition as PD; PD.draw()
import plot_coarse as PCo; PCo.draw() if hasattr(PCo, "draw") else None
runpy.run_path("plot_nx_nm_coupling.py", run_name="__main__")
import plot_core_width as PCW; PCW.draw()
import plot_depo_correlation as PDC; PDC.draw() if hasattr(PDC, "draw") else runpy.run_path("plot_depo_correlation.py", run_name="__main__")
if "--dumps" in sys.argv:
    import plot_pinch_evolution as PPE; PPE.draw(); PPE.histogram()
    import deposition_error as DE; DE.main() if hasattr(DE, "main") else runpy.run_path("deposition_error.py", run_name="__main__")
print("done")
