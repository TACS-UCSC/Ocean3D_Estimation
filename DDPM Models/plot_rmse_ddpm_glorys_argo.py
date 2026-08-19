"""
Per-sample RMSE of the DDPM reconstruction against GLORYS and against Argo.

Layout: rows = the three reported depths (55.8 / 318.1 / 1062.4 m), columns =
variable. Each panel shows the same 100 days scored three ways:

    DDPM vs GLORYS, full field    every ocean cell   (~30,000 per level per day)
    DDPM vs GLORYS, at Argo pts   the Argo cells     (~6 per level per day)
    DDPM vs Argo,   at Argo pts   the Argo cells
    GLORYS vs Argo, at Argo pts   the Argo cells

The middle curve exists so the outer two are comparable: it uses exactly the
cells of the Argo curve, so the gap between them is a real GLORYS-vs-Argo
difference rather than a sampling artefact. The Argo curve is jagged for the
same reason -- a handful of floats per day is a small sample, not instability.

Reads : rmse_ddpm_2023_perday.npz  (written by rmse_ddpm_glorys_argo_2023.py)
Writes: RMSE_DDPM_vs_Argo_2023.png        window 0, 2023-02-13..05-23
        RMSE_DDPM_vs_Argo_2023_w1.png     window 1, 2023-09-01..12-09
        RMSE_profile_vs_Argo_2023.png     all 9 levels, both windows, and only
                                          the two curves that share a reference
                                          (Argo) and a sample (the Argo cells):
                                          DDPM vs Argo and GLORYS vs Argo
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 10, "axes.labelweight": "bold",
    "figure.titlesize": 13, "figure.titleweight": "bold",
    "xtick.labelsize": 9, "ytick.labelsize": 9,
})

ROOT = "/glade/derecho/scratch/nasefi/Ocean3D"
OUT_DIR = (f"{ROOT}/Results/ddpm_output/DDPM_Depth_min_max_level(1_to_9)"
           f"[SST, sal, u, v]_5Dcond_6ch_Obs[SSH,SSH, mask_ssh,land, "
           f"Log_norm_depth_ids]_pwr(2.0)_law(0.015)_hy(128,256)_1stDec_paper_4March"
           f"/fno_ddpm_timesteps")
NPZ = f"{OUT_DIR}/rmse_ddpm_2023_perday.npz"

SHOW_LEVELS = [2, 6, 9]                       # 55.8 m, 318.1 m, 1062.4 m
VARS   = [("thetao", "potential temperature", "RMSE (degC)"),
          ("so",     "salinity",              "RMSE (psu)")]
SERIES = [("glorys_full",    "DDPM vs GLORYS (full field)", "#1a1a1a", "-",  1.8, None),
          ("glorys_at_argo", "DDPM vs GLORYS (at Argo pts)", "#1f6fb4", "--", 1.3, None),
          ("argo",           "DDPM vs Argo",                 "#c0392b", "-",  1.3, "o"),
          ("glorys_vs_argo", "GLORYS vs Argo",               "#1e7d43", "-",  1.3, "^")]
LEVELS = list(range(1, 10))

# The profile figure keeps only the two Argo-referenced metrics: both are scored
# on exactly the same cells against the same floats, so the gap between them is
# the model-vs-reanalysis difference and nothing else.
PROFILE_SERIES = ["argo", "glorys_vs_argo"]


def pooled(v, n=None):
    v = np.asarray(v, float)
    ok = np.isfinite(v)
    if not ok.any():
        return np.nan
    if n is None:
        return float(np.sqrt((v[ok] ** 2).mean()))
    n = np.asarray(n, float)[ok]
    return float(np.sqrt((v[ok] ** 2 * n).sum() / max(n.sum(), 1)))


def per_sample_figure(d, w, out_png):
    depth_m = d["depth_m"]
    label   = str(d["labels"][w])
    fig, axes = plt.subplots(len(SHOW_LEVELS), len(VARS),
                             figsize=(13.5, 9.0), sharex=True)

    for r, l in enumerate(SHOW_LEVELS):
        for c, (var, title, ylab) in enumerate(VARS):
            ax = axes[r, c]
            n = d[f"w{w}_L{l}_{var}_count"]
            x = np.arange(1, n.size + 1)

            for key, lab, col, ls, lw, mk in SERIES:
                y = d[f"w{w}_L{l}_{var}_{key}"]
                ax.plot(x, y, color=col, ls=ls, lw=lw, marker=mk, ms=2.6,
                        alpha=0.95 if key != "glorys_at_argo" else 0.85,
                        label=f"{lab}   (pooled {pooled(y, None if key=='glorys_full' else n):.3f})")

            ax.set_ylim(bottom=0)
            ax.grid(True, color="0.9", lw=0.6)
            ax.set_axisbelow(True)
            ax.set_ylabel(ylab)
            ax.set_title(f"{title}   -   depth {depth_m[l]:.0f} m  (level {l})")
            ax.legend(fontsize=7.5, loc="upper right", frameon=True, framealpha=0.92)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if r == len(SHOW_LEVELS) - 1:
                ax.set_xlabel("sample index (consecutive days)")

    med = int(np.median([d[f"w{w}_L{l}_thetao_count"].mean() for l in SHOW_LEVELS]))
    fig.suptitle(f"DDPM reconstruction RMSE over {n.size} samples   |   {label}   |   "
                 f"~{med} Argo obs per level per day", y=0.995)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print("wrote", out_png)


def profile_figure(d, out_png):
    """RMSE vs depth, both curves scored against Argo at the Argo cells only.

    DDPM vs GLORYS is deliberately absent: that comparison is full-field against
    full-field, so it lives on a different sample and is not commensurate with
    the two observational curves.
    """
    depth_m = d["depth_m"]
    z = depth_m[LEVELS]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.4))
    wstyle = [("-", "o"), ("--", "s")]
    style = {k: (lab, col) for k, lab, col, _, _, _ in SERIES}

    for c, (var, title, xlab) in enumerate(VARS):
        ax = axes[c]
        for w in range(len(d["labels"])):
            ls, mk = wstyle[w % len(wstyle)]
            for key in PROFILE_SERIES:
                lab, col = style[key]
                v = [pooled(d[f"w{w}_L{l}_{var}_{key}"],
                            d[f"w{w}_L{l}_{var}_count"]) for l in LEVELS]
                ax.plot(v, z, color=col, ls=ls, lw=1.8, marker=mk, ms=5,
                        label=f"{lab} | w{w}")
        ax.set_ylim(1100, 0)
        ax.set_xlim(left=0)
        ax.grid(True, color="0.9", lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel(xlab)
        ax.set_ylabel("depth (m)")
        ax.set_title(f"{title}\nRMSE vs Argo at the Argo cells, 2023")
        ax.legend(fontsize=8, loc="lower right", frameon=True, framealpha=0.92)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    med = int(np.median([d[f"w0_L{l}_thetao_count"].mean() for l in LEVELS]))
    fig.suptitle("solid = window 0 (Feb-May)   dashed = window 1 (Sep-Dec)   |   "
                 f"scored only where Argo reports (~{med} obs per level per day)",
                 y=1.0)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print("wrote", out_png)


def main():
    d = np.load(NPZ, allow_pickle=True)
    per_sample_figure(d, 0, f"{OUT_DIR}/RMSE_DDPM_vs_Argo_2023.png")
    per_sample_figure(d, 1, f"{OUT_DIR}/RMSE_DDPM_vs_Argo_2023_w1.png")
    profile_figure(d, f"{OUT_DIR}/RMSE_profile_vs_Argo_2023.png")


if __name__ == "__main__":
    main()
