#!/usr/bin/env python3
"""
Error rate by drug (out-of-fold predictions of the multimodal model)
====================================================================

Input
-----
multimodal_oof_predictions.csv
    Written by the nested 5-fold CV notebook (PLOS_Multimodal_nestedCV.ipynb).
    Required columns: Drug, y_true_code, y_pred_code

    Each of the 622 diplotype-drug records appears exactly once, predicted by
    the model of the outer fold in which it was held out.

Drugs shown
-----------
Only drugs with at least MIN_N records (default 5) and at least one error are
plotted, sorted by increasing error rate.  Drugs with too few records would
produce error rates of 0 or 1 that carry no information.

Output
------
fig_drug_error_rate.pdf   vector
fig_drug_error_rate.png   600 dpi

Usage
-----
    python plot_figure_drug_error_rate.py
    python plot_figure_drug_error_rate.py --min-n 5 --outdir figures/
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

DPI = 600

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


def error_rate_by_drug(path, min_n):
    """Per-drug error counts from the stored out-of-fold predictions."""
    df = pd.read_csv(path)
    for col in ("Drug", "y_true_code", "y_pred_code"):
        if col not in df.columns:
            raise KeyError(f"'{col}' column missing in {path}")

    drug_df = pd.DataFrame({
        "Drug": df["Drug"].values,
        "error": (df["y_true_code"].values != df["y_pred_code"].values).astype(int),
    })
    grp = drug_df.groupby("Drug").agg(n=("error", "size"), errors=("error", "sum"))
    grp["error_rate"] = grp["errors"] / grp["n"]

    kept = grp[(grp["n"] >= min_n) & (grp["errors"] > 0)].sort_values("error_rate")
    return grp, kept


def format_drug_label(drug):
    """Capitalise and wrap long drug names onto two lines."""
    drug = str(drug)
    drug = drug[:1].upper() + drug[1:]

    if len(drug) > 18:
        split_pos = drug.rfind(" ", 0, 16)   # prefer breaking at a space
        if split_pos <= 0:
            split_pos = 18
        drug = drug[:split_pos] + "\n" + drug[split_pos:].lstrip()

    return drug


def plot_error_rate(kept, out_basename, dpi=DPI):
    drugs = kept.index.tolist()
    error_rates = kept["error_rate"].to_numpy()
    labels = [f"{e}/{n}" for e, n in zip(kept["errors"], kept["n"])]
    y = np.arange(len(drugs))

    # Red gradient keyed to the error rate; floor at 0.3 so low bars stay visible.
    cmap = matplotlib.colormaps["Reds"]
    norm = Normalize(vmin=error_rates.min(), vmax=error_rates.max())
    bar_colors = [cmap(0.3 + 0.7 * norm(v)) for v in error_rates]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.42 * len(drugs))))
    ax.barh(y, error_rates, color=bar_colors, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([format_drug_label(d) for d in drugs], fontsize=12)
    ax.set_xlabel("Error rate", fontsize=18)
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", linestyle="--", alpha=0.4, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # errors/n annotation at the end of each bar
    for yi, er, lab in zip(y, error_rates, labels):
        ax.text(er + 0.015, yi, lab, va="center", fontsize=12)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Error rate", fontsize=18)

    ax.set_title("Error Rate by Drug", fontsize=24, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{out_basename}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{out_basename}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", default="multimodal_oof_predictions.csv",
                        help="out-of-fold prediction CSV from the nested CV notebook")
    parser.add_argument("--min-n", type=int, default=5,
                        help="minimum number of records for a drug to be plotted")
    parser.add_argument("--outdir", default=".", help="directory for the figure files")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_basename = str(outdir / "fig_drug_error_rate")

    grp, kept = error_rate_by_drug(args.oof, args.min_n)
    plot_error_rate(kept, out_basename)

    print("=" * 64)
    print("[Error rate by drug] out-of-fold predictions from nested 5-fold CV")
    print(f"Records total        = {int(grp['n'].sum())}")
    print(f"Errors total         = {int(grp['errors'].sum())}")
    print(f"Drugs total          = {len(grp)}")
    print(f"Drugs plotted        = {len(kept)}  (n >= {args.min_n} and at least one error)")
    print("-" * 64)
    for drug, row in kept.iterrows():
        print(f"  {str(drug):<28} {int(row['errors']):>3}/{int(row['n']):<4} "
              f"= {row['error_rate']:.3f}")
    print("=" * 64)
    print(f"saved {out_basename}.{{pdf,png}}")


if __name__ == "__main__":
    main()
