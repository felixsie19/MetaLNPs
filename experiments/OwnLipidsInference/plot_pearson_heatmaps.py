#!/usr/bin/env python3
import re
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def parse_sample(sample: str):
    s = str(sample)
    m = re.search(r"(sirna|mrna)", s, flags=re.I)
    if not m:
        return None, None
    cargo = "siRNA" if "si" in m.group(1).lower() else "mRNA"
    cell = (s[:m.start()] + s[m.end():]).strip(" _-")
    cell = re.sub(r"[_\-\s]+$", "", cell)
    cell = re.sub(r"^[_\-\s]+", "", cell)
    return cell, cargo

def make_heatmap(ax, data: pd.DataFrame, vmin=None, vmax=None):
    # Column order (if both exist)
    cols = [c for c in ["maml", "rf"] if c in data.columns] or list(data.columns)
    data = data.loc[:, cols]

    im = ax.imshow(data.values, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(data.index)))
    ax.set_xticklabels(cols)
    ax.set_yticklabels(list(data.index))

    # Subtle grid
    ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(data.index), 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=1, alpha=0.25)
    ax.tick_params(top=False, bottom=False, left=False, right=False)

    # Annotations
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data.iat[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center")
    # Cleaner look
    for sp in ax.spines.values():
        sp.set_visible(False)
    return im

def main():
    ap = argparse.ArgumentParser(description="Plot Pearson heatmaps for siRNA and mRNA")
    ap.add_argument("excel", type=Path, help="Excel file from collector (sheet 'long')")
    ap.add_argument("--sheet", default="long")
    ap.add_argument("-o", "--outprefix", default="pearson_heatmaps")
    args = ap.parse_args()

    df = pd.read_excel(args.excel, sheet_name=args.sheet)

    # Parse sample → cell_line, cargo
    parsed = df["sample"].apply(parse_sample)
    df[["cell_line", "cargo"]] = pd.DataFrame(parsed.tolist(), index=df.index)

    # Keep valid rows
    df = df[df["cell_line"].notna() & df["cargo"].notna()].copy()
    df["model"] = df["model"].str.strip().str.lower()

    # Split
    df_si = df[df["cargo"] == "siRNA"].copy()
    df_m  = df[df["cargo"] == "mRNA"].copy()

    # Pivot
    piv_si = df_si.pivot_table(index="cell_line", columns="model", values="pearson", aggfunc="mean")
    piv_m  = df_m .pivot_table(index="cell_line", columns="model", values="pearson", aggfunc="mean")

    # Drop all-NaN rows/cols (removes blank space)
    piv_si = piv_si.dropna(how="all", axis=0).dropna(how="all", axis=1)
    piv_m  = piv_m .dropna(how="all", axis=0).dropna(how="all", axis=1)

    # Optional: sort rows alphabetically for consistent order
    piv_si = piv_si.sort_index()
    piv_m  = piv_m.sort_index()

    # Shared color scale from actual values present
    all_vals = pd.concat([piv_si.stack(), piv_m.stack()], axis=0)
    if all_vals.empty:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = float(all_vals.min()), float(all_vals.max())
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = (vmin - 0.01, vmax + 0.01) if np.isfinite(vmin) else (0.0, 1.0)

    # --- siRNA (no title) ---
    fig1, ax1 = plt.subplots(figsize=(5.2, 3.6))
    im1 = make_heatmap(ax1, piv_si, vmin=vmin, vmax=vmax)
    cbar1 = fig1.colorbar(im1, ax=ax1)
    cbar1.set_label("Pearson r")
    fig1.tight_layout(pad=0.5)
    fig1.savefig(f"{args.outprefix}_siRNA.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
    fig1.savefig(f"{args.outprefix}_siRNA.pdf", bbox_inches="tight", pad_inches=0.1)

    # --- mRNA (no title) ---
    fig2, ax2 = plt.subplots(figsize=(5.2, 3.6))
    im2 = make_heatmap(ax2, piv_m, vmin=vmin, vmax=vmax)
    cbar2 = fig2.colorbar(im2, ax=ax2)
    cbar2.set_label("Pearson r")
    fig2.tight_layout(pad=0.5)
    fig2.savefig(f"{args.outprefix}_mRNA.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
    fig2.savefig(f"{args.outprefix}_mRNA.pdf", bbox_inches="tight", pad_inches=0.1)

    print("[OK] Wrote:")
    print(f"  {args.outprefix}_siRNA.png / .pdf")
    print(f"  {args.outprefix}_mRNA.png / .pdf")

if __name__ == "__main__":
    main()
