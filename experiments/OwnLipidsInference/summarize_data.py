#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple, List
import pandas as pd

PREFERRED_NAMES = ("metrics.json", "results.json", "scores.json")

def _casefold(s: str) -> str:
    return s.casefold()  # better than lower() for unicode
from pathlib import Path
import re

def _normalize_name(s: str) -> str:
    # collapse whitespace and casefold for robust matching
    return re.sub(r"\s+", "", s).casefold()

def resolve_model_dir(sample_dir: Path, target: str) -> Path | None:
    """
    Find a subfolder under `sample_dir` whose name equals `target`
    after stripping whitespace and casefolding.
    E.g., matches 'maml', 'MAML', 'maml ', '  maml', etc.
    """
    tgt = _normalize_name(target)
    for p in sample_dir.iterdir():
        if p.is_dir() and _normalize_name(p.name) == tgt:
            return p
    return None


def _iter_json_files(model_dir: Path, max_depth: int = 3) -> List[Path]:
    """
    Recursively find JSON files under model_dir (depth-limited).
    - Case-insensitive extension (.json / .JSON / .Json …)
    - Prefer well-known names (case-insensitive). If multiple, return them first
      in preference order, then the rest sorted by mtime (newest first).
    """
    if not (model_dir.exists() and model_dir.is_dir()):
        return []

    # Collect candidates up to max_depth
    candidates: List[Path] = []
    base_depth = len(model_dir.parts)
    for p in model_dir.rglob("*"):
        if p.is_file() and _casefold(p.suffix) == _casefold(".json"):
            depth = len(p.parts) - base_depth
            if depth <= max_depth:
                candidates.append(p)

    if not candidates:
        return []

    # First, put preferred names (case-insensitive) at the front, preserving order
    preferred = []
    others = []
    pref_fold = [_casefold(n) for n in PREFERRED_NAMES]
    for p in candidates:
        name_fold = _casefold(p.name)
        if name_fold in pref_fold:
            # maintain PREFERRED_NAMES order
            idx = pref_fold.index(name_fold)
            preferred.append((idx, p))
        else:
            others.append(p)

    preferred_sorted = [p for _, p in sorted(preferred, key=lambda t: t[0])]
    # For non-preferred, pick by newest first (mtime)
    others_sorted = sorted(others, key=lambda x: x.stat().st_mtime, reverse=True)

    return preferred_sorted + others_sorted

def _find_metric(
    obj: Any,
    key_variants: Iterable[str],
    nested_value_keys: Iterable[str] = ("r", "rho", "coef", "value", "score"),
    require_ancestor_hint: Optional[str] = None,
) -> Tuple[Optional[float], Optional[str]]:
    """
    Search for a numeric metric located under any of `key_variants`.
    - Matches are case-insensitive and allow substrings (e.g., 'pearson_r', 'spearmanr').
    - If the matched value is a dict, try nested_value_keys (e.g., {'r': 0.71}).
    - If `require_ancestor_hint` is set (like 'spearman'), then generic keys (e.g., 'rho')
      are only accepted if an ancestor key contains that hint (prevents 'rho' under pearson).
    Returns (value, path) where path is a dotted key path for debugging.
    """
    variants = {k.casefold() for k in key_variants}
    nested_keys = tuple(k.casefold() for k in nested_value_keys)

    def key_matches(k: str) -> bool:
        kf = k.casefold()
        return any(kf == v or v in kf for v in variants)

    best_val: Optional[float] = None
    best_path: Optional[str] = None

    def walk(node: Any, path: list[str], ancestors: list[str]):
        nonlocal best_val, best_path
        if isinstance(node, dict):
            for k, v in node.items():
                if best_val is not None:
                    return
                kstr = str(k)
                kf = kstr.casefold()

                # If this key matches our variants, try to extract a number here
                if key_matches(kstr):
                    if isinstance(v, (int, float)):
                        best_val = float(v)
                        best_path = ".".join(path + [kstr])
                        return
                    if isinstance(v, dict):
                        # Try common nested value keys
                        for nk in nested_keys:
                            if nk in {str(kk).casefold() for kk in v.keys()}:
                                vv = v.get(nk)
                                if isinstance(vv, (int, float)):
                                    best_val = float(vv)
                                    best_path = ".".join(path + [kstr, nk])
                                    return
                        # Otherwise, keep walking **within** the matched subtree
                        walk(v, path + [kstr], ancestors + [kstr])
                        return
                    # If it’s a list, keep walking within the matched subtree
                    if isinstance(v, list):
                        walk(v, path + [kstr], ancestors + [kstr])
                        return

                # If not a direct match, still walk, but with ancestors
                walk(v, path + [kstr], ancestors + [kstr])

        elif isinstance(node, list):
            for idx, item in enumerate(node):
                if best_val is not None:
                    return
                walk(item, path + [f"[{idx}]"], ancestors)

        else:
            # primitive: ignore unless we already matched a variant key (handled above)
            return

    walk(obj, [], [])

    # If not found and we’re allowed a **generic** key (like 'rho'), do a constrained search:
    if best_val is None and require_ancestor_hint:
        hint = require_ancestor_hint.casefold()

        def walk_generic(node: Any, path: list[str], ancestors: list[str]):
            nonlocal best_val, best_path
            if isinstance(node, dict):
                ancestors_cf = [a.casefold() for a in ancestors]
                has_hint = any(hint in a for a in ancestors_cf)
                for k, v in node.items():
                    if best_val is not None:
                        return
                    kf = str(k).casefold()
                    # Accept generic nested keys only if an ancestor contained the hint
                    if has_hint and kf in nested_keys and isinstance(v, (int, float)):
                        best_val = float(v)
                        best_path = ".".join(path + [str(k)])
                        return
                    walk_generic(v, path + [str(k)], ancestors + [str(k)])
            elif isinstance(node, list):
                for idx, item in enumerate(node):
                    if best_val is not None:
                        return
                    walk_generic(item, path + [f"[{idx}]"], ancestors)

        walk_generic(obj, [], [])

    return best_val, best_path
def collect(root: Path, verbose: bool = False) -> pd.DataFrame:
    records = []
    missing_rows = []

    # sample directories (sorted, case-insensitive)
    samples = [p for p in root.iterdir() if p.is_dir()]
    samples.sort(key=lambda x: _casefold(x.name))

    for sample_dir in samples:
        sample = sample_dir.name.strip()

        for model in ("maml", "rf"):
            # robustly resolve subdir (handles 'maml ' with trailing space, case, etc.)
            model_dir = resolve_model_dir(sample_dir, model)
            if model_dir is None:
                model_dir = sample_dir / model  # fallback

            jfiles = _iter_json_files(model_dir, max_depth=3)

            if verbose:
                if jfiles:
                    print(f"[FOUND] {sample}/{model}: {len(jfiles)} JSON → using {jfiles[0].name}")
                else:
                    print(f"[MISS]  {sample}/{model}: no JSON found under {model_dir}")

            if not jfiles:
                missing_rows.append((sample, model, "no json found", str(model_dir)))
                continue

            jp = jfiles[0]
            pearson, spearman = _extract_metrics(jp)

            if pearson is None and spearman is None:
                missing_rows.append((sample, model, "could not parse metrics", str(jp)))
                continue

            records.append(
                dict(
                    sample=sample,
                    model=model,
                    pearson=pearson,
                    spearman=spearman,
                    json_file=str(jp.relative_to(root)),
                )
            )

    df = pd.DataFrame.from_records(
        records, columns=["sample", "model", "pearson", "spearman", "json_file"]
    )
    if not df.empty:
        df = df.sort_values(["sample", "model"], ignore_index=True)

    miss_df = pd.DataFrame(missing_rows, columns=["sample", "model", "issue", "path"])
    df.attrs["missing"] = miss_df
    return df


def _extract_metrics(json_path: Path) -> Tuple[Optional[float], Optional[float]]:
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None, None

    pearson, _ = _find_metric(
        data,
        key_variants=("pearson", "pearson_r", "pearsonr", "r_pearson")
    )
    spearman, _ = _find_metric(
        data,
        key_variants=("spearman", "spearman_r", "spearmanr", "spearman_rho", "r_spearman"),
        require_ancestor_hint="spearman"  # only accept generic 'rho' if under a spearman-ancestor
    )
    return pearson, spearman


def write_excel(df: pd.DataFrame, out_path: Path) -> None:
    with pd.ExcelWriter(out_path) as xw:
        df.to_excel(xw, index=False, sheet_name="long")
        if not df.empty:
            df.pivot_table(index="sample", columns="model", values="pearson", aggfunc="mean").to_excel(xw, sheet_name="wide_pearson")
            df.pivot_table(index="sample", columns="model", values="spearman", aggfunc="mean").to_excel(xw, sheet_name="wide_spearman")
        miss = df.attrs.get("missing", pd.DataFrame())
        if not miss.empty:
            miss.to_excel(xw, index=False, sheet_name="missing")

def main():
    ap = argparse.ArgumentParser(description="Collect Pearson/Spearman from sample/maml|rf/**/*.json and export to Excel.")
    ap.add_argument("root", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("lnp_fewshot_vs_rf_metrics.xlsx"))
    ap.add_argument("-v", "--verbose", action="store_true", help="Print files discovered per sample/model")
    args = ap.parse_args()

    df = collect(args.root, verbose=args.verbose)
    write_excel(df, args.out)

    print(f"[OK] Wrote {args.out} with {len(df)} rows.")
    miss = df.attrs.get("missing")
    if miss is not None and not miss.empty:
        print("\n[WARN] Some entries were skipped:")
        print(miss.to_string(index=False))

if __name__ == "__main__":
    main()
