#!/usr/bin/env python
# coding: utf-8

"""
Yield tables from the cf.SelectEvents outputs written by tests/run_analysis, used to compare two
versions of the analysis code (e.g. before and after a refactoring). Only reads existing outputs.

Usage, inside the analysis environment ("source setup.sh <name>"):

    # 1. run the tests with the code version to check, and note the printed version (ci_test_<date>_<time>)
    bash tests/run_analysis

    # 2. dump the yields of that version to a json file (needs awkward, hence the sandbox)
    cf_sandbox venv_multilepton_dev \
        "python tests/yield_table.py --version ci_test_2026-09-17_08-18 --output yields_new.json"

    # 3. compare two dumps, which can come from different people and stores (plain python is enough)
    python tests/yield_table.py --compare yields_old.json yields_new.json

Only the outputs are read, nothing is run. Use --tests to restrict the config:dataset pairs.
"""

from __future__ import annotations

import os
import re
import sys
import json
import argparse
from collections import defaultdict


this_dir = os.path.dirname(os.path.abspath(__file__))


def get_run_analysis_tests() -> list[tuple[str, str]]:
    """
    Returns the (config, dataset) pairs defined in tests/run_analysis, so that both stay in sync.
    """
    with open(os.path.join(this_dir, "run_analysis")) as f:
        content = f.read()
    block = re.search(r"local tests=\((.*?)\)", content, re.S).group(1)
    return [tuple(entry.split(":")) for entry in re.findall(r'"([^"]+:[^"]+)"', block)]


def compute_yields(version: str, config: str, dataset: str, selector: str) -> dict:
    # note: columnflow (and with it the wlcg file system) must be imported before awkward, otherwise
    # remote outputs are reported as missing in the venv_multilepton_dev sandbox
    from columnflow.tasks.selection import SelectEvents
    import awkward as ak

    task = SelectEvents(
        version=version,
        config=config,
        dataset=dataset,
        selector=selector,
        limit_dataset_files=1,
        workflow="local",
        branch=0,
    )
    outputs = task.output()
    if not outputs["results"].exists():
        raise FileNotFoundError(f"missing output {outputs['results'].uri()}")

    def load_parquet(target):
        with target.localize("r") as tmp:
            return ak.from_parquet(tmp.abspath)

    results = load_parquet(outputs["results"])
    columns = load_parquet(outputs["columns"])
    config_inst = task.config_inst
    is_mc = task.dataset_inst.is_mc
    weight = columns.mc_weight if is_mc else ak.ones_like(columns.channel_id, dtype=float)

    def count(mask) -> list[float]:
        return [int(ak.sum(mask)), float(ak.sum(weight[mask]))]

    selected = results.event
    yields = {
        "all": count(ak.ones_like(selected)),
        "selected": count(selected),
        # events passing each single selection step
        "steps": {step: count(results.steps[step]) for step in results.steps.fields},
        "channels": {},
        "categories": {},
    }

    # selected events per channel, with the per-channel flags filled by the lepton selection
    channel_names = {ch.id: ch.name for ch in config_inst.channels}
    sel = columns[selected]
    sel_weight = weight[selected]
    for ch_id in sorted(set(ak.to_list(sel.channel_id))):
        in_ch = sel.channel_id == ch_id
        name = "none (0)" if ch_id == 0 else channel_names.get(ch_id, f"unknown ({ch_id})")
        yields["channels"][name] = {
            flag: [int(ak.sum(mask)), float(ak.sum(sel_weight[mask]))]
            for flag, mask in [
                ("events", in_ch),
                ("tight", in_ch & sel.tight_sel),
                ("trig_match", in_ch & sel.trig_match),
                ("tight_trig_match", in_ch & sel.tight_sel & sel.trig_match),
                ("leptons_os", in_ch & sel.leptons_os),
            ]
        }

    # selected events per category id
    flat_ids = ak.flatten(sel.category_ids)
    flat_weight = ak.flatten(ak.broadcast_arrays(sel_weight, sel.category_ids)[0])
    per_cat = defaultdict(lambda: [0, 0.0])
    for cat_id, w in zip(ak.to_list(flat_ids), ak.to_list(flat_weight)):
        per_cat[cat_id][0] += 1
        per_cat[cat_id][1] += w
    for cat_id, (n, w) in sorted(per_cat.items()):
        name = config_inst.get_category(cat_id).name if config_inst.has_category(cat_id) else str(cat_id)
        yields["categories"][name] = [n, w]

    return yields


def fmt(value: list[float], is_mc: bool) -> str:
    return f"{value[0]:>9d}" + (f" ({value[1]:.4g})" if is_mc else "")


def print_yields(key: str, yields: dict) -> None:
    is_mc = "data_" not in key
    print(f"\n=== {key} ===   (events{' (sum mc_weight)' if is_mc else ''})")
    print(f"  {'all':<30s} {fmt(yields['all'], is_mc)}")
    print(f"  {'selected':<30s} {fmt(yields['selected'], is_mc)}")
    print("  -- steps")
    for name, value in yields["steps"].items():
        print(f"  {name:<30s} {fmt(value, is_mc)}")
    print("  -- channels (selected)      events  tight  trig_match  tight&trig  leptons_os")
    for name, flags in yields["channels"].items():
        print(f"  {name:<22s} " + "  ".join(f"{flags[f][0]:>9d}" for f in flags))
    print("  -- categories (selected)")
    for name, value in yields["categories"].items():
        print(f"  {name:<30s} {fmt(value, is_mc)}")


def compare(file_a: str, file_b: str, rel_tol: float = 1e-6) -> int:
    with open(file_a) as f:
        a = json.load(f)
    with open(file_b) as f:
        b = json.load(f)

    def flatten(d, prefix=""):
        for k, v in d.items():
            if isinstance(v, dict):
                yield from flatten(v, f"{prefix}{k} / ")
            else:
                yield f"{prefix}{k}", v

    n_diff = 0
    print(f"A: {file_a} (version {a['_meta']['version']})")
    print(f"B: {file_b} (version {b['_meta']['version']})")
    for key in sorted((set(a) | set(b)) - {"_meta"}):
        if key not in a or key not in b:
            print(f"\n=== {key}: only in {'A' if key in a else 'B'}")
            n_diff += 1
            continue
        if "error" in a[key] or "error" in b[key]:
            print(f"\n=== {key}: error A={a[key].get('error')} B={b[key].get('error')}")
            n_diff += 1
            continue
        fa, fb = dict(flatten(a[key])), dict(flatten(b[key]))
        rows = []
        for name in sorted(set(fa) | set(fb)):
            va, vb = fa.get(name), fb.get(name)
            if va is None or vb is None:
                rows.append(f"  {name:<50s} {str(va):>22s} {str(vb):>22s}")
                continue
            same_n = va[0] == vb[0]
            same_w = abs(va[1] - vb[1]) <= rel_tol * max(abs(va[1]), abs(vb[1]), 1e-12)
            if not (same_n and same_w):
                rows.append(
                    f"  {name:<50s} {va[0]:>10d} {va[1]:>11.5g} {vb[0]:>10d} {vb[1]:>11.5g}"
                    f"   diff {vb[0] - va[0]:+d}",
                )
        status = "IDENTICAL" if not rows else f"{len(rows)} DIFFERENCES"
        print(f"\n=== {key}: {status}")
        if rows:
            print(f"  {'':<50s} {'A events':>10s} {'A weight':>11s} {'B events':>10s} {'B weight':>11s}")
            print("\n".join(rows))
        n_diff += bool(rows)

    print(f"\n{n_diff} of {len((set(a) | set(b)) - {'_meta'})} config/dataset pairs differ")
    return 1 if n_diff else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", help="version used by tests/run_analysis (ci_test_<date>_<time>)")
    parser.add_argument("--selector", default="default", help="selector, default: %(default)s")
    parser.add_argument("--tests", nargs="*", metavar="CONFIG:DATASET",
        help="config:dataset pairs, default: the ones in tests/run_analysis")
    parser.add_argument("--output", help="json file to write the yields to")
    parser.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"), help="compare two yield files")
    args = parser.parse_args()

    if args.compare:
        return compare(*args.compare)
    if not args.version:
        parser.error("either --version or --compare is required")

    tests = [tuple(t.split(":")) for t in args.tests] if args.tests else get_run_analysis_tests()
    all_yields = {"_meta": {"version": args.version, "selector": args.selector, "user": os.environ.get("USER")}}
    for config, dataset in tests:
        key = f"{config}/{dataset}"
        try:
            all_yields[key] = compute_yields(args.version, config, dataset, args.selector)
        except Exception as e:
            print(f"\n=== {key} === FAILED: {e}")
            all_yields[key] = {"error": str(e)}
            continue
        print_yields(key, all_yields[key])

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_yields, f, indent=1)
        print(f"\nyields written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
