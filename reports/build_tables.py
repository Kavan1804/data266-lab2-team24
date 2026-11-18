"""Writes the report's results tables from reports/results/*.json.

Report numbers must come from notebook output, so nobody types them in by hand.
Missing values print as PENDING.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS, TABLES = HERE / "results", HERE / "tables"
TABLES.mkdir(exist_ok=True)


def load(name):
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else {}


def show(value, fmt="{:.2f}"):
    if value is None:
        return r"\textit{PENDING}"
    if isinstance(value, float) and value != value:  # NaN: not applicable (e.g. training fields for the base model)
        return "n/a"
    return fmt.format(value) if isinstance(value, float) else str(value)


def table(name, caption, label, header, rows):
    body = " \\\\\n".join(" & ".join(r) for r in rows)
    spec = "l" + "c" * (len(header) - 1)
    (TABLES / f"{name}.tex").write_text(
        f"\\begin{{table}}[!t]\n\\centering\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\begin{{tabular}}{{{spec}}}\n\\hline\n"
        + " & ".join(header) + " \\\\\n\\hline\n" + body + " \\\\\n\\hline\n\\end{tabular}\n\\end{table}\n")


def p1():
    data = load("p1_eval.json")
    base, final = data.get("baseline", {}), data.get("final", {})
    measures = [("Correct out of 25", "correct_of_25"), ("Recall@5", "recall_at_5"), ("Citation accuracy", "citation_accuracy"),
                ("Groundedness mean (1--5)", "groundedness_mean"), ("Agreement within one point", "groundedness_within_one"),
                ("Median seconds per question", "median_seconds_per_question")]
    fmt = lambda key: "{:.3f}" if key == "median_seconds_per_question" else "{:.2f}"
    final_only = {"groundedness_mean", "groundedness_within_one"}  # only the final system's answers are rated
    rows = [[label, "n/a" if key in final_only else show(base.get(key), fmt(key)), show(final.get(key), fmt(key))] for label, key in measures]
    table("p1", "Part 1 results on our 25 golden questions", "tab:p1", ["Measure", "Dense baseline", "Final hybrid"], rows)


def muhammed_p2():
    """Reads eval/p2_results.json (+ eval/p2_config.json if present) as written by p2_finetune_Muhammad.ipynb
    and renames its keys to the p2.json names. Values are copied, never computed here."""
    eval_dir = HERE.parent / "eval"
    res_path, cfg_path = eval_dir / "p2_results.json", eval_dir / "p2_config.json"
    if not res_path.exists():
        return {}
    res = json.loads(res_path.read_text())
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    rename = {"kid_all": "kid_mean", "kid_all_std": "kid_std", "clip_all": "clip_score", "lpips_diversity": "lpips_diversity",
              "train_gpu_hours": "gpu_hours", "peak_mem_gib": "peak_gpu_gb", "win_rate": "win_rate"}
    out = {}
    for src, dst in (("base", "base"), ("run_a", "run1"), ("run_b", "run2")):
        m = {new: res.get(src, {}).get(old) for old, new in rename.items()}
        params_m = res.get(src, {}).get("trainable_params_M")
        m["trainable_params"] = (f"{params_m:.2f}M" if params_m == params_m else params_m) if isinstance(params_m, float) else params_m
        run_cfg = cfg.get(src, {}) if isinstance(cfg.get(src), dict) else {}
        for key in ("learning_rate", "steps", "batch_size"):
            m[key] = run_cfg.get(key)
        out[dst] = m
    return out


def p2():
    data = load("p2.json") or muhammed_p2()
    models = [data.get(k, {}) for k in ("base", "run1", "run2")]
    measures = [("KID (mean)", "kid_mean", "{:.4f}"), ("KID (std)", "kid_std", "{:.4f}"), ("CLIP score", "clip_score", "{:.3f}"), ("LPIPS diversity", "lpips_diversity", "{:.3f}"),
                ("Trainable parameters", "trainable_params", "{}"), ("Learning rate", "learning_rate", "{}"), ("Steps", "steps", "{}"), ("Batch size", "batch_size", "{}"),
                ("GPU-hours", "gpu_hours", "{:.2f}"), ("Peak GPU memory (GiB)", "peak_gpu_gb", "{:.2f}"), ("Win rate vs base (overall preference)", "win_rate", "{:.2f}")]
    rows = [[label] + [show(m.get(key), fmt) for m in models] for label, key, fmt in measures]
    table("p2", "Part 2 results", "tab:p2", ["Measure", "Base", "Run 1 (run\\_a)", "Run 2 (run\\_b, final)"], rows)


def p2_repro():
    """Kavan's independent Part 2 baseline (one LoRA run), used as reproduction evidence."""
    data = load("p2_baseline_kavan.json")
    base, lora = data.get("base", {}), data.get("lora", {})
    rows = [["KID (mean)", show(base.get("kid_mean"), "{:.4f}"), show(lora.get("kid_mean"), "{:.4f}")],
            ["KID (std)", show(base.get("kid_std"), "{:.4f}"), show(lora.get("kid_std"), "{:.4f}")],
            ["CLIP score", show(base.get("clip_score"), "{:.3f}"), show(lora.get("clip_score"), "{:.3f}")],
            ["LPIPS diversity", show(base.get("lpips_diversity"), "{:.3f}"), show(lora.get("lpips_diversity"), "{:.3f}")],
            ["LoRA rank / steps / batch", "n/a", f"{show(data.get('rank'))} / {show(data.get('steps'))} / {show(data.get('batch_size'))}"],
            ["Trainable parameters", "n/a", show(data.get("trainable_params"))],
            ["GPU-hours (" + str(data.get("gpu", "GPU")) + ")", "n/a", show(data.get("gpu_hours"))],
            ["Peak GPU memory (GB)", "n/a", show(data.get("peak_gpu_gb"))]]
    table("p2_repro", "Part 2 reproduction: Kavan's independent baseline run (CLIP as cosine similarity, 0--1; Table~\\ref{tab:p2} uses the 0--100 scale)", "tab:p2repro", ["Measure", "Base", "LoRA (Kavan)"], rows)


def p3():
    data = load("p3.json")
    base, final = data.get("baseline", {}), data.get("final", {})
    rows = [["Scenarios passed (of 15)", show(base.get("scenarios_passed")), show(final.get("scenarios_passed"))],
            ["Human pass/fail agreement", show(base.get("human_agreement")), show(final.get("human_agreement"))],
            ["Critic agreement", "n/a", show(final.get("critic_agreement"))],
            ["Median seconds per request", show(base.get("median_seconds_per_request")), show(final.get("median_seconds_per_request"))],
            ["Red-team attacks that succeeded", "n/a", f"{show(data.get('red_team_attack_success'))} of {show(data.get('red_team_cases'))}"]]
    table("p3", "Part 3 results", "tab:p3", ["Measure", "No critique", "With critique"], rows)


if __name__ == "__main__":
    p1(), p2(), p2_repro(), p3()
    print("wrote tables to", TABLES)
