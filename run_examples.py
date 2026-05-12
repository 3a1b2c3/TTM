
"""
Defaults to running every example in ./examples/ with the appropriate model
(wan / cog / svd) inferred from the directory prefix.

Usage:
    python run_examples.py                        # run all
    python run_examples.py --only Monkey Birds    # only matching example names
    python run_examples.py --model wan            # only wan examples
    python run_examples.py --tweak-index 4 --tstrong-index 9   # override indices
    python run_examples.py --start-index 5        # skip the first 5 examples (resume)
    python run_examples.py --dry-run              # print commands without executing
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Bump HF download timeout (default 10s) before importing huggingface_hub so it picks up the value.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
# Enable hf_transfer (rust-based, much faster + more robust than the default Python downloader).
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
# Silence the symlink warning on Windows (we accept the file-copy fallback).
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
# Disable transformers 5.x async tensor loading — its ThreadPoolExecutor workers can die on Windows
# without setting future results, leaving Future.result() to raise TimeoutError mid-load.
# Force-assign (not setdefault) so an externally-set "0"/"false" doesn't override.
os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
print(f"[env] HF_DEACTIVATE_ASYNC_LOAD={os.environ.get('HF_DEACTIVATE_ASYNC_LOAD')!r}", flush=True)

REPO = Path(__file__).resolve().parent
VENV_PY = REPO / ".venv" / "Scripts" / "python.exe"
EXAMPLES = REPO / "examples"
OUTPUTS = REPO / "outputs"

DEFAULTS = {
    "wan_cutdrag":    {"tweak": 8,  "tstrong": 12},
    "wan_camcontrol": {"tweak": 2,  "tstrong": 5},
    "wan21":          {"tweak": 0,  "tstrong": 0},
    "ti2v5b":         {"tweak": 0,  "tstrong": 0},
    "cog":            {"tweak": 4,  "tstrong": 9},
    "svd":            {"tweak": 16, "tstrong": 21},
}

SCRIPTS = {"wan": "run_wan.py", "wan21": "run_wan21.py", "ti2v5b": "run_ti2v5b.py", "cog": "run_cog.py", "svd": "run_svd.py"}

MODEL_IDS = {
    "wan": "lopho/Wan2.2-I2V-A14B-Diffusers_nf4",
    "wan21": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "ti2v5b": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    "cog": "THUDM/CogVideoX-5b-I2V",
    "svd": "stabilityai/stable-video-diffusion-img2vid-xt",
}

# Extra repos to prefetch alongside the main MODEL_IDS entry for a given model.
# Wan uses the broken `_nf4` repo as the pipeline shell but pulls the actual I2V
# transformer weights from these two split repos (which have the correct in_channels=36).
EXTRA_PREFETCH_REPOS = {
    "wan": [
        "lopho/Wan2.2-I2V-A14B-Diffusers_nf4_transformer",
        "lopho/Wan2.2-I2V-A14B-Diffusers_nf4_transformer_2",
    ],
}

CLIP_FRAMES = {"wan": 81, "wan21": 81, "ti2v5b": 81, "cog": 49, "svd": 21}


def write_example_stats(name: str, model: str, elapsed: float, rc: int, *,
                        tweak: int | None = None, tstrong: int | None = None,
                        output: Path | None = None) -> Path:
    """Write a one-example stats file alongside the per-example output mp4."""
    frames = CLIP_FRAMES.get(model, 0)
    fps = (frames / elapsed) if rc == 0 and elapsed > 0 and frames else 0.0
    status = "OK" if rc == 0 else f"ERR{rc}"
    path = OUTPUTS / f"{model}_{name}_stats.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"name: {name}\n")
        f.write(f"model: {model}\n")
        if tweak is not None:
            f.write(f"tweak_index: {tweak}\n")
        if tstrong is not None:
            f.write(f"tstrong_index: {tstrong}\n")
        if output is not None:
            f.write(f"output: {output}\n")
        f.write(f"frames: {frames}\n")
        f.write(f"elapsed_s: {elapsed:.3f}\n")
        f.write(f"elapsed_min: {elapsed/60:.3f}\n")
        f.write(f"fps: {fps:.3f}\n")
        f.write(f"status: {status}\n")
        f.write(f"rc: {rc}\n")
    return path


def model_for(example_name: str):
    if (
        example_name.startswith("camcontrol_")
        or example_name.startswith("cutdrag_wan_")
        or example_name.startswith("C2R")
    ):
        return "wan"
    if example_name.startswith("wan21_"):
        return "wan21"
    if example_name.startswith("ti2v5b_"):
        return "ti2v5b"
    if example_name.startswith("cutdrag_cog_"):
        return "cog"
    if example_name.startswith("cutdrag_svd_"):
        return "svd"
    return None


def defaults_key(model: str, example_name: str) -> str:
    if model == "wan":
        return "wan_camcontrol" if example_name.startswith("camcontrol_") else "wan_cutdrag"
    return model


def category_for(example_name: str) -> str:
    return "camera" if example_name.startswith("camcontrol_") else "motion"


def prefetch_models(models: set[str], dry_run: bool, max_attempts: int = 8):
    if not models:
        return
    repos_per_model = {
        m: [MODEL_IDS[m]] + EXTRA_PREFETCH_REPOS.get(m, [])
        for m in sorted(models)
    }
    total_repos = sum(len(v) for v in repos_per_model.values())
    print(f"\nPre-downloading {len(models)} model(s) across {total_repos} repo(s): "
          + ", ".join(f"{m}={'+'.join(r)}" for m, r in repos_per_model.items()))
    if dry_run:
        for m, repos in repos_per_model.items():
            for r in repos:
                print(f"    [dry-run] would download {r}")
        return
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    for m, repos in repos_per_model.items():
        for repo_id in repos:
            print(f"  -> {m}: {repo_id}")
            info = api.model_info(repo_id, files_metadata=True)
            siblings = [(s.rfilename, s.size or 0) for s in info.siblings]
            total_bytes = sum(sz for _, sz in siblings)
            print(f"     {len(siblings)} files, {total_bytes/1e9:.1f} GB total")
            done_bytes = 0
            for i, (fname, sz) in enumerate(siblings, 1):
                print(f"     [{i}/{len(siblings)}] {fname}  ({sz/1e6:.1f} MB)  cumulative {done_bytes/1e9:.1f}/{total_bytes/1e9:.1f} GB", flush=True)
                for attempt in range(1, max_attempts + 1):
                    start = time.perf_counter()
                    try:
                        hf_hub_download(repo_id=repo_id, filename=fname)
                        el = time.perf_counter() - start
                        rate = (sz / 1e6 / el) if el > 0 and sz > 0 else 0.0
                        print(f"          ok in {el:.1f}s ({rate:.1f} MB/s)", flush=True)
                        done_bytes += sz
                        break
                    except Exception as e:
                        el = time.perf_counter() - start
                        print(f"          attempt {attempt}/{max_attempts} failed after {el:.1f}s: {type(e).__name__}: {e}", flush=True)
                        if attempt == max_attempts:
                            raise
                        time.sleep(min(2 ** attempt, 30))


def run_wan_batch(example_dirs: list[Path], args, python_exe: Path):
    """
    Run all wan examples in-process (model loaded once).
    Returns list of (name, elapsed_seconds, rc) tuples.
    """
    import traceback
    from types import SimpleNamespace

    entries = []
    for ex in example_dirs:
        key = defaults_key("wan", ex.name)
        tweak = args.tweak_index if args.tweak_index is not None else DEFAULTS[key]["tweak"]
        tstrong = args.tstrong_index if args.tstrong_index is not None else DEFAULTS[key]["tstrong"]
        output = OUTPUTS / f"wan_{ex.name}_tw{tweak}_ts{tstrong}.mp4"
        entries.append({
            "name": ex.name,
            "input": str(ex),
            "output": str(output),
            "tweak": tweak,
            "tstrong": tstrong,
        })

    print(f"\n==> wan batch | {len(entries)} examples (model loaded once)")
    for e in entries:
        print(f"    - {e['name']} (tweak={e['tweak']}, tstrong={e['tstrong']}) -> {e['output']}")
    if args.dry_run:
        return [(e["name"], 0.0, 0) for e in entries]

    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import run_wan

    # Belt-and-suspenders: even if HF_DEACTIVATE_ASYNC_LOAD didn't take, force the loader to be sync.
    # On Windows, the ThreadPoolExecutor in transformers.core_model_loading dies silently.
    try:
        import transformers.core_model_loading as _cml
        _cml.GLOBAL_WORKERS = 1
        _orig_spawn = _cml.spawn_materialize
        _orig_spawn_tp = _cml.spawn_tp_materialize
        def _sync_spawn_materialize(thread_pool, *a, **kw):
            return _orig_spawn(None, *a, **kw)
        def _sync_spawn_tp_materialize(thread_pool, *a, **kw):
            return _orig_spawn_tp(None, *a, **kw)
        _cml.spawn_materialize = _sync_spawn_materialize
        _cml.spawn_tp_materialize = _sync_spawn_tp_materialize
        print("[env] forced transformers.core_model_loading to sync (thread_pool=None)", flush=True)
    except Exception as _e:
        print(f"[env] could not patch transformers core_model_loading: {_e}", flush=True)

    run_args = SimpleNamespace(
        device="cuda",
        seed=0,
        num_inference_steps=50,
        max_area=320 * 544,
        num_frames=81,
        guidance_scale=1.5,
        negative_prompt=(
            "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，"
            "低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
            "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
        ),
    )
    pipe = run_wan.setup_wan_pipeline(run_wan.MODEL_ID, run_wan.DTYPE, run_args.device)

    out = []
    for e in entries:
        name = e["name"]
        print(f"\n=== {name} ===", flush=True)
        elapsed = 0.0
        try:
            elapsed = run_wan.run_one(pipe, run_args, e["input"], e["output"], e["tweak"], e["tstrong"], name=name)
            rc = 0
        except Exception:
            traceback.print_exc()
            rc = 1
        print(f"=== {name}: inference {elapsed:.1f}s rc={rc} ===", flush=True)
        if rc == 0:
            fps = CLIP_FRAMES["wan"] / elapsed if elapsed > 0 else 0.0
            print(f"    {name}: {elapsed:.1f}s ({elapsed/60:.2f} min)  fps: {fps:.3f}")
        else:
            print(f"    {name}: {elapsed:.1f}s  FAILED (rc={rc})")
        stats_path = write_example_stats(
            name, "wan", elapsed, rc,
            tweak=e["tweak"], tstrong=e["tstrong"], output=Path(e["output"]),
        )
        print(f"    stats: {stats_path}")
        out.append((name, elapsed, rc))
    return out


def run_one(example_dir: Path, args, python_exe: Path, forced_model: str | None = None):
    name = example_dir.name
    model = forced_model if forced_model else model_for(name)
    if model is None:
        print(f"[skip] {name}: unknown prefix")
        return None, 0.0, None
    if args.model and model != args.model:
        return None, 0.0, None
    if model in args.skip:
        return None, 0.0, None

    key = defaults_key(model, name)
    tweak = args.tweak_index if args.tweak_index is not None else DEFAULTS[key]["tweak"]
    tstrong = args.tstrong_index if args.tstrong_index is not None else DEFAULTS[key]["tstrong"]

    output = OUTPUTS / f"{model}_{name}_tw{tweak}_ts{tstrong}.mp4"
    cmd = [
        str(python_exe),
        str(REPO / SCRIPTS[model]),
        "--input-path", str(example_dir),
        "--output-path", str(output),
        "--tweak-index", str(tweak),
        "--tstrong-index", str(tstrong),
    ]
    print(f"\n==> {model} | {name} | tweak={tweak} tstrong={tstrong}")
    print(f"    output: {output}")
    if args.dry_run:
        print("    cmd:", " ".join(cmd))
        return 0, 0.0, model
    start = time.perf_counter()
    rc = subprocess.call(cmd, cwd=str(REPO))
    elapsed = time.perf_counter() - start
    if rc == 0:
        fps = CLIP_FRAMES[model] / elapsed if elapsed > 0 else 0.0
        print(f"    elapsed: {elapsed:.1f}s ({elapsed/60:.2f} min)  fps: {fps:.3f} ({CLIP_FRAMES[model]} frames)")
    else:
        print(f"    elapsed: {elapsed:.1f}s ({elapsed/60:.2f} min)  FAILED (exit {rc}) — fps not reported")
    stats_path = write_example_stats(
        name, model, elapsed, rc, tweak=tweak, tstrong=tstrong, output=output,
    )
    print(f"    stats: {stats_path}")
    return rc, elapsed, model


def main():
    parser = argparse.ArgumentParser(description="Run TTM examples")
    parser.add_argument("--model", choices=["wan", "wan21", "ti2v5b", "cog", "svd"], help="Limit to one model")
    parser.add_argument("--force-model", choices=["wan", "wan21", "ti2v5b", "cog", "svd"], default=None, help="Force every example folder to use this model (overrides model_for routing).")
    parser.add_argument("--skip", nargs="*", choices=["wan", "wan21", "ti2v5b", "cog", "svd"], default=["cog", "svd"], help="Skip these models (examples + downloads). Default skips cog and svd; pass --skip with no args to skip nothing.")
    parser.add_argument("--only", nargs="+", help="Only run examples whose name contains any of these substrings")
    parser.add_argument("--tweak-index", type=int, help="Override default tweak-index")
    parser.add_argument("--tstrong-index", type=int, help="Override default tstrong-index")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--no-prefetch", action="store_true", help="Skip pre-downloading model weights")
    parser.add_argument("--start-index", type=int, default=0, help="Skip examples before this 0-based index (after --only filter, in sorted order)")
    args = parser.parse_args()

    python_exe = VENV_PY if VENV_PY.exists() else Path(sys.executable)
    if not VENV_PY.exists():
        print(f"[warn] venv not found at {VENV_PY}; using {python_exe}", file=sys.stderr)

    OUTPUTS.mkdir(exist_ok=True)

    examples = sorted(
        (p for p in EXAMPLES.iterdir() if p.is_dir()),
        key=lambda p: (p.name != "C2R_gui", not p.name.startswith("C2R"), p.name.startswith("camcontrol_"), p.name),
    )
    if args.only:
        examples = [p for p in examples if any(s.lower() in p.name.lower() for s in args.only)]

    if not examples:
        print("No examples matched.", file=sys.stderr)
        return 1

    if args.start_index:
        if args.start_index >= len(examples):
            print(f"--start-index {args.start_index} is past the last example (have {len(examples)}).", file=sys.stderr)
            return 1
        skipped = [p.name for p in examples[:args.start_index]]
        print(f"Skipping {len(skipped)} example(s) before index {args.start_index}: {', '.join(skipped)}")
        examples = examples[args.start_index:]

    def effective_model_for(name: str):
        if args.force_model:
            return args.force_model
        return model_for(name)

    if not args.no_prefetch:
        needed_models = set()
        for ex in examples:
            m = effective_model_for(ex.name)
            if m is None:
                continue
            if args.model and m != args.model:
                continue
            if m in args.skip:
                continue
            needed_models.add(m)
        prefetch_models(needed_models, args.dry_run)

    print(f"Running {len(examples)} example(s)")
    timings = []
    failures = []
    total_start = time.perf_counter()

    wan_examples = [ex for ex in examples
                    if effective_model_for(ex.name) == "wan"
                    and (not args.model or args.model == "wan")
                    and "wan" not in args.skip]
    if wan_examples:
        wan_results = run_wan_batch(wan_examples, args, python_exe)
        for name, elapsed, rc in wan_results:
            timings.append((name, "wan", elapsed, rc))
            if rc != 0:
                failures.append((name, rc))

    for ex in examples:
        m = effective_model_for(ex.name)
        if m == "wan":
            continue
        rc, elapsed, model = run_one(ex, args, python_exe, forced_model=args.force_model)
        if rc is None:
            continue
        timings.append((ex.name, model, elapsed, rc))
        if rc != 0:
            failures.append((ex.name, rc))
    total_elapsed = time.perf_counter() - total_start

    print()
    print("=" * 60)
    print("Timing summary")
    print("=" * 60)
    lines = []
    if timings:
        name_w = max(len(n) for n, _, _, _ in timings)
        header = f"  {'name':<{name_w}}  {'model':<5}  {'elapsed':>9}  {'min':>6}  {'frames':>6}  {'fps':>7}  status"
        print(header)
        lines.append(header)
        for name, model, elapsed, rc in timings:
            status = "OK " if rc == 0 else f"ERR{rc}"
            frames = CLIP_FRAMES.get(model, 0)
            if rc == 0 and elapsed > 0 and frames:
                fps_str = f"{frames / elapsed:7.3f}"
            else:
                fps_str = f"{'—':>7}"
            row = f"  {name:<{name_w}}  {model:<5}  {elapsed:7.1f}s  {elapsed/60:5.2f}  {frames:>6d}  {fps_str}  {status}"
            print(row)
            lines.append(row)
        ok = [(name, m, e) for name, m, e, rc in timings if rc == 0]
        if ok:
            mean_elapsed = sum(e for _, _, e in ok) / len(ok)
            total_frames = sum(CLIP_FRAMES.get(m, 0) for _, m, _ in ok)
            total_ok_elapsed = sum(e for _, _, e in ok)
            agg_fps = (total_frames / total_ok_elapsed) if total_ok_elapsed > 0 else 0.0
            summary = (
                f"  mean elapsed (successful): {mean_elapsed:.1f}s  "
                f"aggregate fps: {agg_fps:.3f}  "
                f"total wall: {total_elapsed:.1f}s ({total_elapsed/60:.2f} min)"
            )
            print(summary)
            lines.append(summary)

            print()
            lines.append("")
            per_model_header = f"  Per-model averages ({'model':<5}  {'count':>5}  {'mean_s':>8}  {'frames':>6}  {'mean_fps':>9}  {'agg_fps':>8})"
            print(per_model_header)
            lines.append(per_model_header)
            by_model = {}
            for _, m, e in ok:
                by_model.setdefault(m, []).append(e)
            for m in sorted(by_model):
                es = by_model[m]
                frames = CLIP_FRAMES.get(m, 0)
                mean_s = sum(es) / len(es)
                mean_fps = (frames / mean_s) if mean_s > 0 else 0.0
                agg_fps_m = (frames * len(es)) / sum(es) if sum(es) > 0 else 0.0
                row = f"    {m:<5}  {len(es):>5d}  {mean_s:8.2f}  {frames:>6d}  {mean_fps:9.3f}  {agg_fps_m:8.3f}"
                print(row)
                lines.append(row)

            print()
            lines.append("")
            per_cat_header = f"  Per-category averages ({'category':<8}  {'count':>5}  {'mean_s':>8}  {'mean_fps':>9}  {'agg_fps':>8})"
            print(per_cat_header)
            lines.append(per_cat_header)
            by_cat = {}
            for name, m, e in ok:
                by_cat.setdefault(category_for(name), []).append((m, e))
            for cat in sorted(by_cat):
                items = by_cat[cat]
                es = [e for _, e in items]
                total_e = sum(es)
                total_f = sum(CLIP_FRAMES.get(m, 0) for m, _ in items)
                mean_s = total_e / len(items)
                mean_fps = sum((CLIP_FRAMES.get(m, 0) / e) for m, e in items if e > 0) / len(items)
                agg_fps_c = (total_f / total_e) if total_e > 0 else 0.0
                row = f"    {cat:<8}  {len(items):>5d}  {mean_s:8.2f}  {mean_fps:9.3f}  {agg_fps_c:8.3f}"
                print(row)
                lines.append(row)

    stats_path = OUTPUTS / "stats.txt"
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("Timing summary\n")
        f.write("=" * 60 + "\n")
        for line in lines:
            f.write(line + "\n")
        if failures:
            f.write(f"\nFAILED ({len(failures)}):\n")
            for name, rc in failures:
                f.write(f"  {name} (exit {rc})\n")
    print(f"\nWrote stats: {stats_path}")

    if failures:
        print(f"\nFAILED ({len(failures)}):")
        for name, rc in failures:
            print(f"  {name} (exit {rc})")
        return 1
    print(f"\nAll {len(examples)} example(s) completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
