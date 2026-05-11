try:
    import argparse
    import datetime as _dt
    import json
    import os
    import time
    from pathlib import Path
    import torch
    from diffusers import BitsAndBytesConfig, WanTransformer3DModel
    from diffusers.utils import export_to_video, load_image
    from transformers import BitsAndBytesConfig as TransformersBnbConfig, UMT5EncoderModel
    from pipelines.wan_pipeline import WanImageToVideoTTMPipeline
    from pipelines.utils import (
        validate_inputs,
        compute_hw_from_area,
    )
except ImportError as e:
    raise ImportError(f"Required module not found: {e}. Please install it before running this script. "
                     f"For installation instructions, see: https://github.com/Wan-Video/Wan2.2")

MODEL_ID = "lopho/Wan2.2-I2V-A14B-Diffusers_nf4"
DTYPE = torch.bfloat16


def _gpu_mem(prefix=""):
    if not torch.cuda.is_available():
        return f"{prefix}cuda not available"
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    used = total - free
    alloc = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    peak = torch.cuda.max_memory_allocated()
    return (
        f"{prefix}used {used/1e9:5.2f}/{total/1e9:5.2f} GB | "
        f"alloc {alloc/1e9:5.2f} GB | reserved {reserved/1e9:5.2f} GB | peak {peak/1e9:5.2f} GB"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run Wan Image to Video Pipeline")
    parser.add_argument("--manifest", type=str, default=None, help="JSON list of {name, input, output, tweak, tstrong}; loads model once and runs all entries")
    parser.add_argument("--stats-path", type=str, default=None, help="Append per-example 'name\\telapsed\\trc' lines to this file")
    parser.add_argument("--input-path", type=str, default="./examples/wan_monkey", help="Path to input image (single mode)")
    parser.add_argument("--output-path", type=str, default="./outputs/output_wan_monkey.mp4", help="Path to save output video (single mode)")
    parser.add_argument("--negative-prompt", type=str, default=(
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，"
        "低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
        "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    ), help="Default negative prompt in Wan2.2")
    parser.add_argument("--tweak-index", type=int, default=3, help="t weak timestep index- when to start denoising")
    parser.add_argument("--tstrong-index", type=int, default=6, help="t strong timestep index- when to start denoising within the mask")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--max-area", type=int, default=480 * 832, help="Maximum area for resizing")
    parser.add_argument("--num-frames", type=int, default=81, help="Number of frames to generate")
    parser.add_argument("--guidance-scale", type=float, default=3.5, help="Guidance scale for generation")
    return parser.parse_args()


def setup_wan_pipeline(model_id: str, dtype: torch.dtype, device: str):
    t0 = time.perf_counter()
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
    )

    print(f"[setup] mem at start | {_gpu_mem()}", flush=True)
    print(f"[setup] quant config: load_in_4bit=True nf4 compute_dtype={dtype}", flush=True)

    t_t1 = time.perf_counter()
    print(f"[setup] loading transformer (nf4) from {model_id}/transformer ...", flush=True)
    transformer = WanTransformer3DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        quantization_config=quant_config,
        torch_dtype=dtype,
        local_files_only=True,
    )
    print(f"[setup] transformer loaded in {time.perf_counter()-t_t1:.1f}s | {_gpu_mem()}", flush=True)

    t_t2 = time.perf_counter()
    print(f"[setup] loading transformer_2 (nf4) from {model_id}/transformer_2 ...", flush=True)
    transformer_2 = WanTransformer3DModel.from_pretrained(
        model_id,
        subfolder="transformer_2",
        quantization_config=quant_config,
        torch_dtype=dtype,
        local_files_only=True,
    )
    print(f"[setup] transformer_2 loaded in {time.perf_counter()-t_t2:.1f}s | {_gpu_mem()}", flush=True)

    t_te = time.perf_counter()
    print(f"[setup] loading text_encoder (nf4 UMT5) from {model_id}/text_encoder ...", flush=True)
    te_quant_config = TransformersBnbConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
    )
    text_encoder = UMT5EncoderModel.from_pretrained(
        model_id,
        subfolder="text_encoder",
        quantization_config=te_quant_config,
        torch_dtype=dtype,
        local_files_only=True,
    )
    print(f"[setup] text_encoder loaded in {time.perf_counter()-t_te:.1f}s | {_gpu_mem()}", flush=True)

    t_p = time.perf_counter()
    print(f"[setup] from_pretrained({model_id}) — pipeline assembly with prequantized components ...", flush=True)
    pipe = WanImageToVideoTTMPipeline.from_pretrained(
        model_id,
        transformer=transformer,
        transformer_2=transformer_2,
        text_encoder=text_encoder,
        torch_dtype=dtype,
        local_files_only=True,
    )
    print(f"[setup] from_pretrained done in {time.perf_counter()-t_p:.1f}s | {_gpu_mem()}", flush=True)

    t1 = time.perf_counter()
    print("[setup] vae.enable_tiling + enable_slicing ...", flush=True)
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    print(f"[setup] vae setup done in {time.perf_counter()-t1:.2f}s", flush=True)

    t2 = time.perf_counter()
    print(f"[setup] pipe.to({device!r}) — moving non-quantized components to GPU ...", flush=True)
    pipe.to(device)
    print(f"[setup] pipe.to done in {time.perf_counter()-t2:.1f}s | {_gpu_mem()}", flush=True)

    print(f"[setup] component placement:", flush=True)
    for name in ("transformer", "transformer_2", "text_encoder", "vae", "image_encoder"):
        comp = getattr(pipe, name, None)
        if comp is not None:
            try:
                dev = next(comp.parameters()).device
                dt = next(comp.parameters()).dtype
                n_params = sum(p.numel() for p in comp.parameters())
                print(f"[setup]   {name}: device={dev} dtype={dt} params={n_params/1e9:.2f}B", flush=True)
            except StopIteration:
                print(f"[setup]   {name}: <no params>", flush=True)
    print(f"[setup] total setup time: {time.perf_counter()-t0:.1f}s", flush=True)
    return pipe


def run_one(pipe, args, input_path: str, output_path: str, tweak_index: int, tstrong_index: int, name: str = None):
    tag = f"[{name}]" if name else f"[{Path(input_path).name}]"
    print(f"{tag} input_path={input_path}", flush=True)
    image_path = os.path.join(input_path, "first_frame.png")
    motion_signal_mask_path = os.path.join(input_path, "mask.mp4")
    motion_signal_video_path = os.path.join(input_path, "motion_signal.mp4")
    prompt_path = os.path.join(input_path, "prompt.txt")
    Path(os.path.dirname(output_path) or ".").mkdir(parents=True, exist_ok=True)
    print(f"{tag} validate_inputs ...", flush=True)
    validate_inputs(image_path, motion_signal_mask_path, motion_signal_video_path)

    t_img = time.perf_counter()
    image = load_image(image_path)
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    height, width = compute_hw_from_area(image.height, image.width, args.max_area, mod_value)
    image = image.resize((width, height))
    print(f"{tag} image loaded ({image.width}x{image.height}) in {time.perf_counter()-t_img:.2f}s", flush=True)

    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt = f.read().strip()
    print(f"{tag} prompt loaded ({len(prompt)} chars): {prompt[:80]}{'...' if len(prompt) > 80 else ''}", flush=True)

    gen_device = args.device if args.device.startswith("cuda") else "cpu"
    generator = torch.Generator(device=gen_device).manual_seed(args.seed)

    total_steps = args.num_inference_steps
    print(f"{tag} starting pipe(...) | num_inference_steps={total_steps} num_frames={args.num_frames} tweak={tweak_index} tstrong={tstrong_index}", flush=True)

    pipe.set_progress_bar_config(disable=True)

    pipe._run_wan_tag = tag
    _patched_attrs = ("encode_prompt", "encode_image", "set_timesteps", "prepare_latents", "prepare_motion_signal")
    _SENTINEL = "_run_wan_wrapped"
    for _name in _patched_attrs:
        target = pipe.scheduler if _name == "set_timesteps" else pipe
        if not hasattr(target, _name):
            continue
        existing = getattr(target, _name)
        if getattr(existing, _SENTINEL, False):
            continue
        def _make_wrapped(orig, name):
            def _wrapped(*a, **kw):
                t0 = time.perf_counter()
                cur_tag = getattr(pipe, "_run_wan_tag", tag)
                print(f"{cur_tag}   pre-step: {name} start | {_gpu_mem()}", flush=True)
                try:
                    return orig(*a, **kw)
                finally:
                    print(f"{cur_tag}   pre-step: {name} done in {time.perf_counter()-t0:.1f}s | {_gpu_mem()}", flush=True)
            setattr(_wrapped, _SENTINEL, True)
            return _wrapped
        setattr(target, _name, _make_wrapped(existing, _name))

    step_state = {
        "count": 0,
        "t_first": time.perf_counter(),
        "t_step": time.perf_counter(),
        "step_times": [],
    }
    def _step_cb(pipe_self, step_idx, timestep, callback_kwargs):
        now = time.perf_counter()
        ts = int(timestep) if hasattr(timestep, "__int__") else timestep
        step_dt = now - step_state["t_step"]
        step_state["t_step"] = now
        step_state["count"] += 1
        step_state["step_times"].append(step_dt)
        elapsed = now - step_state["t_first"]
        avg = sum(step_state["step_times"][-5:]) / min(len(step_state["step_times"]), 5)
        remaining = max(0, total_steps - step_state["count"])
        eta_s = remaining * avg
        eta_clock = (_dt.datetime.now() + _dt.timedelta(seconds=eta_s)).strftime("%H:%M:%S")
        print(
            f"{tag}   step {step_state['count']}/{total_steps} | "
            f"{step_dt:6.2f}s | avg(5) {avg:5.2f}s | elapsed {elapsed/60:5.2f}m | "
            f"ETA {eta_s/60:5.2f}m (~{eta_clock}) | timestep={ts} | {_gpu_mem()}",
            flush=True,
        )
        return callback_kwargs

    inf_start = time.perf_counter()
    with torch.inference_mode():
        result = pipe(
            image=image,
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            height=height,
            width=width,
            num_frames=args.num_frames,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
            motion_signal_video_path=motion_signal_video_path,
            motion_signal_mask_path=motion_signal_mask_path,
            tweak_index=tweak_index,
            tstrong_index=tstrong_index,
            callback_on_step_end=_step_cb,
        )
    inf_elapsed = time.perf_counter() - inf_start
    print(f"{tag} pipe(...) returned in {inf_elapsed:.1f}s", flush=True)

    t_exp = time.perf_counter()
    frames = result.frames[0]
    export_to_video(frames, output_path, fps=16)
    print(f"{tag} export_to_video done in {time.perf_counter()-t_exp:.1f}s -> {output_path}", flush=True)
    return inf_elapsed


def main():
    args = parse_args()
    pipe = setup_wan_pipeline(MODEL_ID, DTYPE, args.device)

    if args.manifest:
        with open(args.manifest, "r", encoding="utf-8") as f:
            entries = json.load(f)
        print(f"[main] manifest has {len(entries)} entries", flush=True)
        for i, entry in enumerate(entries, 1):
            name = entry.get("name", entry["input"])
            tweak = entry.get("tweak", args.tweak_index)
            tstrong = entry.get("tstrong", args.tstrong_index)
            print(f"\n=== [{i}/{len(entries)}] {name} ===", flush=True)
            elapsed = 0.0
            try:
                elapsed = run_one(pipe, args, entry["input"], entry["output"], tweak, tstrong, name=name)
                rc = 0
            except Exception as e:
                print(f"[{name}] FAILED: {type(e).__name__}: {e}", flush=True)
                rc = 1
            print(f"=== [{i}/{len(entries)}] {name}: inference {elapsed:.1f}s rc={rc} ===", flush=True)
            if args.stats_path:
                with open(args.stats_path, "a", encoding="utf-8") as f:
                    f.write(f"{name}\t{elapsed:.3f}\t{rc}\n")
    else:
        run_one(pipe, args, args.input_path, args.output_path, args.tweak_index, args.tstrong_index, name=Path(args.input_path).name)


if __name__ == "__main__":
    main()
