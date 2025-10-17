#!/usr/bin/env python
"""
convert_and_eval.py

Convert a VerL checkpoint to HuggingFace format and (optionally) run the
in‑house evaluation framework, saving results to JSON.

Example
-------
python convert_and_eval.py \
  --base_model Qwen/Qwen2.5-3B \
  --proj_name wikiRL \
  --checkpoint_name MAP-7B-10K-_data_zhiheng_cogito_base_models_qwen2.5-3b-1epoch-hard-grpo-n4-b16-t0.5 \
  --output_name 3B-10K-40Step \
  --global_step 40 \
  --cuda_id 0
"""
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

# Files that must be copied after merging
TOKENIZER_CONFIG_FILES = [
    "added_tokens.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
]


def parse_args() -> argparse.Namespace:
    """CLI parser."""
    parser = argparse.ArgumentParser(description="Convert a VerL checkpoint and run evaluation.")
    parser.add_argument("--base_model", required=True, help="HF repo id, e.g. Qwen/Qwen2.5-3B")
    parser.add_argument("--proj_name", required=True, help="Project name, e.g. wikiRL")
    parser.add_argument("--checkpoint_name", required=True, help="Folder name under checkpoints/<proj>")
    parser.add_argument("--output_name", required=True, help="Name for converted checkpoint")
    parser.add_argument("--global_step", type=int, default=40, help="Global step folder to use")
    parser.add_argument("--cuda_id", default="0", help="CUDA device used for later evaluation")
    parser.add_argument(
        "--checkpoints_root",
        default="/data/zhiheng/cogito/verl-tool/checkpoints",
        help="Root where training checkpoints are stored",
    )
    parser.add_argument(
        "--converted_root",
        default="/data/zhiheng/cogito/verl-tool/checkpoint_converted",
        help="Root folder to place converted checkpoints",
    )
    parser.add_argument(
        "--base_model_cache",
        default="./base_model",
        help="Local cache directory for downloading the HF base model",
    )
    return parser.parse_args()


def _download_base_model(repo_id: str, cache_dir: Path) -> Path:
    """
    Download a HuggingFace model (if not cached) and return the local path.
    Uses `huggingface-hub` snapshot_download to avoid re‑downloading.
    """
    from huggingface_hub import snapshot_download

    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = cache_dir / repo_id.replace("/", "__")
    if local_path.exists() and any(local_path.iterdir()):
        return local_path

    print(f"[INFO] Downloading base model '{repo_id}' to '{local_path}' …")
    snapshot_download(repo_id, local_dir=local_path, local_dir_use_symlinks=False)
    return local_path


def _copy_tokenizer_files(src_dir: Path, dst_dir: Path) -> None:
    """Copy tokenizer/config files if they exist in `src_dir`."""
    for fname in TOKENIZER_CONFIG_FILES:
        src_f = src_dir / fname
        if src_f.exists():
            shutil.copy2(src_f, dst_dir / fname)


def model_convert(
    base_model_repo: str,
    checkpoints_root: Path,
    converted_root: Path,
    proj_name: str,
    checkpoint_name: str,
    output_name: str,
    global_step: int,
    base_model_cache: Path,
) -> Path:
    """
    Convert VerL checkpoint to HF format if necessary and return the target path.
    """
    ckpt_source = (
        checkpoints_root
        / proj_name
        / checkpoint_name
        / f"global_step_{global_step}"
        / "actor"
    )
    target_dir = converted_root / proj_name / output_name

    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"[INFO] Target directory '{target_dir}' already exists—skipping conversion.")
        return target_dir

    if not ckpt_source.exists():
        raise FileNotFoundError(f"Source checkpoint directory not found: {ckpt_source}")

    # Ensure parent folders exist
    target_dir.mkdir(parents=True, exist_ok=True)

    # Download base model if necessary
    base_model_local = _download_base_model(base_model_repo, base_model_cache)

    # === Run merger =========================================================
    merger_cmd = [
        "python",
        "/data/zhiheng/cogito/verl-tool/verl/scripts/model_merger.py",
        "--backend",
        "fsdp",
        "--hf_model_path",
        str(base_model_repo),
        "--local_dir",
        str(ckpt_source),
        "--target_dir",
        str(target_dir),
    ]

    print(f"[INFO] Running model merger:\n{' '.join(merger_cmd)}")
    subprocess.run(merger_cmd, check=True)

    # === Copy tokenizer / config files ======================================
    print("[INFO] Copying tokenizer/config files …")
    _copy_tokenizer_files(base_model_local, target_dir)

    print(f"[SUCCESS] Converted checkpoint saved to '{target_dir}'.")
    return target_dir


def eval_result(target_dir: Path, result_json: Path, cuda_id: str = "0") -> None:
    """
    Placeholder for your internal evaluation logic.

    Parameters
    ----------
    target_dir : Path
        Path to the converted HF checkpoint.
    result_json : Path
        File to write evaluation results into.
    cuda_id : str
        CUDA device ID for evaluation.
    """
    # TODO: integrate your evaluation framework here.
    # Write the eval metrics as a dict into `result_json`, e.g.:
    #
    # metrics = {"accuracy": 0.0, "bleu": 0.0}
    # result_json.write_text(json.dumps(metrics, indent=2))
    #
    # For now, we just create an empty file so the calling pipeline does not fail.
    pass


def main() -> None:
    args = parse_args()

    checkpoints_root = Path(args.checkpoints_root).expanduser()
    converted_root = Path(args.converted_root).expanduser()
    base_model_cache = Path(args.base_model_cache).expanduser()

    target_dir = model_convert(
        base_model_repo=args.base_model,
        checkpoints_root=checkpoints_root,
        converted_root=converted_root,
        proj_name=args.proj_name,
        checkpoint_name=args.checkpoint_name,
        output_name=args.output_name,
        global_step=args.global_step,
        base_model_cache=base_model_cache,
    )

    result_json = converted_root / args.proj_name / f"{args.output_name}.json"
    eval_result(target_dir=target_dir, result_json=result_json, cuda_id=args.cuda_id)


if __name__ == "__main__":
    main()
