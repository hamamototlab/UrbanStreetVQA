# Dependencies & Environment Notes

## Main environment (fine-tuning + evaluation)

Covers `finetuning/` and `evaluation/qwen35.py`.

- `transformers>=5.13` — provides `Qwen3_5ForConditionalGeneration`.
- `peft`, `accelerate`, `safetensors` — LoRA + Hugging Face `Trainer` + weight I/O.
- `qwen-vl-utils` — Qwen-VL preprocessing helper.
- `torch` / `torchvision` matching your CUDA build.
- `numpy`, `pandas`, `pillow`, `pyyaml`.

## API evaluation baselines

- **Gemini 2.5 Flash** (`evaluation/gemini.py`): `google-genai`. The API key is read from the
  `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) environment variable — never hard-coded. Free-tier
  quota can block a full 1,731-QA run; a paid key is needed for the complete benchmark.
- **Gemma 4** (`evaluation/gemma.py`): uses `transformers` (`AutoModelForMultimodalLM`). The
  visual token budget is architecturally capped at 280 soft tokens/image.
- **GPT-5.4** (`evaluation/gpt.py`): uses `openai`; the key is read from `OPENAI_API_KEY`.
- **InternVL3.5** (`evaluation/internvl.py`): `transformers` (`AutoModelForImageTextToText`,
  `trust_remote_code`); saves an Excel report via `xlsxwriter`.
- **LLaVA-OneVision-1.5** (`evaluation/llava.py`): `transformers` (`AutoModelForCausalLM`) with a
  checkpoint key-remap loader, `qwen-vl-utils` (`process_vision_info`), `safetensors`; Excel via
  `xlsxwriter`.

## Frame download

`scripts/download_frames.py` requires:

- `yt-dlp` (pip) — downloads the source YouTube videos.
- `ffmpeg` (**system dependency**, install via your OS package manager) — extracts frames.

## Circular Evaluation

- **Multiple-choice:** each question is asked under all cyclic permutations of the option order
  and counts as correct only if the model is correct on every rotation.
- **Counting:** exact numerical match, no rotation.

Implemented consistently across the evaluation scripts.

## Removed dependencies

The following were only needed by research code that is **not** part of this public repository
(dataset construction and the geometry/human fusion model), and are therefore intentionally
excluded: VGGT, 4D-Humans (+ SMPL), Grounding DINO, Depth Anything V2, and the rebuttal
baselines (SpatialThinker, SpatialReasoner, DriveLMM-o1 / InternVL2.5, PerceptionLM).
