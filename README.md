# UrbanStreetVQA

**UrbanStreetVQA: A Benchmark for Visual Understanding of Urban Street Views**

> This repository accompanies our ACCV 2026 paper. See the [License](#license) section for
> code / annotation / source-media terms.

---

## Overview

UrbanStreetVQA is a visual-question-answering benchmark built from **pedestrian-viewpoint**
imagery of real-world urban streets. It evaluates both low-level visual perception (object
recognition, counting) and **spatial reasoning** from the perspective of a person standing on
the street — self-localization and reasoning about directions and relative distances, either
*camera-centric* (relative to the viewer) or *object-centric* (relative to a referenced
person/object in the scene).

This repository provides the **benchmark annotations**, the **training annotations** used for
fine-tuning, **source-video metadata** with a frame-download script (raw frames are not
redistributed), **evaluation code**, and the **Qwen3.5-VL component-wise fine-tuning** code.

## Main Findings

Across the six evaluated VLMs, performance is comparatively strong for
camera-centric spatial reasoning but drops sharply for object-centric direction estimation,
which requires reasoning from another person's viewpoint.

> **Key finding:** Current VLMs perform comparatively well under the camera-centric frame but degrade sharply on object-centric direction estimation.

| Model | Object Recognition | Counting | Self-localization | Camera Dir. | Camera Dist. | Object Dir. | Object Dist. | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-5.4 | 82.2 | 61.3 | 61.4 | 78.6 | 84.7 | 17.8 | 78.9 | 67.9 |
| Gemini 2.5 Flash | 85.5 | 49.3 | 72.9 | 66.9 | 84.7 | 19.0 | 62.5 | 63.1 |
| InternVL3.5 | 83.6 | 64.0 | 72.1 | 47.2 | 83.6 | 22.7 | 66.4 | 60.2 |
| LLaVA-OneVision-1.5 | 84.9 | 38.2 | 59.3 | 35.9 | 75.1 | 17.0 | 55.9 | 49.5 |
| Gemma 4 | 83.6 | 56.0 | 75.7 | 64.9 | 85.0 | 22.3 | 75.0 | 65.2 |
| **Qwen3.5** | **86.2** | **70.2** | 70.7 | 71.2 | 82.7 | **32.0** | 75.7 | **69.5** |

The largest performance drop appears in **object-centric direction estimation**:
even the best-performing model, Qwen3.5, reaches only **32.0%**, while its
camera-centric direction accuracy is **71.2%**. This suggests that the main difficulty
lies not only in recognizing the scene, but in transforming spatial relationships into
another person's reference frame.

## Dataset

Verified directly from `data/benchmark/urbanstreetvqa.jsonl`:

- **1,731** benchmark QA pairs
- **854** benchmark images
- **7** cities
- **7** task categories

**Cities:** Tokyo, Shanghai, Dubai, Mumbai, London, New York, Mexico City.

The **7 categories** are the 5 `category` values, with `directional_localization` and
`relative_distance` each split into camera-centric and object-centric (`viewpoint`) variants.
See [`DATASET.md`](DATASET.md) for the full schema, statistics, and mappings.

A separate **training set** (for fine-tuning) is provided in
`data/train/urbanstreetvqa_train.jsonl` — **3,622 QA / 1,780 images**, disjoint from the
benchmark.

## Data Release

Raw frames are **not** redistributed. Instead, the repository provides:

- source video metadata — `data/sources/videos.csv` (internal `video_key` → YouTube id/url),
- per-frame metadata — `data/sources/frames.csv` (`image` → `video_key`, `timestamp_sec`, split),
- a downloader — `scripts/download_frames.py`,

so that a third party can reconstruct the exact frames referenced by the annotations from the
original videos (35 source YouTube walking videos across the 7 cities). The rights to the source
videos belong to their **original YouTube uploaders**; you are responsible for complying with
YouTube's Terms of Service and those rights.

`videos.csv` provides the `youtube_id` / `youtube_url` for all 35 source videos, and
`frames.csv` provides `timestamp_sec` for all benchmark (854) and training (1,780) frames, so
`scripts/download_frames.py` can reconstruct the images end-to-end.

**Source media policy:**

- Raw videos are **not** redistributed.
- Raw frames are **not** redistributed.
- `data/sources/videos.csv` contains **source references only** (`video_key`, city, YouTube id/url).
- `data/sources/frames.csv` contains **timestamp metadata only** (`image`, `video_key`, `timestamp_sec`, split).
- `scripts/download_frames.py` helps users **retrieve the source media themselves**.
- Use of the YouTube / source media remains subject to the **original uploader / rights-holder
  terms and the platform's terms of service**.
- UrbanStreetVQA does **not** claim ownership of the source videos or the extracted frames.

## Benchmark Annotations

- `data/benchmark/urbanstreetvqa.jsonl` — 1,731 evaluation QA over 854 images.
- `data/train/urbanstreetvqa_train.jsonl` — 3,622 fine-tuning QA over 1,780 images.

Both share the same annotation schema (documented in [`DATASET.md`](DATASET.md)).

## Download Frames

```bash
pip install yt-dlp          # ffmpeg must also be installed as a system dependency

python scripts/download_frames.py --split benchmark --output-dir data/frames/benchmark
python scripts/download_frames.py --split train     --output-dir data/frames/train
```

By default each source video is downloaded once, the listed frames are extracted with ffmpeg,
and the video is deleted afterward (use `--keep-videos` to keep it). See `--help` for
`--metadata`, `--frames`, `--video-cache`, and `--skip-existing`.

## Evaluation

The paper evaluates six models:

| Model | Script |
|---|---|
| Qwen3.5-VL | `evaluation/qwen35.py` |
| GPT-5.4 | `evaluation/gpt.py` |
| Gemini 2.5 Flash | `evaluation/gemini.py` |
| Gemma 4 | `evaluation/gemma.py` |
| InternVL3.5 | `evaluation/internvl.py` |
| LLaVA-OneVision-1.5 | `evaluation/llava.py` |

Evaluation scripts share a common CLI:

```bash
python evaluation/qwen35.py \
    --annotations data/benchmark/urbanstreetvqa.jsonl \
    --image-dir data/frames/benchmark \
    --model-path PATH/TO/Qwen3.5-VL \
    --output-dir results_qwen35
```

Local models (`qwen35.py`, `gemma.py`, `internvl.py`, `llava.py`) take `--model-path`. API
models read their key from an environment variable (`OPENAI_API_KEY` for GPT, `GEMINI_API_KEY`
for Gemini) — **never** hard-code keys.

```bash
export GEMINI_API_KEY=...
python evaluation/gemini.py \
    --annotations data/benchmark/urbanstreetvqa.jsonl \
    --image-dir data/frames/benchmark \
    --output-dir results_gemini
```

### Circular Evaluation

- **Multiple-choice:** each question is asked under all cyclic permutations of the option order;
  it counts as correct only if the model is correct on **every** rotation.
- **Counting:** exact numerical match, with **no** rotation.

This protocol is implemented consistently across the evaluation scripts.

## Fine-tuning

Component-wise fine-tuning of plain Qwen3.5-VL (no fusion) in three cumulative conditions:

- **(a) Projector only** — train the projector (merger); LLM and ViT frozen.
- **(b) Projector + LLM** — projector + LLM LoRA (r=16, α=32).
- **(c) Projector + LLM + ViT** — (b) plus unfreezing the ViT at 1/10 the learning rate.

```bash
python finetuning/train_qwen35.py --config finetuning/config.yaml --tune_mode projector_only
python finetuning/train_qwen35.py --config finetuning/config.yaml --tune_mode projector_lora
python finetuning/train_qwen35.py --config finetuning/config.yaml --tune_mode projector_lora_vit
```

Settings (`finetuning/config.yaml`): LoRA rank 16 / α 32 / dropout 0.05; AdamW
(β1=0.9, β2=0.95); 3 epochs; cosine schedule; 5% warmup; projector/LLM lr 1e-4; ViT lr 1e-5;
effective batch size 8 (batch 1 × grad-accum 8).

### Fine-tuning Results

Component-wise fine-tuning of Qwen3.5 reveals a different behavior between
camera-centric and object-centric direction reasoning.

| Trainable modules | Camera Dir. | Object Dir. | Overall |
|---|---:|---:|---:|
| Baseline | 71.2 | 32.0 | 69.5 |
| Projector only | 91.6 | 25.1 | 71.9 |
| Projector + LLM | **96.1** | 30.8 | **76.3** |
| Projector + LLM + ViT | 93.5 | **40.1** | **76.3** |

Camera-centric direction improves substantially by adapting the projector and LLM,
whereas object-centric direction improves only when the **ViT is also updated**.
This suggests that adapting the visual encoder is particularly important for
object-centric direction reasoning.

## Dependencies

See [`requirements.txt`](requirements.txt) and [`docs/dependencies.md`](docs/dependencies.md).
`ffmpeg` is a **system dependency** (required by `scripts/download_frames.py`); install it via
your OS package manager. `yt-dlp` is installed via pip.

## License

UrbanStreetVQA uses distinct licenses for code, annotations, and source media:

- **Repository code** (`evaluation/`, `finetuning/`, `scripts/`): **Apache-2.0** — see
  [`LICENSE`](LICENSE).
- **QA annotations** (`data/benchmark/`, `data/train/`, `data/sources/`): **CC BY 4.0**
  (<https://creativecommons.org/licenses/by/4.0/>) — see [`data/LICENSE.md`](data/LICENSE.md).
  Attribution is required. This applies only to the author-created annotations.
- **Source media** (the YouTube videos and extracted frames): **not redistributed**. The
  original rights remain with the original uploaders / rights holders; this project claims no
  ownership of them (see [Data Release](#data-release) and [`DATASET.md`](DATASET.md)).

## Citation

See [`CITATION.cff`](CITATION.cff). The paper has been accepted to ACCV 2026. The project page
is available at https://lemonsour3.github.io/UrbanStreetVQA/.
