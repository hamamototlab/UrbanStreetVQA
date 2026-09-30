# UrbanStreetVQA — Dataset Documentation

## Benchmark dataset statistics

Computed directly from `data/benchmark/urbanstreetvqa.jsonl`:

| Quantity | Value |
|---|---|
| QA pairs | **1,731** |
| Unique images | **854** |
| Cities | **7** |
| Source videos | **35** |
| Task categories | **7** (5 `category` values × camera/viewpoint split) |

QA per `category`: directional_localization 709, relative_distance 505, counting 225,
object_detection 152, self_localization 140.

## Training dataset statistics

Computed directly from `data/train/urbanstreetvqa_train.jsonl`:

| Quantity | Value |
|---|---|
| QA pairs | **3,622** |
| Unique images | **1,780** |
| Cities | 7 |
| Source videos | 35 (same set as the benchmark) |

The training set is used for the fine-tuning experiments (paper Section 5, Table 4) and is
disjoint from the benchmark evaluation set. This is the **v4** training set that actually
produced the Table 4 results; the paper text's "3,510 QA / 1,750 images" is a known erratum to
be corrected in the camera-ready.

## Annotation schema

Both jsonl files share the same schema (one JSON object per line):

| Field | Type | Description |
|---|---|---|
| `image` | string | Frame filename, e.g. `akihabara_movie3_12.jpg` (filename only, no path). |
| `video` | string | Source `video_key`, e.g. `akihabara_movie3`. |
| `city` | string | City key (see mapping below). |
| `category` | string | `directional_localization`, `relative_distance`, `counting`, `object_detection`, `self_localization`. |
| `subcategory` | string | Task refinement / centricity — see mapping below. |
| `question` | string | The question text. |
| `answer` | string | Ground-truth answer (option text for MC; an integer string or `"11 or more"` for counting). |
| `options` | list[str] | Multiple-choice options. **Empty for counting** (free-form integer). |
| `ref_object` / `ref_bbox` | string / list / null | Referenced object and bbox (object-centric items). |
| `obj_A` / `obj_A_bbox` / `obj_B` / `obj_B_bbox` | string / list / null | Two referenced objects and boxes for relational questions. |

The internal management fields `_id` and `_review` used during dataset construction have been
**removed** from the public annotations.

## City mapping

| Annotation key | Display name |
|---|---|
| `akihabara` | Tokyo |
| `Shanghai` | Shanghai |
| `dubai` | Dubai |
| `Mumbai` | Mumbai |
| `london` | London |
| `newyork` | New York |
| `MexicoCity` | Mexico City |

## Category / subcategory mapping (7 tasks)

**Camera-centric:**
- Object Recognition — `category=object_detection` (`subcategory` ∈ {existence, facility, sign})
- Counting — `category=counting` (`subcategory=none`)
- Self-localization — `category=self_localization` (`subcategory=scene`)
- Direction (camera) — `category=directional_localization`, `subcategory=camera`
- Relative Distance (camera) — `category=relative_distance`, `subcategory=camera`

**Object-centric** (`subcategory=viewpoint`):
- Direction (viewpoint) — `category=directional_localization`, `subcategory=viewpoint`
- Relative Distance (viewpoint) — `category=relative_distance`, `subcategory=viewpoint`

## File naming conventions

- Frame image: `{city}_movie{N}_{index}.jpg` (e.g. `Shanghai_movie3_57.jpg`). The trailing
  number is a frame/sample index — **not** a timestamp in seconds.
- Source video id (`video` / `video_key`): `{city}_movie{N}`.

## Source metadata schema

### `data/sources/videos.csv`

| Column | Description |
|---|---|
| `video_key` | Internal source id, e.g. `akihabara_movie3` (35 rows). |
| `city` | Display city name. |
| `youtube_id` | YouTube video id. **Populated for all 35 videos.** |
| `youtube_url` | Full YouTube url. **Populated for all 35 videos.** |
| `duration` | Optional. |
| `notes` | Optional. |

### `data/sources/frames.csv`

| Column | Description |
|---|---|
| `image` | Frame filename referenced by the annotations. |
| `video_key` | Source video the frame comes from. |
| `timestamp_sec` | Time offset (seconds) of the frame within the video. **Populated for all rows** from the frame-extraction metadata. |
| `split` | `benchmark` or `train`. |
| `city` | Display city name. |

`frames.csv` has 2,634 rows: 854 benchmark + 1,780 train unique images. Every row has a
`timestamp_sec` (100% coverage; source: per-video `timestamps.json` frame-extraction metadata).

## Frame download procedure

Raw frames are not redistributed. To reconstruct them:

```bash
pip install yt-dlp     # ffmpeg is a system dependency (install separately)
python scripts/download_frames.py --split benchmark --output-dir data/frames/benchmark
python scripts/download_frames.py --split train     --output-dir data/frames/train
```

The script downloads each source video (yt-dlp) and extracts the exact frames listed in
`frames.csv` at their `timestamp_sec` (ffmpeg). It **never guesses**: video rows without a
YouTube id/url and frames without a `timestamp_sec` are skipped with a warning.

## Annotation license

The UrbanStreetVQA **QA annotations** — `data/benchmark/urbanstreetvqa.jsonl`,
`data/train/urbanstreetvqa_train.jsonl`, and the source-reference/timestamp metadata in
`data/sources/` — are released under the **Creative Commons Attribution 4.0 International License
(CC BY 4.0)**: <https://creativecommons.org/licenses/by/4.0/>
(legal code: <https://creativecommons.org/licenses/by/4.0/legalcode>). See
[`data/LICENSE.md`](data/LICENSE.md).

- UrbanStreetVQA QA annotations are released under CC BY 4.0.
- **Attribution is required.**
- This license applies **only to the annotations created by the authors**.
- It does **NOT** grant rights to the underlying YouTube videos or the extracted frames.

The repository **code** is licensed separately under **Apache-2.0** (see the root `LICENSE`).

## Source media policy

- Raw videos are **not** redistributed.
- Raw frames are **not** redistributed.
- `data/sources/videos.csv` contains **source references only** (`video_key`, city, YouTube id/url).
- `data/sources/frames.csv` contains **timestamp metadata only** (`image`, `video_key`, `timestamp_sec`, split).
- `scripts/download_frames.py` helps users **retrieve the source media themselves**.
- Use of the YouTube / source media remains subject to the **original uploader / rights-holder
  terms and the platform's terms of service**.
- UrbanStreetVQA does **not** claim ownership of the source videos or the extracted frames.

The 35 source walking videos are third-party YouTube content; all rights belong to their
original uploaders. Downloading and extracting frames with `scripts/download_frames.py` is at the
user's discretion and subject to those rights and YouTube's Terms of Service.

## Metadata status

- **YouTube ids/urls — complete.** `videos.csv` provides `youtube_id` and `youtube_url` for all
  35 source videos.
- **Frame timestamps — complete.** `timestamp_sec` is populated for all 2,634 rows (854
  benchmark + 1,780 train), taken from the per-video `timestamps.json` extraction metadata
  (100% coverage). Not guessed from filenames.

Together these let `scripts/download_frames.py` reconstruct every referenced frame end-to-end.
