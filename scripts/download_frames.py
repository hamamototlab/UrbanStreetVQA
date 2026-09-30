#!/usr/bin/env python3
"""
download_frames.py — reconstruct UrbanStreetVQA frames from their source videos.

UrbanStreetVQA does NOT redistribute raw frames or videos. Instead it ships:
  - data/sources/videos.csv : internal video_key -> YouTube id / url
  - data/sources/frames.csv : image -> video_key, timestamp_sec, split

This script downloads each source video (yt-dlp) and extracts the exact frames
listed in frames.csv (ffmpeg) into an output directory, so a third party can
reconstruct the images referenced by the annotations on their own machine.

Requirements (installed by the user, not vendored here):
  - yt-dlp   (pip install yt-dlp)
  - ffmpeg   (system dependency; must be on PATH)

The rights to the source videos belong to their original uploaders. This script
only helps you fetch frames for research use; you are responsible for complying
with YouTube's Terms of Service and the uploaders' rights.

IMPORTANT — this script never guesses:
  - If a video_key has no youtube_id/url in videos.csv, its frames are SKIPPED.
  - If a frame has no timestamp_sec in frames.csv, that frame is SKIPPED.
  Those fields are currently unresolved in this repository (see DATASET.md);
  fill them in before the script can extract frames.

Usage:
  python scripts/download_frames.py --split benchmark --output-dir data/frames/benchmark
  python scripts/download_frames.py --split train     --output-dir data/frames/train

Optional:
  --metadata data/sources/videos.csv
  --frames   data/sources/frames.csv
  --video-cache .video_cache      # where downloaded videos are stored
  --skip-existing                 # skip frames already extracted
  --keep-videos                   # do not delete a video after extraction
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path


def _check_tool(name: str) -> None:
    if shutil.which(name) is None:
        sys.exit(f"[error] required tool '{name}' not found on PATH. "
                 f"Install it and try again (yt-dlp via pip, ffmpeg as a system package).")


def load_videos(path: Path) -> dict[str, dict]:
    videos: dict[str, dict] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            videos[row["video_key"]] = row
    return videos


def load_frames(path: Path, split: str) -> list[dict]:
    with open(path, newline="") as f:
        return [r for r in csv.DictReader(f) if r.get("split") == split]


def resolve_url(video_row: dict) -> str | None:
    url = (video_row.get("youtube_url") or "").strip()
    if url:
        return url
    vid = (video_row.get("youtube_id") or "").strip()
    if vid:
        return f"https://www.youtube.com/watch?v={vid}"
    return None


def download_video(url: str, cache_dir: Path, video_key: str) -> Path | None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_tmpl = str(cache_dir / f"{video_key}.%(ext)s")
    # Grab a single mp4-ish progressive/best stream; let yt-dlp pick the container.
    cmd = ["yt-dlp", "-f", "bestvideo[ext=mp4]+bestaudio/best",
           "--merge-output-format", "mp4", "-o", out_tmpl, url]
    print(f"  [yt-dlp] {url}")
    if subprocess.run(cmd).returncode != 0:
        print(f"  [warn] download failed for {video_key}")
        return None
    for p in sorted(cache_dir.glob(f"{video_key}.*")):
        if p.suffix.lower() in (".mp4", ".mkv", ".webm"):
            return p
    return None


def extract_frame(video_path: Path, timestamp_sec: float, out_path: Path) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-ss", str(timestamp_sec), "-i", str(video_path),
           "-frames:v", "1", "-q:v", "2", str(out_path)]
    return subprocess.run(cmd, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode == 0 and out_path.exists()


def main() -> None:
    ap = argparse.ArgumentParser(description="Reconstruct UrbanStreetVQA frames from source videos.")
    ap.add_argument("--split", required=True, choices=["benchmark", "train"])
    ap.add_argument("--output-dir", required=True, help="where extracted frames are written")
    ap.add_argument("--metadata", default="data/sources/videos.csv")
    ap.add_argument("--frames", default="data/sources/frames.csv")
    ap.add_argument("--video-cache", default=".video_cache")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--keep-videos", action="store_true",
                    help="keep downloaded videos (default: delete each after extraction)")
    args = ap.parse_args()

    _check_tool("yt-dlp")
    _check_tool("ffmpeg")

    videos = load_videos(Path(args.metadata))
    frames = load_frames(Path(args.frames), args.split)
    out_dir = Path(args.output_dir)
    cache = Path(args.video_cache)

    # Group frames by video_key so each video is downloaded once.
    by_video: dict[str, list[dict]] = {}
    for fr in frames:
        by_video.setdefault(fr["video_key"], []).append(fr)

    n_done = n_skip_meta = n_skip_ts = n_fail = 0
    for video_key, frs in sorted(by_video.items()):
        vrow = videos.get(video_key)
        url = resolve_url(vrow) if vrow else None
        if url is None:
            print(f"[skip] {video_key}: no youtube_id/url in {args.metadata} (unresolved)")
            n_skip_meta += len(frs)
            continue

        pending = []
        for fr in frs:
            ts = (fr.get("timestamp_sec") or "").strip()
            if ts == "":
                print(f"[skip] {fr['image']}: no timestamp_sec (unresolved)")
                n_skip_ts += 1
                continue
            out_path = out_dir / fr["image"]
            if args.skip_existing and out_path.exists():
                continue
            pending.append((float(ts), out_path, fr["image"]))

        if not pending:
            continue

        video_path = download_video(url, cache, video_key)
        if video_path is None:
            n_fail += len(pending)
            continue

        for ts, out_path, image in pending:
            if extract_frame(video_path, ts, out_path):
                n_done += 1
            else:
                print(f"  [warn] extraction failed: {image} @ {ts}s")
                n_fail += 1

        if not args.keep_videos:
            try:
                video_path.unlink()
            except OSError:
                pass

    print("\n=== summary ===")
    print(f"  frames extracted           : {n_done}")
    print(f"  skipped (no video metadata): {n_skip_meta}")
    print(f"  skipped (no timestamp)     : {n_skip_ts}")
    print(f"  failed                     : {n_fail}")
    if n_skip_meta or n_skip_ts:
        print("  NOTE: source video ids and/or frame timestamps are not yet filled in.")
        print("        See DATASET.md — this metadata must be completed to extract frames.")


if __name__ == "__main__":
    main()
