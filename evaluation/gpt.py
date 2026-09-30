#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gpt.py - UrbanStreetVQA evaluation for GPT-5.4 (OpenAI API).

Based on the paper's benchmark_final_verbose.py. Evaluation logic is preserved:
- Circular Evaluation for all multiple-choice categories.
- Counting is scored as a non-circular exact numeric match.
- Prompt, answer parsing, and scoring logic are unchanged.

The API key is read from the OPENAI_API_KEY environment variable (never hard-coded).
"""

import os
import re
import sys
import json
import argparse
import base64
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict

import pandas as pd
from openai import OpenAI, APITimeoutError


# ====== logging Tee ======

class Tee:
    """Write to both stdout and a log file."""
    def __init__(self, log_path: Path):
        self.terminal = sys.stdout
        self.log = open(log_path, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# ====== utilities ======

def encode_image(image_path: Path) -> Optional[str]:
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        print(f"[image encode error] {image_path}: {e}")
        return None


def print_category_report(cat_accuracies: Dict[str, Dict[str, Any]]):
    if not cat_accuracies:
        return
    print("\n[per-category results]")
    header = f"| {'Category':<24} | {'Acc':<7} | {'Correct':<8} | {'Total':<6} |"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for cat in sorted(cat_accuracies.keys()):
        stats = cat_accuracies[cat]
        acc_str = f"{stats['accuracy']*100:.1f}%"
        print(f"| {cat:<24} | {acc_str:<7} | {stats['correct_samples']:<8} | {stats['total_samples']:<6} |")
    print("-" * len(header))


# ====== main evaluation class ======

class StreetVQABenchmarkerGPT:
    numeric_categories = {"count", "counting"}

    def __init__(self, jsonl_path: str, image_dir: str, model_name: str = "gpt-5.4", api_key: Optional[str] = None):
        self.jsonl_path = Path(jsonl_path)
        self.image_dir = Path(image_dir)
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, timeout=60.0, max_retries=3)

        val = str(os.getenv("CIRCULAR_EVAL", "1")).lower()
        self.enable_circular_eval = val in {"1", "true", "yes", "on"}

        self.samples = self._load_jsonl(self.jsonl_path)
        print(f"Initialized Benchmarker. Model: {self.model_name} | Samples: {len(self.samples)} | CircularEval: {self.enable_circular_eval}")

    def _load_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        data = []
        with path.open("r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    data.append(json.loads(line))
                except Exception as e:
                    print(f"[warn] failed to parse line {ln}: {e}")
        return data

    def _resolve_image_path(self, ex: Dict[str, Any]) -> Path:
        """Resolve the image path using the city/video/image three-level structure."""
        img_name = ex.get("image") or ex.get("image_path", "")
        if Path(img_name).is_absolute():
            return Path(img_name)
        city = ex.get("city", "")
        video = ex.get("video", "")
        if city and video:
            return self.image_dir / city / video / img_name
        return self.image_dir / img_name

    def _build_system_prompt(self, category: str, choices: Optional[List[str]] = None) -> str:
        cat = (category or "").lower()
        if cat in self.numeric_categories:
            return (
                "You MUST answer using STRICT FORMAT.\n"
                "Task: Count based on the image.\n"
                "Output rules:\n"
                "- If the count is between 1 and 10, output the EXACT integer (e.g., 3).\n"
                "- If the count is 11 or more, output EXACTLY: 11 or more\n"
                "Do NOT output any other text or symbols."
            )
        if choices:
            return (
                "You MUST answer using STRICT FORMAT.\n"
                "Task: Choose the correct option letter based on the image.\n"
                "Output MUST be EXACTLY one uppercase letter (A, B, C, or D).\n"
                + "\n".join(choices)
            )
        return "Answer the question based on the image. Choose the correct option letter (A-D) only."

    def ask(self, question: str, image_path: Path, category: str, choices: Optional[List[str]] = None) -> str:
        try:
            base64_img = encode_image(image_path)
            if not base64_img:
                return "IMAGE_ERROR"
            messages = [
                {"role": "system", "content": self._build_system_prompt(category, choices)},
                {"role": "user", "content": [
                    {"type": "text", "text": f"Question: {question}\nAnswer:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}", "detail": "high"}}
                ]}
            ]
            response = self.client.chat.completions.create(
                model=self.model_name, messages=messages, max_completion_tokens=50, temperature=0
            )
            return (response.choices[0].message.content or "").strip()
        except APITimeoutError:
            return "TIMEOUT_ERROR"
        except Exception as e:
            return f"ERROR: {str(e)}"

    def postprocess(self, raw: str, category: str) -> str:
        if raw in ["API_ERROR", "ERROR", "IMAGE_ERROR", "TIMEOUT_ERROR", None] or str(raw).startswith("ERROR:"):
            return "INVALID_ANSWER"
        text = re.sub(r"\*\*", "", raw).strip().strip(".")
        cat = (category or "").lower()

        if cat in self.numeric_categories:
            if "11 or more" in text.lower():
                return "11 or more"
            m = re.search(r"\d+", text)
            return m.group(0) if m else text

        m = re.search(r"[A-D]", text.upper())
        if m:
            return m.group(0)
        return text

    def compare(self, pred: str, gt: str, category: str, options: Optional[List[str]] = None) -> bool:
        if pred == "INVALID_ANSWER":
            return False
        p_val = str(pred).strip()
        g_val = str(gt).strip()
        cat = (category or "").lower()

        if cat in self.numeric_categories:
            if g_val.lower() == "11 or more":
                return p_val.lower() == "11 or more" or (p_val.isdigit() and int(p_val) >= 11)
            return p_val == g_val

        if p_val.upper() == g_val.upper():
            return True

        if options and len(p_val) == 1 and p_val.upper() in "ABCD":
            idx = ord(p_val.upper()) - 65
            if 0 <= idx < len(options):
                opt_text = re.sub(r"^[A-D]:\s*", "", options[idx]).strip().lower()
                return opt_text == g_val.lower()
        return False

    def circular_eval(self, image_path: Path, question: str, category: str, options: List[str], answer: str, verbose: bool = True):
        trials = []
        num = len(options)
        letters = [chr(65 + i) for i in range(num)]
        gt_idx = -1

        for i, o in enumerate(options):
            if str(o).strip().lower() == str(answer).strip().lower() or str(answer).upper() == letters[i]:
                gt_idx = i
                break
        if gt_idx < 0:
            if verbose:
                print(f"    [WARN] ground truth not found in options: answer={answer!r}")
            return False, trials

        all_ok = True
        for shift in range(num):
            shifted = options[shift:] + options[:shift]
            choices_text = [f"{letters[i]}: {opt}" for i, opt in enumerate(shifted)]
            correct_letter = letters[(gt_idx - shift) % num]
            raw = self.ask(question, image_path, category, choices=choices_text)
            pred = self.postprocess(raw, category)
            ok = (pred == correct_letter)
            trials.append({"shift": shift, "is_correct": ok, "raw": raw, "pred": pred, "expected": correct_letter})
            if verbose:
                print(f"    [shift {shift}] choices={shifted}")
                print(f"               raw={raw!r} | pred={pred} | expected={correct_letter} -> {'OK' if ok else 'NG'}")
            if not ok:
                all_ok = False
        return all_ok, trials

    def _process_sample(self, ex: Dict[str, Any], idx: int, total: int, verbose: bool = True) -> Dict[str, Any]:
        path = self._resolve_image_path(ex)
        cat = ex.get("category", "")
        gt_answer = ex.get("answer", "")
        options = ex.get("options", [])

        print(f"\n[{idx}/{total}] image={path.name} | category={cat}")
        print(f"    Q: {ex.get('question','')}")
        print(f"    Options: {options}")
        print(f"    GroundTruth: {gt_answer}")

        if not path.exists():
            print(f"    -> NOT_FOUND: {path}")
            return {**ex, "correct": False, "prediction": "NOT_FOUND", "category": cat}

        is_mcq = isinstance(options, list) and len(options) > 0

        if self.enable_circular_eval and is_mcq and cat.lower() not in self.numeric_categories:
            ok, trials = self.circular_eval(path, ex["question"], cat, options, gt_answer, verbose=verbose)
            res = {"correct": ok, "prediction": "ALL_CORRECT" if ok else "FAIL", "circular_trials": trials}
            print(f"    => Result: {'OK (ALL_CORRECT)' if ok else 'NG (FAIL)'}")
        else:
            choices = [f"{chr(65+j)}: {o}" for j, o in enumerate(options)] if is_mcq else None
            raw = self.ask(ex["question"], path, cat, choices=choices)
            pred = self.postprocess(raw, cat)
            ok = self.compare(pred, gt_answer, cat, options=options)
            res = {"correct": ok, "prediction": pred, "raw_answer": raw}
            print(f"    Raw: {raw!r}")
            print(f"    Prediction: {pred}")
            print(f"    => Result: {'OK' if ok else 'NG'}")

        return {**ex, **res}

    def run(self, data: List[Dict[str, Any]], start_idx: int, verbose: bool = True):
        results = []
        for i, ex in enumerate(data, 1):
            results.append(self._process_sample(ex, start_idx + i - 1, len(data), verbose=verbose))
        stats = defaultdict(lambda: {"correct": 0, "total": 0})
        for r in results:
            c = r.get("category", "OTHER")
            stats[c]["total"] += 1
            if r.get("correct"):
                stats[c]["correct"] += 1
        cat_acc = {
            c: {"accuracy": v["correct"] / v["total"], "correct_samples": v["correct"], "total_samples": v["total"]}
            for c, v in stats.items()
        }
        overall = sum(1 for r in results if r.get("correct")) / len(results) if results else 0
        return results, {"accuracy": overall, "category_accuracies": cat_acc}


def main():
    parser = argparse.ArgumentParser(description="UrbanStreetVQA GPT-5.4 evaluation (Circular Evaluation)")
    parser.add_argument("--annotations", "--jsonl", dest="jsonl",
                        default="data/benchmark/urbanstreetvqa.jsonl", help="input annotations jsonl")
    parser.add_argument("--image-dir", default="data/frames/benchmark", help="frame image root directory")
    parser.add_argument("--output-dir", "--out-dir", dest="out_dir",
                        default="results_gpt5_4", help="output directory")
    parser.add_argument("--model-name", default="gpt-5.4")
    parser.add_argument("--limit", type=int, default=None, help="max number of samples to process")
    parser.add_argument("--log", default=None, help="log file path (auto-generated under logs/ if omitted)")
    args = parser.parse_args()

    # API key from environment only (never hard-coded)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[fatal] environment variable OPENAI_API_KEY is not set.")
        return

    # log file setup
    if args.log:
        log_path = Path(args.log)
    else:
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"benchmark_final_{args.model_name}_{ts}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(log_path)
    print(f"=== log output: {log_path} ===")

    bench = StreetVQABenchmarkerGPT(
        args.jsonl, args.image_dir,
        model_name=args.model_name,
        api_key=api_key
    )
    samples = bench.samples[:args.limit] if args.limit else bench.samples
    if not samples:
        print("no samples found.")
        return

    results, summary = bench.run(samples, 1, verbose=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = f"results_{args.model_name}_{Path(args.jsonl).stem}"
    csv_path = out / f"{base}.csv"
    pd.DataFrame(results).to_csv(csv_path, index=False)

    final_acc = sum(1 for r in results if r.get("correct")) / len(results) * 100
    print(f"\n{'='*50}")
    print(f"Final Accuracy: {final_acc:.2f}% ({sum(1 for r in results if r.get('correct'))}/{len(results)})")
    print(f"Results saved: {csv_path}")
    print(f"Log saved: {log_path}")
    print_category_report(summary["category_accuracies"])


if __name__ == "__main__":
    main()
