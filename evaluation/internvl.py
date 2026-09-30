#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
internvl.py - UrbanStreetVQA evaluation for InternVL3.5.

Based on the paper's benchmark_internvl3.py. Evaluation logic is preserved:
- Circular Evaluation for all multiple-choice categories (2/3/4 options).
- Counting is scored as a non-circular exact numeric match.
- Model handling, pre/post-processing, and scoring are unchanged.
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
from collections import defaultdict

import pandas as pd
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

torch.set_grad_enabled(False)

# ====== utilities ======

def load_image(path: Path) -> Optional[Image.Image]:
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        print(f"[image load error] {path}: {e}")
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
        correct = stats['correct_samples']
        total = stats['total_samples']
        print(f"| {cat:<24} | {acc_str:<7} | {correct:<8} | {total:<6} |")
    print("-" * len(header))

# ====== main evaluation class ======

class StreetVQABenchmarkerInternVL35:

    # Only counting uses plain evaluation; everything else uses Circular Evaluation.
    numeric_categories = {"count", "counting"}

    def __init__(self, jsonl_path: str, image_dir: str, model_path: str = "PATH/TO/InternVL3.5"):
        self.jsonl_path = Path(jsonl_path)
        self.image_dir = Path(image_dir)
        self.model_path = model_path

        print(f"Loading model from: {model_path}...")
        if torch.cuda.is_available():
            self.device = "cuda"
            print(">>> using GPU (CUDA) <<<")
        else:
            self.device = "cpu"
            print(">>> no GPU found; using CPU <<<")

        try:
            self.processor = AutoProcessor.from_pretrained(
                model_path, trust_remote_code=True
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map="auto",
                trust_remote_code=True
            )
            print(f"Model loaded successfully.")
        except Exception as e:
            print(f"[fatal] failed to load model: {e}")
            raise

        val = str(os.getenv("CIRCULAR_EVAL", "1")).lower()
        self.enable_circular_eval = val in {"1", "true", "yes", "on"}
        print(f"Circular Evaluation {'enabled' if self.enable_circular_eval else 'disabled'}")

        self.samples = self._load_jsonl(self.jsonl_path)
        print(f"Loaded {len(self.samples)} samples from {self.jsonl_path}")

    def _load_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        data = []
        with path.open("r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except Exception as e:
                    print(f"[warn] failed to parse jsonl line {ln}: {e}")
        return data

    def _build_system_prompt(self, category: str, choices: Optional[List[str]] = None) -> str:
        cat = (category or "").lower()
        if cat in self.numeric_categories:
            return ("You MUST answer using STRICT FORMAT.\n"
                    "Task: Count based on the image.\n"
                    "Output MUST be EXACTLY one single integer.\n")
        if choices:
            return ("You MUST answer using STRICT FORMAT.\n"
                    "Task: Choose the correct option based on the image.\n"
                    "Output MUST be EXACTLY one single uppercase letter (A, B, C, or D) and NOTHING ELSE.\n"
                    "Do NOT add explanations or the option text.\n"
                    + "\n".join(choices))
        return "Answer the question based on the image. Be concise and accurate."

    def _final_instruction(self, category: str, has_choices: bool) -> str:
        cat = (category or "").lower()
        if has_choices and cat not in self.numeric_categories:
            return "\nChoose the letter corresponding to your answer. Your output must be a single letter only.\nAnswer:"
        return "\nAnswer:"

    @torch.no_grad()
    def ask(self, question: str, image: Image.Image, category: str, choices: Optional[List[str]] = None) -> str:
        try:
            system_prompt = self._build_system_prompt(category, choices)
            final_instruction = self._final_instruction(category, has_choices=bool(choices))
            user_question = f"Question: {question}{final_instruction}"

            # InternVL3.5 input format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": image},
                        {"type": "text", "text": f"{system_prompt}\n{user_question}"}
                    ]
                }
            ]

            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.device)

            input_token_len = inputs["input_ids"].shape[1]
            max_new = 5 if (category or "").lower() not in self.numeric_categories else 15

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                temperature=None,
                top_p=None
            )

            generated_tokens = outputs[0, input_token_len:]
            response = self.processor.decode(generated_tokens, skip_special_tokens=True)
            return response.strip() if response else "INVALID_ANSWER"
        except Exception as e:
            print(f"[error] InternVL3.5 inference error: {e}")
            return "ERROR"

    def postprocess(self, raw_answer: str, category: str) -> str:
        if raw_answer is None or raw_answer in ["ERROR", "INVALID_ANSWER"]:
            return "INVALID_ANSWER"
        text = str(raw_answer).strip()
        text = re.sub(r'\[\s*picture\s*\d+\s*\]|\(\s*picture\s*\d+\s*\)', '', text, flags=re.IGNORECASE).strip()
        text = text.split("###")[0].strip()
        text = re.sub(r"\*\*", "", text)
        if text.endswith("."):
            text = text[:-1]
        text = text.strip()
        cat = (category or "").lower()
        if cat in self.numeric_categories:
            if "11 or more" in text.lower():
                return "11 or more"
            m = re.search(r"\d+", text)
            if m:
                try:
                    val = int(m.group(0))
                    return "11 or more" if val >= 11 else str(val)
                except ValueError:
                    return m.group(0)
        # MCQ: extract an A-D letter
        m = re.search(r"[A-D]", text.upper())
        if m:
            return m.group(0)
        return text

    def compare(self, pred: str, gt: str, category: str, options: Optional[List[str]] = None) -> bool:
        if pred in {"ERROR", "INVALID_ANSWER", "API_ERROR", "TIMEOUT_ERROR"}:
            return False
        pred_str = str(pred).strip()
        gt_str = str(gt).strip()
        cat = (category or "").lower()

        # numeric comparison
        if cat in self.numeric_categories:
            if gt_str.lower() == "11 or more":
                return pred_str.lower() == "11 or more" or (pred_str.isdigit() and int(pred_str) >= 11)
            return pred_str == gt_str

        # MCQ comparison (letter match or option-text match)
        if pred_str.upper() == gt_str.upper():
            return True

        if pred_str.upper() in {"A", "B", "C", "D"} and options:
            idx = ord(pred_str.upper()) - 65
            if 0 <= idx < len(options):
                return str(options[idx]).strip() == gt_str
        return False

    def circular_eval(self, image_path: Path, question: str, category: str, options: List[str], answer: str):
        """
        Circular Evaluation: rotate the options and test every permutation.
        2 options: A,B -> B,A
        3 options: A,B,C -> B,C,A -> C,A,B
        4 options: A,B,C,D -> B,C,D,A -> C,D,A,B -> D,A,B,C
        """
        trials: List[Dict[str, Any]] = []
        num_options = len(options)
        letters = [chr(65 + i) for i in range(num_options)]
        img = load_image(image_path)
        if img is None:
            return False, []

        gt_str = str(answer).strip()
        gt_idx = -1

        # if GT is text, find its index
        if len(gt_str) == 1 and "A" <= gt_str.upper() <= letters[-1]:
            gt_idx = ord(gt_str.upper()) - 65
        else:
            for i, o in enumerate(options):
                if str(o).strip() == gt_str:
                    gt_idx = i
                    break

        if gt_idx < 0:
            print(f"[warn] CircularEval: ground truth '{answer}' not found in options {options}.")
            return False, trials

        all_ok = True
        print(f"    --- Circular Eval ({num_options} options) ---")
        for shift in range(num_options):
            shifted_options = options[shift:] + options[:shift]
            choices_text = [f"{letters[i]}: {opt}" for i, opt in enumerate(shifted_options)]
            current_gt_idx = (gt_idx - shift) % num_options
            correct_letter = letters[current_gt_idx]
            raw = self.ask(question, img, category, choices=choices_text)
            pred = self.postprocess(raw, category)
            ok = (pred == correct_letter)
            print(f"    Shift {shift}/{num_options-1} | Choices: {' / '.join(choices_text)}")
            print(f"      Raw Output: {raw}")
            print(f"      Prediction: {pred} | GT: {correct_letter} | Correct: {ok}")
            trials.append({
                "shift": shift, "shifted_options": shifted_options, "choices_text": choices_text,
                "raw": raw, "pred": pred, "correct_letter": correct_letter, "is_correct": ok
            })
            if not ok:
                all_ok = False
        print("    -------------------------------------")
        del img
        return all_ok, trials

    def _process_sample(self, ex: Dict[str, Any], idx: int, total: int) -> Optional[Dict[str, Any]]:
        image_name = ex.get("image") or ex.get("image_path")
        question = ex.get("question", "")
        category = ex.get("category", "")
        subcategory = ex.get("subcategory", "")
        answer = ex.get("answer", "")
        options = ex.get("options", None)

        # ===== image path resolution using city/video =====
        city = ex.get("city", "")
        video = ex.get("video", "")

        img_path = Path(image_name)
        if not img_path.is_absolute():
            if city and video:
                img_path = self.image_dir / city / video / image_name
            else:
                img_path = self.image_dir / image_name
        # ==================================================

        if not img_path.exists():
            print(f"[{idx}/{total}] [skip] image not found: {img_path}")
            return {
                "index": idx, "image": str(img_path), "category": category, "subcategory": subcategory,
                "question": question, "ground_truth": answer, "prediction": "NOT_FOUND",
                "raw_answer": "", "correct": False, "options_full": " | ".join(options or [])
            }

        print(f"\n[{idx}/{total}] {img_path.name} | Cat={category} | Q={question}")
        is_mcq = isinstance(options, list) and len(options) > 0

        # every multiple-choice category except counting uses Circular Evaluation
        if (self.enable_circular_eval and is_mcq and len(options) >= 2 and
                category.lower() not in self.numeric_categories):
            ok, trials = self.circular_eval(img_path, question, category, options, answer)
            pred = "ALL_CORRECT" if ok else f"FAIL({sum(t['is_correct'] for t in trials)}/{len(trials)})"
            raw = f"[CIRCULAR_EVAL_{len(options)}opt]"
            correct = ok
            extra = {"circular_trials": trials}
            print(f"  Result: {pred} | Correct: {correct}")
        else:
            # counting, or no options -> plain evaluation
            img = load_image(img_path)
            if img is None:
                print(f"  [skip] image load failed")
                return {
                    "index": idx, "image": str(img_path), "category": category, "subcategory": subcategory,
                    "question": question, "ground_truth": answer, "prediction": "IMAGE_LOAD_ERROR",
                    "raw_answer": "", "correct": False, "options_full": " | ".join(options or [])
                }

            choices_text = [f"{chr(65+j)}: {opt}" for j, opt in enumerate(options)] if is_mcq else None
            raw = self.ask(question, img, category, choices=choices_text)
            pred = self.postprocess(raw, category)
            correct = self.compare(pred, answer, category, options=options)
            extra = {}
            print(f"  Raw Output: {raw}")
            print(f"  Prediction: {pred} | GT: {answer} | Correct: {correct}")
            del img

        return {
            "index": idx, "image": str(img_path), "category": category, "subcategory": subcategory,
            "question": question,
            "options_full": " | ".join([f"{chr(65+j)}:{o}" for j, o in enumerate(options)]) if is_mcq else "",
            "ground_truth": answer, "prediction": pred, "raw_answer": raw, "correct": correct, **extra
        }

    def run(self, data_to_run: List[Dict[str, Any]], start_idx: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        total_count = len(data_to_run)
        for i, ex in enumerate(data_to_run, 1):
            current_idx = start_idx + i - 1
            result = self._process_sample(ex, current_idx, total_count)
            if result:
                results.append(result)
                category = result["category"]
                stats[category]['total'] += 1
                if result["correct"]:
                    stats[category]['correct'] += 1
        total_samples = len(results)
        correct_total = sum(1 for r in results if r["correct"])
        acc = (correct_total / total_samples) if total_samples else 0.0
        cat_acc = {}
        for cat, s in stats.items():
            if s["total"] > 0:
                cat_acc[cat] = {
                    "accuracy": s["correct"] / s["total"],
                    "correct_samples": s["correct"],
                    "total_samples": s["total"]
                }
        return results, {"accuracy": acc, "total_samples": total_samples,
                         "correct_samples": correct_total, "category_accuracies": cat_acc}

# ====== save results ======
def save_results(evals: Dict[str, Any], out_dir: Path, base: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    results_list = evals.get("results", [])
    if not results_list:
        print("[warn] no evaluation results to save.")
        return
    try:
        summary_df = pd.DataFrame([{
            "total_samples": evals["total_samples"],
            "correct_samples": evals["correct_samples"],
            "accuracy": evals["accuracy"],
        }])
        cat_df = pd.DataFrame(
            [{"category": c, **v} for c, v in (evals.get("category_accuracies") or {}).items()]
        )
        details_df = pd.DataFrame(results_list)
        if "circular_trials" in details_df.columns:
            details_df["circular_trials"] = details_df["circular_trials"].apply(
                lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, list) else x
            )
        xlsx_path = out_dir / f"{base}_results.xlsx"
        with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as w:
            summary_df.to_excel(w, index=False, sheet_name="summary")
            cat_df.to_excel(w, index=False, sheet_name="by_category")
            details_df.to_excel(w, index=False, sheet_name="details")
        print(f"[saved] Excel: {xlsx_path}")
    except Exception as e:
        print(f"[warn] failed to save Excel: {e}")
    try:
        details_path = out_dir / f"{base}_details.csv"
        pd.DataFrame(results_list).to_csv(details_path, index=False)
        print(f"[saved] CSV: {details_path}")
    except Exception as e:
        print(f"[warn] failed to save CSV: {e}")

# ====== main ======
def main():
    parser = argparse.ArgumentParser(description="UrbanStreetVQA InternVL3.5 evaluation (Circular Evaluation)")
    parser.add_argument("--annotations", "--jsonl", dest="jsonl", required=True, help="input annotations jsonl")
    parser.add_argument("--image-dir", required=True, help="frame image root directory")
    parser.add_argument("--output-dir", "--out-dir", dest="out_dir", default="results_internvl35", help="output directory")
    parser.add_argument("--model-path", default="PATH/TO/InternVL3.5", help="model path or HF id")
    parser.add_argument("--limit", type=int, default=None, help="max number of samples to process")
    parser.add_argument("--auto-yes", action="store_true", help="auto-confirm running the remaining samples")
    args = parser.parse_args()

    if not Path(args.jsonl).exists():
        print(f"[fatal] jsonl not found: {args.jsonl}")
        return
    if not Path(args.image_dir).exists():
        print(f"[fatal] image directory not found: {args.image_dir}")
        return

    bench = StreetVQABenchmarkerInternVL35(args.jsonl, args.image_dir, model_path=args.model_path)
    all_samples = bench.samples
    all_results: List[Dict[str, Any]] = []

    if not all_samples:
        print("no samples to evaluate.")
        return

    samples_by_category = defaultdict(list)
    for sample in all_samples:
        samples_by_category[sample.get("category", "OTHER")].append(sample)

    test_samples: List[Dict[str, Any]] = []
    full_remaining_samples: List[Dict[str, Any]] = []
    for cat in sorted(samples_by_category.keys()):
        items = samples_by_category[cat]
        test_samples.extend(items[:3])
        full_remaining_samples.extend(items[3:])

    if args.limit is not None and args.limit > 0:
        combined_samples = test_samples + full_remaining_samples
        limited_samples = combined_samples[:args.limit]
        test_samples = []
        full_remaining_samples = []
        current_test_count = 0
        for cat in sorted(samples_by_category.keys()):
            original_items = samples_by_category[cat]
            for item in original_items[:3]:
                if item in limited_samples and current_test_count < 3 * len(samples_by_category):
                    test_samples.append(item)
                    current_test_count += 1
            for item in original_items[3:]:
                if item in limited_samples:
                    full_remaining_samples.append(item)
        print(f"[LIMIT] applied a processing limit of {args.limit} samples.")

    if not test_samples and not full_remaining_samples:
        print("no evaluation samples; exiting.")
        return

    print("\n" + "=" * 50)
    print("=== STEP 1: per-category sample test ===")
    print("=" * 50)
    test_results, test_summary = bench.run(test_samples, start_idx=1)
    all_results.extend(test_results)

    print("\n" + "=" * 50)
    print("=== sample test summary ===")
    print(f"Accuracy: {test_summary['accuracy']*100:.2f}%")
    print_category_report(test_summary["category_accuracies"])
    print("=" * 50)

    run_all = False
    if full_remaining_samples:
        if args.auto_yes:
            print(f"\n[auto-yes] running the remaining {len(full_remaining_samples)} samples.")
            run_all = True
        else:
            while True:
                try:
                    user_input = input(
                        f"\nRun the remaining {len(full_remaining_samples)} samples? (yes/no): "
                    ).lower().strip()
                    if user_input == "yes":
                        run_all = True
                        break
                    elif user_input == "no":
                        break
                    else:
                        print("please enter yes/no.")
                except:
                    break
    else:
        print("no remaining samples.")

    if run_all:
        print("\n" + "=" * 50)
        print("=== STEP 2: running all remaining samples ===")
        print("=" * 50)
        remaining_results, _ = bench.run(full_remaining_samples, start_idx=len(test_samples) + 1)
        all_results.extend(remaining_results)

    final_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in all_results:
        cat = r["category"]
        final_stats[cat]["total"] += 1
        if r.get("correct", False):
            final_stats[cat]["correct"] += 1

    final_total = len(all_results)
    final_correct = sum(1 for r in all_results if r.get("correct", False))
    final_acc = (final_correct / final_total) if final_total else 0.0
    final_cat_acc = {}
    for cat, s in final_stats.items():
        if s["total"] > 0:
            final_cat_acc[cat] = {
                "accuracy": s["correct"] / s["total"],
                "correct_samples": s["correct"],
                "total_samples": s["total"],
            }

    final_evals = {
        "accuracy": final_acc, "total_samples": final_total,
        "correct_samples": final_correct, "category_accuracies": final_cat_acc,
        "results": all_results,
    }

    print("\n" + "=" * 50)
    print("=== final results ===")
    print(f"Accuracy: {final_evals['accuracy']*100:.2f}%")
    print_category_report(final_evals["category_accuracies"])
    print("=" * 50)

    base = f"{Path(args.jsonl).stem}_internvl35"
    save_results(final_evals, Path(args.out_dir), base)

if __name__ == "__main__":
    main()
