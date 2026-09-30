#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_gemini.py
UrbanStreetVQA zero-shot baseline: Gemini 2.5 Flash (API)。

既存ハーネス benchmark_qwen3_5.py を厳密に踏襲する。
postprocess / compare / circular_eval / run / save_results / main の規約は
そのまま流用し、変更点は以下の2箇所のみ:
  - __init__ : google-genai クライアント初期化（APIキーは環境変数のみ）
  - ask()    : Gemini API 呼び出し（画像→テキスト順 / system_instruction /
               temperature=0 greedy / thinking OFF / max_output_tokens=16 /
               レート制限リトライ）

APIキーは環境変数 GEMINI_API_KEY から取得する。スクリプトには絶対に書かない。
Gemini 2.5 Flash は zero-shot ベースラインとしてのみ追加（FTなし）。
"""

import os
import re
import json
import time
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
from collections import defaultdict

import pandas as pd
from PIL import Image

from google import genai
from google.genai import types

# ====== ユーティリティ（benchmark_qwen3_5.py と同一） ======

def load_image(path: Path) -> Optional[Image.Image]:
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        print(f"[画像読込エラー] {path}: {e}")
        return None

def print_category_report(cat_accuracies: Dict[str, Dict[str, Any]]):
    if not cat_accuracies:
        return
    print("\n[カテゴリ別 成績詳細]")
    header = f"| {'Category':<24} | {'Acc':<7} | {'Correct':<8} | {'Total':<6} |"
    print("-" * len(header)); print(header); print("-" * len(header))
    for cat in sorted(cat_accuracies.keys()):
        stats = cat_accuracies[cat]
        print(f"| {cat:<24} | {stats['accuracy']*100:.1f}%   | "
              f"{stats['correct_samples']:<8} | {stats['total_samples']:<6} |")
    print("-" * len(header))

def print_subcategory_report(rows: List[Dict[str, Any]]):
    """category/subcategory 別。vp/cam の比較が本研究のコアなので必須。"""
    stat = defaultdict(lambda: [0, 0])
    for r in rows:
        key = f"{r.get('category','')}/{r.get('subcategory','')}"
        stat[key][1] += 1
        if r.get("correct"):
            stat[key][0] += 1
    print("\n[サブカテゴリ別 成績（category/subcategory）]")
    header = f"| {'category/subcategory':<40} | {'Acc':<7} | {'Correct/Total':<14} |"
    print("-" * len(header)); print(header); print("-" * len(header))
    for key in sorted(stat):
        c, t = stat[key]
        print(f"| {key:<40} | {c/t*100:5.1f}% | {f'{c}/{t}':<14} |")
    print("-" * len(header))
    def acc(k):
        c, t = stat.get(k, [0, 0]); return f"{c/t*100:.1f}% ({c}/{t})" if t else "n/a"
    print("\n[コア比較]")
    print(f"  カメラ視点 directional_localization/camera : {acc('directional_localization/camera')}")
    print(f"  カメラ視点 relative_distance/camera       : {acc('relative_distance/camera')}")
    print(f"  人物視点 directional_localization/viewpoint: {acc('directional_localization/viewpoint')}")
    print(f"  人物視点 relative_distance/viewpoint      : {acc('relative_distance/viewpoint')}")

# ====== メイン評価クラス ======

class StreetVQABenchmarkerGemini:

    numeric_categories = {"count", "counting"}

    def __init__(self, jsonl_path: str, image_dir: str,
                 model_name: str = "gemini-2.5-flash", max_retries: int = 6):
        self.jsonl_path = Path(jsonl_path)
        self.image_dir = Path(image_dir)
        self.model_name = model_name
        self.max_retries = max_retries

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("環境変数 GEMINI_API_KEY が未設定です。")
        self.client = genai.Client(api_key=api_key)
        print(f"Gemini client 初期化: model={model_name}")

        self._smoke_logged = 0

        val = str(os.getenv("CIRCULAR_EVAL", "1")).lower()
        self.enable_circular_eval = val in {"1", "true", "yes", "on"}
        print(f"Circular Evaluation {'有効' if self.enable_circular_eval else '無効'}")

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
                    print(f"[警告] JSONL行{ln}の解析に失敗: {e}")
        return data

    def ask(self, question: str, image: Image.Image, category: str,
            choices: Optional[List[str]] = None) -> str:
        """
        Gemini 2.5 Flash 仕様:
          - contents=[image, text]（画像→テキスト順）
          - system_instruction でstrictフォーマット指示
          - temperature=0（greedy・CircularEval再現性）
          - thinking OFF（thinking_budget=0）
          - max_output_tokens=16
          - レート制限/一時エラーは指数バックオフでリトライ
        システムプロンプト / 回答フォーマットは benchmark_qwen3_5.py と同一。
        """
        cat = (category or "").lower()
        # ── システムプロンプト（Qwen版と完全同一） ──
        if cat in self.numeric_categories:
            system_prompt = ("You MUST answer using STRICT FORMAT.\n"
                             "Task: Count based on the image.\n"
                             "Output MUST be EXACTLY one single integer.")
        elif choices:
            system_prompt = ("You MUST answer using STRICT FORMAT.\n"
                             "Task: Choose the correct option based on the image.\n"
                             "Output MUST be EXACTLY one single uppercase letter (A, B, C, or D) and NOTHING ELSE.\n"
                             "Do NOT add explanations or the option text.")
        else:
            system_prompt = "Answer the question based on the image. Be concise and accurate."

        # ── ユーザー質問（Qwen版と完全同一） ──
        if choices:
            choices_text = "\n".join(choices)
            user_question = (f"Question: {question}\n\n"
                             f"Options:\n{choices_text}\n\n"
                             f"Choose the letter corresponding to your answer. Your output must be a single letter only.\n"
                             f"Answer:")
        else:
            user_question = f"Question: {question}\nAnswer:"

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.0,
            max_output_tokens=16,
            thinking_config=types.ThinkingConfig(thinking_budget=0),  # thinking OFF
        )

        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[image, user_question],  # 画像 → テキスト
                    config=config,
                )
                text = (resp.text or "").strip()
                if self._smoke_logged < 3:
                    print(f"  [SMOKE] raw='{text}'")
                    self._smoke_logged += 1
                return text if text else "INVALID_ANSWER"
            except Exception as e:
                last_err = e
                msg = str(e)
                # レート制限/一時エラーは指数バックオフ
                transient = any(s in msg for s in
                                ("429", "RESOURCE_EXHAUSTED", "503", "500",
                                 "UNAVAILABLE", "deadline", "DEADLINE", "overloaded"))
                wait = min(60, (2 ** attempt) + 1)
                if transient:
                    print(f"  [retry {attempt+1}/{self.max_retries}] {type(e).__name__}: {msg[:80]} → {wait}s待機")
                    time.sleep(wait)
                else:
                    print(f"  [エラー] Gemini呼び出し: {type(e).__name__}: {msg[:150]}")
                    time.sleep(min(5, wait))
        print(f"  [致命] リトライ上限。最後のエラー: {last_err}")
        return "ERROR"

    # ===== postprocess / compare / circular_eval / _process_sample / run は
    #       benchmark_qwen3_5.py と完全同一 =====

    def postprocess(self, raw_answer: str, category: str) -> str:
        if raw_answer is None or raw_answer in ["ERROR", "INVALID_ANSWER"]:
            return "INVALID_ANSWER"
        text = str(raw_answer).strip()
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
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
        m = re.search(r'^([A-D])\b', text.upper())
        if m: return m.group(1)
        m = re.search(r'\b([A-D])\b', text.upper())
        if m: return m.group(1)
        m = re.search(r"[A-D]", text.upper())
        if m: return m.group(0)
        return text

    def compare(self, pred: str, gt: str, category: str, options: Optional[List[str]] = None) -> bool:
        if pred in {"ERROR", "INVALID_ANSWER", "API_ERROR", "TIMEOUT_ERROR"}:
            return False
        pred_str = str(pred).strip(); gt_str = str(gt).strip()
        cat = (category or "").lower()
        if cat in self.numeric_categories:
            if gt_str.lower() == "11 or more":
                return pred_str.lower() == "11 or more" or (pred_str.isdigit() and int(pred_str) >= 11)
            return pred_str == gt_str
        if pred_str.upper() == gt_str.upper():
            return True
        if pred_str.upper() in {"A", "B", "C", "D"} and options:
            idx = ord(pred_str.upper()) - 65
            if 0 <= idx < len(options):
                return str(options[idx]).strip() == gt_str
        return False

    def circular_eval(self, image_path: Path, question: str, category: str, options: List[str], answer: str):
        trials: List[Dict[str, Any]] = []
        num_options = len(options)
        letters = [chr(65 + i) for i in range(num_options)]
        img = load_image(image_path)
        if img is None:
            return False, []
        gt_str = str(answer).strip(); gt_idx = -1
        if len(gt_str) == 1 and "A" <= gt_str.upper() <= letters[-1]:
            gt_idx = ord(gt_str.upper()) - 65
        else:
            for i, o in enumerate(options):
                if str(o).strip() == gt_str:
                    gt_idx = i; break
        if gt_idx < 0:
            print(f"[警告] CircularEval: 正解 '{answer}' が options {options} に見つかりません。")
            return False, trials
        all_ok = True
        print(f"    --- Circular Eval ({num_options}択) ---")
        for shift in range(num_options):
            shifted_options = options[shift:] + options[:shift]
            choices_text = [f"{letters[i]}: {opt}" for i, opt in enumerate(shifted_options)]
            current_gt_idx = (gt_idx - shift) % num_options
            correct_letter = letters[current_gt_idx]
            raw = self.ask(question, img, category, choices=choices_text)
            pred = self.postprocess(raw, category)
            ok = (pred == correct_letter)
            print(f"    Shift {shift}/{num_options-1} | Choices: {' / '.join(choices_text)}")
            print(f"      Raw Output: {raw} | Prediction: {pred} | GT: {correct_letter} | Correct: {ok}")
            trials.append({"shift": shift, "shifted_options": shifted_options, "choices_text": choices_text,
                           "raw": raw, "pred": pred, "correct_letter": correct_letter, "is_correct": ok})
            if not ok:
                all_ok = False
        print("    -------------------------------------")
        del img
        return all_ok, trials

    def _process_sample(self, ex: Dict[str, Any], idx: int, total: int) -> Optional[Dict[str, Any]]:
        image_name = ex.get("image") or ex.get("image_path")
        question = ex.get("question", ""); category = ex.get("category", "")
        subcategory = ex.get("subcategory", ""); answer = ex.get("answer", "")
        options = ex.get("options", None)
        city = ex.get("city", ""); video = ex.get("video", "")

        img_path = Path(image_name)
        if not img_path.is_absolute():
            if city and video:
                img_path = self.image_dir / city / video / image_name
            else:
                img_path = self.image_dir / image_name
        if not img_path.exists():
            hits = list(self.image_dir.rglob(Path(image_name).name))
            if hits:
                img_path = hits[0]
        if not img_path.exists():
            print(f"[{idx}/{total}] [スキップ] 画像が見つかりません: {img_path}")
            return {"index": idx, "image": str(img_path), "category": category, "subcategory": subcategory,
                    "question": question, "ground_truth": answer, "prediction": "NOT_FOUND",
                    "raw_answer": "", "correct": False, "options_full": " | ".join(options or [])}

        print(f"\n[{idx}/{total}] {img_path.name} | Cat={category}/{subcategory} | Q={question}")
        is_mcq = isinstance(options, list) and len(options) > 0
        if (self.enable_circular_eval and is_mcq and len(options) >= 2 and
                category.lower() not in self.numeric_categories):
            ok, trials = self.circular_eval(img_path, question, category, options, answer)
            pred = "ALL_CORRECT" if ok else f"FAIL({sum(t['is_correct'] for t in trials)}/{len(trials)})"
            raw = f"[CIRCULAR_EVAL_{len(options)}択]"; correct = ok
            extra = {"circular_trials": trials}
            print(f"  Result: {pred} | Correct: {correct}")
        else:
            img = load_image(img_path)
            if img is None:
                return {"index": idx, "image": str(img_path), "category": category, "subcategory": subcategory,
                        "question": question, "ground_truth": answer, "prediction": "IMAGE_LOAD_ERROR",
                        "raw_answer": "", "correct": False, "options_full": " | ".join(options or [])}
            choices_text = [f"{chr(65+j)}: {opt}" for j, opt in enumerate(options)] if is_mcq else None
            raw = self.ask(question, img, category, choices=choices_text)
            pred = self.postprocess(raw, category)
            correct = self.compare(pred, answer, category, options=options)
            extra = {}
            print(f"  Raw Output: {raw} | Prediction: {pred} | GT: {answer} | Correct: {correct}")
            del img
        return {"index": idx, "image": str(img_path), "category": category, "subcategory": subcategory,
                "question": question,
                "options_full": " | ".join([f"{chr(65+j)}:{o}" for j, o in enumerate(options)]) if is_mcq else "",
                "ground_truth": answer, "prediction": pred, "raw_answer": raw, "correct": correct, **extra}

    def run(self, data_to_run: List[Dict[str, Any]], start_idx: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        total_count = len(data_to_run)
        for i, ex in enumerate(data_to_run, 1):
            result = self._process_sample(ex, start_idx + i - 1, total_count)
            if result:
                results.append(result)
                stats[result["category"]]['total'] += 1
                if result["correct"]:
                    stats[result["category"]]['correct'] += 1
        total_samples = len(results)
        correct_total = sum(1 for r in results if r["correct"])
        acc = (correct_total / total_samples) if total_samples else 0.0
        cat_acc = {c: {"accuracy": s["correct"]/s["total"], "correct_samples": s["correct"],
                       "total_samples": s["total"]} for c, s in stats.items() if s["total"] > 0}
        return results, {"accuracy": acc, "total_samples": total_samples,
                         "correct_samples": correct_total, "category_accuracies": cat_acc}

# ====== 結果保存（benchmark_qwen3_5.py と同一フォーマット） ======
def save_results(evals: Dict[str, Any], out_dir: Path, base: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    results_list = evals.get("results", [])
    if not results_list:
        print("[警告] 保存する評価結果がありません。"); return
    try:
        summary_df = pd.DataFrame([{"total_samples": evals["total_samples"],
                                    "correct_samples": evals["correct_samples"],
                                    "accuracy": evals["accuracy"]}])
        cat_df = pd.DataFrame([{"category": c, **v} for c, v in (evals.get("category_accuracies") or {}).items()])
        details_df = pd.DataFrame(results_list)
        if "circular_trials" in details_df.columns:
            details_df["circular_trials"] = details_df["circular_trials"].apply(
                lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, list) else x)
        xlsx_path = out_dir / f"{base}_results.xlsx"
        with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as w:
            summary_df.to_excel(w, index=False, sheet_name="summary")
            cat_df.to_excel(w, index=False, sheet_name="by_category")
            details_df.to_excel(w, index=False, sheet_name="details")
        print(f"[保存] Excel: {xlsx_path}")
    except Exception as e:
        print(f"[警告] Excel保存に失敗: {e}")
    try:
        details_path = out_dir / f"{base}_details.csv"
        pd.DataFrame(results_list).to_csv(details_path, index=False)
        print(f"[保存] CSV: {details_path}")
    except Exception as e:
        print(f"[警告] CSV保存に失敗: {e}")

# ====== main ======
def main():
    parser = argparse.ArgumentParser(description="UrbanStreetVQA Gemini 2.5 Flash zero-shot baseline")
    parser.add_argument("--annotations", "--jsonl", dest="jsonl", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", "--out-dir", dest="out_dir", default="results_gemini25flash")
    parser.add_argument("--model-name", default="gemini-2.5-flash")
    parser.add_argument("--smoke", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not Path(args.jsonl).exists():
        print(f"[致命的エラー] JSONLが見つかりません: {args.jsonl}"); return
    if not Path(args.image_dir).exists():
        print(f"[致命的エラー] 画像ディレクトリが見つかりません: {args.image_dir}"); return

    bench = StreetVQABenchmarkerGemini(args.jsonl, args.image_dir, model_name=args.model_name)
    all_samples = bench.samples
    if not all_samples:
        print("評価するサンプルがありません。"); return

    if args.smoke and args.smoke > 0:
        print("\n" + "=" * 60)
        print(f"=== スモークテスト: 先頭 {args.smoke} 問 ===")
        print("=" * 60)
        results, summary = bench.run(all_samples[:args.smoke], start_idx=1)
        print(f"\nスモーク精度: {summary['accuracy']*100:.2f}%  "
              f"({summary['correct_samples']}/{summary['total_samples']})")
        print_subcategory_report(results)
        n_bad = sum(1 for r in results if r.get("prediction") in {"INVALID_ANSWER", "ERROR", "NOT_FOUND"})
        print(f"\n出力異常(INVALID/ERROR/NOT_FOUND): {n_bad}/{len(results)}")
        print("→ raw が 'B' 形式で正しくパースできているか確認後、--smoke を外して全件実行。")
        return

    if args.limit:
        all_samples = all_samples[:args.limit]

    print("\n" + "=" * 50)
    print(f"=== 全 {len(all_samples)} 問 評価 (Gemini 2.5 Flash, zero-shot) ===")
    print("=" * 50)
    all_results, _ = bench.run(all_samples, start_idx=1)

    final_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in all_results:
        final_stats[r["category"]]["total"] += 1
        if r.get("correct", False):
            final_stats[r["category"]]["correct"] += 1
    final_total = len(all_results)
    final_correct = sum(1 for r in all_results if r.get("correct", False))
    final_acc = (final_correct / final_total) if final_total else 0.0
    final_cat_acc = {c: {"accuracy": s["correct"]/s["total"], "correct_samples": s["correct"],
                         "total_samples": s["total"]} for c, s in final_stats.items() if s["total"] > 0}
    final_evals = {"accuracy": final_acc, "total_samples": final_total, "correct_samples": final_correct,
                   "category_accuracies": final_cat_acc, "results": all_results}

    print("\n" + "=" * 50)
    print("=== 最終結果 (Gemini 2.5 Flash zero-shot) ===")
    print(f"全体精度: {final_acc*100:.2f}%  ({final_correct}/{final_total})")
    print_category_report(final_cat_acc)
    print_subcategory_report(all_results)
    print("=" * 50)

    base = f"{Path(args.jsonl).stem}_gemini25flash"
    save_results(final_evals, Path(args.out_dir), base)

if __name__ == "__main__":
    main()
