#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_gemma4.py
UrbanStreetVQA zero-shot baseline: Gemma 4 12B (instruction-tuned)。

既存ハーネス benchmark_qwen3_5.py を厳密に踏襲する。
postprocess / compare / circular_eval / run / save_results / main の規約は
そのまま流用し、変更点は以下の2箇所のみ:
  - __init__ : AutoProcessor + AutoModelForMultimodalLM でロード
  - ask()    : Gemma 4 仕様（thinking OFF / 画像→テキスト順 /
               parse_response / visual token budget 560 / greedy max_new_tokens=16）

Gemma 4 はFTせず zero-shot ベースラインとしてのみ追加する。

注意（知識カットオフ後のAPI。スモークテストで必ず検証すること）:
  - AutoModelForMultimodalLM / processor.parse_response / token budget の
    挙動は実機で確認する前提。ask() は防御的に実装しログを厚くしている。
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
from transformers import AutoProcessor, AutoModelForMultimodalLM

torch.set_grad_enabled(False)

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

def print_subcategory_report(rows: List[Dict[str, Any]]):
    """category/subcategory 別の精度。vp/cam の比較が本研究のコアなので必須。"""
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
    # カメラ視点 / 人物視点 を明示
    def acc(k):
        c, t = stat.get(k, [0, 0]); return f"{c/t*100:.1f}% ({c}/{t})" if t else "n/a"
    print("\n[コア比較]")
    print(f"  カメラ視点 directional_localization/camera : {acc('directional_localization/camera')}")
    print(f"  カメラ視点 relative_distance/camera       : {acc('relative_distance/camera')}")
    print(f"  人物視点 directional_localization/viewpoint: {acc('directional_localization/viewpoint')}")
    print(f"  人物視点 relative_distance/viewpoint      : {acc('relative_distance/viewpoint')}")

# ====== メイン評価クラス ======

class StreetVQABenchmarkerGemma4:

    # カウントカテゴリのみ通常評価、それ以外はサーキュラー評価（Qwen版と同一）
    numeric_categories = {"count", "counting"}

    def __init__(self, jsonl_path: str, image_dir: str,
                 model_path: str = "google/gemma-4-12B-it",
                 visual_tokens: int = 560):
        self.jsonl_path = Path(jsonl_path)
        self.image_dir = Path(image_dir)
        self.model_path = model_path
        self.visual_tokens = visual_tokens

        print(f"Loading Gemma 4: {model_path} ...")
        if torch.cuda.is_available():
            self.device = "cuda"
            print(">>> GPU (CUDA) を使用します <<<")
        else:
            self.device = "cpu"
            print(">>> GPUが見つからないため CPU を使用します <<<")

        # ── ロード: dtype="auto", device_map="auto" ──
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_path,
            dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        self.has_parse_response = hasattr(self.processor, "parse_response")
        print(f"Model loaded. parse_response available: {self.has_parse_response}")
        print(f"visual token budget(指定): {self.visual_tokens}")

        val = str(os.getenv("CIRCULAR_EVAL", "1")).lower()
        self.enable_circular_eval = val in {"1", "true", "yes", "on"}
        print(f"Circular Evaluation {'有効' if self.enable_circular_eval else '無効'}")

        self._smoke_logged = 0  # スモーク時に詳細ログを出す回数

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

    @torch.no_grad()
    def ask(self, question: str, image: Image.Image, category: str,
            choices: Optional[List[str]] = None) -> str:
        """
        Gemma 4 仕様:
          - thinking OFF: apply_chat_template(..., enable_thinking=False)
          - messages は 画像→テキスト の順
          - 出力は processor.parse_response() を通してからシンボル抽出
          - greedy (do_sample=False), max_new_tokens=16
          - visual token budget = 560（実渡し量をログ確認）
        システムプロンプト / 回答フォーマットは benchmark_qwen3_5.py と同一。
        """
        try:
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

            # ── messages: 画像 → テキスト の順。画像オブジェクトを content に埋める ──
            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_question},
                ]},
            ]

            # ── トークナイズ ──
            # 注: gemma-4-12B-it は Gemma4UnifiedImageProcessor で
            #     num_soft_tokens=max_soft_tokens=280 にアーキテクチャ固定。
            #     pan_and_scan/num_crops 等のマルチクロップ機構は無く、
            #     画像あたり280ソフトトークンが上限（=ネイティブ最大の視覚予算）。
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=False,   # thinking OFF
            )

            inputs = {k: (v.to(self.model.device) if isinstance(v, torch.Tensor) else v)
                      for k, v in inputs.items()}
            input_token_len = inputs["input_ids"].shape[1]

            # ── スモーク時: visual token 数を実測ログ ──
            if self._smoke_logged < 3:
                self._log_visual_tokens(inputs, input_token_len)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=False,
            )
            generated = outputs[0, input_token_len:]
            raw = self.processor.decode(generated, skip_special_tokens=True)

            # ── parse_response でthoughtタグ等を除去してから返す ──
            parsed = raw
            if self.has_parse_response:
                try:
                    pr = self.processor.parse_response(raw)
                    # parse_response の戻り値型はAPI次第。dict/str両対応で防御。
                    if isinstance(pr, dict):
                        parsed = pr.get("content") or pr.get("text") or pr.get("answer") or raw
                    elif isinstance(pr, str):
                        parsed = pr
                except Exception as e:
                    print(f"[警告] parse_response 失敗、raw使用: {e}")

            if self._smoke_logged < 3:
                print(f"  [SMOKE] raw='{raw}'  parsed='{parsed}'")
                self._smoke_logged += 1

            return parsed.strip() if parsed and parsed.strip() else "INVALID_ANSWER"
        except Exception as e:
            print(f"[エラー] Gemma4モデル質問処理エラー: {e}")
            return "ERROR"

    def _log_visual_tokens(self, inputs: dict, input_token_len: int):
        """visual token budget が実際に渡せているか確認するためのログ。"""
        n_img = None
        # image token id の特定を試みる（API差異に防御的）
        for attr in ("image_token_id", "image_token_index"):
            tid = getattr(self.model.config, attr, None)
            if tid is not None and "input_ids" in inputs:
                n_img = int((inputs["input_ids"][0] == tid).sum().item())
                break
        pv_shape = None
        for k in ("pixel_values", "pixel_values_images", "image_patches"):
            if k in inputs and hasattr(inputs[k], "shape"):
                pv_shape = tuple(inputs[k].shape); break
        print(f"  [SMOKE] input_len={input_token_len}  image_tokens={n_img}  "
              f"pixel_values_shape={pv_shape}  (budget指定={self.visual_tokens})")

    # ===== 以下 postprocess / compare / circular_eval / _process_sample / run は
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
        if m:
            return m.group(1)
        m = re.search(r'\b([A-D])\b', text.upper())
        if m:
            return m.group(1)
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

        gt_str = str(answer).strip()
        gt_idx = -1
        if len(gt_str) == 1 and "A" <= gt_str.upper() <= letters[-1]:
            gt_idx = ord(gt_str.upper()) - 65
        else:
            for i, o in enumerate(options):
                if str(o).strip() == gt_str:
                    gt_idx = i
                    break
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
        city = ex.get("city", "")
        video = ex.get("video", "")

        img_path = Path(image_name)
        if not img_path.is_absolute():
            if city and video:
                img_path = self.image_dir / city / video / image_name
            else:
                img_path = self.image_dir / image_name
        # フォールバック: stem 一致で探索（benchmark_qwen3_5.py と同様に堅牢化）
        if not img_path.exists():
            hits = list(self.image_dir.rglob(Path(image_name).name))
            if hits:
                img_path = hits[0]

        if not img_path.exists():
            print(f"[{idx}/{total}] [スキップ] 画像が見つかりません: {img_path}")
            return {
                "index": idx, "image": str(img_path), "category": category, "subcategory": subcategory,
                "question": question, "ground_truth": answer, "prediction": "NOT_FOUND",
                "raw_answer": "", "correct": False, "options_full": " | ".join(options or [])
            }

        print(f"\n[{idx}/{total}] {img_path.name} | Cat={category}/{subcategory} | Q={question}")
        is_mcq = isinstance(options, list) and len(options) > 0

        if (self.enable_circular_eval and is_mcq and len(options) >= 2 and
                category.lower() not in self.numeric_categories):
            ok, trials = self.circular_eval(img_path, question, category, options, answer)
            pred = "ALL_CORRECT" if ok else f"FAIL({sum(t['is_correct'] for t in trials)}/{len(trials)})"
            raw = f"[CIRCULAR_EVAL_{len(options)}択]"
            correct = ok
            extra = {"circular_trials": trials}
            print(f"  Result: {pred} | Correct: {correct}")
        else:
            img = load_image(img_path)
            if img is None:
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

# ====== 結果保存（benchmark_qwen3_5.py と同一フォーマット） ======
def save_results(evals: Dict[str, Any], out_dir: Path, base: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    results_list = evals.get("results", [])
    if not results_list:
        print("[警告] 保存する評価結果がありません。")
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
        print(f"[保存] Excel: {xlsx_path}")
    except Exception as e:
        print(f"[警告] Excel保存に失敗: {e}")
    try:
        details_path = out_dir / f"{base}_details.csv"
        pd.DataFrame(results_list).to_csv(details_path, index=False)
        print(f"[保存] CSV: {details_path}")
    except Exception as e:
        print(f"[警告] CSV保存に失敗: {e}")

# ====== main（benchmark_qwen3_5.py の構造を踏襲 + スモーク + サブカテゴリ出力） ======
def main():
    parser = argparse.ArgumentParser(description="UrbanStreetVQA Gemma 4 12B-it zero-shot baseline")
    parser.add_argument("--annotations", "--jsonl", dest="jsonl", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", "--out-dir", dest="out_dir", default="results_gemma4")
    parser.add_argument("--model-path", default="google/gemma-4-12B-it")
    parser.add_argument("--visual-tokens", type=int, default=560)
    parser.add_argument("--smoke", type=int, default=0,
                        help=">0 なら先頭N問だけ実行（スモークテスト）")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not Path(args.jsonl).exists():
        print(f"[致命的エラー] JSONLが見つかりません: {args.jsonl}"); return
    if not Path(args.image_dir).exists():
        print(f"[致命的エラー] 画像ディレクトリが見つかりません: {args.image_dir}"); return

    bench = StreetVQABenchmarkerGemma4(
        args.jsonl, args.image_dir,
        model_path=args.model_path, visual_tokens=args.visual_tokens,
    )
    all_samples = bench.samples
    if not all_samples:
        print("評価するサンプルがありません。"); return

    # ── スモークテスト: 先頭N問 ──
    if args.smoke and args.smoke > 0:
        print("\n" + "=" * 60)
        print(f"=== スモークテスト: 先頭 {args.smoke} 問 ===")
        print("=" * 60)
        smoke_samples = all_samples[:args.smoke]
        results, summary = bench.run(smoke_samples, start_idx=1)
        print("\n" + "=" * 60)
        print(f"スモーク精度: {summary['accuracy']*100:.2f}%  "
              f"({summary['correct_samples']}/{summary['total_samples']})")
        print_subcategory_report(results)
        n_bad = sum(1 for r in results
                    if r.get("prediction") in {"INVALID_ANSWER", "ERROR", "NOT_FOUND"})
        print(f"\n出力異常(INVALID/ERROR/NOT_FOUND): {n_bad}/{len(results)}")
        print("→ raw が 'B' 形式で正しくパースできているか上のログを確認してください。")
        print("  問題なければ --smoke を外して全件実行。")
        return

    if args.limit:
        all_samples = all_samples[:args.limit]

    print("\n" + "=" * 50)
    print(f"=== 全 {len(all_samples)} 問 評価 (Gemma 4 12B-it, zero-shot) ===")
    print("=" * 50)
    all_results, _ = bench.run(all_samples, start_idx=1)

    final_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in all_results:
        cat = r["category"]
        final_stats[cat]["total"] += 1
        if r.get("correct", False):
            final_stats[cat]["correct"] += 1
    final_total = len(all_results)
    final_correct = sum(1 for r in all_results if r.get("correct", False))
    final_acc = (final_correct / final_total) if final_total else 0.0
    final_cat_acc = {c: {"accuracy": s["correct"]/s["total"],
                         "correct_samples": s["correct"], "total_samples": s["total"]}
                     for c, s in final_stats.items() if s["total"] > 0}

    final_evals = {
        "accuracy": final_acc, "total_samples": final_total,
        "correct_samples": final_correct, "category_accuracies": final_cat_acc,
        "results": all_results,
    }

    print("\n" + "=" * 50)
    print("=== 最終結果 (Gemma 4 12B-it zero-shot) ===")
    print(f"全体精度: {final_acc*100:.2f}%  ({final_correct}/{final_total})")
    print_category_report(final_cat_acc)
    print_subcategory_report(all_results)
    print("=" * 50)

    base = f"{Path(args.jsonl).stem}_gemma4"
    save_results(final_evals, Path(args.out_dir), base)

if __name__ == "__main__":
    main()
