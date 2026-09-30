#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_qwen35.py (v2.1, プロンプト改善版 + 全MCQA対応サーキュラー評価版)
---------------------------------------------------
Qwen3.5対応版（Qwen3-VLからの変更点）:
- Qwen3_5ForConditionalGeneration を使用
- apply_chat_template / generateでenable_thinking=Falseを指定
- max_new_tokensを64に増加（thinking modeオフでも安全な長さ）
- processor(text=prompt, images=[image])で画像を渡す
- qwen_vl_utils不要
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
from transformers import Qwen3_5ForConditionalGeneration, AutoTokenizer, AutoProcessor
from peft import PeftModel

torch.set_grad_enabled(False)

# ====== ユーティリティ ======

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

# ====== メイン評価クラス ======

class StreetVQABenchmarkerJSONLQwen:

    # カウントカテゴリのみ通常評価、それ以外はサーキュラー評価
    numeric_categories = {"count", "counting"}

    def __init__(self, jsonl_path: str, image_dir: str, model_path: str = "./Qwen3.5-9B", adapter_path: str = None):
        self.jsonl_path = Path(jsonl_path)
        self.image_dir = Path(image_dir)
        self.model_path = model_path
        self.adapter_path = adapter_path

        print(f"Loading model from local path: {model_path}...")
        if torch.cuda.is_available():
            self.device = "cuda"
            print(">>> GPU (CUDA) を使用します <<<")
        else:
            self.device = "cpu"
            print(">>> GPUが見つからないため CPU を使用します <<<")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True
            ).to(self.device)

            # ===== LoRAアダプターを読み込み =====
            if self.adapter_path:
                print(f"Loading LoRA adapter from: {self.adapter_path}")
                self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
                self.model = self.model.merge_and_unload()
                print("✅ LoRA adapter loaded and merged successfully")
            else:
                print("⚠️  LoRAアダプターなし（ベースモデルのみで評価）")
            # ====================================

            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            print(f"Model loaded successfully to device: {self.model.device}")
        except Exception as e:
            print(f"[致命的エラー] モデルの読み込みに失敗: {e}")
            raise

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

    @torch.no_grad()
    def ask(self, question: str, image: Image.Image, category: str, choices: Optional[List[str]] = None) -> str:
        """
        Qwen3.5対応版：
        - enable_thinking=False でthinking modeをOFF
        - max_new_tokens=64 で十分な出力長を確保
        - processor(text=prompt, images=[image]) で画像を渡す
        """
        try:
            cat = (category or "").lower()

            # システムプロンプト
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

            # ユーザー質問（選択肢を含める）
            if choices:
                choices_text = "\n".join(choices)
                user_question = (f"Question: {question}\n\n"
                                 f"Options:\n{choices_text}\n\n"
                                 f"Choose the letter corresponding to your answer. Your output must be a single letter only.\n"
                                 f"Answer:")
            elif cat in self.numeric_categories:
                user_question = f"Question: {question}\nAnswer:"
            else:
                user_question = f"Question: {question}\nAnswer:"

            # メッセージ形式（画像はプレースホルダーのみ）
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_question}
                ]}
            ]

            # Step1: enable_thinking=False でprompt文字列を取得
            prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,   # thinking modeをOFF
            )

            # Step2: processor()でpromptと画像をエンコード
            inputs = self.processor(
                text=prompt,
                images=[image],
                return_tensors="pt"
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

            input_token_len = inputs["input_ids"].shape[1]

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=64,       # thinking modeオフでも十分な長さ
                do_sample=False,
                temperature=None,
                top_p=None,
            )
            generated_tokens = outputs[0, input_token_len:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            return response.strip() if response else "INVALID_ANSWER"
        except Exception as e:
            print(f"[エラー] Qwenモデル質問処理エラー: {e}")
            return "ERROR"

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

        # MCQの場合：より厳密な抽出
        # 1. 行頭または単独のA-D文字を優先的に抽出
        m = re.search(r'^([A-D])\b', text.upper())
        if m:
            return m.group(1)

        # 2. 単語境界のA-D
        m = re.search(r'\b([A-D])\b', text.upper())
        if m:
            return m.group(1)

        # 3. 最後の手段として任意のA-D
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

        # 数値比較
        if cat in self.numeric_categories:
            if gt_str.lower() == "11 or more":
                return pred_str.lower() == "11 or more" or (pred_str.isdigit() and int(pred_str) >= 11)
            return pred_str == gt_str

        # MCQ比較（記号一致 または テキスト一致）
        if pred_str.upper() == gt_str.upper():
            return True

        if pred_str.upper() in {"A", "B", "C", "D"} and options:
            idx = ord(pred_str.upper()) - 65
            if 0 <= idx < len(options):
                return str(options[idx]).strip() == gt_str
        return False

    def circular_eval(self, image_path: Path, question: str, category: str, options: List[str], answer: str):
        """
        サーキュラー評価: 選択肢を回転させて全パターンでテスト
        2択: A,B → B,A
        3択: A,B,C → B,C,A → C,A,B
        4択: A,B,C,D → B,C,D,A → C,D,A,B → D,A,B,C
        """
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

        if not img_path.exists():
            print(f"[{idx}/{total}] [スキップ] 画像が見つかりません: {img_path}")
            return {
                "index": idx, "image": str(img_path), "category": category, "subcategory": subcategory,
                "question": question, "ground_truth": answer, "prediction": "NOT_FOUND",
                "raw_answer": "", "correct": False, "options_full": " | ".join(options or [])
            }

        print(f"\n[{idx}/{total}] {img_path.name} | Cat={category} | Q={question}")
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
                print(f"  [スキップ] 画像読込失敗")
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

# ====== 結果保存 ======
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

# ====== main ======
def main():
    parser = argparse.ArgumentParser(description="StreetVQA Qwen3.5 Benchmark v2.1 (プロンプト改善版 + 全MCQA サーキュラー評価)")
    parser.add_argument("--annotations", "--jsonl", dest="jsonl", required=True,
                        help="input annotations jsonl")
    parser.add_argument("--image-dir", required=True, help="frame image directory")
    parser.add_argument("--output-dir", "--out-dir", dest="out_dir",
                        default="results_qwen35", help="output directory")
    parser.add_argument("--model-path", default="PATH/TO/Qwen3.5-VL", help="base model path")
    parser.add_argument("--adapter-path", default=None, help="LoRAアダプターのパス（オプション）")
    parser.add_argument("--limit", type=int, default=None, help="処理上限件数")
    parser.add_argument("--auto-yes", action="store_true", help="残りサンプルの実行確認を自動でyesにする")
    args = parser.parse_args()

    if not Path(args.jsonl).exists():
        print(f"[致命的エラー] JSONLが見つかりません: {args.jsonl}")
        return
    if not Path(args.image_dir).exists():
        print(f"[致命的エラー] 画像ディレクトリが見つかりません: {args.image_dir}")
        return

    bench = StreetVQABenchmarkerJSONLQwen(
        args.jsonl,
        args.image_dir,
        model_path=args.model_path,
        adapter_path=args.adapter_path
    )
    all_samples = bench.samples
    all_results: List[Dict[str, Any]] = []

    if not all_samples:
        print("評価するサンプルがありません。")
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
        print(f"[LIMIT] 処理上限 {args.limit} 件を適用しました。")

    if not test_samples and not full_remaining_samples:
        print("評価サンプルなし。終了。")
        return

    print("\n" + "=" * 50)
    print("=== STEP 1: カテゴリ別サンプルテスト ===")
    print("=" * 50)
    test_results, test_summary = bench.run(test_samples, start_idx=1)
    all_results.extend(test_results)

    print("\n" + "=" * 50)
    print("=== サンプルテスト結果サマリー ===")
    print(f"精度: {test_summary['accuracy']*100:.2f}%")
    print_category_report(test_summary["category_accuracies"])
    print("=" * 50)

    run_all = False
    if full_remaining_samples:
        if args.auto_yes:
            print(f"\n[auto-yes] 残り {len(full_remaining_samples)} 件を自動実行します。")
            run_all = True
        else:
            while True:
                try:
                    user_input = input(
                        f"\n残り {len(full_remaining_samples)} 件を実行しますか？ (yes/no): "
                    ).lower().strip()
                    if user_input == "yes":
                        run_all = True
                        break
                    elif user_input == "no":
                        break
                    else:
                        print("yes/no で入力してください。")
                except:
                    break
    else:
        print("残りサンプルなし。")

    if run_all:
        print("\n" + "=" * 50)
        print("=== STEP 2: 残り全てのサンプルを実行 ===")
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
    print("=== 最終結果 ===")
    print(f"精度: {final_evals['accuracy']*100:.2f}%")
    print_category_report(final_evals["category_accuracies"])
    print("=" * 50)

    base = f"{Path(args.jsonl).stem}_qwen35"
    save_results(final_evals, Path(args.out_dir), base)

if __name__ == "__main__":
    main()