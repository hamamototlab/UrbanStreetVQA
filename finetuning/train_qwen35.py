"""
train_qwen35.py — Component-wise fine-tuning of Qwen3.5-VL in the ACCV 2026 paper, conditions (a)(b)(c)

学習範囲を段階的に拡大し、人物視点推論の改善が学習範囲に依存するかを調べる。
VGGT/4D-Humans 融合は一切使わない（素の Qwen3.5-VL のみ）。

3条件:
  (a) projector_only      : merger のみ学習。LLM 凍結（LoRAなし）、ViT 凍結
  (b) projector_lora      : merger + LLM LoRA(r=16,α=32)。ViT 凍結
  (c) projector_lora_vit  : merger + LLM LoRA + ViT 凍結解除（lr を 1/10）

累積設計: merger は全条件で学習。範囲を (a)→(b)→(c) と広げる。
公平性のため学習データ・epoch・batch・grad_accum・seed・lr schedule は3条件で固定。
変数は「学習範囲」のみ。

保存: 各条件で学習後 LoRA をマージし、完全な単独モデルとして保存する
      （merger / ViT はフル微調整のため、この方式なら欠落しない）。

Usage:
  python finetuning/train_qwen35.py --config finetuning/config.yaml \
      --tune_mode projector_only        # or: projector_lora | projector_lora_vit
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import (
    AutoProcessor,
    Qwen3_5ForConditionalGeneration,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType

# PEFT の torchao バージョン競合を回避
try:
    import peft.tuners.lora.torchao as _peft_torchao
    _peft_torchao.is_torchao_available = lambda: False
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).parent))
from data import PlainVQADataset, plain_collate_fn

VALID_MODES = ("projector_only", "projector_lora", "projector_lora_vit")

# 確立済みベースライン（LLaMA-Factory版 output_lora_v3_qwen35）と同一の全モジュールLoRA。
# Qwen3.5-VL LLM は 32層中24層が linear_attn(Mamba) 層で、ここがモデル容量の大半。
# 標準Transformer8層の self_attn(q/k/v/o) と全32層の MLP に加え、
# Mamba層の in_proj_*/out_proj も対象にしてLLM全体を低ランク適応する。
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
                "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]


# ─────────────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_merger(name: str) -> bool:
    return ".visual.merger." in name


def is_vit(name: str) -> bool:
    """visual encoder（merger を除く）。patch_embed / pos_embed / blocks。"""
    return ".visual." in name and ".visual.merger." not in name


def is_lora(name: str) -> bool:
    return "lora_" in name


# ─────────────────────────────────────────────────────────────────
# 学習対象パラメータのログ出力（条件設定ミス防止）
# ─────────────────────────────────────────────────────────────────
def log_trainable(model, tune_mode: str) -> None:
    groups = defaultdict(lambda: {"params": 0, "count": 0})
    total = 0
    trainable = 0
    for name, p in model.named_parameters():
        total += p.numel()
        if not p.requires_grad:
            continue
        trainable += p.numel()
        if is_lora(name):
            key = "LoRA (LLM)"
        elif is_merger(name):
            key = "merger (Projector)"
        elif is_vit(name):
            key = "ViT encoder"
        else:
            key = "OTHER(!)"
        groups[key]["params"] += p.numel()
        groups[key]["count"] += 1

    print("\n" + "=" * 70)
    print(f"  学習対象パラメータ  [tune_mode={tune_mode}]")
    print("=" * 70)
    for key in sorted(groups.keys()):
        g = groups[key]
        print(f"  {key:22s}: {g['params']:>14,} params  ({g['count']} tensors)")
    print("-" * 70)
    print(f"  {'TOTAL trainable':22s}: {trainable:>14,} params "
          f"({trainable/total*100:.3f}% of {total:,})")
    print("=" * 70)

    if "OTHER(!)" in groups:
        print("  ⚠️ 想定外の学習対象パラメータが存在します。設定を確認してください。")
        # 想定外パラメータ名を一部表示
        shown = 0
        for name, p in model.named_parameters():
            if p.requires_grad and not (is_lora(name) or is_merger(name) or is_vit(name)):
                print(f"      {name}")
                shown += 1
                if shown >= 10:
                    print("      …")
                    break

    # モード別の整合性チェック
    has_lora = "LoRA (LLM)" in groups
    has_vit = "ViT encoder" in groups
    has_merger = "merger (Projector)" in groups
    print("  [整合性チェック]")
    print(f"    merger 学習: {has_merger}  (全条件 True 期待)")
    print(f"    LoRA  学習: {has_lora}  ((b)(c) True 期待)")
    print(f"    ViT   学習: {has_vit}  ((c) のみ True 期待)")
    if tune_mode == "projector_only":
        assert has_merger and not has_lora and not has_vit, "projector_only 設定不整合"
    elif tune_mode == "projector_lora":
        assert has_merger and has_lora and not has_vit, "projector_lora 設定不整合"
    elif tune_mode == "projector_lora_vit":
        assert has_merger and has_lora and has_vit, "projector_lora_vit 設定不整合"
    print("    ✓ 設定整合性 OK")
    print("=" * 70 + "\n", flush=True)


# ─────────────────────────────────────────────────────────────────
# eval_loss 最小（best）の学習対象パラメータのみを保存するコールバック
#   - フルチェックポイント（18GB×複数）を残さずディスク節約
#   - 学習対象パラメータ（merger / LoRA / ViT）を名前指定で保存するため
#     PEFT 標準保存の merger/ViT 欠落バグを回避
#   - 凍結パラメータ（LLM本体）は学習中変化しないので base のまま使える
# ─────────────────────────────────────────────────────────────────
class BestStateCallback(TrainerCallback):
    def __init__(self, out_dir: Path, best_min_step: int = 0, save_all_ckpts: bool = False):
        self.best = float("inf")
        self.best_step = -1
        self.path = Path(out_dir) / "best_trainable_state.pt"
        # best_min_step: このステップ未満の checkpoint は best 選択対象外にする。
        # ViT解凍(c)条件で、学習データ縮小により early-overfit の偽の eval_loss 最小が
        # 前半に出て、人物視点が未発達な不安定 checkpoint を拾う問題への対処。
        # 条件(c)の best が後半(epoch~1.8)だったのに合わせ、後半レジームで選択する。
        self.best_min_step = int(best_min_step)
        # save_all_ckpts: 全evalステップの学習対象stateを ckpts/step{N}.pt に保存する。
        # 任意checkpointを後から評価できるようにする（checkpoint感度分析用）。
        self.save_all_ckpts = bool(save_all_ckpts)
        self.ckpt_dir = Path(out_dir) / "ckpts"
        if self.save_all_ckpts:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_log = []  # (step, eval_loss, path)

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        if metrics is None or model is None:
            return
        el = metrics.get("eval_loss")
        step = int(state.global_step)
        trainable = None
        # --- 全checkpoint保存 ---
        if self.save_all_ckpts:
            trainable = {n: p.detach().cpu().clone()
                         for n, p in model.named_parameters() if p.requires_grad}
            cp = self.ckpt_dir / f"step{step}.pt"
            torch.save(trainable, cp)
            self.ckpt_log.append((step, float(el) if el is not None else None, str(cp)))
            print(f"  [ckpt保存] step{step} eval_loss={el} → {cp.name}", flush=True)
        # --- best選択（best_min_step以降のみ） ---
        if step < self.best_min_step:
            print(f"  (best選択スキップ: step{step} < min_step{self.best_min_step}, "
                  f"eval_loss={el})", flush=True)
            return
        if el is None or el >= self.best:
            return
        self.best = float(el)
        self.best_step = step
        if trainable is None:
            trainable = {n: p.detach().cpu().clone()
                         for n, p in model.named_parameters() if p.requires_grad}
        torch.save(trainable, self.path)
        print(f"  ★ best eval_loss={el:.5f} @ step{step} "
              f"→ 学習対象 {len(trainable)} tensors を保存 ({self.path.name})",
              flush=True)


# ─────────────────────────────────────────────────────────────────
# ViT に小さい学習率を割り当てる Trainer
# ─────────────────────────────────────────────────────────────────
class StagedTrainer(Trainer):
    def __init__(self, *args, vit_lr_ratio: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.vit_lr_ratio = vit_lr_ratio

    def create_optimizer(self):
        base_lr = self.args.learning_rate
        vit_params, other_params = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if is_vit(name):
                vit_params.append(p)
            else:
                other_params.append(p)

        param_groups = [{"params": other_params, "lr": base_lr}]
        if vit_params:
            param_groups.append({
                "params": vit_params,
                "lr": base_lr * self.vit_lr_ratio,
                "name": "vit_lowlr",
            })

        self.optimizer = torch.optim.AdamW(
            param_groups,
            betas=(0.9, 0.95),
            weight_decay=self.args.weight_decay,
        )
        msg = f"  AdamW: base_lr={base_lr:.2e}"
        if vit_params:
            msg += f"  vit_lr={base_lr * self.vit_lr_ratio:.2e} (×{self.vit_lr_ratio})"
        print(msg, flush=True)
        return self.optimizer


# ─────────────────────────────────────────────────────────────────
def configure_requires_grad(model, tune_mode: str):
    """tune_mode に応じて requires_grad を設定。
    LoRA は (b)(c) で get_peft_model 済みの前提。
    """
    for name, p in model.named_parameters():
        if is_lora(name):
            p.requires_grad_(True)          # LoRA は常に学習
        elif is_merger(name):
            p.requires_grad_(True)          # merger は全条件で学習
        elif is_vit(name):
            p.requires_grad_(tune_mode == "projector_lora_vit")
        else:
            p.requires_grad_(False)         # LLM 本体は常に凍結（学習は LoRA 経由）


# ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tune_mode", required=True, choices=VALID_MODES)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    mcfg, dcfg, tcfg = cfg["model"], cfg["data"], cfg["train"]

    tune_mode = args.tune_mode
    seed = int(tcfg.get("seed", 42))
    set_seed(seed)

    # 出力先は tune_mode ごと
    out_dir = Path(tcfg["output_dir_base"] + "_" + {
        "projector_only": "proj_only",
        "projector_lora": "proj_lora",
        "projector_lora_vit": "proj_lora_vit",
    }[tune_mode])
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== 段階的チューニング学習: {tune_mode} ===")
    print(f"Output: {out_dir}")

    # ── モデル ──
    qwen_path = mcfg["qwen_path"]
    processor = AutoProcessor.from_pretrained(
        qwen_path, trust_remote_code=True,
        max_pixels=int(mcfg.get("max_pixels", 1280 * 28 * 28)),
        min_pixels=int(mcfg.get("min_pixels", 256 * 28 * 28)),
    )
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        qwen_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda:0")

    # まず全凍結
    for p in model.parameters():
        p.requires_grad_(False)

    # (b)(c): LoRA を付与
    if tune_mode in ("projector_lora", "projector_lora_vit"):
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(mcfg.get("lora_rank", 16)),
            lora_alpha=int(mcfg.get("lora_alpha", 32)),
            lora_dropout=float(mcfg.get("lora_dropout", 0.05)),
            target_modules=LORA_TARGETS,
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)

    # requires_grad をモード別に設定
    configure_requires_grad(model, tune_mode)

    # gradient checkpointing 用に入力埋め込みの grad を有効化
    model.enable_input_require_grads()
    if bool(tcfg.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()

    # 学習対象ログ + 整合性チェック
    log_trainable(model, tune_mode)

    # ── データ（3条件で完全に同一） ──
    import json as _json
    with open(dcfg["jsonl_path"]) as _f:
        _all = [_json.loads(l) for l in _f if l.strip()]
    _rng = torch.Generator().manual_seed(seed)
    _perm = torch.randperm(len(_all), generator=_rng).tolist()
    val_size = max(1, int(len(_perm) * float(dcfg.get("val_split", 0.1))))
    _val_idx = _perm[:val_size]
    _train_idx = _perm[val_size:]

    train_dataset = PlainVQADataset(
        jsonl_path=dcfg["jsonl_path"], image_dir=dcfg["image_dir"],
        processor=processor, max_len=int(dcfg.get("max_len", 4096)),
        is_train=True, sample_indices=_train_idx,
    )
    val_dataset = PlainVQADataset(
        jsonl_path=dcfg["jsonl_path"], image_dir=dcfg["image_dir"],
        processor=processor, max_len=int(dcfg.get("max_len", 4096)),
        is_train=False, sample_indices=_val_idx,
    )
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # ── TrainingArguments ──
    epochs = int(tcfg.get("num_epochs", 3))
    lr = float(tcfg.get("lr", 2e-4))
    max_steps = int(tcfg.get("max_steps", -1))

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=int(tcfg.get("batch_size", 1)),
        per_device_eval_batch_size=int(tcfg.get("batch_size", 1)),
        gradient_accumulation_steps=int(tcfg.get("grad_accum", 8)),
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=float(tcfg.get("warmup_ratio", 0.05)),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
        bf16=True,
        gradient_checkpointing=bool(tcfg.get("gradient_checkpointing", True)),
        eval_strategy="steps",
        eval_steps=int(tcfg.get("eval_steps", 50)),
        save_strategy="no",                 # フルモデルは末尾で手動保存
        logging_steps=int(tcfg.get("logging_steps", 10)),
        report_to="tensorboard",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        seed=seed,
    )

    print(f"epochs={epochs}  lr={lr}  max_steps={max_steps}  "
          f"grad_accum={tcfg.get('grad_accum', 8)}  seed={seed}")

    best_cb = BestStateCallback(out_dir, best_min_step=int(tcfg.get("best_min_step", 0)),
                                save_all_ckpts=bool(tcfg.get("save_all_ckpts", False)))
    trainer = StagedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=plain_collate_fn,
        vit_lr_ratio=float(tcfg.get("vit_lr_ratio", 0.1)),
        callbacks=[best_cb],
    )

    trainer.train()

    # ── best（eval_loss 最小）の学習対象パラメータをロード ──
    if best_cb.best_step >= 0 and best_cb.path.exists():
        print(f"\nbest モデルをロード: step{best_cb.best_step}, "
              f"eval_loss={best_cb.best:.5f} ({best_cb.path.name})")
        state = torch.load(best_cb.path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            print(f"  ⚠️ unexpected keys: {len(unexpected)}（先頭: {unexpected[:3]}）")
        loaded = sum(1 for n, p in model.named_parameters()
                     if p.requires_grad and n in state)
        print(f"  ✓ 学習対象 {loaded} tensors を best 重みで上書き")
    else:
        print("\n⚠️ best 状態が見つかりません。最終ステップの重みで保存します。")

    # ── フルモデル保存 ──
    final_dir = out_dir / "final_model"
    print(f"学習完了。フルモデルを保存します → {final_dir}")
    save_model = model
    if tune_mode in ("projector_lora", "projector_lora_vit"):
        print("  LoRA をマージしてフルモデル化…")
        save_model = model.merge_and_unload()   # merger/ViT 更新は in-place で保持

    save_model.save_pretrained(str(final_dir), safe_serialization=True)
    processor.save_pretrained(str(final_dir))
    print(f"  ✓ 保存完了: {final_dir}  (best step{best_cb.best_step}, "
          f"eval_loss={best_cb.best:.5f})")
    print(f"  評価: benchmark_qwen3_5.py --model-path {final_dir}")


if __name__ == "__main__":
    main()
