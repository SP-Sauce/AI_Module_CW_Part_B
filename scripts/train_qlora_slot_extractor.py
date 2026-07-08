"""Fine-tune the LLM slot extractor with LoRA/QLoRA.

This script is intentionally optional. The normal coursework demo does not
depend on GPU-only packages, but this provides a reproducible route for tuning
the JSON intent/slot extraction component when a CUDA environment is available.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_FILE = ROOT / "data" / "training" / "slot_instruction_examples.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "models" / "slot-extractor-qlora"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a LoRA/QLoRA adapter for JSON slot extraction.")
    parser.add_argument("--base-model", default="google/flan-t5-small", help="Seq2seq base model to adapt.")
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN_FILE, help="JSONL instruction examples.")
    parser.add_argument("--eval-file", type=Path, default=None, help="Optional JSONL evaluation examples for eval_loss.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for the adapter.")
    parser.add_argument("--metrics-output", type=Path, default=None, help="Optional JSON file for training metadata.")
    parser.add_argument("--max-steps", type=int, default=120, help="Training steps for the adapter.")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device training batch size.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-source-length", type=int, default=256)
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="q,v", help="Comma-separated LoRA target modules.")
    parser.add_argument("--no-4bit", action="store_false", dest="load_in_4bit", help="Use LoRA without 4-bit quantization.")
    parser.set_defaults(load_in_4bit=True)
    return parser


def slot_prompt(text: str) -> str:
    return (
        "You are the language-understanding component for a MultiWOZ restaurant assistant. "
        "Return only compact JSON with this schema: "
        '{"intent":"<intent>","slots":{...}}. '
        "Allowed intents: search, book, reschedule, cancel, greeting, thanks, alternative, list, "
        "correct, booking_info, booking_list, table_view, restaurant_info, filter_info, "
        "cuisine_help, dish_preference, distance_info, date_clarification, unsupported, unknown. "
        "Allowed slots: food, food_candidates, cuisine_group, dish, area, pricerange, day, "
        "relative_day, day_modifier, time, people, booking_reference. "
        "For broad regional cuisine phrases such as Middle Eastern, South Asian, East Asian, "
        "Southeast Asian, North African or West African, prefer cuisine_group plus food_candidates. "
        "For incomplete booking requests, return intent book with only the booking slots the user gave. "
        "A book or reserve command remains intent book when a restaurant name contains a dish or cuisine "
        "word such as curry; do not reinterpret the restaurant name as a dish preference. "
        f"User: {text}"
    )


def load_examples(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    examples: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(row.get("text") or row.get("input") or "").strip()
            if not text:
                raise ValueError(f"Missing text/input at {path}:{line_number}")
            if "output" in row:
                target = row["output"]
            else:
                target = {"intent": row.get("intent", "unknown"), "slots": row.get("slots", {})}
            examples.append({"prompt": slot_prompt(text), "target": json.dumps(target, ensure_ascii=True)})
    if not examples:
        raise ValueError(f"No training examples found in {path}")
    return examples


def import_training_deps() -> dict[str, Any]:
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForSeq2Seq,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit(
            "Missing optional QLoRA dependencies. Install them with: "
            "pip install -r requirements-qlora.txt"
        ) from exc
    return locals()


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    deps = import_training_deps()
    torch = deps["torch"]
    Dataset = deps["Dataset"]
    LoraConfig = deps["LoraConfig"]
    TaskType = deps["TaskType"]
    get_peft_model = deps["get_peft_model"]
    prepare_model_for_kbit_training = deps["prepare_model_for_kbit_training"]
    AutoModelForSeq2SeqLM = deps["AutoModelForSeq2SeqLM"]
    AutoTokenizer = deps["AutoTokenizer"]
    BitsAndBytesConfig = deps["BitsAndBytesConfig"]
    DataCollatorForSeq2Seq = deps["DataCollatorForSeq2Seq"]
    Seq2SeqTrainer = deps["Seq2SeqTrainer"]
    Seq2SeqTrainingArguments = deps["Seq2SeqTrainingArguments"]

    if args.load_in_4bit and not torch.cuda.is_available():
        raise SystemExit(
            "QLoRA 4-bit training requires a CUDA environment. "
            "Use WSL/Colab/GPU Linux, or rerun with --no-4bit for a small LoRA smoke test."
        )

    examples = load_examples(args.train_file)
    eval_examples = load_examples(args.eval_file) if args.eval_file else None
    dataset = Dataset.from_list(examples)
    eval_dataset = Dataset.from_list(eval_examples) if eval_examples else None
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    bf16_supported = bool(
        torch.cuda.is_available()
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )
    qlora_compute_dtype = torch.bfloat16 if bf16_supported else torch.float32
    quantization_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=qlora_compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        if args.load_in_4bit
        else None
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.base_model,
        quantization_config=quantization_config,
        device_map="auto" if args.load_in_4bit else None,
    )
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    target_modules = [item.strip() for item in args.target_modules.split(",") if item.strip()]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_config)

    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        inputs = tokenizer(batch["prompt"], max_length=args.max_source_length, truncation=True)
        labels = tokenizer(text_target=batch["target"], max_length=args.max_target_length, truncation=True)
        inputs["labels"] = labels["input_ids"]
        return inputs

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    tokenized_eval = (
        eval_dataset.map(tokenize, batched=True, remove_columns=eval_dataset.column_names)
        if eval_dataset is not None
        else None
    )
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        logging_steps=10,
        save_strategy="no",
        fp16=False,
        bf16=bf16_supported,
        logging_nan_inf_filter=False,
        report_to="none",
    )
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        eval_dataset=tokenized_eval,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    train_result = trainer.train()
    train_loss = float(train_result.metrics.get("train_loss", math.nan))
    if not math.isfinite(train_loss) or train_loss <= 0:
        raise SystemExit(
            f"Training produced an invalid train_loss ({train_loss}); no adapter was saved. "
            "Try a fresh runtime and verify the CUDA/package setup."
        )
    eval_metrics: dict[str, Any] = {}
    if tokenized_eval is not None:
        eval_metrics = trainer.evaluate()
        if "eval_loss" in eval_metrics:
            print(f"eval_loss: {eval_metrics['eval_loss']}")
            eval_loss = float(eval_metrics["eval_loss"])
            if not math.isfinite(eval_loss):
                raise SystemExit(
                    f"Evaluation produced an invalid eval_loss ({eval_loss}); no adapter was saved."
                )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    if args.metrics_output:
        metrics = {
            "base_model": args.base_model,
            "train_file": str(args.train_file),
            "eval_file": str(args.eval_file) if args.eval_file else None,
            "output_dir": str(args.output_dir),
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "load_in_4bit": args.load_in_4bit,
            "compute_dtype": str(qlora_compute_dtype),
            "bf16_enabled": bf16_supported,
            "cuda_available": torch.cuda.is_available(),
            "train_metrics": train_result.metrics,
            "eval_metrics": eval_metrics,
        }
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
        print(f"Saved training metadata to {args.metrics_output}")
    print(f"Saved LoRA adapter to {args.output_dir}")
    print("Use it for extraction with:")
    print(f"  python scripts/run_chat.py --enable-llm --slot-model-name {args.output_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])
