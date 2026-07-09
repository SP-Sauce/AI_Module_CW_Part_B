"""Train a strict JSON LoRA slot extractor for Kaggle GPU notebooks."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from restaurant_assistant.slot_extraction import adapter_slot_prompt, strict_parse_llm_json_output


DEFAULT_TRAIN_FILE = ROOT / "data" / "training" / "slot_train_strict.jsonl"
DEFAULT_DEV_FILE = ROOT / "data" / "training" / "slot_dev_strict.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "models" / "slot-extractor-lora-strict"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a strict JSON LoRA adapter for slot extraction.")
    parser.add_argument("--base-model", default="google/flan-t5-base")
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--dev-file", type=Path, default=DEFAULT_DEV_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-source-length", type=int, default=256)
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=6062026)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="q,v")
    parser.add_argument("--load-in-4bit", action="store_true", help="Use optional bitsandbytes 4-bit loading.")
    parser.add_argument("--generation-max-new-tokens", type=int, default=96)
    parser.add_argument("--generation-num-beams", type=int, default=1)
    return parser


def load_jsonl_examples(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    examples: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            user_input = str(row.get("input") or "").strip()
            target = str(row.get("target") or "").strip()
            if not user_input or not target:
                raise ValueError(f"Missing input/target at {path}:{line_number}")
            parsed = strict_parse_llm_json_output(target)
            canonical = json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            if target != canonical:
                raise ValueError(f"Target is not canonical strict JSON at {path}:{line_number}")
            examples.append({"input": user_input, "prompt": adapter_slot_prompt(user_input), "target": target})
    if not examples:
        raise ValueError(f"No examples found in {path}")
    return examples


def _slot_pairs(slots: dict[str, Any]) -> set[tuple[str, Any]]:
    pairs = set()
    for key, value in slots.items():
        if isinstance(value, list):
            value = tuple(value)
        pairs.add((key, value))
    return pairs


def strict_generation_metrics(predictions: list[str], targets: list[str]) -> dict[str, Any]:
    parse_success = 0
    raw_parse_errors = 0
    intent_correct = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    cases: list[dict[str, Any]] = []
    for prediction, target_text in zip(predictions, targets):
        expected = strict_parse_llm_json_output(target_text)
        parsed_prediction: dict[str, Any] | None = None
        parse_error: str | None = None
        try:
            parsed_prediction = strict_parse_llm_json_output(prediction)
            parse_success += 1
        except ValueError as exc:
            raw_parse_errors += 1
            parse_error = str(exc)

        if parsed_prediction is not None:
            intent_correct += int(parsed_prediction.get("intent") == expected.get("intent"))
            predicted_pairs = _slot_pairs(parsed_prediction.get("slots") or {})
        else:
            predicted_pairs = set()
        expected_pairs = _slot_pairs(expected.get("slots") or {})
        true_positive += len(predicted_pairs & expected_pairs)
        false_positive += len(predicted_pairs - expected_pairs)
        false_negative += len(expected_pairs - predicted_pairs)
        cases.append(
            {
                "target": expected,
                "prediction": parsed_prediction,
                "raw_prediction": prediction,
                "parse_error": parse_error,
            }
        )

    total = max(len(targets), 1)
    precision = round(true_positive / (true_positive + false_positive), 4) if true_positive + false_positive else 0.0
    recall = round(true_positive / (true_positive + false_negative), 4) if true_positive + false_negative else 0.0
    f1 = round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0
    return {
        "strict_json_parse_success_rate": round(parse_success / total, 4),
        "raw_parse_error_count": raw_parse_errors,
        "strict_intent_accuracy": round(intent_correct / total, 4),
        "strict_slot_precision": precision,
        "strict_slot_recall": recall,
        "strict_slot_f1": f1,
        "evaluated_cases": len(targets),
        "cases": cases,
    }


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
            set_seed,
        )
    except ImportError as exc:
        raise SystemExit(
            "Missing Kaggle training dependencies. Install with: "
            "pip install -U transformers datasets accelerate peft sentencepiece scikit-learn evaluate"
        ) from exc
    return locals()


def gpu_info(torch_module: Any) -> dict[str, Any]:
    info = {
        "cuda_available": bool(torch_module.cuda.is_available()),
        "device_count": int(torch_module.cuda.device_count()) if torch_module.cuda.is_available() else 0,
        "devices": [],
        "python": sys.version,
        "platform": platform.platform(),
    }
    if torch_module.cuda.is_available():
        for index in range(torch_module.cuda.device_count()):
            props = torch_module.cuda.get_device_properties(index)
            info["devices"].append(
                {
                    "index": index,
                    "name": torch_module.cuda.get_device_name(index),
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                }
            )
    return info


def generate_predictions(
    *,
    model: Any,
    tokenizer: Any,
    examples: list[dict[str, str]],
    max_new_tokens: int,
    num_beams: int,
) -> list[str]:
    predictions: list[str] = []
    try:
        model_device = model.get_input_embeddings().weight.device
    except AttributeError:
        model_device = None
    for example in examples:
        inputs = tokenizer(example["prompt"], return_tensors="pt", truncation=True, max_length=256)
        if model_device is not None:
            inputs = {key: value.to(model_device) for key, value in inputs.items()}
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=num_beams,
            early_stopping=True,
        )
        predictions.append(tokenizer.decode(generated[0], skip_special_tokens=True).strip())
    return predictions


def main(argv: list[str] | None = None) -> int:
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
    set_seed = deps["set_seed"]

    set_seed(args.seed)
    train_examples = load_jsonl_examples(args.train_file)
    dev_examples = load_jsonl_examples(args.dev_file)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    bf16_supported = bool(
        torch.cuda.is_available()
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )
    fp16_enabled = bool(torch.cuda.is_available() and not bf16_supported)
    quantization_config = None
    if args.load_in_4bit:
        if not torch.cuda.is_available():
            raise SystemExit("--load-in-4bit requires CUDA.")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16_supported else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.base_model,
        quantization_config=quantization_config,
        device_map="auto" if args.load_in_4bit else None,
    )
    model.config.use_cache = False
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    target_modules = [module.strip() for module in args.target_modules.split(",") if module.strip()]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_config)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    train_dataset = Dataset.from_list(train_examples)
    dev_dataset = Dataset.from_list(dev_examples)

    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        inputs = tokenizer(batch["prompt"], max_length=args.max_source_length, truncation=True)
        labels = tokenizer(text_target=batch["target"], max_length=args.max_target_length, truncation=True)
        inputs["labels"] = labels["input_ids"]
        return inputs

    tokenized_train = train_dataset.map(tokenize, batched=True, remove_columns=train_dataset.column_names)
    tokenized_dev = dev_dataset.map(tokenize, batched=True, remove_columns=dev_dataset.column_names)

    training_kwargs = {
        "output_dir": str(args.output_dir),
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "max_steps": args.max_steps,
        "logging_steps": 25,
        "eval_steps": args.eval_steps,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "predict_with_generate": False,
        "fp16": fp16_enabled,
        "bf16": bf16_supported,
        "report_to": "none",
        "seed": args.seed,
    }
    signature_parameters = inspect.signature(Seq2SeqTrainingArguments).parameters
    if "eval_strategy" in signature_parameters:
        training_kwargs["eval_strategy"] = "steps"
    else:
        training_kwargs["evaluation_strategy"] = "steps"
    training_args = Seq2SeqTrainingArguments(**training_kwargs)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_dev,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    train_result = trainer.train()
    eval_metrics = trainer.evaluate()
    raw_predictions = generate_predictions(
        model=trainer.model,
        tokenizer=tokenizer,
        examples=dev_examples,
        max_new_tokens=args.generation_max_new_tokens,
        num_beams=args.generation_num_beams,
    )
    strict_metrics = strict_generation_metrics(raw_predictions, [example["target"] for example in dev_examples])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    metadata = {
        "base_model": args.base_model,
        "train_file": str(args.train_file),
        "dev_file": str(args.dev_file),
        "num_train_examples": len(train_examples),
        "num_dev_examples": len(dev_examples),
        "seed": args.seed,
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": target_modules,
            "load_in_4bit": args.load_in_4bit,
        },
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "train_loss": train_result.metrics.get("train_loss"),
        "eval_loss": eval_metrics.get("eval_loss"),
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "strict_dev_metrics": {key: value for key, value in strict_metrics.items() if key != "cases"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cuda": gpu_info(torch),
    }
    metadata_path = args.output_dir / "training_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print(json.dumps(metadata, indent=2, default=str))
    print(f"Saved strict LoRA adapter to {args.output_dir}")
    print(f"Saved training metadata to {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
