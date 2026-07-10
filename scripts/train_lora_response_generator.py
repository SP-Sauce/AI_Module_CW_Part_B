"""Train an optional LoRA response generator for grounded restaurant replies."""

from __future__ import annotations

import argparse
import inspect
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from restaurant_assistant.llm_generator import DEFAULT_RESPONSE_MODEL, validate_generated_response


DEFAULT_TRAIN_FILE = ROOT / "data" / "training" / "response_generation_examples.jsonl"
DEFAULT_EVAL_FILE = ROOT / "data" / "evaluation" / "response_generation_eval.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "models" / "response-generator-lora"
DEFAULT_METADATA_PATH = ROOT / "outputs" / "evaluation" / "response_lora_training_metadata.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a guarded response-generation LoRA adapter.")
    parser.add_argument("--base-model", default=DEFAULT_RESPONSE_MODEL)
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--max-source-length", type=int, default=384)
    parser.add_argument("--max-target-length", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=6062026)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="q,v")
    parser.add_argument("--load-in-4bit", action="store_true", help="Use optional bitsandbytes 4-bit loading.")
    parser.add_argument("--generation-max-new-tokens", type=int, default=120)
    return parser


def format_prompt(row: dict[str, str]) -> str:
    return f"{row['instruction']}\n{row['input']}"


def evidence_records_from_input(input_text: str) -> list[dict[str, str]]:
    for line in input_text.splitlines():
        if not line.startswith("Evidence:"):
            continue
        raw = line.split(":", 1)[1].strip()
        records: list[dict[str, str]] = []
        for chunk in raw.split("|"):
            record: dict[str, str] = {}
            for part in chunk.split(";"):
                if "=" not in part:
                    continue
                key, value = part.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key and value:
                    record[key] = value
            if record:
                records.append(record)
        return records
    return []


def load_jsonl_examples(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    examples: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            instruction = str(row.get("instruction") or "").strip()
            input_text = str(row.get("input") or "").strip()
            output = str(row.get("output") or "").strip()
            if not instruction or not input_text or not output:
                raise ValueError(f"Missing instruction/input/output at {path}:{line_number}")
            evidence_records = evidence_records_from_input(input_text)
            validation = validate_generated_response(
                output,
                evidence_records=evidence_records,
                known_restaurant_records=evidence_records,
            )
            if not validation.ok:
                raise ValueError(f"Unsafe output at {path}:{line_number}: {validation.reason}")
            examples.append(
                {
                    "instruction": instruction,
                    "input": input_text,
                    "prompt": format_prompt(
                        {
                            "instruction": instruction,
                            "input": input_text,
                        }
                    ),
                    "target": output,
                }
            )
    if not examples:
        raise ValueError(f"No examples found in {path}")
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
            set_seed,
        )
    except ImportError as exc:
        raise SystemExit(
            "Missing training dependencies. Install with: "
            "pip install -U -r requirements-qlora.txt"
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
    eval_examples = load_jsonl_examples(args.eval_file)
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
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)

    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        inputs = tokenizer(
            batch["prompt"],
            max_length=args.max_source_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["target"],
            max_length=args.max_target_length,
            truncation=True,
        )
        inputs["labels"] = labels["input_ids"]
        return inputs

    train_dataset = Dataset.from_list(train_examples).map(tokenize, batched=True, remove_columns=list(train_examples[0]))
    eval_dataset = Dataset.from_list(eval_examples).map(tokenize, batched=True, remove_columns=list(eval_examples[0]))
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    args_kwargs = {
        "output_dir": str(args.output_dir),
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_steps": args.max_steps,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "logging_steps": max(1, min(args.eval_steps, 25)),
        "save_total_limit": 2,
        "predict_with_generate": True,
        "generation_max_length": args.max_target_length,
        "fp16": fp16_enabled,
        "bf16": bf16_supported,
        "report_to": [],
        "seed": args.seed,
    }
    signature = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    if "eval_strategy" in signature:
        args_kwargs["eval_strategy"] = "steps"
    else:
        args_kwargs["evaluation_strategy"] = "steps"

    training_args = Seq2SeqTrainingArguments(**args_kwargs)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    train_result = trainer.train()
    eval_metrics = trainer.evaluate()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_model": args.base_model,
        "output_dir": str(args.output_dir),
        "train_file": str(args.train_file),
        "eval_file": str(args.eval_file),
        "train_examples": len(train_examples),
        "eval_examples": len(eval_examples),
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "load_in_4bit": args.load_in_4bit,
        "gpu_info": gpu_info(torch),
        "train_metrics": getattr(train_result, "metrics", {}),
        "eval_metrics": eval_metrics,
    }
    args.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print(f"Saved response adapter to {args.output_dir}")
    print(f"Saved training metadata to {args.metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
