# Kaggle Training: Strict Slot Extractor LoRA

Use this in a Kaggle Notebook with GPU enabled. The strict LoRA model trains only
the intent/slot extractor. The assistant still keeps validation, repair,
rule-based fallback, retrieval grounding and the ResponsePlan/NLG safety
boundary.

## 1. Check GPU

```python
!nvidia-smi
```

## 2. Install Dependencies

```python
!pip install -U pip
!pip install -r requirements.txt
!pip install -U transformers datasets accelerate peft sentencepiece scikit-learn evaluate
```

## 3. Build Strict Training Data

```python
!python scripts/build_slot_training_data.py
```

## 4. Smoke Test

```python
!python scripts/train_strict_slot_extractor_lora.py \
  --base-model google/flan-t5-base \
  --train-file data/training/slot_train_strict.jsonl \
  --dev-file data/training/slot_dev_strict.jsonl \
  --output-dir models/slot-extractor-lora-strict-smoke \
  --max-steps 50
```

## 5. Full Training

```python
!python scripts/train_strict_slot_extractor_lora.py \
  --base-model google/flan-t5-base \
  --train-file data/training/slot_train_strict.jsonl \
  --dev-file data/training/slot_dev_strict.jsonl \
  --output-dir models/slot-extractor-lora-strict \
  --max-steps 800
```

## 6. Evaluate Standard Set

```python
!mkdir -p reports
!python scripts/evaluate.py \
  --sample-data \
  --slot-fixture data/evaluation/slot_eval_cases.jsonl \
  --enable-llm \
  --slot-model-name models/slot-extractor-lora-strict \
  --report-path reports/evaluation_lora_strict_eval.json
```

## 7. Evaluate Challenge Set

```python
!python scripts/evaluate.py \
  --sample-data \
  --slot-fixture data/evaluation/slot_challenge_cases.jsonl \
  --enable-llm \
  --slot-model-name models/slot-extractor-lora-strict \
  --report-path reports/evaluation_lora_strict_challenge.json
```

## 8. Compare Reports

```python
!python scripts/compare_model_reports.py \
  --report strict_eval=reports/evaluation_lora_strict_eval.json \
  --report strict_challenge=reports/evaluation_lora_strict_challenge.json \
  --output reports/model_comparison.md
```

## 9. Zip Outputs

```python
!zip -r slot_extractor_lora_strict_outputs.zip models/slot-extractor-lora-strict reports
```

Download `slot_extractor_lora_strict_outputs.zip` from Kaggle, unzip it into the
repo root locally, and run:

```powershell
python scripts/run_web.py --enable-llm --slot-model-name models/slot-extractor-lora-strict --debug
```
