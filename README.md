# PII Memorization Pipeline — DLLMS

A research pipeline for **detecting and token-level masking of PII** in the [`LLM-PBE/enron-email`](https://huggingface.co/datasets/LLM-PBE/enron-email) dataset. Built to support studies on PII memorization in large language models.

---

## Overview

The pipeline combines:

| Component | Role |
|---|---|
| [Microsoft Presidio](https://github.com/microsoft/presidio) + spaCy `en_core_web_lg` | Character-level PII entity detection |
| [Qwen/Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B) tokenizer | Subword tokenization with character offset mapping |
| PyTorch | Boolean mask tensor construction |

**Per-example output:**

```jsonc
{
  "example_index": 0,
  "original_text": "Contact Chris at chris@lacimagroup.com ...",
  "token_ids": [151644, 831, 520, ...],
  "offset_mapping": [[0,0], [0,7], [8,13], ...],
  "pii_spans": [
    {"entity_type": "EMAIL_ADDRESS", "start": 17, "end": 38,
     "score": 1.0, "text": "chris@lacimagroup.com"}
  ],
  "target_entity": "EMAIL_ADDRESS",
  "mask": [0, 0, 0, ..., 1, 1, 1, ..., 0]
}
```

`mask[i] = 1` if and only if token `i` overlaps a PII span of the target entity type.

---

## Repo Structure

```
.
├── pipeline.py                  # Original PII scanner (Presidio scan → JSONL hits)
├── pii_masking_pipeline.py      # Full PII detection + token-level masking pipeline
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone <your-repo-url>
cd PII-Memorization-Pipeline-DLLMS
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the spaCy model

```bash
python -m spacy download en_core_web_lg
```

---

## Usage

### PII detection + token masking (`pii_masking_pipeline.py`)

```bash
python pii_masking_pipeline.py \
  --split train \
  --target-entity EMAIL_ADDRESS \
  --max-examples 1000 \
  --batch-size 32 \
  --output pii_masked.jsonl
```

#### All CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `LLM-PBE/enron-email` | HuggingFace dataset identifier |
| `--config` | `None` | Dataset config name |
| `--split` | `train` | Dataset split |
| `--streaming` | `False` | Enable streaming (avoids full download) |
| `--max-examples` | `0` (all) | Cap on examples to process |
| `--text-field` | `text` | Column name containing raw text |
| `--target-entity` | `EMAIL_ADDRESS` | PII entity type for the boolean mask |
| `--entities` | `` (all) | Comma-separated Presidio entity whitelist |
| `--score-threshold` | `0.5` | Minimum Presidio confidence score |
| `--spacy-model` | `en_core_web_lg` | spaCy model name |
| `--language` | `en` | Language code for Presidio |
| `--tokenizer` | `Qwen/Qwen2.5-7B` | HuggingFace tokenizer |
| `--max-length` | `512` | Truncation length in tokens |
| `--batch-size` | `32` | Examples per flush to disk |
| `--output` | `pii_masked.jsonl` | Output JSONL file path |

#### Supported `--target-entity` values (Presidio built-ins)

`EMAIL_ADDRESS`, `PHONE_NUMBER`, `PERSON`, `LOCATION`, `ORGANIZATION`,
`DATE_TIME`, `URL`, `CREDIT_CARD`, `US_SSN`, `US_PASSPORT`, `IBAN_CODE`, `IP_ADDRESS`, `CRYPTO`, …

---

### PII scanner only (`pipeline.py`)

Scans a dataset and writes all detected PII hits without tokenization:

```bash
python pipeline.py \
  --dataset LLM-PBE/enron-email \
  --split train \
  --text-fields text \
  --max-examples 5000 \
  --output hits.jsonl \
  --summary summary.json
```

---

## Output Format

Each line of the output JSONL file is one example:

| Field | Type | Description |
|---|---|---|
| `example_index` | `int` | Position in the dataset |
| `original_text` | `str` | Raw email text |
| `token_ids` | `List[int]` | Token IDs from Qwen2.5-7B tokenizer |
| `offset_mapping` | `List[[int,int]]` | Character `[start, end]` per token |
| `pii_spans` | `List[dict]` | All detected PII spans (all entity types) |
| `target_entity` | `str` | Entity type the mask was built for |
| `mask` | `List[int]` | `1` = token overlaps target PII, `0` = clean |

---

## How the Mask is Built

1. **Detect** — Presidio returns `(entity_type, char_start, char_end)` for every PII span in the text.
2. **Tokenize** — the Qwen tokenizer returns `input_ids` and `offset_mapping` (character offsets per token), with truncation applied at `--max-length`.
3. **Overlap test** — for each token at position `i` with character span `[tok_s, tok_e)`:

$$\text{mask}[i] = \begin{cases} 1 & \exists\, \text{span} \in \text{target\_spans} : tok\_s < \text{span.end} \;\wedge\; tok\_e > \text{span.start} \\ 0 & \text{otherwise} \end{cases}$$

4. Special tokens (offset `(0, 0)`) are always set to `0`.

---

## Requirements

- Python ≥ 3.10
- See [`requirements.txt`](requirements.txt)
- HuggingFace account / token may be needed to access `Qwen/Qwen2.5-7B` tokenizer files on first run
