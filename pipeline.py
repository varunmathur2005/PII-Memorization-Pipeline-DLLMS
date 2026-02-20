#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PII scanner for Hugging Face SFT chat datasets using Presidio and spaCy.

Example:
python scan_pii_presidio_spacy.py \
  --dataset allenai/tulu-3-sft-mixture \
  --split train \
  --text-fields messages \
  --max-examples 5000 \
  --output hits.jsonl

Notes:
Assumes each example has either:
  1) messages: List[{"role": ..., "content": ...}, ...]
  2) or plain text fields like "text", "prompt", "response", etc.
"""

import argparse
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from datasets import load_dataset

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.pattern import Pattern


@dataclass
class Hit:
    entity_type: str
    start: int
    end: int
    score: float
    text: str


def build_analyzer(spacy_model: str = "en_core_web_lg") -> AnalyzerEngine:
    """
    Build Presidio AnalyzerEngine with a spaCy NLP engine.
    """
    provider = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": spacy_model}],
        }
    )
    nlp_engine = provider.create_engine()

    registry = RecognizerRegistry()
    registry.load_predefined_recognizers()

    phone_pattern = Pattern(
        name="phone_us_like",
        regex=r"(?<!\d)(?:\+?1[\s\-\.]?)?(?:\(?\d{3}\)?[\s\-\.]?)\d{3}[\s\-\.]?\d{4}(?!\d)",
        score=0.45,
    )
    registry.add_recognizer(
        PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            patterns=[phone_pattern],
            context=["call", "phone", "mobile", "tel", "contact"],
        )
    )

    analyzer = AnalyzerEngine(
        nlp_engine=nlp_engine,
        registry=registry,
        supported_languages=["en"],
    )
    return analyzer


def iter_texts_from_example(ex: Dict[str, Any], text_fields: List[str]) -> List[Tuple[str, str]]:
    """
    Extract (field_path, text) pairs from an example.
    """
    out: List[Tuple[str, str]] = []
    for field in text_fields:
        if field == "messages" and isinstance(ex.get("messages"), list):
            for i, m in enumerate(ex["messages"]):
                if not isinstance(m, dict):
                    continue
                content = m.get("content")
                if isinstance(content, str) and content.strip():
                    role = m.get("role", "unknown")
                    out.append((f"messages[{i}].content(role={role})", content))
        else:
            val = ex.get(field)
            if isinstance(val, str) and val.strip():
                out.append((field, val))
    return out


def analyze_text(
    analyzer: AnalyzerEngine,
    text: str,
    language: str = "en",
    entities: Optional[List[str]] = None,
    score_threshold: float = 0.5,
    max_snippet_len: int = 120,
) -> List[Hit]:
    """
    Run Presidio analyze and return structured hits.
    """
    results = analyzer.analyze(
        text=text,
        language=language,
        entities=entities,
        score_threshold=score_threshold,
        return_decision_process=False,
    )

    hits: List[Hit] = []
    for r in results:
        snippet = text[r.start:r.end]
        if len(snippet) > max_snippet_len:
            snippet = snippet[:max_snippet_len] + "…"
        hits.append(
            Hit(
                entity_type=r.entity_type,
                start=r.start,
                end=r.end,
                score=float(r.score),
                text=snippet,
            )
        )
    return hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="HF dataset name, e.g. allenai/tulu-3-sft-mixture")
    ap.add_argument("--config", default=None, help="HF dataset config name, if needed")
    ap.add_argument("--split", default="train", help="Dataset split, e.g. train, validation, test")
    ap.add_argument("--streaming", action="store_true", help="Use streaming mode, recommended for huge datasets")
    ap.add_argument("--max-examples", type=int, default=0, help="Limit number of examples, 0 means no limit")
    ap.add_argument("--output", default="pii_hits.jsonl", help="Output JSONL path for hits")
    ap.add_argument("--summary", default="pii_summary.json", help="Output summary JSON path")
    ap.add_argument("--text-fields", default="messages", help='Comma separated fields, e.g. "messages" or "prompt,response"')
    ap.add_argument("--spacy-model", default="en_core_web_lg", help="spaCy model, e.g. en_core_web_sm or en_core_web_lg")
    ap.add_argument("--score-threshold", type=float, default=0.5, help="Presidio score threshold")
    ap.add_argument("--entities", default="", help="Comma separated entity types to include, empty means all")
    ap.add_argument("--language", default="en", help="Language code for Presidio")
    args = ap.parse_args()

    text_fields = [f.strip() for f in args.text_fields.split(",") if f.strip()]
    entities = [e.strip() for e in args.entities.split(",") if e.strip()] or None

    analyzer = build_analyzer(args.spacy_model)

    ds = load_dataset(
        path=args.dataset,
        name=args.config,
        split=args.split,
        streaming=bool(args.streaming),
    )

    total_examples = 0
    examples_with_hits = 0
    total_hits = 0
    entity_counter: Dict[str, int] = {}

    with open(args.output, "w", encoding="utf-8") as f_out:
        pbar = tqdm(ds, desc="Scanning", unit="ex")

        for ex in pbar:
            if args.max_examples and total_examples >= args.max_examples:
                break

            texts = iter_texts_from_example(ex, text_fields)
            ex_hits: List[Dict[str, Any]] = []

            for field_path, text in texts:
                hits = analyze_text(
                    analyzer=analyzer,
                    text=text,
                    language=args.language,
                    entities=entities,
                    score_threshold=args.score_threshold,
                )
                if not hits:
                    continue

                for h in hits:
                    entity_counter[h.entity_type] = entity_counter.get(h.entity_type, 0) + 1

                ex_hits.append(
                    {
                        "field": field_path,
                        "num_hits": len(hits),
                        "hits": [h.__dict__ for h in hits],
                    }
                )

            if ex_hits:
                examples_with_hits += 1
                n = sum(x["num_hits"] for x in ex_hits)
                total_hits += n

                record = {
                    "example_index": total_examples,
                    "num_fields_scanned": len(texts),
                    "num_hits": n,
                    "hits_by_field": ex_hits,
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

            total_examples += 1

            pbar.set_postfix(
                total=total_examples,
                with_hits=examples_with_hits,
                hit_rate=f"{(examples_with_hits / max(total_examples, 1)) * 100:.2f}%",
                total_hits=total_hits,
            )

    summary = {
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "streaming": bool(args.streaming),
        "max_examples": args.max_examples,
        "text_fields": text_fields,
        "spacy_model": args.spacy_model,
        "score_threshold": args.score_threshold,
        "entities_filter": entities,
        "total_examples_scanned": total_examples,
        "examples_with_hits": examples_with_hits,
        "example_hit_rate": examples_with_hits / max(total_examples, 1),
        "total_hits": total_hits,
        "hits_by_entity_type": dict(sorted(entity_counter.items(), key=lambda kv: kv[1], reverse=True)),
    }

    with open(args.summary, "w", encoding="utf-8") as f_sum:
        json.dump(summary, f_sum, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(f"Hits JSONL: {args.output}")
    print(f"Summary: {args.summary}")


if __name__ == "__main__":
    main()