#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PII Detection and Masking Pipeline for LLM-PBE/enron-email.

Combines Microsoft Presidio (backed by spaCy) for entity recognition with a
Qwen2.5-7B tokenizer to produce per-token boolean masks that indicate which
tokens overlap a target PII entity type.

Workflow per example
--------------------
1. Load raw text from the dataset.
2. Run Presidio → list of (entity_type, char_start, char_end) spans.
3. Tokenize with the Qwen tokenizer (return_offsets_mapping=True, truncation).
4. For each token, check overlap with spans whose entity_type == target_entity.
5. Emit a Boolean PyTorch tensor of shape (seq_len,):  1 = PII, 0 = other.
6. Write result to JSONL.

Usage
-----
python pii_masking_pipeline.py \
    --split train \
    --target-entity EMAIL_ADDRESS \
    --max-examples 500 \
    --output results.jsonl
"""

import argparse
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, cast

import torch
from datasets import load_dataset
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.pattern import Pattern
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PIISpan:
    """One detected PII entity within a text string."""
    entity_type: str
    start: int          # character-level start offset (inclusive)
    end: int            # character-level end offset (exclusive)
    score: float        # Presidio confidence score
    text: str           # raw substring


@dataclass
class MaskedExample:
    example_index: int
    original_text: str
    pii_spans: List[PIISpan]
    target_entity: str
    mask: List[int]


# ---------------------------------------------------------------------------
# Presidio analyzer
# ---------------------------------------------------------------------------

def build_analyzer(spacy_model: str = "en_core_web_lg") -> AnalyzerEngine:
    """
    Build a Presidio AnalyzerEngine backed by a spaCy NLP engine.

    Also registers a supplementary US-style phone-number recognizer with a
    custom regex and context words to boost its confidence.

    Args:
        spacy_model: Name of the spaCy model to load (e.g. ``en_core_web_lg``).

    Returns:
        A ready-to-use :class:`AnalyzerEngine`.
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

    # Supplementary phone-number pattern (US-like formats)
    phone_pattern = Pattern(
        name="phone_us_like",
        regex=(
            r"(?<!\d)"
            r"(?:\+?1[\s\-\.]?)?"
            r"(?:\(?\d{3}\)?[\s\-\.]?)\d{3}[\s\-\.]?\d{4}"
            r"(?!\d)"
        ),
        score=0.45,
    )
    registry.add_recognizer(
        PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            patterns=[phone_pattern],
            context=["call", "phone", "mobile", "tel", "contact"],
        )
    )

    return AnalyzerEngine(
        nlp_engine=nlp_engine,
        registry=registry,
        supported_languages=["en"],
    )


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def load_tokenizer(
    model_name: str = "Qwen/Qwen2.5-7B",
) -> PreTrainedTokenizerBase:
    """
    Load the HuggingFace tokenizer for *model_name*.

    ``trust_remote_code=False`` is intentionally used so that no remote code
    is executed.  If the tokenizer has no pad token, the EOS token is used as
    a fallback.

    Args:
        model_name: HuggingFace model / tokenizer identifier.

    Returns:
        A :class:`PreTrainedTokenizerBase` instance.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log.info("Loaded tokenizer: %s  (vocab_size=%d)", model_name, tokenizer.vocab_size)
    return tokenizer


# ---------------------------------------------------------------------------
# PII detection
# ---------------------------------------------------------------------------

def detect_pii(
    analyzer: AnalyzerEngine,
    text: str,
    language: str = "en",
    entities: Optional[List[str]] = None,
    score_threshold: float = 0.5,
) -> List[PIISpan]:
    """
    Run Presidio on *text* and return a sorted list of :class:`PIISpan` objects.

    Args:
        analyzer: Initialised Presidio :class:`AnalyzerEngine`.
        text: Input string to scan for PII.
        language: Language code passed to Presidio (default ``"en"``).
        entities: Optional whitelist of entity types.  ``None`` means all.
        score_threshold: Minimum confidence required to include a result.

    Returns:
        PII spans sorted by ``start`` offset (ascending).
    """
    results = analyzer.analyze(
        text=text,
        language=language,
        entities=entities,
        score_threshold=score_threshold,
        return_decision_process=False,
    )
    spans = []
    for r in results:
        start = r.start
        end = r.end
        while start < end and not text[start].isalnum():
            start += 1
        spans.append(
            PIISpan(
                entity_type=r.entity_type,
                start=start,
                end=end,
                score=float(r.score),
                text=text[start:end],
            )
        )
    return sorted(spans, key=lambda s: s.start)


# ---------------------------------------------------------------------------
# Tokenisation with character-offset mapping
# ---------------------------------------------------------------------------

def tokenize_with_offsets(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    max_length: int = 512,
    add_special_tokens: bool = True,
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Tokenize *text* and return token IDs **and** a character-level offset
    mapping for every token position.

    The HuggingFace fast tokenizer returns ``offset_mapping`` entries of
    ``(0, 0)`` for special tokens (BOS, EOS, etc.).  Those positions will be
    masked as 0 by :func:`build_pii_mask`.

    Args:
        tokenizer: A fast HuggingFace tokenizer supporting
            ``return_offsets_mapping``.
        text: The raw input string.
        max_length: Token-level truncation ceiling.
        add_special_tokens: Whether to prepend / append model special tokens.

    Returns:
        A ``(token_ids, offset_mapping)`` pair where both lists have the same
        length equal to the actual (possibly truncated) sequence length.
    """
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=add_special_tokens,
    )
    token_ids: List[int] = cast(List[int], encoding["input_ids"])
    offset_mapping: List[Tuple[int, int]] = [
        (int(s), int(e)) for s, e in cast(List[Tuple[int, int]], encoding["offset_mapping"])
    ]
    return token_ids, offset_mapping


# ---------------------------------------------------------------------------
# Boolean mask construction
# ---------------------------------------------------------------------------

def build_pii_mask(
    offset_mapping: List[Tuple[int, int]],
    pii_spans: List[PIISpan],
    target_entity: str,
) -> torch.Tensor:
    """
    Build a 1-D boolean :class:`torch.Tensor` of shape ``(num_tokens,)``.

    A token at position *i* is set to ``True`` (1) when **all** of the
    following hold:

    * The token's character span is not ``(0, 0)`` (i.e. it is a real token,
      not a special/padding token).
    * The token's character interval ``[tok_start, tok_end)`` overlaps at
      least one PII span of type *target_entity*, i.e.
      ``tok_start < span_end AND tok_end > span_start``.

    Args:
        offset_mapping: Per-token ``(char_start, char_end)`` pairs, as
            returned by :func:`tokenize_with_offsets`.
        pii_spans: All detected PII spans for the text.
        target_entity: Only spans with this ``entity_type`` contribute to
            the mask (e.g. ``"EMAIL_ADDRESS"``).

    Returns:
        Boolean tensor of shape ``(len(offset_mapping),)``.
    """
    target_spans: List[Tuple[int, int]] = [
        (s.start, s.end)
        for s in pii_spans
        if s.entity_type == target_entity
    ]

    mask = torch.zeros(len(offset_mapping), dtype=torch.bool)

    if not target_spans:
        return mask

    for i, (tok_start, tok_end) in enumerate(offset_mapping):
        if tok_start == 0 and tok_end == 0:
            continue
        for span_start, span_end in target_spans:
            if tok_start < span_end and tok_end > span_start:
                mask[i] = True
                break 

    return mask


# ---------------------------------------------------------------------------
# Single-example end-to-end processing
# ---------------------------------------------------------------------------

def process_example(
    example_index: int,
    text: str,
    analyzer: AnalyzerEngine,
    tokenizer: PreTrainedTokenizerBase,
    target_entity: str,
    language: str = "en",
    entities: Optional[List[str]] = None,
    score_threshold: float = 0.5,
    max_length: int = 512,
) -> MaskedExample:
    """
    Run the full pipeline for a single text string.

    Steps:
        1. Detect PII spans with Presidio.
        2. Tokenize and obtain the character-offset mapping.
        3. Build the boolean PII mask for *target_entity*.

    Args:
        example_index: Position of this example in the dataset iteration.
        text: Raw input text.
        analyzer: Presidio :class:`AnalyzerEngine`.
        tokenizer: HuggingFace tokenizer.
        target_entity: PII type for the boolean mask.
        language: Presidio language code.
        entities: Optional entity whitelist for Presidio.
        score_threshold: Presidio confidence threshold.
        max_length: Tokenizer truncation length.

    Returns:
        A populated :class:`MaskedExample`.
    """
    try:
        all_spans = detect_pii(
            analyzer=analyzer,
            text=text,
            language=language,
            entities=entities,
            score_threshold=score_threshold,
        )

        pii_spans = [
            s for s in all_spans
            if s.entity_type == target_entity
        ]
        
        token_ids, offset_mapping = tokenize_with_offsets(
            tokenizer=tokenizer,
            text=text,
            max_length=max_length,
        )

        mask_tensor = build_pii_mask(
            offset_mapping=offset_mapping,
            pii_spans=pii_spans,
            target_entity=target_entity,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "Example %d failed (%s: %s) — returning empty result.",
            example_index, type(exc).__name__, exc,
        )
        return MaskedExample(
            example_index=example_index,
            original_text=text,
            token_ids=[],
            offset_mapping=[],
            pii_spans=[],
            target_entity=target_entity,
            mask=[],
        )

    return MaskedExample(
        example_index=example_index,
        original_text=text,
        pii_spans=pii_spans,
        target_entity=target_entity,
        mask=mask_tensor.int().tolist(),
    )


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_batch(
    examples: List[Tuple[int, str]],
    analyzer: AnalyzerEngine,
    tokenizer: PreTrainedTokenizerBase,
    target_entity: str,
    language: str = "en",
    entities: Optional[List[str]] = None,
    score_threshold: float = 0.5,
    max_length: int = 512,
) -> List[MaskedExample]:
    """
    Process a batch of ``(index, text)`` pairs through the full pipeline.

    Presidio does not support native batching, so examples are processed
    sequentially.  The tokenizer is also called per example so that each
    example retains its own offset alignment without padding artefacts.

    Args:
        examples: List of ``(example_index, text)`` tuples.
        analyzer: Presidio :class:`AnalyzerEngine`.
        tokenizer: HuggingFace tokenizer.
        target_entity: PII entity type for the boolean mask.
        language: Language code for Presidio.
        entities: Optional entity whitelist for Presidio.
        score_threshold: Presidio confidence threshold.
        max_length: Tokenizer truncation length.

    Returns:
        A list of :class:`MaskedExample` objects in the same order as
        *examples*.
    """
    results: List[MaskedExample] = []
    for idx, text in examples:
        results.append(
            process_example(
                example_index=idx,
                text=text,
                analyzer=analyzer,
                tokenizer=tokenizer,
                target_entity=target_entity,
                language=language,
                entities=entities,
                score_threshold=score_threshold,
                max_length=max_length,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def masked_example_to_dict(me: MaskedExample) -> Dict[str, Any]:
    """
    Convert a :class:`MaskedExample` to a JSON-serialisable :class:`dict`.

    The ``offset_mapping`` is stored as a list of ``[char_start, char_end]``
    pairs.  The ``mask`` field is a flat list of ``int`` values (0 or 1).
    """
    return {
        "example_index": me.example_index,
        "original_text": me.original_text,
        "pii_spans": [
            {
                "entity_type": s.entity_type,
                "start": s.start,
                "end": s.end,
                "score": round(s.score, 4),
                "text": s.text,
            }
            for s in me.pii_spans
        ],
        "target_entity": me.target_entity,
        "mask": me.mask,
    }


# ---------------------------------------------------------------------------
# Text-field extraction
# ---------------------------------------------------------------------------

def extract_text(example: Dict[str, Any], text_field: str) -> Optional[str]:
    """
    Extract and return the text string from a dataset row.

    For the ``LLM-PBE/enron-email`` dataset the primary field is ``"text"``.
    If the value is empty or missing, returns ``None`` so the caller can skip
    the example cleanly.

    Args:
        example: A single dataset row (dict).
        text_field: Key of the text column.

    Returns:
        Stripped text string, or ``None`` if absent / blank.
    """
    val = example.get(text_field)
    if isinstance(val, str) and val.strip():
        return val.strip()
    return None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    """Return a configured :class:`argparse.ArgumentParser`."""
    ap = argparse.ArgumentParser(
        description="PII Detection & Masking Pipeline – LLM-PBE/enron-email",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Dataset ---
    ap.add_argument(
        "--dataset", default="LLM-PBE/enron-email",
        help="HuggingFace dataset identifier.",
    )
    ap.add_argument(
        "--config", default=None,
        help="Dataset config name, if required.",
    )
    ap.add_argument(
        "--split", default="train",
        help="Dataset split: train / test / validation.",
    )
    ap.add_argument(
        "--streaming", action="store_true",
        help="Enable streaming mode (recommended for large datasets).",
    )
    ap.add_argument(
        "--max-examples", type=int, default=0,
        help="Maximum number of examples to process. 0 = no limit.",
    )
    ap.add_argument(
        "--text-field", default="text",
        help="Name of the text column in the dataset.",
    )

    # --- PII detection ---
    ap.add_argument(
        "--target-entity", default="EMAIL_ADDRESS",
        help="PII entity type to build the boolean mask for.",
    )
    ap.add_argument(
        "--entities", default="",
        help="Comma-separated entity types passed to Presidio. Empty = all.",
    )
    ap.add_argument(
        "--score-threshold", type=float, default=0.5,
        help="Presidio minimum confidence score.",
    )
    ap.add_argument(
        "--spacy-model", default="en_core_web_lg",
        help="spaCy model name.",
    )
    ap.add_argument(
        "--language", default="en",
        help="Language code for Presidio.",
    )

    # --- Tokenizer ---
    ap.add_argument(
        "--tokenizer", default="Qwen/Qwen2.5-7B",
        help="HuggingFace model ID for the tokenizer.",
    )
    ap.add_argument(
        "--max-length", type=int, default=2048,
        help=(
            "Maximum token sequence length (truncation applied beyond this). "
            "Enron emails average ~1 000 chars; default 2048 tokens keeps most "
            "emails whole.  Lower to 512 to trade coverage for speed."
        ),
    )

    # --- Batching & output ---
    ap.add_argument(
        "--batch-size", type=int, default=32,
        help="Number of examples accumulated before flushing to disk.",
    )
    ap.add_argument(
        "--output", default="pii_masked.jsonl",
        help="Output JSONL file path.",
    )
    ap.add_argument(
        "--summary", default="pii_masked_summary.json",
        help="Output summary JSON file path.",
    )

    return ap


def main() -> None:
    args = build_arg_parser().parse_args()

    entities_filter: Optional[List[str]] = (
        [e.strip() for e in args.entities.split(",") if e.strip()] or None
    )

    # ------------------------------------------------------------------ setup
    log.info("Building Presidio analyzer (spaCy model: %s)…", args.spacy_model)
    analyzer = build_analyzer(args.spacy_model)

    log.info("Loading tokenizer '%s'…", args.tokenizer)
    tokenizer = load_tokenizer(args.tokenizer)

    # ----------------------------------------------------------------- dataset
    log.info(
        "Loading dataset '%s'  config=%s  split=%s…",
        args.dataset, args.config, args.split,
    )
    ds = load_dataset(
        path=args.dataset,
        name=args.config,
        split=args.split,
        streaming=bool(args.streaming),
    )

    # ---------------------------------------------------------------- counters
    total_examples = 0
    skipped = 0
    examples_with_target_pii = 0
    entity_counter: Dict[str, int] = {}

    # ----------------------------------------- batch accumulator & flush logic
    batch: List[Tuple[int, str]] = []

    def flush_batch() -> None:
        """Process the current batch and write results to disk."""
        nonlocal examples_with_target_pii
        processed = process_batch(
            examples=batch,
            analyzer=analyzer,
            tokenizer=tokenizer,
            target_entity=args.target_entity,
            language=args.language,
            entities=entities_filter,
            score_threshold=args.score_threshold,
            max_length=args.max_length,
        )
        for me in processed:
            for span in me.pii_spans:
                entity_counter[span.entity_type] = (
                    entity_counter.get(span.entity_type, 0) + 1
                )
            if any(me.mask):
                examples_with_target_pii += 1
            f_out.write(
                json.dumps(masked_example_to_dict(me), ensure_ascii=False) + "\n"
            )

    # --------------------------------------------------------------- main loop
    with open(args.output, "w", encoding="utf-8") as f_out:
        pbar = tqdm(ds, desc="Processing", unit="ex")

        for ex in pbar:
            if args.max_examples and total_examples >= args.max_examples:
                break

            text = extract_text(ex, args.text_field)
            if text is None:
                skipped += 1
                continue

            batch.append((total_examples, text))
            total_examples += 1

            if len(batch) >= args.batch_size:
                flush_batch()
                batch.clear()

            pbar.set_postfix(
                processed=total_examples,
                skipped=skipped,
                with_target=examples_with_target_pii,
            )

        # Flush any remaining examples
        if batch:
            flush_batch()
            batch.clear()

    # -------------------------------------------------------------- summary
    summary = {
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "streaming": bool(args.streaming),
        "max_examples": args.max_examples,
        "text_field": args.text_field,
        "tokenizer": args.tokenizer,
        "spacy_model": args.spacy_model,
        "score_threshold": args.score_threshold,
        "entities_filter": entities_filter,
        "target_entity": args.target_entity,
        "max_length": args.max_length,
        "total_examples_processed": total_examples,
        "examples_skipped": skipped,
        "examples_with_target_pii": examples_with_target_pii,
        "target_pii_hit_rate": examples_with_target_pii / max(total_examples, 1),
        "hits_by_entity_type": dict(
            sorted(entity_counter.items(), key=lambda kv: kv[1], reverse=True)
        ),
    }
    with open(args.summary, "w", encoding="utf-8") as f_sum:
        json.dump(summary, f_sum, ensure_ascii=False, indent=2)

    # -------------------------------------------------------------- final log
    log.info("─" * 60)
    log.info("Done.")
    log.info("  Examples processed : %d", total_examples)
    log.info("  Examples skipped   : %d", skipped)
    log.info(
        "  Examples with '%s' in mask: %d (%.2f%%)",
        args.target_entity,
        examples_with_target_pii,
        examples_with_target_pii / max(total_examples, 1) * 100,
    )
    log.info(
        "  Entity counts:\n%s",
        json.dumps(
            dict(sorted(entity_counter.items(), key=lambda kv: kv[1], reverse=True)),
            indent=4,
        ),
    )
    log.info("  Output written to  : %s", args.output)
    log.info("  Summary written to : %s", args.summary)


if __name__ == "__main__":
    main()
