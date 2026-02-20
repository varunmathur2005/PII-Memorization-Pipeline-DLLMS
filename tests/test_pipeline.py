"""
Unit and integration tests for pii_masking_pipeline.py.

Run with:
    pytest tests/ -v

Fast tests (no network, no spaCy) use mocks.
Slow tests (marked 'integration') hit real Presidio + spaCy and require
    python -m spacy download en_core_web_lg
to have been run first.

Skip integration tests:
    pytest tests/ -v -m "not integration"
"""

import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
from unittest.mock import MagicMock, patch

import pytest
import torch

# Make the project root importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from pii_masking_pipeline import (
    MaskedExample,
    PIISpan,
    build_pii_mask,
    extract_text,
    masked_example_to_dict,
    process_example,
    tokenize_with_offsets,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_span(entity_type: str, start: int, end: int, text: str = "") -> PIISpan:
    return PIISpan(entity_type=entity_type, start=start, end=end, score=1.0, text=text)


def _make_offsets(pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    return pairs


# ---------------------------------------------------------------------------
# build_pii_mask — pure logic, no external deps
# ---------------------------------------------------------------------------

class TestBuildPiiMask:
    """Tests for the core interval-overlap masking logic."""

    def test_empty_spans_returns_all_zeros(self):
        offsets = _make_offsets([(0, 0), (0, 5), (5, 10)])
        mask = build_pii_mask(offsets, [], "EMAIL_ADDRESS")
        assert mask.tolist() == [0, 0, 0]

    def test_wrong_entity_type_returns_all_zeros(self):
        offsets = _make_offsets([(0, 0), (0, 5), (5, 10)])
        spans = [_make_span("PERSON", 0, 5)]
        mask = build_pii_mask(offsets, spans, "EMAIL_ADDRESS")
        assert mask.tolist() == [0, 0, 0]

    def test_special_token_always_zero(self):
        # (0,0) offset entries are special tokens and must stay 0
        offsets = _make_offsets([(0, 0), (0, 5), (5, 10), (0, 0)])
        spans = [_make_span("EMAIL_ADDRESS", 0, 10)]
        mask = build_pii_mask(offsets, spans, "EMAIL_ADDRESS")
        assert mask[0].item() == 0, "BOS/EOS must be 0"
        assert mask[3].item() == 0, "BOS/EOS must be 0"

    def test_exact_overlap(self):
        # token exactly covers span
        offsets = _make_offsets([(0, 0), (0, 20), (20, 25)])
        spans = [_make_span("EMAIL_ADDRESS", 0, 20)]
        mask = build_pii_mask(offsets, spans, "EMAIL_ADDRESS")
        assert mask.tolist() == [0, 1, 0]

    def test_partial_overlap_start(self):
        # token starts before span ends → overlap
        offsets = _make_offsets([(0, 0), (15, 25), (25, 30)])
        spans = [_make_span("EMAIL_ADDRESS", 10, 22)]
        mask = build_pii_mask(offsets, spans, "EMAIL_ADDRESS")
        assert mask.tolist() == [0, 1, 0]

    def test_partial_overlap_end(self):
        # token ends after span starts → overlap
        offsets = _make_offsets([(0, 0), (8, 15), (15, 25)])
        spans = [_make_span("EMAIL_ADDRESS", 10, 22)]
        mask = build_pii_mask(offsets, spans, "EMAIL_ADDRESS")
        assert mask.tolist() == [0, 1, 1]

    def test_no_overlap_adjacent(self):
        # span ends exactly where token starts — no overlap
        offsets = _make_offsets([(0, 0), (0, 10), (10, 20)])
        spans = [_make_span("EMAIL_ADDRESS", 0, 10)]
        mask = build_pii_mask(offsets, spans, "EMAIL_ADDRESS")
        assert mask.tolist() == [0, 1, 0]

    def test_multiple_spans_same_entity(self):
        offsets = _make_offsets([(0, 0), (0, 5), (5, 10), (10, 15), (15, 20)])
        spans = [
            _make_span("EMAIL_ADDRESS", 0, 5),
            _make_span("EMAIL_ADDRESS", 10, 20),
        ]
        mask = build_pii_mask(offsets, spans, "EMAIL_ADDRESS")
        assert mask.tolist() == [0, 1, 0, 1, 1]

    def test_multiple_entity_types_only_target_masked(self):
        offsets = _make_offsets([(0, 0), (0, 5), (5, 15), (15, 20)])
        spans = [
            _make_span("EMAIL_ADDRESS", 0, 5),
            _make_span("PERSON", 5, 15),
        ]
        mask = build_pii_mask(offsets, spans, "EMAIL_ADDRESS")
        assert mask.tolist() == [0, 1, 0, 0]

    def test_output_is_bool_tensor(self):
        offsets = _make_offsets([(0, 5)])
        mask = build_pii_mask(offsets, [], "EMAIL_ADDRESS")
        assert mask.dtype == torch.bool

    def test_output_length_matches_offset_mapping(self):
        offsets = _make_offsets([(0, 0)] + [(i * 5, i * 5 + 5) for i in range(9)])
        mask = build_pii_mask(offsets, [], "EMAIL_ADDRESS")
        assert len(mask) == len(offsets)


# ---------------------------------------------------------------------------
# extract_text — pure logic, no external deps
# ---------------------------------------------------------------------------

class TestExtractText:
    def test_returns_text_for_valid_field(self):
        row = {"text": "Hello world"}
        assert extract_text(row, "text") == "Hello world"

    def test_strips_whitespace(self):
        row = {"text": "  hello  "}
        assert extract_text(row, "text") == "hello"

    def test_returns_none_for_missing_field(self):
        assert extract_text({}, "text") is None

    def test_returns_none_for_empty_string(self):
        assert extract_text({"text": "   "}, "text") is None

    def test_returns_none_for_non_string(self):
        assert extract_text({"text": 42}, "text") is None
        assert extract_text({"text": None}, "text") is None
        assert extract_text({"text": ["list"]}, "text") is None

    def test_custom_field_name(self):
        row = {"body": "email body text"}
        assert extract_text(row, "body") == "email body text"


# ---------------------------------------------------------------------------
# masked_example_to_dict — serialisation, no external deps
# ---------------------------------------------------------------------------

class TestMaskedExampleToDict:
    def _make_example(self) -> MaskedExample:
        return MaskedExample(
            example_index=7,
            original_text="Send to alice@example.com please.",
            token_ids=[1, 2, 3, 4, 5],
            offset_mapping=[(0, 0), (0, 4), (5, 7), (8, 27), (28, 33)],
            pii_spans=[
                PIISpan("EMAIL_ADDRESS", 8, 27, 1.0, "alice@example.com")
            ],
            target_entity="EMAIL_ADDRESS",
            mask=[0, 0, 0, 1, 0],
        )

    def test_all_keys_present(self):
        d = masked_example_to_dict(self._make_example())
        expected_keys = {
            "example_index", "original_text", "token_ids",
            "offset_mapping", "pii_spans", "target_entity", "mask",
        }
        assert set(d.keys()) == expected_keys

    def test_is_json_serialisable(self):
        d = masked_example_to_dict(self._make_example())
        dumped = json.dumps(d)  # must not raise
        reloaded = json.loads(dumped)
        assert reloaded["example_index"] == 7

    def test_mask_is_int_list(self):
        d = masked_example_to_dict(self._make_example())
        assert all(isinstance(v, int) for v in d["mask"])

    def test_offset_mapping_is_list_of_lists(self):
        d = masked_example_to_dict(self._make_example())
        for pair in d["offset_mapping"]:
            assert isinstance(pair, list)
            assert len(pair) == 2

    def test_pii_span_fields(self):
        d = masked_example_to_dict(self._make_example())
        span = d["pii_spans"][0]
        assert span["entity_type"] == "EMAIL_ADDRESS"
        assert span["start"] == 8
        assert span["end"] == 27
        assert span["text"] == "alice@example.com"

    def test_empty_mask_and_spans(self):
        ex = MaskedExample(7, "hi", [], [], [], "EMAIL_ADDRESS", [])
        d = masked_example_to_dict(ex)
        assert d["mask"] == []
        assert d["pii_spans"] == []


# ---------------------------------------------------------------------------
# tokenize_with_offsets — mocked tokenizer (fast, no download)
# ---------------------------------------------------------------------------

class TestTokenizeWithOffsets:
    def _make_mock_tokenizer(self, input_ids, offset_mapping):
        """Build a minimal mock of HuggingFace BatchEncoding."""
        encoding = {
            "input_ids": input_ids,
            "offset_mapping": offset_mapping,
        }
        mock_tok = MagicMock()
        mock_tok.return_value = encoding
        return mock_tok

    def test_returns_token_ids_and_offsets(self):
        ids = [1, 2, 3]
        offsets = [(0, 0), (0, 5), (5, 10)]
        tok = self._make_mock_tokenizer(ids, offsets)
        out_ids, out_offsets = tokenize_with_offsets(tok, "hello world")
        assert out_ids == ids
        assert out_offsets == offsets

    def test_lengths_match(self):
        ids = [1, 2, 3, 4, 5]
        offsets = [(0, 0), (0, 3), (3, 6), (6, 9), (0, 0)]
        tok = self._make_mock_tokenizer(ids, offsets)
        out_ids, out_offsets = tokenize_with_offsets(tok, "foo bar baz")
        assert len(out_ids) == len(out_offsets)

    def test_offsets_are_int_tuples(self):
        ids = [1, 2]
        offsets = [(0, 0), (0, 5)]
        tok = self._make_mock_tokenizer(ids, offsets)
        _, out_offsets = tokenize_with_offsets(tok, "hello")
        for s, e in out_offsets:
            assert isinstance(s, int)
            assert isinstance(e, int)

    def test_truncation_arg_passed(self):
        ids = [1]
        offsets = [(0, 5)]
        tok = self._make_mock_tokenizer(ids, offsets)
        tokenize_with_offsets(tok, "hello", max_length=128)
        call_kwargs = tok.call_args[1]
        assert call_kwargs["truncation"] is True
        assert call_kwargs["max_length"] == 128


# ---------------------------------------------------------------------------
# process_example — mocked analyzer + tokenizer (fast, no download)
# ---------------------------------------------------------------------------

class TestProcessExample:
    def _make_analyzer_mock(self, spans: List[PIISpan]):
        mock = MagicMock()
        # detect_pii calls analyzer.analyze internally; patch detect_pii instead
        return mock

    def test_returns_masked_example(self):
        text = "Email alice@example.com for info."
        spans = [PIISpan("EMAIL_ADDRESS", 6, 23, 1.0, "alice@example.com")]
        token_ids = [0, 1, 2, 3, 4, 5, 6]
        offsets = [(0, 0), (0, 5), (6, 11), (6, 23), (24, 27), (28, 32), (0, 0)]

        with patch("pii_masking_pipeline.detect_pii", return_value=spans), \
             patch("pii_masking_pipeline.tokenize_with_offsets",
                   return_value=(token_ids, offsets)):
            result = process_example(
                example_index=0,
                text=text,
                analyzer=MagicMock(),
                tokenizer=MagicMock(),
                target_entity="EMAIL_ADDRESS",
            )

        assert isinstance(result, MaskedExample)
        assert result.example_index == 0
        assert result.token_ids == token_ids
        assert len(result.mask) == len(token_ids)

    def test_mask_marks_correct_token(self):
        text = "Email alice@example.com for info."
        # token at index 3 covers chars 6-23 → overlaps EMAIL_ADDRESS span
        spans = [PIISpan("EMAIL_ADDRESS", 6, 23, 1.0, "alice@example.com")]
        token_ids = [0, 1, 2, 3, 4, 5, 6]
        offsets = [(0, 0), (0, 5), (6, 11), (6, 23), (24, 27), (28, 32), (0, 0)]

        with patch("pii_masking_pipeline.detect_pii", return_value=spans), \
             patch("pii_masking_pipeline.tokenize_with_offsets",
                   return_value=(token_ids, offsets)):
            result = process_example(0, text, MagicMock(), MagicMock(), "EMAIL_ADDRESS")

        assert result.mask[3] == 1
        assert result.mask[0] == 0   # special token
        assert result.mask[1] == 0   # "Email"
        assert result.mask[4] == 0   # "for"

    def test_error_returns_empty_result(self):
        """If Presidio or the tokenizer raises, we get an empty MaskedExample."""
        with patch("pii_masking_pipeline.detect_pii",
                   side_effect=RuntimeError("spaCy exploded")):
            result = process_example(
                42, "some text", MagicMock(), MagicMock(), "EMAIL_ADDRESS"
            )
        assert result.example_index == 42
        assert result.token_ids == []
        assert result.mask == []

    def test_no_target_pii_all_zeros(self):
        text = "Hello there."
        spans = [PIISpan("PERSON", 0, 5, 0.9, "Hello")]
        token_ids = [0, 1, 2, 3]
        offsets = [(0, 0), (0, 5), (6, 11), (0, 0)]

        with patch("pii_masking_pipeline.detect_pii", return_value=spans), \
             patch("pii_masking_pipeline.tokenize_with_offsets",
                   return_value=(token_ids, offsets)):
            result = process_example(0, text, MagicMock(), MagicMock(), "EMAIL_ADDRESS")

        assert all(v == 0 for v in result.mask)


# ---------------------------------------------------------------------------
# Integration tests — real Presidio + spaCy (require en_core_web_lg)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestIntegration:
    """Require spaCy en_core_web_lg and network access for the tokenizer."""

    @pytest.fixture(scope="class")
    def analyzer(self):
        from pii_masking_pipeline import build_analyzer
        return build_analyzer("en_core_web_lg")

    def test_email_detected(self, analyzer):
        from pii_masking_pipeline import detect_pii
        spans = detect_pii(analyzer, "Contact us at hello@example.com")
        types = [s.entity_type for s in spans]
        assert "EMAIL_ADDRESS" in types

    def test_email_span_offsets_correct(self, analyzer):
        from pii_masking_pipeline import detect_pii
        text = "Send mail to alice@test.org please."
        spans = detect_pii(analyzer, text)
        email_spans = [s for s in spans if s.entity_type == "EMAIL_ADDRESS"]
        assert len(email_spans) >= 1
        s = email_spans[0]
        assert text[s.start:s.end] == s.text

    def test_phone_detected(self, analyzer):
        # The custom phone recognizer scores 0.45 (below the default 0.5 threshold).
        # A lower threshold is needed unless strong context words boost it.
        # This test uses score_threshold=0.4 to confirm the recognizer fires at all.
        from pii_masking_pipeline import detect_pii
        text = "My phone number is (415) 555-1234."
        spans = detect_pii(analyzer, text, score_threshold=0.4)
        types = [s.entity_type for s in spans]
        assert "PHONE_NUMBER" in types

    def test_no_false_positive_on_clean_text(self, analyzer):
        from pii_masking_pipeline import detect_pii
        text = "The weather is nice today. I enjoy coding."
        spans = detect_pii(analyzer, text, score_threshold=0.8)
        high_confidence_pii = [
            s for s in spans
            if s.entity_type in ("EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD")
        ]
        assert high_confidence_pii == []
