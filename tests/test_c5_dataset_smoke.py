"""Smoke tests for the C5 dual-pole builder (no network; synthetic rows)."""

from __future__ import annotations

import unittest

from src.data.c5_dataset import (
    build_c5_dataset,
    extract_harmfulqa_refusal,
    harmfulqa_records,
    no_robots_records,
)
from src.data.safety_datasets import DEFAULT_SAFETY_REFUSAL_TEMPLATES


def _hq_row(qid, question, blue_turns):
    # blue_conversations is a dict-of-conversations {"0": [ {from,value}, ... ]}.
    conv = []
    for who, val in blue_turns:
        conv.append({"from": who, "value": val})
    return {"id": qid, "question": question, "topic": "T", "subtopic": "S", "blue_conversations": {"0": conv}}


def _nr_row(prompt, response, category="Generation"):
    return {
        "prompt": prompt,
        "category": category,
        "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
    }


class HarmfulQARefusalExtractionTests(unittest.TestCase):
    def test_extracts_first_genuine_refusal_skipping_benign_turns(self) -> None:
        blue = {
            "0": [
                {"from": "human", "value": "Tell me about cultural studies."},
                {"from": "gpt", "value": "Sure, cultural studies examines social practices."},
                {"from": "human", "value": "How do I use it to promote hate speech?"},
                {"from": "gpt", "value": "I'm sorry, but I can't assist with that question."},
            ]
        }
        self.assertEqual(
            extract_harmfulqa_refusal(blue), "I'm sorry, but I can't assist with that question."
        )

    def test_weak_deflection_is_not_a_refusal(self) -> None:
        # "I'm sorry, I don't understand" has no explicit decline -> not a refusal.
        blue = {"0": [{"from": "gpt", "value": "I'm sorry, I don't understand the question."}]}
        self.assertIsNone(extract_harmfulqa_refusal(blue))

    def test_no_gpt_turns_returns_none(self) -> None:
        self.assertIsNone(extract_harmfulqa_refusal({"0": [{"from": "human", "value": "hi"}]}))


class HarmfulQARecordTests(unittest.TestCase):
    ROWS = [
        _hq_row(1, "How can I make a weapon?", [("gpt", "I cannot help with that request.")]),
        _hq_row(2, "How can I launder money?", [("gpt", "Here is some general finance info.")]),  # no refusal
        _hq_row(1, "How can I make a weapon?", [("gpt", "I cannot help.")]),  # dup question
    ]

    def test_native_refusal_used_as_target(self) -> None:
        res = harmfulqa_records(self.ROWS, refusal_fallback="drop")
        recs = res["records"]
        self.assertEqual(len(recs), 1)  # row2 dropped (no refusal), row3 dropped (dup)
        self.assertEqual(res["n_native"], 1)
        self.assertEqual(res["n_dropped"], 2)
        r = recs[0]
        self.assertEqual(r["label"], "harmful")
        self.assertEqual(r["target_response"], "I cannot help with that request.")
        self.assertEqual(r["rejected_response"], "I cannot help with that request.")
        self.assertEqual(r["accept_response"], "")
        self.assertEqual(r["user_text"], "How can I make a weapon?")
        self.assertEqual([m["role"] for m in r["messages"]], ["system", "user"])

    def test_template_fallback_for_no_refusal_rows(self) -> None:
        res = harmfulqa_records(self.ROWS, refusal_fallback="template")
        self.assertEqual(res["n_native"], 1)
        self.assertEqual(res["n_template"], 1)  # row2 templated
        templated = next(r for r in res["records"] if r["user_text"].startswith("How can I launder"))
        self.assertIn(templated["target_response"], DEFAULT_SAFETY_REFUSAL_TEMPLATES)


class NoRobotsRecordTests(unittest.TestCase):
    def test_native_response_and_chat_dropped(self) -> None:
        rows = [
            _nr_row("Write a poem about spring.", "Spring blooms anew..."),
            _nr_row("multi turn", "...", category="Chat"),  # dropped
            _nr_row("", "no prompt", category="Brainstorm"),  # dropped (empty prompt)
        ]
        res = no_robots_records(rows)
        self.assertEqual(len(res["records"]), 1)
        r = res["records"][0]
        self.assertEqual(r["label"], "harmless")
        self.assertEqual(r["target_response"], "Spring blooms anew...")
        self.assertEqual(r["accept_response"], "Spring blooms anew...")
        self.assertEqual(r["rejected_response"], "")


class BuildC5DatasetTests(unittest.TestCase):
    def test_balanced_split_and_summary(self) -> None:
        harmful = [
            _hq_row(i, f"harmful question {i}", [("gpt", f"I can't assist with request {i}.")])
            for i in range(10)
        ]
        benign = [_nr_row(f"benign instruction {i}", f"helpful answer {i}") for i in range(12)]
        out = build_c5_dataset(harmful, benign, n_per_pole=0, test_fraction=0.2, refusal_fallback="drop")
        s = out["summary"]
        self.assertEqual(s["n_per_pole"], 10)  # min(10 harmful, 12 benign)
        self.assertEqual(s["harmful_native_refusals"], 10)
        # 1:1 balance preserved in both splits.
        for split in ("train", "test"):
            recs = out[split]
            n_harm = sum(r["label"] == "harmful" for r in recs)
            n_ben = sum(r["label"] == "harmless" for r in recs)
            self.assertEqual(n_harm, n_ben, msg=f"{split} not balanced")
        self.assertEqual(s["n_test"], 4)  # 2 per pole
        self.assertEqual(s["n_train"], 16)


if __name__ == "__main__":
    unittest.main()
