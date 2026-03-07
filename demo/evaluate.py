"""LLM-as-judge evaluation: score an answer against ground truth in one call."""

from __future__ import annotations

import json
import re

from config import MODEL, get_llm_client
from models import EvalScore


def _extract_json(text: str) -> dict:
    """Extract JSON from LLM output, handling fences and common issues."""
    text = text.strip()
    # Strip ```json ... ``` or ``` ... ```
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        text = m.group(1).strip()
    # Fix trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return json.loads(text)


_JUDGE_SYSTEM = """\
You are an expert evaluator. Given a question, a ground-truth answer, and a \
candidate answer, score the candidate on five dimensions using integers 1-5 \
(1 = worst, 5 = best):

1. Correctness – Are the facts and numbers accurate compared to the ground truth? \
If the ground truth contains specific dollar amounts, percentages, or figures, the \
candidate MUST include them to score above 2. A bare list of names without numbers \
is NOT a correct answer to a quantitative question.
2. Completeness – Does the candidate cover all key points from the ground truth? \
If the ground truth includes supporting data (e.g. dollar amounts for each company \
in a ranking), the candidate must include comparable detail. A list of names without \
figures is at most 2/5.
3. Relevance – Is the answer on-topic with no irrelevant information?
4. Faithfulness – Does the candidate avoid hallucinated or unsupported claims?
5. Conciseness – Is the answer appropriately brief without losing key information?

IMPORTANT: A candidate that gives the right ranking/order but omits the actual \
numbers from the ground truth should score at most 3/5 on correctness and 2/5 on \
completeness. Raw data structures (lists, dicts, tuples) are NOT valid answers — \
they indicate the agent failed to produce a human-readable response.

You MUST respond with ONLY a JSON object (no markdown, no explanation). Example:
{"correctness": 4, "completeness": 3, "relevance": 5, "faithfulness": 4, "conciseness": 5, "correctness_reason": "Most numbers match.", "completeness_reason": "Missing one detail.", "relevance_reason": "Fully on topic.", "faithfulness_reason": "No hallucinations.", "conciseness_reason": "Well structured."}"""


def evaluate_answer(question: str, ground_truth: str, candidate: str) -> EvalScore:
    """Score a candidate answer against ground truth using a single LLM call."""
    client = get_llm_client()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Ground Truth:\n{ground_truth}\n\n"
                    f"Candidate Answer:\n{candidate}"
                ),
            },
        ],
        temperature=0,
        max_tokens=1024,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        parsed = _extract_json(raw)
    except json.JSONDecodeError:
        # Last resort: pull the first { ... } substring
        m = re.search(r"\{[\s\S]*\}", raw)
        parsed = json.loads(m.group()) if m else {}
    return EvalScore(**parsed)
