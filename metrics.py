import re
import math
from collections import Counter
from typing import Union


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'["""]', '"', text)
    text = re.sub(r"[''']", "'", text)
    text = re.sub(r'[–—]', '-', text)
    return text


def exact_match(prediction: str, accepted_answers: list[str]) -> float:
    pred = normalize_text(prediction)
    answers = [normalize_text(a) for a in accepted_answers]
    return 1.0 if pred in answers else 0.0


def contains_match(prediction: str, accepted_answers: list[str]) -> float:
    pred = normalize_text(prediction)
    for ans in accepted_answers:
        if normalize_text(ans) in pred:
            return 1.0
    return 0.0


def anls(prediction: str, ground_truth: str, threshold: float = 0.5) -> float:
    pred = normalize_text(prediction)
    gt = normalize_text(ground_truth)
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0
    dist = levenshtein_distance(pred, gt)
    max_len = max(len(pred), len(gt))
    similarity = 1.0 - (dist / max_len)
    return similarity if similarity >= threshold else 0.0


def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


def edit_distance_similarity(prediction: str, ground_truth: str) -> float:
    pred = normalize_text(prediction)
    gt = normalize_text(ground_truth)
    if not pred and not gt:
        return 1.0
    dist = levenshtein_distance(pred, gt)
    max_len = max(len(pred), len(gt))
    return 1.0 - (dist / max_len)


def _ngrams(text: str, n: int) -> list[tuple]:
    tokens = text.split()
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def bleu_score(prediction: str, references: list[str], max_n: int = 4) -> float:
    pred = normalize_text(prediction)
    refs = [normalize_text(r) for r in references]
    if not pred.split():
        return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        pred_ngrams = Counter(_ngrams(pred, n))
        if not pred_ngrams:
            precisions.append(0.0)
            continue
        max_ref_counts = Counter()
        for ref in refs:
            ref_ngrams = Counter(_ngrams(ref, n))
            for ng, count in ref_ngrams.items():
                max_ref_counts[ng] = max(max_ref_counts[ng], count)
        clipped = {ng: min(count, max_ref_counts.get(ng, 0)) for ng, count in pred_ngrams.items()}
        total = sum(pred_ngrams.values())
        matched = sum(clipped.values())
        precisions.append(matched / total if total > 0 else 0.0)
    if all(p == 0 for p in precisions):
        return 0.0
    log_avg = sum(math.log(max(p, 1e-10)) for p in precisions) / max_n
    pred_len = len(pred.split())
    ref_lens = [len(r.split()) for r in refs]
    closest_ref_len = min(ref_lens, key=lambda rl: abs(rl - pred_len))
    bp = 1.0 if pred_len > closest_ref_len else math.exp(1 - closest_ref_len / max(pred_len, 1))
    return bp * math.exp(log_avg)


def rouge_l(prediction: str, reference: str) -> float:
    pred = normalize_text(prediction).split()
    ref = normalize_text(reference).split()
    if not pred or not ref:
        return 0.0
    lcs_len = _lcs_length(pred, ref)
    precision = lcs_len / len(pred)
    recall = lcs_len / len(ref)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _lcs_length(seq1: list, seq2: list) -> int:
    m, n = len(seq1), len(seq2)
    if m < n:
        return _lcs_length(seq2, seq1)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(curr[j-1], prev[j])
        prev = curr
    return prev[n]


def field_extraction_accuracy(
    predicted_fields: dict[str, str],
    expected_fields: dict[str, str],
    accepted_fields: list[str] | None = None,
) -> dict:
    fields_to_check = accepted_fields or list(expected_fields.keys())
    correct = 0
    total = len(fields_to_check)
    field_scores = {}
    for field in fields_to_check:
        expected = expected_fields.get(field, "")
        predicted = predicted_fields.get(field, "")
        score = anls(predicted, expected)
        field_scores[field] = score
        correct += score
    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "field_scores": field_scores,
        "total_fields": total,
        "correct_fields": correct,
    }


def compute_all_metrics(
    prediction: str,
    expected_answer: Union[str, dict],
    accepted_answers: list[str] | None = None,
    references: list[str] | None = None,
    task_type: str = "qa",
) -> dict:
    if task_type == "form_extraction" and isinstance(expected_answer, dict):
        predicted_fields = _parse_fields_from_text(prediction)
        return field_extraction_accuracy(predicted_fields, expected_answer)

    pred = prediction
    if isinstance(expected_answer, str):
        expected = expected_answer
    else:
        expected = str(expected_answer)

    if accepted_answers is None:
        accepted_answers = [expected]
    if references is None:
        references = [expected]

    return {
        "exact_match": exact_match(pred, accepted_answers),
        "contains_match": contains_match(pred, accepted_answers),
        "anls": anls(pred, expected),
        "edit_distance_similarity": edit_distance_similarity(pred, expected),
        "bleu": bleu_score(pred, references),
        "rouge_l": max(rouge_l(pred, ref) for ref in references),
    }


def _parse_fields_from_text(text: str) -> dict[str, str]:
    fields = {}
    for line in text.strip().split('\n'):
        line = line.strip()
        if ':' in line:
            key, _, value = line.partition(':')
            fields[key.strip()] = value.strip()
        elif '=' in line:
            key, _, value = line.partition('=')
            fields[key.strip()] = value.strip()
    if not fields and text.strip():
        fields["raw"] = text.strip()
    return fields


def aggregate_metrics(results: list[dict]) -> dict:
    if not results:
        return {}
    all_keys = set()
    for r in results:
        all_keys.update(k for k in r if isinstance(r[k], (int, float)))
    aggregated = {}
    for key in sorted(all_keys):
        values = [r[key] for r in results if key in r and isinstance(r[key], (int, float))]
        if values:
            aggregated[key] = {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "count": len(values),
            }
    return aggregated
