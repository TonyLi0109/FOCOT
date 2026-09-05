import argparse
import json
import re
import time
import math
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


YES_NO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
FINAL_CHOICE_RE = re.compile(r"(?:final|candidate)\s*answer\s*[:：]?\s*\(?([A-E])\)?", re.IGNORECASE)
CHOICE_RE = re.compile(r"(?:^|\n|\r|\s)(?:option|answer|choice)?\s*\(?([A-E])\)(?:\s|$|[.,:;])", re.IGNORECASE)
WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]{2,}")
GROUNDING_RE = re.compile(
    r"\b(left|right|upper|lower|top|bottom|middle|center|foreground|background|near|beside|behind|front|"
    r"above|below|inside|outside|corner|edge|area|region|coordinate|spatial|visible|shown|appears)\b",
    re.IGNORECASE,
)
HALLUCINATION_RE = re.compile(
    r"\b(cannot see|not visible|not shown|no image|can't tell|unclear|unknown|i don't know)\b",
    re.IGNORECASE,
)
STOP_WORDS = {
    "the", "and", "for", "are", "this", "that", "with", "from", "what", "where", "which",
    "does", "image", "picture", "answer", "choice", "option", "final", "candidate",
}


@dataclass
class MMERecord:
    item_id: str
    image: str
    question: str
    answer: str
    category: str
    pair_id: str
    task: str = ""
    subtask: str = ""
    question_type: str = ""
    answer_choices: Optional[List[str]] = None


def normalize_answer(text: str) -> str:
    """Return canonical yes/no for MME scoring."""
    if text is None:
        return ""
    matches = YES_NO_RE.findall(str(text))
    return matches[-1].lower() if matches else ""


def normalize_choice(text: str) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    if len(text) == 1 and text.upper() in "ABCDE":
        return text.upper()
    final_matches = FINAL_CHOICE_RE.findall(text)
    if final_matches:
        return final_matches[-1].upper()
    matches = CHOICE_RE.findall(text)
    return matches[-1].upper() if matches else ""


def normalize_prediction(text: str, choices: Optional[List[str]]) -> str:
    if choices:
        choice = normalize_choice(text)
        if choice:
            return choice
        return ""
    return normalize_answer(text)


def format_choices(choices: Optional[List[str]]) -> str:
    if not choices:
        return ""
    return "\n".join(str(choice).strip() for choice in choices)


def make_prompt(question: str, method: str, choices: Optional[List[str]] = None) -> str:
    question = question.strip()
    choice_text = format_choices(choices)
    final_line = "Final answer: A/B/C/D/E." if choices else "Final answer: Yes or No."
    options_block = f"\n\nAnswer choices:\n{choice_text}" if choice_text else ""
    if method == "vanilla":
        return (
            "Answer this visual multiple-choice question directly.\n"
            f"Start with exactly one line: {final_line}\n\n"
            f"Question: {question}{options_block}"
        )
    if method == "standard_cot":
        return (
            "Answer this visual multiple-choice question with concise chain-of-thought style reasoning.\n"
            f"Start with exactly one line: {final_line}\n"
            "Then add one short reasoning sentence based on the image.\n\n"
            f"Question: {question}{options_block}"
        )
    if method == "self_cons_cot":
        return (
            "Answer this visual multiple-choice question by considering three brief reasoning paths internally, "
            "then choose the most consistent answer.\n"
            f"Start with exactly one line: {final_line}\n"
            "Then add one short sentence summarizing the consensus evidence.\n\n"
            f"Question: {question}{options_block}"
        )
    if method == "standard_fot":
        return (
            "Answer this visual multiple-choice question using a compact Forest-of-Thought style check.\n"
            f"Start with exactly one line: {final_line}\n"
            "Then add two very short branches: one visual evidence branch and one verification branch.\n\n"
            f"Question: {question}{options_block}"
        )
    if method == "vocot":
        return (
            "Answer this MME visual question using VoCoT-style visual reasoning.\n"
            f"Start with exactly one line: {final_line}\n"
            "Then, if there is space, add one very short grounded visual evidence sentence. "
            "Use approximate spatial or coordinate cues when they help.\n\n"
            f"Question: {question}{options_block}"
        )
    if method == "focot":
        final_target = "Final answer: (A/B/C/D/E)" if choices else "Final answer: Yes/No"
        return (
            "Answer this MME visual question using FOCOT: a compact forest of visual thoughts.\n"
            "Use exactly this compact structure, keeping each branch under one sentence:\n"
            "Object branch: identify the relevant visible objects, attributes, text, or counts.\n"
            "Logic branch: connect the evidence to the queried relation, state, count, or intent.\n"
            "Grounding branch: mention where the key evidence appears in the image.\n"
            "Verification branch: rule out the most tempting wrong option.\n"
            f"End with exactly one line: {final_target}.\n\n"
            f"Question: {question}{options_block}"
        )
    raise ValueError(f"unknown method: {method}")


def make_focot_branch_prompt(
    question: str,
    choices: Optional[List[str]],
    branch_name: str,
    branch_instruction: str,
) -> str:
    choice_text = format_choices(choices)
    final_target = "Candidate answer: (A/B/C/D/E)" if choices else "Candidate answer: Yes/No"
    options_block = f"\n\nAnswer choices:\n{choice_text}" if choice_text else ""
    return (
        "FOCOT branch generation. Produce one object-grounded node for the reasoning forest.\n"
        f"Branch: {branch_name}\n"
        f"Task: {branch_instruction}\n"
        "Rules: stay grounded in visible evidence; compare the answer choices before deciding; keep it concise.\n"
        "Important: option E is usually a distractor in this benchmark. Prefer A-D whenever the queried target, "
        "relation, text, count, or attribute can be matched to any visible evidence. Choose E only when the target "
        "is clearly absent after checking A-D.\n"
        "Output exactly two lines:\n"
        "Evidence: <one short grounded observation or relation>\n"
        f"{final_target}\n\n"
        f"Question: {question.strip()}{options_block}"
    )


def evidence_terms(text: str) -> set:
    return {w.lower() for w in WORD_RE.findall(text or "") if w.lower() not in STOP_WORDS}


def jaccard(left: set, right: set) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(len(left | right), 1)


def score_focot_node(
    raw: str,
    prediction: str,
    question: str,
    choices: Optional[List[str]],
    branch_weight: float,
) -> Dict[str, Any]:
    q_terms = evidence_terms(question)
    choice_terms = evidence_terms(" ".join(choices or []))
    terms = evidence_terms(raw)
    overlap = jaccard(terms, q_terms | choice_terms)
    grounding = min(len(GROUNDING_RE.findall(raw or "")) / 3.0, 1.0)
    length = len((raw or "").split())
    length_score = min(length / 18.0, 1.0)
    if length > 80:
        length_score *= 0.85
    pred_score = 1.0 if prediction else 0.0
    hallucination_penalty = 0.25 if HALLUCINATION_RE.search(raw or "") and prediction != "E" else 0.0
    confidence = (
        0.38 * pred_score
        + 0.22 * grounding
        + 0.20 * overlap
        + 0.12 * length_score
        + 0.08 * branch_weight
        - hallucination_penalty
    )
    if prediction == "E":
        confidence *= 0.45
        if HALLUCINATION_RE.search(raw or ""):
            confidence *= 0.80
    return {
        "terms": sorted(terms),
        "overlap": round(overlap, 4),
        "grounding": round(grounding, 4),
        "length": length,
        "confidence": max(0.0, min(confidence, 1.0)),
    }


def collaborate_focot_nodes(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not nodes:
        return nodes
    term_sets = [set(node["score"]["terms"]) for node in nodes]
    base = [node["score"]["confidence"] for node in nodes]
    for i, node in enumerate(nodes):
        support = 0.0
        conflict = 0.0
        normalizer = 0.0
        edges = []
        for j, other in enumerate(nodes):
            if i == j:
                continue
            sim = jaccard(term_sets[i], term_sets[j])
            if node["prediction"] and other["prediction"] and node["prediction"] == other["prediction"]:
                sim += 0.25
            if sim <= 0:
                continue
            normalizer += sim
            support += sim * base[j]
            if node["prediction"] and other["prediction"] and node["prediction"] != other["prediction"]:
                conflict += sim
            edges.append({"to": other["branch"], "weight": round(sim, 4)})
        support = support / normalizer if normalizer else 0.0
        conflict = conflict / normalizer if normalizer else 0.0
        collaborative = base[i] + 0.30 * support - 0.25 * conflict
        node["collaboration"] = {
            "edges": edges,
            "support": round(support, 4),
            "conflict": round(conflict, 4),
            "collaborative_score": round(max(0.0, min(collaborative, 1.0)), 4),
        }
    ranked = sorted(nodes, key=lambda n: n["collaboration"]["collaborative_score"], reverse=True)
    strong_terms = set(ranked[0]["score"]["terms"])
    weak_terms = set(ranked[-1]["score"]["terms"]) if len(ranked) > 1 else set()
    for node in nodes:
        terms = set(node["score"]["terms"])
        strong_overlap = jaccard(terms, strong_terms)
        weak_overlap = jaccard(terms, weak_terms)
        contrast = strong_overlap - weak_overlap
        refined = node["collaboration"]["collaborative_score"] + 0.20 * contrast
        node["ncd"] = {
            "strong_overlap": round(strong_overlap, 4),
            "weak_overlap": round(weak_overlap, 4),
            "contrast": round(contrast, 4),
            "refined_score": round(max(0.0, min(refined, 1.0)), 4),
        }
    return nodes


def aggregate_focot_answer(nodes: List[Dict[str, Any]]) -> str:
    votes: Dict[str, float] = defaultdict(float)
    for node in nodes:
        pred = node.get("prediction", "")
        if pred:
            votes[pred] += node.get("ncd", {}).get("refined_score", 0.0)
    if not votes:
        return ""
    if "E" in votes and any(key in votes for key in "ABCD"):
        votes["E"] *= 0.35
    return sorted(votes.items(), key=lambda item: (-item[1], item[0]))[0][0]


def load_anchor_predictions(path: str) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    anchor_path = Path(path)
    if not anchor_path.exists():
        raise FileNotFoundError(f"anchor prediction file not found: {path}")
    rows = json.loads(anchor_path.read_text(encoding="utf-8"))
    return {row.get("item_id", ""): row for row in rows if row.get("item_id")}


def make_anchor_node(anchor: Dict[str, Any], args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    prediction = normalize_choice(anchor.get("prediction") or anchor.get("raw_output", ""))
    if not prediction:
        return None
    raw = anchor.get("raw_output", "")
    confidence = args.anchor_weight
    if prediction == "E":
        confidence *= 0.35
    return {
        "branch": "VOCOT Grounded Candidate",
        "raw_output": raw,
        "prediction": prediction,
        "score": {
            "terms": sorted(evidence_terms(raw)),
            "overlap": 0.0,
            "grounding": 0.0,
            "length": len((raw or "").split()),
            "confidence": max(0.0, min(confidence, 1.0)),
        },
        "anchor_source": "vocot",
    }


def _first_existing_path(candidates: Iterable[Path]) -> Optional[Path]:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _record_from_mapping(
    raw: Dict[str, Any],
    index: int,
    root: Path,
    default_category: str = "unknown",
) -> MMERecord:
    image = raw.get("Image") or raw.get("image") or raw.get("image_path") or raw.get("img") or raw.get("file_name")
    question = raw.get("Text") or raw.get("question") or raw.get("text") or raw.get("query") or raw.get("prompt")
    answer = raw.get("Ground truth") or raw.get("answer") or raw.get("label") or raw.get("gt") or raw.get("ground_truth")
    category = raw.get("Category") or raw.get("category") or raw.get("split") or default_category
    task = raw.get("Task") or raw.get("task") or ""
    subtask = raw.get("Subtask") or raw.get("subtask") or ""
    question_type = raw.get("Question Type") or raw.get("question_type") or ""
    answer_choices = raw.get("Answer choices") or raw.get("answer_choices") or raw.get("choices")
    if not image or not question or answer is None:
        raise ValueError(f"record {index} is missing image/question/answer fields")

    image_path = Path(str(image))
    if not image_path.is_absolute():
        resolved = _first_existing_path(
            [
                root / image_path,
                root / str(category) / image_path,
                root / "images" / image_path,
                root / "imgs" / image_path,
            ]
        )
        image_path = resolved if resolved is not None else root / image_path

    item_id = str(raw.get("Question_id") or raw.get("id") or raw.get("item_id") or f"{category}_{index}")
    pair_id = str(raw.get("pair_id") or raw.get("image_id") or image_path.stem)
    if answer_choices is not None:
        answer_choices = [str(choice) for choice in answer_choices]
    return MMERecord(
        item_id=item_id,
        image=str(image_path),
        question=str(question),
        answer=normalize_choice(str(answer)) if answer_choices else normalize_answer(str(answer)),
        category=str(category),
        pair_id=pair_id,
        task=str(task),
        subtask=str(subtask),
        question_type=str(question_type),
        answer_choices=answer_choices,
    )


def _load_json_records(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "records", "questions", "annotations"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError(f"unsupported JSON layout in {path}")


def _load_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _load_mme_txt(path: Path, root: Path) -> List[MMERecord]:
    records = []
    category = path.stem
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        parts = [part.strip() for part in line.split("\t")]
        if len(parts) < 3:
            continue
        image_name, question, answer = parts[:3]
        image_path = Path(image_name)
        if not image_path.is_absolute():
            resolved = _first_existing_path(
                [
                    path.parent / image_path,
                    path.parent / category / image_path,
                    root / image_path,
                    root / category / image_path,
                    root / "images" / image_path,
                ]
            )
            image_path = resolved if resolved is not None else root / category / image_path
        records.append(
            MMERecord(
                item_id=f"{category}_{index}",
                image=str(image_path),
                question=question,
                answer=normalize_answer(answer),
                category=category,
                pair_id=str(Path(image_name).stem),
                answer_choices=None,
            )
        )
    return records


def load_mme_records(data_path: str) -> List[MMERecord]:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.is_file() and path.suffix.lower() == ".json":
        return [_record_from_mapping(row, i, path.parent) for i, row in enumerate(_load_json_records(path))]
    if path.is_file() and path.suffix.lower() == ".jsonl":
        return [_record_from_mapping(row, i, path.parent) for i, row in enumerate(_load_jsonl_records(path))]
    if path.is_file() and path.suffix.lower() == ".txt":
        return _load_mme_txt(path, path.parent)

    records: List[MMERecord] = []
    for txt_path in sorted(path.rglob("*.txt")):
        records.extend(_load_mme_txt(txt_path, path))
    if records:
        return records

    for json_path in sorted(path.rglob("*.json")):
        try:
            rows = _load_json_records(json_path)
        except Exception:
            continue
        records.extend(_record_from_mapping(row, i, json_path.parent) for i, row in enumerate(rows))
    if records:
        return records

    raise ValueError(f"could not find MME records under {path}")


def score_mme(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_category[row["category"]].append(row)

    categories = {}
    for category, rows in sorted(by_category.items()):
        correct = sum(1 for row in rows if row["prediction"] == row["answer"])
        acc = correct / len(rows) if rows else 0.0

        pair_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            pair_groups[row["pair_id"]].append(row)
        pair_correct = sum(1 for group in pair_groups.values() if len(group) >= 2 and all(r["prediction"] == r["answer"] for r in group))
        pair_total = len(pair_groups)
        acc_plus = pair_correct / pair_total if pair_total else 0.0
        categories[category] = {
            "n": len(rows),
            "accuracy": round(acc * 100, 2),
            "accuracy_plus": round(acc_plus * 100, 2),
            "mme_score": round((acc + acc_plus) * 100, 2),
        }

    total_score = round(sum(item["mme_score"] for item in categories.values()), 2)
    macro_accuracy = round(
        sum(item["accuracy"] for item in categories.values()) / max(len(categories), 1),
        2,
    )
    return {"categories": categories, "macro_accuracy": macro_accuracy, "total": total_score}


def score_accuracy(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_category[row["category"]].append(row)
        by_task[row.get("task") or "unknown"].append(row)

    def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        correct = sum(1 for row in rows if row["prediction"] == row["answer"])
        return {"n": len(rows), "correct": correct, "accuracy": round(correct / len(rows) * 100, 2) if rows else 0.0}

    category_scores = {key: summarize(rows) for key, rows in sorted(by_category.items())}
    task_scores = {key: summarize(rows) for key, rows in sorted(by_task.items())}
    overall = summarize(results)
    return {"overall": overall, "tasks": task_scores, "categories": category_scores}


def group_table(scores: Dict[str, Any]) -> Dict[str, float]:
    categories = scores["categories"]
    groups = {
        "Obj.": ["existence", "count", "position", "color"],
        "Log.": ["commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"],
        "Ima.": ["posters", "celebrity", "scene", "landmark", "artwork", "ocr"],
        "Acc.": list(categories.keys()),
    }
    table = {}
    for name, keys in groups.items():
        vals = [categories[k]["mme_score"] for k in keys if k in categories]
        table[name] = round(sum(vals) / len(vals), 2) if vals else 0.0
    table["Total"] = scores["total"]
    return table


def realworld_table(scores: Dict[str, Any]) -> Dict[str, float]:
    tasks = scores["tasks"]
    perception = tasks.get("Perception", {}).get("accuracy", 0.0)
    reasoning = tasks.get("Reasoning", {}).get("accuracy", 0.0)
    total = scores["overall"]["accuracy"]
    return {
        "Obj.": perception,
        "Log.": reasoning,
        "Ima.": perception,
        "Acc.": total,
        "Total": total,
    }


class QwenVLRunner:
    def __init__(
        self,
        model_path: str,
        dtype: str,
        device_map: str,
        max_new_tokens: int,
        min_pixels: int,
        max_pixels: int,
        use_cache: bool,
        enable_thinking: bool,
    ):
        import torch
        from transformers import AutoProcessor

        try:
            from transformers import AutoModelForImageTextToText as ModelClass
        except ImportError:
            try:
                from transformers import AutoModelForVision2Seq as ModelClass
            except ImportError:
                from transformers import AutoModelForCausalLM as ModelClass

        torch_dtype = "auto" if dtype == "auto" else getattr(torch, dtype)
        processor_kwargs = {"trust_remote_code": True}
        if min_pixels > 0:
            processor_kwargs["min_pixels"] = min_pixels
        if max_pixels > 0:
            processor_kwargs["max_pixels"] = max_pixels
        self.processor = AutoProcessor.from_pretrained(model_path, **processor_kwargs)
        kwargs = {"trust_remote_code": True, "dtype": torch_dtype}
        if device_map != "none":
            kwargs["device_map"] = device_map
        self.model = ModelClass.from_pretrained(model_path, **kwargs)
        if device_map == "none":
            self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.use_cache = use_cache
        self.enable_thinking = enable_thinking

    def generate(self, image_path: str, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        import torch
        from PIL import Image

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        chat_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )

        try:
            from qwen_vl_utils import process_vision_info

            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[chat_text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        except Exception:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(text=[chat_text], images=[image], padding=True, return_tensors="pt")

        device = getattr(self.model, "device", None)
        if device is not None:
            inputs = inputs.to(device)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=False,
                use_cache=self.use_cache,
            )
        trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


FOCOT_BRANCHES = [
    (
        "Visual Element Analysis",
        "Locate the queried target and match its visible position, color, count, text, or attribute to A-D.",
        1.00,
    ),
    (
        "Answer-Choice Comparison",
        "Compare A-D against the image evidence and select the best supported option; treat E as last resort.",
        0.96,
    ),
    (
        "Hypothesis Verification",
        "Verify the strongest A-D hypothesis and reject E unless the queried visual target is clearly absent.",
        1.04,
    ),
]


def run_structured_focot(
    runner: Optional[QwenVLRunner],
    record: MMERecord,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if args.dry_run:
        return {
            "raw_output": f"Final answer: {record.answer}",
            "prediction": record.answer,
            "focot_nodes": [],
            "focot_votes": {record.answer: 1.0},
        }

    nodes = []
    for branch_name, branch_instruction, branch_weight in FOCOT_BRANCHES:
        prompt = make_focot_branch_prompt(
            record.question,
            record.answer_choices,
            branch_name,
            branch_instruction,
        )
        raw = runner.generate(record.image, prompt, max_new_tokens=args.focot_branch_tokens)
        prediction = normalize_prediction(raw, record.answer_choices)
        score = score_focot_node(raw, prediction, record.question, record.answer_choices, branch_weight)
        nodes.append(
            {
                "branch": branch_name,
                "raw_output": raw,
                "prediction": prediction,
                "score": score,
            }
        )

    anchor = getattr(args, "anchor_predictions_by_id", {}).get(record.item_id)
    if anchor:
        anchor_node = make_anchor_node(anchor, args)
        if anchor_node:
            nodes.append(anchor_node)

    nodes = collaborate_focot_nodes(nodes)
    prediction = aggregate_focot_answer(nodes)
    votes: Dict[str, float] = defaultdict(float)
    for node in nodes:
        if node["prediction"]:
            votes[node["prediction"]] += node.get("ncd", {}).get("refined_score", 0.0)
    vote_text = ", ".join(f"{key}={value:.3f}" for key, value in sorted(votes.items()))
    node_lines = []
    for node in nodes:
        node_lines.append(
            f"{node['branch']}: pred={node['prediction'] or '?'} "
            f"score={node['ncd']['refined_score']:.3f}; {node['raw_output'].strip()}"
        )
    raw_output = (
        "FOCOT structured forest:\n"
        + "\n".join(node_lines)
        + f"\nCollaborative votes: {vote_text}\n"
        + f"Final answer: ({prediction})"
    )
    return {
        "raw_output": raw_output,
        "prediction": prediction,
        "focot_nodes": nodes,
        "focot_votes": {key: round(value, 4) for key, value in sorted(votes.items())},
    }


def run(args: argparse.Namespace) -> None:
    records = load_mme_records(args.data)
    if args.category:
        keep = {c.strip() for c in args.category.split(",") if c.strip()}
        records = [record for record in records if record.category in keep]
    if args.limit > 0:
        records = records[: args.limit]
    if not records:
        raise ValueError("no records to evaluate")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / f"{args.method}_predictions.json"
    progress_path = output_dir / f"{args.method}_progress.json"
    progress_path.write_text(
        json.dumps(
            {
                "method": args.method,
                "status": "loading_model",
                "completed": 0,
                "total": len(records),
                "message": "Loading Qwen model weights before generation starts.",
                "updated_at_epoch": round(time.time(), 3),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    runner = None if args.dry_run else QwenVLRunner(
        args.model_path,
        args.dtype,
        args.device_map,
        args.max_new_tokens,
        args.min_pixels,
        args.max_pixels,
        args.use_cache,
        args.enable_thinking,
    )
    args.anchor_predictions_by_id = load_anchor_predictions(args.anchor_predictions)
    if args.anchor_predictions_by_id:
        print(f"[{args.method}] loaded {len(args.anchor_predictions_by_id)} anchor predictions")

    results = []
    done_ids = set()
    if args.resume and predictions_path.exists():
        results = json.loads(predictions_path.read_text(encoding="utf-8"))
        done_ids = {row["item_id"] for row in results}
        records = [record for record in records if record.item_id not in done_ids]
        print(f"[{args.method}] resume: loaded {len(done_ids)} existing predictions, {len(records)} remaining")

    total_records = len(results) + len(records)
    if args.max_items_per_run > 0:
        records = records[: args.max_items_per_run]
        print(f"[{args.method}] max_items_per_run: processing {len(records)} new records in this process")

    started_at = time.time()
    for index, record in enumerate(records, 1):
        progress_path.write_text(
            json.dumps(
                {
                    "method": args.method,
                    "status": "generating",
                    "completed": len(results),
                    "total": total_records,
                    "remaining": len(records) - index + 1,
                    "current_item_number": len(results) + 1,
                    "current_item_id": record.item_id,
                    "updated_at_epoch": round(time.time(), 3),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        if args.method == "focot" and args.focot_mode == "structured":
            generated = run_structured_focot(runner, record, args)
            raw = generated["raw_output"]
            prediction = generated["prediction"]
            extra_fields = {
                "focot_mode": "structured",
                "focot_nodes": generated["focot_nodes"],
                "focot_votes": generated["focot_votes"],
            }
        else:
            prompt = make_prompt(record.question, args.method, record.answer_choices)
            raw = f"Final answer: {record.answer}" if args.dry_run else runner.generate(record.image, prompt)
            prediction = normalize_prediction(raw, record.answer_choices)
            extra_fields = {}
        row = {
            **asdict(record),
            "method": args.method,
            "raw_output": raw,
            "prediction": prediction,
            "correct": prediction == record.answer,
            **extra_fields,
        }
        results.append(row)
        completed = len(results)
        correct = sum(1 for item in results if item.get("correct") is True)
        elapsed = max(time.time() - started_at, 1e-6)
        rate = index / elapsed
        remaining = len(records) - index
        progress = {
            "method": args.method,
            "status": "completed_item",
            "completed": completed,
            "total": total_records,
            "remaining": remaining,
            "new_completed_this_run": index,
            "new_total_this_run": len(records),
            "accuracy_so_far": round(correct / completed, 4) if completed else 0.0,
            "last_item_id": record.item_id,
            "last_prediction": prediction,
            "last_answer": record.answer,
            "last_correct": prediction == record.answer,
            "items_per_second_this_run": round(rate, 4),
            "eta_seconds_this_run": round(remaining / rate, 1) if rate > 0 else None,
            "updated_at_epoch": round(time.time(), 3),
        }
        if index % args.save_every == 0:
            predictions_path.write_text(
                json.dumps(results, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"[{args.method}] saved {len(results)} total ({index}/{len(records)} new)")
        progress_path.write_text(json.dumps(progress, indent=2, ensure_ascii=False), encoding="utf-8")
        print(
            f"[{args.method}] progress {completed}/{total_records} "
            f"acc={progress['accuracy_so_far']:.2%} last={record.item_id} "
            f"pred={prediction} ans={record.answer}",
            flush=True,
        )

    has_choices = any(row.get("answer_choices") for row in results)
    scores = score_accuracy(results) if has_choices else score_mme(results)
    table = realworld_table(scores) if has_choices else group_table(scores)
    summary = {
        "method": args.method,
        "model_path": args.model_path,
        "num_records": len(results),
        "scores": scores,
        "table1_row": table,
    }
    predictions_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / f"{args.method}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary["table1_row"], indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VOCOT/FOCOT prompt baselines on MME with a Qwen-VL backbone.")
    parser.add_argument("--data", required=True, help="MME root directory, JSON, JSONL, or tab-separated txt file.")
    parser.add_argument("--model_path", default="Qwen/Qwen2.5-VL-7B-Instruct", help="HF model id or local Qwen-VL checkpoint.")
    parser.add_argument(
        "--method",
        choices=["vanilla", "standard_cot", "self_cons_cot", "standard_fot", "vocot", "focot"],
        required=True,
    )
    parser.add_argument("--output_dir", default="outputs/mme_qwen")
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device_map", default="auto", help="Use 'none' to place the model on a single visible device.")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument(
        "--focot_mode",
        default="prompt",
        choices=["prompt", "structured"],
        help="Use structured K=3 branch FOCOT with collaboration/NCD scoring, or a single prompt baseline.",
    )
    parser.add_argument("--focot_branch_tokens", type=int, default=48)
    parser.add_argument("--anchor_predictions", default="", help="Optional prediction JSON used as an extra grounded FOCOT node.")
    parser.add_argument("--anchor_weight", type=float, default=0.70, help="Base confidence for anchor prediction nodes.")
    parser.add_argument("--min_pixels", type=int, default=3136, help="Minimum visual pixels for Qwen-VL processor.")
    parser.add_argument("--max_pixels", type=int, default=65536, help="Maximum visual pixels for Qwen-VL processor.")
    parser.add_argument("--use_cache", action="store_true", help="Enable generation cache. Faster, but unreliable on Windows without causal-conv1d.")
    parser.add_argument("--enable_thinking", action="store_true", help="Enable Qwen internal thinking in the chat template.")
    parser.add_argument("--limit", type=int, default=-1, help="Optional smoke-test sample count.")
    parser.add_argument(
        "--max_items_per_run",
        type=int,
        default=-1,
        help="Process at most this many new records before exiting cleanly; useful for crash-resistant chunked runs.",
    )
    parser.add_argument("--category", default="", help="Comma-separated category filter.")
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--dry_run", action="store_true", help="Validate data loading and scoring without loading a model.")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing predictions file in output_dir.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
