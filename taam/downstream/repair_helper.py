from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .downstream import load_hard_samples, split_records, write_jsonl


def _tokenize_text(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_'.]+", str(text).lower())


def _dedupe_preserving_order(values: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def load_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl_rows(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def hard_sample_to_helper_record(sample: Dict[str, Any]) -> Dict[str, Any]:
    positives = _dedupe_preserving_order(
        [str(sample.get("failed_on", "")).strip(), *[str(x) for x in sample.get("hidden_nodes", [])]]
    )
    negatives = [
        item
        for item in _dedupe_preserving_order([str(x) for x in sample.get("visible_nodes", [])])
        if item not in positives
    ]
    return {
        "sample_id": f'{sample.get("theorem_id", "unknown")}::{sample.get("target_id", "T")}::{sample.get("failed_on", "")}',
        "theorem_id": str(sample.get("theorem_id", "")).strip(),
        "target_id": str(sample.get("target_id", "")).strip(),
        "query": str(sample.get("lean_problem", "")),
        "positive_premises": sorted(positives),
        "negative_premises": negatives,
        "failed_on": str(sample.get("failed_on", "")).strip(),
        "hidden_nodes": list(sample.get("hidden_nodes", [])),
        "visible_nodes": list(sample.get("visible_nodes", [])),
        "theorem_domain": str(sample.get("theorem_domain", "")).strip(),
        "well_posed": sample.get("well_posed", None),
        "source_path": str(sample.get("_source_path", sample.get("source_path", ""))).strip(),
        "full_lean_problem": str(sample.get("full_lean_problem", "")).strip(),
        "proof_completion": str(sample.get("proof_completion", "")).strip(),
    }


def build_helper_dataset_records(
    samples: Sequence[Dict[str, Any]],
    only_well_posed: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    stats = {
        "total_samples": len(samples),
        "kept_samples": 0,
        "skipped_not_well_posed": 0,
        "skipped_missing_query": 0,
        "skipped_missing_positive_premises": 0,
    }
    for sample in samples:
        if only_well_posed and sample.get("well_posed") is not True:
            stats["skipped_not_well_posed"] += 1
            continue
        record = hard_sample_to_helper_record(sample)
        if not record["query"]:
            stats["skipped_missing_query"] += 1
            continue
        if not record["positive_premises"]:
            stats["skipped_missing_positive_premises"] += 1
            continue
        records.append(record)
    stats["kept_samples"] = len(records)
    return records, stats


def export_helper_dataset_bundle(
    samples_root: Path,
    sample_glob: str,
    out_dir: Path,
    only_well_posed: bool,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, Any]:
    samples = load_hard_samples(samples_root, sample_glob)
    records, stats = build_helper_dataset_records(samples, only_well_posed=only_well_posed)
    splits = split_records(records, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_paths: Dict[str, str] = {}
    for split_name, split_rows in splits.items():
        split_path = out_dir / f"{split_name}.jsonl"
        write_jsonl(split_rows, split_path)
        split_paths[split_name] = str(split_path)
    manifest = {
        "source_type": "hard_samples",
        "dataset_format": "premise_helper",
        "samples_root": str(samples_root),
        "sample_glob": sample_glob,
        "seed": seed,
        "split_paths": split_paths,
        "split_sizes": {name: len(rows) for name, rows in splits.items()},
        "stats": stats,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def build_premise_corpus(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    corpus: List[Dict[str, Any]] = []
    for row in rows:
        premise_id = str(row.get("theorem_id", "")).strip()
        target_statement = str(row.get("target_statement", "")).strip()
        if not premise_id or not target_statement:
            continue
        corpus.append(
            {
                "premise_id": premise_id,
                "target_statement": target_statement,
                "source_file_path": str(row.get("file_path", "")).strip(),
                "text": f"{premise_id} : {target_statement}",
                "tokens": _tokenize_text(f"{premise_id} {target_statement}"),
            }
        )
    return corpus


def write_premise_corpus(corpus: Sequence[Dict[str, Any]], path: Path) -> None:
    write_jsonl_rows(corpus, path)


def _build_corpus_map(corpus: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    corpus_map: Dict[str, Dict[str, Any]] = {}
    for row in corpus:
        premise_id = str(row.get("premise_id", "")).strip()
        if premise_id:
            corpus_map[premise_id] = row
    return corpus_map


def _copy_hit(row: Dict[str, Any], *, source: str, score: float | None = None) -> Dict[str, Any]:
    item = dict(row)
    item["source"] = source
    if score is not None:
        item["score"] = float(score)
    return item


def _oracle_positive_premise_ids(task_metadata: Dict[str, Any]) -> List[str]:
    positives = task_metadata.get("positive_premises", [])
    if isinstance(positives, list) and positives:
        return _dedupe_preserving_order([str(item) for item in positives])
    fallback = [
        str(task_metadata.get("failed_on", "")).strip(),
        *[str(item) for item in task_metadata.get("hidden_nodes", [])],
    ]
    return _dedupe_preserving_order(fallback)


def render_premise_hint_block(premises: Sequence[Dict[str, Any]]) -> str:
    lines = ["/- TAAM premise hints:"]
    for premise in premises:
        lines.append(f"{premise['premise_id']} : {premise['target_statement']}")
    lines.append("-/")
    return "\n".join(lines) + "\n\n"


def inject_premise_hints(lean_problem: str, premises: Sequence[Dict[str, Any]]) -> str:
    if not premises:
        return lean_problem
    hint_block = render_premise_hint_block(premises)
    lines = lean_problem.splitlines(keepends=True)
    insert_at = 0
    for idx, line in enumerate(lines):
        if line.startswith("import "):
            insert_at = idx + 1
            continue
        if insert_at and not line.strip():
            insert_at = idx + 1
            continue
        break
    rebuilt = "".join(lines[:insert_at]) + hint_block + "".join(lines[insert_at:])
    return rebuilt


def select_best_budget_schedule(schedule_candidates: Sequence[str], schedule_scores: Dict[str, Dict[str, Any]]) -> str:
    best_name = ""
    best_score = (-1, -1.0)
    for name in schedule_candidates:
        stats = schedule_scores.get(name, {})
        successes = int(stats.get("successes", -1))
        pass_rate = float(stats.get("pass_rate", 0.0))
        score = (successes, pass_rate)
        if score > best_score:
            best_name = name
            best_score = score
    if not best_name:
        raise ValueError("No valid budget schedule scores were provided")
    return best_name


def is_algebra_like_task_name(task_name: str) -> bool:
    lowered = str(task_name).lower()
    return any(token in lowered for token in ("algebra", "polynomial", "ring", "field"))


class BM25Index:
    def __init__(self, documents: Sequence[Dict[str, Any]], k1: float = 1.5, b: float = 0.75) -> None:
        self.documents = list(documents)
        self.k1 = k1
        self.b = b
        self.doc_freqs: List[Counter[str]] = [Counter(doc["tokens"]) for doc in self.documents]
        self.doc_lengths: List[int] = [sum(freq.values()) for freq in self.doc_freqs]
        self.avgdl = (sum(self.doc_lengths) / len(self.doc_lengths)) if self.doc_lengths else 0.0
        self.idf: Dict[str, float] = {}
        token_docs = Counter()
        for freqs in self.doc_freqs:
            for token in freqs:
                token_docs[token] += 1
        total_docs = len(self.documents)
        for token, doc_count in token_docs.items():
            self.idf[token] = math.log(1 + (total_docs - doc_count + 0.5) / (doc_count + 0.5))

    def score(self, query: str) -> List[tuple[float, Dict[str, Any]]]:
        query_tokens = _tokenize_text(query)
        results: List[tuple[float, Dict[str, Any]]] = []
        for doc, freqs, dl in zip(self.documents, self.doc_freqs, self.doc_lengths):
            score = 0.0
            for token in query_tokens:
                if token not in freqs:
                    continue
                idf = self.idf.get(token, 0.0)
                tf = freqs[token]
                denom = tf + self.k1 * (1 - self.b + self.b * (dl / self.avgdl if self.avgdl else 0.0))
                score += idf * ((tf * (self.k1 + 1)) / denom if denom else 0.0)
            results.append((score, doc))
        results.sort(key=lambda item: item[0], reverse=True)
        return results

    def top_k(self, query: str, k: int) -> List[Dict[str, Any]]:
        return [doc for score, doc in self.score(query)[:k] if score > 0]


def build_helper_training_pairs(
    helper_rows: Sequence[Dict[str, Any]],
    corpus: Sequence[Dict[str, Any]],
    bm25_candidate_count: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    corpus_map = {row["premise_id"]: row for row in corpus}
    bm25 = BM25Index(corpus)
    pairs: List[Dict[str, Any]] = []
    corpus_ids = list(corpus_map.keys())
    for row in helper_rows:
        query = row["query"]
        positives = [pid for pid in row.get("positive_premises", []) if pid in corpus_map]
        negatives = [pid for pid in row.get("negative_premises", []) if pid in corpus_map and pid not in positives]
        confusers = [
            item["premise_id"]
            for item in bm25.top_k(query, bm25_candidate_count)
            if item["premise_id"] not in positives and item["premise_id"] not in negatives
        ]
        same_domain_pool = [
            pid
            for pid in corpus_ids
            if pid not in positives and pid not in negatives and pid not in confusers
        ]
        sampled_random = rng.sample(same_domain_pool, k=min(4, len(same_domain_pool)))
        for pid in positives:
            pairs.append(
                {
                    "sample_id": row.get("sample_id", ""),
                    "query": query,
                    "premise_text": corpus_map[pid]["text"],
                    "label": 1,
                    "premise_id": pid,
                }
            )
        for pid in _dedupe_preserving_order([*negatives, *confusers, *sampled_random]):
            pairs.append(
                {
                    "sample_id": row.get("sample_id", ""),
                    "query": query,
                    "premise_text": corpus_map[pid]["text"],
                    "label": 0,
                    "premise_id": pid,
                }
            )
    return pairs


def build_pairwise_ranking_examples(
    candidate_rows: Sequence[Dict[str, Any]],
    negatives_per_positive: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for row in candidate_rows:
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            continue
        group = grouped.setdefault(
            sample_id,
            {
                "query": str(row.get("query", "")),
                "positives": [],
                "negatives": [],
            },
        )
        label = int(row.get("label", 0))
        item = {
            "premise_id": str(row.get("premise_id", "")).strip(),
            "premise_text": str(row.get("premise_text", "")),
        }
        if label == 1:
            group["positives"].append(item)
        else:
            group["negatives"].append(item)

    pairwise_rows: List[Dict[str, Any]] = []
    for sample_id, group in grouped.items():
        positives = list(group["positives"])
        negatives = list(group["negatives"])
        if not positives or not negatives:
            continue
        for positive in positives:
            sampled_negatives = list(negatives)
            rng.shuffle(sampled_negatives)
            if negatives_per_positive > 0:
                sampled_negatives = sampled_negatives[: min(negatives_per_positive, len(sampled_negatives))]
            for negative in sampled_negatives:
                pairwise_rows.append(
                    {
                        "sample_id": sample_id,
                        "query": group["query"],
                        "positive_premise_id": positive["premise_id"],
                        "positive_premise_text": positive["premise_text"],
                        "negative_premise_id": negative["premise_id"],
                        "negative_premise_text": negative["premise_text"],
                    }
                )
    return pairwise_rows


def load_premise_corpus_from_inventory(inventory_jsonl: Path) -> List[Dict[str, Any]]:
    return build_premise_corpus(load_jsonl_rows(inventory_jsonl))


def load_helper_metadata(helper_model_dir: Path) -> Dict[str, Any]:
    metadata_path = helper_model_dir / "helper_manifest.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def load_helper_corpus(helper_model_dir: Path | None, inventory_jsonl: Path | None = None) -> List[Dict[str, Any]]:
    if helper_model_dir is not None:
        corpus_path = helper_model_dir / "premise_corpus.jsonl"
        if corpus_path.exists():
            return load_jsonl_rows(corpus_path)
    if inventory_jsonl is None or not str(inventory_jsonl):
        raise FileNotFoundError(f"No premise corpus found under {helper_model_dir}")
    return load_premise_corpus_from_inventory(inventory_jsonl)


class PremiseReranker:
    def __init__(self, model_dir: Path, max_length: int = 512, batch_size: int = 8, device: str = "cpu") -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_dir = model_dir
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = torch.device(device or "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), local_files_only=True)
        self.model.to(self.device)
        self.model.eval()
        self._torch = torch

    def score(self, query: str, candidates: Sequence[Dict[str, Any]]) -> List[float]:
        if not candidates:
            return []
        scores: List[float] = []
        for start in range(0, len(candidates), self.batch_size):
            batch = candidates[start : start + self.batch_size]
            enc = self.tokenizer(
                [query] * len(batch),
                [item["text"] for item in batch],
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with self._torch.no_grad():
                logits = self.model(**enc).logits
            if logits.ndim == 1 or logits.shape[-1] == 1:
                batch_scores = logits.reshape(-1).tolist()
            else:
                probs = self._torch.softmax(logits, dim=-1)[:, 1]
                batch_scores = probs.tolist()
            scores.extend(float(value) for value in batch_scores)
        return scores


def retrieve_premise_hits(
    query: str,
    bm25: BM25Index,
    reranker: PremiseReranker | None,
    bm25_candidate_count: int,
    rerank_top_n: int,
    hint_top_k: int,
) -> List[Dict[str, Any]]:
    candidates = bm25.top_k(query, bm25_candidate_count)
    if not candidates:
        return []
    if reranker is None:
        return candidates[:hint_top_k]
    scores = reranker.score(query, candidates)
    ranked = []
    for score, candidate in zip(scores, candidates):
        item = dict(candidate)
        item["score"] = float(score)
        ranked.append(item)
    ranked.sort(key=lambda row: row["score"], reverse=True)
    top_ranked = ranked[: max(1, rerank_top_n)]
    return top_ranked[:hint_top_k]


def select_hint_premises(
    *,
    query: str,
    task_metadata: Dict[str, Any],
    corpus: Sequence[Dict[str, Any]],
    bm25: BM25Index,
    reranker: PremiseReranker | None,
    hint_source: str,
    bm25_candidate_count: int,
    rerank_top_n: int,
    hint_top_k: int,
    rng: random.Random | None = None,
) -> List[Dict[str, Any]]:
    mode = str(hint_source or "learned").strip().lower()
    if hint_top_k <= 0 or not corpus:
        return []

    if mode == "oracle":
        corpus_map = _build_corpus_map(corpus)
        hits: List[Dict[str, Any]] = []
        for premise_id in _oracle_positive_premise_ids(task_metadata):
            row = corpus_map.get(premise_id)
            if row is not None:
                hits.append(_copy_hit(row, source="oracle", score=1.0))
            if len(hits) >= hint_top_k:
                break
        return hits

    if mode == "random":
        random_gen = rng if rng is not None else random.Random(0)
        sample_size = min(hint_top_k, len(corpus))
        sampled = random_gen.sample(list(corpus), k=sample_size)
        return [_copy_hit(row, source="random", score=0.0) for row in sampled]

    if mode == "bm25":
        hits = retrieve_premise_hits(
            query,
            bm25=bm25,
            reranker=None,
            bm25_candidate_count=bm25_candidate_count,
            rerank_top_n=rerank_top_n,
            hint_top_k=hint_top_k,
        )
        return [_copy_hit(row, source="bm25", score=row.get("score", 0.0)) for row in hits]

    if mode != "learned":
        raise ValueError(f"Unsupported hint_source={hint_source}")

    hits = retrieve_premise_hits(
        query,
        bm25=bm25,
        reranker=reranker,
        bm25_candidate_count=bm25_candidate_count,
        rerank_top_n=rerank_top_n,
        hint_top_k=hint_top_k,
    )
    return [_copy_hit(row, source="learned", score=row.get("score", 0.0)) for row in hits]
