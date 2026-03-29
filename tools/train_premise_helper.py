from __future__ import annotations

import argparse
import inspect
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from huggingface_hub import snapshot_download
except ImportError:  # pragma: no cover - validated at runtime in main()
    snapshot_download = None

from taam.downstream.repair_helper import (
    build_pairwise_ranking_examples,
    build_helper_training_pairs,
    load_jsonl_rows,
    load_premise_corpus_from_inventory,
    write_jsonl_rows,
    write_premise_corpus,
)


MODEL_ALLOW_PATTERNS = [
    "config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.json",
    "pytorch_model.bin",
    "model.safetensors",
    "model.safetensors.index.json",
]
MODEL_WEIGHT_FILENAMES = ("pytorch_model.bin", "model.safetensors")


def _has_helper_weights(model_dir: Path) -> bool:
    return any((model_dir / filename).exists() for filename in MODEL_WEIGHT_FILENAMES)


def resolve_helper_model_dir(model_name_or_path: str) -> Path:
    candidate = Path(model_name_or_path)
    if candidate.exists():
        if not _has_helper_weights(candidate):
            raise SystemExit(f"Helper model directory is missing model weights: {candidate}")
        return candidate
    if snapshot_download is None:
        raise SystemExit("huggingface_hub is required to download the helper model")
    local_dir = Path(
        snapshot_download(
            repo_id=model_name_or_path,
            allow_patterns=MODEL_ALLOW_PATTERNS,
        )
    )
    if not _has_helper_weights(local_dir):
        raise SystemExit(f"Helper model snapshot is missing model weights: {local_dir}")
    return local_dir


class PairDataset:
    def __init__(self, rows: List[Dict[str, Any]], tokenizer, max_length: int) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        encoded = self.tokenizer(
            row["query"],
            row["premise_text"],
            truncation=True,
            max_length=self.max_length,
        )
        encoded["labels"] = int(row["label"])
        return encoded


class PairwiseRankingDataset:
    def __init__(self, rows: List[Dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.rows[idx]


class PairwiseRankingCollator:
    def __init__(self, tokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        positive_enc = self.tokenizer(
            [row["query"] for row in rows],
            [row["positive_premise_text"] for row in rows],
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        negative_enc = self.tokenizer(
            [row["query"] for row in rows],
            [row["negative_premise_text"] for row in rows],
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {"positive": positive_enc, "negative": negative_enc}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _compute_pair_metrics(eval_pred) -> Dict[str, float]:
    predictions, labels = eval_pred
    total = len(labels)
    if total == 0:
        return {"accuracy": 0.0}
    correct = 0
    for pred, label in zip(predictions, labels):
        if hasattr(pred, "tolist"):
            pred = pred.tolist()
        if isinstance(pred, list):
            predicted_label = 0 if len(pred) == 1 and pred[0] < 0.5 else (pred.index(max(pred)) if len(pred) > 1 else 0)
        else:
            predicted_label = int(pred > 0)
        if int(predicted_label) == int(label):
            correct += 1
    return {"accuracy": correct / total}


def _logits_to_scores(logits):
    import torch

    if logits.ndim == 1:
        return logits.reshape(-1)
    if logits.shape[-1] == 1:
        return logits.reshape(-1)
    return logits[:, 1] - logits[:, 0]


def compute_ranking_metrics(
    rows: Sequence[Dict[str, Any]],
    scores: Sequence[float],
    top_ks: Sequence[int],
) -> Dict[str, float]:
    groups: Dict[str, List[tuple[float, int]]] = defaultdict(list)
    for row, score in zip(rows, scores):
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            continue
        groups[sample_id].append((float(score), int(row.get("label", 0))))

    if not groups:
        metrics = {"mrr": 0.0, "pairwise_win_rate": 0.0}
        for k in sorted({int(k) for k in top_ks if int(k) > 0}):
            metrics[f"hit_at_{k}"] = 0.0
            metrics[f"recall_at_{k}"] = 0.0
        return metrics

    normalized_top_ks = sorted({int(k) for k in top_ks if int(k) > 0})
    hit_sums = {k: 0.0 for k in normalized_top_ks}
    recall_sums = {k: 0.0 for k in normalized_top_ks}
    mrr_sum = 0.0
    pairwise_wins = 0.0
    pairwise_total = 0.0

    for entries in groups.values():
        ranked = sorted(entries, key=lambda item: item[0], reverse=True)
        positives = [score for score, label in ranked if label == 1]
        negatives = [score for score, label in ranked if label == 0]
        positive_ranks = [index + 1 for index, (_score, label) in enumerate(ranked) if label == 1]
        if positive_ranks:
            mrr_sum += 1.0 / min(positive_ranks)
        positive_count = max(1, len(positive_ranks))
        for k in normalized_top_ks:
            top_labels = [label for _score, label in ranked[:k]]
            positive_hits = sum(1 for label in top_labels if label == 1)
            hit_sums[k] += 1.0 if positive_hits > 0 else 0.0
            recall_sums[k] += positive_hits / positive_count
        if positives and negatives:
            for pos_score in positives:
                for neg_score in negatives:
                    pairwise_total += 1.0
                    if pos_score > neg_score:
                        pairwise_wins += 1.0

    total_groups = float(len(groups))
    metrics = {
        "mrr": mrr_sum / total_groups,
        "pairwise_win_rate": (pairwise_wins / pairwise_total) if pairwise_total else 0.0,
    }
    for k in normalized_top_ks:
        metrics[f"hit_at_{k}"] = hit_sums[k] / total_groups
        metrics[f"recall_at_{k}"] = recall_sums[k] / total_groups
    return metrics


def is_better_ranking_checkpoint(
    current_metrics: Dict[str, float],
    best_metrics: Dict[str, float] | None,
    *,
    hint_top_k: int,
) -> bool:
    if not best_metrics:
        return True
    current_key = (
        float(current_metrics.get("mrr", 0.0)),
        float(current_metrics.get(f"recall_at_{hint_top_k}", 0.0)),
        float(current_metrics.get(f"hit_at_{hint_top_k}", 0.0)),
        float(current_metrics.get("pairwise_win_rate", 0.0)),
    )
    best_key = (
        float(best_metrics.get("mrr", 0.0)),
        float(best_metrics.get(f"recall_at_{hint_top_k}", 0.0)),
        float(best_metrics.get(f"hit_at_{hint_top_k}", 0.0)),
        float(best_metrics.get("pairwise_win_rate", 0.0)),
    )
    return current_key > best_key


def _mean_positive_rank_score(rows: List[Dict[str, Any]], scores: List[float]) -> float:
    positives: List[float] = []
    for row, score in zip(rows, scores):
        if int(row["label"]) == 1:
            positives.append(float(score))
    if not positives:
        return 0.0
    return sum(positives) / len(positives)


def _score_rows(model, tokenizer, rows: List[Dict[str, Any]], max_length: int, batch_size: int) -> List[float]:
    import torch

    device = next(model.parameters()).device
    scores: List[float] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        enc = tokenizer(
            [row["query"] for row in batch],
            [row["premise_text"] for row in batch],
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        batch_scores = _logits_to_scores(logits).tolist()
        scores.extend(float(value) for value in batch_scores)
    return scores


def _build_training_arguments(TrainingArguments, *, output_dir: str, args: argparse.Namespace, train_pairs: int):
    kwargs: Dict[str, Any] = {
        "output_dir": output_dir,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.train_num_epochs,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "load_best_model_at_end": True,
        "metric_for_best_model": "accuracy",
        "greater_is_better": True,
        "logging_strategy": "steps",
        "logging_steps": max(1, train_pairs // max(1, args.train_batch_size * 4)),
        "report_to": [],
        "seed": args.seed,
        "remove_unused_columns": False,
        "do_train": True,
        "do_eval": True,
    }
    supported = set(inspect.signature(TrainingArguments.__init__).parameters)
    if "evaluation_strategy" in supported:
        kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in supported:
        kwargs["eval_strategy"] = "epoch"
    if "save_strategy" in supported:
        kwargs["save_strategy"] = "epoch"
    if "overwrite_output_dir" in supported:
        kwargs["overwrite_output_dir"] = True
    if "save_total_limit" in supported:
        kwargs["save_total_limit"] = 1
    return TrainingArguments(**kwargs)


def _evaluate_ranking_rows(
    model,
    tokenizer,
    rows: List[Dict[str, Any]],
    *,
    max_length: int,
    batch_size: int,
    hint_top_k: int,
    rerank_top_n: int,
) -> Dict[str, float]:
    top_ks = tuple(k for k in (1, hint_top_k, rerank_top_n) if int(k) > 0)
    scores = _score_rows(model, tokenizer, rows, max_length=max_length, batch_size=batch_size)
    metrics = compute_ranking_metrics(rows, scores, top_ks=top_ks)
    metrics["mean_positive_score"] = _mean_positive_rank_score(rows, scores)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small premise reranker from TAAM hard samples")
    parser.add_argument("--train-jsonl", type=str, required=True)
    parser.add_argument("--eval-jsonl", type=str, required=True)
    parser.add_argument("--test-jsonl", type=str, required=True)
    parser.add_argument("--inventory-jsonl", type=str, required=True)
    parser.add_argument("--output-model-dir", type=str, required=True)
    parser.add_argument("--result-json", type=str, required=True)
    parser.add_argument("--mode", type=str, default="repair_helper_budget32")
    parser.add_argument("--helper-model-name", type=str, default="microsoft/codebert-base")
    parser.add_argument("--train-num-epochs", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--bm25-candidate-count", type=int, default=32)
    parser.add_argument("--rerank-top-n", type=int, default=8)
    parser.add_argument("--hint-top-k", type=int, default=8)
    parser.add_argument("--train-negatives-per-positive", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    try:
        import torch
        import torch.nn.functional as F
        from torch.optim import AdamW
        from torch.utils.data import DataLoader
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("transformers and torch are required to train the premise helper") from exc

    _set_seed(args.seed)

    train_rows = load_jsonl_rows(Path(args.train_jsonl))
    eval_rows = load_jsonl_rows(Path(args.eval_jsonl))
    test_rows = load_jsonl_rows(Path(args.test_jsonl))
    corpus = load_premise_corpus_from_inventory(Path(args.inventory_jsonl))
    if not corpus:
        raise SystemExit("Premise inventory is empty; cannot train premise helper")

    train_candidate_rows = build_helper_training_pairs(
        train_rows,
        corpus,
        bm25_candidate_count=args.bm25_candidate_count,
        rng=random.Random(args.seed),
    )
    eval_candidate_rows = build_helper_training_pairs(
        eval_rows,
        corpus,
        bm25_candidate_count=args.bm25_candidate_count,
        rng=random.Random(args.seed + 1),
    )
    test_candidate_rows = build_helper_training_pairs(
        test_rows,
        corpus,
        bm25_candidate_count=args.bm25_candidate_count,
        rng=random.Random(args.seed + 2),
    )
    if not train_candidate_rows or not eval_candidate_rows or not test_candidate_rows:
        raise SystemExit("Premise helper training pairs are empty for at least one split")

    train_pairwise_rows = build_pairwise_ranking_examples(
        train_candidate_rows,
        negatives_per_positive=args.train_negatives_per_positive,
        rng=random.Random(args.seed + 3),
    )
    if not train_pairwise_rows:
        raise SystemExit("Pairwise ranking examples are empty for training")

    output_model_dir = Path(args.output_model_dir)
    output_model_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_rows(train_candidate_rows, output_model_dir / "train_pairs.jsonl")
    write_jsonl_rows(eval_candidate_rows, output_model_dir / "eval_pairs.jsonl")
    write_jsonl_rows(test_candidate_rows, output_model_dir / "test_pairs.jsonl")
    write_jsonl_rows(train_pairwise_rows, output_model_dir / "train_pairwise.jsonl")
    write_premise_corpus(corpus, output_model_dir / "premise_corpus.jsonl")

    local_model_dir = resolve_helper_model_dir(args.helper_model_name)
    tokenizer = AutoTokenizer.from_pretrained(str(local_model_dir), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(local_model_dir),
        num_labels=2,
        local_files_only=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_dataset = PairwiseRankingDataset(train_pairwise_rows)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=PairwiseRankingCollator(tokenizer, max_length=args.max_length),
    )
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    train_history: List[Dict[str, float]] = []
    best_eval_metrics: Dict[str, float] | None = None
    best_epoch = 0

    for epoch in range(args.train_num_epochs):
        model.train()
        epoch_loss = 0.0
        batches = 0
        for batch in train_loader:
            positive_enc = {key: value.to(device) for key, value in batch["positive"].items()}
            negative_enc = {key: value.to(device) for key, value in batch["negative"].items()}
            positive_logits = model(**positive_enc).logits
            negative_logits = model(**negative_enc).logits
            positive_scores = _logits_to_scores(positive_logits)
            negative_scores = _logits_to_scores(negative_logits)
            loss = -F.logsigmoid(positive_scores - negative_scores).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().item())
            batches += 1

        eval_metrics = _evaluate_ranking_rows(
            model,
            tokenizer,
            eval_candidate_rows,
            max_length=args.max_length,
            batch_size=args.eval_batch_size,
            hint_top_k=args.hint_top_k,
            rerank_top_n=args.rerank_top_n,
        )
        train_history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": epoch_loss / max(1, batches),
                "eval_mrr": float(eval_metrics.get("mrr", 0.0)),
                f"eval_recall_at_{args.hint_top_k}": float(eval_metrics.get(f"recall_at_{args.hint_top_k}", 0.0)),
                "eval_pairwise_win_rate": float(eval_metrics.get("pairwise_win_rate", 0.0)),
            }
        )
        if is_better_ranking_checkpoint(eval_metrics, best_eval_metrics, hint_top_k=args.hint_top_k):
            best_eval_metrics = dict(eval_metrics)
            best_epoch = epoch + 1
            model.save_pretrained(str(output_model_dir))
            tokenizer.save_pretrained(str(output_model_dir))

    if best_eval_metrics is None:
        raise SystemExit("No ranking checkpoint was selected during helper training")

    best_model = AutoModelForSequenceClassification.from_pretrained(str(output_model_dir), local_files_only=True)
    best_model.to(device)
    best_model.eval()
    test_metrics = _evaluate_ranking_rows(
        best_model,
        tokenizer,
        test_candidate_rows,
        max_length=args.max_length,
        batch_size=args.eval_batch_size,
        hint_top_k=args.hint_top_k,
        rerank_top_n=args.rerank_top_n,
    )

    helper_manifest = {
        "success": True,
        "mode": args.mode,
        "model_name": args.helper_model_name,
        "local_model_dir": str(local_model_dir),
        "output_model_dir": str(output_model_dir),
        "selection_metric": "mrr",
        "best_epoch": best_epoch,
        "train_pairs": len(train_candidate_rows),
        "eval_pairs": len(eval_candidate_rows),
        "test_pairs": len(test_candidate_rows),
        "train_pairwise_examples": len(train_pairwise_rows),
        "premise_corpus_size": len(corpus),
        "bm25_candidate_count": args.bm25_candidate_count,
        "rerank_top_n": args.rerank_top_n,
        "hint_top_k": args.hint_top_k,
        "max_length": args.max_length,
        "train_negatives_per_positive": args.train_negatives_per_positive,
        "seed": args.seed,
        "train_history": train_history,
        "eval_metrics": dict(best_eval_metrics),
        "test_metrics": dict(test_metrics),
    }
    (output_model_dir / "helper_manifest.json").write_text(
        json.dumps(helper_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_model_dir / "bm25_config.json").write_text(
        json.dumps(
            {
                "bm25_candidate_count": args.bm25_candidate_count,
                "rerank_top_n": args.rerank_top_n,
                "hint_top_k": args.hint_top_k,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    Path(args.result_json).write_text(
        json.dumps(
            {
                "success": True,
                "output_model_dir": str(output_model_dir),
                "mode": args.mode,
                "local_model_dir": str(local_model_dir),
                "train_pairs": len(train_candidate_rows),
                "eval_pairs": len(eval_candidate_rows),
                "test_pairs": len(test_candidate_rows),
                "train_pairwise_examples": len(train_pairwise_rows),
                "best_epoch": best_epoch,
                "selection_metric": "mrr",
                "eval_mrr": float(best_eval_metrics.get("mrr", 0.0)),
                f"eval_recall_at_{args.hint_top_k}": float(best_eval_metrics.get(f"recall_at_{args.hint_top_k}", 0.0)),
                f"eval_hit_at_{args.hint_top_k}": float(best_eval_metrics.get(f"hit_at_{args.hint_top_k}", 0.0)),
                "eval_pairwise_win_rate": float(best_eval_metrics.get("pairwise_win_rate", 0.0)),
                "test_mrr": float(test_metrics.get("mrr", 0.0)),
                f"test_recall_at_{args.hint_top_k}": float(test_metrics.get(f"recall_at_{args.hint_top_k}", 0.0)),
                f"test_hit_at_{args.hint_top_k}": float(test_metrics.get(f"hit_at_{args.hint_top_k}", 0.0)),
                "test_pairwise_win_rate": float(test_metrics.get("pairwise_win_rate", 0.0)),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
