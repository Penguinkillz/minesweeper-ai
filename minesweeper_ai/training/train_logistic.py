"""Fit the logistic-regression model on the same rows and splits as the forest.

Run (from the repository root)::

    python -m minesweeper_ai.training.train_logistic
"""

from pathlib import Path

from minesweeper_ai.agents.forest_features import FEATURE_NAMES
from minesweeper_ai.agents.logistic import LogisticModel
from minesweeper_ai.training.forest_data import load_rows
from minesweeper_ai.training.train_forest import (
    INTERMEDIATE_WEIGHT,
    TRAIN_ROW_CAP,
    VALIDATION_SHARD,
    _stack,
    evaluate,
)

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "logistic.json"
METRICS_PATH = HERE / "logistic_metrics.txt"


def main() -> None:
    beginner_train = load_rows("beginner", "train")
    inter_train = load_rows("intermediate", "train", exclude=VALIDATION_SHARD)
    train_rows, train_labels = _stack([beginner_train], TRAIN_ROW_CAP, 0)
    inter_rows, inter_labels = _stack([inter_train], None, 0)
    train_rows += inter_rows * INTERMEDIATE_WEIGHT
    train_labels += inter_labels * INTERMEDIATE_WEIGHT
    model = LogisticModel().fit(train_rows, train_labels)
    model.save(MODEL_PATH)
    report = [f"train_rows={len(train_rows)} (intermediate x{INTERMEDIATE_WEIGHT}), same rows as the forest", ""]
    report.append("weights on standardised features:")
    for name, weight in sorted(zip(FEATURE_NAMES, model.weights), key=lambda item: -abs(item[1])):
        report.append(f"  {name:<20} {weight:+.3f}")
    report.append(f"  bias                 {model.bias:+.3f}")
    report.append("")
    report += evaluate(model, load_rows("beginner", "validation"), "beginner validation")  # type: ignore[arg-type]
    report += evaluate(model, load_rows("intermediate", "train", only=VALIDATION_SHARD), "intermediate validation")  # type: ignore[arg-type]
    for size in ("beginner", "intermediate"):
        report += evaluate(model, load_rows(size, "test"), f"{size} test")  # type: ignore[arg-type]
        report.append("")
    METRICS_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
