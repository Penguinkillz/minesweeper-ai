"""Logistic regression: one sigmoid on standardized features.

``p(mine | x) = sigmoid(b + sum_i w_i * (x_i - mean_i) / std_i)``.

The score is a probability, not a proof. ``predict`` matches
``RandomForest.predict``, so the forest players can use this model.
"""

import json
import math
import random
from collections.abc import Sequence
from pathlib import Path


def _sigmoid(value: float) -> float:
    value = max(-30.0, min(30.0, value))
    return 1.0 / (1.0 + math.exp(-value))


class LogisticModel:
    def __init__(self, l2: float = 1e-4, lr: float = 0.3, epochs: int = 15, batch: int = 256, seed: int = 0) -> None:
        self.l2, self.lr, self.epochs, self.batch, self.seed = l2, lr, epochs, batch, seed
        self.mean: list[float] = []
        self.std: list[float] = []
        self.weights: list[float] = []
        self.bias = 0.0

    def fit(self, rows: Sequence[Sequence[float]], labels: Sequence[int]) -> "LogisticModel":
        n, d = len(rows), len(rows[0])
        self.mean = [sum(r[i] for r in rows) / n for i in range(d)]
        self.std = [
            math.sqrt(sum((r[i] - self.mean[i]) ** 2 for r in rows) / n) or 1.0 for i in range(d)
        ]
        scaled = [[(r[i] - self.mean[i]) / self.std[i] for i in range(d)] for r in rows]
        self.weights = [0.0] * d
        self.bias = 0.0
        rng = random.Random(self.seed)
        order = list(range(n))
        for epoch in range(self.epochs):
            rng.shuffle(order)
            step = self.lr / (1.0 + 0.2 * epoch)
            for start in range(0, n, self.batch):
                chunk = order[start : start + self.batch]
                grad = [0.0] * d
                grad_b = 0.0
                for index in chunk:
                    x = scaled[index]
                    error = _sigmoid(self.bias + sum(w * v for w, v in zip(self.weights, x))) - labels[index]
                    grad_b += error
                    for i in range(d):
                        grad[i] += error * x[i]
                scale = step / len(chunk)
                self.bias -= scale * grad_b
                for i in range(d):
                    self.weights[i] -= scale * grad[i] + step * self.l2 * self.weights[i]
        return self

    def predict(self, row: Sequence[float]) -> float:
        total = self.bias
        for w, m, s, v in zip(self.weights, self.mean, self.std, row):
            total += w * (v - m) / s
        return _sigmoid(total)

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps({"mean": self.mean, "std": self.std, "weights": self.weights, "bias": self.bias}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "LogisticModel":
        data = json.loads(path.read_text(encoding="utf-8"))
        model = cls()
        model.mean, model.std = data["mean"], data["std"]
        model.weights, model.bias = data["weights"], data["bias"]
        return model
