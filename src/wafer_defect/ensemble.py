"""Utilities for multiclass classifier ensembles and stacking."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from wafer_defect.data.supervised import DEFAULT_CLASS_NAMES
from wafer_defect.models.classifier import WaferClassifier


SUPPORTED_STACKING_FEATURE_TYPES = ("probabilities", "logits")
SUPPORTED_STACKING_SELECTION_METRICS = ("accuracy", "balanced_accuracy")


def _build_classifier_from_checkpoint(checkpoint: dict[str, object], class_names: list[str]) -> WaferClassifier:
    model_cfg = dict(checkpoint["model_config"])
    return WaferClassifier(
        num_classes=len(class_names),
        base_channels=int(model_cfg["base_channels"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        dropout=float(model_cfg["dropout"]),
        variant=str(model_cfg.get("variant", "baseline")),
        block_dropout=float(model_cfg.get("block_dropout", 0.0)),
        se_reduction=int(model_cfg.get("se_reduction", 8)),
    )


def load_classifier_models(
    checkpoint_paths: list[Path],
    device: torch.device,
) -> tuple[list[WaferClassifier], list[str]]:
    models: list[WaferClassifier] = []
    class_names: list[str] | None = None

    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        current_class_names = list(checkpoint.get("class_names", DEFAULT_CLASS_NAMES))
        if class_names is None:
            class_names = current_class_names
        elif current_class_names != class_names:
            raise ValueError(f"Checkpoint class-name mismatch for {checkpoint_path}")

        model = _build_classifier_from_checkpoint(checkpoint, current_class_names)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()
        models.append(model)

    if class_names is None:
        raise ValueError("No checkpoints were loaded")
    return models, class_names


@torch.no_grad()
def collect_model_outputs(
    models: list[WaferClassifier],
    inputs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits_per_model = []
    probabilities_per_model = []
    for model in models:
        logits = model(inputs)
        logits_per_model.append(logits)
        probabilities_per_model.append(torch.softmax(logits, dim=1))

    logits = torch.stack(logits_per_model, dim=1)
    probabilities = torch.stack(probabilities_per_model, dim=1)
    return logits, probabilities


@torch.no_grad()
def average_probabilities(
    models: list[WaferClassifier],
    inputs: torch.Tensor,
) -> torch.Tensor:
    _, probabilities = collect_model_outputs(models, inputs)
    return probabilities.mean(dim=1)


def build_stacking_features(
    model_outputs: np.ndarray | torch.Tensor,
    feature_type: str,
) -> np.ndarray:
    if feature_type not in SUPPORTED_STACKING_FEATURE_TYPES:
        raise ValueError(
            f"Unsupported stacking feature type '{feature_type}'. "
            f"Choose from {SUPPORTED_STACKING_FEATURE_TYPES}."
        )

    if isinstance(model_outputs, torch.Tensor):
        output_array = model_outputs.detach().cpu().numpy()
    else:
        output_array = np.asarray(model_outputs)

    if output_array.ndim != 3:
        raise ValueError(f"Expected model outputs with shape (n_samples, n_models, n_classes), got {output_array.shape}")

    return output_array.reshape(output_array.shape[0], -1).astype(np.float32, copy=False)


def _fit_standardization(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = features.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    return mean, scale


def _apply_standardization(features: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((features - mean) / scale).astype(np.float32, copy=False)


def _softmax(scores: np.ndarray) -> np.ndarray:
    centered = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(centered)
    return exp_scores / np.clip(exp_scores.sum(axis=1, keepdims=True), 1e-12, None)


def _make_logistic_regression(
    C: float,
    max_iter: int,
    class_weight: str | None,
) -> LogisticRegression:
    return LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver="lbfgs",
        class_weight=class_weight,
    )


def _score_predictions(
    targets: np.ndarray,
    predictions: np.ndarray,
    selection_metric: str,
) -> float:
    if selection_metric == "accuracy":
        return float(accuracy_score(targets, predictions))
    if selection_metric == "balanced_accuracy":
        return float(balanced_accuracy_score(targets, predictions))
    raise ValueError(
        f"Unsupported stacking selection metric '{selection_metric}'. "
        f"Choose from {SUPPORTED_STACKING_SELECTION_METRICS}."
    )


def select_stacking_configuration(
    *,
    outputs_by_type: dict[str, np.ndarray],
    targets: np.ndarray,
    feature_types: list[str],
    c_grid: list[float],
    cv_folds: int,
    max_iter: int,
    random_seed: int,
    selection_metric: str = "balanced_accuracy",
    class_weight_options: list[str | None] | None = None,
) -> dict[str, object]:
    if cv_folds < 2:
        raise ValueError(f"cv_folds must be at least 2, got {cv_folds}")

    class_weight_options = class_weight_options or [None, "balanced"]
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)

    best_result: dict[str, object] | None = None
    results: list[dict[str, object]] = []

    for feature_type in feature_types:
        if feature_type not in outputs_by_type:
            raise KeyError(f"Missing model outputs for feature type '{feature_type}'")
        features = build_stacking_features(outputs_by_type[feature_type], feature_type=feature_type)

        for c_value in c_grid:
            for class_weight in class_weight_options:
                fold_scores: list[float] = []
                for train_index, holdout_index in splitter.split(features, targets):
                    train_features = features[train_index]
                    holdout_features = features[holdout_index]
                    train_targets = targets[train_index]
                    holdout_targets = targets[holdout_index]

                    mean, scale = _fit_standardization(train_features)
                    standardized_train = _apply_standardization(train_features, mean, scale)
                    standardized_holdout = _apply_standardization(holdout_features, mean, scale)

                    model = _make_logistic_regression(
                        C=c_value,
                        max_iter=max_iter,
                        class_weight=class_weight,
                    )
                    model.fit(standardized_train, train_targets)
                    predictions = model.predict(standardized_holdout)
                    fold_scores.append(
                        _score_predictions(
                            holdout_targets,
                            predictions,
                            selection_metric=selection_metric,
                        )
                    )

                result = {
                    "feature_type": feature_type,
                    "C": float(c_value),
                    "class_weight": class_weight or "none",
                    "cv_score_mean": float(np.mean(fold_scores)),
                    "cv_score_std": float(np.std(fold_scores)),
                    "fold_scores": [float(score) for score in fold_scores],
                }
                results.append(result)

                if best_result is None:
                    best_result = result
                    continue

                if result["cv_score_mean"] > best_result["cv_score_mean"]:
                    best_result = result

    if best_result is None:
        raise ValueError("No stacking configuration candidates were evaluated")

    return {
        "selection_metric": selection_metric,
        "cv_folds": cv_folds,
        "results": results,
        "best": best_result,
    }


@dataclass
class StackingCombiner:
    class_names: list[str]
    checkpoint_paths: list[str]
    feature_type: str
    fitted_classes: list[int]
    coefficient_rows: list[list[float]]
    intercept: list[float]
    feature_mean: list[float]
    feature_scale: list[float]
    C: float
    max_iter: int
    random_seed: int
    cv_folds: int
    class_weight: str | None
    selection_metric: str = "balanced_accuracy"
    solver: str = "lbfgs"

    def to_dict(self) -> dict[str, object]:
        return {
            "type": "multiclass_stacking_logistic_regression",
            "class_names": self.class_names,
            "checkpoint_paths": self.checkpoint_paths,
            "feature_type": self.feature_type,
            "fitted_classes": self.fitted_classes,
            "coefficient_rows": self.coefficient_rows,
            "intercept": self.intercept,
            "feature_mean": self.feature_mean,
            "feature_scale": self.feature_scale,
            "C": self.C,
            "max_iter": self.max_iter,
            "random_seed": self.random_seed,
            "cv_folds": self.cv_folds,
            "class_weight": self.class_weight,
            "selection_metric": self.selection_metric,
            "solver": self.solver,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "StackingCombiner":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("type") != "multiclass_stacking_logistic_regression":
            raise ValueError(f"Unsupported combiner type in {path}: {payload.get('type')}")

        return cls(
            class_names=list(payload["class_names"]),
            checkpoint_paths=list(payload.get("checkpoint_paths", [])),
            feature_type=str(payload["feature_type"]),
            fitted_classes=[int(value) for value in payload["fitted_classes"]],
            coefficient_rows=[[float(item) for item in row] for row in payload["coefficient_rows"]],
            intercept=[float(item) for item in payload["intercept"]],
            feature_mean=[float(item) for item in payload["feature_mean"]],
            feature_scale=[float(item) for item in payload["feature_scale"]],
            C=float(payload["C"]),
            max_iter=int(payload["max_iter"]),
            random_seed=int(payload["random_seed"]),
            cv_folds=int(payload["cv_folds"]),
            class_weight=None if payload.get("class_weight") in {None, "none"} else str(payload["class_weight"]),
            selection_metric=str(payload.get("selection_metric", "balanced_accuracy")),
            solver=str(payload.get("solver", "lbfgs")),
        )

    def _predict_present_class_probabilities(self, standardized_features: np.ndarray) -> np.ndarray:
        coefficients = np.asarray(self.coefficient_rows, dtype=np.float32)
        intercept = np.asarray(self.intercept, dtype=np.float32)
        scores = standardized_features @ coefficients.T + intercept
        return _softmax(scores)

    def predict_proba_from_features(self, features: np.ndarray) -> np.ndarray:
        mean = np.asarray(self.feature_mean, dtype=np.float32)
        scale = np.asarray(self.feature_scale, dtype=np.float32)
        standardized_features = _apply_standardization(features.astype(np.float32, copy=False), mean, scale)
        present_class_probabilities = self._predict_present_class_probabilities(standardized_features)

        all_probabilities = np.zeros((features.shape[0], len(self.class_names)), dtype=np.float32)
        all_probabilities[:, self.fitted_classes] = present_class_probabilities
        return all_probabilities

    def predict_proba(self, model_outputs: np.ndarray | torch.Tensor) -> np.ndarray:
        features = build_stacking_features(model_outputs, feature_type=self.feature_type)
        return self.predict_proba_from_features(features)


def fit_stacking_combiner(
    *,
    class_names: list[str],
    checkpoint_paths: list[Path],
    outputs_by_type: dict[str, np.ndarray],
    targets: np.ndarray,
    feature_types: list[str],
    c_grid: list[float],
    cv_folds: int,
    max_iter: int,
    random_seed: int,
    selection_metric: str = "balanced_accuracy",
    class_weight_options: list[str | None] | None = None,
) -> tuple[StackingCombiner, dict[str, object]]:
    selection = select_stacking_configuration(
        outputs_by_type=outputs_by_type,
        targets=targets,
        feature_types=feature_types,
        c_grid=c_grid,
        cv_folds=cv_folds,
        max_iter=max_iter,
        random_seed=random_seed,
        selection_metric=selection_metric,
        class_weight_options=class_weight_options,
    )
    best = dict(selection["best"])

    selected_feature_type = str(best["feature_type"])
    final_features = build_stacking_features(outputs_by_type[selected_feature_type], feature_type=selected_feature_type)
    feature_mean, feature_scale = _fit_standardization(final_features)
    standardized_features = _apply_standardization(final_features, feature_mean, feature_scale)

    class_weight = None if best["class_weight"] == "none" else str(best["class_weight"])
    classifier = _make_logistic_regression(
        C=float(best["C"]),
        max_iter=max_iter,
        class_weight=class_weight,
    )
    classifier.fit(standardized_features, targets)

    combiner = StackingCombiner(
        class_names=list(class_names),
        checkpoint_paths=[str(path) for path in checkpoint_paths],
        feature_type=selected_feature_type,
        fitted_classes=[int(value) for value in classifier.classes_.tolist()],
        coefficient_rows=classifier.coef_.astype(float).tolist(),
        intercept=classifier.intercept_.astype(float).tolist(),
        feature_mean=feature_mean.astype(float).tolist(),
        feature_scale=feature_scale.astype(float).tolist(),
        C=float(best["C"]),
        max_iter=max_iter,
        random_seed=random_seed,
        cv_folds=cv_folds,
        class_weight=class_weight,
        selection_metric=str(selection["selection_metric"]),
        solver="lbfgs",
    )
    return combiner, selection
