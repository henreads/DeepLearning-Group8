"""Supervised multiclass wafer-defect classification package."""

from wafer_defect.classification.data import (
    DEFAULT_CLASS_NAMES,
    NORMAL_CLASS,
    LabeledWaferDataset,
    RawWaferInferenceDataset,
    build_labeled_metadata,
    prepare_supervised_dataframe,
)
from wafer_defect.classification.ensemble import (
    SUPPORTED_STACKING_FEATURE_TYPES,
    SUPPORTED_STACKING_SELECTION_METRICS,
    StackingCombiner,
    average_probabilities,
    collect_model_outputs,
    fit_stacking_combiner,
    load_classifier_models,
)
from wafer_defect.classification.models import WaferClassifier

__all__ = [
    "DEFAULT_CLASS_NAMES",
    "NORMAL_CLASS",
    "LabeledWaferDataset",
    "RawWaferInferenceDataset",
    "build_labeled_metadata",
    "prepare_supervised_dataframe",
    "SUPPORTED_STACKING_FEATURE_TYPES",
    "SUPPORTED_STACKING_SELECTION_METRICS",
    "StackingCombiner",
    "average_probabilities",
    "collect_model_outputs",
    "fit_stacking_combiner",
    "load_classifier_models",
    "WaferClassifier",
]
