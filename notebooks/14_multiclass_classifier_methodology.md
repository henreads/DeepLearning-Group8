## Multiclass Classifier Methodology

This classifier is trained as a supervised multiclass wafer-defect model on the labeled WM-811K subset before being used for unlabeled pseudo-labeling.

### Why this structure

- keep the primary task clean: predict one of the known wafer-defect classes on labeled data
- evaluate the classifier on held-out labeled validation and test splits before trusting it on unlabeled wafers
- optimize for class-aware quality, not just raw overall accuracy, because the dataset is imbalanced

### Current research-aligned choices

- residual CNN backbone instead of the original plain CNN
- train-time augmentation only on the training split
- weighted sampling enabled by default for class imbalance
- class-weighted loss disabled by default to avoid over-correcting imbalance
- label smoothing enabled to reduce overconfidence
- best checkpoint selected by validation balanced accuracy
- learning-rate decay and early stopping to stabilize training
- unlabeled predictions filtered by confidence for safer pseudo-labeling

### Practical implication

The classifier should be judged first by labeled validation and test performance, especially balanced accuracy and per-class recall. Unlabeled predictions should be treated as candidate pseudo-labels, not as guaranteed ground truth.
