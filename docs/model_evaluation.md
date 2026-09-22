# Model Evaluation

The project evaluates multiple classifiers using stratified five-fold cross-validation and a held-out test set. Metrics include accuracy, precision, recall, F1, ROC-AUC and PR-AUC.

Model selection is configured around mean cross-validation F1 in `src.models.train`; the actual selected model is written to `models/metadata.json` after execution. This is a documented selection rule, not a pre-selected result.

`src.models.evaluate` creates confusion-matrix, ROC and precision-recall artifacts. `src.analysis.error_analysis` labels held-out predictions as correct, false positive, false negative, and identifies borderline probabilities around 0.5 for further inspection.
