# Data Science Pipeline

1. **Data understanding**: `src.data.loader` loads the public IBM Telco sample and creates a data-quality report covering shape, dtypes, missingness and cardinality.
2. **Cleaning**: `src.data.preprocessing.clean_frame` removes duplicate rows, normalizes the identifier, converts `TotalCharges`, and maps `Churn` to a binary target.
3. **EDA**: `src.analysis.eda` produces focused charts for churn, contract, tenure, charges, payment method and numeric correlation.
4. **Statistics**: `src.analysis.statistics` uses Mann-Whitney U for selected continuous variables and chi-square tests for selected categorical relationships. The test choice is explicit rather than automatic.
5. **Feature engineering**: `src.features.engineering` creates tenure-adjusted charges, estimated tenure value, tenure buckets, service count and electronic-payment indicator where source fields exist.
6. **Feature selection**: `src.features.selection` computes mutual information on a training partition as an analysis artifact. Selection is not performed using the test set.
7. **Imbalance**: `src.models.imbalance` compares unweighted versus class-weighted logistic regression using stratified CV.

Run each module with `python -m ...`. Results are generated from execution and are not hard-coded.
