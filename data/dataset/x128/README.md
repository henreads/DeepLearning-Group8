# x128 Dataset Branches

This folder groups dataset-generation notebooks for `128 x 128` processed wafer maps.

Why keep `x128` as its own branch:
- it lets us test whether a higher-resolution reconstruction input preserves more local defect detail than `x64`
- it keeps the dataset-generation logic separate from the model notebooks that consume the processed CSV and arrays
- it gives graders a direct place to verify that the `x128` processed split can be rebuilt

Current branch:
- `benchmark_50k_5pct/`: the curated 50k-normal benchmark split with a test anomaly count equal to `5%` of the test-normal count
