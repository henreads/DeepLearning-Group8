# x240 Benchmark 120k / 5% Split

This dataset branch prepares the `240 x 240` EfficientNet-style benchmark with:

- `120,000` total normal wafers
- `96,000` train normals
- `12,000` validation normals
- `12,000` test normals
- `600` test anomalies (`5%` of the test-normal count)

It keeps the same `normal_only_test_defects` protocol as the original `50k / 5%` benchmark while scaling the normal-only training pool for larger follow-up runs.
