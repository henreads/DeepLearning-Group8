from __future__ import annotations

import argparse
from pathlib import Path

from wafer_defect.data.legacy_pickle import read_legacy_pickle, unwrap_legacy_value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/LSWMD.pkl")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {input_path}")

    df = read_legacy_pickle(input_path)

    print(f"rows={len(df)}")
    print(f"columns={list(df.columns)}")
    print()
    print(df.head(3).to_string())
    print()

    for column in ("failureType", "trianTestLabel"):
        if column in df.columns:
            flattened = df[column].map(unwrap_legacy_value)
            print(f"{column} unique sample:")
            print(flattened.value_counts(dropna=False).head(10).to_string())
            print()


if __name__ == "__main__":
    main()
