"""Build the public single-lead model without loading any dataset."""

from __future__ import annotations

import argparse

from single_lead_ecg_fm import build_single_lead_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to the official ECG-FM checkpoint")
    parser.add_argument(
        "--source-lead-index",
        type=int,
        default=1,
        help="Zero-based source channel used to initialize the single-lead convolution",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = build_single_lead_classifier(
        args.checkpoint,
        source_lead_index=args.source_lead_index,
    )
    print(model)


if __name__ == "__main__":
    main()
