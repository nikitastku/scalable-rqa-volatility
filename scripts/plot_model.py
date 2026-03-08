from __future__ import annotations

import argparse
from pathlib import Path

from scalable_rqa_volatility.plots.io import load_predictions_npz, plot_confusion, plot_roc, ensure_dir


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, required=True)
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()

    pred_path = Path(args.pred)
    bundle = load_predictions_npz(pred_path)
    if args.name is not None:
        bundle = bundle.__class__(name=args.name, y_true=bundle.y_true, y_score=bundle.y_score, y_pred=bundle.y_pred)

    out_dir = ensure_dir(repo_root() / "reports" / "figures" / "models" / bundle.name)

    plot_roc(bundle, out_dir / "roc.png")
    plot_confusion(bundle, out_dir / "confusion.png")


if __name__ == "__main__":
    main()