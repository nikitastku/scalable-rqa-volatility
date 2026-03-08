from __future__ import annotations

import argparse
from pathlib import Path

from scalable_rqa_volatility.plots.io import load_predictions_npz, plot_combined_roc, ensure_dir


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", nargs="+", required=True)
    args = parser.parse_args()

    bundles = [load_predictions_npz(Path(p)) for p in args.preds]
    out_dir = ensure_dir(repo_root() / "reports" / "figures" / "models")

    plot_combined_roc(bundles, out_dir / "roc_all.png", title="ROC Curves (Test)")


if __name__ == "__main__":
    main()