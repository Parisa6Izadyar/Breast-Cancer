import argparse
from pathlib import Path
import sys

import joblib
import pandas as pd

from .data import load_data
from .experiment import run_experiment

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate recorded vital-status classifiers on the bundled dataset.")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--input", type=Path, default=ROOT / "data/Breast_Cancer.csv")
    train.add_argument("--output", type=Path, default=ROOT / "results/reproduced")
    predict = commands.add_parser("predict")
    predict.add_argument("--model", type=Path, default=ROOT / "results/reproduced/model.joblib")
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--output", type=Path, default=ROOT / "results/reproduced/predictions.csv")
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            frame, quality = load_data(args.input)
            run_experiment(frame, quality, args.output)
        else:
            artifact = joblib.load(args.model)
            frame = pd.read_csv(args.input)
            model = artifact["pipeline"]
            probability = model.predict_proba(frame)[:, list(model.classes_).index(1)]
            args.output.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"recorded_dead_score": probability, "predicted_status": ["Dead" if p >= 0.5 else "Alive" for p in probability]}).to_csv(args.output, index=False)
            print(f"Predictions saved: {args.output.resolve()}")
        return 0
    except (ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
