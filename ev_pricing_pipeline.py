"""Compatibility entry point for the maintained EV-price training CLI."""

from src.ev_price_prediction_xgb import build_parser, run


def main() -> None:
    args = build_parser().parse_args()
    rmse = run(args.train_csv, args.test_csv, args.output_csv, args.metrics_json)
    print(f"Validation RMSE: {rmse:.4f}")


if __name__ == "__main__":
    main()
