"""Backward-compatible entry point; prefer `python mushroom_train.py`."""

from mushroom_train import parse_args, train


if __name__ == "__main__":
    arguments = parse_args()
    train(arguments.data, arguments.output)
