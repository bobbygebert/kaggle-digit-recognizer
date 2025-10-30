#!/usr/bin/env python

import argparse
from pathlib import Path

from torch.utils.data import random_split

import kaggle_digit_recognizer.evaluate
import kaggle_digit_recognizer.train
from kaggle_digit_recognizer.dataset import DigitDataset, RawDigitDataset
from kaggle_digit_recognizer.model import DigitRecognizer
from kaggle_digit_recognizer.transforms import get_transform


def train(args: argparse.Namespace) -> None:
    raw_dataset = RawDigitDataset(Path(args.data_path))

    training_size = int((1 - args.validation_split) * len(raw_dataset))
    validation_size = len(raw_dataset) - training_size
    training_subset, validation_subset = random_split(
        raw_dataset, [training_size, validation_size]
    )

    training_dataset = DigitDataset(training_subset, transform=get_transform())
    validation_dataset = DigitDataset(validation_subset, transform=get_transform())

    model = DigitRecognizer()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")

    kaggle_digit_recognizer.train.train(
        model, training_dataset, args.batch_size, args.epochs
    )

    training_accuracy = kaggle_digit_recognizer.evaluate.evaluate(
        model, training_dataset, args.batch_size
    )
    print(f"Training accuracy: {training_accuracy:.4f}")

    model.save(Path(args.model_path))

    if len(validation_dataset) > 0:
        validation_accuracy = kaggle_digit_recognizer.evaluate.evaluate(
            model, validation_dataset, args.batch_size
        )
        print(f"Validation accuracy: {validation_accuracy:.4f}")


def evaluate(args: argparse.Namespace) -> None:
    model = DigitRecognizer.load(Path(args.model_path))
    raw_dataset = RawDigitDataset(Path(args.data_path))
    dataset = DigitDataset(raw_dataset, transform=get_transform())
    accuracy = kaggle_digit_recognizer.evaluate.evaluate(
        model, dataset, args.batch_size
    )
    print(f"Accuracy: {accuracy:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Digit recognition model training and evaluation"
    )
    subparsers = parser.add_subparsers(required=True)

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--data-path", type=str, required=True)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--model-path", type=str, required=True)
    train_parser.add_argument("--validation-split", type=float, default=0.2)
    train_parser.set_defaults(func=train)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate the model")
    evaluate_parser.add_argument("--model-path", type=str, required=True)
    evaluate_parser.add_argument("--data-path", type=str, required=True)
    evaluate_parser.add_argument("--batch-size", type=int, default=64)
    evaluate_parser.set_defaults(func=evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
