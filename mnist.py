import argparse
import os.path

from lib.data_processing import DataProcessing
from lib.model import MnistModel
from lib.model_functions import ModelFunctions
from lib.show_result import ShowFunction
from utils.path_utils import create_dir


def train_model(*args, **kwargs):
    print(f"train {args[0]}")
    train_dataset_path, model_path = args[0].dataset, args[0].model

    # Check file and file extension
    print("Preprocess data")
    # check_dataset_path(train_dataset_path)
    # data_processing = DataProcessing(train_dataset_path)
    train_dataset, validation_dataset = DataProcessing(train_dataset_path).get_train_generators()

    # Train Model
    print("Train model")
    model_sess = ModelFunctions(MnistModel().model())
    # model_sess.train(train_dataset, validation_dataset)

    model_sess.train_model_all_layers(
        validation_dataset,
        train_dataset, )

    ShowFunction(model_sess.history.history).show_history()

    # Save model
    create_dir(model_path)
    model_sess.save_model(model_path)

    print(f"model saved by path {model_path}")
    pass


def test_model(*args, **kwargs):
    # TODO : fixed evaluate to Test Model!
    print(f"train {args[0]}")
    test_dataset_path, model_path = args[0].dataset, args[0].model

    # Check file and file extension
    print("Preprocess data")
    # check_dataset_path(train_dataset_path)
    # data_processing = DataProcessing(train_dataset_path)
    test_generator = DataProcessing(test_dataset_path).get_test_generator()

    # Train Model
    print("Train model")
    model_sess = ModelFunctions(MnistModel().model())
    # model_sess.train(train_dataset, validation_dataset)
    model_sess.evaluate(test_generator)

    # model_sess.load_latest_checkpoint()
    model_sess.restore_model(model_path)

    model_sess.evaluate(test_generator)

    # Save model
    # create_dir(model_path)
    # model_sess.save_model(model_path)
    # print(f"model saved by path {model_path}")
    pass


def inference_model(*args, **kwargs):
    # TODO : add model predict
    print("inference")
    pass


def check_dataset_path(dataset_path):
    if not os.path.isfile(dataset_path) or os.path.splitext(dataset_path)[-1].lower() != ".csv":
        raise FileNotFoundError("File not found")


def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')

    subparsers = parser.add_subparsers()
    parser_train = subparsers.add_parser('train', help='Train model')
    parser_train.add_argument('--dataset', type=str, help='Path to dataset .csv')
    parser_train.add_argument('--model', type=str, help='Path to model')
    parser_train.set_defaults(func=train_model)

    parser_inference = subparsers.add_parser('predict', help='Inference for model')
    parser_inference.add_argument('--model', type=str, help='Path to model')
    parser_inference.add_argument('--input', type=str, help='Path to input dataset file')
    parser_inference.add_argument('--output', type=str, help='Path to output predictions.csv')
    parser_inference.set_defaults(func=inference_model)

    parser_train = subparsers.add_parser('test', help='Test model')
    parser_train.add_argument('--dataset', type=str, help='Path to dataset .csv')
    parser_train.add_argument('--model', type=str, help='Path to model')
    parser_train.set_defaults(func=test_model)

    args = parser.parse_args()
    print(args)
    return args


def main():
    args = parse_args()
    try:
        args.func(args)
    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
