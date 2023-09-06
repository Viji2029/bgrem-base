"""
Each run file needs to export 4 functions:
1. train: trainer
2. test: test using loaders, etc
3. export: export onnx version of the model
4. predict: production run using the onnx model. this code would be ported to C++ as required

Keep logic to a minimum here. Use it to call respective functions
"""
import argparse
import sys


def run_train(args):
    from train import train

    print("Inside train")
    train(args)


def run_test(args):
    print("Not implemented. Export to onnx and run predict")


def run_export(args):
    from export import export

    print("Inside export")
    export(args.model_path, args.out_path)


def run_predict(args):
    from predict import predict

    print("Inside predict")
    predict(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(help="subparsers")

    if len(sys.argv) < 2:
        print("Add valid subcommand: train/test/export/predict")
        sys.exit(1)

    # ------------- Train args -------------
    train_parser = subparsers.add_parser("train", help="train args")
    train_parser.add_argument("--name", type=str, default="", help="experiment name")
    train_parser.add_argument("--azure_job", type=bool, default=False, help="whether running as azure job")
    train_parser.add_argument("--ablation",type=bool,default=False,help="whether doing an ablation run")

    train_parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    train_parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    train_parser.add_argument("--epochs", type=int, default=100, help="number of epochs to train for")
    train_parser.add_argument("--seed", type=int, default=42, help="manual seed")
    train_parser.add_argument("--accelerator", type=str, default="gpu")
    train_parser.add_argument("--devices", type=int, default=1)
    # Dataset
    train_parser.add_argument(
        "--train_base_path",
        type=str,
        default="./datasets/u2net_lite_test",
        help="train dataset path",
    )
    train_parser.add_argument(
        "--val_base_path",
        type=str,
        default="./datasets/u2net_lite_test",
        help="val dataset path",
    )
    train_parser.add_argument(
        "--out_path",
        type=str,
        default="./outputs",
        help="output path",
    )
    train_parser.set_defaults(func=run_train)

    # ------------- Test args -------------
    test_parser = subparsers.add_parser("test", help="test args")
    # add arguments
    test_parser.set_defaults(func=run_test)

    # ------------- Export args -------------
    export_parser = subparsers.add_parser("export", help="export args. model_path: reqd, out_path: optional")
    export_parser.add_argument("--model_path", type=str, required=True, help="trained model ckpt path")
    export_parser.add_argument("--out_path", type=str, required=True, help="output onnx model full path")
    export_parser.set_defaults(func=run_export)

    # ------------- Predict args -------------
    predict_parser = subparsers.add_parser(
        "predict",
        help="predict args. model_path is reqd. either give input_image or input_dir",
    )
    predict_parser.add_argument("--model_path", type=str, required=True, help="trained model ckpt path")
    predict_parser.add_argument("--input_image", type=str, help="path to input image")
    predict_parser.add_argument(
        "--input_dir",
        type=str,
        default="../../tests/test_images",
        help="path to dir containing input images",
    )
    predict_parser.add_argument(
        "--out_dir",
        type=str,
        default="./outputs",
        help="path to write output files. will write insisde a `curr_date_time` subfolder",
    )
    predict_parser.set_defaults(func=run_predict)

    args = parser.parse_args()
    args.func(args)
