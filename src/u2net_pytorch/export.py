import torch
from train_lightning import U2NET_full, U2NetLightning


def export(checkpoint_path, out_path="./u2net_kg_export.onnx"):
    # TODO: Make sure sigmoid is returned in model, uncomment this line in the class below: [torch.sigmoid(x) for x in maps]
    u2net = U2NET_full()
    # u2net = U2NET_lite()
    pl_model = U2NetLightning({}, u2net)
    print(pl_model)

    ckpt = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    pl_model.load_state_dict(ckpt["state_dict"])
    print("model loaded")

    x = torch.randn(1, 3, 320, 320, requires_grad=True)

    torch.onnx.export(
        pl_model,
        x,
        out_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input_0"],
        output_names=["output_0"],
        dynamic_axes={
            "input_0": {0: "batch_size", 2: "w", 3: "h"},
            "output_0": {0: "batch_size", 2: "w", 3: "h"},
        },
    )


if __name__ == "__main__":
    # checkpoint_path = "/Users/dhruv/Downloads/u2net_epoch=0099_train_loss=0.29_val_loss=0.11_val_mae=0.0277.ckpt"

    checkpoint_path = "/Users/dhruv/Downloads/u2net_epoch=0091_train_loss=0.32_val_loss=0.09_val_mae=0.0273.ckpt"
    export(checkpoint_path)
