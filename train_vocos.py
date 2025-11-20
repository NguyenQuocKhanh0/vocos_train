import torch
from pytorch_lightning.cli import LightningCLI
from vocos import Vocos   # pip install vocos


if __name__ == "__main__":
    # 1) Khởi tạo CLI, tạo model + datamodule theo configs/vocos.yaml
    cli = LightningCLI(run=False)

    # 2) Load Vocos pretrain từ HuggingFace
    #    (cần có internet; nếu không có thì lát nữa mình nói cách offline)
    vocos_pre = Vocos.from_pretrained("charactr/vocos-mel-24khz")

    # 3) Lấy state_dict từ model pretrain
    sd_hf = vocos_pre.state_dict()

    # 4) Chỉ load vào backbone + head của model đang train
    #    (tránh đụng vào discriminator, loss, v.v.)
    filtered_sd = {k: v for k, v in sd_hf.items()
                   if k.startswith("backbone.") or k.startswith("head.")}

    # 5) Nạp vào LightningModule của bạn
    #    strict=False để cho phép thiếu các key khác (disc, loss, ...)
    result = cli.model.load_state_dict(filtered_sd, strict=False)
    print("Loaded HF Vocos weights:")
    print("  Missing keys :", result.missing_keys)
    print("  Unexpected   :", result.unexpected_keys)

    # 6) Train tiếp
    cli.trainer.fit(model=cli.model, datamodule=cli.datamodule)
