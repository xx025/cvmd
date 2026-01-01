import torch


if __name__ == "__main__":
    # reference: https://github.com/facebookresearch/detr/releases/tag/v0.2
    model = torch.hub.load("facebookresearch/detr", "detr_resnet50", pretrained=True)
    model.eval()  
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    model.to(device)
    ts_model = torch.jit.script(model).half()  
    ts_model.save("/home/user/worksapce/cvmd/temp/detr_resnet50.torchscript")
    print("Saved: detr_resnet50.torchscript")
