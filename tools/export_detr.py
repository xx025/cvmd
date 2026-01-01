import torch


if __name__ == "__main__":
    # reference: https://github.com/facebookresearch/detr/releases/tag/v0.2
    model = torch.hub.load("facebookresearch/detr", "detr_resnet50", pretrained=True)
    model.eval()  
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    model.to(device)
    
    if device == "cuda":
        model = model.half()

    dummy_input = torch.randn(1, 3, 800, 800).to(device)
    if device == "cuda":
        dummy_input = dummy_input.half()
    
    # must set strict=False due to some ops not supported in half precision
    ts_model = torch.jit.trace(model, dummy_input, strict=False)
        
    save_path = "/home/user/worksapce/cvmd/temp/model_weights/detr_resnet50.torchscript"
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ts_model.save(save_path)
    print(f"Saved: {save_path}")
