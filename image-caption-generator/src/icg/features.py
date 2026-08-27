import torch
import torchvision
from PIL import Image

class ResNet50FeatureExtractor:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.weights = torchvision.models.ResNet50_Weights.DEFAULT
        self.transform = self.weights.transforms()

        base = torchvision.models.resnet50(weights=self.weights)
        self.backbone = torch.nn.Sequential(*list(base.children())[:-1]).to(self.device).eval()

    @torch.no_grad()
    def extract(self, image: Image.Image) -> torch.Tensor:
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)  # (1,3,H,W)
        feat = self.backbone(x).squeeze(-1).squeeze(-1)                        # (1,2048)
        return feat.squeeze(0).float()
