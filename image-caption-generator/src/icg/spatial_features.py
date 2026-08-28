import torch
import torchvision
from PIL import Image


class ResNet50SpatialFeatureExtractor:
    """
    Returns spatial feature map from ResNet50:
      (2048, 7, 7)
    """
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        weights = torchvision.models.ResNet50_Weights.DEFAULT
        self.transform = weights.transforms()

        resnet = torchvision.models.resnet50(weights=weights)
        # keep spatial map (exclude avgpool + fc)
        self.backbone = torch.nn.Sequential(*list(resnet.children())[:-2]).to(self.device).eval()

    @torch.no_grad()
    def extract(self, image: Image.Image) -> torch.Tensor:
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)  # (1,3,H,W)
        fmap = self.backbone(x).squeeze(0)                                      # (2048,7,7)
        return fmap.float()
