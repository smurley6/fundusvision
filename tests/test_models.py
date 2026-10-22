import torch

from fundusvision.models import FundusClassifier, UNet
from fundusvision.models.classifier import count_parameters


def test_classifier_output_shape_and_size():
    model = FundusClassifier("resnet34", pretrained=False).eval()
    x = torch.randn(3, 3, 64, 64)
    assert model(x).shape == (3,)
    p = model.predict_proba(x)
    assert ((p >= 0) & (p <= 1)).all()
    assert count_parameters(model) == 21_449_793
    assert model.feature_layer is model.backbone[7]


def test_unet_output_shape_and_size():
    model = UNet().eval()
    x = torch.randn(2, 3, 64, 64)
    assert model(x).shape == (2, 2, 64, 64)
    assert model.predict_mask(x).shape == (2, 64, 64)
    assert count_parameters(model) == 31_043_586


def test_unet_handles_odd_sizes():
    model = UNet(widths=(8, 16)).eval()
    assert model(torch.randn(1, 3, 50, 70)).shape == (1, 2, 50, 70)
