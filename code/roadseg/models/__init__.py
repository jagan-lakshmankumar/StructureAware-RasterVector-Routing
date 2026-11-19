"""Model factory — every model lives in its own file, named after the original.

Supported:
    unet            classic Ronneberger U-Net (from scratch)
    unet_resnet34   SMP U-Net with ResNet-34 encoder
    unetpp          U-Net++ (ResNet-34 encoder)
    dlinknet        D-LinkNet (ResNet-34 + dilated cascade)
    deeplabv3plus   DeepLabV3+ (configurable encoder)
    segformer       SegFormer MiT-B0..B5
"""

try:
    import segmentation_models_pytorch as smp
except Exception:  # pragma: no cover
    smp = None

from .unet import UNet
from .unetpp import UNetPP
from .dlinknet import DLinkNet
from .deeplabv3plus import DeepLabV3Plus
from .segformer import SegFormer


def build_model(name, **kwargs):
    name = name.lower().replace("-", "_")
    pretrained = kwargs.pop("pretrained", False)

    # ---- U-Net ----
    if name in ("unet", "unet_classic"):
        return UNet(in_channels=3, out_channels=1, base=kwargs.get("base", 32))

    if name in ("unet_resnet34", "unet_r34"):
        if smp is None:
            raise ImportError("segmentation_models_pytorch is required")
        return smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3, classes=1, activation=None,
        )

    # ---- U-Net++ ----
    if name in ("unetpp", "unet_plus_plus", "unetplusplus"):
        return UNetPP(encoder=kwargs.get("encoder", "resnet34"), pretrained=pretrained)

    # ---- D-LinkNet ----
    if name in ("dlinknet", "d_linknet"):
        return DLinkNet(in_channels=3, out_channels=1)

    # ---- DeepLabV3+ ----
    if name in ("deeplabv3plus", "deeplabv3_plus", "deeplab", "deeplabv3"):
        return DeepLabV3Plus(encoder=kwargs.get("encoder", "resnet34"),
                             pretrained=pretrained)

    # ---- SegFormer ----
    if name in ("segformer", "segformer_mit"):
        return SegFormer(variant=kwargs.get("variant", "mit_b0"), pretrained=pretrained)

    raise ValueError(f"Unknown model: {name}")


MODEL_NAMES = ["unet", "unet_resnet34", "unetpp", "dlinknet",
               "deeplabv3plus", "segformer"]
