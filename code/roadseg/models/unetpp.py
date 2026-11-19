"""U-Net++ (nested skip-connections, ResNet-34 encoder)."""

try:
    import segmentation_models_pytorch as smp
except Exception:  # pragma: no cover
    smp = None


class UNetPP:
    """Thin factory returning an SMP U-Net++ instance."""

    def __new__(cls, encoder="resnet34", pretrained=False, in_channels=3, classes=1):
        if smp is None:
            raise ImportError("segmentation_models_pytorch is required for U-Net++")
        return smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=in_channels,
            classes=classes,
            activation=None,
        )
