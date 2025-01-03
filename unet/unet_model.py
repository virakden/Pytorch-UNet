from .unet_parts import *
import torch.utils.checkpoint as checkpoint


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Use gradient checkpointing here
        x1 = checkpoint.checkpoint(self.inc, x)
        x2 = checkpoint.checkpoint(self.down1, x1)
        x3 = checkpoint.checkpoint(self.down2, x2)
        x4 = checkpoint.checkpoint(self.down3, x3)
        x5 = checkpoint.checkpoint(self.down4, x4)
        x = checkpoint.checkpoint(self.up1, x5, x4)
        x = checkpoint.checkpoint(self.up2, x, x3)
        x = checkpoint.checkpoint(self.up3, x, x2)
        x = checkpoint.checkpoint(self.up4, x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        # This method is no longer needed with the current implementation
        # Keeping it in case you want to apply it later manually if required
        pass
