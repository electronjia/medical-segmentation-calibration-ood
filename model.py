import torch
import torch.nn as nn

# -------------------------
# Basic Blocks
# -------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k=3, p=0.1):  # ↓ dropout reduced
        super().__init__()
        self.conv = nn.Conv3d(in_c, out_c, k, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout3d(p=p)

    def forward(self, x):
        return self.drop(self.relu(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, c, p=0.1):
        super().__init__()
        self.conv1 = ConvBlock(c, c, p=p)
        self.conv2 = ConvBlock(c, c, p=p)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


# -------------------------
# Model
# -------------------------

class Model(nn.Module):
    def __init__(self, in_channels=1, base_feat=16, depth=4, n_classes=3):
        super().__init__()

        self.depth = depth

        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()

        in_c = in_channels
        for d in range(depth):
            out_c = base_feat * (2 ** d)

            self.enc_blocks.append(ConvBlock(in_c, out_c))
            self.enc_blocks.append(ResBlock(out_c))

            if d < depth - 1:
                self.downsample.append(nn.Conv3d(out_c, out_c, 2, stride=2))

            in_c = out_c

        # Decoder
        self.upconvs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        for d in range(depth - 1, 0, -1):
            in_c = base_feat * (2 ** d)
            out_c = base_feat * (2 ** (d - 1))

            self.upconvs.append(nn.ConvTranspose3d(in_c, out_c, 2, stride=2))

            # 🔥 FIX: concat doubles channels → use out_c * 2
            self.dec_blocks.append(ConvBlock(out_c * 2, out_c))
            self.dec_blocks.append(ResBlock(out_c))

        # Final output
        self.out = nn.Conv3d(base_feat, n_classes, 1)

    def forward(self, x):
        skips = []

        # Encoder
        for i in range(self.depth):
            x = self.enc_blocks[2*i](x)
            x = self.enc_blocks[2*i + 1](x)

            if i < self.depth - 1:
                skips.append(x)
                x = self.downsample[i](x)

        # Decoder
        for i in range(self.depth - 1):
            x = self.upconvs[i](x)
            skip = skips[-(i + 1)]

            # Optional safety (keep commented unless needed)
            # if x.shape != skip.shape:
            #     x = nn.functional.interpolate(x, size=skip.shape[2:])

            x = torch.cat([skip, x], dim=1)
            x = self.dec_blocks[2*i](x)
            x = self.dec_blocks[2*i + 1](x)

        return self.out(x)