# the script will contain the model from the paper
# the script will only use pytorch

# remember that tensorflow uses channels last: N, D, H, W, C
# pytorch uses channels first: N C, D, H, W

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.optimizer as optimizer

# define the blocks for convolution and residual blocks
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k=3, p=0.3):
        """
        Arguments:
        inc_c is for number of input channels. Eample grayscale has 1 chanel similar to CT. I think CT also has a single channel
        out_c is for number of output channels
        k is for kernel size which usually is 3x3
        p is for dropout probability for regularization
        """
        super().__init__()
        self.conv = nn.Conv3d(in_c, out_c, k, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout3d(p=p)

    def forward(self, x):
        # the input x should have the shape of batch size, channels, depth, height, width aligning with pytorch
        return self.drop(self.relu(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, c, p=0.3):

        """
        Arguments:
        c is for number of channels. since this is residual block, the number of input and output channels should be equal
        p is for dropout probability
        """
        super().__init__()
        self.conv1 = ConvBlock(c, c, p=p)
        self.conv2 = ConvBlock(c, c, p=p)

    def forward(self, x):
        # the input x should have the shape of batch size, channels, depth, height, width aligning with pytorch
        # input and output must have the same shape
        return x + self.conv2(self.conv1(x))


# the davoodnet model

class Model(nn.Module):
    def __init__(self, in_channels, base_feat, depth, n_classes, p_drop=0.3):
        """
        Arguments:
        in_channels is for number of input channels. in my CT data it should be 1 grayscale channel
        base_feat is for starting number of features maps. usual levels are 16,32,64,128
        depth: number of encoder and decoder levels. Depth can be 4
        n_classes is for number of output classes which should be coded in the label array
        """
        super().__init__()

        self.depth = depth
        self.p_drop = p_drop

        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()

        in_c = in_channels
        for d in range(depth):
            out_c = base_feat * (2 ** d)
            self.enc_blocks.append(ConvBlock(in_c, out_c, p=p_drop))
            self.enc_blocks.append(ResBlock(out_c, p=p_drop))
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
            self.dec_blocks.append(ConvBlock(in_c, out_c, p=p_drop))
            self.dec_blocks.append(ResBlock(out_c, p=p_drop))

        # Outputs
        self.out_main = nn.Conv3d(base_feat, n_classes, 1)
        self.out_aux = nn.Conv3d(base_feat, 1, 1)

    def forward(self, x):
        # the input x should have the shape of batch size, channels, depth, height, width aligning with pytorch
        # Encoder
        skips = []
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

            # Match shapes if needed
            # if x.shape != skip.shape:
            #     x = F.interpolate(x, size=skip.shape[2:])

            x = torch.cat([skip, x], dim=1)

            x = self.dec_blocks[2*i](x)
            x = self.dec_blocks[2*i + 1](x)

        # Outputs
        out_main = self.out_main(x)
        out_aux = self.out_aux(x)

        return out_main, out_aux


# -----------------------------
# TensorBoard Example
# -----------------------------

if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter
    x = torch.randn(1, 1, 32, 32, 32)

    model = Model(in_channels=1, base_feat=16, depth=4, n_classes=3)
    criterion_main = nn.CrossEntropyLoss()
    criterion_aux = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Multi-class labels (for CrossEntropyLoss)
    y_main_labels = torch.randint(0, 3, (1, 32, 32, 32))  # shape: (N, D, H, W)

    # Binary labels (for BCEWithLogitsLoss)
    y_aux_labels = torch.randint(0, 2, (1, 1, 32, 32, 32)).float()  # shape: (N, 1, D, H, W)

    writer = SummaryWriter()

    for step in range(10):

        optimizer.zero_grad()    
        # Forward pass
        y_main, y_aux = model(x)

        # Compute loss per type
        loss_main = criterion_main(y_main, y_main_labels)   # multi-class
        loss_aux = criterion_aux(y_aux, y_aux_labels)       # binary

        # Weighted sum
        loss = loss_main + 0.5 * loss_aux
        loss.backward()
        optimizer.step()
        print(f"Step {step}, Loss: {loss.item()}")


    # Log graph
    writer.add_graph(model, x)

    # Log outputs
    writer.add_histogram("output_main", y_main)
    writer.add_histogram("output_aux", y_aux)

    writer.close()
