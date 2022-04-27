import torch
import torch.nn as nn

# This model is inspired by ResNet 
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class EncBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = conv3x3(in_planes=in_ch, out_planes=in_ch)
        self.bn1    = nn.BatchNorm2d(in_ch)
        self.relu  = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(in_planes=in_ch, out_planes=in_ch)
        self.bn2    = nn.BatchNorm2d(in_ch)

        # This 1 X 1 layer downsamples and inc channel as required
        self.conv3 = conv1x1(in_planes=in_ch, out_planes=out_ch, stride=2)
        self.bn3 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        
        # Till now channel and spatial dims of out and identity are same 
        # Now we inc channel length
        out = self.conv3(out)
        out = self.bn3(out)
        return out

class Encoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # First conv to inc channels from 1 to 16
        self.conv1 = conv3x3(in_planes=1, out_planes=16)
        self.bn1    = nn.BatchNorm2d(16)
        self.relu  = nn.ReLU(inplace=True)

        self.enc_blocks = nn.ModuleList([EncBlock(channels[i], channels[i+1]) for i in range(len(channels)-1)])

    def forward(self, x):
        # Input = [N, 1, 64, 64]

#       If channel = 1 then additional conv 
        if x.shape[1] == 1:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

        # Output = [N, 16, 64, 64]
        
        for block in self.enc_blocks:
            x = block(x)
        return x


class Embedded_Encoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.firstEncoder= Encoder((16, 32, 64))

        self.secondEncoder= Encoder((64, 128))

        self.thirdEncoder= Encoder((128, 256))

    def forward(self, x):

        input_shape = x.shape
        batch_size, seq_len = x.shape[0:2]
        
        # Changed to 4D
        x = x.reshape(-1, *input_shape[2:])

        encoder_outputs = []
        
        x=self.firstEncoder(x)
        encoder_outputs.append(x)

        x=self.secondEncoder(x)
        encoder_outputs.append(x)

        x=self.thirdEncoder(x)
        encoder_outputs.append(x)

# Back to 5D
        for idx, emds in enumerate(encoder_outputs):
            emds_dims = emds.shape[1:]
            encoder_outputs[idx] = emds.reshape(batch_size, seq_len, *emds_dims)
            
        return encoder_outputs