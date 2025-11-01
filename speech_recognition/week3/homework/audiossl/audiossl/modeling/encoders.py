"""
Audio Encoders: 1D (waveform) and 2D (spectrogram) ResNet architectures
Copied from SSL_Seminar.ipynb
"""  # noqa: E501
import torch
import torch.nn as nn


class Block1D(nn.Module):
    """ResNet block for 1D convolutions (waveform input)"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        identity_downsample: nn.Module | None = None,
        stride: int = 1,
    ):
        super(Block1D, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.identity_downsample = identity_downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)

        return x


class ResNet1D(nn.Module):
    """
    1D ResNet encoder for raw waveform input
    Architecture: ResNet-18 [2, 2, 2, 2]
    Output: [batch_size, 512, 1] after adaptive pooling
    """  # noqa: E501

    def __init__(self, block: nn.Module | None = None):
        super(ResNet1D, self).__init__()
        if block is None:
            block = Block1D

        # ResNet-18 configuration
        layers = [2, 2, 2, 2]
        self.expansion = 1

        self.in_channels = 64
        self.conv1 = nn.Conv1d(
            1, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm1d(self.in_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, layers[0], 64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], 128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], 256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, 1, sequence_length] - raw waveform
        Returns:
            x: [batch_size, 512, 1] - encoded features
        """  # noqa: E501
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        return x

    def _make_layer(
        self, block: nn.Module, num_residual_block: int, out_channels: int, stride: int
    ) -> nn.Sequential:
        identity_downsample = None
        layers = []

        if stride != 1:
            identity_downsample = nn.Sequential(
                nn.Conv1d(
                    self.in_channels,
                    out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels * self.expansion),
            )

        layers.append(
            block(self.in_channels, out_channels, identity_downsample, stride)
        )
        self.in_channels = out_channels * self.expansion

        for i in range(1, num_residual_block):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)


class Block2D(nn.Module):
    """ResNet block for 2D convolutions (spectrogram input)"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        identity_downsample: nn.Module | None = None,
        stride: int = 1,
    ):
        super(Block2D, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.identity_downsample = identity_downsample
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)

        return x


class ResNet2D(nn.Module):
    """
    2D ResNet encoder for spectrogram input
    Architecture: ResNet-18 [2, 2, 2, 2]
    Output: [batch_size, 512, 1, 1] after adaptive pooling
    """  # noqa: E501

    def __init__(self, image_channels: int = 1, block: nn.Module | None = None):
        super(ResNet2D, self).__init__()
        if block is None:
            block = Block2D

        # ResNet-18 configuration
        layers = [2, 2, 2, 2]
        self.expansion = 1

        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            image_channels,
            self.in_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, layers[0], 64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], 128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], 256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, 1, freq_bins, time_steps] - spectrogram
        Returns:
            x: [batch_size, 512, 1, 1] - encoded features
        """  # noqa: E501
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        return x

    def _make_layer(
        self, block: nn.Module, num_residual_block: int, out_channels: int, stride: int
    ):
        identity_downsample = None
        layers = []

        if stride != 1:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )

        layers.append(
            block(self.in_channels, out_channels, identity_downsample, stride)
        )
        self.in_channels = out_channels * self.expansion

        for i in range(1, num_residual_block):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)


def create_resnet1d():
    """Factory function for 1D ResNet encoder"""
    return ResNet1D()


def create_resnet2d(img_channels=1):
    """Factory function for 2D ResNet encoder"""
    return ResNet2D(image_channels=img_channels)
