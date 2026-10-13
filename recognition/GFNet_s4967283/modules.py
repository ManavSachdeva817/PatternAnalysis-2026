"""
modules.py

Model definitions for the ADNI Alzheimer's classification project
(AD = Alzheimer's Disease, CN = Cognitively Normal).

This file holds two models:

  1. BaselineCNN  - a plain convolutional network. This is the comparison point.
                    The assignment asks me to implement a simple baseline myself
                    and benchmark the main model against it on the same splits,
                    so this is not filler, it is half of the experiment.

  2. GFNet        - the main model. Added in a later commit.

Note on imports: the assignment says the model code must not depend on NumPy or
anything outside PyTorch, so everything here is pure torch.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    One convolution step: convolution -> batch norm -> ReLU -> max pool.

    I pulled this out into its own class because the baseline repeats the same
    shape four times with different channel counts. Writing it once means there
    is only one place to change if I want to adjust the block later.

    Why each piece is here:
      - Conv2d with padding=1 on a 3x3 kernel keeps the width and height the
        same, so only the pooling layer changes the spatial size. That makes the
        size arithmetic easy to follow.
      - BatchNorm2d steadies training. MRI slices vary a lot in overall
        brightness between scanners and patients, so normalising the activations
        helps the network stop caring about absolute intensity.
      - MaxPool2d halves the width and height each block, which is what lets the
        later layers see a larger area of the original image.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.convolution = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=False,  # no bias needed, the batch norm right after has its own shift
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.convolution(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pooling(x)
        return x


class BaselineCNN(nn.Module):
    """
    A straightforward CNN for classifying a brain MRI slice as AD or CN.

    Shape of the thing: four ConvBlocks that grow the channel count while
    shrinking the image, then pool down to a single value per channel, then a
    small fully connected head.

    Design decisions worth being able to explain:

      - I deliberately kept this plain. It is meant to be a fair but ordinary
        reference point. If I tuned it heavily it would stop being a baseline
        and the comparison against GFNet would say less.

      - AdaptiveAvgPool2d((1, 1)) instead of flattening a fixed grid. Global
        average pooling collapses each channel to one number no matter what the
        input size was, so the model works whatever resolution I settle on after
        the data audit. It also cuts the parameter count a lot compared to
        flattening, which makes it less likely to overfit on a dataset this size.

      - The final layer outputs 2 logits rather than 1 value with a sigmoid.
        Two logits plus CrossEntropyLoss gives me a proper probability for each
        class after a softmax, and I need calibrated per-class confidence later
        for the uncertainty investigation. One sigmoid would work too but this
        keeps the confidence handling identical between the baseline and GFNet.

      - Dropout before the final layer, because the AD/CN split has relatively
        few patients and the network will happily memorise them otherwise.
    """

    def __init__(self, in_channels=1, num_classes=2, dropout_rate=0.3):
        super().__init__()

        # Channel counts double each block: 1 -> 32 -> 64 -> 128 -> 256.
        # Doubling as the image shrinks is the usual pattern, it keeps the
        # amount of information per layer roughly steady.
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )

        # Squash each of the 256 channels down to a single number.
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),           # shape becomes (batch, 256)
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        """
        x comes in with shape (batch, channels, height, width).
        Returns raw logits with shape (batch, num_classes).

        Logits, not probabilities - CrossEntropyLoss applies the softmax
        itself, so applying one here would be doing it twice.
        """
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


def count_trainable_parameters(model):
    """
    Count how many parameters the model will actually learn.

    The assignment asks for a resource profiling table comparing the baseline
    and the main model, and parameter count is one of the three numbers it wants
    (alongside peak GPU memory and inference time), so this lives here rather
    than being worked out by hand.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # A quick self-check so I can confirm the shapes work before wiring this
    # into the training script. Run with: python modules.py
    #
    # I feed it two different input sizes on purpose, to prove the adaptive
    # pooling really does make the model size-independent.

    for image_size in (224, 256):
        model = BaselineCNN(in_channels=1, num_classes=2)
        fake_batch = torch.randn(4, 1, image_size, image_size)
        output = model(fake_batch)

        print("input %dx%d -> output shape %s" % (image_size, image_size, tuple(output.shape)))
        assert output.shape == (4, 2), "expected 4 samples and 2 class scores"

    print("trainable parameters in BaselineCNN: %d" % count_trainable_parameters(BaselineCNN()))
    print("shape checks passed")
