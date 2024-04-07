"""Layers for processing persistence diagrams.
this code does not consider all persistence homology dimenssions, only covers 2 dimensions.
this code is not designed for cases where the shape of diagrams for homology dimension is not same. >> I added padding and then called this code."""

import torch


class StructureElementLayer(torch.nn.Module):
    def __init__(
        self,
        n_elements
    ):
        super().__init__()

        self.n_elements = n_elements
        self.dim = 2    # TODO: Make configurable

        size = (self.n_elements, self.dim)

        self.centres = torch.nn.Parameter(
            torch.rand(*size)
        )

        self.sharpness = torch.nn.Parameter(
            torch.ones(*size) * 3
        )

    def forward(self, x):
        batch = torch.cat([x] * self.n_elements, 1)
        B, N, D = x.shape

        centres = torch.cat([self.centres] * N, 1)
        centres = centres.view(-1, self.dim)
        centres = torch.stack([centres] * B, 0)

        sharpness = torch.pow(self.sharpness, 2)
        sharpness = torch.cat([sharpness] * N, 1)
        sharpness = sharpness.view(-1, self.dim)
        sharpness = torch.stack([sharpness] * B, 0)

        x = centres - batch
        x = x.pow(2)
        x = torch.mul(x, sharpness)
        x = torch.nansum(x, 2)
        x = torch.exp(-x)
        x = x.view(B, self.n_elements, -1)
        x = torch.sum(x, 2)
        x = x.squeeze()

        return x