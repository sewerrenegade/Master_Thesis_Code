
from torch import nn

from . import PersistenceInformation
from .torch_betti_curve import make_betti_curve
from .torch_persistence_landscape import make_persistence_landscape

import gudhi
import torch

import torch


class CubicalComplex(nn.Module):
    def __init__(self, 
                 superlevel=False, 
                 dim=None, 
                 topo_threshold=100, 
                 topo_representor="BettiCurve",
                 topo_feat_dims=1,
    ):
        super().__init__()

        # TODO: This is handled somewhat inelegantly below. Might be
        # smarter to update.
        self.superlevel = superlevel
        self.dim = dim
        # for vector rep aclculation
        self.topo_feat_dims = topo_feat_dims
        self.topo_representor = topo_representor
        self.topo_threshold = topo_threshold

    def forward(self, x):
        if not x.is_cuda:
            x = x.to('cuda')

        if self.dim is not None:
            shape = x.shape[:-self.dim]
            dims = len(shape)
        else:
            dims = len(x.shape) - 2

        if dims == 0:
            return self._forward(x)  # Move to GPU

        elif dims == 1:
            return [self._forward(x_.to('cuda')) for x_ in x]

        elif dims == 2:
            return [[self._forward(x__.to('cuda')) for x__ in x_] for x_ in x]

    def _forward(self, x):
        if self.superlevel:
            x = -x

        cubical_complex = gudhi.CubicalComplex(
            dimensions=x.shape,
            top_dimensional_cells=x.flatten().cpu().numpy()  # Convert to Numpy
        )

        cubical_complex.persistence()
        cofaces = cubical_complex.cofaces_of_persistence_pairs()

        max_dim = len(x.shape)

        persistence_information = [
            self._extract_generators_and_diagrams(
                x,
                cofaces,
                dim
            ) for dim in range(0, max_dim)
        ]

        return persistence_information

        # thresholds = torch.linspace(0, 1, self.topo_threshold)
        # betti_numbers = [cubical_complex.persistent_betti_numbers(from_value=threshold, to_value=threshold)[self.topo_feat_dims-1] for threshold in thresholds]
        # topo_rep_new = torch.tensor(betti_numbers)


        # # to get the vector rep directly instead of diagram
        # if self.topo_representor == "BettiCurve":
        #     topo_rep = make_betti_curve(
        #         persistence_information[self.topo_feat_dims-1].diagram, 
        #         num_thresholds=self.topo_threshold
        #         )[:,1]
        # elif self.topo_representor == "PersistenceLandscape":
        #     topo_rep = make_persistence_landscape(
        #         persistence_information[self.topo_feat_dims-1].diagram,
        #         num_thresholds=self.topo_threshold
        #         )[:,1]
        # else:
        #     raise ValueError("Invalid topological representation method. Use 'BettiCurve' or 'PersistenceLandscape'.")
        
        # return topo_rep.cuda()


    def _extract_generators_and_diagrams(self, x, cofaces, dim):
        pairs = torch.empty((0, 2), dtype=torch.long, device=x.device)

        try:
            regular_pairs = torch.as_tensor(
                cofaces[0][dim], dtype=torch.long, device='cuda'
            )
            pairs = torch.cat((pairs, regular_pairs))
        except IndexError:
            pass

        try:
            infinite_pairs = torch.as_tensor(
                cofaces[1][dim], dtype=torch.long, device='cuda'
            )
        except IndexError:
            infinite_pairs = None

        if infinite_pairs is not None:
            max_index = torch.argmax(x)
            fake_destroyers = torch.full_like(infinite_pairs, fill_value=max_index)

            infinite_pairs = torch.stack(
                (infinite_pairs, fake_destroyers), 1
            )

            pairs = torch.cat((pairs, infinite_pairs))

        return self._create_tensors_from_pairs(x, pairs, dim)

    def _create_tensors_from_pairs(self, x, pairs, dim):

        creators = torch.as_tensor(
            torch.nonzero(pairs[:, 0]).t(), dtype=torch.long, device='cuda'
        )
        destroyers = torch.as_tensor(
            torch.nonzero(pairs[:, 1]).t(), dtype=torch.long, device='cuda'
        )
        gens = torch.cat((creators, destroyers), dim=1)

        persistence_diagram = torch.stack((
            x.view(-1)[pairs[:, 0]],
            x.view(-1)[pairs[:, 1]]
        ), 1)

        return PersistenceInformation(
            pairing=gens,
            diagram=persistence_diagram,
            dimension=dim
        )
