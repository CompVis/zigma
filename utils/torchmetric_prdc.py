# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module

from torchmetrics.image.fid import NoTrainInceptionV3
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.imports import _TORCH_FIDELITY_AVAILABLE
import sklearn.metrics 
import numpy as np
import sys

__doctest_requires__ = {("PrecisionRecallDensityCoverage", "PRDC"): ["torch_fidelity"]}


def compute_pairwise_distance(data_x, data_y=None):
    """
    Args:
        data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
        data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
    Returns:
        numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
    """
    if data_y is None:
        data_y = data_x
    dists = sklearn.metrics.pairwise_distances(
        data_x.detach().cpu().numpy(), data_y.detach().cpu().numpy(), metric='euclidean', n_jobs=8)
    return dists


def get_kth_value(unsorted, k, axis=-1):
    """
    Args:
        unsorted: numpy.ndarray of any dimensionality.
        k: int
    Returns:
        kth values along the designated axis.
    """
    indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    kth_values = k_smallests.max(axis=axis)
    return kth_values


def compute_nearest_neighbour_distances(input_features, nearest_k):
    """
    Args:
        input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int
    Returns:
        Distances to kth nearest neighbours.
    """
    distances = compute_pairwise_distance(input_features)
    radii = get_kth_value(distances, k=nearest_k + 1, axis=-1)
    return radii


class PRDC(Metric):
    """
    Example:
        >>> import torch
        >>> _ = torch.manual_seed(123)
        >>> from torchmetric_prdc import PRDC
        >>> prdc = PRDC(nearest_k=5)
        >>> # generate two slightly overlapping image intensity distributions
        >>> imgs_dist1 = torch.randint(0, 200, (100, 3, 299, 299), dtype=torch.uint8)
        >>> imgs_dist2 = torch.randint(100, 255, (100, 3, 299, 299), dtype=torch.uint8)
        >>> prdc.update(imgs_dist1, real=True)
        >>> prdc.update(imgs_dist2, real=False)
        >>> result = prdc.compute()
        >>> print(result)
        {'precision': 0.07, 'recall': 0.05, 'density': 0.016, 'coverage': 0.07}

    """
    real_features: List[Tensor]
    fake_features: List[Tensor]
    higher_is_better: bool = True
    is_differentiable: bool = False

    def __init__(
        self,
        feature: Union[str, int, torch.nn.Module] = 2048,
        reset_real_features: bool = True,
        nearest_k: int = 5, 
        realism: bool = False,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(**kwargs)

        if isinstance(feature, (str, int)):
            if not _TORCH_FIDELITY_AVAILABLE:
                raise ModuleNotFoundError(
                    "Precision Recall Density Coverage metric requires that `Torch-fidelity` is installed."
                    " Either install as `pip install torchmetrics[image]` or `pip install torch-fidelity`."
                )
            valid_int_input = ("logits_unbiased", 64, 192, 768, 2048)
            if feature not in valid_int_input:
                raise ValueError(
                    f"Integer input to argument `feature` must be one of {valid_int_input}," f" but got {feature}."
                )

            self.inception: Module = NoTrainInceptionV3(name="inception-v3-compat", features_list=[str(feature)])
        elif isinstance(feature, Module):
            self.inception = feature
        else:
            raise TypeError("Got unknown input to argument `feature`")

        if not isinstance(reset_real_features, bool):
            raise ValueError("Arugment `reset_real_features` expected to be a bool")
        self.reset_real_features = reset_real_features
        self.nearest_k = nearest_k
        self.realism = realism
        # states for extracted features
        self.add_state("real_features", [], dist_reduce_fx=None)
        self.add_state("fake_features", [], dist_reduce_fx=None)

    def update(self, imgs: Tensor, real: bool) -> None:  # type: ignore
        """Update the state with extracted features.

        Args:
            imgs: tensor with images feed to the feature extractor
            real: bool indicating if ``imgs`` belong to the real or the fake distribution
        """
        features = self.inception(imgs)
        if real:
            self.real_features.append(features)
        else:
            self.fake_features.append(features)

    def compute(self) -> Tuple[Tensor, Tensor]:
        """Calculate PRDC score based on accumulated extracted features from the two distributions. 
        Implementation inspired by `Fid Score`_
        """
        real_features = dim_zero_cat(self.real_features)
        fake_features = dim_zero_cat(self.fake_features)

        real_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        real_features, self.nearest_k)
        fake_nearest_neighbour_distances = compute_nearest_neighbour_distances(
            fake_features, self.nearest_k)
        distance_real_fake = compute_pairwise_distance(
            real_features, fake_features)

        precision = (
                distance_real_fake <
                np.expand_dims(real_nearest_neighbour_distances, axis=1)
        ).any(axis=0).mean()

        recall = (
                distance_real_fake <
                np.expand_dims(fake_nearest_neighbour_distances, axis=0)
        ).any(axis=1).mean()

        density = (1. / float(self.nearest_k)) * (
                distance_real_fake <
                np.expand_dims(real_nearest_neighbour_distances, axis=1)
        ).sum(axis=0).mean()

        coverage = (
                distance_real_fake.min(axis=1) <
                real_nearest_neighbour_distances
        ).mean()

        d = dict(precision=precision, recall=recall,
                    density=density, coverage=coverage)
        if self.realism:
            """
            Large errors, even if they are rare, would undermine the usefulness of the metric.
            We tackle this problem by discarding half of the hyperspheres with the largest radii.
            In other words, the maximum in Equation 3 is not taken over all φr ∈ Φr but only over 
            those φr whose associated hypersphere is smaller than the median.
            """
            mask = real_nearest_neighbour_distances < np.median(real_nearest_neighbour_distances)

            d['realism'] = (
                    np.expand_dims(real_nearest_neighbour_distances[mask], axis=1)/distance_real_fake[mask]
            ).max(axis=0)
        return d

    def reset(self) -> None:
        if not self.reset_real_features:
            # remove temporarily to avoid resetting
            value = self._defaults.pop("real_features")
            super().reset()
            self._defaults["real_features"] = value
        else:
            super().reset()


