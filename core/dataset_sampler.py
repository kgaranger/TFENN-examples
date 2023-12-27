# Copyright 2023 Kévin Garanger
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

import logging
from typing import Tuple

import jax
from TFENN.util.enum_input import EnumInputClass

from .dataset import Dataset


class DatasetSamplerType(EnumInputClass):
    RANDOM_SHUFFLE = "random_shuffle"
    WHOLE_DATASET = "whole_dataset"
    SEQUENTIAL = "sequential"

    @classmethod
    @property
    def obj_map(cls) -> dict["DatasetSamplerType", "DatasetSampler"]:
        return {
            cls.RANDOM_SHUFFLE: RandomShuffleSampler,
            cls.WHOLE_DATASET: WholeDatasetSampler,
            cls.SEQUENTIAL: SequentialSampler,
        }


class DatasetSampler:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        key: jax.random.PRNGKey = None,
        seed: int = 0,
    ):
        self.dataset = dataset
        if not self.dataset.scaled:
            logging.warning("Dataset is not scaled.")

        self.batch_size = batch_size
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size is not None and batch_size > len(dataset):
            raise ValueError(
                "batch_size must be smaller than the length of dataset, "
                f"but batch_size={batch_size} and len(dataset)={len(dataset)}"
            )

        self.key = jax.random.PRNGKey(seed) if key is None else key

    def sample(self) -> Tuple[bool, Tuple[jax.typing.ArrayLike, jax.typing.ArrayLike]]:
        raise NotImplementedError

    def __len__(self) -> int:
        return self.batch_count


class RandomShuffleSampler(DatasetSampler):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        key: jax.random.PRNGKey = None,
        seed: int = 0,
    ):
        super().__init__(dataset, batch_size, key, seed)

        self.batch_count = len(dataset) // self.batch_size

        self._reset()

    def _reset(self):
        self.current_batch = 0
        self.shuffled_indices = jax.random.permutation(self.key, len(self.dataset))
        _, self.key = jax.random.split(self.key)

    def sample(self) -> Tuple[bool, Tuple[jax.typing.ArrayLike, jax.typing.ArrayLike]]:
        if self.current_batch >= self.batch_count:
            self._reset()
            new_epoch = True
        else:
            new_epoch = False

        indices = self.shuffled_indices[
            self.current_batch
            * self.batch_size : (self.current_batch + 1)
            * self.batch_size
        ]
        self.current_batch += 1

        return (
            new_epoch,
            self.dataset[indices, ...],
        )


class WholeDatasetSampler(DatasetSampler):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int | None = None,
        key: jax.random.PRNGKey = None,
        seed: int = 0,
    ):
        super().__init__(dataset, batch_size, key, seed)

        if self.batch_size is None:
            self.batch_size = len(self.dataset)

    def sample(self) -> Tuple[bool, Tuple[jax.typing.ArrayLike, jax.typing.ArrayLike]]:
        return True, self.dataset[: self.batch_size, ...]


class SequentialSampler(DatasetSampler):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        key: jax.random.PRNGKey = None,
        seed: int = 0,
    ):
        super().__init__(dataset, batch_size, key, seed)

        self.batch_count = len(dataset) // self.batch_size

        self._reset()

    def _reset(self):
        self.current_batch = 0

    def sample(self) -> Tuple[bool, Tuple[jax.typing.ArrayLike, jax.typing.ArrayLike]]:
        if self.current_batch >= self.batch_count:
            self._reset()
            new_epoch = True
        else:
            new_epoch = False

        indices = slice(
            self.current_batch * self.batch_size,
            (self.current_batch + 1) * self.batch_size,
        )

        self.current_batch += 1

        return (
            new_epoch,
            self.dataset[indices, ...],
        )
