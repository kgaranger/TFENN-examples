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

import jax
from flax import linen as nn
from TFENN.util.enum_input import EnumInputClass


class InitializerType(EnumInputClass):
    NORMAL = "normal"
    UNIFORM = "uniform"
    XAVIER_UNIFORM = "xavier_uniform"
    XAVIER_NORMAL = "xavier_normal"
    HE_NORMAL = "he_normal"
    HE_UNIFORM = "he_uniform"
    LE_CUN_NORMAL = "le_cun_normal"
    LE_CUN_UNIFORM = "le_cun_uniform"
    ZERO = "zero"

    @classmethod
    @property
    def obj_map(cls) -> dict["InitializerType", jax.nn.initializers.Initializer]:
        return {
            InitializerType.NORMAL: nn.initializers.normal,
            InitializerType.UNIFORM: nn.initializers.uniform,
            InitializerType.XAVIER_UNIFORM: nn.initializers.xavier_uniform,
            InitializerType.XAVIER_NORMAL: nn.initializers.xavier_normal,
            InitializerType.HE_NORMAL: nn.initializers.he_normal,
            InitializerType.HE_UNIFORM: nn.initializers.he_uniform,
            InitializerType.LE_CUN_NORMAL: nn.initializers.lecun_normal,
            InitializerType.LE_CUN_UNIFORM: nn.initializers.lecun_uniform,
            InitializerType.ZERO: nn.initializers.zeros_init,
        }
