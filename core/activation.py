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

from flax import linen as nn
from TFENN.util.enum_input import EnumInputFun


class ActivationType(EnumInputFun):
    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    TANH = "tanh"
    LOGISTIC = "logistic"
    NONE = "none"

    @classmethod
    @property
    def fun_map(cls) -> dict["ActivationType", nn.Module]:
        return {
            ActivationType.RELU: nn.activation.relu,
            ActivationType.LEAKY_RELU: nn.activation.leaky_relu,
            ActivationType.TANH: nn.activation.tanh,
            ActivationType.LOGISTIC: nn.activation.sigmoid,
            ActivationType.NONE: lambda x: x,
        }
