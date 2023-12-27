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

import optax
from TFENN.util.enum_input import EnumInputClass


class SchedulerType(EnumInputClass):
    NONE = None
    COC = "coc"

    @classmethod
    @property
    def obj_map(cls) -> dict["SchedulerType", optax.Schedule]:
        return {
            SchedulerType.NONE: optax.constant_schedule,
            SchedulerType.COC: optax.cosine_onecycle_schedule,
        }
