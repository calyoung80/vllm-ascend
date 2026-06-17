#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#

import os

from vllm_ascend.patch.platform.patch_mamba_config import _disable_batch_invariant_for_hybrid_mamba


def test_hybrid_mamba_disables_batch_invariant(monkeypatch):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")

    _disable_batch_invariant_for_hybrid_mamba()

    assert os.environ["VLLM_BATCH_INVARIANT"] == "0"
