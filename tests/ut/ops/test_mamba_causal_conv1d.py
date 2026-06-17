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

import inspect

import torch

from vllm_ascend.ops.triton.mamba.causal_conv1d import causal_conv1d_fn


def test_causal_conv1d_fn_accepts_vllm_block_cache_kwargs():
    signature = inspect.signature(causal_conv1d_fn)

    signature.bind(
        torch.empty(4, 8),
        torch.empty(4, 3),
        None,
        activation="silu",
        conv_states=torch.empty(1, 4, 2),
        has_initial_state=torch.empty(1, dtype=torch.bool),
        cache_indices=torch.empty(1, dtype=torch.int32),
        query_start_loc=torch.tensor([0, 8], dtype=torch.int32),
        block_idx_first_scheduled_token=torch.empty(1, dtype=torch.int32),
        block_idx_last_scheduled_token=torch.empty(1, dtype=torch.int32),
        initial_state_idx=torch.empty(1, dtype=torch.int32),
        num_computed_tokens=torch.empty(1, dtype=torch.int32),
        block_size_to_align=8,
        metadata=None,
    )
