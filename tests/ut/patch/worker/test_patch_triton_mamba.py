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

import torch
import vllm.model_executor.layers.mamba.ops.ssd_chunk_scan as ssd_chunk_scan
import vllm.model_executor.layers.mamba.ops.ssd_combined as ssd_combined

from vllm_ascend.patch.worker import patch_triton


def test_chunk_scan_dummy_optional_kernel_args_are_non_null():
    reference = torch.empty(2, 3)

    actual = patch_triton._chunk_scan_optional_kernel_args(reference, None, (1, 1))

    assert actual.shape == (1, 1)
    assert actual.device == reference.device
    assert actual.dtype == reference.dtype


def test_chunk_scan_initial_states_dummy_reuses_states_pointer():
    states = torch.empty(2, 3, 4, 5)

    actual = patch_triton._chunk_scan_initial_states_kernel_arg(states, None)

    assert actual is states
    assert actual.data_ptr() == states.data_ptr()


def test_chunk_scan_patch_updates_combined_import_binding():
    assert ssd_chunk_scan._chunk_scan_fwd is patch_triton._chunk_scan_fwd_npu
    assert ssd_combined._chunk_scan_fwd is patch_triton._chunk_scan_fwd_npu
