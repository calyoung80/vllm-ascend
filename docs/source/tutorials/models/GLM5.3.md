# GLM-5.3

## 1 Introduction

[GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) uses the same base model as GLM-5.2 — every gain comes from post-training. Compared with GLM-5.2, it is much better at complex coding and long-horizon tasks.

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, multi-node deployment, accuracy and performance evaluation.

## 2 Supported Features

Refer to [Supported Features List](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [Feature Guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## 3 Prerequisites

### 3.1 Model Weight

|  Weight Version          | Hardware Requirements                                         | Download Links |
|--------------------------|---------------------------------------------------------------|----------------|
|  `GLM-5.3-w8a8c8`        | 2 Atlas 800I A3 (128GB × 8) node or 4 Atlas 800I A2 (64GB × 8) | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/GLM-5.3-w8a8c8) |

- You can use [msmodelslim](https://gitcode.com/Ascend/msmodelslim) to quantize the model directly.

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`.

>**Path description**: Download the model weights to a directory of your choice and record it. Ensure the model path in the subsequent deployment command matches this directory.

### 3.2 Verify Multi-node Communication (Optional)

If you want to deploy multi-node environment, you need to verify multi-node communication according to [verify multi-node communication environment](../../getting_started/installation.md#installation-multi-node-interconnect).

## 4 Installation

### 4.1 Docker Image Installation

- You can use our official docker image to run GLM-5.3 directly.

=== "A3 series"

    Start the docker image on each node.

    ```shell

    export IMAGE=quay.io/ascend/vllm-ascend:v0.30.0rc1-a3
    export NAME=vllm-ascend

    # Run the container using the defined variables
    # Note: If you are running bridge network with docker, please expose available ports for multiple nodes communication in advance
    docker run --rm \
    --name $NAME \
    --net=host \
    --shm-size=1g \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci4 \
    --device /dev/davinci5 \
    --device /dev/davinci6 \
    --device /dev/davinci7 \
    --device /dev/davinci8 \
    --device /dev/davinci9 \
    --device /dev/davinci10 \
    --device /dev/davinci11 \
    --device /dev/davinci12 \
    --device /dev/davinci13 \
    --device /dev/davinci14 \
    --device /dev/davinci15 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /root/.cache:/root/.cache \
    -it $IMAGE bash
    ```

=== "A2 series"

    Start the docker image on each of your nodes.

    ```shell

    export IMAGE=quay.io/ascend/vllm-ascend:v0.30.0rc1
    docker run --rm \
        --name vllm-ascend \
        --shm-size=1g \
        --net=host \
        --device /dev/davinci0 \
        --device /dev/davinci1 \
        --device /dev/davinci2 \
        --device /dev/davinci3 \
        --device /dev/davinci4 \
        --device /dev/davinci5 \
        --device /dev/davinci6 \
        --device /dev/davinci7 \
        --device /dev/davinci_manager \
        --device /dev/devmm_svm \
        --device /dev/hisi_hdc \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
        -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        -v /root/.cache:/root/.cache \
        -it $IMAGE bash
    ```

If you want to deploy multi-node environment, you need to set up environment on each node.

### 4.2 Source Code Installation

If you don't want to use the docker image as above, you can also build all from source:

- Install `vllm-ascend` from source, refer to [installation](../../getting_started/installation.md).

## 5 Online Service Deployment

### 5.1 Context Below 1M

#### 5.1.1 Multi-node Deployment

If you want to deploy multi-node environment, you need to verify multi-node communication according to [verify multi-node communication environment](../../getting_started/installation.md#installation-multi-node-interconnect). The scripts in Section 5.1.1 were tested on **v0.23.0**.

Common Issues Tip: If you encounter issues, Refer to [Public FAQs](../../faqs.md).

=== "A3 series"
    -  `GLM-5.3-w8a8c8`: can be deployed on 2 Atlas 800I A3 (64GB × 16).

    Run the following scripts on two nodes respectively.

    **node 0**

    ```shell
    # this obtained through ifconfig
    # nic_name is the network interface name corresponding to local_ip of the current node
    nic_name="xxx"
    local_ip="xxx"

    # The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
    node0_ip="xxxx"

    export HCCL_OP_EXPANSION_MODE="AIV"
    export HCCL_IF_IP=$local_ip
    export GLOO_SOCKET_IFNAME=$nic_name
    export TP_SOCKET_IFNAME=$nic_name
    export HCCL_SOCKET_IFNAME=$nic_name
    export HCCL_TRANSFER_TIMEOUT=600
    export HCCL_EXEC_TIMEOUT=3600
    export HCCL_CONNECT_TIMEOUT=3600
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=1
    export HCCL_BUFFSIZE=400
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export VLLM_ASCEND_ENABLE_MLAPO=1

    # Ensure the model path matches the directory recorded during download
    vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.3-w8a8c8 \
        --host 0.0.0.0 \
        --port 8000 \
        --safetensors-load-strategy prefetch \
        --api-server-count 1 \
        --data-parallel-size 8 \
        --data-parallel-start-rank 0 \
        --data-parallel-size-local 4 \
        --data-parallel-address $node0_ip \
        --data-parallel-rpc-port 12980 \
        --tensor-parallel-size 4 \
        --enable-expert-parallel \
        --seed 1024 \
        --served-model-name glm-5 \
        --tool-call-parser glm47 \
        --reasoning-parser glm47 \
        --enable-auto-tool-choice \
        --max-num-seqs 6 \
        --max-model-len 202752 \
        --max-num-batched-tokens 4096 \
        --trust-remote-code \
        --gpu-memory-utilization 0.92 \
        --quantization ascend \
        --enable-chunked-prefill \
        --enable-prefix-caching \
        --async-scheduling \
        --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
        --kv-cache-dtype int8 \
        --attention_config.indexer_kv_dtype int8 \
        --additional-config '{"enable_dsa_cp": true, "enable_balance_scheduling": true, "enable_fused_mc2": 1, "enable_flashcomm1": true}'  \
        --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}'
    ```

    **node 1**

    ```shell
    # this obtained through ifconfig
    # nic_name is the network interface name corresponding to local_ip of the current node
    nic_name="xxx"
    local_ip="xxx"

    # The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
    node0_ip="xxxx"

    export HCCL_OP_EXPANSION_MODE="AIV"
    export HCCL_IF_IP=$local_ip
    export GLOO_SOCKET_IFNAME=$nic_name
    export TP_SOCKET_IFNAME=$nic_name
    export HCCL_SOCKET_IFNAME=$nic_name
    export HCCL_TRANSFER_TIMEOUT=600
    export HCCL_EXEC_TIMEOUT=3600
    export HCCL_CONNECT_TIMEOUT=3600
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=1
    export HCCL_BUFFSIZE=400
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export VLLM_ASCEND_ENABLE_MLAPO=1

    # Ensure the model path matches the directory recorded during download
    vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.3-w8a8c8 \
        --host 0.0.0.0 \
        --port 8000 \
        --safetensors-load-strategy prefetch \
        --headless \
        --data-parallel-size 8 \
        --data-parallel-start-rank 4 \
        --data-parallel-size-local 4 \
        --data-parallel-address $node0_ip \
        --data-parallel-rpc-port 12980 \
        --tensor-parallel-size 4 \
        --enable-expert-parallel \
        --seed 1024 \
        --served-model-name glm-5 \
        --tool-call-parser glm47 \
        --reasoning-parser glm47 \
        --enable-auto-tool-choice \
        --max-num-seqs 6 \
        --max-model-len 202752 \
        --max-num-batched-tokens 4096 \
        --trust-remote-code \
        --gpu-memory-utilization 0.92 \
        --quantization ascend \
        --enable-chunked-prefill \
        --enable-prefix-caching \
        --async-scheduling \
        --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
        --kv-cache-dtype int8 \
        --attention_config.indexer_kv_dtype int8 \
        --additional-config '{"enable_dsa_cp": true, "enable_balance_scheduling": true, "enable_fused_mc2": 1, "enable_flashcomm1": true}' \
        --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}'
    ```

=== "A2 series"

    - `GLM-5.3-w8a8c8`: can be deployed on 4 Atlas 800I A2 (64GB × 8).

    Run the following scripts on four nodes respectively.

    **node 0**

    ```shell
    # this obtained through ifconfig
    # nic_name is the network interface name corresponding to local_ip of the current node
    nic_name="xxx"
    local_ip="xxx"

    # The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
    node0_ip="xxx"

    export HCCL_OP_EXPANSION_MODE="AIV"
    export HCCL_IF_IP=$local_ip
    export GLOO_SOCKET_IFNAME=$nic_name
    export TP_SOCKET_IFNAME=$nic_name
    export HCCL_SOCKET_IFNAME=$nic_name
    export VLLM_RPC_TIMEOUT=360000
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
    export HCCL_EXEC_TIMEOUT=200
    export HCCL_CONNECT_TIMEOUT=120
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=10
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export ACL_OP_INIT_MODE=1
    export CPU_AFFINITY_CONF=1
    export VLLM_ASCEND_ENABLE_MLAPO=1
    export VLLM_ENGINE_READY_TIMEOUT_S=1200

    # Ensure the model path matches the directory recorded during download
    vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.3-w8a8c8 \
        --host 0.0.0.0 \
        --port 8000 \
        --max-model-len 135000 \
        --data-parallel-size 4 \
        --data-parallel-size-local 1 \
        --data-parallel-start-rank 0 \
        --data-parallel-address "${node0_ip}" \
        --data-parallel-rpc-port 12980 \
        --tensor-parallel-size 8 \
        --enable-expert-parallel \
        --seed 1024 \
        --served-model-name glm-5 \
        --reasoning-parser glm47 \
        --safetensors-load-strategy prefetch \
        --max-num-seqs 128 \
        --max-num-batched-tokens 8192 \
        --trust-remote-code \
        --quantization ascend \
        --gpu-memory-utilization 0.92 \
        --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}' \
        --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
        --kv-cache-dtype int8 \
        --attention_config.indexer_kv_dtype int8 \
        --additional-config '{"enable_dsa_cp": true, "enable_balance_scheduling": true, "fuse_muls_add": true, "multistream_overlap_shared_expert": true, "enable_flashcomm1": true}' \
        --enable-prefix-caching \
        --async-scheduling \
        --api-server-count 1
    ```

    **node 1-3**

    ```shell
    # this obtained through ifconfig
    # nic_name is the network interface name corresponding to local_ip of the current node
    nic_name="xxx"
    local_ip="xxx"

    # The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
    node0_ip="xxx"

    # node1: dp_start_rank=1, node2: dp_start_rank=2, node3: dp_start_rank=3
    dp_start_rank=1

    export HCCL_OP_EXPANSION_MODE="AIV"
    export HCCL_IF_IP=$local_ip
    export GLOO_SOCKET_IFNAME=$nic_name
    export TP_SOCKET_IFNAME=$nic_name
    export HCCL_SOCKET_IFNAME=$nic_name
    export VLLM_RPC_TIMEOUT=360000
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
    export HCCL_EXEC_TIMEOUT=200
    export HCCL_CONNECT_TIMEOUT=120
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=10
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export ACL_OP_INIT_MODE=1
    export CPU_AFFINITY_CONF=1
    export VLLM_ASCEND_ENABLE_MLAPO=1
    export VLLM_ENGINE_READY_TIMEOUT_S=1200

    # Ensure the model path matches the directory recorded during download
    vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.3-w8a8c8 \
        --host 0.0.0.0 \
        --port 8000 \
        --headless \
        --max-model-len 135000 \
        --data-parallel-size 4 \
        --data-parallel-size-local 1 \
        --data-parallel-start-rank ${dp_start_rank} \
        --data-parallel-address "${node0_ip}" \
        --data-parallel-rpc-port 12980 \
        --tensor-parallel-size 8 \
        --enable-expert-parallel \
        --seed 1024 \
        --served-model-name glm-5 \
        --reasoning-parser glm47 \
        --safetensors-load-strategy prefetch \
        --max-num-seqs 128 \
        --max-num-batched-tokens 8192 \
        --trust-remote-code \
        --quantization ascend \
        --gpu-memory-utilization 0.92 \
        --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}' \
        --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
        --kv-cache-dtype int8 \
        --attention_config.indexer_kv_dtype int8 \
        --additional-config '{"enable_dsa_cp": true, "enable_balance_scheduling": true,"fuse_muls_add": true, "multistream_overlap_shared_expert": true, "enable_flashcomm1": true}' \
        --enable-prefix-caching \
        --async-scheduling
    ```

Key Parameter Descriptions:

Only the key parameters specific to this model/scenario are described below. max-model-len and max-num-seqs need to be set according to the actual usage scenario.

**Multi-node network and data parallel configuration:**

- `HCCL_IF_IP`, `GLOO_SOCKET_IFNAME`, `TP_SOCKET_IFNAME`, `HCCL_SOCKET_IFNAME`: Network interface configuration for multi-node communication. Set `nic_name` to the network interface name (obtained via `ifconfig`) and `local_ip` to the current node's IP address. These must be correctly configured on each node for successful multi-node communication.
- `--data-parallel-size 8`: Total number of data parallel ranks across all nodes (4 ranks per node in this scenario).
- `--data-parallel-size-local 4`: Number of data parallel ranks on the current node.
- `--data-parallel-start-rank`: Starting rank offset for data parallel ranks on this node. Node 0 uses `0`, node 1 uses `4`.
- `--data-parallel-address`: IP address of the data parallel master node (node 0). Must match the `local_ip` of the master node.
- `--data-parallel-rpc-port 12980`: RPC port for data parallel master communication. Must be the same across all nodes.
- `--headless`: Indicates a non-master node (used on node 1-3). Do not use on node 0.

**A2-specific environment variables:**

- `CPU_AFFINITY_CONF=1`: Enables CPU core affinity binding for worker processes.
- `ACL_OP_INIT_MODE=1`: ACL operator initialization mode to speed up operator compilation.
- `VLLM_RPC_TIMEOUT=360000` / `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000` / `HCCL_EXEC_TIMEOUT=200` / `HCCL_CONNECT_TIMEOUT=120` / `VLLM_ENGINE_READY_TIMEOUT_S=1200`: Timeout settings for multi-node startup and model execution on the slower A2 platform. Increase them if the engine fails to become ready in time.

**Notice:**
This scenario enables `additional_config.enable_fused_mc2=1` (fused `dispatch_ffn_combine`/`mega_moe` operators). Fused MC2 conflicts with `multistream_overlap_shared_expert` — the two optimizations must not be enabled at the same time (the runtime forcibly disables `multistream_overlap_shared_expert` when fused MC2 is on).

#### 5.1.2 Prefill-Decode Disaggregation

The A3 reference below serves GLM-5.3 W8A8C8 with a **200,000-token**
maximum sequence length for the 198K workload. It uses one Prefill engine
spanning P0/P1 with TP16 and PP2, and sixteen TP2 Decode engines on D0/D1.
The following A2 reference uses a separate eight-node topology. Section 5.1.3
adds MemCache KV pooling to the A3 deployment.

!!! caution

    Keep `MooncakeConnectorV2` for these P/D layouts. Do not substitute a V1
    or Hybrid connector: their port-derived remote-host lookup can select an
    incorrect peer in non-uniform DP x TP topologies and cause requests to hang.

##### 5.1.2.1 A3 Series: Topology and Prerequisites

| Role | Nodes | Global DP | TP | PP | Placement | Devices per role |
| --- | --- | --- | --- | --- | --- | --- |
| Prefill | P0, P1 | 1 | 16 | 2 | One PP stage per node | 32 |
| Decode | D0, D1 | 16 | 2 | 1 | Eight DP ranks per node | 32 |

Each container exposes devices 0-15. P0 exposes the only Prefill API;
P1 runs the second PP stage without an API server. Decode ranks 0-7 run
on D0 and ranks 8-15 on D1, with API ports 8000-8007 on each node.

Use the A3 `v0.30.0rc1-a3` image from Section 4.1 on all four nodes.

1. Prepare the same weights and software on all four nodes. Set `MODEL_PATH`
   and `PYTHON_LIB_DIR` in each role script to the installed paths. Install
   `fastokens` on Prefill nodes for `VLLM_USE_FASTOKENS=1`.
2. Verify [multi-node communication](../../getting_started/installation.md#installation-multi-node-interconnect).
   Set `LOCAL_IP="<PREFILL_NODE_IP>"` on each Prefill node and
   `LOCAL_IP="<DECODE_NODE_IP>"` on each Decode node to that node's IP.
   Set `NIC_NAME="<NETWORK_INTERFACE>"` to its matching interface and
   `MODEL_PATH="<YOUR_MODEL_PATH>"` to its model weight directory.
3. Reserve the PP master port 7060, Decode DP RPC port 16600, API ports,
   proxy port 8000, and the topology-derived Mooncake transfer/handshake ports.
   Prefill and Decode run on separate node groups.
4. Keep the 78-layer PP partition `42,36` consistent in the Prefill
   environment and both roles' connector topology. Both roles explicitly
   use int8 KV cache and int8 indexer KV cache for these W8A8C8 weights.

##### 5.1.2.2 Prepare the Scripts

Before you start, please

1. prepare the script `launch_online_dp.py` on each Decode node:

    ```python
    import argparse
    import multiprocessing
    import subprocess
    import sys


    def run_command(args, local_rank):
        dp_rank = args.dp_rank_start + local_rank
        devices = ",".join(str(i) for i in range(
            local_rank * args.tp_size, (local_rank + 1) * args.tp_size))
        subprocess.run([
            "bash", args.script, devices, str(args.vllm_start_port + local_rank),
            str(args.dp_size), str(dp_rank), args.dp_address,
            str(args.dp_rpc_port), str(args.tp_size),
        ], check=True)


    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("--dp-size", type=int, required=True)
        parser.add_argument("--tp-size", type=int, required=True)
        parser.add_argument("--dp-size-local", type=int, required=True)
        parser.add_argument("--dp-rank-start", type=int, required=True)
        parser.add_argument("--dp-address", required=True)
        parser.add_argument("--dp-rpc-port", type=int, default=16600)
        parser.add_argument("--vllm-start-port", type=int, default=8000)
        parser.add_argument("--script", default="./run_dp_template.sh")
        args = parser.parse_args()
        if min(args.dp_size, args.tp_size, args.dp_size_local) <= 0:
            parser.error("Parallel sizes must be positive")
        if args.dp_rank_start < 0 or args.dp_rank_start + args.dp_size_local > args.dp_size:
            parser.error("Local ranks must be within the global DP group")
        if args.dp_size_local * args.tp_size > 16:
            parser.error("This A3 configuration exposes 16 devices per node")
        processes = []
        for local_rank in range(args.dp_size_local):
            process = multiprocessing.Process(target=run_command, args=(args, local_rank))
            processes.append(process)
            process.start()
        for process in processes:
            process.join()
        return 1 if any(process.exitcode != 0 for process in processes) else 0


    if __name__ == "__main__":
        sys.exit(main())
    ```

2. prepare the script `run_dp_template.sh` on each node.

    1. Prefill node 0

        ```shell
        #!/usr/bin/env bash

        LOCAL_IP="<PREFILL_NODE_IP>"
        NIC_NAME="<NETWORK_INTERFACE>"
        MODEL_PATH="<YOUR_MODEL_PATH>"
        PYTHON_LIB_DIR=/path/to/python/lib

        # Set the current node IP and shared P0 master IP before starting.

        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export HCCL_EXEC_TIMEOUT=1800
        export HCCL_CONNECT_TIMEOUT=1800
        export ASCEND_TRANSFER_TIMEOUT=10000

        NODE_P0_IP="$LOCAL_IP"

        export VLLM_HOST_IP="$LOCAL_IP"
        export HCCL_IF_IP="$LOCAL_IP"
        export GLOO_SOCKET_IFNAME="$NIC_NAME"
        export TP_SOCKET_IFNAME="$NIC_NAME"
        export HCCL_SOCKET_IFNAME="$NIC_NAME"

        export LD_LIBRARY_PATH="${PYTHON_LIB_DIR}:/usr/local/lib:${LD_LIBRARY_PATH}"

        export HCCL_BUFFSIZE=1024
        export HCCL_OP_EXPANSION_MODE="AIV"

        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export TASK_QUEUE_ENABLE=1

        export VLLM_USE_V2_MODEL_RUNNER=0

        export VLLM_USE_FASTOKENS=1
        export VLLM_PP_LAYER_PARTITION="42,36"

        export ASCEND_LOCAL_COMM_RES='{"version":"1.2"}'

        exec vllm serve "${MODEL_PATH}" \
            --host 0.0.0.0 \
            --port 8000 \
            --tensor-parallel-size 16 \
            --enable-expert-parallel \
            --pipeline-parallel-size 2 \
            --distributed-executor-backend mp \
            --master-addr "$NODE_P0_IP" \
            --master-port 7060 \
            --nnodes 2 \
            --node-rank 0 \
            --enable-chunked-prefill \
            --enable-prefix-caching \
            --seed 1024 \
            --served-model-name glm5 \
            --max-model-len 200000 \
            --max-num-batched-tokens 16384 \
            --trust-remote-code \
            --max-num-seqs 64 \
            --gpu-memory-utilization 0.92 \
            --async-scheduling \
            --quantization ascend \
            --safetensors-load-strategy 'prefetch' \
            --enable-auto-tool-choice \
            --tool-call-parser glm47 \
            --reasoning-parser glm47 \
            --enforce-eager \
            --kv-cache-dtype int8 \
            --attention_config.indexer_kv_dtype int8 \
            --additional-config '{
                "enable_dsa_cp":true,
                "enable_fused_mc2": 1,
                "enable_flashcomm1": true
            }' \
            --speculative-config '{"num_speculative_tokens": 1,  "method":"deepseek_mtp","enforce_eager":true}' \
            --kv-transfer-config '{
                "kv_connector": "MooncakeConnectorV2",
                "kv_role": "kv_producer",
                "kv_port": "30000",
                "engine_id": "glm53-prefill",
                "kv_connector_extra_config": {
                    "use_ascend_direct": true,
                    "prefill": {"dp_size": 1, "pp_size": 2, "tp_size": 16, "pp_layer_partition": "42,36"},
                    "decode": {"dp_size": 16, "tp_size": 2}
                }
            }'
        ```

    2. Prefill node 1

        ```shell
        #!/usr/bin/env bash

        LOCAL_IP="<PREFILL_NODE_IP>"
        NIC_NAME="<NETWORK_INTERFACE>"
        MODEL_PATH="<YOUR_MODEL_PATH>"
        PYTHON_LIB_DIR=/path/to/python/lib

        # Set the current node IP and shared P0 master IP before starting.

        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export HCCL_EXEC_TIMEOUT=1800
        export HCCL_CONNECT_TIMEOUT=1800
        export ASCEND_TRANSFER_TIMEOUT=10000

        NODE_P0_IP="<PREFILL_NODE0_IP>"

        export VLLM_HOST_IP="$LOCAL_IP"
        export HCCL_IF_IP="$LOCAL_IP"
        export GLOO_SOCKET_IFNAME="$NIC_NAME"
        export TP_SOCKET_IFNAME="$NIC_NAME"
        export HCCL_SOCKET_IFNAME="$NIC_NAME"

        export LD_LIBRARY_PATH="${PYTHON_LIB_DIR}:/usr/local/lib:${LD_LIBRARY_PATH}"

        export HCCL_BUFFSIZE=1024
        export HCCL_OP_EXPANSION_MODE="AIV"

        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export TASK_QUEUE_ENABLE=1

        export VLLM_USE_V2_MODEL_RUNNER=0

        export VLLM_USE_FASTOKENS=1
        export VLLM_PP_LAYER_PARTITION="42,36"

        export ASCEND_LOCAL_COMM_RES='{"version":"1.2"}'

        exec vllm serve "${MODEL_PATH}" \
            --host 0.0.0.0 \
            --port 8000 \
            --tensor-parallel-size 16 \
            --enable-expert-parallel \
            --pipeline-parallel-size 2 \
            --distributed-executor-backend mp \
            --master-addr "$NODE_P0_IP" \
            --master-port 7060 \
            --nnodes 2 \
            --node-rank 1 \
            --enable-chunked-prefill \
            --enable-prefix-caching \
            --seed 1024 \
            --served-model-name glm5 \
            --max-model-len 200000 \
            --max-num-batched-tokens 16384 \
            --trust-remote-code \
            --max-num-seqs 64 \
            --gpu-memory-utilization 0.92 \
            --async-scheduling \
            --quantization ascend \
            --safetensors-load-strategy 'prefetch' \
            --enable-auto-tool-choice \
            --tool-call-parser glm47 \
            --reasoning-parser glm47 \
            --enforce-eager \
            --kv-cache-dtype int8 \
            --attention_config.indexer_kv_dtype int8 \
            --additional-config '{
                "enable_dsa_cp":true,
                "enable_fused_mc2": 1,
                "enable_flashcomm1": true
            }' \
            --speculative-config '{"num_speculative_tokens": 1,  "method":"deepseek_mtp","enforce_eager":true}' \
            --kv-transfer-config '{
                "kv_connector": "MooncakeConnectorV2",
                "kv_role": "kv_producer",
                "kv_port": "30000",
                "engine_id": "glm53-prefill",
                "kv_connector_extra_config": {
                    "use_ascend_direct": true,
                    "prefill": {"dp_size": 1, "pp_size": 2, "tp_size": 16, "pp_layer_partition": "42,36"},
                    "decode": {"dp_size": 16, "tp_size": 2}
                }
            }' \
            --headless
        ```

    3. Decode node 0

        ```shell
        #!/usr/bin/env bash

        LOCAL_IP="<DECODE_NODE_IP>"
        NIC_NAME="<NETWORK_INTERFACE>"
        MODEL_PATH="<YOUR_MODEL_PATH>"
        PYTHON_LIB_DIR=/path/to/python/lib

        # Arguments $1-$7 are supplied by launch_online_dp.py.

        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export HCCL_EXEC_TIMEOUT=1800
        export HCCL_CONNECT_TIMEOUT=1800
        export ASCEND_TRANSFER_TIMEOUT=10000


        export VLLM_HOST_IP="$LOCAL_IP"
        export HCCL_IF_IP="$LOCAL_IP"
        export GLOO_SOCKET_IFNAME="$NIC_NAME"
        export TP_SOCKET_IFNAME="$NIC_NAME"
        export HCCL_SOCKET_IFNAME="$NIC_NAME"

        export LD_LIBRARY_PATH="${PYTHON_LIB_DIR}:/usr/local/lib:${LD_LIBRARY_PATH}"

        export HCCL_BUFFSIZE=1024
        export HCCL_OP_EXPANSION_MODE="AIV"

        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export TASK_QUEUE_ENABLE=1

        export VLLM_USE_V2_MODEL_RUNNER=0
        export ASCEND_RT_VISIBLE_DEVICES=$1
        export ASCEND_LOCAL_COMM_RES='{"version":"1.2"}'

        exec vllm serve "${MODEL_PATH}" \
            --host 0.0.0.0 \
            --port $2 \
            --data-parallel-size $3 \
            --data-parallel-rank $4 \
            --data-parallel-address $5 \
            --data-parallel-rpc-port $6 \
            --tensor-parallel-size $7 \
            --enable-expert-parallel \
            --enable-chunked-prefill \
            --enable-prefix-caching \
            --seed 1024 \
            --served-model-name glm5 \
            --max-model-len 200000 \
            --max-num-batched-tokens 256 \
            --trust-remote-code \
            --max-num-seqs 64 \
            --gpu-memory-utilization 0.92 \
            --async-scheduling \
            --quantization ascend \
            --safetensors-load-strategy 'prefetch' \
            --enable-auto-tool-choice \
            --tool-call-parser glm47 \
            --reasoning-parser glm47 \
            --kv-cache-dtype int8 \
            --attention_config.indexer_kv_dtype int8 \
            --additional-config '{
                "recompute_scheduler_enable": true,
                "enable_fused_mc2": 1
            }' \
            --speculative-config '{"num_speculative_tokens": 5,  "method":"deepseek_mtp","enforce_eager":true}' \
            --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
            --kv-transfer-config '{
                "kv_connector": "MooncakeConnectorV2",
                "kv_role": "kv_consumer",
                "kv_port": "30100",
                "engine_id": "glm53-decode-dp'"$4"'",
                "kv_connector_extra_config": {
                    "use_ascend_direct": true,
                    "prefill": {"dp_size": 1, "pp_size": 2, "tp_size": 16, "pp_layer_partition": "42,36"},
                    "decode": {"dp_size": 16, "tp_size": 2}
                }
            }'
        ```

    4. Decode node 1

        ```shell
        #!/usr/bin/env bash

        LOCAL_IP="<DECODE_NODE_IP>"
        NIC_NAME="<NETWORK_INTERFACE>"
        MODEL_PATH="<YOUR_MODEL_PATH>"
        PYTHON_LIB_DIR=/path/to/python/lib

        # Arguments $1-$7 are supplied by launch_online_dp.py.

        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export HCCL_EXEC_TIMEOUT=1800
        export HCCL_CONNECT_TIMEOUT=1800
        export ASCEND_TRANSFER_TIMEOUT=10000


        export VLLM_HOST_IP="$LOCAL_IP"
        export HCCL_IF_IP="$LOCAL_IP"
        export GLOO_SOCKET_IFNAME="$NIC_NAME"
        export TP_SOCKET_IFNAME="$NIC_NAME"
        export HCCL_SOCKET_IFNAME="$NIC_NAME"

        export LD_LIBRARY_PATH="${PYTHON_LIB_DIR}:/usr/local/lib:${LD_LIBRARY_PATH}"

        export HCCL_BUFFSIZE=1024
        export HCCL_OP_EXPANSION_MODE="AIV"

        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export TASK_QUEUE_ENABLE=1

        export VLLM_USE_V2_MODEL_RUNNER=0
        export ASCEND_RT_VISIBLE_DEVICES=$1
        export ASCEND_LOCAL_COMM_RES='{"version":"1.2"}'

        exec vllm serve "${MODEL_PATH}" \
            --host 0.0.0.0 \
            --port $2 \
            --data-parallel-size $3 \
            --data-parallel-rank $4 \
            --data-parallel-address $5 \
            --data-parallel-rpc-port $6 \
            --tensor-parallel-size $7 \
            --enable-expert-parallel \
            --enable-chunked-prefill \
            --enable-prefix-caching \
            --seed 1024 \
            --served-model-name glm5 \
            --max-model-len 200000 \
            --max-num-batched-tokens 256 \
            --trust-remote-code \
            --max-num-seqs 64 \
            --gpu-memory-utilization 0.92 \
            --async-scheduling \
            --quantization ascend \
            --safetensors-load-strategy 'prefetch' \
            --enable-auto-tool-choice \
            --tool-call-parser glm47 \
            --reasoning-parser glm47 \
            --kv-cache-dtype int8 \
            --attention_config.indexer_kv_dtype int8 \
            --additional-config '{
                "recompute_scheduler_enable": true,
                "enable_fused_mc2": 1
            }' \
            --speculative-config '{"num_speculative_tokens": 5,  "method":"deepseek_mtp","enforce_eager":true}' \
            --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
            --kv-transfer-config '{
                "kv_connector": "MooncakeConnectorV2",
                "kv_role": "kv_consumer",
                "kv_port": "30100",
                "engine_id": "glm53-decode-dp'"$4"'",
                "kv_connector_extra_config": {
                    "use_ascend_direct": true,
                    "prefill": {"dp_size": 1, "pp_size": 2, "tp_size": 16, "pp_layer_partition": "42,36"},
                    "decode": {"dp_size": 16, "tp_size": 2}
                }
            }'
        ```

##### 5.1.2.3 Start the Engines

Once the preparation is done, you can start the server with the following command on each node:

1. Prefill node 0

    ```shell
    bash run_dp_template.sh
    ```

2. Prefill node 1

    ```shell
    bash run_dp_template.sh
    ```

3. Decode node 0

    ```shell
    D0_IP="<DECODE_NODE0_IP>"
    python launch_online_dp.py --dp-size 16 --tp-size 2 --dp-size-local 8 --dp-rank-start 0 --dp-address "$D0_IP" --dp-rpc-port 16600 --vllm-start-port 8000
    ```

4. Decode node 1

    ```shell
    D0_IP="<DECODE_NODE0_IP>"
    python launch_online_dp.py --dp-size 16 --tp-size 2 --dp-size-local 8 --dp-rank-start 8 --dp-address "$D0_IP" --dp-rpc-port 16600 --vllm-start-port 8000
    ```

##### 5.1.2.4 Start the Proxy

Replace each node's IP and installation-path placeholders before starting.
Run the four commands in Section 5.1.2.3 in separate terminals. Keep the
master processes running while the other nodes join; start both nodes of
each group before waiting for the group to become healthy.

After the Prefill API on P0:8000 and all sixteen Decode APIs respond
to `GET /v1/models`, start the
[PD load-balancing proxy](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py)
from the matching checkout on a separate host that can reach all P/D
API endpoints. The proxy listens on port 8000; using a separate host
avoids a port conflict with P0:8000 and the Decode APIs on 8000-8007.
Register P0 once and all sixteen Decode endpoints. P1 is not a separate
Prefill API endpoint.

```shell
P0_IP="<P0_IP>"
D0_IP="<D0_IP>"
D1_IP="<D1_IP>"

python examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
    --host 0.0.0.0 --port 8000 \
    --prefiller-hosts "$P0_IP" --prefiller-ports 8000 \
    --decoder-hosts "$D0_IP" "$D0_IP" "$D0_IP" "$D0_IP" \
        "$D0_IP" "$D0_IP" "$D0_IP" "$D0_IP" \
        "$D1_IP" "$D1_IP" "$D1_IP" "$D1_IP" \
        "$D1_IP" "$D1_IP" "$D1_IP" "$D1_IP" \
    --decoder-ports 8000 8001 8002 8003 8004 8005 8006 8007 \
        8000 8001 8002 8003 8004 8005 8006 8007
```

Key parameters:

- Prefill uses TP16/PP2 over two nodes with the `mp` executor and
  `VLLM_PP_LAYER_PARTITION="42,36"`. Its batch budget is 16,384 tokens,
  maximum sequence count is 64, and it runs in eager mode with one MTP token.
- Decode uses DP16/TP2, a 256-token batch budget, 64 sequences, five MTP
  tokens with an eager draft model, and `FULL_DECODE_ONLY` graph mode.
  `recompute_scheduler_enable=true` allows recomputation scheduling.
- Both roles retain chunked prefill, prefix caching, async scheduling,
  seed 1024, memory utilization 0.92, and fused MC2. Prefill also enables
  DSA CP and FlashComm1. Both use `glm47` tool and reasoning parsers and
  the served model name `glm5`.
- `MooncakeConnectorV2` declares the same Prefill DP1/PP2/TP16 and
  Decode DP16/TP2 topology on both sides. The non-pooled base KV ports
  are 30000 on Prefill and 30100 on Decode; reserve derived worker ports.
- The examples give Prefill and each Decode engine distinct `engine_id`
  values so P-to-D peer registration cannot conflict.

##### 5.1.2.5 A2 Series: 8-Node Deployment

The following reference deploys GLM-5.3 W8A8C8 prefill-decode disaggregation
on **eight Atlas 800 A2 (64GB × 8) nodes** running vLLM v0.30.0. It was
validated end-to-end (launch, serving, benchmark). Because one A2 node holds
only 8 × 64GB devices, neither role can keep the whole checkpoint (roughly
700 GB on disk) inside a single node group the way the A3 guide does, so
**both roles split across four nodes each** and rely on expert parallelism:
with `--enable-expert-parallel`, the MoE expert weights are distributed
across all DP×TP ranks, so each decode engine only holds its own expert
shard plus the TP-sharded dense part, which is what makes the 2-device
decode engines fit.

This follows the same launcher-based pattern as the 8-node A2 guide in
[GLM-5.2](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/tutorials/models/GLM5.2.md#5132-deployment-on-8-atlas-800-a2)
(`launch_online_dp.py` on the Decode side, reused as-is), but the
topology differs: GLM-5.2 A2 uses Prefill DP4×TP8 / Decode DP8×TP4 with
one prefill DP rank per node, while GLM-5.3 W8A8C8 needs TP16 per
prefill DP rank, so each prefill DP rank spans two nodes via
`--nnodes 4`.

| Role | Nodes | Global DP | TP | Placement | Devices per role |
| --- | --- | --- | --- | --- | --- |
| Prefill | 4 nodes | 2 | 16 | One DP rank spans two nodes (nodes 0+1 = DP rank 0, nodes 2+3 = DP rank 1) | 32 |
| Decode | 4 nodes | 16 | 2 | Four DP ranks per node (ranks 0-3, 4-7, 8-11, 12-15) | 32 |

Key characteristics, and where this reference differs from typical
single-node-group guides:

- **Prefill = TP16 × DP2 over 4 nodes.** Each prefill DP rank is one
  TP16 group spanning two nodes (`--nnodes 4`, `mp` backend). Node 0
  hosts the only Prefill API server (`--api-server-count 1`); the other
  three nodes run `--headless`.
- **Decode = DP16 × TP2 over 4 nodes.** Each decode engine is a
  2-device engine on a single node; sixteen engines are coordinated by
  one DP RPC master. This is the same `launch_online_dp.py` pattern as
  the GLM-5.3-Flash PD guide.
- **No pipeline parallelism.** TP16 spans two nodes directly over the
  node-to-node interconnect, so no PP partition is configured.
- **MooncakeConnectorV2** with `engine_id` `"0"` (Prefill) and `"1"`
  (Decode).

!!! warning "Version requirements"

    This reference configuration was validated on **vLLM v0.30.0** with the
    V1 model runner (`VLLM_USE_V2_MODEL_RUNNER=0`). The flag surface differs from other
    releases: on 0.30.0 multi-node `--nnodes` forces TCP rendezvous, and
    `--headless` must be passed on every non-API-server node.

Prerequisites:

1. Prepare the same weights and software on all eight nodes. Set
   `MODEL_PATH` to the installed GLM-5.3-w8a8c8 weight directory.
2. Verify [multi-node communication](../../getting_started/installation.md#installation-multi-node-interconnect).
   The Prefill command takes the node's own IP as `$3`; each Decode node
   sets its own `LOCAL_IP` in its local engine script before invoking
   `launch_online_dp.py`.
3. Reserve the Prefill master port 29501 (TCP store for `--nnodes 4`),
   Prefill DP RPC port 12980, Decode DP RPC port 12990, Decode API ports
   10010-10013 per node, Prefill API port 8077, proxy port 8000, and the
   topology-derived Mooncake transfer/handshake ports (base 30000 Prefill
   / 30100 Decode; each engine derives `kv_port + dp_rank × tp × pp`).
4. Both roles use int8 KV cache and int8 indexer KV cache for these
   W8A8C8 weights.

###### 5.1.2.5.1 Prepare the Decode DP Launcher

Save the `launch_online_dp.py` launcher from Section 5.1.2.2 on all four
Decode nodes. It starts one process per local DP rank and propagates child
failures. Prefill does not use this launcher (its
`--nnodes 4` is handled by `vllm serve` directly).

This reference invokes the launcher with the A2 topology (see the
commands in Section 5.1.2.5.4): `--dp-size 16 --tp-size 2 --dp-size-local 4`,
`--dp-rank-start` 0/4/8/12 per node, DP RPC port 12990 on
`<DECODE_NODE0_IP>`, and API ports 10010-10013.

The Decode script follows the GLM-5.2 `run_dp_template.sh` convention: the
launcher supplies `$1`-`$7`, while `LOCAL_IP`, `NIC_NAME`, and `MODEL_PATH`
are node-level settings. Prefill is intentionally different because TP16 with
DP2 spans four nodes; `vllm serve` must receive `--nnodes 4` and `--node-rank`
directly instead of being started as independent DP processes.

###### 5.1.2.5.2 Prefill Engine Script

Save the following complete script as `start_p_engine.sh`. One copy
serves all four Prefill nodes; the node role is selected by `$1`
(`node_rank`) and the local visible devices are supplied as `$4`.

```shell
#!/usr/bin/env bash
set -euo pipefail

NODE_RANK=${1:?need node_rank}
PORT=${2:-8077}
LOCAL_IP=${3:?need local_ip}
VISIBLE_DEVICES=${4:-0,1,2,3,4,5,6,7}

# ---- Per-node role (the only difference between the four nodes) ----
case "$NODE_RANK" in
  0)
    HEADLESS_ARGS=""                  # node 0: hosts the API server
    ;;
  1|2|3)
    HEADLESS_ARGS="--headless"        # nodes 1-3: headless workers
    ;;
  *)
    echo "node_rank must be 0-3" >&2
    exit 1
    ;;
esac
API_SERVER_COUNT=""
if [ "$NODE_RANK" = "0" ]; then
  API_SERVER_COUNT="--api-server-count 1"
fi

NIC_NAME="<NETWORK_INTERFACE>"
P_MASTER_IP="<PREFILL_NODE0_IP>"
MODEL_PATH="<YOUR_MODEL_PATH>"

unset ftp_proxy FTP_PROXY https_proxy HTTPS_PROXY http_proxy HTTP_PROXY
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

# Cross-node TP16: do NOT set HCCL_IF_IP here. Device-side HCCL
# communication still auto-selects the NPU RoCE interface; pinning
# HCCL_IF_IP to the business IP fails with EI0006, pinning it to the
# RoCE IP fails with EI0001. Only the host-side topology socket is
# pinned, because a host-agent vNIC can otherwise be incorrectly selected and
# hang the startup in SYN_SENT.
export HCCL_SOCKET_IFNAME=$NIC_NAME
export VLLM_HOST_IP=$LOCAL_IP
export GLOO_SOCKET_IFNAME=$NIC_NAME
export TP_SOCKET_IFNAME=$NIC_NAME
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=7200
export HCCL_CONNECT_TIMEOUT=7200
export VLLM_RPC_TIMEOUT=360000
export VLLM_ENGINE_READY_TIMEOUT_S=1200
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export VLLM_USE_V2_MODEL_RUNNER=0
export CPU_AFFINITY_CONF=1
export ACL_OP_INIT_MODE=1
export ASCEND_RT_VISIBLE_DEVICES="$VISIBLE_DEVICES"
export VLLM_USE_FASTOKENS=1

exec vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --data-parallel-size 2 \
  --tensor-parallel-size 16 \
  --nnodes 4 \
  --node-rank "$NODE_RANK" \
  --master-addr "$P_MASTER_IP" \
  --master-port 29501 \
  --data-parallel-address "$P_MASTER_IP" \
  --data-parallel-rpc-port 12980 \
  --enable-expert-parallel \
  --quantization ascend \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --seed 1024 \
  --served-model-name glm5 \
  --max-model-len 135000 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.92 \
  --trust-remote-code \
  --async-scheduling \
  --enforce-eager \
  --safetensors-load-strategy prefetch \
  --kv-cache-dtype int8 \
  --attention_config.indexer_kv_dtype int8 \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm47 \
  --speculative-config \
  '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}' \
  --additional-config \
  '{"enable_dsa_cp": true, "enable_fused_mc2": 1, "enable_flashcomm1": true, "ascend_compilation_config": {"fuse_muls_add": true}}' \
  --kv-transfer-config \
  '{
    "kv_connector": "MooncakeConnectorV2",
    "kv_role": "kv_producer",
    "kv_port": "30000",
    "engine_id": "0",
    "kv_connector_extra_config": {
      "use_ascend_direct": true,
      "prefill": {"dp_size": 2, "tp_size": 16},
      "decode": {"dp_size": 16, "tp_size": 2}
    }
  }' \
  $API_SERVER_COUNT $HEADLESS_ARGS
```

###### 5.1.2.5.3 Decode Engine Script

Save the following complete script as `start_d_engine.sh` on all four
Decode nodes. Set `LOCAL_IP` to the current node's IP in each copy;
`launch_online_dp.py` supplies `$1`-`$7`.

```shell
#!/usr/bin/env bash
set -euo pipefail

VISIBLE_DEVICES=$1
PORT=$2
DP_SIZE=$3
DP_RANK=$4
DP_ADDRESS=$5
DP_RPC_PORT=$6
TP_SIZE=$7
MODEL_PATH="<YOUR_MODEL_PATH>"

LOCAL_IP="<CURRENT_DECODE_NODE_IP>"
NIC_NAME="<NETWORK_INTERFACE>"

unset ftp_proxy FTP_PROXY https_proxy HTTPS_PROXY http_proxy HTTP_PROXY
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

# Intra-node TP2: HCCL_IF_IP to the local business IP is fine here.
export VLLM_HOST_IP=$LOCAL_IP
export HCCL_IF_IP=$LOCAL_IP
export GLOO_SOCKET_IFNAME=$NIC_NAME
export TP_SOCKET_IFNAME=$NIC_NAME
export HCCL_SOCKET_IFNAME=$NIC_NAME
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=7200
export HCCL_CONNECT_TIMEOUT=7200
export VLLM_RPC_TIMEOUT=360000
export VLLM_ENGINE_READY_TIMEOUT_S=1200
export VLLM_ASCEND_ENABLE_MLAPO=1
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export VLLM_USE_V2_MODEL_RUNNER=0
export CPU_AFFINITY_CONF=1
export ACL_OP_INIT_MODE=1
export ASCEND_RT_VISIBLE_DEVICES="$VISIBLE_DEVICES"

exec vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --data-parallel-size "$DP_SIZE" \
  --data-parallel-rank "$DP_RANK" \
  --data-parallel-address "$DP_ADDRESS" \
  --data-parallel-rpc-port "$DP_RPC_PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  --enable-expert-parallel \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --seed 1024 \
  --served-model-name glm5 \
  --max-model-len 135000 \
  --max-num-batched-tokens 256 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.92 \
  --trust-remote-code \
  --async-scheduling \
  --quantization ascend \
  --safetensors-load-strategy prefetch \
  --kv-cache-dtype int8 \
  --attention_config.indexer_kv_dtype int8 \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm47 \
  --additional-config \
  '{"recompute_scheduler_enable": true, "enable_sparse_sfa_c8": true, "enable_sparse_li_c8": true, "enable_fused_mc2": 1}' \
  --speculative-config \
  '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}' \
  --compilation-config \
  '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --kv-transfer-config \
  '{
    "kv_connector": "MooncakeConnectorV2",
    "kv_role": "kv_consumer",
    "kv_port": "30100",
    "engine_id": "1",
    "kv_connector_extra_config": {
      "use_ascend_direct": true,
      "prefill": {"dp_size": 2, "tp_size": 16},
      "decode": {"dp_size": 16, "tp_size": 2}
    }
  }'
```

###### 5.1.2.5.4 Start the Engines

Start the Decode group first, then Prefill. The Decode DP master waits
for all sixteen engines to register, so the four Decode nodes should be
started nearly simultaneously.

1. Decode node 0 (DP ranks 0-3, hosts the DP master):

    ```shell
    python launch_online_dp.py --script ./start_d_engine.sh --dp-size 16 --tp-size 2 --dp-size-local 4 --dp-rank-start 0 --dp-address "<DECODE_NODE0_IP>" --dp-rpc-port 12990 --vllm-start-port 10010
    ```

2. Decode node 1 (DP ranks 4-7), node 2 (8-11), and node 3 (12-15):

    Run the corresponding command directly on each node. Replace the
    `LOCAL_IP` and `DP_RANK_START` values for that node.

    ```shell
    # Decode node 1
    python launch_online_dp.py --script ./start_d_engine.sh --dp-size 16 --tp-size 2 --dp-size-local 4 --dp-rank-start 4 --dp-address "<DECODE_NODE0_IP>" --dp-rpc-port 12990 --vllm-start-port 10010

    # Decode node 2
    python launch_online_dp.py --script ./start_d_engine.sh --dp-size 16 --tp-size 2 --dp-size-local 4 --dp-rank-start 8 --dp-address "<DECODE_NODE0_IP>" --dp-rpc-port 12990 --vllm-start-port 10010

    # Decode node 3
    python launch_online_dp.py --script ./start_d_engine.sh --dp-size 16 --tp-size 2 --dp-size-local 4 --dp-rank-start 12 --dp-address "<DECODE_NODE0_IP>" --dp-rpc-port 12990 --vllm-start-port 10010
    ```

3. Prefill node 0 (API server) first, then nodes 2, 1, and 3 (leader before
   workers). Run each command directly on its target node:

    ```shell
    # Prefill node 0
    bash start_p_engine.sh 0 8077 <PREFILL_NODE0_IP> 0,1,2,3,4,5,6,7

    # Prefill node 2
    bash start_p_engine.sh 2 8077 <PREFILL_NODE2_IP> 0,1,2,3,4,5,6,7

    # Prefill node 1
    bash start_p_engine.sh 1 8077 <PREFILL_NODE1_IP> 0,1,2,3,4,5,6,7

    # Prefill node 3
    bash start_p_engine.sh 3 8077 <PREFILL_NODE3_IP> 0,1,2,3,4,5,6,7
    ```

For background launches inside containers, prefer `docker exec -d`
(`nohup ... &` may not survive a non-interactive `docker exec`).

###### 5.1.2.5.5 Start the Proxy

After the Prefill API (`http://<PREFILL_NODE0_IP>:8077/v1/models`) and
all sixteen Decode APIs (`http://<node_ip>:1001x/v1/models`) answer,
start the
[PD load-balancing proxy](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py)
on Prefill node 0. The proxy's `--decoder-hosts`/`--decoder-ports` are
zipped 1:1 with a length check — all sixteen Decode endpoints must be
listed explicitly:

```shell
python examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py --host 0.0.0.0 --port 8000 --workers 8 --prefiller-hosts <PREFILL_NODE0_IP> --prefiller-ports 8077 --decoder-hosts <DECODE_NODE0_IP> <DECODE_NODE0_IP> <DECODE_NODE0_IP> <DECODE_NODE0_IP> <DECODE_NODE1_IP> <DECODE_NODE1_IP> <DECODE_NODE1_IP> <DECODE_NODE1_IP> <DECODE_NODE2_IP> <DECODE_NODE2_IP> <DECODE_NODE2_IP> <DECODE_NODE2_IP> <DECODE_NODE3_IP> <DECODE_NODE3_IP> <DECODE_NODE3_IP> <DECODE_NODE3_IP> --decoder-ports 10010 10011 10012 10013 10010 10011 10012 10013 10010 10011 10012 10013 10010 10011 10012 10013
```

Key parameters:

- **Expert parallelism is what makes the topology fit**: with
  `--enable-expert-parallel` on both roles, the MoE expert weights are
  spread across all DP×TP ranks, so each 2-device Decode engine holds
  only its expert shard plus the dense part of the W8A8C8 checkpoint.
- Prefill runs TP16×DP2 over four nodes (`--nnodes 4`, `mp` backend,
  TCP store on node 0:29501, DP RPC 12980). Node 0 is the only API
  server (`--api-server-count 1`); the other three nodes are
  `--headless`. It runs eager with a 135,000-token model length, an
  8,192-token batch budget, 64 sequences, 3 MTP tokens, and DSA CP,
  fused MC2, and FlashComm1.
- Decode runs DP16×TP2 with a 135,000-token model length, a 256-token batch
  budget, 32 sequences, 3 MTP tokens with an eager draft,
  `FULL_DECODE_ONLY` graph mode, and
  `recompute_scheduler_enable=true`.
- On the Prefill side, do **not** set `HCCL_IF_IP` (see the comment in
  Section 5.1.2.5.2): device-side HCCL auto-selects the NPU RoCE interface, and
  pinning it to the business or RoCE IP breaks startup (EI0006/EI0001).
  Pin only `HCCL_SOCKET_IFNAME` to the business NIC so the host-side
  topology socket cannot bind a host-agent vNIC. The Decode side is
  intra-node TP2, where `HCCL_IF_IP=<local business IP>` is fine.
- Both roles declare the same Mooncake topology
  (`prefill {dp_size 2, tp_size 16}` / `decode {dp_size 16, tp_size 2}`)
  with `engine_id` `"0"`/`"1"`. Base KV ports are 30000 (Prefill) and
  30100 (Decode); each engine derives its side-channel port as
  `kv_port + dp_rank × tp × pp`, so reserve the derived ranges too.
- Both roles use `glm47` tool parser and `glm47` reasoning parser and
  the served model name `glm5`, int8 KV cache + int8 indexer KV cache,
  chunked prefill, prefix caching, async scheduling, seed 1024, memory
  utilization 0.92, and fused MC2.
- `VLLM_USE_V2_MODEL_RUNNER=0` selects the V1 model runner for this
  validated configuration.

#### 5.1.3 MemCache KV Cache Pool Deployment

This A3 reference extends the same four-node PP/PD topology with
`MultiConnector`: `MooncakeConnectorV2` transfers KV cache from Prefill to
Decode, and `AscendStoreConnector` provides non-layerwise MemCache pooling.
The dedicated eight-node A2 pooling reference is in Section 5.1.3.5. Retain
the launcher, PP placement, and proxy routing from Section 5.1.2 for this A3
reference.

##### 5.1.3.1 Prepare MemCache

Complete [MemCache backend setup](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md#scenario-2-memcache-backend),
including the hardware/CANN prerequisites, `memfabric-hybrid` and
`memcache-hybrid` installation, configuration files, and metadata service.
All P/D instances must access the same pool; connector arguments do not
start the MemCache service. Set `MEMCACHE_ROOT` to the installed
`memcache_hybrid` directory and `PYTHON_LIB_DIR` to the library directory
of the Python installation actually used by vLLM on each node.

##### 5.1.3.2 Prepare the Scripts

Before you start, please

1. prepare the script `launch_online_dp.py` on each Decode node, using the
   launcher in Section 5.1.2.2.

2. prepare the pooled script `run_dp_template.sh` on each node.

    1. Prefill node 0

        ```shell
        #!/usr/bin/env bash

        LOCAL_IP="<PREFILL_NODE_IP>"
        NIC_NAME="<NETWORK_INTERFACE>"
        MODEL_PATH="<YOUR_MODEL_PATH>"
        PYTHON_LIB_DIR=/path/to/python/lib
        MEMCACHE_ROOT=/path/to/site-packages/memcache_hybrid

        # Set the current node IP and shared P0 master IP before starting.

        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export HCCL_EXEC_TIMEOUT=1800
        export HCCL_CONNECT_TIMEOUT=1800
        export ASCEND_TRANSFER_TIMEOUT=10000

        NODE_P0_IP="$LOCAL_IP"

        export VLLM_HOST_IP="$LOCAL_IP"
        export HCCL_IF_IP="$LOCAL_IP"
        export GLOO_SOCKET_IFNAME="$NIC_NAME"
        export TP_SOCKET_IFNAME="$NIC_NAME"
        export HCCL_SOCKET_IFNAME="$NIC_NAME"

        export LD_LIBRARY_PATH="${PYTHON_LIB_DIR}:/usr/local/lib:${LD_LIBRARY_PATH}"

        export HCCL_BUFFSIZE=1024
        export HCCL_OP_EXPANSION_MODE="AIV"

        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export TASK_QUEUE_ENABLE=1

        export VLLM_USE_V2_MODEL_RUNNER=0

        export VLLM_USE_FASTOKENS=1
        export VLLM_PP_LAYER_PARTITION="42,36"

        export PYTHONHASHSEED=0
        export ACL_OP_INIT_MODE=1
        export MMC_LOCAL_CONFIG_PATH="${MEMCACHE_ROOT}/config/mmc-local.conf"
        export LD_LIBRARY_PATH="${MEMCACHE_ROOT}/lib:${PYTHON_LIB_DIR}:${LD_LIBRARY_PATH}"

        export ASCEND_LOCAL_COMM_RES='{"version":"1.2"}'

        exec vllm serve "${MODEL_PATH}" \
            --host 0.0.0.0 \
            --port 8000 \
            --tensor-parallel-size 16 \
            --enable-expert-parallel \
            --pipeline-parallel-size 2 \
            --distributed-executor-backend mp \
            --master-addr "$NODE_P0_IP" \
            --master-port 7060 \
            --nnodes 2 \
            --node-rank 0 \
            --enable-chunked-prefill \
            --async-scheduling \
            --enable-prefix-caching \
            --seed 1024 \
            --served-model-name glm5 \
            --max-model-len 200000 \
            --max-num-batched-tokens 16384 \
            --trust-remote-code \
            --max-num-seqs 64 \
            --gpu-memory-utilization 0.92 \
            --quantization ascend \
            --safetensors-load-strategy 'prefetch' \
            --enable-auto-tool-choice \
            --tool-call-parser glm47 \
            --reasoning-parser glm47 \
            --enforce-eager \
            --kv-cache-dtype int8 \
            --attention_config.indexer_kv_dtype int8 \
            --additional-config '{
                "enable_dsa_cp":true,
                "enable_fused_mc2": 1,
                "enable_flashcomm1": true
            }' \
            --speculative-config '{"num_speculative_tokens": 1,  "method":"deepseek_mtp","enforce_eager":true}' \
            --kv-transfer-config '{
                "kv_connector": "MultiConnector",
                "kv_role": "kv_producer",
                "kv_port": "30000",
                "engine_id": "glm53-prefill",
                "kv_connector_extra_config": {
                    "connectors":[
                    {
                        "kv_connector": "MooncakeConnectorV2",
                        "kv_role": "kv_producer",
                        "kv_port": "30000",
                        "kv_connector_extra_config": {
                            "use_ascend_direct": true,
                            "prefill": {"dp_size": 1, "pp_size": 2, "tp_size": 16, "pp_layer_partition": "42,36"},
                            "decode": { "dp_size": 16, "tp_size": 2}
                        }
                    },
                    {
                        "kv_connector": "AscendStoreConnector",
                        "kv_role": "kv_producer",
                        "kv_connector_extra_config": {
                            "lookup_rpc_port":"0",
                            "backend": "memcache",
                            "use_layerwise": false
                        }
                    }
                    ]
                }
            }'
        ```

    2. Prefill node 1

        ```shell
        #!/usr/bin/env bash

        LOCAL_IP="<PREFILL_NODE_IP>"
        NIC_NAME="<NETWORK_INTERFACE>"
        MODEL_PATH="<YOUR_MODEL_PATH>"
        PYTHON_LIB_DIR=/path/to/python/lib
        MEMCACHE_ROOT=/path/to/site-packages/memcache_hybrid

        # Set the current node IP and shared P0 master IP before starting.

        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export HCCL_EXEC_TIMEOUT=1800
        export HCCL_CONNECT_TIMEOUT=1800
        export ASCEND_TRANSFER_TIMEOUT=10000

        NODE_P0_IP="<PREFILL_NODE0_IP>"

        export VLLM_HOST_IP="$LOCAL_IP"
        export HCCL_IF_IP="$LOCAL_IP"
        export GLOO_SOCKET_IFNAME="$NIC_NAME"
        export TP_SOCKET_IFNAME="$NIC_NAME"
        export HCCL_SOCKET_IFNAME="$NIC_NAME"

        export LD_LIBRARY_PATH="${PYTHON_LIB_DIR}:/usr/local/lib:${LD_LIBRARY_PATH}"

        export HCCL_BUFFSIZE=1024
        export HCCL_OP_EXPANSION_MODE="AIV"

        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export TASK_QUEUE_ENABLE=1

        export VLLM_USE_V2_MODEL_RUNNER=0

        export VLLM_USE_FASTOKENS=1
        export VLLM_PP_LAYER_PARTITION="42,36"

        export PYTHONHASHSEED=0
        export ACL_OP_INIT_MODE=1
        export MMC_LOCAL_CONFIG_PATH="${MEMCACHE_ROOT}/config/mmc-local.conf"
        export LD_LIBRARY_PATH="${MEMCACHE_ROOT}/lib:${PYTHON_LIB_DIR}:${LD_LIBRARY_PATH}"

        export ASCEND_LOCAL_COMM_RES='{"version":"1.2"}'

        exec vllm serve "${MODEL_PATH}" \
            --host 0.0.0.0 \
            --port 8000 \
            --tensor-parallel-size 16 \
            --enable-expert-parallel \
            --pipeline-parallel-size 2 \
            --distributed-executor-backend mp \
            --master-addr "$NODE_P0_IP" \
            --master-port 7060 \
            --nnodes 2 \
            --node-rank 1 \
            --enable-chunked-prefill \
            --async-scheduling \
            --enable-prefix-caching \
            --seed 1024 \
            --served-model-name glm5 \
            --max-model-len 200000 \
            --max-num-batched-tokens 16384 \
            --trust-remote-code \
            --max-num-seqs 64 \
            --gpu-memory-utilization 0.92 \
            --quantization ascend \
            --safetensors-load-strategy 'prefetch' \
            --enable-auto-tool-choice \
            --tool-call-parser glm47 \
            --reasoning-parser glm47 \
            --enforce-eager \
            --kv-cache-dtype int8 \
            --attention_config.indexer_kv_dtype int8 \
            --additional-config '{
                "enable_dsa_cp":true,
                "enable_fused_mc2": 1,
                "enable_flashcomm1": true
            }' \
            --speculative-config '{"num_speculative_tokens": 1,  "method":"deepseek_mtp","enforce_eager":true}' \
            --kv-transfer-config '{
                "kv_connector": "MultiConnector",
                "kv_role": "kv_producer",
                "kv_port": "30000",
                "engine_id": "glm53-prefill",
                "kv_connector_extra_config": {
                    "connectors":[
                    {
                        "kv_connector": "MooncakeConnectorV2",
                        "kv_role": "kv_producer",
                        "kv_port": "30000",
                        "kv_connector_extra_config": {
                            "use_ascend_direct": true,
                            "prefill": {"dp_size": 1, "pp_size": 2, "tp_size": 16, "pp_layer_partition": "42,36"},
                            "decode": { "dp_size": 16, "tp_size": 2}
                        }
                    },
                    {
                        "kv_connector": "AscendStoreConnector",
                        "kv_role": "kv_producer",
                        "kv_connector_extra_config": {
                            "lookup_rpc_port":"0",
                            "backend": "memcache",
                            "use_layerwise": false
                        }
                    }
                    ]
                }
            }' \
            --headless
        ```

    3. Decode node 0

        ```shell
        #!/usr/bin/env bash

        LOCAL_IP="<DECODE_NODE_IP>"
        NIC_NAME="<NETWORK_INTERFACE>"
        MODEL_PATH="<YOUR_MODEL_PATH>"
        PYTHON_LIB_DIR=/path/to/python/lib
        MEMCACHE_ROOT=/path/to/site-packages/memcache_hybrid

        # Arguments $1-$7 are supplied by launch_online_dp.py.

        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export HCCL_EXEC_TIMEOUT=1800
        export HCCL_CONNECT_TIMEOUT=1800
        export ASCEND_TRANSFER_TIMEOUT=10000


        export VLLM_HOST_IP="$LOCAL_IP"
        export HCCL_IF_IP="$LOCAL_IP"
        export GLOO_SOCKET_IFNAME="$NIC_NAME"
        export TP_SOCKET_IFNAME="$NIC_NAME"
        export HCCL_SOCKET_IFNAME="$NIC_NAME"

        export LD_LIBRARY_PATH="${PYTHON_LIB_DIR}:/usr/local/lib:${LD_LIBRARY_PATH}"

        export HCCL_BUFFSIZE=1024
        export HCCL_OP_EXPANSION_MODE="AIV"

        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export TASK_QUEUE_ENABLE=1

        export VLLM_USE_V2_MODEL_RUNNER=0

        export ASCEND_RT_VISIBLE_DEVICES=$1

        export PYTHONHASHSEED=0
        export ACL_OP_INIT_MODE=1
        export MMC_LOCAL_CONFIG_PATH="${MEMCACHE_ROOT}/config/mmc-local.conf"
        export LD_LIBRARY_PATH="${MEMCACHE_ROOT}/lib:${PYTHON_LIB_DIR}:${LD_LIBRARY_PATH}"

        export ASCEND_LOCAL_COMM_RES='{"version":"1.2"}'

        exec vllm serve "${MODEL_PATH}" \
            --host 0.0.0.0 \
            --port $2 \
            --data-parallel-size $3 \
            --data-parallel-rank $4 \
            --data-parallel-address $5 \
            --data-parallel-rpc-port $6 \
            --tensor-parallel-size $7 \
            --enable-expert-parallel \
            --enable-chunked-prefill \
            --enable-prefix-caching \
            --seed 1024 \
            --served-model-name glm5 \
            --max-model-len 200000 \
            --max-num-batched-tokens 256 \
            --trust-remote-code \
            --max-num-seqs 64 \
            --gpu-memory-utilization 0.92 \
            --async-scheduling \
            --quantization ascend \
            --safetensors-load-strategy 'prefetch' \
            --enable-auto-tool-choice \
            --tool-call-parser glm47 \
            --reasoning-parser glm47 \
            --kv-cache-dtype int8 \
            --attention_config.indexer_kv_dtype int8 \
            --additional-config '{
                "recompute_scheduler_enable": true,
                "enable_fused_mc2": 1
            }' \
            --speculative-config '{"num_speculative_tokens": 5,  "method":"deepseek_mtp","enforce_eager":true}' \
            --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
            --kv-transfer-config '{
                "kv_connector": "MultiConnector",
                "kv_role": "kv_consumer",
                "kv_port": "30200",
                "engine_id": "glm53-decode-dp'"$4"'",
                "kv_connector_extra_config": {
                    "connectors":[
                    {
                        "kv_connector": "MooncakeConnectorV2",
                        "kv_role": "kv_consumer",
                        "kv_port": "30200",
                        "kv_connector_extra_config": {
                            "use_ascend_direct": true,
                            "prefill": {"dp_size": 1, "pp_size": 2, "tp_size": 16, "pp_layer_partition": "42,36"},
                            "decode": { "dp_size": 16, "tp_size": 2}
                        }
                    },
                    {
                        "kv_connector": "AscendStoreConnector",
                        "kv_role": "kv_consumer",
                        "kv_connector_extra_config": {
                            "lookup_rpc_port":"0",
                            "backend": "memcache",
                            "use_layerwise": false
                        }
                    }
                    ]
                }
            }'
        ```

    4. Decode node 1

        ```shell
        #!/usr/bin/env bash

        LOCAL_IP="<DECODE_NODE_IP>"
        NIC_NAME="<NETWORK_INTERFACE>"
        MODEL_PATH="<YOUR_MODEL_PATH>"
        PYTHON_LIB_DIR=/path/to/python/lib
        MEMCACHE_ROOT=/path/to/site-packages/memcache_hybrid

        # Arguments $1-$7 are supplied by launch_online_dp.py.

        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export HCCL_EXEC_TIMEOUT=1800
        export HCCL_CONNECT_TIMEOUT=1800
        export ASCEND_TRANSFER_TIMEOUT=10000


        export VLLM_HOST_IP="$LOCAL_IP"
        export HCCL_IF_IP="$LOCAL_IP"
        export GLOO_SOCKET_IFNAME="$NIC_NAME"
        export TP_SOCKET_IFNAME="$NIC_NAME"
        export HCCL_SOCKET_IFNAME="$NIC_NAME"

        export LD_LIBRARY_PATH="${PYTHON_LIB_DIR}:/usr/local/lib:${LD_LIBRARY_PATH}"

        export HCCL_BUFFSIZE=1024
        export HCCL_OP_EXPANSION_MODE="AIV"

        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export TASK_QUEUE_ENABLE=1

        export VLLM_USE_V2_MODEL_RUNNER=0

        export ASCEND_RT_VISIBLE_DEVICES=$1

        export PYTHONHASHSEED=0
        export ACL_OP_INIT_MODE=1
        export MMC_LOCAL_CONFIG_PATH="${MEMCACHE_ROOT}/config/mmc-local.conf"
        export LD_LIBRARY_PATH="${MEMCACHE_ROOT}/lib:${PYTHON_LIB_DIR}:${LD_LIBRARY_PATH}"

        export ASCEND_LOCAL_COMM_RES='{"version":"1.2"}'

        exec vllm serve "${MODEL_PATH}" \
            --host 0.0.0.0 \
            --port $2 \
            --data-parallel-size $3 \
            --data-parallel-rank $4 \
            --data-parallel-address $5 \
            --data-parallel-rpc-port $6 \
            --tensor-parallel-size $7 \
            --enable-expert-parallel \
            --enable-chunked-prefill \
            --enable-prefix-caching \
            --seed 1024 \
            --served-model-name glm5 \
            --max-model-len 200000 \
            --max-num-batched-tokens 256 \
            --trust-remote-code \
            --max-num-seqs 64 \
            --gpu-memory-utilization 0.92 \
            --async-scheduling \
            --quantization ascend \
            --safetensors-load-strategy 'prefetch' \
            --enable-auto-tool-choice \
            --tool-call-parser glm47 \
            --reasoning-parser glm47 \
            --kv-cache-dtype int8 \
            --attention_config.indexer_kv_dtype int8 \
            --additional-config '{
                "recompute_scheduler_enable": true,
                "enable_fused_mc2": 1
            }' \
            --speculative-config '{"num_speculative_tokens": 5,  "method":"deepseek_mtp","enforce_eager":true}' \
            --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
            --kv-transfer-config '{
                "kv_connector": "MultiConnector",
                "kv_role": "kv_consumer",
                "kv_port": "30200",
                "engine_id": "glm53-decode-dp'"$4"'",
                "kv_connector_extra_config": {
                    "connectors":[
                    {
                        "kv_connector": "MooncakeConnectorV2",
                        "kv_role": "kv_consumer",
                        "kv_port": "30200",
                        "kv_connector_extra_config": {
                            "use_ascend_direct": true,
                            "prefill": {"dp_size": 1, "pp_size": 2, "tp_size": 16, "pp_layer_partition": "42,36"},
                            "decode": { "dp_size": 16, "tp_size": 2}
                        }
                    },
                    {
                        "kv_connector": "AscendStoreConnector",
                        "kv_role": "kv_consumer",
                        "kv_connector_extra_config": {
                            "lookup_rpc_port":"0",
                            "backend": "memcache",
                            "use_layerwise": false
                        }
                    }
                    ]
                }
            }'
        ```

The pooled configuration uses base KV ports 30000 on Prefill and
30200 on Decode. The outer `engine_id` is unique per engine and is
inherited by the child connectors. This differs from the supplied pooled
reference's repeated `"0"` IDs to avoid Mooncake peer identity conflicts.
`lookup_rpc_port="0"` is retained: Ascend Store uses an IPC endpoint that
also includes the DP rank, and P/D run on separate hosts. It is not a TCP
listener on port zero. `backend="memcache"` and `use_layerwise=false`
are set on both sides.

##### 5.1.3.3 Start the Engines

Start the MemCache metadata service and complete its configuration on all
four nodes before starting vLLM.

Once the preparation is done, you can start the server with the following command on each node:

1. Prefill node 0

    ```shell
    bash run_dp_template.sh
    ```

2. Prefill node 1

    ```shell
    bash run_dp_template.sh
    ```

3. Decode node 0

    ```shell
    D0_IP="<DECODE_NODE0_IP>"
    python launch_online_dp.py --dp-size 16 --tp-size 2 --dp-size-local 8 --dp-rank-start 0 --dp-address "$D0_IP" --dp-rpc-port 16600 --vllm-start-port 8000
    ```

4. Decode node 1

    ```shell
    D0_IP="<DECODE_NODE0_IP>"
    python launch_online_dp.py --dp-size 16 --tp-size 2 --dp-size-local 8 --dp-rank-start 8 --dp-address "$D0_IP" --dp-rpc-port 16600 --vllm-start-port 8000
    ```

##### 5.1.3.4 Start the Proxy and Verify Pooling

1. Start the MemCache metadata service and complete its configuration on
   all four nodes before starting vLLM.
2. Start P0, P1, D0, and D1 with the commands in
   Section 5.1.3.3. Both Prefill scripts already specify port 8000.
3. Wait for P0:8000 and all sixteen Decode APIs to become healthy. Start
   the proxy from Section 5.1.2.4 on a separate host, using the same
   Prefill endpoint and Decode ports 8000-8007. Its public port is 8000.
4. Send requests to the proxy using the served model name `glm5`:

```shell
curl "http://<proxy_ip>:8000/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "glm5",
        "messages": [{"role": "user", "content": "Explain prefill and decode in an LLM."}],
        "max_tokens": 128,
        "temperature": 0
    }'
```

For either mode, check P/D logs for successful KV transfer. For pooling,
also check MemCache initialization, then warm the pool with a prompt longer
than one cache block and repeat requests with the same prefix. Inspect
Ascend Store/MemCache saves, lookups, loads, and hit metrics. A successful
response or a local prefix-cache hit alone does not demonstrate shared pool
reuse; verify a pool load when the prefix is absent from the engine's local
cache. The A3 reference scripts above and the A2 scripts in Sections 5.1.2.5
and 5.1.3.5 use different topologies and are not interchangeable. These
references do not add accuracy or performance claims for the 198K workload.

##### 5.1.3.5 A2 Series: 8-Node Deployment with MemCache KV Cache Pool

This section adds MemCache KV-cache pooling to an 8-node A2
prefill-decode deployment of GLM-5.3 W8A8C8 on vLLM v0.30.0. It uses the
same connector pattern as the A3 pooling reference above: `MultiConnector`
combines `MooncakeConnectorV2`
for prefill-to-decode KV transfer with `AscendStoreConnector` for
non-layerwise MemCache pooling. The pool lets repeated or shared
prefixes be served from pooled DRAM instead of re-running prefill, which
is what moves high-repetition workloads from cold-path to warm-path
TTFT.

!!! warning

    This reference uses a different Prefill topology than Section 5.1.2.5:
    here each Prefill instance is TP8 × PP2 spanning two nodes, and two
    such instances run as two separate API servers. The pooled scripts
    were validated only in this topology — do not combine them with the
    Section 5.1.2.5 parallel layout.

| Role | Nodes | Parallelism | Placement | Devices per role |
| --- | --- | --- | --- | --- |
| Prefill | 4 nodes (2 instances) | TP8 × PP2 per instance | One PP stage per node; instance 0 on nodes 0+1, instance 1 on nodes 2+3 | 32 |
| Decode | 4 nodes | DP16 × TP2 | Four DP ranks per node (ranks 0-3, 4-7, 8-11, 12-15) | 32 |

Each Prefill instance exposes one API server (ports 8000 on both
instance leaders); the proxy registers both. Decode keeps the Section
5.3 arrangement: one 2-device engine per DP rank, coordinated by the
`launch_online_dp.py` launcher from Section 5.1.2.5.1 (reused unchanged),
DP RPC master on `<DECODE_NODE0_IP>`:12990, API ports 10010-10013.

###### 5.1.3.5.1 Prepare MemCache

Complete [MemCache backend setup](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md#scenario-2-memcache-backend),
including the `memfabric-hybrid` and `memcache-hybrid` installation and
the hardware prerequisites. Connector arguments do not start the
MemCache service; all Prefill and Decode nodes must reach the same
MetaService.

1. Install on every node:

    ```shell
    pip install memfabric-hybrid
    pip install memcache-hybrid
    ```

2. Prepare `mmc-meta.conf` (used by the MetaService on one node, here
   `<MEMCACHE_META_IP>`):

    ```ini
    ock.mmc.meta_service_url = tcp://<MEMCACHE_META_IP>:5000
    ock.mmc.meta_service.config_store_url = tcp://<MEMCACHE_META_IP>:6000
    ock.mmc.meta_service.metrics_url = http://<MEMCACHE_META_IP>:8009
    ock.mmc.log_level = info
    ```

3. Prepare `mmc-local.conf` in shared storage (or a copy on each node
   with identical contents), referenced later by `MMC_LOCAL_CONFIG_PATH`.
   `world_size` must cover every LocalService across both roles (here
   256 with headroom); `device_rdma` is the A2/RoCE protocol; the pool
   size is `dram.size` per die across all connected ranks. This reference
   reserves `40GB` for each LocalService.

    ```ini
    ock.mmc.meta_service_url = tcp://<MEMCACHE_META_IP>:5000
    ock.mmc.local_service.config_store_url = tcp://<MEMCACHE_META_IP>:6000
    ock.mmc.log_level = error
    ock.mmc.local_service.world_size = 256
    ock.mmc.local_service.protocol = device_rdma
    ock.mmc.local_service.dram.size = 40GB
    ```

4. Set `MEMCACHE_ROOT` to the installed `memcache_hybrid` directory and
   `PYTHON_LIB_DIR` to the library directory of the Python installation
   actually used by vLLM on each node (both are used by the scripts
   below).

###### 5.1.3.5.2 Prefill Engine Script (Pooled)

Save the following script as `start_p_pool.sh` on all four Prefill nodes. Each instance spans
two nodes (`--nnodes 2`, `mp` backend): the instance leader runs
`node_rank 0` and hosts the API server; its partner runs `node_rank 1`.
Run it once per node with the node's role and visible-device list.

```shell
#!/usr/bin/env bash
set -euo pipefail

NODE_RANK=${1:?"need node_rank (0=instance leader, 1=partner)"}
PORT=${2:-8000}
LOCAL_IP=${3:?need local_ip}
INSTANCE_LEADER_IP=${4:?need instance_leader_ip}
INSTANCE_ID=${5:?need instance_id}
VISIBLE_DEVICES=${6:-0,1,2,3,4,5,6,7}

case "$NODE_RANK" in
  0) HEADLESS_ARGS="";;                  # instance leader: API server
  1) HEADLESS_ARGS="--headless";;        # partner node: second PP stage
  *) echo "node_rank must be 0 or 1" >&2; exit 1;;
esac

MODEL_PATH=${MODEL_PATH:-/path/to/GLM-5.3-w8a8c8}
NIC_NAME=${NIC_NAME:?"export NIC_NAME to this node's business interface"}
MEMCACHE_ROOT=${MEMCACHE_ROOT:?export MEMCACHE_ROOT to the memcache_hybrid install dir}
PYTHON_LIB_DIR=${PYTHON_LIB_DIR:?export PYTHON_LIB_DIR to the vLLM Python lib dir}
# Instance leader address: nodes 0+1 form instance 0 (leader = <PREFILL_NODE0_IP>),
# nodes 2+3 form instance 1 (leader = <PREFILL_NODE2_IP>).

unset ftp_proxy FTP_PROXY https_proxy HTTPS_PROXY http_proxy HTTP_PROXY
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

export VLLM_SERVER_DEV_MODE=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=1800
export HCCL_CONNECT_TIMEOUT=1800
# TP8 stays inside one node per PP stage, so HCCL_IF_IP to the business IP is fine here.
export VLLM_HOST_IP=$LOCAL_IP
export HCCL_IF_IP=$LOCAL_IP
export GLOO_SOCKET_IFNAME=$NIC_NAME
export TP_SOCKET_IFNAME=$NIC_NAME
export HCCL_SOCKET_IFNAME=$NIC_NAME
export LD_LIBRARY_PATH="${PYTHON_LIB_DIR}:/usr/local/lib:${LD_LIBRARY_PATH}"
export HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_USE_FASTOKENS=1
export VLLM_PP_LAYER_PARTITION="42,36"
export PYTHONHASHSEED=0
export ACL_OP_INIT_MODE=1
export MMC_LOCAL_CONFIG_PATH="${MEMCACHE_ROOT}/config/mmc-local.conf"
export LD_LIBRARY_PATH="${MEMCACHE_ROOT}/lib:${PYTHON_LIB_DIR}:${LD_LIBRARY_PATH}"
export ASCEND_LOCAL_COMM_RES='{"version":"1.2"}'
export ASCEND_RT_VISIBLE_DEVICES="$VISIBLE_DEVICES"

exec vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --pipeline-parallel-size 2 \
  --distributed-executor-backend mp \
  --master-addr "$INSTANCE_LEADER_IP" \
  --master-port 7060 \
  --nnodes 2 \
  --node-rank "$NODE_RANK" \
  --enable-chunked-prefill \
  --no-async-scheduling \
  --enable-prefix-caching \
  --seed 1024 \
  --served-model-name glm5 \
  --max-model-len 200000 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.92 \
  --trust-remote-code \
  --quantization ascend \
  --safetensors-load-strategy prefetch \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm47 \
  --enforce-eager \
  --kv-cache-dtype int8 \
  --attention_config.indexer_kv_dtype int8 \
  --additional-config '{"enable_dsa_cp":true,"enable_flashcomm1":true,"scheduler_config":{"profiling_chunk_config":{"enabled":true}}}' \
  --speculative-config '{"num_speculative_tokens":3,"method":"deepseek_mtp","enforce_eager":true}' \
  --kv-transfer-config '{
      "kv_connector": "MultiConnector",
      "kv_role": "kv_producer",
      "kv_port": "30000",
      "engine_id": "glm53-a2-prefill-pool-'"$INSTANCE_ID"'",
      "kv_connector_extra_config": {
          "connectors": [
          {
              "kv_connector": "MooncakeConnectorV2",
              "kv_role": "kv_producer",
              "kv_port": "30000",
              "kv_connector_extra_config": {
                  "use_ascend_direct": true,
                  "prefill": {"dp_size": 1, "pp_size": 2, "tp_size": 8, "pp_layer_partition": "42,36"},
                  "decode": {"dp_size": 16, "tp_size": 2}
              }
          },
          {
              "kv_connector": "AscendStoreConnector",
              "kv_role": "kv_producer",
              "kv_connector_extra_config": {
                  "lookup_rpc_port": "0",
                  "backend": "memcache",
                  "use_layerwise": false
              }
          }
          ]
      }
   }' \
  $HEADLESS_ARGS
```

The four Prefill start commands are grouped in Section 5.1.3.5.4. The two
Prefill instances are fully independent vLLM deployments (each with its own
TCP store on its leader's port 7060); they share only the weight directory,
the MemCache pool, and the proxy in front of them.

###### 5.1.3.5.3 Decode Engine Script (Pooled)

Save the following complete script as `start_d_engine_pool.sh` on all
four Decode nodes. Set `LOCAL_IP` to the current node's IP in each copy.
`launch_online_dp.py` (Section 5.1.2.5.1) supplies `$1`-`$7`; the NIC, model,
and MemCache paths are configured in this script.

```shell
#!/usr/bin/env bash
set -euo pipefail

VISIBLE_DEVICES=$1
PORT=$2
DP_SIZE=$3
DP_RANK=$4
DP_ADDRESS=$5
DP_RPC_PORT=$6
TP_SIZE=$7
MODEL_PATH="<YOUR_MODEL_PATH>"

LOCAL_IP="<CURRENT_DECODE_NODE_IP>"
NIC_NAME="<NETWORK_INTERFACE>"
MEMCACHE_ROOT="<MEMCACHE_ROOT>"
PYTHON_LIB_DIR="<PYTHON_LIB_DIR>"

unset ftp_proxy FTP_PROXY https_proxy HTTPS_PROXY http_proxy HTTP_PROXY
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

export VLLM_SERVER_DEV_MODE=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=1800
export HCCL_CONNECT_TIMEOUT=1800
export ASCEND_TRANSFER_TIMEOUT=10000
# Intra-node TP2: HCCL_IF_IP to the local business IP is fine here.
export VLLM_HOST_IP=$LOCAL_IP
export HCCL_IF_IP=$LOCAL_IP
export GLOO_SOCKET_IFNAME=$NIC_NAME
export TP_SOCKET_IFNAME=$NIC_NAME
export HCCL_SOCKET_IFNAME=$NIC_NAME
export LD_LIBRARY_PATH="${PYTHON_LIB_DIR}:/usr/local/lib:${LD_LIBRARY_PATH}"
export HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export VLLM_USE_V2_MODEL_RUNNER=0
export ASCEND_RT_VISIBLE_DEVICES="$VISIBLE_DEVICES"
export PYTHONHASHSEED=0
export ACL_OP_INIT_MODE=1
export MMC_LOCAL_CONFIG_PATH="${MEMCACHE_ROOT}/config/mmc-local.conf"
export LD_LIBRARY_PATH="${MEMCACHE_ROOT}/lib:${PYTHON_LIB_DIR}:${LD_LIBRARY_PATH}"
export ASCEND_LOCAL_COMM_RES='{"version":"1.2"}'

exec vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --data-parallel-size "$DP_SIZE" \
  --data-parallel-rank "$DP_RANK" \
  --data-parallel-address "$DP_ADDRESS" \
  --data-parallel-rpc-port "$DP_RPC_PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  --enable-expert-parallel \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --seed 1024 \
  --served-model-name glm5 \
  --max-model-len 200000 \
  --max-num-batched-tokens 256 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.92 \
  --async-scheduling \
  --quantization ascend \
  --safetensors-load-strategy prefetch \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm47 \
  --kv-cache-dtype int8 \
  --attention_config.indexer_kv_dtype int8 \
  --additional-config '{"recompute_scheduler_enable":true}' \
  --speculative-config '{"num_speculative_tokens":3,"method":"deepseek_mtp","enforce_eager":true}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --kv-transfer-config '{
      "kv_connector": "MultiConnector",
      "kv_role": "kv_consumer",
      "kv_port": "30200",
      "engine_id": "glm53-a2-decode-pool-dp'"$DP_RANK"'",
      "kv_connector_extra_config": {
          "connectors": [
          {
              "kv_connector": "MooncakeConnectorV2",
              "kv_role": "kv_consumer",
              "kv_port": "30200",
              "kv_connector_extra_config": {
                  "use_ascend_direct": true,
                  "prefill": {"dp_size": 1, "pp_size": 2, "tp_size": 8, "pp_layer_partition": "42,36"},
                  "decode": {"dp_size": 16, "tp_size": 2}
              }
          },
          {
              "kv_connector": "AscendStoreConnector",
              "kv_role": "kv_consumer",
              "kv_connector_extra_config": {
                  "lookup_rpc_port": "0",
                  "backend": "memcache",
                  "use_layerwise": false
              }
          }
          ]
      }
   }'
```

The four Decode start commands are grouped in Section 5.1.3.5.4. They use
`--script ./start_d_engine_pool.sh` and the same launcher arguments as
Section 5.1.2.5.4.

###### 5.1.3.5.4 Start Services

Start services in this order: MemCache MetaService, Prefill, Decode, then
Proxy.

1. Start the MemCache MetaService on `<MEMCACHE_META_IP>` before any vLLM
   process:

    ```shell
    export MMC_META_CONFIG_PATH="${MEMCACHE_ROOT}/config/mmc-meta.conf"
    python3 -c "from memcache_hybrid import MetaService; MetaService.main()"
    ```

2. Start the two Prefill instances. Run each command on its target node and
   export the shared paths and the instance leader IP on that node:

    ```shell
    # Prefill node 0: instance 0 leader and API server
    export MODEL_PATH="<YOUR_MODEL_PATH>"
    export NIC_NAME="<NETWORK_INTERFACE>"
    export MEMCACHE_ROOT="<MEMCACHE_ROOT>"
    export PYTHON_LIB_DIR="<PYTHON_LIB_DIR>"
    bash start_p_pool.sh 0 8000 <PREFILL_NODE0_IP> <PREFILL_NODE0_IP> 0 0,1,2,3,4,5,6,7

    # Prefill node 1: instance 0 partner
    bash start_p_pool.sh 1 8000 <PREFILL_NODE1_IP> <PREFILL_NODE0_IP> 0 0,1,2,3,4,5,6,7

    # Prefill node 2: instance 1 leader and API server
    bash start_p_pool.sh 0 8000 <PREFILL_NODE2_IP> <PREFILL_NODE2_IP> 1 0,1,2,3,4,5,6,7

    # Prefill node 3: instance 1 partner
    bash start_p_pool.sh 1 8000 <PREFILL_NODE3_IP> <PREFILL_NODE2_IP> 1 0,1,2,3,4,5,6,7
    ```

3. After both Prefill instances are ready, start Decode on all four Decode
    nodes. Run each command on its target node; `LOCAL_IP` is configured in
    that node's copy of `start_d_engine_pool.sh`:

    ```shell
    # Decode node 0, DP ranks 0-3
    python launch_online_dp.py --script ./start_d_engine_pool.sh --dp-size 16 --tp-size 2 --dp-size-local 4 --dp-rank-start 0 --dp-address "<DECODE_NODE0_IP>" --dp-rpc-port 12990 --vllm-start-port 10010

    # Decode node 1, DP ranks 4-7
    python launch_online_dp.py --script ./start_d_engine_pool.sh --dp-size 16 --tp-size 2 --dp-size-local 4 --dp-rank-start 4 --dp-address "<DECODE_NODE0_IP>" --dp-rpc-port 12990 --vllm-start-port 10010

    # Decode node 2, DP ranks 8-11
    python launch_online_dp.py --script ./start_d_engine_pool.sh --dp-size 16 --tp-size 2 --dp-size-local 4 --dp-rank-start 8 --dp-address "<DECODE_NODE0_IP>" --dp-rpc-port 12990 --vllm-start-port 10010

    # Decode node 3, DP ranks 12-15
    python launch_online_dp.py --script ./start_d_engine_pool.sh --dp-size 16 --tp-size 2 --dp-size-local 4 --dp-rank-start 12 --dp-address "<DECODE_NODE0_IP>" --dp-rpc-port 12990 --vllm-start-port 10010
    ```

4. After the Prefill and Decode APIs are ready, start the proxy on a host
   that can reach every endpoint:

```shell
python examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py --host 0.0.0.0 --port 1999 --workers 8 --prefiller-hosts <PREFILL_NODE0_IP> <PREFILL_NODE2_IP> --prefiller-ports 8000 8000 --decoder-hosts <DECODE_NODE0_IP> <DECODE_NODE0_IP> <DECODE_NODE0_IP> <DECODE_NODE0_IP> <DECODE_NODE1_IP> <DECODE_NODE1_IP> <DECODE_NODE1_IP> <DECODE_NODE1_IP> <DECODE_NODE2_IP> <DECODE_NODE2_IP> <DECODE_NODE2_IP> <DECODE_NODE2_IP> <DECODE_NODE3_IP> <DECODE_NODE3_IP> <DECODE_NODE3_IP> <DECODE_NODE3_IP> --decoder-ports 10010 10011 10012 10013 10010 10011 10012 10013 10010 10011 10012 10013 10010 10011 10012 10013
```

###### 5.1.3.5.5 Verify Pooling

Send requests to the proxy using the served model name `glm5`. Keep
thinking enabled (Section 5 note). Then verify that reuse actually comes
from the shared pool, not only from an engine-local prefix cache:

1. Warm the pool with a prompt longer than one cache block, then repeat
   requests with the same prefix.
2. Inspect MemCache saves, lookups, loads, and hit metrics
   (`ock.mmc.log_level = info` in `mmc-meta.conf` logs them; a pool load
   when the prefix is absent from the engine's local cache is the
   evidence of shared-pool reuse — a local prefix-cache hit alone does
   not demonstrate pooling).
3. Check the vLLM connector statistics for
   `connector_prefix_cache_stats` hits on the Prefill side.

Key parameters:

- **Two independent Prefill instances**: each is a TP8 × PP2 deployment
  over two nodes (`--nnodes 2`, PP partition `42,36` pinned via
  `VLLM_PP_LAYER_PARTITION` and the connector's `pp_layer_partition` —
  these must stay consistent). The proxy round-robins between them.
- **MultiConnector on both roles**: `MooncakeConnectorV2` (P2P KV
  transfer, `use_ascend_direct`) plus `AscendStoreConnector`
  (`backend: "memcache"`, `use_layerwise: false`). Both sides declare
  the same topology: `prefill {dp_size 1, pp_size 2, tp_size 8,
  pp_layer_partition "42,36"}` and `decode {dp_size 16, tp_size 2}`.
  Both roles and both instances must agree on this declaration or the
  KV transfer handshake fails.
- **Base KV ports 30000 (Prefill) and 30200 (Decode)**; each engine
  derives its side-channel ports from `kv_port + dp_rank × tp × pp`, so
  reserve the derived ranges too. `lookup_rpc_port "0"` is not a TCP
  listener: Ascend Store uses an IPC endpoint that also includes the DP
  rank.
- **Distinct `engine_id` values** (`glm53-a2-prefill-pool-<instance>`,
   `glm53-a2-decode-pool-dp<rank>`) avoid Mooncake peer-identity conflicts;
  the same ID is never reused across engines.
- **Pooling env**: `PYTHONHASHSEED=0` must be identical on every node
  (pool-key hashing); `MMC_LOCAL_CONFIG_PATH` points at the shared
  `mmc-local.conf`; `MEMCACHE_ROOT/lib` must be on `LD_LIBRARY_PATH`
  for the MemCache runtime libraries.
- **Prefill budget**: eager mode, `--no-async-scheduling`, 16,384-token
  batch budget, 64 sequences, three MTP tokens, DSA CP + FlashComm1,
  `max-model-len 200000`, and the chunked-prefill profiling config
  (`scheduler_config.profiling_chunk_config`).
- **Decode budget**: 256-token batch budget, 64 sequences, three MTP
  tokens with an eager draft, `FULL_DECODE_ONLY` graph mode, and
  `recompute_scheduler_enable=true`, `max-model-len 200000`.
- **`VLLM_USE_V2_MODEL_RUNNER=0`** pins the V1 model runner on this
  release, matching the Section 5.1.2.5 value.
- Both roles use `glm47` tool parser and `glm47` reasoning parser and
  the served model name `glm5`, int8 KV cache + int8 indexer KV cache,
  chunked prefill, prefix caching, seed 1024, and memory utilization
  0.92.

### 5.2 1M Context Configuration

#### 5.2.1 1M Prefill-Decode Disaggregation (DCP)

This configuration uses GLM-5.3 W8A8C8 weights on four Atlas 800I A3
nodes, with 16 logical NPU devices per node. Use the A3 image from
[Docker Image Installation](#41-docker-image-installation) on all nodes.
The maximum context length is 1,048,576 tokens, including input and output.

##### 5.2.1.1 Topology and Prerequisites

| Role | Nodes | Global DP | TP | PP | DCP | Local DP ranks | Devices per role |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Prefill | P0, P1 | 2 | 16 | 1 | 16 | One per node | 32 |
| Decode | D0, D1 | 4 | 8 | 1 | 8 | Two per node | 32 |

P0 and P1 run DP ranks 0 and 1. D0 runs ranks 0-1 and D1 runs ranks 2-3.
Prefill context parallel size is 1 on both roles. This DCP layout is
separate from the PP2 layout in Sections 5.1.2 and 5.1.3.

1. Prepare identical weights on all four nodes and verify
   [multi-node communication](../../getting_started/installation.md#installation-multi-node-interconnect).
2. Set `LOCAL_IP`, `NIC_NAME`, and `MODEL_PATH` separately on each node.
   Replace the master address placeholders in the startup commands with
   P0's IP for Prefill and D0's IP for Decode.
3. Reserve DP RPC ports 16591 and 16600, API port 8000 on each Prefill
   node, API ports 8000-8001 on each Decode node, and Mooncake transfer
   ports derived from the configured base ports 30000 and 30200.
   Run the public proxy on a separate host at port 8000.

##### 5.2.1.2 Prepare the Scripts

Before you start, please

1. prepare the script `launch_online_dp.py` on each node:

    ```python
    import argparse
    import multiprocessing
    import subprocess
    import sys


    def run_command(args, local_rank):
        dp_rank = args.dp_rank_start + local_rank
        devices = ",".join(str(i) for i in range(
            local_rank * args.tp_size, (local_rank + 1) * args.tp_size))
        subprocess.run([
            "bash", args.script, devices, str(args.vllm_start_port + local_rank),
            str(args.dp_size), str(dp_rank), args.dp_address,
            str(args.dp_rpc_port), str(args.tp_size),
        ], check=True)


    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("--dp-size", type=int, required=True)
        parser.add_argument("--tp-size", type=int, required=True)
        parser.add_argument("--dp-size-local", type=int, required=True)
        parser.add_argument("--dp-rank-start", type=int, required=True)
        parser.add_argument("--dp-address", required=True)
        parser.add_argument("--dp-rpc-port", type=int, default=16600)
        parser.add_argument("--vllm-start-port", type=int, default=8000)
        parser.add_argument("--script", default="./run_dp_template.sh")
        args = parser.parse_args()
        if min(args.dp_size, args.tp_size, args.dp_size_local) <= 0:
            parser.error("Parallel sizes must be positive")
        if args.dp_rank_start < 0 or args.dp_rank_start + args.dp_size_local > args.dp_size:
            parser.error("Local ranks must be within the global DP group")
        if args.dp_size_local * args.tp_size > 16:
            parser.error("This A3 configuration exposes 16 devices per node")
        processes = []
        for local_rank in range(args.dp_size_local):
            process = multiprocessing.Process(target=run_command, args=(args, local_rank))
            processes.append(process)
            process.start()
        for process in processes:
            process.join()
        return 1 if any(process.exitcode != 0 for process in processes) else 0


    if __name__ == "__main__":
        sys.exit(main())
    ```

2. prepare the script `run_dp_template.sh` on each node.

    1. Prefill node 0

        ```shell
        #!/usr/bin/env bash

        LOCAL_IP="<PREFILL_NODE_IP>"
        NIC_NAME="<NETWORK_INTERFACE>"
        MODEL_PATH="<YOUR_MODEL_PATH>"

        export HCCL_OP_EXPANSION_MODE="AIV"
        export VLLM_ENGINE_READY_TIMEOUT_S=100000
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export HCCL_EXEC_TIMEOUT=204
        export HCCL_CONNECT_TIMEOUT=7200
        export ASCEND_CONNECT_TIMEOUT=10000
        export ASCEND_TRANSFER_TIMEOUT=10000
        export MF_GROUP_JOIN_MAX_TIMEOUT=1200

        export VLLM_HOST_IP="$LOCAL_IP"
        export HCCL_IF_IP="$LOCAL_IP"
        export GLOO_SOCKET_IFNAME="$NIC_NAME"
        export TP_SOCKET_IFNAME="$NIC_NAME"
        export HCCL_SOCKET_IFNAME="$NIC_NAME"

        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export HCCL_BUFFSIZE=400

        export ACL_OP_INIT_MODE=1
        export ASCEND_A3_ENABLE=1
        export ASCEND_RT_VISIBLE_DEVICES="$1"

        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

        export VLLM_USE_V2_MODEL_RUNNER=0

        exec vllm serve "$MODEL_PATH" \
            --host 0.0.0.0 \
            --port "$2" \
            --data-parallel-size "$3" \
            --data-parallel-rank "$4" \
            --data-parallel-address "$5" \
            --data-parallel-rpc-port "$6" \
            --tensor-parallel-size "$7" \
            --enable-expert-parallel \
            --prefill-context-parallel-size 1 \
            --decode-context-parallel-size 16 \
            --cp-kv-cache-interleave-size 128 \
            --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp","enforce_eager":true}' \
            --seed 1024 \
            --served-model-name glm-5 \
            --max-model-len 1048576 \
            --safetensors-load-strategy 'prefetch' \
            --additional-config '{
                "enable_dsa_cp": true,
                "enable_fused_mc2": 1,
                "enable_flashcomm1": true
            }' \
            --max-num-batched-tokens 8192 \
            --trust-remote-code \
            --enable-prefix-caching \
            --max-num-seqs 64 \
            --quantization ascend \
            --gpu-memory-utilization 0.85 \
            --enforce-eager \
            --enable-auto-tool-choice \
            --tool-call-parser glm47 \
            --reasoning-parser glm47 \
            --kv-cache-dtype int8 \
            --attention_config.indexer_kv_dtype int8 \
            --kv-transfer-config '{
                "kv_connector": "MooncakeConnectorV1",
                "kv_role": "kv_producer",
                "kv_port": "30000",
                "engine_id": "0",
                "kv_connector_extra_config": {
                    "use_ascend_direct": true,
                    "prefill": {
                        "dp_size": 2,
                        "tp_size": 16
                    },
                    "decode": {
                        "dp_size": 4,
                        "tp_size": 8
                    }
                }
            }'
        ```

    2. Prefill node 1

        ```shell
        #!/usr/bin/env bash

        LOCAL_IP="<PREFILL_NODE_IP>"
        NIC_NAME="<NETWORK_INTERFACE>"
        MODEL_PATH="<YOUR_MODEL_PATH>"

        export HCCL_OP_EXPANSION_MODE="AIV"
        export VLLM_ENGINE_READY_TIMEOUT_S=100000
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export HCCL_EXEC_TIMEOUT=204
        export HCCL_CONNECT_TIMEOUT=7200
        export ASCEND_CONNECT_TIMEOUT=10000
        export ASCEND_TRANSFER_TIMEOUT=10000
        export MF_GROUP_JOIN_MAX_TIMEOUT=1200

        export VLLM_HOST_IP="$LOCAL_IP"
        export HCCL_IF_IP="$LOCAL_IP"
        export GLOO_SOCKET_IFNAME="$NIC_NAME"
        export TP_SOCKET_IFNAME="$NIC_NAME"
        export HCCL_SOCKET_IFNAME="$NIC_NAME"

        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export HCCL_BUFFSIZE=400

        export ACL_OP_INIT_MODE=1
        export ASCEND_A3_ENABLE=1
        export ASCEND_RT_VISIBLE_DEVICES="$1"

        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

        export VLLM_USE_V2_MODEL_RUNNER=0

        exec vllm serve "$MODEL_PATH" \
            --host 0.0.0.0 \
            --port "$2" \
            --data-parallel-size "$3" \
            --data-parallel-rank "$4" \
            --data-parallel-address "$5" \
            --data-parallel-rpc-port "$6" \
            --tensor-parallel-size "$7" \
            --enable-expert-parallel \
            --prefill-context-parallel-size 1 \
            --decode-context-parallel-size 16 \
            --cp-kv-cache-interleave-size 128 \
            --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp","enforce_eager":true}' \
            --seed 1024 \
            --served-model-name glm-5 \
            --max-model-len 1048576 \
            --safetensors-load-strategy 'prefetch' \
            --additional-config '{
                "enable_dsa_cp": true,
                "enable_fused_mc2": 1,
                "enable_flashcomm1": true
            }' \
            --max-num-batched-tokens 8192 \
            --trust-remote-code \
            --enable-prefix-caching \
            --max-num-seqs 64 \
            --quantization ascend \
            --gpu-memory-utilization 0.85 \
            --enforce-eager \
            --enable-auto-tool-choice \
            --tool-call-parser glm47 \
            --reasoning-parser glm47 \
            --kv-cache-dtype int8 \
            --attention_config.indexer_kv_dtype int8 \
            --kv-transfer-config '{
                "kv_connector": "MooncakeConnectorV1",
                "kv_role": "kv_producer",
                "kv_port": "30000",
                "engine_id": "0",
                "kv_connector_extra_config": {
                    "use_ascend_direct": true,
                    "prefill": {
                        "dp_size": 2,
                        "tp_size": 16
                    },
                    "decode": {
                        "dp_size": 4,
                        "tp_size": 8
                    }
                }
            }'
        ```

    3. Decode node 0

        ```shell
        #!/usr/bin/env bash

        LOCAL_IP="<DECODE_NODE_IP>"
        NIC_NAME="<NETWORK_INTERFACE>"
        MODEL_PATH="<YOUR_MODEL_PATH>"

        export HCCL_OP_EXPANSION_MODE="AIV"
        export VLLM_ENGINE_READY_TIMEOUT_S=100000
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export HCCL_EXEC_TIMEOUT=204
        export HCCL_CONNECT_TIMEOUT=7200
        export ASCEND_CONNECT_TIMEOUT=10000
        export ASCEND_TRANSFER_TIMEOUT=10000
        export MF_GROUP_JOIN_MAX_TIMEOUT=1200

        export VLLM_HOST_IP="$LOCAL_IP"
        export HCCL_IF_IP="$LOCAL_IP"
        export GLOO_SOCKET_IFNAME="$NIC_NAME"
        export TP_SOCKET_IFNAME="$NIC_NAME"
        export HCCL_SOCKET_IFNAME="$NIC_NAME"

        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export HCCL_BUFFSIZE=400

        export ACL_OP_INIT_MODE=1
        export ASCEND_A3_ENABLE=1
        export TASK_QUEUE_ENABLE=1
        export ASCEND_RT_VISIBLE_DEVICES="$1"

        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
        export VLLM_USE_V2_MODEL_RUNNER=0

        exec vllm serve "$MODEL_PATH" \
            --host 0.0.0.0 \
            --port "$2" \
            --data-parallel-size "$3" \
            --data-parallel-rank "$4" \
            --data-parallel-address "$5" \
            --data-parallel-rpc-port "$6" \
            --tensor-parallel-size "$7" \
            --enable-expert-parallel \
            --prefill-context-parallel-size 1 \
            --decode-context-parallel-size 8 \
            --cp-kv-cache-interleave-size 128 \
            --seed 1024 \
            --served-model-name glm-5 \
            --max-model-len 1048576 \
            --safetensors-load-strategy 'prefetch' \
            --max-num-batched-tokens 192 \
            --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
            --speculative-config '{"num_speculative_tokens": 5,  "method":"deepseek_mtp","enforce_eager":true}' \
            --additional-config '{
                "recompute_scheduler_enable": true,
                "enable_mlapo": true,
                "enable_fused_mc2": 1
            }' \
            --trust-remote-code \
            --max-num-seqs 32 \
            --gpu-memory-utilization 0.90 \
            --async-scheduling \
            --quantization ascend \
            --enable-auto-tool-choice \
            --tool-call-parser glm47 \
            --reasoning-parser glm47 \
            --kv-cache-dtype int8 \
            --attention_config.indexer_kv_dtype int8 \
            --kv-transfer-config '{
                "kv_connector": "MooncakeConnectorV1",
                "kv_role": "kv_consumer",
                "kv_port": "30200",
                "engine_id": "2",
                "kv_connector_extra_config": {
                    "use_ascend_direct": true,
                    "prefill": {
                        "dp_size": 2,
                        "tp_size": 16
                    },
                    "decode": {
                        "dp_size": 4,
                        "tp_size": 8
                    }
                }
            }'
        ```

    4. Decode node 1

        ```shell
        #!/usr/bin/env bash

        LOCAL_IP="<DECODE_NODE_IP>"
        NIC_NAME="<NETWORK_INTERFACE>"
        MODEL_PATH="<YOUR_MODEL_PATH>"

        export HCCL_OP_EXPANSION_MODE="AIV"
        export VLLM_ENGINE_READY_TIMEOUT_S=100000
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export HCCL_EXEC_TIMEOUT=204
        export HCCL_CONNECT_TIMEOUT=7200
        export ASCEND_CONNECT_TIMEOUT=10000
        export ASCEND_TRANSFER_TIMEOUT=10000
        export MF_GROUP_JOIN_MAX_TIMEOUT=1200

        export VLLM_HOST_IP="$LOCAL_IP"
        export HCCL_IF_IP="$LOCAL_IP"
        export GLOO_SOCKET_IFNAME="$NIC_NAME"
        export TP_SOCKET_IFNAME="$NIC_NAME"
        export HCCL_SOCKET_IFNAME="$NIC_NAME"

        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export HCCL_BUFFSIZE=400

        export ACL_OP_INIT_MODE=1
        export ASCEND_A3_ENABLE=1
        export TASK_QUEUE_ENABLE=1
        export ASCEND_RT_VISIBLE_DEVICES="$1"

        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
        export VLLM_USE_V2_MODEL_RUNNER=0

        exec vllm serve "$MODEL_PATH" \
            --host 0.0.0.0 \
            --port "$2" \
            --data-parallel-size "$3" \
            --data-parallel-rank "$4" \
            --data-parallel-address "$5" \
            --data-parallel-rpc-port "$6" \
            --tensor-parallel-size "$7" \
            --enable-expert-parallel \
            --prefill-context-parallel-size 1 \
            --decode-context-parallel-size 8 \
            --cp-kv-cache-interleave-size 128 \
            --seed 1024 \
            --served-model-name glm-5 \
            --max-model-len 1048576 \
            --safetensors-load-strategy 'prefetch' \
            --max-num-batched-tokens 192 \
            --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
            --speculative-config '{"num_speculative_tokens": 5,  "method":"deepseek_mtp","enforce_eager":true}' \
            --additional-config '{
                "recompute_scheduler_enable": true,
                "enable_mlapo": true,
                "enable_fused_mc2": 1
            }' \
            --trust-remote-code \
            --max-num-seqs 32 \
            --gpu-memory-utilization 0.90 \
            --async-scheduling \
            --quantization ascend \
            --enable-auto-tool-choice \
            --tool-call-parser glm47 \
            --reasoning-parser glm47 \
            --kv-cache-dtype int8 \
            --attention_config.indexer_kv_dtype int8 \
            --kv-transfer-config '{
                "kv_connector": "MooncakeConnectorV1",
                "kv_role": "kv_consumer",
                "kv_port": "30200",
                "engine_id": "2",
                "kv_connector_extra_config": {
                    "use_ascend_direct": true,
                    "prefill": {
                        "dp_size": 2,
                        "tp_size": 16
                    },
                    "decode": {
                        "dp_size": 4,
                        "tp_size": 8
                    }
                }
            }'
        ```

##### 5.2.1.3 Start the Engines

Run these commands from the directory containing both scripts on each node.
The two Prefill commands use the same P0 master address, and the two Decode
commands use the same D0 master address.

1. Prefill node 0

    ```shell
    P0_IP="<PREFILL_NODE0_IP>"
    python launch_online_dp.py --dp-size 2 --tp-size 16 --dp-size-local 1 --dp-rank-start 0 --dp-address "$P0_IP" --dp-rpc-port 16591 --vllm-start-port 8000
    ```

2. Prefill node 1

    ```shell
    P0_IP="<PREFILL_NODE0_IP>"
    python launch_online_dp.py --dp-size 2 --tp-size 16 --dp-size-local 1 --dp-rank-start 1 --dp-address "$P0_IP" --dp-rpc-port 16591 --vllm-start-port 8000
    ```

3. Decode node 0

    ```shell
    D0_IP="<DECODE_NODE0_IP>"
    python launch_online_dp.py --dp-size 4 --tp-size 8 --dp-size-local 2 --dp-rank-start 0 --dp-address "$D0_IP" --dp-rpc-port 16600 --vllm-start-port 8000
    ```

4. Decode node 1

    ```shell
    D0_IP="<DECODE_NODE0_IP>"
    python launch_online_dp.py --dp-size 4 --tp-size 8 --dp-size-local 2 --dp-rank-start 2 --dp-address "$D0_IP" --dp-rpc-port 16600 --vllm-start-port 8000
    ```

##### 5.2.1.4 Start the Proxy

After all six APIs become healthy, start the proxy on its separate host:

```shell
P0_IP="<P0_IP>"
P1_IP="<P1_IP>"
D0_IP="<D0_IP>"
D1_IP="<D1_IP>"

python examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
    --host 0.0.0.0 --port 8000 \
    --prefiller-hosts "$P0_IP" "$P1_IP" --prefiller-ports 8000 8000 \
    --decoder-hosts "$D0_IP" "$D0_IP" "$D1_IP" "$D1_IP" \
    --decoder-ports 8000 8001 8000 8001
```

Send requests to `http://<PROXY_IP>:8000` using the served model name `glm-5`.

Key Parameter Descriptions:

- The launcher passes `$1` = visible devices, `$2` = API port, `$3` = global
  DP size, `$4` = DP rank, `$5` = DP master IP, `$6` = DP RPC port, and
  `$7` = TP size to `run_dp_template.sh`.
- `--decode-context-parallel-size`: 16 on Prefill and 8 on Decode;
  `--prefill-context-parallel-size` remains 1 and
  `--cp-kv-cache-interleave-size` is 128 on both roles.
- `--max-num-batched-tokens`: 8192 on Prefill and 192 on Decode.
  `--max-num-seqs`: 64 on Prefill and 32 on Decode. These are scheduler
  limits; start the 1M workload with one concurrent request and increase
  concurrency according to available KV cache capacity.
- MTP uses `deepseek_mtp` with an eager draft: 1 speculative token on
  Prefill and 5 on Decode. Decode uses `FULL_DECODE_ONLY` graph mode.
- `MooncakeConnectorV1` uses `kv_producer` on Prefill and `kv_consumer`
  on Decode, with the same `prefill: DP2/TP16` and `decode: DP4/TP8`
  topology on both roles. This example does not enable MemCache pooling.
- Both roles use int8 KV and indexer KV caches. Prefill enables DSA CP,
  fused MC2 and FlashComm1; Decode enables fused MC2, MLAPO and the
  recomputation scheduler. See [Performance Tuning](#9-performance-tuning).

## 6 Functional Verification

Once your server is started, you can query the model with input prompts:

```shell
curl http://<node0_ip>:<port>/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "glm-5",
        "messages":[
            {
                "role": "user",
                "content": "Who are you?"
            }
        ],
        "temperature": 0
    }'
```

Expected result should be have this:

```text
"message":{"role":"assistant","content":"I'm GLM, a large language model developed by Z.ai. I'm designed to understand and generate human-like text based on the conversations we have together. My role involves processing diverse text data to help answer questions and provide assistance across many topics.\n\nI don't store your personal data, and I'm continually learning to improve my capabilities. Is there something specific I can help you with today?","refusal":null,"annotations":null,"audio":null,"function_call":oning":"Let me analyze this question about my identity. First, I should acknowledge that this is a fundamental question about who and what I am. The key aspects are my identity as GLM, a large language model by Z.ai, and my core capabilities. I should explain my primary function of text processing and be transparent about my nature as an AI system. It's also important to clarify my role in helping users and my ability to engage with various topics. Mention my text processing abilities and learning from diverse datasets, but avoid making claims about consciousness or emotions. The response should be structured logically, starting with my basic identity and moving on to my capabilities and purpose. I'll organize this information in a clear, straightforward manner that addresses the user's query directly."}
```

## 7 Accuracy Evaluation

Here are two accuracy evaluation methods.

### 7.1 Using AISBench

1. Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for details.

2. The following table shows the results of `GLM-5.3-w8a8c8` with Prefill-Decode disaggregation.

| dataset | model | hardware | deployment | metric | mode | vllm-api-general-chat |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| GPQA Diamond | GLM-5.3-w8a8c8 | A3 | PD disaggregation | accuracy | gen | 91.92 |
| GPQA Diamond | GLM-5.3-w8a8c8 | A2 | PD disaggregation | accuracy | gen | 89.39 |

### 7.2 Using Language Model Evaluation Harness

Not tested yet.

## 8 Performance Evaluation

### 8.1 Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### 8.2 Using vLLM Benchmark

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

## 9 Performance Tuning

The following A3 W8A8C8 settings summarize the deployment scripts in
Sections 5.1.2, 5.1.3 and 5.2.1. Global DP includes all nodes of one role;
local DP is the number of ranks on one node. Each role uses 32 logical
NPU devices.

### 9.1 Detailed Node Configuration

| Scenario | Role | TP | Global DP | Local DP | PP | DCP | Max Num Seqs | Max Num Batched Tokens | Max Model Len | MTP Spec Num |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 198K PD, with or without pooling | Prefill | 16 | 1 | 1 | 2 | 1 | 64 | 16384 | 200000 | 1 |
| 198K PD, with or without pooling | Decode | 2 | 16 | 8 | 1 | 1 | 64 | 256 | 200000 | 5 |
| 1M PD (A3) | Prefill | 16 | 2 | 1 | 1 | 16 | 64 | 8192 | 1048576 | 1 |
| 1M PD (A3) | Decode | 8 | 4 | 2 | 1 | 8 | 32 | 192 | 1048576 | 5 |

### 9.2 Performance Workloads

The 0.30 RC performance reference covers the following workloads.
Lengths use `input/output` notation. The 1M A3 case uses one concurrent
request; reserve output tokens within the 1,048,576-token context limit.

| Scenario | Input / Output | Prefix cache hit rate | Deployment reference |
| --- | --- | --- | --- |
| Prefill-only | 16K / 1, 128K / 1 | 0% | Section 5.1.2 |
| Decode-only | 16K / 1K, 128K / 1K | 100% | Section 5.1.2 |
| Complete PD | 16K / 1K | 0% | Section 5.1.2 |
| Complete PD, 1M context (A3) | Approximately 1M / 1K | 0% | Section 5.2.1 |
| Pooling | 128K / 1K | 90%, 32 distinct prefixes | Section 5.1.3 |
| Pooling | 40K-80K / 2.5K | Repeated prefixes | Section 5.1.3 |

### 9.3 Optimizations Requiring Explicit Enablement

Combine enabled options in one `--additional-config` JSON object, as in
the deployment scripts.

| Optimization | Scenario | Enablement | Purpose and constraints |
| --- | --- | --- | --- |
| FlashComm1 | A3 Prefill | `"enable_flashcomm1": true` | Reduces TP communication overhead; the Decode reference keeps it disabled. |
| Fused MC2 | A3 PD | `"enable_fused_mc2": 1` | Fuses MoE dispatch, computation and combine. Do not combine with shared-expert multistream overlap. |
| DSA CP | A3 Prefill | `"enable_dsa_cp": true` | Distributes long-context attention work; use the layout specified in each scenario. |
| DCP | 1M PD (A3) | `--decode-context-parallel-size 16` on Prefill, `8` on Decode | Shards KV cache across the TP group. Keep PCP at 1 and KV interleave at 128 on both roles. |
| Int8 attention KV cache | A3 W8A8C8 | `--kv-cache-dtype int8` | Reduces attention KV cache memory. |
| Int8 indexer KV cache | A3 W8A8C8 | `--attention_config.indexer_kv_dtype int8` | Quantizes the indexer cache separately from the attention cache. |
| MLAPO | A3 1M Decode | `"enable_mlapo": true` | Fuses MLA preprocessing; account for its memory requirements. |
| Recompute scheduler | A3 Decode | `"recompute_scheduler_enable": true` | Enables Prefill-side recomputation when Decode KV cache capacity is insufficient. |
| Decode graph | A3 Decode | `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` | Captures Decode execution; the reference Prefill uses eager mode. |
| KV pooling | Repeated-prefix PD | Ascend Store/MemCache configuration in Section 5.1.3 | Reuses KV across requests and engines; confirm pool hits in backend metrics. |

## 10 FAQ

- **Q: How to enable function calling for GLM-5.3?**

  A: Please add following configurations in vLLM startup command

  ```shell
  --tool-call-parser glm47 \
  --reasoning-parser glm47 \
  --enable-auto-tool-choice \
  ```

- **Q: Does GLM-5.3 support `enable_thinking: false`?**

  A: No, GLM-5.3 does not support `enable_thinking`.
