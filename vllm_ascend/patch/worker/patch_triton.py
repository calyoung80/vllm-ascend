import torch
import vllm.model_executor.layers.fla.ops
import vllm.model_executor.layers.mamba.ops.causal_conv1d
import vllm.model_executor.layers.mamba.ops.ssd_chunk_scan
import vllm.model_executor.layers.mamba.ops.ssd_combined
import vllm.v1.worker.gpu.sample.gumbel
from vllm.triton_utils import HAS_TRITON, triton
from vllm.utils.math_utils import next_power_of_2

from vllm_ascend.ops.triton.fla.chunk import chunk_gated_delta_rule
from vllm_ascend.ops.triton.fla.layernorm_guard import LayerNormFn
from vllm_ascend.ops.triton.fla.sigmoid_gating import fused_recurrent_gated_delta_rule_fwd_kernel
from vllm_ascend.ops.triton.mamba.causal_conv1d import causal_conv1d_fn, causal_conv1d_update_npu

triton.next_power_of_2 = next_power_of_2

_ssd_chunk_scan = vllm.model_executor.layers.mamba.ops.ssd_chunk_scan
_ssd_combined = vllm.model_executor.layers.mamba.ops.ssd_combined
_ORIGINAL_CHUNK_SCAN_FWD = _ssd_chunk_scan._chunk_scan_fwd


def _chunk_scan_optional_kernel_args(reference, optional_tensor, shape):
    if optional_tensor is not None:
        return optional_tensor
    return reference.new_empty(shape)


def _chunk_scan_fwd_npu(
    cb,
    x,
    dt,
    dA_cumsum,
    C,
    states,
    cu_chunk_seqlens,
    out,
    seq_idx,
    D=None,
    z=None,
    initial_states=None,
):
    if x.device.type != "npu" or initial_states is not None:
        return _ORIGINAL_CHUNK_SCAN_FWD(
            cb,
            x,
            dt,
            dA_cumsum,
            C,
            states,
            cu_chunk_seqlens,
            out,
            seq_idx,
            D=D,
            z=z,
            initial_states=initial_states,
        )

    assert seq_idx is not None, "this implementation requires seq_idx"

    seqlen, nheads, headdim = x.shape
    _, nchunks, chunk_size = dt.shape
    _, ngroups, dstate = C.shape
    assert nheads % ngroups == 0
    assert C.shape == (seqlen, ngroups, dstate)
    assert cb.shape == (nchunks, ngroups, chunk_size, chunk_size)
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
    if z is not None:
        assert z.shape == x.shape
    assert dt.shape == (nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == (nheads, nchunks, chunk_size)
    assert states.shape == (nchunks, nheads, headdim, dstate)
    assert seq_idx.shape == (nchunks,)

    grid = lambda META: (
        triton.cdiv(chunk_size, META["BLOCK_SIZE_M"])
        * triton.cdiv(headdim, META["BLOCK_SIZE_N"]),
        nchunks,
        nheads,
    )

    z_strides = (z.stride(0), z.stride(1), z.stride(2)) if z is not None else (0, 0, 0)
    initial_states_strides = (0, 0, 0, 0)

    # Triton-Ascend may type-check pointers in constexpr-false branches. Keep
    # semantic flags false, but pass non-null tensors for optional pointers.
    z_ptr = _chunk_scan_optional_kernel_args(x, z, (1, 1, 1))
    D_ptr = _chunk_scan_optional_kernel_args(x, D, (1,))
    initstates_ptr = _chunk_scan_optional_kernel_args(states, None, (1, 1, 1, 1))

    _ssd_chunk_scan._chunk_scan_fwd_kernel[grid](
        cb_ptr=cb,
        x_ptr=x,
        z_ptr=z_ptr,
        out_ptr=out,
        dt_ptr=dt,
        dA_cumsum_ptr=dA_cumsum,
        seq_idx_ptr=seq_idx,
        C_ptr=C,
        states_ptr=states,
        D_ptr=D_ptr,
        initstates_ptr=initstates_ptr,
        cu_chunk_seqlens_ptr=cu_chunk_seqlens,
        chunk_size=chunk_size,
        hdim=headdim,
        dstate=dstate,
        seqlen=seqlen,
        nheads_ngroups_ratio=nheads // ngroups,
        stride_cb_chunk=cb.stride(0),
        stride_cb_head=cb.stride(1),
        stride_cb_csize_m=cb.stride(2),
        stride_cb_csize_k=cb.stride(3),
        stride_x_seqlen=x.stride(0),
        stride_x_head=x.stride(1),
        stride_x_hdim=x.stride(2),
        stride_z_seqlen=z_strides[0],
        stride_z_head=z_strides[1],
        stride_z_hdim=z_strides[2],
        stride_out_seqlen=out.stride(0),
        stride_out_head=out.stride(1),
        stride_out_hdim=out.stride(2),
        stride_dt_chunk=dt.stride(1),
        stride_dt_head=dt.stride(0),
        stride_dt_csize=dt.stride(2),
        stride_dA_cs_chunk=dA_cumsum.stride(1),
        stride_dA_cs_head=dA_cumsum.stride(0),
        stride_dA_cs_csize=dA_cumsum.stride(2),
        stride_seq_idx_chunk=seq_idx.stride(0),
        stride_C_seqlen=C.stride(0),
        stride_C_head=C.stride(1),
        stride_C_dstate=C.stride(2),
        stride_states_chunk=states.stride(0),
        stride_states_head=states.stride(1),
        stride_states_hdim=states.stride(2),
        stride_states_dstate=states.stride(3),
        stride_init_states_batch=initial_states_strides[0],
        stride_init_states_head=initial_states_strides[1],
        stride_init_states_hdim=initial_states_strides[2],
        stride_init_states_dstate=initial_states_strides[3],
        stride_D_head=D.stride(0) if D is not None else 0,
        IS_CAUSAL=True,
        HAS_D=D is not None,
        D_HAS_HDIM=D.dim() == 2 if D is not None else True,
        HAS_Z=z is not None,
        BLOCK_SIZE_DSTATE=max(triton.next_power_of_2(dstate), 16),
        IS_TRITON_22=_ssd_chunk_scan.TRITON_22,
        HAS_INITSTATES=False,
    )
    return None


vllm.model_executor.layers.mamba.ops.causal_conv1d.causal_conv1d_update = causal_conv1d_update_npu
vllm.model_executor.layers.mamba.ops.causal_conv1d.causal_conv1d_fn = causal_conv1d_fn
_ssd_chunk_scan._chunk_scan_fwd = _chunk_scan_fwd_npu
_ssd_combined._chunk_scan_fwd = _chunk_scan_fwd_npu
vllm.model_executor.layers.fla.ops.fused_recurrent.fused_recurrent_gated_delta_rule_fwd_kernel = (
    fused_recurrent_gated_delta_rule_fwd_kernel
)
vllm.model_executor.layers.fla.ops.layernorm_guard.LayerNormFn = LayerNormFn
vllm.model_executor.layers.fla.ops.chunk_gated_delta_rule = chunk_gated_delta_rule

# On NPU platforms without an active Triton backend (e.g. 310P), replace the
# Triton-based fused_post_conv_prep with a pure-PyTorch fallback so that
# qwen_gdn_linear_attn's from-import picks up the replacement before model
# load.
if not HAS_TRITON:
    import torch
    import torch.nn.functional as _F

    def _fused_post_conv_prep_pytorch(
        conv_output,
        a,
        b,
        A_log,
        dt_bias,
        num_k_heads,
        head_k_dim,
        head_v_dim,
        apply_l2norm=True,
        output_g_exp=False,
    ):
        L = conv_output.shape[0]
        H, K, V = num_k_heads, head_k_dim, head_v_dim
        HV = A_log.shape[0]

        q = conv_output[:, : H * K].reshape(L, H, K)
        k = conv_output[:, H * K : 2 * H * K].reshape(L, H, K)
        v = conv_output[:, 2 * H * K :].reshape(L, HV, V)

        if apply_l2norm:
            # x / sqrt(sum(x^2) + eps) — matches Triton kernel, in fp32
            def _l2norm(t):
                t_f = t.float()
                return (t_f / torch.sqrt((t_f * t_f).sum(-1, keepdim=True) + 1e-6)).to(t.dtype)

            q, k = _l2norm(q), _l2norm(k)

        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

        x = (a + dt_bias.unsqueeze(0)).float()
        g = -torch.exp(A_log.float().unsqueeze(0)) * _F.softplus(x)
        if output_g_exp:
            g = torch.exp(g)

        return q, k, v, g, torch.sigmoid(b.float())

    vllm.model_executor.layers.fla.ops.fused_post_conv_prep = _fused_post_conv_prep_pytorch

    def _fused_recurrent_packed_decode_pytorch(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        scale,
        initial_state,
        out,
        ssm_state_indices,
        use_qk_l2norm_in_kernel=False,
    ):
        B = mixed_qkv.shape[0]
        HV, V, K = initial_state.shape[-3:]
        H = (mixed_qkv.shape[1] - HV * V) // (2 * K)
        ratio = HV // H

        q = mixed_qkv[:, : H * K].reshape(B, H, K)
        k = mixed_qkv[:, H * K : 2 * H * K].reshape(B, H, K)
        v = mixed_qkv[:, 2 * H * K :].reshape(B, HV, V)

        SOFTPLUS_THRESHOLD = 20.0
        x = (a + dt_bias.unsqueeze(0)).float()
        softplus_x = torch.where(x <= SOFTPLUS_THRESHOLD, torch.log1p(torch.exp(x)), x)
        g = -torch.exp(A_log.float().unsqueeze(0)) * softplus_x  # [B, HV]
        beta = torch.sigmoid(b.float())  # [B, HV]

        for n in range(B):
            state_idx = int(ssm_state_indices[n].item())
            if state_idx <= 0:
                out[n, 0] = 0
                continue

            h = initial_state[state_idx].float()  # [HV, V, K]
            q_n = q[n].float().repeat_interleave(ratio, dim=0)  # [HV, K]
            k_n = k[n].float().repeat_interleave(ratio, dim=0)  # [HV, K]
            v_n = v[n].float()  # [HV, V]

            if use_qk_l2norm_in_kernel:

                def _l2norm(t):
                    t_f = t.float()
                    return t_f / torch.sqrt((t_f * t_f).sum(-1, keepdim=True) + 1e-6)

                q_n, k_n = _l2norm(q_n), _l2norm(k_n)
            q_n = q_n * scale

            h = h * torch.exp(g[n]).view(HV, 1, 1)
            v_n = v_n - torch.einsum("hvk,hk->hv", h, k_n)
            v_n = v_n * beta[n].view(HV, 1)
            h = h + torch.einsum("hv,hk->hvk", v_n, k_n)
            out[n, 0] = torch.einsum("hvk,hk->hv", h, q_n).to(out.dtype)
            initial_state[state_idx] = h.to(initial_state.dtype)

        return out, initial_state

    vllm.model_executor.layers.fla.ops.fused_recurrent.fused_recurrent_gated_delta_rule_packed_decode = (
        _fused_recurrent_packed_decode_pytorch
    )
