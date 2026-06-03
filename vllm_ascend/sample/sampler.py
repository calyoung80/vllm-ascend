import torch
import vllm.envs as envs
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import logger
from vllm.triton_utils import HAS_TRITON, triton
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler
from vllm.v1.sample.sampler import Sampler
from vllm.v1.outputs import LogprobsTensors, SamplerOutput

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.sample.penalties import apply_all_penalties
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type, global_stream, npu_stream_switch
from vllm_ascend.worker.v2.sample.gumbel import gumbel_sample

DEFAULT_LOGPROBS_MODE = "raw_logprobs"

_SAMPLING_EPS = 1e-5


def _log_sample_trace(message: str, **kwargs) -> None:
    if kwargs:
        details = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        logger.info("[sample-trace][sampler] %s | %s", message, details)
    else:
        logger.info("[sample-trace][sampler] %s", message)


def random_sample(
    probs: torch.Tensor,
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    """Randomly sample from the probabilities.

    We use this function instead of torch.multinomial because torch.multinomial
    causes CPU-NPU synchronization.
    """
    # NOTE(woosuk): To batch-process the requests without their own seeds,
    # which is the common case, we first assume that every request does
    # not have its own seed. Then, we overwrite the values for the requests
    # that have their own seeds.
    _log_sample_trace(
        "random_sample_enter",
        rows=probs.shape[0],
        cols=probs.shape[1],
        generator_count=len(generators),
    )
    with npu_stream_switch(global_stream()):
        q = torch.empty_like(probs)
        if len(generators) != probs.shape[0]:
            q.exponential_()
        if generators:
            # TODO(woosuk): This can be slow because we handle each request
            # one by one. Optimize this.
            for i, generator in generators.items():
                q[i].exponential_(generator=generator)
    torch.npu.current_stream().wait_stream(global_stream())
    _log_sample_trace("random_sample_ready")
    return probs.div_(q).argmax(dim=-1).view(-1)


def _ensure_runtime_state_tensor(
    tensor: torch.Tensor | None,
    needed: int,
    default_value,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if tensor is not None and tensor.shape[0] >= needed:
        return tensor
    out = torch.full((needed,), default_value, dtype=dtype, device=device)
    if tensor is not None and tensor.numel() > 0:
        out[: tensor.shape[0]].copy_(tensor)
    return out


def sample_with_runtime_state(
    logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    positions: torch.Tensor,
    temperature: torch.Tensor | None,
    top_k: torch.Tensor | None,
    top_p: torch.Tensor | None,
    seeds: torch.Tensor | None,
    all_greedy: bool = False,
    all_random: bool = False,
) -> torch.Tensor:
    needed = max(int(idx_mapping.shape[0]), 1)
    if all_greedy or temperature is None:
        return logits.argmax(dim=-1)

    temperature = _ensure_runtime_state_tensor(
        temperature,
        needed,
        0.0,
        torch.float32,
        logits.device,
    )
    seeds = _ensure_runtime_state_tensor(
        seeds,
        needed,
        0,
        torch.int64,
        logits.device,
    )

    idx_mapping_long = idx_mapping.to(torch.long)
    row_temperature = temperature[idx_mapping_long]
    greedy_sampled = None if all_random else logits.argmax(dim=-1)
    safe_temperature = (
        row_temperature
        if all_random
        else torch.where(
            row_temperature < _SAMPLING_EPS,
            torch.ones_like(row_temperature),
            row_temperature,
        )
    )
    logits = logits.div_(safe_temperature.unsqueeze(dim=1))

    if top_k is not None or top_p is not None:
        top_k_base = _ensure_runtime_state_tensor(
            top_k,
            needed,
            logits.shape[1],
            torch.int32,
            logits.device,
        )
        top_p_base = _ensure_runtime_state_tensor(
            top_p,
            needed,
            1.0,
            torch.float32,
            logits.device,
        )
        logits = apply_top_k_top_p(
            logits,
            top_k_base[idx_mapping_long],
            top_p_base[idx_mapping_long],
        )

    if logits.device.type != "npu" or not hasattr(triton, "cdiv"):
        return logits.argmax(dim=-1) if greedy_sampled is None else greedy_sampled

    random_sampled = gumbel_sample(
        logits,
        idx_mapping.to(torch.int32),
        temperature,
        seeds,
        positions.to(torch.int32),
        apply_temperature=False,
    )
    if all_random:
        return random_sampled
    assert greedy_sampled is not None
    return torch.where(
        temperature[idx_mapping_long] < _SAMPLING_EPS,
        greedy_sampled,
        random_sampled,
        out=greedy_sampled,
    )


def apply_sampling_constraints_with_runtime_state(
    logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    temperature: torch.Tensor | None,
    top_k: torch.Tensor | None,
    top_p: torch.Tensor | None,
) -> torch.Tensor:
    needed = max(int(idx_mapping.shape[0]), 1)
    if temperature is None and top_k is None and top_p is None:
        return logits
    temperature = _ensure_runtime_state_tensor(
        temperature,
        needed,
        0.0,
        torch.float32,
        logits.device,
    )
    idx_mapping_long = idx_mapping.to(torch.long)
    row_temperature = temperature[idx_mapping_long]
    safe_temperature = torch.where(
        row_temperature < _SAMPLING_EPS,
        torch.ones_like(row_temperature),
        row_temperature,
    )
    logits.div_(safe_temperature.unsqueeze(dim=1))
    if top_k is None and top_p is None:
        return logits
    top_k_base = _ensure_runtime_state_tensor(
        top_k,
        needed,
        logits.shape[1],
        torch.int32,
        logits.device,
    )
    top_p_base = _ensure_runtime_state_tensor(
        top_p,
        needed,
        1.0,
        torch.float32,
        logits.device,
    )
    return apply_top_k_top_p(
        logits,
        top_k_base[idx_mapping_long],
        top_p_base[idx_mapping_long],
    )


class AscendSampler(Sampler):
    uses_seeded_gumbel = True

    @staticmethod
    def apply_penalties(
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        output_token_ids: list[list[int]],
    ) -> torch.Tensor:
        """Use Triton-Ascend penalties on NPU when Triton is available; else vLLM default."""
        if not HAS_TRITON:
            return Sampler.apply_penalties(logits, sampling_metadata, output_token_ids)

        if sampling_metadata.no_penalties:
            return logits
        assert sampling_metadata.prompt_token_ids is not None
        return apply_all_penalties(
            logits,
            sampling_metadata.prompt_token_ids,
            sampling_metadata.presence_penalties,
            sampling_metadata.frequency_penalties,
            sampling_metadata.repetition_penalties,
            output_token_ids,
        )

    def __init__(self, logprobs_mode=DEFAULT_LOGPROBS_MODE):
        # TODO: support logprobs_mode in vllm-ascend
        super().__init__(logprobs_mode=logprobs_mode)
        self.topk_topp_sampler = AscendTopKTopPSampler(logprobs_mode=logprobs_mode)
        self.async_exponential_event = torch.npu.Event()

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        predict_bonus_token: bool = False,
        logprobs_mode_override=None,
    ) -> SamplerOutput:
        logprobs_mode = logprobs_mode_override or self.logprobs_mode
        _log_sample_trace(
            "forward_enter",
            logits_rows=logits.shape[0],
            logits_cols=logits.shape[1],
            predict_bonus_token=predict_bonus_token,
            max_num_logprobs=sampling_metadata.max_num_logprobs,
            logprobs_mode=logprobs_mode,
        )
        num_logprobs = sampling_metadata.max_num_logprobs
        if num_logprobs is not None:
            if logprobs_mode == "raw_logprobs":
                raw_logprobs = self.compute_logprobs(logits)
            elif logprobs_mode == "raw_logits":
                if logits.dtype == torch.float32:
                    raw_logprobs = logits.clone()
                else:
                    raw_logprobs = logits.to(torch.float32)

        logits = logits.to(torch.float32)
        logits = self.apply_logits_processors(
            logits, sampling_metadata, predict_bonus_token)
        _log_sample_trace("forward_after_logits_processors")

        sampled, processed_logprobs = self.sample(logits, sampling_metadata)
        _log_sample_trace(
            "forward_after_sample",
            sampled_rows=sampled.shape[0],
            sampled_dtype=str(sampled.dtype),
            processed_logprobs=processed_logprobs is not None,
        )
        if processed_logprobs is not None:
            raw_logprobs = processed_logprobs

        sampled = sampled.long()
        _log_sample_trace("forward_after_long_cast")

        logprob_token_ids_tensors = None
        if sampling_metadata.logprob_token_ids:
            logprob_token_ids_tensors = self.gather_specific_token_logprobs(
                logits, sampling_metadata.logprob_token_ids, sampled
            )
            _log_sample_trace("forward_specific_logprob_ids")

        if num_logprobs is None:
            logprobs_tensors = logprob_token_ids_tensors
        elif num_logprobs == -1:
            logprobs_tensors = LogprobsTensors(
                torch.empty(0), raw_logprobs, torch.empty(0)
            )
        else:
            logprobs_tensors = self.gather_logprobs(
                raw_logprobs, num_logprobs, token_ids=sampled
            )
            _log_sample_trace("forward_gather_logprobs")

        if logprob_token_ids_tensors is not None and num_logprobs is not None:
            logprobs_tensors = logprob_token_ids_tensors

        sampled = sampled.to(torch.int32)
        sampler_output = SamplerOutput(
            sampled_token_ids=sampled.unsqueeze(-1).contiguous(),
            logprobs_tensors=logprobs_tensors,
        )
        _log_sample_trace(
            "forward_exit",
            sampled_rows=sampler_output.sampled_token_ids.shape[0],
            sampled_cols=sampler_output.sampled_token_ids.shape[1],
            sampled_dtype=str(sampler_output.sampled_token_ids.dtype),
            has_logprobs=logprobs_tensors is not None,
        )
        return sampler_output

    def set_q_event(self, q, event):
        self.topk_topp_sampler.set_q_event(q, event)

    def prepare_sampling(self, top_k):
        self.topk_topp_sampler.prepare_sampling(top_k)

    def do_async_exponential(self, b_s, head_dim, generators):
        if self.uses_seeded_gumbel:
            _log_sample_trace(
                "skip_async_exponential",
                reason="seeded_gumbel_active",
                batch_size=b_s,
                head_dim=head_dim,
            )
            return
        _log_sample_trace(
            "run_async_exponential",
            batch_size=b_s,
            head_dim=head_dim,
            generator_count=len(generators),
        )
        # Calculating exponential randoms in a different stream
        # and overlapping with model executing.
        with torch.npu.stream(global_stream()):
            global_stream().wait_stream(torch.npu.current_stream())
            q = torch.empty((b_s, head_dim), device="npu", dtype=torch.float32)
            # Goes to async exponential with AI-CPU exponential or default exponential.
            if len(generators) != q.shape[0]:
                q.exponential_()
            if generators:
                for i, generator in generators.items():
                    q[i].exponential_(generator=generator)
            self.async_exponential_event.record()
        self.set_q_event(q, self.async_exponential_event)

    def _sample_seeded_gumbel(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        logprobs_mode_override=None,
    ):
        if getattr(sampling_metadata, "_ascend_disable_runtime_sampling", False):
            _log_sample_trace(
                "seeded_gumbel_disabled",
                reason="runtime_sampling_disabled_for_path",
            )
            return None
        logprobs_mode = logprobs_mode_override or self.logprobs_mode
        positions = getattr(sampling_metadata, "_ascend_positions", None)
        idx_mapping = getattr(sampling_metadata, "_ascend_idx_mapping", None)
        seeds = getattr(sampling_metadata, "seeds", None)
        if positions is None or idx_mapping is None or seeds is None:
            _log_sample_trace(
                "seeded_gumbel_unavailable",
                has_positions=positions is not None,
                has_idx_mapping=idx_mapping is not None,
                has_seeds=seeds is not None,
            )
            return None
        needed = max(int(idx_mapping.shape[0]), 1)
        _log_sample_trace(
            "seeded_gumbel_enter",
            needed=needed,
            logits_rows=logits.shape[0],
            vocab=logits.shape[1],
        )

        all_greedy = sampling_metadata.all_greedy
        all_random = sampling_metadata.all_random and not all_greedy
        temperature_base = _ensure_runtime_state_tensor(
            sampling_metadata.temperature,
            needed,
            1.0,
            torch.float32,
            logits.device,
        )
        seeds = _ensure_runtime_state_tensor(
            seeds,
            needed,
            0,
            torch.int64,
            logits.device,
        )

        if all_random:
            greedy_sampled = None
        else:
            greedy_sampled = self.greedy_sample(logits)
            if all_greedy:
                _log_sample_trace("seeded_gumbel_all_greedy")
                processed_logprobs = None
                if sampling_metadata.max_num_logprobs is not None:
                    if logprobs_mode == "processed_logits":
                        processed_logprobs = logits
                    elif logprobs_mode == "processed_logprobs":
                        processed_logprobs = self.compute_logprobs(logits)
                return greedy_sampled, processed_logprobs

        idx_mapping_long = idx_mapping.to(torch.long)
        row_temperature = temperature_base[idx_mapping_long]
        if not all_random:
            row_temperature = torch.where(
                row_temperature < _SAMPLING_EPS,
                torch.ones_like(row_temperature),
                row_temperature,
            )
        logits = logits.div_(row_temperature.unsqueeze(dim=1))

        for processor in sampling_metadata.logitsprocs.argmax_invariant:
            logits = processor.apply(logits)

        if sampling_metadata.top_k is not None or sampling_metadata.top_p is not None:
            _log_sample_trace(
                "seeded_gumbel_apply_topk_topp",
                has_top_k=sampling_metadata.top_k is not None,
                has_top_p=sampling_metadata.top_p is not None,
            )
            top_k_base = _ensure_runtime_state_tensor(
                sampling_metadata.top_k,
                needed,
                logits.shape[1],
                torch.int32,
                logits.device,
            )
            top_p_base = _ensure_runtime_state_tensor(
                sampling_metadata.top_p,
                needed,
                1.0,
                torch.float32,
                logits.device,
            )
            row_top_k = top_k_base[idx_mapping_long]
            row_top_p = top_p_base[idx_mapping_long]
            logits = apply_top_k_top_p(logits, row_top_k, row_top_p)

        processed_logprobs = None
        if logprobs_mode == "processed_logits":
            processed_logprobs = logits
        elif logprobs_mode == "processed_logprobs":
            processed_logprobs = logits.log_softmax(dim=-1, dtype=torch.float32)

        random_sampled = gumbel_sample(
            logits,
            idx_mapping.to(torch.int32),
            temperature_base,
            seeds,
            positions.to(torch.int32),
            apply_temperature=False,
        )

        if greedy_sampled is None:
            _log_sample_trace("seeded_gumbel_random_only")
            return random_sampled, processed_logprobs

        sampled = torch.where(
            temperature_base[idx_mapping_long] < _SAMPLING_EPS,
            greedy_sampled,
            random_sampled,
            out=greedy_sampled,
        )
        _log_sample_trace("seeded_gumbel_mixed_complete")
        return sampled, processed_logprobs

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        logprobs_mode_override=None,
    ):
        seeded = self._sample_seeded_gumbel(logits, sampling_metadata, logprobs_mode_override)
        if seeded is not None:
            _log_sample_trace("sample_return_seeded_gumbel")
            return seeded
        logprobs_mode = logprobs_mode_override or self.logprobs_mode
        _log_sample_trace(
            "sample_regular_enter",
            all_greedy=sampling_metadata.all_greedy,
            all_random=sampling_metadata.all_random,
            has_temperature=sampling_metadata.temperature is not None,
            has_top_k=sampling_metadata.top_k is not None,
            has_top_p=sampling_metadata.top_p is not None,
            argmax_invariant_count=len(
                sampling_metadata.logitsprocs.argmax_invariant),
        )
        assert not (
            sampling_metadata.all_greedy and sampling_metadata.all_random)
        if sampling_metadata.all_random:
            greedy_sampled = None
        else:
            greedy_sampled = self.greedy_sample(logits)
            if sampling_metadata.all_greedy:
                _log_sample_trace("sample_regular_all_greedy")
                processed_logprobs = None
                if sampling_metadata.max_num_logprobs is not None:
                    if logprobs_mode == "processed_logits":
                        processed_logprobs = logits
                    elif logprobs_mode == "processed_logprobs":
                        processed_logprobs = self.compute_logprobs(logits)
                return greedy_sampled, processed_logprobs

        assert sampling_metadata.temperature is not None
        _log_sample_trace("sample_regular_apply_temperature")
        logits = self.apply_temperature(
            logits, sampling_metadata.temperature, sampling_metadata.all_random
        )

        for processor in sampling_metadata.logitsprocs.argmax_invariant:
            logits = processor.apply(logits)
        if sampling_metadata.logitsprocs.argmax_invariant:
            _log_sample_trace("sample_regular_after_argmax_invariant")

        _log_sample_trace("sample_regular_topk_topp_enter")
        random_sampled, processed_logprobs = self.topk_topp_sampler(
            logits,
            sampling_metadata.generators,
            sampling_metadata.top_k,
            sampling_metadata.top_p,
        )
        _log_sample_trace(
            "sample_regular_topk_topp_exit",
            sampled_rows=random_sampled.shape[0],
        )

        if greedy_sampled is None:
            _log_sample_trace("sample_regular_return_random")
            return random_sampled, processed_logprobs

        sampled = torch.where(
            sampling_metadata.temperature < _SAMPLING_EPS,
            greedy_sampled,
            random_sampled,
            out=greedy_sampled,
        )
        _log_sample_trace("sample_regular_return_mixed")
        return sampled, processed_logprobs

    @staticmethod
    def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
        if get_ascend_config().enable_reduce_sample:
            tp_group = get_tp_group()
            B, V_local = logits.shape
            rank = tp_group.rank_in_group

            local_max_logits, local_max_indices = logits.max(dim=-1)
            local_global_idx = local_max_indices + rank * V_local  # [B]
            # [B, world_size]
            gathered_logits = tp_group.all_gather(local_max_logits.unsqueeze(-1), dim=-1)
            gathered_global_idx = tp_group.all_gather(local_global_idx.unsqueeze(-1), dim=-1)  # [B, world_size]
            global_max_rank = gathered_logits.argmax(dim=-1)  # [B]
            target_argmax = gathered_global_idx.gather(dim=-1, index=global_max_rank.unsqueeze(-1)).squeeze(-1)  # [B]
            return target_argmax
        else:
            return logits.argmax(dim=-1).view(-1)


class AscendTopKTopPSampler(TopKTopPSampler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.apply_top_k_top_p = apply_top_k_top_p
        self.top_k = None

    def set_q_event(self, q, event):
        # Pass in async exponential results.
        # Also pass in event to prevent synchronize errors.
        self.q = q
        self.async_event = event
        _log_sample_trace(
            "topk_topp_set_q_event",
            q_rows=q.shape[0],
            q_cols=q.shape[1],
        )

    def prepare_sampling(self, top_k):
        if top_k is not None:
            self.top_k = top_k
        else:
            self.top_k = None

    def forward_native(self, logits, generators, k, p):
        """Override pytorch native implementation to torch_npu"""
        _log_sample_trace(
            "topk_topp_forward_enter",
            logits_rows=logits.shape[0],
            logits_cols=logits.shape[1],
            has_k=k is not None,
            has_p=p is not None,
            reduce_sample=get_ascend_config().enable_reduce_sample,
            batch_invariant=envs.VLLM_BATCH_INVARIANT,
        )
        # when batch_invariant mode is enabled, we should use vllm's implementation.
        # or it will make batch_invariant mode not working.
        if envs.VLLM_BATCH_INVARIANT:
            _log_sample_trace("topk_topp_forward_super_batch_invariant")
            return super().forward_native(logits, generators, k, p)

        if get_ascend_config().enable_reduce_sample:
            _log_sample_trace("topk_topp_forward_reduce_sample")
            cand_logits, cand_idx = self.apply_top_k_top_p(logits, k, p, self.top_k)
            logits_to_return = None
            if self.logprobs_mode == "processed_logits":
                logits_to_return = cand_logits
            elif self.logprobs_mode == "processed_logprobs":
                logits_to_return = cand_logits.log_softmax(dim=-1, dtype=torch.float32)

            probs = torch.softmax(cand_logits, dim=-1)
            pos = random_sample(probs, generators)  # [B]

            next_token = cand_idx.gather(dim=1, index=pos.unsqueeze(1)).squeeze(1)  # [B]
            _log_sample_trace("topk_topp_forward_reduce_sample_done")
            return next_token, logits_to_return
        else:
            _log_sample_trace(
                "topk_topp_forward_native_path",
                async_exponential=get_ascend_config().enable_async_exponential,
            )
            logits = self.apply_top_k_top_p(logits, k, p)
            logits_to_return = None
            if self.logprobs_mode == "processed_logits":
                logits_to_return = logits
            elif self.logprobs_mode == "processed_logprobs":
                logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)

            probs = logits.softmax(dim=-1, dtype=torch.float32)
            if get_ascend_config().enable_async_exponential:
                # Add synchronize to prevent synchronize error.
                _log_sample_trace("topk_topp_forward_wait_async_event")
                self.async_event.synchronize()
                _log_sample_trace("topk_topp_forward_async_event_ready")
                return probs.div_(self.q).argmax(dim=-1).view(-1), logits_to_return
            _log_sample_trace("topk_topp_forward_random_sample")
            return random_sample(probs, generators), logits_to_return


def _apply_top_k_top_p_pytorch(
    logits: torch.Tensor,  # [B, V_local]
    k: torch.Tensor,  # [B] or None
    p: torch.Tensor,  # [B] or None
    top_k: int | None = None,
) -> torch.Tensor:
    if get_ascend_config().enable_reduce_sample:
        tp_group = get_tp_group()
        B, V_local = logits.shape
        world_size = tp_group.world_size
        rank = tp_group.rank_in_group
        V_global = V_local * world_size

        local_vals, local_idx = torch.topk(logits, k=top_k, dim=-1)  # [B, top_k], [B, top_k]
        local_global_idx = local_idx + rank * V_local  # [B, top_k]

        gathered_vals = tp_group.all_gather(local_vals, dim=-1)  # [B, top_k*tp]
        gathered_idx = tp_group.all_gather(local_global_idx, dim=-1)  # [B, top_k*tp]

        full_logits = logits.new_full((B, V_global), -float("inf"))
        full_logits.scatter_(dim=-1, index=gathered_idx, src=gathered_vals)

        if p is None and k is None:
            return full_logits
        probs = full_logits.softmax(dim=-1)
        probs_sort, _ = probs.sort(dim=-1, descending=False)
        if k is not None:
            kk = k.to(torch.long).clamp(min=1, max=V_global)
            top_k_count = (probs_sort.size(1) - kk).unsqueeze(1)  # [B,1]
            top_k_cutoff = probs_sort.gather(-1, top_k_count)
            no_top_k_mask = (kk == V_global).unsqueeze(1)
            top_k_cutoff.masked_fill_(no_top_k_mask, -float("inf"))
            elements_to_discard = probs < top_k_cutoff
            full_logits.masked_fill_(elements_to_discard, -float("inf"))
        if p is not None:
            cumprob = torch.cumsum(probs_sort, dim=-1)
            top_p_mask = cumprob <= (1 - p.unsqueeze(1))
            top_p_mask[:, -1] = False  # at least one
            top_p_count = top_p_mask.sum(dim=-1, keepdim=True)
            top_p_cutoff = probs_sort.gather(-1, top_p_count)
            elements_to_discard = probs < top_p_cutoff
            full_logits.masked_fill_(elements_to_discard, -float("inf"))
        return full_logits
    else:
        if p is None and k is None:
            return logits

        probs = logits.softmax(dim=-1)
        probs_sort, _ = probs.sort(dim=-1, descending=False)

        if k is not None:
            top_k_count = probs_sort.size(1) - k.to(torch.long)  # shape: (batch, )
            top_k_count = top_k_count.unsqueeze(dim=1)
            top_k_cutoff = probs_sort.gather(-1, top_k_count)

            # Make sure the no top-k rows are no-op.
            no_top_k_mask = (k == logits.shape[1]).unsqueeze(dim=1)
            top_k_cutoff.masked_fill_(no_top_k_mask, -float("inf"))

            elements_to_discard = probs < top_k_cutoff
            logits.masked_fill_(elements_to_discard, -float("inf"))

        if p is not None:
            cumprob = torch.cumsum(probs_sort, dim=-1)
            top_p_mask = cumprob <= 1 - p.unsqueeze(dim=1)
            top_p_mask[:, -1] = False  # at least one

            top_p_count = top_p_mask.sum(dim=-1).unsqueeze(1)
            top_p_cutoff = probs_sort.gather(-1, top_p_count)
            elements_to_discard = probs < top_p_cutoff
            logits.masked_fill_(elements_to_discard, -float("inf"))

        return logits


def _apply_top_k_top_p_ascendc(
    logits: torch.Tensor,
    k: torch.Tensor,
    p: torch.Tensor,
    top_k: int | None = None,
) -> torch.Tensor:
    if get_ascend_config().enable_reduce_sample:
        tp_group = get_tp_group()
        B, V_local = logits.shape
        rank = tp_group.rank_in_group

        local_vals, local_idx = torch.topk(logits, k=top_k, dim=-1)  # [B, top_k], [B, top_k]

        local_global_idx = local_idx + rank * V_local  # [B, top_k]

        gathered_vals = tp_group.all_gather(local_vals, dim=-1)  # [B, top_k*tp]
        gathered_idx = tp_group.all_gather(local_global_idx, dim=-1)  # [B, top_k*tp]

        if p is None and k is None:
            return logits
        gathered_vals = torch.ops._C_ascend.npu_apply_top_k_top_p(gathered_vals, k=k, p=p)
        return gathered_vals, gathered_idx
    else:
        if p is None and k is None:
            return logits
        return torch.ops._C_ascend.npu_apply_top_k_top_p(logits, k=k, p=p)


apply_top_k_top_p = (
    _apply_top_k_top_p_ascendc
    if get_ascend_device_type() in [AscendDeviceType.A2, AscendDeviceType.A3]
    else _apply_top_k_top_p_pytorch
)
