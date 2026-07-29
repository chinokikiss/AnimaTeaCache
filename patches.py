import torch

import comfy.ldm.common_dit


def _polynomial(coefficients, value):
    result = 0.0
    for coefficient in coefficients:
        result = result * value + coefficient
    return result


def _relative_l1(previous, current):
    return ((previous - current).abs().mean() / previous.abs().mean().clamp_min(1e-8)).item()


def _is_last_step(transformer_options):
    sigmas = transformer_options.get("sigmas")
    sample_sigmas = transformer_options.get("sample_sigmas")
    if sigmas is None or sample_sigmas is None or sample_sigmas.numel() < 2:
        return False
    return torch.isclose(sigmas[0], sample_sigmas[-2].to(sigmas), rtol=1e-4, atol=1e-6).item()


def _modulated_input(block, x, embedding, adaln_lora, extra_pos_emb):
    if extra_pos_emb is not None:
        x = x + extra_pos_emb

    if block.use_adaln_lora:
        shift, scale, _ = (block.adaln_modulation_self_attn(embedding) + adaln_lora).chunk(3, dim=-1)
    else:
        shift, scale, _ = block.adaln_modulation_self_attn(embedding).chunk(3, dim=-1)

    shift = shift.unsqueeze(2).unsqueeze(2)
    scale = scale.unsqueeze(2).unsqueeze(2)
    return block.layer_norm_self_attn(x) * (1 + scale) + shift


class _TeaCacheEntry:
    def __init__(self):
        self.accumulated_rel_l1_distance = 0.0
        self.previous_modulated_input = None
        self.previous_residual = None

    def reset(self):
        self.accumulated_rel_l1_distance = 0.0
        self.previous_modulated_input = None
        self.previous_residual = None


class TeaCacheState:
    def __init__(self, rel_l1_thresh, force_recalc_thresh, coefficients):
        self.rel_l1_thresh = rel_l1_thresh
        self.force_recalc_thresh = force_recalc_thresh
        self.coefficients = coefficients
        self.entries = {}
        self.model_calls = 0
        self.cache_hits = 0

    def _cache_key(self, x, transformer_options):
        context_window = transformer_options.get("context_window")
        context_key = None if context_window is None else tuple(context_window.index_list)
        return (
            tuple(transformer_options.get("uuids", ())),
            tuple(transformer_options.get("cond_or_uncond", ())),
            context_key,
            tuple(x.shape),
            x.dtype,
            x.device,
        )

    def get_entry(self, x, transformer_options):
        key = self._cache_key(x, transformer_options)
        entry = self.entries.get(key)
        if entry is None:
            entry = _TeaCacheEntry()
            self.entries[key] = entry
        self.model_calls += 1
        return entry

    def reset(self):
        for entry in self.entries.values():
            entry.reset()
        self.entries.clear()
        self.model_calls = 0
        self.cache_hits = 0

    def forward_wrapper(self, executor, *args, **kwargs):
        return _anima_teacache_forward(executor.class_obj, self, *args, **kwargs)


def _anima_teacache_forward(
    self,
    state,
    x: torch.Tensor,
    timesteps: torch.Tensor,
    context: torch.Tensor,
    fps=None,
    padding_mask=None,
    **kwargs,
):
    orig_shape = x.shape
    x = comfy.ldm.common_dit.pad_to_patch_size(x, (self.patch_temporal, self.patch_spatial, self.patch_spatial))
    x_input = x
    crossattn_emb = context

    x, rope_emb, extra_pos_emb = self.prepare_embedded_sequence(x_input, fps=fps, padding_mask=padding_mask)

    if timesteps.ndim == 1:
        timesteps = timesteps.unsqueeze(1)
    t_embedding, adaln_lora = self.t_embedder[1](self.t_embedder[0](timesteps).to(x.dtype))
    t_embedding = self.t_embedding_norm(t_embedding)

    self.affline_scale_log_info = {"t_embedding_B_T_D": t_embedding.detach()}
    self.affline_emb = t_embedding
    self.crossattn_emb = crossattn_emb

    if extra_pos_emb is not None:
        assert x.shape == extra_pos_emb.shape, f"{x.shape} != {extra_pos_emb.shape}"

    transformer_options = kwargs.get("transformer_options", {})
    patches = transformer_options.get("patches", {})
    if "post_input" in patches:
        transformer_options = transformer_options.copy()
        transformer_options["model_patch_data"] = {}
        for patch in patches["post_input"]:
            out = patch({"img": x, "x": x_input, "transformer_options": transformer_options})
            x = out["img"]

    block_kwargs = {
        "rope_emb_L_1_1_D": rope_emb.unsqueeze(1).unsqueeze(0),
        "adaln_lora_B_T_3D": adaln_lora,
        "extra_per_block_pos_emb": extra_pos_emb,
        "transformer_options": transformer_options,
    }

    if x.dtype == torch.float16:
        x = x.float()

    modulated_input = _modulated_input(self.blocks[0], x, t_embedding, adaln_lora, extra_pos_emb)
    entry = state.get_entry(modulated_input, transformer_options)

    if entry.previous_modulated_input is None or entry.previous_residual is None or _is_last_step(transformer_options):
        should_calc = True
        entry.accumulated_rel_l1_distance = 0.0
    else:
        raw_dist = _relative_l1(entry.previous_modulated_input, modulated_input)
        if raw_dist < state.force_recalc_thresh:
            should_calc = True
            entry.accumulated_rel_l1_distance = 0.0
        else:
            entry.accumulated_rel_l1_distance += max(0.0, _polynomial(state.coefficients, raw_dist))
            should_calc = entry.accumulated_rel_l1_distance >= state.rel_l1_thresh
            if should_calc:
                entry.accumulated_rel_l1_distance = 0.0

    entry.previous_modulated_input = modulated_input

    if should_calc:
        for block_index, block in enumerate(self.blocks):
            transformer_options["block_index"] = block_index
            x = block(x, t_embedding, crossattn_emb, **block_kwargs)
        entry.previous_residual = x - modulated_input
    else:
        state.cache_hits += 1
        x = entry.previous_residual + modulated_input

    x = self.final_layer(x.to(crossattn_emb.dtype), t_embedding, adaln_lora_B_T_3D=adaln_lora)
    return self.unpatchify(x)[:, :, :orig_shape[-3], :orig_shape[-2], :orig_shape[-1]]
