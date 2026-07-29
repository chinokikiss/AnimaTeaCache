import logging

import comfy.model_base
import comfy.patcher_extension
import comfy.samplers
from nodes import common_ksampler

from .patches import TeaCacheState


TEACACHE_CONFIGS = {
    ("euler", "normal"): (0.03, [5.32030879e+00, -9.40726208e-01, 7.64571376e-02]),
    ("euler", "simple"): (0.03, [4.08388264e+00, -4.86643619e-01, 6.14516484e-02]),
    ("euler", "beta57"): (0.25, [-1.05261817e+00, 2.87565126e+00, -2.44596262e+00, 8.01109257e-01, 3.33674738e-02]),
    ("er_sde", "simple"): (0.075, [5.09107664e+00, -2.23509746e+01, 3.75468173e+01, -2.99490918e+01, 1.13310514e+01, -1.60137682e+00, 1.24625360e-01]),
    ("uni_pc", "ddim_uniform"): (0.055, [6.49855575e+00, -1.33734900e+00, 1.14792987e-01]),
}


class TeaCacheKSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "rel_l1_thresh": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "Anima_TeaCache"

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, rel_l1_thresh, denoise=1.0):
        if rel_l1_thresh == 0:
            return common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)

        if not isinstance(model.model, comfy.model_base.Anima):
            logging.warning("TeaCache disabled: model is not Anima.")
            return common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)

        config = TEACACHE_CONFIGS.get((sampler_name, scheduler))
        if config is None:
            logging.warning(f"TeaCache disabled: unsupported sampler '{sampler_name}' with scheduler '{scheduler}'.")
            return common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)

        force_recalc_thresh, coefficients = config
        state = TeaCacheState(rel_l1_thresh, force_recalc_thresh, coefficients)
        model = model.clone()
        model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "anima_teacache", state.forward_wrapper)

        try:
            out = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)
        finally:
            model_calls = state.model_calls
            cache_hits = state.cache_hits
            model.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "anima_teacache")
            state.reset()

        if model_calls:
            logging.info(f"TeaCache: reused {cache_hits}/{model_calls} model calls ({cache_hits / model_calls * 100:.2f}%).")
        return out
