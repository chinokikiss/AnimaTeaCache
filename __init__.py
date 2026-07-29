from .sampler import TeaCacheKSampler

NODE_CLASS_MAPPINGS = {
    "TeaCacheKSampler": TeaCacheKSampler,
}

__all__ = ["NODE_CLASS_MAPPINGS"]
