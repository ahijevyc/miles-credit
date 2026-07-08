import torch.nn as nn


class BasePreblock(nn.Module):
    """
    Base class for all preblocks. Enforces the forward signature and
    provides the from_config classmethod used by the registry.
    """

    VALID_DATA_TYPES = ("input", "target")

    def _copy_batch(self, batch: dict) -> dict:
        """Shallow-copy the batch so forward() never mutates the caller's dict.

        Creates new dict objects at the data_type and source levels; tensor
        values are shared (not copied) since preblocks must not mutate tensors
        in-place anyway (doing so would break autograd).
        """
        copied_batch = {}
        for data_type, sources in batch.items():
            if not isinstance(sources, dict):
                copied_batch[data_type] = sources
                continue

            copied_sources = {}
            for source, variables in sources.items():
                copied_sources[source] = dict(variables) if isinstance(variables, dict) else variables
            copied_batch[data_type] = copied_sources

        return copied_batch

    def forward(self, batch: dict) -> dict:
        pass

    @classmethod
    def from_config(cls, **kwargs):
        return cls(**kwargs)
