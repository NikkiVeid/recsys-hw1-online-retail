"""iALS с непрерывным preference target; confidence совпадает с обычным iALS."""
import numpy as np
from .ials import IALS


class CustomIALS(IALS):
    def __init__(self, tau_days=180., saturation=2., **kwargs):
        super().__init__(**kwargs)
        self.tau_days, self.saturation = tau_days, saturation

    def make_targets(self, data):
        # Counts and age_days have exactly the same CSR ordering.
        assert np.array_equal(data.counts.indices, data.age_days.indices)
        assert np.array_equal(data.counts.indptr, data.age_days.indptr)
        return (1. - np.exp(-data.counts.data / self.saturation)) * np.exp(
            -data.age_days.data / self.tau_days)
