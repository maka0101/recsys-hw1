from __future__ import annotations

import numpy as np
from scipy import sparse

from .ials import IALS


class CustomALS(IALS):

    def __init__(
        self,
        factors: int = 32,
        regularization: float = 0.1,
        alpha: float = 20.0,
        iterations: int = 10,
        seed: int = 42,
        positive_threshold: float = 4.0,
        tolerance: float = 0.0,
        objective_batch_size: int = 65_536,
        rho: float = 0.25,
        kappa: float = 10.0,
    ) -> None:
        super().__init__(
            factors=factors,
            regularization=regularization,
            alpha=alpha,
            iterations=iterations,
            seed=seed,
            positive_threshold=positive_threshold,
            tolerance=tolerance,
            objective_batch_size=objective_batch_size,
        )
        if not np.isfinite(rho) or not 0 <= rho <= 0.5:
            raise ValueError("rho must be in [0, 0.5] to retain clear rating signs")
        if not np.isfinite(kappa) or kappa <= 0:
            raise ValueError("kappa must be finite and positive")
        self.rho = float(rho)
        self.kappa = float(kappa)

    def _make_targets(self, ratings: sparse.csr_matrix) -> np.ndarray:
        counts = np.diff(ratings.indptr)
        sums = np.asarray(ratings.sum(axis=1)).ravel()
        rows = self._observation_rows_
        # Среднее без текущей оценки, со сглаживанием к общему.
        self.loo_user_means_ = (
            sums[rows] - ratings.data + self.kappa * self.global_mean_
        ) / (counts[rows] - 1 + self.kappa)
        absolute_satisfaction = (ratings.data - 3.0) / 2.0
        relative_satisfaction = (ratings.data - self.loo_user_means_) / 4.0
        return (
            (1.0 - self.rho) * absolute_satisfaction
            + self.rho * relative_satisfaction
        )
