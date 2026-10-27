from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.linalg import solve


class IALS:

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
    ) -> None:
        if not isinstance(factors, (int, np.integer)) or factors < 1:
            raise ValueError("factors must be a positive integer")
        if not isinstance(iterations, (int, np.integer)) or iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not np.isfinite(regularization) or regularization <= 0:
            raise ValueError("regularization must be finite and positive")
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be finite and nonnegative")
        if not np.isfinite(positive_threshold) or not 1 <= positive_threshold <= 5:
            raise ValueError("positive_threshold must be in [1, 5]")
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("tolerance must be finite and nonnegative")
        if (
            not isinstance(objective_batch_size, (int, np.integer))
            or objective_batch_size < 1
        ):
            raise ValueError("objective_batch_size must be a positive integer")
        self.factors = int(factors)
        self.regularization = float(regularization)
        self.alpha = float(alpha)
        self.iterations = int(iterations)
        self.seed = seed
        self.positive_threshold = float(positive_threshold)
        self.tolerance = float(tolerance)
        self.objective_batch_size = int(objective_batch_size)

    @staticmethod
    def _validate_ratings(ratings: sparse.csr_matrix) -> sparse.csr_matrix:
        if not sparse.issparse(ratings) or ratings.format != "csr":
            raise TypeError("ratings must be a scipy CSR sparse matrix")
        if ratings.ndim != 2 or min(ratings.shape) < 1:
            raise ValueError("ratings must have at least one user and one item")
        result = sparse.csr_matrix(ratings, dtype=np.float64, copy=True)
        result.eliminate_zeros()
        result.sort_indices()
        if not np.all(np.isfinite(result.data)) or np.any(
            (result.data < 1) | (result.data > 5)
        ):
            raise ValueError("observed ratings must be finite and in [1, 5]")
        if not result.has_canonical_format:
            for start, end in zip(result.indptr[:-1], result.indptr[1:]):
                if np.any(np.diff(result.indices[start:end]) == 0):
                    raise ValueError("duplicate user/item ratings must be resolved first")
        return result

    def _make_targets(self, ratings: sparse.csr_matrix) -> np.ndarray:
        return (ratings.data >= self.positive_threshold).astype(np.float64)

    def fit(self, ratings: sparse.csr_matrix) -> IALS:
        self.ratings_ = self._validate_ratings(ratings)
        self.n_users_, self.n_items_ = self.ratings_.shape
        self.global_mean_ = (
            float(self.ratings_.data.mean()) if self.ratings_.nnz else 3.0
        )
        self._observation_rows_ = np.repeat(
            np.arange(self.n_users_), np.diff(self.ratings_.indptr)
        )
        self.positive_counts_ = np.bincount(
            self._observation_rows_[
                self.ratings_.data >= self.positive_threshold
            ],
            minlength=self.n_users_,
        )
        self.cold_positive_users_ = self.positive_counts_ == 0
        self.known_item_mask_ = (
            np.bincount(self.ratings_.indices, minlength=self.n_items_) > 0
        )
        self.confidence_on_observed_ = 1.0 + self.alpha
        target_values = self._make_targets(self.ratings_)
        # Нулевые цели наблюдений сохраняют вес 1 + alpha.
        self.targets_ = sparse.csr_matrix(
            (
                target_values,
                self.ratings_.indices.copy(),
                self.ratings_.indptr.copy(),
            ),
            shape=self.ratings_.shape,
        )
        observed_items = self.ratings_.T.tocsr()
        target_items = self.targets_.T.tocsr()

        rng = np.random.default_rng(self.seed)
        self.user_factors_ = rng.normal(
            0.0, 0.01, (self.n_users_, self.factors)
        )
        self.item_factors_ = rng.normal(
            0.0, 0.01, (self.n_items_, self.factors)
        )
        self.objective_history_ = [self.objective()]
        self.n_iterations_ = 0
        for iteration in range(self.iterations):
            self._update_factors(
                self.ratings_, self.targets_, self.item_factors_, self.user_factors_
            )
            self._update_factors(
                observed_items, target_items, self.user_factors_, self.item_factors_
            )
            current = self.objective()
            previous = self.objective_history_[-1]
            if current > previous + 1e-8 * max(1.0, abs(previous)):
                raise FloatingPointError(
                    "ALS objective increased after exact alternating updates"
                )
            self.objective_history_.append(current)
            self.n_iterations_ = iteration + 1
            if self.tolerance and previous - current <= self.tolerance * max(
                1.0, abs(previous)
            ):
                break
        self.objective_ = self.objective_history_[-1]
        return self

    def _update_factors(
        self,
        observations: sparse.csr_matrix,
        targets: sparse.csr_matrix,
        fixed_factors: np.ndarray,
        out: np.ndarray,
    ) -> None:
        gram = fixed_factors.T @ fixed_factors
        gram.flat[:: self.factors + 1] += self.regularization
        for row in range(observations.shape[0]):
            start, end = observations.indptr[row : row + 2]
            values = targets.data[start:end]
            if start == end or not np.any(values):
                out[row].fill(0.0)
                continue
            factors = fixed_factors[observations.indices[start:end]]
            coefficient = gram + self.alpha * (factors.T @ factors)
            rhs = self.confidence_on_observed_ * (values @ factors)
            out[row] = solve(
                coefficient,
                rhs,
                assume_a="pos",
                check_finite=False,
                overwrite_a=True,
            )

    def objective(self) -> float:
        if not hasattr(self, "user_factors_"):
            raise RuntimeError("fit the model before computing its objective")
        user_gram = self.user_factors_.T @ self.user_factors_
        item_gram = self.item_factors_.T @ self.item_factors_
        loss = float(np.sum(user_gram * item_gram))
        for start in range(0, self.ratings_.nnz, self.objective_batch_size):
            end = min(start + self.objective_batch_size, self.ratings_.nnz)
            prediction = np.einsum(
                "ij,ij->i",
                self.user_factors_[self._observation_rows_[start:end]],
                self.item_factors_[self.ratings_.indices[start:end]],
            )
            target = self.targets_.data[start:end]
            # Для наблюдений: pred² → (1 + alpha) * (target - pred)².
            loss += float(
                self.alpha * (prediction @ prediction)
                - 2.0 * self.confidence_on_observed_ * (target @ prediction)
                + self.confidence_on_observed_ * (target @ target)
            )
        loss += self.regularization * float(
            np.sum(self.user_factors_**2) + np.sum(self.item_factors_**2)
        )
        if not np.isfinite(loss):
            raise FloatingPointError("ALS produced a nonfinite objective")
        if loss < -1e-7:
            raise FloatingPointError("ALS produced a negative squared-error objective")
        return max(0.0, loss)

    def score_users(self, user_ids: np.ndarray) -> np.ndarray:
        if not hasattr(self, "user_factors_"):
            raise RuntimeError("fit the model before requesting scores")
        ids = np.asarray(user_ids)
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("user_ids must be a one-dimensional integer array")
        if np.any((ids < 0) | (ids >= self.n_users_)):
            raise IndexError("user ID outside the fitted user catalog")
        return self.user_factors_[ids] @ self.item_factors_.T
