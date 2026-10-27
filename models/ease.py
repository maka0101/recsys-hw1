import numpy as np
from scipy import linalg, sparse


class EASE:

    def __init__(self, regularization=1000):
        if not np.isfinite(regularization) or regularization <= 0:
            raise ValueError("regularization must be finite and positive")
        self.regularization = float(regularization)

    def fit(self, ratings: sparse.csr_matrix):
        if not sparse.isspmatrix_csr(ratings):
            raise TypeError("ratings must be a scipy.sparse.csr_matrix")
        if min(ratings.shape) == 0:
            raise ValueError("ratings must have at least one user and one item")
        history = ratings.copy()
        history.sort_indices()
        if not history.has_canonical_format:
            raise ValueError("ratings must not contain duplicate user-item entries")
        if (
            not np.isfinite(history.data).all()
            or (history.data < 1).any()
            or (history.data > 5).any()
        ):
            raise ValueError("stored ratings must be finite and in [1, 5]")
        self.history_ = history.astype(np.float64, copy=False)
        self.n_users_, self.n_items_ = self.history_.shape
        positive = self.history_.copy()
        positive.data = (positive.data >= 4).astype(np.float64)
        positive.eliminate_zeros()
        self.positive_history_ = positive

        gram = (positive.T @ positive).toarray(order="F")
        gram.flat[:: self.n_items_ + 1] += self.regularization
        factor = linalg.cho_factor(
            gram, lower=True, overwrite_a=True, check_finite=False
        )
        precision = linalg.cho_solve(
            factor,
            np.eye(self.n_items_, dtype=np.float64, order="F"),
            overwrite_b=True,
            check_finite=False,
        )
        diagonal = np.diag(precision).copy()
        # B_ij = -P_ij / P_jj; нормировка по целевым столбцам.
        precision /= -diagonal[np.newaxis, :]
        np.fill_diagonal(precision, 0.0)
        self.coef_ = precision
        return self

    def score_users(self, user_ids: np.ndarray) -> np.ndarray:
        if not hasattr(self, "coef_"):
            raise RuntimeError("fit must be called before score_users")
        user_ids = np.asarray(user_ids)
        if user_ids.ndim != 1:
            raise ValueError("user_ids must be one-dimensional")
        if user_ids.size == 0:
            return np.empty((0, self.n_items_), dtype=np.float64)
        if not np.issubdtype(user_ids.dtype, np.integer):
            raise TypeError("user_ids must contain integers")
        if (user_ids < 0).any() or (user_ids >= self.n_users_).any():
            raise IndexError("user_ids are outside the fitted user range")
        return np.asarray(self.positive_history_[user_ids] @ self.coef_)
