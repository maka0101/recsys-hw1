from numbers import Integral

import numpy as np
from scipy import sparse


class ItemKNN:

    def __init__(self, k=50, shrinkage=10):
        if isinstance(k, bool) or not isinstance(k, Integral) or k < 1:
            raise ValueError("k must be a positive integer")
        if not np.isfinite(shrinkage) or shrinkage < 0:
            raise ValueError("shrinkage must be finite and nonnegative")
        self.k = int(k)
        self.shrinkage = float(shrinkage)

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

        overlaps = (positive.T @ positive).tocsc()
        overlaps.setdiag(0)
        overlaps.eliminate_zeros()
        norms = np.sqrt(np.asarray(positive.sum(axis=0)).ravel())
        neighbor_indices = []
        neighbor_weights = []
        indptr = np.zeros(self.n_items_ + 1, dtype=np.int64)
        # Top-k выбирается для каждого целевого столбца.
        for target in range(self.n_items_):
            start, end = overlaps.indptr[target : target + 2]
            sources = overlaps.indices[start:end]
            counts = overlaps.data[start:end]
            values = counts / (norms[sources] * norms[target])
            values *= counts / (counts + self.shrinkage)
            selected = np.lexsort((sources, -values))[: self.k]
            neighbor_indices.append(sources[selected])
            neighbor_weights.append(values[selected])
            indptr[target + 1] = indptr[target] + selected.size

        similarity = sparse.csc_matrix(
            (
                np.concatenate(neighbor_weights),
                np.concatenate(neighbor_indices),
                indptr,
            ),
            shape=(self.n_items_, self.n_items_),
            dtype=np.float64,
        )
        self.similarity_ = similarity.tocsr()
        self.similarity_.sort_indices()
        return self

    def score_users(self, user_ids: np.ndarray) -> np.ndarray:
        if not hasattr(self, "similarity_"):
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
        return (self.positive_history_[user_ids] @ self.similarity_).toarray()
