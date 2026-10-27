from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import sparse
import torch
from torch import nn
from torch.nn import functional as F


def fm_interactions(feature_vectors: torch.Tensor) -> torch.Tensor:
    # Попарные взаимодействия: ((Σv)² - Σv²) / 2.
    summed = feature_vectors.sum(dim=-2)
    return 0.5 * (summed.square() - feature_vectors.square().sum(dim=-2)).sum(dim=-1)


class _FMNetwork(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_item_features: int,
        factors: int,
        item_indices: torch.Tensor,
        item_values: torch.Tensor,
        init_std: float,
        generator: torch.Generator,
    ) -> None:
        super().__init__()
        self.padding_index = n_item_features
        self.user_vectors = nn.Embedding(n_users, factors)
        self.user_linear = nn.Embedding(n_users, 1)
        self.item_vectors = nn.Embedding(
            n_item_features + 1, factors, padding_idx=self.padding_index
        )
        self.item_linear = nn.Embedding(
            n_item_features + 1, 1, padding_idx=self.padding_index
        )
        self.bias = nn.Parameter(torch.zeros(()))
        self.register_buffer("item_indices", item_indices)
        self.register_buffer("item_values", item_values)
        with torch.no_grad():
            nn.init.normal_(self.user_vectors.weight, std=init_std, generator=generator)
            nn.init.normal_(self.item_vectors.weight, std=init_std, generator=generator)
            self.user_linear.weight.zero_()
            self.item_linear.weight.zero_()
            self.item_vectors.weight[self.padding_index].zero_()

    def item_representation(self, item_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        indices = self.item_indices[item_ids]
        values = self.item_values[item_ids]
        vectors = self.item_vectors(indices) * values.unsqueeze(-1)
        summed_vectors = vectors.sum(dim=1)
        item_offset = (
            self.item_linear(indices).squeeze(-1) * values
        ).sum(dim=1) + fm_interactions(vectors)
        return summed_vectors, item_offset

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        item_vectors, item_offset = self.item_representation(item_ids)
        user_vectors = self.user_vectors(user_ids)
        return (
            self.bias
            + self.user_linear(user_ids).squeeze(-1)
            + item_offset
            + (user_vectors * item_vectors).sum(dim=1)
        )

    def l2_penalty(self) -> torch.Tensor:
        return 0.5 * sum(
            parameter.square().sum()
            for name, parameter in self.named_parameters()
            if name != "bias"
        )


class FactorizationMachine:

    def __init__(
        self,
        factors: int = 16,
        learning_rate: float = 0.003,
        regularization: float = 1e-4,
        epochs: int = 10,
        batch_size: int = 4096,
        seed: int = 42,
        use_metadata: bool = True,
        init_std: float = 0.01,
        score_batch_size: int = 2048,
        verbose: bool = False,
    ) -> None:
        if factors < 1 or epochs < 1 or batch_size < 1 or score_batch_size < 1:
            raise ValueError("factors, epochs and batch sizes must be positive")
        if learning_rate <= 0 or regularization < 0 or init_std <= 0:
            raise ValueError("learning_rate/init_std must be positive and regularization nonnegative")
        self.factors = int(factors)
        self.learning_rate = float(learning_rate)
        self.regularization = float(regularization)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.use_metadata = bool(use_metadata)
        self.init_std = float(init_std)
        self.score_batch_size = int(score_batch_size)
        self.verbose = bool(verbose)

    def _prepare_item_features(
        self, item_features: sparse.spmatrix | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_metadata and item_features is not None:
            if not sparse.issparse(item_features):
                raise TypeError("item_features must be a scipy sparse matrix")
            metadata = sparse.csr_matrix(item_features, dtype=np.float32, copy=True)
            if metadata.shape[0] != self.n_items_:
                raise ValueError("item_features needs one row for each ratings column")
            metadata.sum_duplicates()
            metadata.eliminate_zeros()
            metadata.sort_indices()
            if not np.isfinite(metadata.data).all():
                raise ValueError("item_features must contain only finite values")
        else:
            metadata = sparse.csr_matrix((self.n_items_, 0), dtype=np.float32)
        self.n_metadata_features_ = metadata.shape[1]
        self.n_features_ = self.n_users_ + self.n_items_ + self.n_metadata_features_
        total_item_features = self.n_items_ + self.n_metadata_features_
        width = int(np.diff(metadata.indptr).max(initial=0)) + 1
        indices = np.full((self.n_items_, width), total_item_features, dtype=np.int64)
        values = np.zeros((self.n_items_, width), dtype=np.float32)
        indices[:, 0] = np.arange(self.n_items_)
        values[:, 0] = 1.0
        for item in range(self.n_items_):
            start, end = metadata.indptr[item : item + 2]
            count = end - start
            indices[item, 1 : count + 1] = metadata.indices[start:end] + self.n_items_
            values[item, 1 : count + 1] = metadata.data[start:end]
        return torch.from_numpy(indices), torch.from_numpy(values)

    def fit(
        self,
        ratings: sparse.spmatrix,
        item_features: sparse.spmatrix | None = None,
        validation_callback: Callable[["FactorizationMachine"], float] | None = None,
    ) -> "FactorizationMachine":
        if not sparse.issparse(ratings):
            raise TypeError("ratings must be a scipy sparse matrix")
        data = sparse.csr_matrix(ratings, dtype=np.float32, copy=True)
        data.sum_duplicates()
        data.eliminate_zeros()
        data.sort_indices()
        if min(data.shape) < 1 or data.nnz < 1:
            raise ValueError("ratings must contain at least one observed interaction")
        if not np.isfinite(data.data).all() or np.any((data.data < 1) | (data.data > 5)):
            raise ValueError("observed ratings must be finite and in [1, 5]")
        self.n_users_, self.n_items_ = data.shape
        item_indices, item_values = self._prepare_item_features(item_features)
        initialization_generator = torch.Generator(device="cpu").manual_seed(self.seed)
        # Один порядок батчей для сравнения вариантов с разными признаками.
        batch_generator = torch.Generator(device="cpu").manual_seed((self.seed + 1) % (2**64))
        self._network = _FMNetwork(
            self.n_users_,
            self.n_items_ + self.n_metadata_features_,
            self.factors,
            item_indices,
            item_values,
            self.init_std,
            initialization_generator,
        )
        users = torch.from_numpy(np.repeat(np.arange(self.n_users_, dtype=np.int64), np.diff(data.indptr)))
        items = torch.from_numpy(data.indices.astype(np.int64, copy=True))
        labels = torch.from_numpy((data.data >= 4).astype(np.float32))
        optimizer = torch.optim.Adam(self._network.parameters(), lr=self.learning_rate)
        self.history_: list[dict[str, float | int]] = []
        self.best_epoch_ = self.epochs
        self.best_validation_score_: float | None = None
        self._cache_dirty = True
        best_state = None
        for epoch in range(1, self.epochs + 1):
            self._network.train()
            permutation = torch.randperm(data.nnz, generator=batch_generator)
            objective_sum = 0.0
            bce_sum = 0.0
            for start in range(0, data.nnz, self.batch_size):
                batch = permutation[start : start + self.batch_size]
                optimizer.zero_grad(set_to_none=True)
                logits = self._network(users[batch], items[batch])
                bce = F.binary_cross_entropy_with_logits(logits, labels[batch])
                loss = bce + self.regularization * self._network.l2_penalty()
                loss.backward()
                optimizer.step()
                objective_sum += float(loss.detach()) * len(batch)
                bce_sum += float(bce.detach()) * len(batch)
            self._cache_dirty = True
            record: dict[str, float | int] = {
                "epoch": epoch,
                "loss": objective_sum / data.nnz,
                "bce": bce_sum / data.nnz,
            }
            if validation_callback is not None:
                self._network.eval()
                metric = float(validation_callback(self))
                if not np.isfinite(metric):
                    raise ValueError("validation_callback must return a finite scalar metric")
                record["validation_metric"] = metric
                if self.best_validation_score_ is None or metric > self.best_validation_score_:
                    self.best_validation_score_ = metric
                    self.best_epoch_ = epoch
                    best_state = {
                        name: value.detach().clone()
                        for name, value in self._network.state_dict().items()
                    }
            self.history_.append(record)
            if self.verbose:
                print(f"FM epoch {epoch}/{self.epochs}: {record}")
        self.trained_epochs_ = self.epochs
        if best_state is not None:
            self._network.load_state_dict(best_state)
        self._network.eval()
        self._cache_dirty = True
        return self

    def _check_ids(self, ids: np.ndarray, size: int, label: str) -> np.ndarray:
        values = np.asarray(ids)
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"{label} must be a one-dimensional integer array")
        if np.any((values < 0) | (values >= size)):
            raise ValueError(f"{label} contains an out-of-range ID")
        return values.astype(np.int64, copy=False)

    def _require_fitted(self) -> None:
        if not hasattr(self, "_network"):
            raise RuntimeError("call fit before scoring")

    @torch.no_grad()
    def _refresh_item_cache(self) -> None:
        vectors = []
        offsets = []
        for start in range(0, self.n_items_, self.score_batch_size):
            item_ids = torch.arange(start, min(start + self.score_batch_size, self.n_items_))
            batch_vectors, batch_offsets = self._network.item_representation(item_ids)
            vectors.append(batch_vectors.numpy())
            offsets.append(batch_offsets.numpy())
        self._item_vectors_ = np.concatenate(vectors, axis=0)
        self._item_offsets_ = np.concatenate(offsets, axis=0)
        self._cache_dirty = False

    @torch.no_grad()
    def score_users(self, user_ids: np.ndarray) -> np.ndarray:
        self._require_fitted()
        users = self._check_ids(user_ids, self.n_users_, "user_ids")
        if self._cache_dirty:
            self._refresh_item_cache()
        tensor_users = torch.from_numpy(users)
        user_vectors = self._network.user_vectors(tensor_users).numpy()
        user_offset = self._network.user_linear(tensor_users).numpy()
        return (
            user_vectors @ self._item_vectors_.T
            + user_offset
            + self._item_offsets_[None, :]
            + float(self._network.bias)
        ).astype(np.float32, copy=False)

    @torch.no_grad()
    def score_pairs(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        self._require_fitted()
        users = self._check_ids(user_ids, self.n_users_, "user_ids")
        items = self._check_ids(item_ids, self.n_items_, "item_ids")
        if len(users) != len(items):
            raise ValueError("user_ids and item_ids must have equal lengths")
        outputs = []
        for start in range(0, len(users), self.score_batch_size):
            stop = start + self.score_batch_size
            outputs.append(
                self._network(
                    torch.from_numpy(users[start:stop]), torch.from_numpy(items[start:stop])
                ).numpy()
            )
        return np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float32)
