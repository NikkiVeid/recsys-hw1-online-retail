"""Item-based CF: cosine similarity, shrinkage, top neighbours per target."""
import numpy as np
from scipy import sparse
from .base import Recommender


class ItemKNN(Recommender):
    def __init__(self, k=50, shrink=20):
        self.k, self.shrink = k, shrink

    def fit(self, data):
        self.x = data.binary
        co = (self.x.T @ self.x).toarray().astype(np.float64)
        norm = np.sqrt(np.diag(co))
        sim = co / np.maximum(norm[:, None] * norm[None, :], 1e-12)
        sim *= co / (co + self.shrink + 1e-12)
        np.fill_diagonal(sim, 0)
        neighbours = np.argsort(-sim, axis=0, kind='stable')[:min(self.k, sim.shape[0]-1)]
        cols = np.broadcast_to(np.arange(sim.shape[1]), neighbours.shape)
        self.weights = sparse.csc_matrix(
            (sim[neighbours, cols].ravel(), (neighbours.ravel(), cols.ravel())),
            shape=sim.shape)
        return self

    def score(self, users):
        return (self.x[users] @ self.weights).toarray()
