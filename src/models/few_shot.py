import torch
from torch import nn
import torch

from rdkit import Chem

import sys, enum, typing

# 1) Backfill enum.StrEnum (Python 3.11 feature)
if not hasattr(enum, "StrEnum"):
    class StrEnum(str, enum.Enum):
        """Minimal backport of Python 3.11's StrEnum for runtime use."""
        pass
    # attach to the already-imported stdlib module so `from enum import StrEnum` works
    enum.StrEnum = StrEnum
    sys.modules["enum"].StrEnum = StrEnum

# 2) Backfill typing.Self if needed (used only for type hints at runtime)
if not hasattr(typing, "Self"):
    try:
        # if typing_extensions is available, prefer the proper one
        from typing_extensions import Self as _Self
        typing.Self = _Self
    except Exception:
        class _Self:  # minimal sentinel; fine for runtime annotations
            pass
        typing.Self = _Self
from chemprop.featurizers.molgraph import SimpleMoleculeMolGraphFeaturizer
from chemprop.data.collate import BatchMolGraph
from chemprop.data.molgraph import MolGraph
from chemprop.nn.message_passing import BondMessagePassing
from chemprop.nn.agg import MeanAggregation, NormAggregation



class DummyNN(nn.Module):
    def __init__(self, dim1: int, dim2: int, max_mol: int):
        """
        Args
        ----
        dim1, dim2 : hidden layer sizes
        max_mol    : length of concatenated feature vector
        """
        super().__init__()

        #self.norm = nn.LayerNorm(max_mol)        # ← NEW
        self.fc1  = nn.Linear(max_mol, dim1)
        self.fc2  = nn.Linear(dim1,    dim2)
        self.out  = nn.Linear(dim2, 1)

    def forward(self, x):
        # x: [batch, max_mol]
        #x = self.norm(x)                         # ← NEW
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)
    



class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):

        return self.net(x).squeeze(-1)



class ChempropModel(nn.Module):
    """
    Minimal chemprop-style GNN:
      - Featurizes SMILES -> BatchMolGraph
      - Message passing to node embeddings
      - Aggregation to graph embeddings
      - Optional concatenation of extra features X_d
      - MLP head for regression
    """
    def __init__(
        self,
        hidden_dim: int = 300,
        depth: int = 3,
        dropout: float = 0.0,
        use_norm_agg: bool = False,
        xtra_dim: int = 0,          # number of extra formulation features per sample
        head_hidden: int = 256,     # hidden size in the MLP head
    ):
        super().__init__()

        # 1) Build atom/bond featurizer (used each forward to convert SMILES -> MolGraph)
        self.featurizer = SimpleMoleculeMolGraphFeaturizer()
        d_v = self.featurizer.atom_fdim
        d_e = self.featurizer.bond_fdim
        #print("DEBUG dims:", d_v, d_e, hidden_dim)  # sanity check
        # 2) Message passing (learnable)
        self.mp = BondMessagePassing(
            d_v=d_v,
            d_e=d_e,
            d_vd=0,                # no per-atom extra descriptors
            depth=depth,
            d_h=hidden_dim,
            dropout=dropout,
            undirected=False,
            
        )

        # 3) Aggregation (no learnable parameters if Mean/Norm)
        self.agg = MeanAggregation() if not use_norm_agg else NormAggregation(norm=100.0)

        # 4) Prediction head (learnable)
        in_dim = hidden_dim + xtra_dim
        self.head = nn.Sequential(
            nn.Linear(in_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1)
        )

    @torch.no_grad()
    def _smiles_to_bmg(self, smiles: list[str]) -> BatchMolGraph:
        """SMILES -> list[MolGraph] -> BatchMolGraph."""
        mol_graphs = []
        for s in smiles:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                raise ValueError(f"Bad SMILES: {s}")
            mg: MolGraph = self.featurizer(mol)
            mol_graphs.append(mg)
        return BatchMolGraph(mol_graphs)

    def forward(self, smiles: list[str], X_d: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args
        ----
        smiles : list[str]  length N
        X_d    : (optional) tensor [N, xtra_dim] of extra formulation features

        Returns
        -------
        yhat : tensor [N, 1]
        """
        device = next(self.parameters()).device

        # Build graph batch
        bmg = self._smiles_to_bmg(smiles)
        bmg.to(device)  # move all graph tensors (V, E, edge_index, batch, ...) to the module's device

        # Message passing -> node embeddings H (shape [V_total, hidden_dim])
        H_v = self.mp(bmg)

        # Aggregate to per-graph embeddings Z (shape [N, hidden_dim])
        Z = self.agg(H_v, bmg.batch)

        # Concatenate extra features if provided
        if X_d is not None:
            if not torch.is_tensor(X_d):
                X_d = torch.as_tensor(X_d, dtype=torch.float32)
            X_d = X_d.to(device)
            if X_d.ndim != 2 or X_d.size(0) != Z.size(0):
                raise ValueError(f"X_d must be [N, xtra_dim], got {tuple(X_d.shape)} vs N={Z.size(0)}")
            Z = torch.cat([Z, X_d], dim=1)

        # MLP head -> prediction [N, 1]
        yhat = self.head(Z)
        return yhat

class MLPEncoder(nn.Module):
    """
    Feed-forward encoder that maps x -> embedding (N, emb_dim).

    Expects batches as dicts with key 'x' (tensor [N, in_dim]).
    """
    def __init__(
        self,
        in_dim: int,
        hidden: int = 256,
        emb_dim: int = 128,
        dropout: float = 0.10,
        normalize: bool = False,   # set True if you use cosine distance
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.proj = nn.Linear(hidden, emb_dim)
        self.normalize = normalize

    def forward(self, x: torch.Tensor | None = None, **x_dict) -> torch.Tensor:
        if x is None:
            # support dict-style batches used by your trainer: {"x": tensor,...}
            assert "x" in x_dict, "MLPEncoder expects key 'x' in the sample dict."
            x = x_dict["x"]
        h = self.trunk(x)
        z = self.proj(h)  # (N, emb_dim)
        if self.normalize:
            z = F.normalize(z, dim=1, eps=1e-8)
        return z
    

class ChempropEncoder(nn.Module):
    """
    Chemprop-style GNN encoder that returns graph embeddings (N, emb_dim).

    Expects dict batches with keys:
      - 'smiles': list[str]
      - optional 'X_d': tensor [N, xtra_dim]
    """
    def __init__(
        self,
        hidden_dim: int = 300,
        depth: int = 3,
        dropout: float = 0.0,
        use_norm_agg: bool = False,
        xtra_dim: int = 0,
        emb_dim: int = 256,         # final embedding size for ProtoNet
        normalize: bool = False,    # set True if you use cosine distance
    ):
        super().__init__()

        # 1) Featurizer
        self.featurizer = SimpleMoleculeMolGraphFeaturizer()
        d_v = self.featurizer.atom_fdim
        d_e = self.featurizer.bond_fdim

        # 2) Message passing
        self.mp = BondMessagePassing(
            d_v=d_v,
            d_e=d_e,
            d_vd=0,
            depth=depth,
            d_h=hidden_dim,
            dropout=dropout,
            undirected=False,
        )

        # 3) Aggregation
        self.agg = MeanAggregation() if not use_norm_agg else NormAggregation(norm=100.0)

        # 4) Projection to embedding
        in_dim = hidden_dim + xtra_dim
        self.proj = nn.Sequential(
            nn.Linear(in_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim),
        )
        self.normalize = normalize
        self.xtra_dim = xtra_dim

    @torch.no_grad()
    def _smiles_to_bmg(self, smiles: list[str]) -> "BatchMolGraph":
        mol_graphs = []
        for s in smiles:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                raise ValueError(f"Bad SMILES: {s}")
            mg: "MolGraph" = self.featurizer(mol)
            mol_graphs.append(mg)
        return BatchMolGraph(mol_graphs)

    def forward(self, smiles: list[str] | None = None, X_d: torch.Tensor | None = None, **x_dict) -> torch.Tensor:
        # Accept dict-style input
        if smiles is None:
            assert "smiles" in x_dict, "ChempropEncoder expects key 'smiles' in the sample dict."
            smiles = x_dict["smiles"]
            X_d = x_dict.get("X_d", None)

        device = next(self.parameters()).device

        # Build graph batch
        bmg = self._smiles_to_bmg(smiles)
        bmg.to(device)

        # Node embeddings -> graph embeddings
        H_v = self.mp(bmg)              # (V_total, hidden_dim)
        Z   = self.agg(H_v, bmg.batch)  # (N, hidden_dim)

        # Optional extra features
        if self.xtra_dim > 0:
            if X_d is None:
                raise ValueError(f"X_d (shape [N, {self.xtra_dim}]) is required but missing.")
            if not torch.is_tensor(X_d):
                X_d = torch.as_tensor(X_d, dtype=torch.float32)
            X_d = X_d.to(device)
            if X_d.ndim != 2 or X_d.size(0) != Z.size(0) or X_d.size(1) != self.xtra_dim:
                raise ValueError(f"X_d must be [N, {self.xtra_dim}] with same N as smiles; got {tuple(X_d.shape)}.")
            Z = torch.cat([Z, X_d], dim=1)

        # Projection to final embedding
        z = self.proj(Z)  # (N, emb_dim)
        if self.normalize:
            z = F.normalize(z, dim=1, eps=1e-8)
        return z
  