
import os
import math
import time
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import R2Score

try:
    import learn2learn as l2l
except Exception as e:
    raise RuntimeError("This trainer requires `learn2learn` (l2l). Please install it first.") from e

def _to_device_batch(x_batch, device):
    out = []
    for sample in x_batch:
        s = {}
        for k, v in sample.items():
            s[k] = v.to(device) if torch.is_tensor(v) else v  # <- keep lists as-is
        out.append(s)
    return out


def _unsqueeze_targets(y: torch.Tensor) -> torch.Tensor:
    return y.unsqueeze(1) if y.ndim == 1 else y


class FewShotTrainer:
    """
    Few-shot regression trainer with learn2learn (MAML/FoMAML/MetaSGD/ANIL/Reptile).

    - Episode-wise logging to TensorBoard
    - Early stopping on validation loss (patience) with checkpointing best state
    - Evaluation helper for val/test/holdout
    - Optional episode (task) target normalization using support stats

    Data interface:
        data.train_loader() -> iterable of (x_sup, y_sup, x_que, y_que)
        data.val_loader()   -> iterable of (x_sup, y_sup, x_que, y_que)
        data.test_loader() or data.holdout_loader() -> same (optional)
        data.tasks_per_batch: int

    Model:
        model(**x_dict) -> (B, 1) or (B,)

    Example:
        # base_model = MyModel(...)
        # trainer = FewShotTrainer(base_model, data, algorithm="FoMAML", episode_norm=True)
        # trainer.fit(epochs=20, eval_every=10)
        # print(trainer.evaluate_split("holdout"))
    """
    def __init__(
        self,
        model: nn.Module,
        data,
        loss: str = "MSE",
        algorithm: str = "FoMAML",
        adapt_lr: float = 1e-2,
        meta_lr: float = 3e-4,
        adapt_steps: int = 3,
        episode_norm: bool = False,
        head_only: bool = False,
        head_name: str = "head",
        patience: int = 20,
        log_dir: Optional[str] = None,
        ckpt_dir: str = "./checkpoints",
        max_grad_norm: float = 1.0,
        mixed_precision: bool = False,
        seed: int = 1337,
    ):
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.base_model = model
        self.algorithm_name = algorithm
        self.adapt_lr = adapt_lr
        self.meta_lr = meta_lr
        self.adapt_steps = adapt_steps
        self.episode_norm = episode_norm
        self.head_only = head_only
        self.head_name = head_name
        self.patience = patience
        self.ckpt_dir = ckpt_dir
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.max_grad_norm = max_grad_norm
        self.mixed_precision = mixed_precision

        # Data
        self.data = data
        self.train_loader = data.train_loader()
        self.val_loader = data.val_loader()
        self.test_loader = None
        if hasattr(data, "holdout_loader"):
            self.test_loader = data.holdout_loader()
        elif hasattr(data, "test_loader"):
            self.test_loader = data.test_loader()
        self.tasks_per_batch = getattr(data, "tasks_per_batch", 1)

        # Device & AMP
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_model.to(self.device)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

        # Loss
        if loss.strip().upper() == "MSE":
            self.lossfn = nn.MSELoss()
        elif loss.strip().upper() == "HUBER":
            self.lossfn = nn.HuberLoss()
        else:
            raise ValueError(f"Unsupported loss: {loss}")

        # l2l algorithm wrapper
        self.meta = self._make_meta_wrapper(self.base_model, self.algorithm_name, self.adapt_lr)

        # Optimizer over meta-parameters
        self.opt = optim.Adam(self.meta.parameters(), lr=self.meta_lr)

        # Logging
        if log_dir is None:
            log_dir = "./runs/fewshot"
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

        # Bookkeeping
        self.global_episode = 0
        self.best_val = math.inf
        self.best_state_dict: Optional[Dict[str, Any]] = None
        self.best_episode = -1

    def _make_meta_wrapper(self, model: nn.Module, algorithm: str, adapt_lr: float):
        name = algorithm.lower()
        if name in ["maml", "fomaml"]:
            first_order = (name == "fomaml")
            return l2l.algorithms.MAML(model, lr=adapt_lr, first_order=first_order)
        elif name == "metasgd":
            return l2l.algorithms.MetaSGD(model, lr=adapt_lr, first_order=True)
        elif name == "anil":
            # Implement with FoMAML but freeze non-head params during inner-loop.
            return l2l.algorithms.MAML(model, lr=adapt_lr, first_order=True)
        elif name == "reptile":
            return l2l.algorithms.Reptile(model, lr=adapt_lr)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    @staticmethod
    def _episode_norm_targets(
        y_sup: torch.Tensor, y_que: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
        mu = y_sup.mean().item()
        std = y_sup.std(unbiased=False).item()
        if std < 1e-8:
            std = 1.0
        y_sup_n = (y_sup - mu) / std
        y_que_n = (y_que - mu) / std
        return y_sup_n, y_que_n, mu, std

    @staticmethod
    def _denorm(y_pred_n: torch.Tensor, mu: float, std: float) -> torch.Tensor:
        return y_pred_n * std + mu

    def _adapt_and_predict_single_task(
        self,
        learner,
        x_sup_t: Dict[str, torch.Tensor],
        y_sup_t: torch.Tensor,
        x_que_t: Dict[str, torch.Tensor],
        y_que_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # ANIL / head-only adaptation
        if self.algorithm_name.lower() == "anil" or self.head_only:
            for name, p in learner.named_parameters():
                p.requires_grad = (self.head_name in name)

        y_sup_t = _unsqueeze_targets(y_sup_t)
        y_que_t = _unsqueeze_targets(y_que_t)

        if self.episode_norm:
            y_sup_n, y_que_n, mu, std = self._episode_norm_targets(y_sup_t, y_que_t)
        else:
            y_sup_n, y_que_n = y_sup_t, y_que_t
            mu, std = 0.0, 1.0

        # Inner-loop
        for _ in range(self.adapt_steps):
            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                sup_pred = learner(**x_sup_t)
                if sup_pred.ndim == 1:
                    sup_pred = _unsqueeze_targets(sup_pred)
                loss = self.lossfn(sup_pred, y_sup_n)
            learner.adapt(loss)

        # Query forward
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            que_pred_n = learner(**x_que_t)
            if que_pred_n.ndim == 1:
                que_pred_n = _unsqueeze_targets(que_pred_n)
            que_loss = self.lossfn(que_pred_n, y_que_n)

        que_pred = self._denorm(que_pred_n, mu, std) if self.episode_norm else que_pred_n
        return que_pred, que_loss

    def _evaluate_loader(
        self,
        loader,
        max_batches: Optional[int] = None,
        return_arrays: bool = False,
    ) -> Dict[str, float] | Tuple[Dict[str, float], torch.Tensor, torch.Tensor]:

        self.meta.eval()
        total_loss = 0.0
        n_batches = 0
        preds, trues = [], []

        for batch_idx, (x_sup, y_sup, x_que, y_que) in enumerate(loader):
            if (max_batches is not None) and (batch_idx >= max_batches):
                break

            x_sup = _to_device_batch(x_sup, self.device)
            x_que = _to_device_batch(x_que, self.device)
            y_sup = y_sup.to(self.device)
            y_que = y_que.to(self.device)

            batch_loss = 0.0
            for t in range(min(self.tasks_per_batch, len(x_sup))):
                learner = self.meta.clone()
                learner.train()  # grads ON for support adaptation

                # ---- Support adapt (grads) + query predict ----
                que_pred, que_loss = self._adapt_and_predict_single_task(
                    learner, x_sup[t], y_sup[t], x_que[t], y_que[t]
                )

                # ---- Accumulate metrics (no grads needed) ----
                with torch.no_grad():
                    batch_loss += float(que_loss.item())
                    preds.append(que_pred.detach().squeeze(1).cpu())
                    trues.append(_unsqueeze_targets(y_que[t]).detach().squeeze(1).cpu())

            total_loss += batch_loss / self.tasks_per_batch
            n_batches += 1

        y_pred = torch.cat(preds, dim=0)
        y_true = torch.cat(trues, dim=0)

        mse = torch.mean((y_pred - y_true) ** 2).item()
        rmse = math.sqrt(mse)
        r2 = R2Score().to("cpu")(y_pred, y_true).item()
        avg_loss = total_loss / max(n_batches, 1)

        metrics = {"loss": avg_loss, "rmse": rmse, "r2": r2}
        if return_arrays:
            return metrics, y_true, y_pred
        return metrics

    def _log_eval(self, split: str, metrics: Dict[str, float], step: int,
              y_true: Optional[torch.Tensor] = None,
              y_pred: Optional[torch.Tensor] = None,
              make_scatter: bool = True):
        # Scalars
        self.writer.add_scalar(f"{split}/loss", metrics["loss"], step)
        self.writer.add_scalar(f"{split}/rmse", metrics["rmse"], step)
        self.writer.add_scalar(f"{split}/r2",   metrics["r2"],   step)

        # Optional scatter figure
        if make_scatter and (y_true is not None) and (y_pred is not None):
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.scatter(y_true.numpy(), y_pred.numpy(), s=8, alpha=0.6)
            ax.set_xlabel("y_true"); ax.set_ylabel("y_pred")
            ax.set_title(f"{split}: y_true vs y_pred")
            # y=x reference
            lo = float(min(y_true.min().item(), y_pred.min().item()))
            hi = float(max(y_true.max().item(), y_pred.max().item()))
            ax.plot([lo, hi], [lo, hi], linestyle="--")
            self.writer.add_figure(f"{split}/scatter_ytrue_ypred", fig, step)
            plt.close(fig)
    def fit(self, epochs: int = 50, eval_every: int = 10, save_tag: str = "best") -> Dict[str, float]:
        self.meta.train()
        start_time = time.time()
        no_improve = 0
        best_val_metrics = {"loss": float("inf"), "rmse": float("inf"), "r2": -float("inf")}

        for epoch in range(1, epochs + 1):
            for episode, (x_sup, y_sup, x_que, y_que) in enumerate(self.train_loader):
                self.meta.train()

                x_sup = _to_device_batch(x_sup, self.device)
                x_que = _to_device_batch(x_que, self.device)
                y_sup = y_sup.to(self.device)
                y_que = y_que.to(self.device)

                meta_loss = 0.0
                for t in range(min(self.tasks_per_batch, len(x_sup))):
                    learner = self.meta.clone()
                    que_pred, que_loss = self._adapt_and_predict_single_task(
                        learner, x_sup[t], y_sup[t], x_que[t], y_que[t]
                    )
                    meta_loss += que_loss

                meta_loss = meta_loss / self.tasks_per_batch

                # Log training loss
                self.writer.add_scalar("train/episode_loss", meta_loss.item(), self.global_episode)

                # Outer update
                self.opt.zero_grad(set_to_none=True)
                if self.mixed_precision:
                    self.scaler.scale(meta_loss).backward()
                    clip_grad_norm_(self.meta.parameters(), self.max_grad_norm)
                    self.scaler.step(self.opt)
                    self.scaler.update()
                else:
                    meta_loss.backward()
                    clip_grad_norm_(self.meta.parameters(), self.max_grad_norm)
                    self.opt.step()

                # Validation
                if (self.global_episode % eval_every) == 0:
                    val_metrics = self._evaluate_loader(self.val_loader)
                    self.writer.add_scalar("val/loss", val_metrics["loss"], self.global_episode)
                    self.writer.add_scalar("val/rmse", val_metrics["rmse"], self.global_episode)
                    self.writer.add_scalar("val/r2", val_metrics["r2"], self.global_episode)

                    if val_metrics["loss"] + 1e-8 < self.best_val:
                        self.best_val = val_metrics["loss"]
                        self.best_state_dict = {k: v.cpu().clone() for k, v in self.meta.state_dict().items()}
                        self.best_episode = self.global_episode
                        no_improve = 0
                        # Save checkpoint
                        ckpt = os.path.join(self.ckpt_dir, f"{save_tag}_episode{self.global_episode}.pt")
                        torch.save(self.best_state_dict, ckpt)
                    else:
                        no_improve += 1

                    elapsed = time.time() - start_time
                    self.writer.add_scalar("time/elapsed_sec", elapsed, self.global_episode)

                    if val_metrics["loss"] <= best_val_metrics["loss"]:
                        best_val_metrics = val_metrics

                    if no_improve >= self.patience:
                        print(f"⏹️ Early stop at episode {self.global_episode} "
                              f"(best val loss {self.best_val:.4f} at episode {self.best_episode}).")
                        return best_val_metrics

                self.global_episode += 1

        print(f"Training done. Best val loss {self.best_val:.4f} at episode {self.best_episode}.")
        return best_val_metrics

    def evaluate_split(self, split: str = "val", log: bool = True, step: Optional[int] = None):
        if split == "val":
            loader = self.val_loader
        elif split in ["test", "holdout"]:
            if self.test_loader is None:
                raise ValueError("No test/holdout loader available.")
            loader = self.test_loader
        else:
            raise ValueError(f"Unknown split: {split}")

        if self.best_state_dict is not None:
            self.meta.load_state_dict(self.best_state_dict)

        out = self._evaluate_loader(loader, return_arrays=True)
        metrics, y_true, y_pred = out

        if log:
            if step is None:
                step = self.global_episode
            self._log_eval(split, metrics, step, y_true, y_pred)

        return metrics
