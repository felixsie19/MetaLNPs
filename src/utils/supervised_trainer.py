 
import os
import math
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
from torch import nn as tnn
from torch.nn.utils import clip_grad_norm_
from torchmetrics import R2Score
from torch.utils.tensorboard import SummaryWriter
import copy


def _to_device_dict(d: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move only tensors to device; leave strings/lists (e.g., SMILES) alone."""
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in d.items()}


def _concat_dicts(list_of_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Concatenate values across a list of dicts.
    - Tensors            -> torch.cat(dim=0)
    - List[str] (SMILES) -> flatten lists
    - Other types        -> keep list of values
    """
    buckets: Dict[str, List[Any]] = {}
    for d in list_of_dicts:
        for k, v in d.items():
            buckets.setdefault(k, []).append(v)

    out: Dict[str, Any] = {}
    for k, vs in buckets.items():
        v0 = vs[0]
        if torch.is_tensor(v0):
            out[k] = torch.cat(vs, dim=0)
        elif isinstance(v0, list) and (len(v0) == 0 or isinstance(v0[0], str)):
            flat = []
            for lst in vs:
                flat.extend(lst)
            out[k] = flat
        else:
            out[k] = vs
    return out


def _flatten_episode_to_batch(
    x_sup: List[Dict[str, Any]],
    y_sup: torch.Tensor,
    x_que: List[Dict[str, Any]],
    y_que: torch.Tensor,
) -> Tuple[Dict[str, Any], torch.Tensor]:
    """
    Works for both:
      • FP mode:    x dicts contain {"x": Tensor}
      • Chemprop mode: x dicts contain {"smiles": list[str], "X_d": Tensor}
    """
    x_all = x_sup + x_que
    y_all = torch.cat([y_sup, y_que], dim=0)
    X = _concat_dicts(x_all)
    y = y_all.view(-1).unsqueeze(1)
    return X, y


class SupervisedTrainer:
    """
    Supervised trainer that:
    - Flattens episodic loaders into (X, y).
    - Early-stops on validation.
    - Can evaluate with optional K-shot fine-tuning.
    """

    def __init__(
        self,
        model: tnn.Module,
        data,
        loss: str = "MSE",
        lr: float = 3e-4,
        weight_decay: float = 0.0,
        max_grad_norm: float = 1.0,
        patience: int = 20,
        log_dir: Optional[str] = None,
        ckpt_dir: str = "./checkpoints_sup",
        mixed_precision: bool = False,
        seed: int = 1337,
    ):
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.model = model
        self.data = data
        self.train_loader = data.train_loader()
        self.val_loader = data.val_loader()
        self.test_loader = getattr(data, "holdout_loader", getattr(data, "test_loader", None))
        if callable(self.test_loader):
            self.test_loader = self.test_loader()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        if loss.upper() == "MSE":
            self.lossfn = tnn.MSELoss()
        elif loss.upper() == "HUBER":
            self.lossfn = tnn.HuberLoss()
        else:
            raise ValueError(f"Unsupported loss: {loss}")

        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.max_grad_norm = max_grad_norm
        self.patience = patience
        self.mixed_precision = mixed_precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

        if log_dir is None:
            log_dir = "./runs/supervised_baseline"
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        os.makedirs(ckpt_dir, exist_ok=True)
        self.ckpt_dir = ckpt_dir

        self.best_val = float("inf")
        self.best_state = None
        self.global_step = 0

    def _forward_loss(self, X: Dict[str, Any], y: torch.Tensor) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            pred = self.model(**X)   # supports {"x": ...} or {"smiles": ..., "X_d": ...}
            if pred.ndim == 1:
                pred = pred.unsqueeze(1)
            loss = self.lossfn(pred, y)
        return loss

    @torch.no_grad()
    def _eval_flat_loader(self, loader) -> Dict[str, float]:
        self.model.eval()
        r2_metric = R2Score().to("cpu")
        preds, trues = [], []
        total_loss = 0.0
        n_batches = 0

        for x_sup, y_sup, x_que, y_que in loader:
            X_cpu, y_cpu = _flatten_episode_to_batch(x_sup, y_sup, x_que, y_que)
            X = _to_device_dict(X_cpu, self.device)
            y = y_cpu.to(self.device)

            pred = self.model(**X)
            if pred.ndim == 1:
                pred = pred.unsqueeze(1)
            loss = self.lossfn(pred, y).item()
            total_loss += loss
            n_batches += 1

            preds.append(pred.detach().cpu().squeeze(1))
            trues.append(y.detach().cpu().squeeze(1))

        if n_batches == 0:
            return {"loss": float("nan"), "rmse": float("nan"), "r2": float("nan")}

        y_pred = torch.cat(preds, dim=0)
        y_true = torch.cat(trues, dim=0)
        mse = torch.mean((y_pred - y_true) ** 2).item()
        rmse = math.sqrt(mse)
        r2 = r2_metric(y_pred, y_true).item()
        return {"loss": total_loss / n_batches, "rmse": rmse, "r2": r2}

    def fit(self, epochs: int = 50, eval_every: int = 50, save_tag: str = "sup") -> Dict[str, float]:
        self.model.train()
        no_improve = 0
        best_val_metrics = {"loss": float("inf"), "rmse": float("inf"), "r2": -float("inf")}

        for epoch in range(1, epochs + 1):
            for x_sup, y_sup, x_que, y_que in self.train_loader:
                self.model.train()
                X_cpu, y_cpu = _flatten_episode_to_batch(x_sup, y_sup, x_que, y_que)
                X = _to_device_dict(X_cpu, self.device)
                y = y_cpu.to(self.device)

                loss = self._forward_loss(X, y)

                self.opt.zero_grad(set_to_none=True)
                if self.mixed_precision:
                    self.scaler.scale(loss).backward()
                    clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.opt)
                    self.scaler.update()
                else:
                    loss.backward()
                    clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.opt.step()

                self.writer.add_scalar("train/loss", loss.item(), self.global_step)
                self.global_step += 1

                if (self.global_step % eval_every) == 0:
                    val_metrics = self._eval_flat_loader(self.val_loader)
                    self.writer.add_scalar("val/loss", val_metrics["loss"], self.global_step)
                    self.writer.add_scalar("val/rmse", val_metrics["rmse"], self.global_step)
                    self.writer.add_scalar("val/r2",   val_metrics["r2"],   self.global_step)

                    if val_metrics["loss"] + 1e-8 < self.best_val:
                        self.best_val = val_metrics["loss"]
                        self.best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                        no_improve = 0
                        ckpt = os.path.join(self.ckpt_dir, f"{save_tag}_step{self.global_step}.pt")
                        torch.save(self.best_state, ckpt)
                    else:
                        no_improve += 1

                    if val_metrics["loss"] <= best_val_metrics["loss"]:
                        best_val_metrics = val_metrics

                    if no_improve >= self.patience:
                        print(f"⏹️ Early stop at step {self.global_step} (best val loss {self.best_val:.4f}).")
                        return best_val_metrics

        print(f"Training done. Best val loss {self.best_val:.4f}.")
        return best_val_metrics

    def evaluate_meta_loader(
        self,
        loader,
        k_shot_finetune: bool = False,
        finetune_steps: int = 5,
        finetune_lr: float = 5e-3,
        head_only: bool = True,
        head_name: str = "head",
        log_prefix: str = "holdout",
        log_scatter: bool = True,
    ) -> Dict[str, float]:
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        self.model.eval()

        preds, trues = [], []

        for x_sup, y_sup, x_que, y_que in loader:
            B = len(x_sup)
            for t in range(B):
                work_model = self.model
                if k_shot_finetune:
                    work_model = copy.deepcopy(self.model).to(self.device)
                    work_model.load_state_dict(self.model.state_dict())
                    if head_only:
                        for name, p in work_model.named_parameters():
                            p.requires_grad = (head_name in name)
                    opt_ft = torch.optim.Adam(filter(lambda p: p.requires_grad, work_model.parameters()), lr=finetune_lr)

                    xs = _to_device_dict(x_sup[t], self.device)
                    ys = y_sup[t].to(self.device).view(-1, 1)

                    for _ in range(finetune_steps):
                        work_model.train()
                        pred = work_model(**xs)
                        if pred.ndim == 1: pred = pred.unsqueeze(1)
                        loss = self.lossfn(pred, ys)
                        opt_ft.zero_grad(set_to_none=True)
                        loss.backward()
                        clip_grad_norm_(filter(lambda p: p.requires_grad, work_model.parameters()), 1.0)
                        opt_ft.step()

                xq = _to_device_dict(x_que[t], self.device)
                yq = y_que[t].to(self.device).view(-1, 1)
                work_model.eval()
                pred_q = work_model(**xq)
                if pred_q.ndim == 1: pred_q = pred_q.unsqueeze(1)
                preds.append(pred_q.detach().cpu().squeeze(1))
                trues.append(yq.detach().cpu().squeeze(1))

        y_pred = torch.cat(preds, dim=0)
        y_true = torch.cat(trues, dim=0)
        mse = torch.mean((y_pred - y_true) ** 2).item()
        rmse = math.sqrt(mse)
        r2 = R2Score().to("cpu")(y_pred, y_true).item()
        metrics = {"rmse": rmse, "r2": r2}

        step = self.global_step
        self.writer.add_scalar(f"{log_prefix}/rmse", rmse, step)
        self.writer.add_scalar(f"{log_prefix}/r2", r2, step)

        if log_scatter:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.scatter(y_true.numpy(), y_pred.numpy(), s=8, alpha=0.6)
            ax.set_xlabel("y_true"); ax.set_ylabel("y_pred")
            ax.set_title(f"{log_prefix}: y_true vs y_pred")
            lo = float(min(y_true.min().item(), y_pred.min().item()))
            hi = float(max(y_true.max().item(), y_pred.max().item()))
            ax.plot([lo, hi], [lo, hi], linestyle="--")
            self.writer.add_figure(f"{log_prefix}/scatter_ytrue_ypred", fig, step)
            plt.close(fig)

        return metrics
