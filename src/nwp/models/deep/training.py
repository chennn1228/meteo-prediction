"""Reusable deep quantile training with a disjoint early-stop block."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Sequence

import numpy as np


def pinball_loss(prediction, target, quantiles: Sequence[float]):
    import torch
    levels = torch.tensor(quantiles, dtype=prediction.dtype,
                          device=prediction.device).view(1, -1)
    residual = target.view(-1, 1) - prediction
    base = torch.maximum(levels * residual, (levels - 1) * residual).mean()
    crossing = torch.relu(prediction[:, :-1] - prediction[:, 1:]).mean()
    return base + 0.1 * crossing


def fit_quantile_model(
        model, fit_x: np.ndarray, fit_y: np.ndarray,
        early_stop_x: np.ndarray, early_stop_y: np.ndarray, *,
        quantiles: Sequence[float], learning_rate: float, max_epochs: int,
        batch_size: int, patience: int = 4, seed: int = 0,
        device: str = "cpu") -> tuple[Any, dict[str, Any]]:
    """Train on fit only and select duration only on the early-stop block."""
    import torch
    if min(len(fit_x), len(early_stop_x)) <= 0:
        raise ValueError("deep fit and early-stop blocks must be nonempty")
    if fit_x.ndim != 3 or early_stop_x.ndim != 3:
        raise ValueError("deep features must have shape [sample, sequence, feature]")
    torch.manual_seed(seed)
    np.random.seed(seed)
    destination = torch.device(device)
    model.to(destination)
    generator = torch.Generator().manual_seed(seed)

    def loader(x, y, shuffle):
        dataset = torch.utils.data.TensorDataset(
            torch.as_tensor(x, dtype=torch.float32),
            torch.as_tensor(y, dtype=torch.float32))
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            generator=generator if shuffle else None)

    fit_loader = loader(fit_x, fit_y, True)
    early_loader = loader(early_stop_x, early_stop_y, False)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_loss, best_state, best_epoch, wait = np.inf, None, 0, 0
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_total, train_rows = 0.0, 0
        for x_batch, y_batch in fit_loader:
            x_batch, y_batch = x_batch.to(destination), y_batch.to(destination)
            output = model(x_batch)
            prediction, penalty = output if isinstance(output, tuple) else (
                output, output.new_zeros(()))
            loss = pinball_loss(prediction, y_batch, quantiles) + 0.1 * penalty
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_total += float(loss.detach()) * len(y_batch)
            train_rows += len(y_batch)
        model.eval()
        early_total, early_rows = 0.0, 0
        with torch.no_grad():
            for x_batch, y_batch in early_loader:
                x_batch, y_batch = x_batch.to(destination), y_batch.to(destination)
                output = model(x_batch)
                prediction, penalty = output if isinstance(output, tuple) else (
                    output, output.new_zeros(()))
                loss = pinball_loss(prediction, y_batch, quantiles) + 0.1 * penalty
                early_total += float(loss.detach()) * len(y_batch)
                early_rows += len(y_batch)
        train_loss = train_total / train_rows
        early_loss = early_total / early_rows
        history.append({"epoch": epoch, "fit_loss": train_loss,
                        "early_stop_loss": early_loss})
        if early_loss < best_loss - 1e-4:
            best_loss, best_epoch = early_loss, epoch
            best_state, wait = deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is None:
        raise RuntimeError("deep training produced no selectable epoch")
    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch, "best_early_stop_loss": float(best_loss),
        "epochs_executed": len(history), "fit_rows": len(fit_x),
        "early_stop_rows": len(early_stop_x), "history": history,
    }


def predict_quantiles(model, features: np.ndarray, *,
                      device: str = "cpu") -> np.ndarray:
    import torch
    model.eval()
    with torch.no_grad():
        output = model(torch.as_tensor(
            features, dtype=torch.float32, device=torch.device(device)))
        prediction = output[0] if isinstance(output, tuple) else output
    return prediction.detach().cpu().numpy()
