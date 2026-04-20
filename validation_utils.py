from pathlib import Path

import torch
from torch.utils.data import Subset


def split_train_val_dataset(dataset, val_split, seed):
    if not 0 <= val_split < 1:
        raise ValueError("val_split must be in [0, 1).")

    dataset_len = len(dataset)
    val_len = int(dataset_len * val_split)
    if dataset_len > 1 and val_split > 0 and val_len == 0:
        val_len = 1
    train_len = dataset_len - val_len
    if train_len <= 0:
        raise ValueError("val_split leaves no training samples.")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(dataset_len, generator=generator).tolist()
    train_indices = indices[:train_len]
    val_indices = indices[train_len:]
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def should_run_validation(epoch_index, eval_interval):
    if eval_interval <= 0:
        raise ValueError("eval_interval must be positive.")
    return (epoch_index + 1) % eval_interval == 0


def maybe_save_best_checkpoint(model_state_dict, save_dir, val_loss, best_val_loss):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "best.pth"
    if val_loss < best_val_loss:
        torch.save(model_state_dict, best_path)
        return val_loss, best_path
    return best_val_loss, best_path
