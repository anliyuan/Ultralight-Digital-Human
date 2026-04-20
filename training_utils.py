from torch.utils.data import DataLoader


def build_train_dataloader(dataset, batch_size, num_workers=4):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
    )
