import itertools
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset

from src.datamodule import CriteoIterable
from src.models.deepfm import DeepFM
from src.models.logistic import LogisticCTR
from src.models.dlrm import DLRMmini
from src.utils import load_config, set_seed


def build_model(name: str, d_dense: int, bucket_sizes: list[int], emb_dim: int):
    name = name.lower()
    if name == "logistic":
        return LogisticCTR(d_dense, len(bucket_sizes), emb_dim, bucket_sizes)
    if name == "deepfm":
        return DeepFM(d_dense, bucket_sizes, emb_dim)
    if name == "dlrm":
        return DLRMmini(d_dense, bucket_sizes, emb_dim)
    raise ValueError(f"Unknown model: {name}")


def init_distributed():
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank) if torch.cuda.is_available() else None
    return local_rank, dist.get_world_size()


def is_main_process():
    return int(os.environ.get("RANK", 0)) == 0


def main(config_path: str = "configs/criteo_stream.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))
    local_rank, world_size = init_distributed()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    dataset_name = cfg.get("dataset", "criteo/CriteoClickLogs")
    train_split = cfg.get("splits", {}).get("train", "train")
    val_split = cfg.get("splits", {}).get("val", "validation")
    sample_size = int(cfg.get("sample_size", 100000))
    bucket_size_default = int(cfg.get("bucket_size_default", 1_000_003))
    batch_size = int(cfg.get("batch_size", 2048))
    epochs = int(cfg.get("epochs", 3))
    lr = float(cfg.get("learning_rate", 1e-3))
    emb_dim = int(cfg.get("emb_dim", 16))

    # rank 0 estimates stats, broadcasts to others via env for simplicity
    d_dense = n_cats = None
    if is_main_process():
        ds_stream = load_dataset(dataset_name, split=train_split, streaming=True)
        sample = list(itertools.islice(ds_stream, sample_size))
        d_dense = len(sample[0]["dense_features"])  # type: ignore[index]
        n_cats = len(sample[0]["cat_features"])  # type: ignore[index]
        dense_mean = [
            sum(ex["dense_features"][i] for ex in sample) / len(sample) for i in range(d_dense)
        ]
        dense_std = [1.0] * d_dense
    else:
        dense_mean = dense_std = []

    # broadcast basic sizes
    obj_list = [d_dense, n_cats]
    dist.broadcast_object_list(obj_list, src=0)
    d_dense, n_cats = obj_list

    if not dense_mean:
        dense_mean = [0.0] * d_dense
        dense_std = [1.0] * d_dense

    bucket_sizes = [bucket_size_default] * n_cats

    # create sharded streams
    train_stream = load_dataset(dataset_name, split=train_split, streaming=True).shard(num_shards=world_size, index=dist.get_rank())
    val_stream = load_dataset(dataset_name, split=val_split, streaming=True).shard(num_shards=world_size, index=dist.get_rank())

    train_ds = CriteoIterable(train_stream, bucket_sizes, dense_mean, dense_std)
    val_ds = CriteoIterable(val_stream, bucket_sizes, dense_mean, dense_std)

    model_name = cfg.get("model", "deepfm")
    model = build_model(model_name, d_dense, bucket_sizes, emb_dim).to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank] if torch.cuda.is_available() else None)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCELoss()

    def run_epoch(loader: DataLoader, max_steps: int = 2000):
        model.train()
        total_loss = torch.tensor(0.0, device=device)
        n = 0
        for step, (dense, cats, y) in enumerate(loader):
            if step >= max_steps:
                break
            dense, cats, y = dense.to(device), cats.to(device), y.to(device).unsqueeze(1)
            p = model(dense, cats)
            loss = bce(p, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.detach()
            n += 1
        # average across workers
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        n_tensor = torch.tensor(float(n), device=device)
        dist.all_reduce(n_tensor, op=dist.ReduceOp.SUM)
        return (total_loss / torch.clamp(n_tensor, min=1.0)).item()

    def evaluate(loader: DataLoader, max_steps: int = 200):
        # For simplicity, compute local loss and average; skip AUC in DDP eval
        model.eval()
        total_loss = torch.tensor(0.0, device=device)
        n = 0
        with torch.no_grad():
            for step, (dense, cats, y) in enumerate(loader):
                if step >= max_steps:
                    break
                p = model(dense.to(device), cats.to(device))
                y = y.to(device).unsqueeze(1)
                loss = bce(p, y)
                total_loss += loss
                n += 1
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        n_tensor = torch.tensor(float(n), device=device)
        dist.all_reduce(n_tensor, op=dist.ReduceOp.SUM)
        return (total_loss / torch.clamp(n_tensor, min=1.0)).item()

    train_loader = DataLoader(train_ds, batch_size=batch_size)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    for epoch in range(epochs):
        tr_loss = run_epoch(train_loader)
        va_loss = evaluate(val_loader)
        if is_main_process():
            print(f"[Epoch {epoch}] train_loss={tr_loss:.5f}  val_loss={va_loss:.5f}")


if __name__ == "__main__":
    config = os.environ.get("CRITEO_CTR_CONFIG", "configs/criteo_stream.yaml")
    main(config)

