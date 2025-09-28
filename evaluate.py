import os

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from sklearn.metrics import log_loss, roc_auc_score

from src.datamodule import CriteoIterable
from src.models.deepfm import DeepFM
from src.models.logistic import LogisticCTR
from src.models.dlrm import DLRMmini
from src.utils import load_config, set_seed, get_device


def build_model(name: str, d_dense: int, bucket_sizes: list[int], emb_dim: int):
    name = name.lower()
    if name == "logistic":
        return LogisticCTR(d_dense, len(bucket_sizes), emb_dim, bucket_sizes)
    if name == "deepfm":
        return DeepFM(dense_dim := d_dense, bucket_sizes=bucket_sizes, emb_dim=emb_dim)
    if name == "dlrm":
        return DLRMmini(d_dense, bucket_sizes, emb_dim)
    raise ValueError(f"Unknown model: {name}")


def evaluate(config_path: str = "configs/criteo_stream.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))
    device = get_device()

    dataset_name = cfg.get("dataset", "criteo/CriteoClickLogs")
    val_split = cfg.get("splits", {}).get("val", "validation")
    bucket_size_default = int(cfg.get("bucket_size_default", 1_000_003))
    batch_size = int(cfg.get("batch_size", 2048))
    emb_dim = int(cfg.get("emb_dim", 16))

    # Infer dimensions from a small sample of train
    sample_stream = load_dataset(dataset_name, split="train", streaming=True)
    ex = next(iter(sample_stream))
    d_dense = len(ex["dense_features"])  # type: ignore[index]
    n_cats = len(ex["cat_features"])  # type: ignore[index]
    dense_mean = [0.0] * d_dense
    dense_std = [1.0] * d_dense
    bucket_sizes = [bucket_size_default] * n_cats

    # Build dataset/loader
    val_stream = load_dataset(dataset_name, split=val_split, streaming=True)
    val_ds = CriteoIterable(val_stream, bucket_sizes, dense_mean, dense_std)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Load a fresh model for evaluation (random weights here; replace with checkpoint loading as needed)
    model_name = cfg.get("model", "deepfm")
    model = build_model(model_name, d_dense, bucket_sizes, emb_dim).to(device)

    ys, ps = [], []
    model.eval()
    with torch.no_grad():
        for dense, cats, y in val_loader:
            p = model(dense.to(device), cats.to(device)).cpu().numpy().ravel().tolist()
            ys += y.numpy().tolist()
            ps += p

    print("val_logloss=", log_loss(ys, ps))
    print("val_auc=", roc_auc_score(ys, ps))


if __name__ == "__main__":
    config = os.environ.get("CRITEO_CTR_CONFIG", "configs/criteo_stream.yaml")
    evaluate(config)

