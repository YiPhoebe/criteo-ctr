# train.py (핵심 부분만, 그대로 교체해도 됨)
import os, json, csv, itertools, argparse, time, datetime
from tqdm import tqdm
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss, roc_auc_score
from datasets import load_dataset
from datasets.exceptions import DataFilesNotFoundError
import numpy as np
import numpy as np
from datamodule import CriteoIterable
from models.deepfm import DeepFM
from models.logistic import LogisticCTR
from models.dlrm import DLRMmini
from torch.utils.tensorboard import SummaryWriter

# NEW: YAML 로딩
def load_yaml(path):
    import yaml
    if not path or not os.path.exists(path):
        return {}  # 파일 없으면 빈 설정 반환 (예외 X)
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=str,
        default=os.environ.get("CRITEO_CTR_CONFIG", "configs/criteo_stream.yaml"),
        help="YAML config path",
    )
    p.add_argument("--epochs", type=int, default=50)   # 🔥 여기 숫자를 바꾸면 기본 에폭이 바뀜
    p.add_argument("--batch-size", type=int, default=1024)  # 🔥 batch-size 기본값
    p.add_argument("--model", type=str, default="deepfm", choices=["deepfm","logistic","dlrm"])
    p.add_argument("--emb-dim", type=int, default=32)  # 🔥 embedding dimension 기본값
    p.add_argument("--lr", type=float, default=5e-4)   # 🔥 learning rate 기본값
    p.add_argument("--max-train-steps", type=int, default=2000)
    p.add_argument("--max-val-steps", type=int, default=200)
    p.add_argument("--logdir", type=str, default="../runs")
    p.add_argument("--run-name", type=str, default="")
    p.add_argument("--save-metric", type=str, default="auc", choices=["logloss", "auc"], help="Criterion to store best checkpoint")  # 🔥 저장 기준을 AUC로
    p.add_argument("--early-stop-metric", type=str, default="auc", choices=["logloss", "auc"], help="Metric to monitor for early stopping")  # 🔥 조기 종료 기준도 AUC로
    p.add_argument("--patience", type=int, default=5, help="Early stopping patience (0 to disable)")  # 🔥 5 epoch 개선 없으면 종료
    p.add_argument("--shuffle-buffer", type=int, default=200_000, help="Streaming shuffle buffer size for training")  # 🔥 shuffle buffer 키움
    p.add_argument("--dropout", type=float, default=0.15, help="Dropout for DeepFM MLP")  # 🔥 dropout 기본값
    p.add_argument("--weight-decay", type=float, default=1e-5, help="Optimizer weight decay")
    p.add_argument("--lr-scheduler", type=str, default="cosine", choices=["none","cosine","step"])
    p.add_argument("--lr-warmup-epochs", type=int, default=1)
    p.add_argument("--step-size", type=int, default=15)   # step scheduler 주기
    p.add_argument("--gamma", type=float, default=0.5)    # step scheduler 감쇠
    return p.parse_args()

def build_model(name, d_dense, bucket_sizes, emb_dim, dropout: float = 0.0):
    if name == "deepfm":
        return DeepFM(d_dense, bucket_sizes, emb_dim=emb_dim, dropout=dropout)
    if name == "logistic":
        return LogisticCTR(d_dense, n_cats=len(bucket_sizes), emb_dim=emb_dim, bucket_sizes=bucket_sizes)
    if name == "dlrm":
        return DLRMmini(d_dense, bucket_sizes, emb_dim=emb_dim)
    raise ValueError(name)

def merge_cfg(args):
    # YAML → args로 병합 (CLI가 우선)
    cfg = {}
    if args.config:
        yaml_cfg = load_yaml(args.config) or {}
        cfg.update(yaml_cfg)
    # 호환: learning_rate -> lr 매핑
    if "learning_rate" in cfg and "lr" not in cfg:
        cfg["lr"] = cfg["learning_rate"]
    # CLI에서 지정된 값은 YAML보다 우선
    # (argparse 기본값이 아닌, 실제로 지정한 값만 우선하게 하려면 별도 처리 가능)
    cfg.setdefault("epochs", args.epochs)
    cfg.setdefault("batch_size", args.batch_size)
    cfg.setdefault("model", args.model)
    cfg.setdefault("emb_dim", args.emb_dim)
    cfg.setdefault("lr", args.lr)
    cfg.setdefault("max_train_steps", args.max_train_steps)
    cfg.setdefault("max_val_steps", args.max_val_steps)
    cfg.setdefault("logdir", args.logdir)
    cfg.setdefault("save_metric", args.save_metric)
    cfg.setdefault("early_stop_metric", args.early_stop_metric)
    cfg.setdefault("patience", args.patience)
    cfg.setdefault("shuffle_buffer", args.shuffle_buffer)
    cfg.setdefault("dropout", args.dropout)
    cfg.setdefault("weight_decay", args.weight_decay)
    # 데이터셋/스플릿/샘플 사이즈/버킷/워커/시드 기본값
    cfg.setdefault("dataset", "criteo/CriteoClickLogs")
    cfg.setdefault("splits", {"train": "train", "val": "validation"})
    cfg.setdefault("sample_size", 100000)
    cfg.setdefault("bucket_size_default", 1_000_003)
    cfg.setdefault("num_workers", 0)
    cfg.setdefault("seed", 42)
    # run_name 비어있으면 자동 생성
    if args.run_name:
        cfg["run_name"] = args.run_name
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg["run_name"] = f"{cfg.get('model','run')}_{ts}"
    return cfg

def main():
    args = get_args()
    cfg = merge_cfg(args)

    os.makedirs(cfg["logdir"], exist_ok=True)
    run_dir = os.path.join(cfg["logdir"], cfg["run_name"])
    os.makedirs(run_dir, exist_ok=True)

    # logger & writers (이미 로그 저장 기능 존재 + 유지/보강)
    log_path = os.path.join(run_dir, "train.log")
    csv_path = os.path.join(run_dir, "metrics.csv")
    tb = SummaryWriter(run_dir)

    def log(msg: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    # 시드 고정
    try:
        import random, numpy as np
        random.seed(cfg["seed"]) ; np.random.seed(cfg["seed"]) ; torch.manual_seed(cfg["seed"]) ; torch.cuda.manual_seed_all(cfg["seed"])  # noqa: E702
    except Exception:
        pass

    # 데이터 (스트리밍) — YAML 구성 적용: csv | parquet | hf
    ds_cfg = cfg.get("dataset", {}) if isinstance(cfg.get("dataset"), dict) else {}

    def build_streams(cfg):
        ds_cfg = cfg.get("dataset", {}) if isinstance(cfg.get("dataset"), dict) else {}
        ds_type = str(ds_cfg.get("type", cfg.get("dataset_type", "hf"))).lower()
        try:
            if ds_type == "hf":
                ds_name = ds_cfg.get("name", "criteo/criteo-attribution-dataset")
                tr_split = ds_cfg.get("train_split", "train")
                va_split = ds_cfg.get("val_split", "validation")
                train_stream = load_dataset(ds_name, split=tr_split, streaming=True)
                try:
                    val_stream = load_dataset(ds_name, split=va_split, streaming=True)
                    return train_stream, val_stream
                except Exception:
                    # Try test split; if unavailable, derive val via take/skip from train
                    try:
                        val_stream = load_dataset(ds_name, split="test", streaming=True)
                        return train_stream, val_stream
                    except Exception:
                        val_take = int(
                            ds_cfg.get(
                                "val_take",
                                cfg.get("val_take", int(cfg.get("batch_size", 2048)) * int(cfg.get("max_val_steps", 200))),
                            )
                        )
                        # Create small validation stream from the head of train; use the tail for training
                        val_head = train_stream.take(val_take)
                        train_tail = train_stream.skip(val_take)
                        return train_tail, val_head

            elif ds_type == "csv":
                data_files = ds_cfg["data_files"]
                sep = ds_cfg.get("sep", "\t")
                column_names = [
                    "label",
                    *[f"I{i}" for i in range(1, 14)],
                    *[f"C{i}" for i in range(1, 27)],
                ]
                tr_raw = load_dataset(
                    "csv",
                    data_files=data_files["train"],
                    sep=sep,
                    column_names=column_names,
                    streaming=True,
                )["train"]
                va_raw = load_dataset(
                    "csv",
                    data_files=data_files["validation"],
                    sep=sep,
                    column_names=column_names,
                    streaming=True,
                )["train"]

                def to_criteo_schema(stream):
                    for ex in stream:
                        try:
                            label = int(ex.get("label", 0) or 0)
                        except Exception:
                            label = 0
                        dense = []
                        for i in range(1, 14):
                            v = ex.get(f"I{i}")
                            try:
                                dense.append(float(v) if v not in (None, "") else 0.0)
                            except Exception:
                                dense.append(0.0)
                        cats = []
                        for i in range(1, 27):
                            v = ex.get(f"C{i}")
                            cats.append(v if v not in (None, "") else "MISS")
                        yield {"label": label, "dense_features": dense, "cat_features": cats}

                return to_criteo_schema(tr_raw), to_criteo_schema(va_raw)

            elif ds_type == "parquet":
                data_files = ds_cfg["data_files"]
                train_stream = load_dataset("parquet", data_files=data_files["train"], streaming=True)[
                    "train"
                ]
                val_stream = load_dataset("parquet", data_files=data_files["validation"], streaming=True)[
                    "train"
                ]
                return train_stream, val_stream

            else:
                raise ValueError(f"Unknown dataset.type: {ds_type}")

        except DataFilesNotFoundError as e:
            raise RuntimeError(
                "데이터 파일을 찾을 수 없어요. 만약 'criteo/CriteoClickLogs'를 지정했다면 그 레포엔 실제 파일이 없습니다.\n"
                "- 허브에 실데이터가 있는 'criteo/criteo-attribution-dataset'로 바꾸거나\n"
                "- 1TB 원본 TSV를 받아서 configs에 로컬 경로(csv/parquet)로 지정하세요."
            ) from e

    train_stream, val_stream = build_streams(cfg)

    # 통계 추정(소샘플)
    sample = list(itertools.islice(train_stream, int(cfg.get("sample_size", 100000))))
    # 스키마 자동 적응: {label, dense_features, cat_features}로 매핑
    adapter_fn = None
    if sample and all(k in sample[0] for k in ("label", "dense_features", "cat_features")):
        sample_mapped = sample
        d_dense = len(sample_mapped[0]["dense_features"])  # type: ignore[index]
        n_cats = len(sample_mapped[0]["cat_features"])  # type: ignore[index]
    else:
        ds_cfg_local = cfg.get("dataset", {}) if isinstance(cfg.get("dataset"), dict) else {}
        ex0 = sample[0]
        cand = [ds_cfg_local.get("label")] if isinstance(ds_cfg_local, dict) and ds_cfg_local.get("label") else []
        cand += ["label", "clicked", "click", "conversion", "converted", "is_click", "target", "y"]
        label_key = next((k for k in cand if k in ex0), None)
        exclude = set(ds_cfg_local.get("exclude", [])) if isinstance(ds_cfg_local, dict) else set()
        numeric_cols = [k for k, v in ex0.items() if k != label_key and k not in exclude and isinstance(v, (int, float)) and not isinstance(v, bool)]
        cat_cols = [k for k, v in ex0.items() if k != label_key and k not in exclude and not (isinstance(v, (int, float)) and not isinstance(v, bool))]
        def adapter_fn(ex):
            val = ex.get(label_key, 0) if label_key else 0
            try:
                yv = int(val)
            except Exception:
                try:
                    yv = int(float(val))
                except Exception:
                    yv = 0
            dense_vals = []
            for k in numeric_cols:
                v = ex.get(k, 0.0)
                try:
                    dense_vals.append(float(v) if v is not None else 0.0)
                except Exception:
                    dense_vals.append(0.0)
            cat_vals = [str(ex.get(k, "MISS")) for k in cat_cols]
            return {"label": yv, "dense_features": dense_vals, "cat_features": cat_vals}
        sample_mapped = [adapter_fn(ex) for ex in sample]
        d_dense = len(numeric_cols)
        n_cats = len(cat_cols)
    # Compute robust mean/std over the sample
    dense_mean = [
        sum(ex["dense_features"][i] for ex in sample_mapped) / max(1, len(sample_mapped))
        for i in range(d_dense)
    ]
    dense_var = []
    for i in range(d_dense):
        mu = dense_mean[i]
        sse = 0.0
        for ex in sample_mapped:
            dx = ex["dense_features"][i] - mu
            sse += dx * dx
        var = sse / max(1, len(sample_mapped))
        dense_var.append(var)
    dense_std = [(v ** 0.5 if v > 1e-12 else 1.0) for v in dense_var]
    bucket_sizes = [int(cfg.get("bucket_size_default", 1_000_003))] * n_cats

    # stream 재생성 (샘플 소비 후) + 필요시 어댑터 적용
    train_stream, val_stream = build_streams(cfg)
    if adapter_fn is not None:
        def wrap(stream):
            for ex in stream:
                yield adapter_fn(ex)
        train_stream = wrap(train_stream)
        val_stream = wrap(val_stream)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)
    model = build_model(cfg["model"], d_dense, bucket_sizes, cfg["emb_dim"], dropout=float(cfg.get("dropout", 0.1))).to(device)
    lr = float(cfg.get("lr", 1e-3))
    wd = float(cfg.get("weight_decay", 1e-5))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    bce = nn.BCEWithLogitsLoss()

    # config 백업
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({**cfg, "device": str(device)}, f, indent=2)

    # CSV 헤더
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_logloss", "val_auc"])

    def run_epoch(dl, epoch, max_steps):
        model.train()
        losses = []
        for step, (dense, cats, y) in enumerate(
            tqdm(dl, total=max_steps, desc=f"Epoch {epoch}", ncols=100)
        ):
            dense, cats = dense.to(device), cats.to(device)
            y = y.to(device).unsqueeze(1)
            logits = model(dense, cats)
            loss = bce(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())
            if step + 1 >= max_steps:
                break
        return sum(losses) / len(losses) if losses else float("nan")

    @torch.no_grad()
    def evaluate(dl, max_steps):
        model.eval()
        ys, ps = [], []
        for step, (dense, cats, y) in enumerate(
            tqdm(dl, total=max_steps, desc="Val", ncols=100)
        ):
            logits = model(dense.to(device), cats.to(device)).detach().cpu().numpy().ravel()
            probs = 1.0 / (1.0 + np.exp(-logits))
            ys += y.numpy().tolist()
            ps += probs.tolist()
            if step + 1 >= max_steps:
                break
        if len(ys) == 0:
            return float("nan"), float("nan")
        # Be robust to single-class eval windows
        try:
            va_logloss = log_loss(ys, ps, labels=[0, 1])
        except Exception:
            va_logloss = float("nan")
        try:
            va_auc = roc_auc_score(ys, ps) if len(set(ys)) >= 2 else float("nan")
        except Exception:
            va_auc = float("nan")
        return va_logloss, va_auc

    @torch.no_grad()
    def evaluate_collect(dl, max_steps):
        model.eval()
        ys, ps = [], []
        for step, (dense, cats, y) in enumerate(
            tqdm(dl, total=max_steps, desc="Val-Collect", ncols=100)
        ):
            logits = model(dense.to(device), cats.to(device)).detach().cpu().numpy().ravel()
            probs = 1.0 / (1.0 + np.exp(-logits))
            ys += y.numpy().tolist()
            ps += probs.tolist()
            if step + 1 >= max_steps:
                break
        return ys, ps

    best = {"val_logloss": float("inf"), "val_auc": 0.0, "epoch": 0}
    # NEW: 타임스탬프가 들어간 best 파일명
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    best_path = os.path.join(run_dir, f"best_{stamp}.pth")

    log(f"Start training: model={cfg['model']}, device={device}, epochs={cfg['epochs']}, run={cfg['run_name']}")

    # Early stopping state
    es_metric = cfg.get("early_stop_metric", "logloss").lower()
    patience = int(cfg.get("patience", 0))
    best_es = float("inf") if es_metric == "logloss" else -float("inf")
    stall = 0

    for epoch in range(1, cfg["epochs"] + 1):
        # Fresh train/val streams each epoch to avoid exhaustion
        fresh_train, fresh_val = build_streams(cfg)
        # Streaming shuffle for training if available
        try:
            fresh_train = fresh_train.shuffle(buffer_size=int(cfg.get("shuffle_buffer", 100_000)), seed=epoch)
        except Exception:
            pass
        if adapter_fn is not None:
            def wrap(stream):
                for ex in stream:
                    yield adapter_fn(ex)
            fresh_train = wrap(fresh_train)
            fresh_val = wrap(fresh_val)
        train_ds_epoch = CriteoIterable(fresh_train, bucket_sizes, dense_mean, dense_std)
        val_ds_epoch = CriteoIterable(fresh_val, bucket_sizes, dense_mean, dense_std)
        train_loader = DataLoader(train_ds_epoch, batch_size=cfg["batch_size"], num_workers=int(cfg.get("num_workers", 0)))
        val_loader = DataLoader(val_ds_epoch, batch_size=cfg["batch_size"], num_workers=int(cfg.get("num_workers", 0)))

        tr_loss = run_epoch(train_loader, epoch, cfg["max_train_steps"])
        va_logloss, va_auc = evaluate(val_loader, cfg["max_val_steps"])

        log(f"Epoch {epoch:03d} | train_loss={tr_loss:.6f}  val_logloss={va_logloss:.6f}  val_auc={va_auc:.6f}")
        tb.add_scalar("loss/train", tr_loss, epoch)
        tb.add_scalar("loss/val_logloss", va_logloss, epoch)
        tb.add_scalar("metric/val_auc", va_auc, epoch)
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{tr_loss:.6f}", f"{va_logloss:.6f}", f"{va_auc:.6f}"])

        # Best checkpoint policy
        save_metric = cfg.get("save_metric", "logloss").lower()
        if save_metric == "auc":
            is_better = va_auc > best["val_auc"] or (va_auc == best["val_auc"] and va_logloss < best["val_logloss"])
        else:
            is_better = va_logloss < best["val_logloss"] or (va_logloss == best["val_logloss"] and va_auc > best["val_auc"]) 

        if is_better:
            best.update({"val_logloss": va_logloss, "val_auc": va_auc, "epoch": epoch})
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": cfg,
                    "epoch": epoch,
                    "val_logloss": va_logloss,
                    "val_auc": va_auc,
                    "saved_at": stamp
                },
                best_path
            )
            log(f"** best updated: epoch={epoch}, val_logloss={va_logloss:.6f}, val_auc={va_auc:.6f} -> {best_path}")
            # Calibration plot (optional)
            try:
                import matplotlib.pyplot as plt
                from sklearn.calibration import calibration_curve
                ys_det, ps_det = evaluate_collect(val_loader, cfg["max_val_steps"])
                if ys_det:
                    frac_pos, mean_pred = calibration_curve(ys_det, ps_det, n_bins=10, strategy="uniform")
                    fig = plt.figure(figsize=(4, 4))
                    plt.plot([0, 1], [0, 1], "k--", label="perfectly calibrated")
                    plt.plot(mean_pred, frac_pos, marker="o", label="model")
                    plt.xlabel("Mean predicted value")
                    plt.ylabel("Fraction of positives")
                    plt.title("Reliability Diagram")
                    plt.legend()
                    fig_path = os.path.join(run_dir, f"calibration_epoch{epoch}.png")
                    plt.tight_layout()
                    plt.savefig(fig_path)
                    tb.add_figure("calibration/reliability", fig, epoch)
                    plt.close(fig)
                    log(f"Saved calibration plot: {fig_path}")
            except Exception as _:
                pass

        # Early stopping check
        metric_val = va_logloss if es_metric == "logloss" else va_auc
        improved = (metric_val < best_es) if es_metric == "logloss" else (metric_val > best_es)
        if improved:
            best_es = metric_val
            stall = 0
        else:
            stall += 1
        if patience and stall >= patience:
            log(f"Early stopping at epoch {epoch} (no improvement in {patience} epochs)")
            break

    tb.close()
    log(f"Done. Best @ epoch {best['epoch']}: val_logloss={best['val_logloss']:.6f}, val_auc={best['val_auc']:.6f}")


if __name__ == "__main__":
    main()

# scheduler
sched = None
if cfg.get("lr_scheduler","cosine") == "cosine":
    # warmup 후 cosine
    total_epochs = int(cfg["epochs"])
    warmup = int(cfg.get("lr_warmup_epochs", 1))
    def lr_lambda(ep):
        if ep <= warmup:
            return max(1e-3, ep / max(1, warmup))  # 선형 워밍업
        # cosine (ep는 1부터 시작하니 -1 보정 가능)
        t = (ep - warmup) / max(1, total_epochs - warmup)
        return 0.5 * (1 + math.cos(math.pi * t))
    import math
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
elif cfg.get("lr_scheduler") == "step":
    sched = torch.optim.lr_scheduler.StepLR(
        opt, step_size=int(cfg.get("step_size", 15)), gamma=float(cfg.get("gamma", 0.5))
)