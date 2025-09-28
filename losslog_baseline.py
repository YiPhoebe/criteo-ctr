# losslog_baseline.py  (견고한 baseline 체크 스크립트)
import itertools, argparse, math
import numpy as np
from datasets import load_dataset, load_dataset_builder
from sklearn.metrics import log_loss, roc_auc_score

def pick_val_stream(ds_name: str, preferred: str, take: int):
    """preferred가 없으면 test -> (없으면) train 앞부분을 잘라 임시 val 생성"""
    b = load_dataset_builder(ds_name)
    avail = set(b.info.splits.keys())
    if preferred in avail:
        return load_dataset(ds_name, split=preferred, streaming=True)
    if "test" in avail:
        return load_dataset(ds_name, split="test", streaming=True)
    # fallback: train에서 앞부분 take개 사용
    train_stream = load_dataset(ds_name, split="train", streaming=True)
    return train_stream.take(take)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="criteo/criteo-attribution-dataset")
    ap.add_argument("--val-split", default="validation")  # 없으면 자동 대체
    ap.add_argument("--max-samples", type=int, default=200_000)
    ap.add_argument("--fallback-take", type=int, default=300_000)  # validation/test 없을 때 사용
    args = ap.parse_args()

    val_stream = pick_val_stream(args.ds, args.val_split, args.fallback_take)
    buf = list(itertools.islice(val_stream, args.max_samples))
    if not buf:
        raise RuntimeError("검증 샘플을 1개도 못 가져왔어요. 데이터셋/스플릿을 확인하세요.")

    # 라벨 키 자동 감지 (label/click/target 등)
    ex0 = buf[0]
    cand = ["label", "clicked", "click", "conversion", "converted", "is_click", "target", "y"]
    label_key = next((k for k in cand if k in ex0), None)
    if label_key is None:
        raise RuntimeError(f"라벨 컬럼을 찾지 못했어요. 후보: {cand}")

    y = np.array([int(bool(x[label_key])) for x in buf], dtype=np.int32)
    p = float(y.mean())

    # 상수예측 baseline (항상 p만 예측)
    base_probs = np.full_like(y, fill_value=p, dtype=np.float32)
    base_ll = log_loss(y, base_probs, labels=[0,1])
    try:
        base_auc = roc_auc_score(y, base_probs)  # 상수면 예외 발생할 수 있음
    except Exception:
        base_auc = float("nan")

    print(f"Samples={len(y)}  PosRate(CTR)={p:.6f}")
    print(f"Baseline (const={p:.6f})  logloss={base_ll:.6f}  auc={base_auc}")

if __name__ == "__main__":
    main()