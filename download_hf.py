from datasets import load_dataset


def get_criteo_stream(split: str = "train"):
    """
    Return streaming dataset iterator for Criteo 1TB click logs.
    Each example expected keys: 'label', 'dense_features', 'cat_features'.
    """
    ds = load_dataset(
        "criteo/CriteoClickLogs",
        split=split,
        streaming=True,
    )
    return ds


if __name__ == "__main__":
    ds = get_criteo_stream("train")
    for i, ex in enumerate(ds):
        if i < 3:
            print(ex)
        else:
            break

