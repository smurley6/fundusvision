import numpy as np
import pandas as pd
import pytest

from fundusvision.data.odir import Split, assert_no_leakage, build_eye_table, patient_grouped_split


def test_eye_table_uses_each_image_once(odir_dir):
    table = build_eye_table(odir_dir / "full_df.csv", odir_dir / "preprocessed_images")
    assert len(table) == 48
    assert table["filename"].is_unique
    assert set(table["label"].unique()) == {0, 1}
    assert table["path"].map(lambda p: p.endswith(".jpg")).all()


def test_patient_grouped_split_has_no_leakage(odir_dir):
    table = build_eye_table(odir_dir / "full_df.csv", odir_dir / "preprocessed_images")
    split = patient_grouped_split(table, seed=0, test_folds=4, val_folds=3)
    assert len(split.train) + len(split.val) + len(split.test) == len(table)
    for a, b in ((split.train, split.val), (split.train, split.test), (split.val, split.test)):
        assert not set(a["patient_id"]) & set(b["patient_id"])
        assert not set(a["path"]) & set(b["path"])
    frame = split.as_frame()
    assert set(frame["split"]) == {"train", "val", "test"}


def test_split_is_deterministic(odir_dir):
    table = build_eye_table(odir_dir / "full_df.csv", odir_dir / "preprocessed_images")
    a = patient_grouped_split(table, seed=7, test_folds=4, val_folds=3)
    b = patient_grouped_split(table, seed=7, test_folds=4, val_folds=3)
    assert a.test["path"].tolist() == b.test["path"].tolist()


def test_assert_no_leakage_detects_shared_patient():
    rows = pd.DataFrame(
        {
            "path": [f"{i}.jpg" for i in range(6)],
            "patient_id": [0, 0, 1, 1, 2, 2],
            "label": [0, 1] * 3,
        }
    )
    leaky = Split(
        train=rows.iloc[:3], val=rows.iloc[3:5], test=rows.iloc[5:]
    )  # patient 1 in train and val
    with pytest.raises(ValueError, match="patient_id leakage"):
        assert_no_leakage(leaky)


def test_split_stratification_roughly_preserved():
    rng = np.random.default_rng(0)
    n = 400
    patients = np.repeat(np.arange(n // 2), 2)
    table = pd.DataFrame(
        {
            "path": [f"{i}.jpg" for i in range(n)],
            "patient_id": patients,
            "label": rng.integers(0, 2, n),
            "filename": "",
            "side": "",
            "keywords": "",
        }
    )
    split = patient_grouped_split(table, seed=1)
    overall = table["label"].mean()
    for part in (split.train, split.val, split.test):
        assert abs(part["label"].mean() - overall) < 0.1
