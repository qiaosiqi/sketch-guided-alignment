"""data/apps_loader.py 的 smoke tests。"""
from v2.data.apps_loader import load_one, iter_split, load_train_val


def test_load_one_fncall(fake_apps_root):
    p = load_one(fake_apps_root / "train" / "0001", split="train")
    assert p is not None
    assert p.task_id == "apps_train_0001"
    assert p.io_format == "fncall"
    assert p.fn_name == "twice"
    assert p.difficulty == "interview"
    assert p.starter_code is not None


def test_load_one_stdio(fake_apps_root):
    p = load_one(fake_apps_root / "train" / "0002", split="train")
    assert p is not None
    assert p.io_format == "stdio"
    assert p.fn_name is None
    assert p.difficulty == "interview"


def test_load_one_introductory_filtered(fake_apps_root):
    p = load_one(fake_apps_root / "train" / "0003", split="train")
    assert p is None   # introductory 被过滤掉


def test_load_one_competition_filtered(fake_apps_root):
    p = load_one(fake_apps_root / "train" / "0004", split="train")
    assert p is None   # competition 也被过滤,只保留 interview


def test_iter_split(fake_apps_root):
    probs = list(iter_split(fake_apps_root, "train"))
    # introductory(0003)与 competition(0004)被过滤,只剩 0001 + 0002
    assert len(probs) == 2
    diffs = sorted(p.difficulty for p in probs)
    assert diffs == ["interview", "interview"]


def test_load_train_val_split(fake_apps_root):
    train, val = load_train_val(fake_apps_root, val_ratio=0.5, seed=1)
    # 只有 2 题:val_ratio=0.5 → 1 val + 1 train
    assert len(train) + len(val) == 2
