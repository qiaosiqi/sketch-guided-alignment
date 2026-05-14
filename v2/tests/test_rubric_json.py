"""annotation/rubric.py 中 _extract_json_scores 的 smoke tests。

不调 GLM,只测 JSON 抽取逻辑。
"""
from v2.annotation.rubric import _extract_json_scores


SK_KEYS = ("S1", "S2", "S3", "S4")
CD_KEYS = ("C1", "C2", "C3", "C4", "C5")


def test_clean_json_at_end():
    text = "Analysis here.\n{\"S1\": 8, \"S2\": 7, \"S3\": 5, \"S4\": 6}"
    out = _extract_json_scores(text, SK_KEYS)
    assert out == {"S1": 8, "S2": 7, "S3": 5, "S4": 6}


def test_json_with_trailing_whitespace():
    text = "blah\n{\"S1\": 8, \"S2\": 7, \"S3\": 5, \"S4\": 6}\n\n"
    out = _extract_json_scores(text, SK_KEYS)
    assert out["S1"] == 8


def test_multiple_jsons_picks_last_valid():
    text = ('Maybe {"S1": 1} or {"S1": 2}.\n'
            'Final: {"S1": 8, "S2": 7, "S3": 5, "S4": 6}')
    out = _extract_json_scores(text, SK_KEYS)
    assert out["S1"] == 8


def test_clamp_above_ten():
    text = '{"S1": 11, "S2": 7, "S3": 5, "S4": 6}'
    out = _extract_json_scores(text, SK_KEYS)
    assert out["S1"] == 10


def test_clamp_below_zero():
    text = '{"S1": -3, "S2": 7, "S3": 5, "S4": 6}'
    out = _extract_json_scores(text, SK_KEYS)
    assert out["S1"] == 0


def test_missing_key_returns_none():
    text = '{"S1": 8, "S2": 7, "S3": 5}'
    out = _extract_json_scores(text, SK_KEYS)
    assert out is None


def test_garbage_text_returns_none():
    text = "no json here at all"
    out = _extract_json_scores(text, SK_KEYS)
    assert out is None


def test_string_values_rejected():
    """非数字值不能强制转换 → 该候选丢弃。"""
    # 这条 JSON 是垃圾,但旁边可能有合法的。下面构造只有垃圾的情形:
    text = '{"S1": "high", "S2": "low", "S3": "mid", "S4": "ok"}'
    out = _extract_json_scores(text, SK_KEYS)
    assert out is None


def test_code_keys_5_dims():
    text = '{"C1": 9, "C2": 8, "C3": 7, "C4": 6, "C5": 5}'
    out = _extract_json_scores(text, CD_KEYS)
    assert out == {"C1": 9, "C2": 8, "C3": 7, "C4": 6, "C5": 5}
