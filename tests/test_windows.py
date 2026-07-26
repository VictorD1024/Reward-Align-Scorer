import pytest

from raise_scorer.scorer import sliding_windows


def test_sliding_windows_overlaps():
    text = "abcdefghij"
    windows = sliding_windows(text, window=4, stride=2, max_windows=10)
    assert windows[0] == ("abcd", 0, 4)
    assert windows[1] == ("cdef", 2, 6)
    assert windows[-1][2] == len(text)


def test_sliding_windows_empty_text():
    assert sliding_windows("", window=4, stride=2) == []


@pytest.mark.parametrize("token_count", [4096, 8192])
def test_sliding_windows_long_response_keeps_head_and_tail(token_count):
    text = "token " * token_count
    windows = sliding_windows(text, window=48, stride=24, max_windows=256)

    assert len(windows) == 256
    assert windows[0][1] == 0
    assert windows[-1][2] == len(text.strip())
    assert any(start > len(text) // 2 for _, start, _ in windows)


def test_sliding_windows_capped_sampling_is_ordered():
    text = "x" * 40_000
    windows = sliding_windows(text, window=128, stride=64, max_windows=32)
    starts = [start for _, start, _ in windows]

    assert starts == sorted(starts)
    assert len(starts) == len(set(starts))
    assert windows[-1][2] == len(text)


@pytest.mark.parametrize(
    ("window", "stride", "max_windows", "message"),
    [
        (0, 1, 10, "window and stride must be positive"),
        (4, 0, 10, "window and stride must be positive"),
        (4, 4, 10, "stride must be smaller than window"),
        (4, 2, 0, "max_windows must be positive"),
    ],
)
def test_sliding_windows_rejects_invalid_config(window, stride, max_windows, message):
    with pytest.raises(ValueError, match=message):
        sliding_windows("abcdefghij", window=window, stride=stride, max_windows=max_windows)
