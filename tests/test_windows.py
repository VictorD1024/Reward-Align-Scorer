from reward_align_scorer.scorer import sliding_windows


def test_sliding_windows_overlaps():
    text = "abcdefghij"
    windows = sliding_windows(text, window=4, stride=2, max_windows=10)
    assert windows[0] == ("abcd", 0, 4)
    assert windows[1] == ("cdef", 2, 6)
    assert windows[-1][2] == len(text)


def test_sliding_windows_empty_text():
    assert sliding_windows("", window=4, stride=2) == []
