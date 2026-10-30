import pytest

from agentenv_sandbox_index.scoring import index_under, metric_score, rank, sub_scores

# The methodology's worked example: eleven made-up providers with made-up statistics. Not measurements.
COLUMNS = ("a1", "a2", "a2p", "a3", "s1", "s1p", "c1", "c2", "c3", "c4", "r1", "r2", "r3")
ROWS = {
    "p01": ((3.8, 6.0, 15, 2.9, 0.45, 1.6, 8, 99.95, 42, 230, 99.95, 99.97, 99.98), 1.5),
    "p02": ((9.5, 12, 30, 3.0, 2.2, 5.5, 25, 99.9, 210, 150, 99.9, 99.95, 99.97), 1.7),
    "p03": ((4.6, 7.2, 18, 3.2, 0.3, 1.1, 10, 99.9, 70, 200, 99.9, 99.93, 99.95), 1.6),
    "p04": ((3.4, 6.1, 20, 3.6, 0.12, 0.7, 14, 99.8, 150, 170, 99.85, 99.9, 99.9), 1.9),
    "p05": ((6.5, 9.5, 24, 3.4, 0.8, 2.6, 18, 99.85, 180, 140, 99.97, 99.98, 99.99), 1.4),
    "p06": ((5.8, 8.8, 23, 3.5, 0.9, 2.9, 16, 99.85, 160, 160, 99.9, 99.93, 99.95), 1.8),
    "p07": ((5.5, 8.5, 26, 3.9, 0.6, 2.4, 20, 99.7, 260, 120, 99.8, 99.85, 99.9), 2.0),
    "p08": ((7.2, 11, 32, 4.1, 1.1, 3.8, 28, 99.6, None, 105, 99.7, 99.8, 99.85), 2.6),
    "p09": ((8.0, 12, 38, 3.7, 1.4, 4.5, 32, 99.5, 400, 95, 99.6, 99.8, 99.8), 2.3),
    "p10": ((11, 15, 45, 4.4, 2.5, 7.0, 45, 99.4, None, 80, 99.5, 99.7, 99.7), 2.5),
    "p11": ((5.0, 8.0, 22, 3.8, 0.5, 2.0, 24, 99.6, 520, 110, 99.4, 99.7, 99.8), 2.4),
}
EXAMPLE = {pid: (dict(zip(COLUMNS, values, strict=True)), ci) for pid, (values, ci) in ROWS.items()}


def test_reference_points():
    assert metric_score("S1", 0.1) == 100
    assert metric_score("S1", 30) == 1
    assert metric_score("R1", 99) == pytest.approx(100 / 3)
    assert metric_score("R1", 99.99) == pytest.approx(100)
    assert metric_score("R1", 90) == 1
    assert metric_score("C4", 600) == pytest.approx(100)


def test_example_ranking():
    ranked = rank(EXAMPLE)
    assert [(r.id, round(r.index, 1), r.band_label) for r in ranked] == [
        ("p01", 71.5, "1"),
        ("p03", 68.0, "2="),
        ("p04", 67.0, "2="),
        ("p05", 64.4, "3"),
        ("p06", 60.9, "4="),
        ("p07", 57.8, "4="),
        ("p02", 55.8, "5="),
        ("p11", 53.5, "5="),
        ("p08", 53.4, "5="),
        ("p09", 49.7, "6="),
        ("p10", 45.4, "6="),
    ]
    for r in ranked:
        assert r.rank_lo <= r.rank <= r.rank_hi or abs(r.rank - r.rank_lo) <= 1


def test_missing_statistic_rescales_its_sub_index():
    stats = dict(EXAMPLE["p08"][0])
    assert set(sub_scores(stats)) == {"agent", "startup", "scale", "rel"}
    stats["c3"] = 100.0
    assert sub_scores(stats)["scale"] != sub_scores(EXAMPLE["p08"][0])["scale"]


def test_reliability_cap():
    sub = {"agent": 90.0, "startup": 90.0, "scale": 90.0, "rel": 90.0}
    assert index_under(sub, 98.9, {"agent": 1, "startup": 1, "scale": 1, "rel": 1}) == 40
    assert index_under(sub, 99.0, {"agent": 1, "startup": 1, "scale": 1, "rel": 1}) == pytest.approx(90)


def test_rank_needs_every_sub_index():
    with pytest.raises(ValueError, match="Reliability"):
        rank({"p01": ({"a1": 1, "s1": 1, "s1p": 1, "c1": 1}, 1.0)})
