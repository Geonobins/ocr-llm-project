import statistics

from evaluation.latency_experiment import percentile, poisson_arrivals, spread_indices


def test_spread_indices_cover_both_ends_evenly():
    indices = spread_indices(60)
    assert len(indices) == 60
    assert indices[0] == 1 and indices[-1] == 600
    assert 27 <= sum(16 <= i <= 315 for i in indices) <= 33


def test_poisson_arrivals_are_reproducible_and_match_the_rate():
    first = poisson_arrivals(2000, 120, seed=7)
    assert first == poisson_arrivals(2000, 120, seed=7)
    assert first[0] == 0.0
    assert all(b >= a for a, b in zip(first, first[1:]))
    gaps = [b - a for a, b in zip(first, first[1:])]
    assert abs(statistics.fmean(gaps) - 0.5) < 0.05


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([0, 10], 95) == 9.5
