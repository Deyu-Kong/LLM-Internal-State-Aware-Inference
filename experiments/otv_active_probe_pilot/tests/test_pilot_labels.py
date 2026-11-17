"""Guard the outcome labels and tie handling used by the scientific pilot."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'active_probe_pilot', Path(__file__).parents[1] / 'scripts' / 'run_probe_pilot.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def test_final_nested_box_and_equivalent_rationals():
    text = r'First \boxed{2}; revised final \boxed{\frac{295}{323}}'
    assert pilot.rational(pilot.boxed(text)) == '295/323'
    assert pilot.rational('0.50') == pilot.rational(r'\frac{1}{2}')
    assert pilot.rational('1,234.5') == '2469/2'
    assert pilot.boxed(r'final \boxed{\frac{1}{2}') is None


@pytest.mark.parametrize('text', ['1/0', r'\sqrt{2}', 'x=2', '3 meters', '2 or 3', 'nan', '', None])
def test_ambiguous_and_unsupported_answers_remain_unlabeled(text):
    assert pilot.rational(text) is None


def test_metrics_handle_ties_and_single_class():
    assert pilot.auc([0, 0, 1, 1], [0, 1, 1, 2]) == .875
    assert pilot.auc([0, 1], [1, 1]) == .5
    assert pilot.auc([1, 1], [0, 1]) is None
    assert pilot.average_precision([0, 1], [1, 1]) == .5
    assert pilot.average_precision([1, 0, 1], [3, 2, 1]) == pytest.approx(5/6)
