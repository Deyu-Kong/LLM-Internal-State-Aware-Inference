import runpy
from pathlib import Path
import pytest


def test_exposure_replays_each_shuffle_epoch_without_replacement():
    f=runpy.run_path(str(Path(__file__).resolve().parents[2]/'scripts/audit_training_exposure.py'))['exposure_indices']
    a=f(5,5,2,42)
    assert len(a)==10 and set(a[:5])==set(range(5)) and set(a[5:])==set(range(5))
    assert a==f(5,5,2,42)
    with pytest.raises(ValueError):f(0,1,1,42)
