import numpy as np
import pytest
from incwear.utils.thresholds import get_ind_acc_threshold2
from incwear.utils.data_models import Thresholds
from incwear.errors import ThresholdComputationError


def test_thresholds_computed_for_valid_signal():
    x = np.array([0, 0.2, 1.3, 0.4, 0, -0.2, -1.4, -0.3, 0, 1.1, 0])
    th = get_ind_acc_threshold2(x)

    assert isinstance(th, Thresholds)
    assert np.isfinite(th.accth)
    assert np.isfinite(th.naccth)
    assert th.accth > 0
    assert th.naccth < 0


def test_thresholds_fail_when_no_valid_peaks():
    x = np.zeros(100)

    with pytest.raises(ThresholdComputationError):
        get_ind_acc_threshold2(x)


