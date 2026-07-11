import numpy as np
from incwear.core.movement_detection import get_mov


def test_single_movement_detected():
    # get_mov requires a movement to close: after the negative phase the
    # signal must swing back positive (with angular velocity > 0) for the
    # end of the movement to be located.
    accmags = np.array([0, 0.2, 1.5, 1.2, 0.3, -0.4, -1.3, -0.8, 0.2, 0.1, 0])
    velmags = np.array([0, 0.1, 0.8, 1.0, 0.5, 0.4, 0.7, 0.3, 0.5, 0.3, 0])

    over_accth = accmags > 1.0
    under_naccth = accmags < -1.0
    th_crossed = over_accth.astype(int) - under_naccth.astype(int)

    mov = get_mov(
        accmags=accmags,
        velmags=velmags,
        fs=20,
        th_crossed=th_crossed,
        over_accth=over_accth,
        under_naccth=under_naccth,
        ttdist=8,
    )

    assert mov.shape[0] >= 1


def test_no_movement_detected_for_flat_signal():
    accmags = np.zeros(100)
    velmags = np.zeros(100)

    over_accth = accmags > 1.0
    under_naccth = accmags < -1.0
    th_crossed = over_accth.astype(int) - under_naccth.astype(int)

    mov = get_mov(
        accmags=accmags,
        velmags=velmags,
        fs=20,
        th_crossed=th_crossed,
        over_accth=over_accth,
        under_naccth=under_naccth,
        ttdist=8,
    )

    assert mov.shape[0] == 0

