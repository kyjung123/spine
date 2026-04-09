import numpy as np

from spine.utils.globals import MUON_MASS
from spine.utils.mcs import highland, mcs_fit


def test_mcs_fit_with_csda_prior_pulls_solution():
    """A strong CSDA prior should dominate the fit solution."""
    dx = 5.0
    M = MUON_MASS
    p = np.full(8, 400.0, dtype=np.float64)
    theta = highland(p, M, dx)

    csda_ke = 260.0
    fit = mcs_fit(
        theta=theta,
        M=M,
        dx=dx,
        csda_ke=csda_ke,
        csda_ke_frac=0.01,
        csda_weight=200.0,
    )

    assert np.isclose(fit, csda_ke, atol=5.0)
