import numpy as np
import pytest

from open_kardia.leads import LEAD_NAMES_6, derive_leads


def test_identities_on_random_signals():
    rng = np.random.default_rng(1)
    I = rng.normal(size=1000)
    II = rng.normal(size=1000)
    L = derive_leads(I, II)
    np.testing.assert_allclose(L.III, II - I)
    np.testing.assert_allclose(L.aVR, -(I + II) / 2)
    np.testing.assert_allclose(L.aVL, (I - L.III) / 2)
    np.testing.assert_allclose(L.aVF, (II + L.III) / 2)
    # Goldberger: aVR + aVL + aVF == 0 ; Einthoven: I + III == II
    np.testing.assert_allclose(L.aVR + L.aVL + L.aVF, 0, atol=1e-12)
    np.testing.assert_allclose(L.I + L.III, L.II, atol=1e-12)


def test_known_vector():
    # I = 1 mV, II = 2 mV -> III = 1, aVR = -1.5, aVL = 0, aVF = 1.5
    L = derive_leads([1.0], [2.0])
    assert L.III[0] == pytest.approx(1.0)
    assert L.aVR[0] == pytest.approx(-1.5)
    assert L.aVL[0] == pytest.approx(0.0)
    assert L.aVF[0] == pytest.approx(1.5)


def test_matrix_shape_and_order():
    L = derive_leads(np.zeros(10), np.ones(10))
    m = L.as_matrix()
    assert m.shape == (10, 6)
    assert LEAD_NAMES_6[2] == "III"
    assert np.all(m[:, 2] == 1.0)


def test_shape_mismatch():
    with pytest.raises(ValueError):
        derive_leads(np.zeros(3), np.zeros(4))
