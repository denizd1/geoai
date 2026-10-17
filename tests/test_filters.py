"""Fourier tabanlı potansiyel alan filtrelerinin analitik çözümlerle doğrulanması."""
import numpy as np
import pytest

from geoai.core.preprocessor import GeophysicalFilters as F

G = 6.674e-11
MASS = 1e12          # kg
N = 201
DX = 100.0           # m
x = (np.arange(N) - N // 2) * DX
X, Y = np.meshgrid(x, x)
C = slice(50, 151)   # kenar etkilerinden uzak merkez bölge


def g_sphere(d):
    """Nokta kütlenin z = 0 düzlemindeki dikey gravitesi (mGal)."""
    return G * MASS * d / (X**2 + Y**2 + d**2) ** 1.5 * 1e5


def dgdz_sphere(d):
    """∂g/∂z (z aşağı pozitif), analitik."""
    r2 = X**2 + Y**2
    return G * MASS * (2 * d**2 - r2) / (r2 + d**2) ** 2.5 * 1e5


@pytest.mark.parametrize('depth,height', [(500.0, 200.0), (1000.0, 500.0)])
def test_upward_continuation_matches_deeper_source(depth, height):
    up = F.upward_continuation(g_sphere(depth), DX, DX, height)
    ref = g_sphere(depth + height)
    assert np.abs(up - ref)[C, C].max() / ref.max() < 2e-3


@pytest.mark.parametrize('depth', [500.0, 1000.0])
def test_vertical_derivative_matches_analytic(depth):
    vd = F.vertical_derivative(g_sphere(depth), DX, DX)
    ref = dgdz_sphere(depth)
    assert np.abs(vd - ref)[C, C].max() / np.abs(ref).max() < 2e-3


def test_second_vertical_derivative_is_derivative_of_first():
    g = g_sphere(800.0)
    v2 = F.vertical_derivative(g, DX, DX, order=2)
    v1_of_v1 = F.vertical_derivative(F.vertical_derivative(g, DX, DX), DX, DX)
    assert np.abs(v2 - v1_of_v1)[C, C].max() / np.abs(v2).max() < 1e-2


def test_stabilized_derivative_equals_derivative_of_continued_field():
    g = g_sphere(600.0)
    h = 150.0
    a = F.vertical_derivative(g, DX, DX, stabilize_height=h)
    ref = dgdz_sphere(600.0 + h)
    assert np.abs(a - ref)[C, C].max() / np.abs(ref).max() < 2e-3


def test_analytic_signal_and_tilt():
    d = 700.0
    g = g_sphere(d)
    asa = F.analytic_signal(g, DX, DX)
    gx = np.gradient(g, DX, axis=1)
    gy = np.gradient(g, DX, axis=0)
    ref = np.sqrt(gx**2 + gy**2 + dgdz_sphere(d) ** 2)
    assert np.abs(asa - ref)[C, C].max() / ref.max() < 2e-3

    tilt = F.tilt_angle(g, DX, DX)
    assert tilt.min() >= -np.pi / 2 and tilt.max() <= np.pi / 2
    assert tilt[N // 2, N // 2] == pytest.approx(np.pi / 2, abs=1e-3)   # kaynak üstü
    # Tilt sıfır konturu: ∂g/∂z = 0 → r = √2·d
    r0 = np.sqrt(2) * d
    k = N // 2 + int(round(r0 / DX))
    assert abs(tilt[N // 2, k]) < 0.1

    lw = F.local_wavenumber(g, DX, DX)
    assert (lw >= 0).all() and np.isfinite(lw).all()


def test_upward_continuation_preserves_mean_and_rejects_bad_input():
    g = g_sphere(500.0) + 25.0
    up = F.upward_continuation(g, DX, DX, 300.0)
    assert up.mean() == pytest.approx(g.mean(), rel=1e-3)
    with pytest.raises(ValueError):
        F.upward_continuation(g, DX, DX, -1.0)
    bad = g.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        F.vertical_derivative(bad, DX, DX)


def test_horizontal_gradient_of_plane():
    g = 3.0 * X - 4.0 * Y
    thg = F.horizontal_gradient(g, DX, DX)
    np.testing.assert_allclose(thg, 5.0)
