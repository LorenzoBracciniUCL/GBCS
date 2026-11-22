"""Geometry of the covariance contours (2-D) and the off-diagonal tilt (3-D)."""
import matplotlib
matplotlib.use('Agg')

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest

from gbcs import Plots_Functions as PF


# ── Task 1: the contour must grow with the noise ─────────────────────────────

def test_x_squeezed_state_is_short_along_x():
    """The level set is xi^T sigma^-1 xi = c^2, so semi-axes go as sqrt(mu).  Plotting the
    level set of xi^T sigma xi instead gives 1/sqrt(mu) and inverts the picture, making the
    squeezed quadrature look long."""
    s = 4.0
    sigma = np.diag([1/s, s])                     # x-squeezed
    g = PF.Covariance_Geometry(sigma)
    assert g['theta'] == pytest.approx(np.pi/2)   # major axis along p
    assert g['semi_minor'] == pytest.approx(np.sqrt(1/s))
    assert g['semi_major'] == pytest.approx(np.sqrt(s))
    assert g['semi_minor'] < 1.0 < g['semi_major']


def test_more_noise_gives_a_bigger_contour():
    small = PF.Covariance_Geometry(np.eye(2))
    big = PF.Covariance_Geometry(9*np.eye(2))
    assert big['semi_major'] > small['semi_major']
    assert big['semi_major'] == pytest.approx(3.0)      # sqrt(9), not 1/3


def test_thermal_state_is_not_the_vacuum():
    """sigma_th = (1 + 2 n_bar) 1: the variance is linear in (1+2n_bar), the radius its
    square root."""
    n_bar = 4.0
    g = PF.Covariance_Geometry((1 + 2*n_bar)*np.eye(2))
    assert g['is_isotropic']
    assert g['theta'] == 0.0
    assert g['semi_major'] == pytest.approx(np.sqrt(1 + 2*n_bar))


def test_isotropic_returns_no_garbage_angle():
    for sigma in (np.eye(2), 7.0*np.eye(2)):
        g = PF.Covariance_Geometry(sigma)
        assert g['is_isotropic'] and g['theta'] == 0.0


def test_eigenvalues_survive_deep_squeezing():
    """mu_minus = det/mu_plus, not T - D, which cancels catastrophically past ~10 dB."""
    for s in (1e3, 1e6, 1e8):
        g = PF.Covariance_Geometry(np.diag([1/s, s]))
        assert g['mu_minus'] == pytest.approx(1/s, rel=1e-10)
        assert g['mu_plus']*g['mu_minus'] == pytest.approx(1.0, rel=1e-10)


def test_orientation_uses_atan2_not_arctan():
    """A 45-degree correlated state: arctan would be ambiguous by pi/2 and could swap
    major and minor."""
    sigma = np.array([[2.0, 1.0], [1.0, 2.0]])
    g = PF.Covariance_Geometry(sigma)
    assert g['theta'] == pytest.approx(np.pi/4)
    assert g['semi_major'] == pytest.approx(np.sqrt(3.0))
    assert g['semi_minor'] == pytest.approx(1.0)


def test_radial_conventions_differ_as_documented():
    sigma = np.diag([1/4.0, 4.0])
    a = PF.Covariance_Geometry(sigma, radial='sqrt')
    b = PF.Covariance_Geometry(sigma, radial='linear')
    assert a['semi_major']/a['semi_minor'] == pytest.approx(4.0)     # exp(2r)
    assert b['semi_major']/b['semi_minor'] == pytest.approx(16.0)    # exp(4r), squared


def test_vacuum_calibrates_the_convention():
    """det(sigma) = hbar^2 exactly for the vacuum: any stray factor of 2 shows up here."""
    inv = PF.Covariance_Invariants(np.eye(2), hbar=1.0)
    assert inv['det_sigma'] == pytest.approx(1.0)
    assert inv['purity'] == pytest.approx(1.0)
    assert inv['n_bar'] == pytest.approx(0.0)
    assert inv['physical']


def test_squeezing_invariants_are_consistent():
    r_true, nu = 0.8, 2.5
    sigma = nu*np.diag([np.exp(-2*r_true), np.exp(2*r_true)])
    inv = PF.Covariance_Invariants(sigma)
    assert inv['r'] == pytest.approx(r_true)
    assert inv['nu'] == pytest.approx(nu)
    assert inv['mu_plus'] == pytest.approx(nu*np.exp(2*r_true))
    assert inv['mu_minus'] == pytest.approx(nu*np.exp(-2*r_true))


def test_determinant_criterion_is_stronger_than_the_product_of_variances():
    """Var_x Var_p >= hbar^2/4 passes states that det(sigma) >= hbar^2 correctly rejects,
    once correlations are present."""
    sigma = np.array([[1.0, 0.9], [0.9, 1.0]])          # det = 0.19 < 1
    inv = PF.Covariance_Invariants(sigma)
    assert not inv['physical']
    assert sigma[0, 0]*sigma[1, 1] >= 0.25              # the weaker test would pass it


# ── Task 3: the tilt, the shadow identity, and the shear ─────────────────────

def test_non_diagonal_block_returns_none_without_raising():
    bad = np.array([[1.0 + 0j, 0.3], [0.3, 2.0 + 0j]])
    assert PF.Off_Diagonal_Polar(bad) is None


def test_inversion_reverses_every_tilt():
    zeta, psi = 2.0, 0.4
    so = np.diag([zeta*np.exp(1j*psi), np.exp(-1j*psi)/zeta])
    a = PF.Off_Diagonal_Polar(so, use_inverse=False)
    b = PF.Off_Diagonal_Polar(so, use_inverse=True)
    assert (b['zeta_x'], b['zeta_p']) == pytest.approx((1/a['zeta_x'], 1/a['zeta_p']))
    assert (b['psi_x'], b['psi_p']) == pytest.approx((-a['psi_x'], -a['psi_p']))


def _uv_from_library(sigma_off, use_inverse=True):
    """Semi-axis vectors as the library actually draws them.

    This must go through Draw_Off_Diagonal_Tilt_3D rather than re-derive u and v: a test
    that rebuilds the construction cannot see an error in the construction.  Replacing the
    linear radial coordinate with sqrt(zeta) used to leave the whole suite green.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    try:
        pol = PF.Draw_Off_Diagonal_Tilt_3D(ax, sigma_off, use_inverse=use_inverse)
    finally:
        plt.close(fig)
    return pol['u'], pol['v'], pol


def _sigma_off_for(zeta_x, zeta_p, psi_x, psi_p, use_inverse=True):
    """Build sigma_off so that the decomposed object has the requested (zeta, psi)."""
    b = np.array([zeta_x*np.exp(1j*psi_x), zeta_p*np.exp(1j*psi_p)])
    return np.diag(1.0/b if use_inverse else b)


def _uv(zeta_x, zeta_p, psi_x, psi_p):
    """Same, via the library, addressed by the decomposed (zeta, psi)."""
    u, v, _ = _uv_from_library(_sigma_off_for(zeta_x, zeta_p, psi_x, psi_p))
    return u, v


def test_shadow_and_tip_identities_are_exact():
    """The radial coordinate is linear in the matrix entry, so the xp-shadow has semi-axes
    Re(b_j) and the tips sit at Im(b_j).  An accidental sqrt breaks the first identity."""
    b = np.array([1.7*np.exp(1j*0.6), 0.45*np.exp(-1j*0.9)])
    so = np.diag(1.0/b)                                   # so that use_inverse recovers b
    u, v, pol = _uv_from_library(so)
    assert u[0] == pytest.approx(b[0].real)
    assert v[1] == pytest.approx(b[1].real)
    assert u[2] == pytest.approx(b[0].imag)
    assert v[2] == pytest.approx(b[1].imag)
    # the sqrt error would break the shadow but not the tips
    assert np.sqrt(pol['zeta_x'])*np.cos(pol['psi_x']) != pytest.approx(b[0].real)


def test_worked_example_shear():
    """zeta_x = zeta_p = 1, psi_x = psi_p = 60 deg: u.v = 0.75, arrows 41.4 deg apart, and
    the drawn ellipse's principal axes are 1.3229 / 0.5 -- not (1, 1).  The mismatch is
    expected; asserting it catches a future change that accidentally makes them agree.

    ERRATUM in the task description, which quotes the minor semi-axis as 0.35: the Gram
    matrix [u v]^T [u v] = [[1, 0.75], [0.75, 1]] has eigenvalues 1.75 and 0.25, so the
    singular values are sqrt(1.75) = 1.3229 and sqrt(0.25) = 0.5.  Geometrically the minor
    semi-axis is |v - u|/sqrt(2) = 0.5; 0.3536 is |v - u|/2, i.e. the same vector divided
    by 2 instead of sqrt(2).  The major semi-axis 1.32 and the angle 41.4 deg are correct.
    """
    u, v = _uv(1.0, 1.0, np.radians(60), np.radians(60))
    assert u == pytest.approx([0.5, 0.0, np.sqrt(3)/2])
    assert v == pytest.approx([0.0, 0.5, np.sqrt(3)/2])
    assert float(u @ v) == pytest.approx(0.75)
    ang = np.degrees(np.arccos(u @ v/(np.linalg.norm(u)*np.linalg.norm(v))))
    assert ang == pytest.approx(41.409, abs=1e-3)
    sv = np.linalg.svd(np.column_stack([u, v]), compute_uv=False)
    assert sv == pytest.approx([np.sqrt(1.75), 0.5], abs=1e-9)
    assert float(np.linalg.norm(v - u)/np.sqrt(2)) == pytest.approx(0.5)
    assert not np.allclose(sv, [1.0, 1.0])


def test_ellipse_axes_differ_whenever_both_phases_are_nonzero():
    for zx, zp, px, pp in ((1.0, 1.0, 0.6, 0.6), (2.0, 0.5, 0.4, -0.4), (1.3, 0.8, 0.9, 0.2)):
        u, v = _uv(zx, zp, px, pp)
        sv = np.sort(np.linalg.svd(np.column_stack([u, v]), compute_uv=False))[::-1]
        assert not np.allclose(sv, np.sort([zx, zp])[::-1]), \
            "shear must make the drawn ellipse's axes differ from the moduli"


def test_zero_phase_gives_no_shear():
    u, v = _uv(2.0, 0.5, 0.0, 0.0)
    assert float(u @ v) == pytest.approx(0.0)
    sv = np.sort(np.linalg.svd(np.column_stack([u, v]), compute_uv=False))[::-1]
    assert sv == pytest.approx([2.0, 0.5])


# ── conventions that must not drift silently ─────────────────────────────────

def test_use_inverse_defaults_to_the_inverse():
    """The default decomposes (sigma^off)^-1, consistent with the on-diagonal contours and
    with the paper's geometry section.  Flipping it reverses every tilt, so the default is
    load-bearing and is pinned here rather than only passed explicitly elsewhere."""
    so = np.diag([2.0*np.exp(0.4j), np.exp(-0.4j)/2.0])
    default = PF.Off_Diagonal_Polar(so)
    explicit = PF.Off_Diagonal_Polar(so, use_inverse=True)
    assert default['use_inverse'] is True
    assert default['psi_x'] == pytest.approx(explicit['psi_x'])
    assert default['psi_x'] == pytest.approx(-PF.Off_Diagonal_Polar(so, use_inverse=False)['psi_x'])


def test_S_dB_is_ten_log10_not_twenty():
    """Variance is power-like, so squeezing in dB carries 10, not 20.  sigma = diag(0.1, 10)
    is 10 dB squeezed."""
    inv = PF.Covariance_Invariants(np.diag([0.1, 10.0]))
    assert inv['S_dB'] == pytest.approx(10.0)
    assert inv['S_dB'] != pytest.approx(20.0)


def test_physicality_tolerance_scales_with_hbar():
    """The vacuum must read as physical at any hbar; an absolute epsilon would not."""
    for hbar in (1.0, 1e-3, 1e3):
        assert PF.Covariance_Invariants(hbar*np.eye(2), hbar=hbar)['physical']


# ── guards and warnings that the spec asked for ──────────────────────────────

def test_non_positive_definite_is_rejected():
    with pytest.raises(ValueError, match='positive definite'):
        PF.Covariance_Geometry(np.diag([0.0, 0.0]))


def test_non_symmetric_off_diagonal_warns():
    """Symmetry (s01 == s10) is a different property from diagonality, and a different bug."""
    # A Hermitian block passed where a complex-symmetric one belongs: s10 = conj(s01).
    # This must warn even though the block is then rejected as non-diagonal -- ordered
    # after the diagonality gate the check was unreachable (the gate's own tolerance makes
    # the symmetry condition unsatisfiable by the triangle inequality).
    so = np.array([[1.0 + 0j, 0.3 + 0.4j], [0.3 - 0.4j, 2.0 + 0j]])
    with pytest.warns(RuntimeWarning, match='not symmetric'):
        PF.Off_Diagonal_Polar(so)

    # and a symmetric block does not warn
    good = np.diag([1.0 + 0.2j, 2.0 - 0.1j])
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        PF.Off_Diagonal_Polar(good)


def test_non_normalisable_phase_warns():
    """Normalisability of Eq.(wigner_jk) forces Re(sigma_off^-1) > 0, i.e. cos(psi) > 0."""
    so = np.diag([np.exp(1j*2.0), np.exp(-1j*0.3)])            # psi_x = -2.0 rad after inversion
    with pytest.warns(RuntimeWarning, match='not normalisable'):
        PF.Off_Diagonal_Polar(so)


def test_unphysical_state_warns_with_its_value():
    """A warning carrying the number, not an exception -- it is a convention calibrator."""
    bad = np.array([[1.0, 0.9], [0.9, 1.0]])                   # det = 0.19 < 1
    with pytest.warns(RuntimeWarning, match='unphysical'):
        PF.Plot_On_Diagonal_Contours(bad, bad)
    plt.close('all')


# ── the figure-level functions ───────────────────────────────────────────────

def test_plot_on_diagonal_contours_returns_consistent_metadata():
    sp, sm = np.diag([0.5, 2.0]), np.diag([2.0, 0.5])
    fig, ax, info = PF.Plot_On_Diagonal_Contours(sp, sm, c=1.0, radial='sqrt')
    assert set(info) == {'+', '-'}
    assert info['+']['semi_major'] == pytest.approx(np.sqrt(2.0))
    assert info['+']['radial'] == 'sqrt' and info['+']['c'] == 1.0
    assert ax.get_aspect() == 1.0
    plt.close(fig)


def test_plot_on_diagonal_contours_marks_first_moments():
    """Spec: non-zero first moments are a marker only; the contours stay at the origin."""
    sp = sm = np.eye(2)
    r = np.array([[1.5], [-0.5]])
    fig, ax, info = PF.Plot_On_Diagonal_Contours(sp, sm, r_on_plus=r)
    assert info['+']['r_offset'] == pytest.approx([1.5, -0.5])
    assert '-' in info and 'r_offset' not in info['-']
    plt.close(fig)


def test_plot_off_diagonal_3d_returns_none_when_not_diagonal():
    bad = np.array([[1.0 + 0j, 0.4], [0.4, 2.0 + 0j]])
    fig, ax, info = PF.Plot_Off_Diagonal_3D(bad)
    assert (fig, ax, info) == (None, None, None)


def test_plot_off_diagonal_3d_geometry():
    zeta, psi = 0.75, 0.5
    so = np.diag([zeta*np.exp(1j*psi), np.exp(-1j*psi)/zeta])
    fig, ax, info = PF.Plot_Off_Diagonal_3D(so, use_inverse=True)
    assert info['zeta_x'] == pytest.approx(1/zeta)
    assert info['psi_x'] == pytest.approx(-psi)
    assert info['u'][0] == pytest.approx(info['b_x'].real)      # shadow identity, end to end
    assert info['u'][2] == pytest.approx(info['b_x'].imag)
    plt.close(fig)


@pytest.mark.parametrize('z_comp', [0, 1])
def test_plot_first_moments_3d_projections(z_comp):
    """The single off-diagonal arrow carries Im[x] or Im[p] according to z_comp, and its
    projections are the in-plane point and the height on the z axis."""
    r_off = np.array([0.8 + 0.3j, -0.5 + 0.6j])
    fig, ax, info = PF.Plot_First_Moments_3D(np.array([1.0, 0.0]), np.array([-1.0, 0.0]),
                                             r_off, z_comp=z_comp)
    assert info['off'] == pytest.approx([0.8, -0.5, r_off[z_comp].imag])
    assert info['proj_plane'] == pytest.approx([0.8, -0.5, 0.0])
    assert info['proj_z'] == pytest.approx([0.0, 0.0, r_off[z_comp].imag])
    assert info['+'] == pytest.approx([1.0, 0.0, 0.0])          # on-diagonal stays in plane
    plt.close(fig)


# ── Draw_Covariance_Contour_3D: on-diagonal contours inside the 3-D moment scene ──

def _contour_3d_from_library(sigma, centre, c=1.0, radial='sqrt'):
    """Recover the polyline the library actually drew, rather than re-deriving it.

    Drawing on a throwaway 3-D axis and reading the artist back is deliberate: a test that
    rebuilds the ellipse itself would pass even if Draw_Covariance_Contour_3D were never
    called.
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    g = PF.Draw_Covariance_Contour_3D(ax, sigma, centre=centre, c=c, radial=radial)
    line = ax.get_lines()[-1]
    xs, ys = np.asarray(line.get_xdata()), np.asarray(line.get_ydata())
    zs = np.asarray(line.get_data_3d()[2])
    plt.close(fig)
    return g, np.column_stack([xs, ys]), zs


def test_contour_3d_points_satisfy_the_level_set():
    """Every drawn vertex must satisfy (xi-centre)^T sigma^-1 (xi-centre) = c^2."""
    sigma = np.array([[2.5, -0.8], [-0.8, 1.1]])
    centre = np.array([0.7, -1.3])
    c = 1.0
    _, pts, zs = _contour_3d_from_library(sigma, centre, c=c)
    d = pts - centre
    q = np.einsum('ij,jk,ik->i', d, np.linalg.inv(sigma), d)
    assert q == pytest.approx(np.full(len(q), c**2), abs=1e-10)
    assert zs == pytest.approx(np.zeros_like(zs))          # on-diagonal lives at Im = 0


def test_contour_3d_agrees_with_the_2d_view():
    """The 3-D and 2-D contours of one sigma are the same ellipse."""
    sigma = np.array([[1.75, -0.375], [-0.375, 0.8125]])   # Example 1 steady state
    g3, _, _ = _contour_3d_from_library(sigma, (0.0, 0.0))
    fig, ax = plt.subplots()
    g2 = PF.Draw_Covariance_Contour(ax, sigma)
    plt.close(fig)
    for k in ('semi_major', 'semi_minor', 'theta', 'mu_plus', 'mu_minus'):
        assert g3[k] == pytest.approx(g2[k])


def test_contour_3d_centre_tracks_the_moment():
    """The contour sits on the arrow tip, so translating the centre translates the curve."""
    sigma = np.array([[1.6, 0.3], [0.3, 0.9]])
    _, p0, _ = _contour_3d_from_library(sigma, (0.0, 0.0))
    shift = np.array([-1.77, -0.88])
    _, p1, _ = _contour_3d_from_library(sigma, shift)
    assert p1 - p0 == pytest.approx(np.broadcast_to(shift, p1.shape))


def test_driving_frame_congruence_rotates_theta_and_preserves_shape():
    """R sigma R^T with R = exp(phi Omega) turns the ellipse by -phi and leaves its
    semi-axes, and hence the symplectic eigenvalue, untouched.

    The sign matters and is easy to get wrong: exp(phi Omega) = [[c, s], [-s, c]] rotates
    CLOCKWISE, consistently with to_driving_frame mapping a drive at angle +phi onto +x.
    phi = pi/2 cannot detect this -- theta is defined modulo pi, so -pi/2 and +pi/2 are the
    same angle -- which is why phi = pi/3 is in the parameter list."""
    from scipy.linalg import expm
    sigma = np.array([[1.75, -0.375], [-0.375, 0.8125]])
    Omega = np.array([[0.0, 1.0], [-1.0, 0.0]])
    g0, _, _ = _contour_3d_from_library(sigma, (0.0, 0.0))
    for phi in (np.pi/2, np.pi/3, -0.4):
        R = expm(phi*Omega)
        g1, _, _ = _contour_3d_from_library(R @ sigma @ R.T, (0.0, 0.0))
        assert g1['semi_major'] == pytest.approx(g0['semi_major'])
        assert g1['semi_minor'] == pytest.approx(g0['semi_minor'])
        dtheta = (g1['theta'] - g0['theta'] + phi + np.pi/2) % np.pi - np.pi/2
        assert dtheta == pytest.approx(0.0, abs=1e-10)


def test_plot_first_moments_3d_reports_contour_geometry():
    """The contours are opt-in, and their geometry comes back in the returned dict."""
    sp = np.array([[1.75, -0.375], [-0.375, 0.8125]])
    sm = np.array([[1.75, 0.375], [0.375, 0.8125]])
    rp, rm = np.array([-1.77, -0.88]), np.array([1.77, -0.88])
    fig, ax, info = PF.Plot_First_Moments_3D(rp, rm)
    assert 'geom+' not in info and 'geom-' not in info
    plt.close(fig)
    fig, ax, info = PF.Plot_First_Moments_3D(rp, rm, sigma_on_plus=sp, sigma_on_minus=sm)
    assert info['geom+']['semi_major'] == pytest.approx(info['geom-']['semi_major'])
    assert info['geom+']['theta'] == pytest.approx(-info['geom-']['theta'])   # p mirror
    # the axis limits must open up to hold the contour, not just the arrow tips
    lim = ax.get_xlim3d()
    assert max(abs(lim[0]), abs(lim[1])) > abs(rp[0]) + info['geom+']['semi_minor']
    plt.close(fig)


# ── the tilted disc's intersection with the xp-plane ──────────────────────────

@pytest.mark.parametrize('zeta,psi_x,psi_p', [(0.75, 0.5, -0.3), (1.4, 0.2, 0.9),
                                              (0.6, -0.8, 0.45)])
def test_plane_intersection_is_exact_and_on_the_rim(zeta, psi_x, psi_p):
    """The chord must lie in z = 0, be antipodal, and consist of genuine rim points.

    A projection would satisfy none of these; that is the whole reason the intersection
    replaced it.
    """
    so = np.diag([zeta*np.exp(1j*psi_x), np.exp(1j*psi_p)/zeta])
    fig, ax, info = PF.Plot_Off_Diagonal_3D(so, use_inverse=True)
    w = info['plane_intersection']
    plt.close(fig)

    assert w[:, 2] == pytest.approx([0.0, 0.0], abs=1e-12)      # genuinely in the plane
    assert w[0] == pytest.approx(-w[1])                          # a diameter through 0
    # both endpoints satisfy the rim parametrisation u cos t + v sin t with cos^2+sin^2 = 1
    M = np.column_stack([info['u'], info['v']])
    for pt in w:
        coef, *_ = np.linalg.lstsq(M, pt, rcond=None)
        assert M @ coef == pytest.approx(pt, abs=1e-12)
        assert np.hypot(*coef) == pytest.approx(1.0)


def test_plane_intersection_absent_when_the_disc_lies_in_the_plane():
    """Zero phases put the whole disc at z = 0, so there is no distinguished chord."""
    so = np.diag([0.8 + 0j, 1.25 + 0j])
    fig, ax, info = PF.Plot_Off_Diagonal_3D(so, use_inverse=True)
    plt.close(fig)
    assert 'plane_intersection' not in info


def test_plane_intersection_can_be_switched_off():
    so = np.diag([0.75*np.exp(0.5j), np.exp(-0.5j)/0.75])
    fig, ax, info = PF.Plot_Off_Diagonal_3D(so, use_inverse=True,
                                            show_plane_intersection=False)
    plt.close(fig)
    assert 'plane_intersection' not in info
