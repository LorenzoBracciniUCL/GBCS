"""Guards for the generic Wigner plotting helpers added to Plots_Functions."""
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest

from gbcs import Plots_Functions as PF

X = np.linspace(-4, 4, 21)
P = np.linspace(-4, 4, 21)
TICK = np.array([-4, 0, 4])


def test_callable_and_array_agree():
    f = lambda x, p: np.exp(-(x**2 + p**2))*np.cos(3*x)
    W = PF.Evaluate_Wigner_On_Grid(f, X, P)
    XX, PP = np.meshgrid(X, P)
    assert np.allclose(W, f(XX, PP))
    assert np.allclose(W, PF.Evaluate_Wigner_On_Grid(W, X, P))


def test_wrong_shape_is_rejected():
    with pytest.raises(ValueError, match='shape'):
        PF.Evaluate_Wigner_On_Grid(np.zeros((3, 4)), X, P)


def test_non_finite_is_rejected():
    """A NaN used to reach imshow, where it made vmin/vmax NaN and rendered a blank
    figure with no error at all."""
    bad = np.zeros((len(P), len(X)))
    bad[2, 2] = np.nan
    with pytest.raises(ValueError, match='non-finite'):
        PF.Evaluate_Wigner_On_Grid(bad, X, P)
    bad[2, 2] = np.inf
    with pytest.raises(ValueError, match='non-finite'):
        PF.Evaluate_Wigner_On_Grid(bad, X, P)


def test_non_square_grid_axes_are_correct():
    """The house formula folds len(p)/len(x) into both the limits and the extent, which
    is only right for a square grid with equal ranges.  With 81x41 it put the p axis at
    (-4, 4) and the image extent at (-1.012, 1.012) for data spanning (-2, 2)."""
    x = np.linspace(-4, 4, 81)
    p = np.linspace(-2, 2, 41)
    fig, im, W = PF.Plot_Wigner_Generic(lambda xx, pp: np.exp(-(xx**2 + pp**2)),
                                        x, p, np.array([-2, 0, 2]), False, False)
    assert W.shape == (len(p), len(x))
    assert fig.axes[0].get_xlim() == pytest.approx((-4.0, 4.0))
    assert fig.axes[0].get_ylim() == pytest.approx((-2.0, 2.0))
    assert im.get_extent() == pytest.approx((-4.0, 4.0, -2.0, 2.0))


def test_square_grid_rendering_is_unchanged():
    """The fix must not move the square-grid case, which every notebook uses."""
    fig, im, _ = PF.Plot_Wigner_Generic(np.zeros((len(P), len(X))), X, P, TICK, False, False)
    assert im.get_extent() == pytest.approx((-4.0, 4.0, -4.0, 4.0))
    assert fig.axes[0].get_ylim() == pytest.approx((-4.0, 4.0))


def test_weight_warning_is_a_real_warning():
    """It used to be a bare print, so it could not be filtered or asserted on."""
    W0 = np.zeros((len(P), len(X)))
    with pytest.warns(RuntimeWarning, match='weights sum'):
        PF.Plot_Wigner_GCS_Weighted(W0, W0, W0, 0.5, 0.3, 0.9, X, P, TICK, False, False)


def test_weighted_sum_is_exact_and_unnormalised():
    rng = np.random.default_rng(0)
    Wc, Wq = rng.normal(size=(len(P), len(X))), rng.normal(size=(len(P), len(X)))
    g = lambda xx, pp: np.exp(-(xx**2 + pp**2))
    _, _, W = PF.Plot_Wigner_GCS_Weighted(Wc, Wq, g, 0.5, 0.2, 0.3, X, P, TICK, False, False)
    assert np.allclose(W, 0.5*Wc + 0.2*Wq + 0.3*PF.Evaluate_Wigner_On_Grid(g, X, P))


def test_quantum_state_weighted_method():
    """Plot_Wigner_Function_GCS_Weighted is public API that no notebook calls (the
    notebook already holds W_c and W_q and would pay for them twice), so it is exercised
    here instead of being left untested."""
    from gbcs.Objects import Quantum_State, Hamiltonian, Decoherence

    H = Hamiltonian(1, 1)
    H.Initialize_Constant_Hamiltonians(
        [np.eye(2)], [np.array([[0.5], [0.]]), np.array([[2.0], [0.]])],
        np.array([0.0 + 0j]))
    G = Quantum_State(1, 1)
    G.Initialize_Gaussian_State(np.zeros((2, 1)), 2.6*np.diag([0.5, 2.0]),
                                np.ones((2, 2))/2)
    G.Open_Dynamics_Numerical(
        H, Decoherence(1, 1, 'Canonical', 0.05*np.array([[1., 0.], [0., 0.]])),
        np.linspace(0, 2*np.pi, 11))

    fig, im, W_sum = G.Plot_Wigner_Function_GCS_Weighted(
        9, 5, 0, 1.0, np.array([-4, 0, 4]),
        W_g=lambda x, p: np.zeros_like(x), p_c=1.0, p_q=1.0, p_g=0.0)
    assert W_sum.shape == (9, 9)
    assert np.all(np.isfinite(W_sum))


def test_post_measurement_wigner_takes_the_scalar_of_each_gaussian():
    """Wigner_PMS_func stored the 1x1 exponent array into a scalar slot, which numpy 2
    rejects ('only 0-dimensional arrays can be converted') and numpy 1.25+ deprecates."""
    import warnings
    from gbcs import Wigner_Functions
    from gbcs.Objects import Quantum_State, Hamiltonian

    H = Hamiltonian(1, 1)
    H.Initialize_Constant_Hamiltonians(
        [np.eye(2)], [np.array([[0.5], [0.]]), np.array([[2.0], [0.]])],
        np.array([0.0 + 0j]))
    G = Quantum_State(1, 1)
    G.Initialize_Gaussian_State(np.zeros((2, 1)), np.eye(2), np.ones((2, 2))/2)
    G.Unitary_Dynamics_Numerical(H, np.linspace(0, np.pi, 5))
    G.Qubit_Ideal_Measurament(np.array([[0, 1], [1, 0]])/2, 'Time')

    r_tilde_array, *_ = Wigner_Functions.Create_r_tilde_array(G, 1.0, 0, 9)
    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        W = Wigner_Functions.Wigner_PMS_func(G, 2, r_tilde_array)
    assert W.shape == (len(G.POVM_array), 9, 9)
    assert np.all(np.isfinite(W))
    assert np.max(np.abs(W)) > 0
