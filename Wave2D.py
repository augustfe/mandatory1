import numpy as np
import sympy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib import cm

from typing import Optional

x, y, t = sp.symbols("x,y,t")


class Wave2D:

    def create_mesh(self, N: int, use_sparse: bool = False):
        """Create 2D mesh and store in self.xij and self.yij"""
        # self.xji, self.yij = ...
        raise NotImplementedError

    def D2(self, N: int):
        """Return second order differentiation matrix"""
        raise NotImplementedError

    @property
    def w(self):
        """Return the dispersion coefficient"""
        raise NotImplementedError

    def ue(self, mx: int, my: int):
        """Return the exact standing wave"""
        return sp.sin(mx * sp.pi * x) * sp.sin(my * sp.pi * y) * sp.cos(self.w * t)

    def initialize(self, N: int, mx: int, my: int):
        r"""Initialize the solution at $U^{n}$ and $U^{n-1}$

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave
        """
        raise NotImplementedError

    @property
    def dt(self):
        """Return the time step"""
        raise NotImplementedError

    def l2_error(self, u: np.ndarray, t0: float):
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        raise NotImplementedError

    def apply_bcs(self):
        raise NotImplementedError

    def __call__(
        self,
        N: int,
        Nt: int,
        cfl: float = 0.5,
        c: float = 1.0,
        mx: int = 3,
        my: int = 3,
        store_data: Optional[int] = None,
    ):
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
        cfl : number
            The CFL number
        c : number
            The wave speed
        mx, my : int
            Parameters for the standing wave
        store_data : int
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (h, l2-error)
        """
        raise NotImplementedError

    def convergence_rates(
        self, m: int = 4, cfl: float = 0.1, Nt: int = 10, mx: int = 3, my: int = 3
    ):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        cfl : number
            The CFL number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for m in range(m):
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err[-1])
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [
            np.log(E[i - 1] / E[i]) / np.log(h[i - 1] / h[i])
            for i in range(1, m + 1, 1)
        ]
        return r, np.array(E), np.array(h)


class Wave2D_Neumann(Wave2D):

    def D2(self, N: int):
        raise NotImplementedError

    def ue(self, mx: int, my: int):
        raise NotImplementedError

    def apply_bcs(self):
        raise NotImplementedError


def test_convergence_wave2d() -> None:
    sol = Wave2D()
    r, E, h = sol.convergence_rates(mx=2, my=3)
    assert abs(r[-1] - 2) < 1e-2


def test_convergence_wave2d_neumann() -> None:
    solN = Wave2D_Neumann()
    r, E, h = solN.convergence_rates(mx=2, my=3)
    assert abs(r[-1] - 2) < 0.05


def test_exact_wave2d():
    raise NotImplementedError
