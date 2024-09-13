import numpy as np
import sympy as sp
import scipy.sparse as sparse
from scipy.sparse import diags_array

# import matplotlib.pyplot as plt
# from matplotlib import cm

from typing import Optional
import pytest

x, y, t = sp.symbols("x,y,t")


class Wave2D:

    def create_mesh(self, N: int, use_sparse: bool = True) -> None:
        """Create 2D mesh and store in self.xij and self.yij.

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction

        use_sparse : bool
            Whether to use a sparse meshgrid, defaults to True
        """
        # self.xji, self.yij = ...
        xi = np.linspace(0, 1, N + 1)
        yj = np.linspace(0, 1, N + 1)

        self.xij, self.yij = np.meshgrid(xi, yj, indexing="ij", sparse=use_sparse)
        self.h = 1 / N
        self.N = N

    def D2(self, N: int) -> sparse.dia_matrix:
        """Set up second order differentiation matrix.

        This version includes Dirichlet boundary conditions

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction

        Returns
        -------
        The differentiation matrix
        """
        D = diags_array(
            [1, -2, 1],
            offsets=[-1, 0, 1],
            shape=(N + 1, N + 1),
            format="lil",
        )
        D[0] = 0
        D[-1] = 0

        return D

    @property
    def w(self) -> float:
        """The dispersion coefficient.

        The dispersion coefficient was computed analytically from the
        true solution of the wave equation with Dirichlet conditions.

        Returns
        -------
        float
            The dispersion coefficient
        """
        kx = sp.pi * self.mx
        ky = sp.pi * self.my
        return sp.sqrt(self.c**2 * (kx**2 + ky**2))

    def ue(self, mx: int, my: int) -> sp.Expr:
        """Return the exact standing wave"""
        return sp.sin(mx * sp.pi * x) * sp.sin(my * sp.pi * y) * sp.cos(self.w * t)

    def initialize(self, N: int, mx: int, my: int) -> None:
        r"""Initialize the solution at $U^{n}$ and $U^{n-1}$.

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave
        """
        init = sp.lambdify((x, y), self.ue(mx, my).subs(t, 0))

        self.U_prev[:] = init(self.xij, self.yij)

        self.U[:] = self.U_prev + 0.5 * (self.c * self.dt) ** 2 * (
            self.D @ self.U_prev + self.U_prev @ (self.D.T)
        )

    @property
    def dt(self) -> float:
        """Return the time step"""
        return self.cfl * self.h / self.c

    def l2_error(self, u: np.ndarray, t0: float) -> float:
        """Return l2-error norm.

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        func = self.ue(self.mx, self.my)
        u_exact = sp.lambdify((x, y), func.subs(t, t0))
        diff = u_exact(self.xij, self.yij) - u

        return np.sqrt(np.sum(diff**2) * self.h**2)

    def apply_bcs(self) -> None:
        """Apply Dirichlet boundary conditions."""
        # Boundary conditions are specified directly in D2
        pass

    def __call__(
        self,
        N: int,
        Nt: int,
        cfl: float = 0.5,
        c: float = 1.0,
        mx: int = 3,
        my: int = 3,
        store_data: Optional[int] = None,
    ) -> dict[int, np.ndarray] | tuple[float, np.ndarray]:
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
        self.cfl = cfl
        self.c = c
        self.mx = mx
        self.my = my

        self.create_mesh(N)

        self.D = self.D2(N) / self.h**2
        self.D = self.D.tocsr()
        self.U_next, self.U, self.U_prev = np.zeros((3, N + 1, N + 1))

        self.initialize(N, mx, my)

        if store_data > 0:
            data = {0: self.U_prev.copy()}
        if store_data == 1:
            data[1] = self.U.copy()

        for n in range(1, Nt):
            self.U_next[:] = (
                2 * self.U
                - self.U_prev
                + (self.c * self.dt) ** 2 * (self.D @ self.U + self.U @ (self.D.T))
            )

            self.U_prev[:] = self.U
            self.U[:] = self.U_next

            if store_data > 0 and n % store_data == 0:
                data[n] = self.U_prev.copy()

        if store_data > 0:
            return data

        return self.h, self.l2_error(self.U, Nt * self.dt)

    def convergence_rates(
        self, m: int = 4, cfl: float = 0.1, Nt: int = 10, mx: int = 3, my: int = 3
    ) -> tuple[list[float], np.ndarray, np.ndarray]:
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
        for i in range(m):
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err)
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [np.log(E[i - 1] / E[i]) / np.log(h[i - 1] / h[i]) for i in range(1, m, 1)]
        return r, np.array(E), np.array(h)


class Wave2D_Neumann(Wave2D):

    def D2(self, N: int) -> sparse.dia_matrix:
        """Set up second order differentiation matrix.

        This version includes Neumann boundary conditions

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction

        Returns
        -------
        The differentiation matrix
        """
        D = diags_array(
            [1, -2, 1],
            offsets=[-1, 0, 1],
            shape=(N + 1, N + 1),
            format="lil",
        )
        D[0, 1] = 2
        D[-1, -2] = 2

        return D

    def ue(self, mx: int, my: int) -> sp.Expr:
        """Exact solution with Neumann boundary conditions.

        Parameters
        ----------
        mx, my : int
            Parameters for the wave equation

        Returns
        -------
        The exact solution
        """
        return sp.cos(mx * sp.pi * x) * sp.cos(my * sp.pi * y) * sp.cos(self.w * t)

    def apply_bcs(self) -> None:
        """Apply Neumann boundary conditions."""
        # Neumann boundary conditions are specified directly in D2
        pass


def test_convergence_wave2d() -> None:
    sol = Wave2D()
    r, E, h = sol.convergence_rates(mx=2, my=3)
    assert abs(r[-1] - 2) < 1e-2


def test_convergence_wave2d_neumann() -> None:
    solN = Wave2D_Neumann()
    r, E, h = solN.convergence_rates(mx=2, my=3)
    assert abs(r[-1] - 2) < 0.05


@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 10])
def test_exact_wave2d(m) -> None:
    sol = Wave2D()

    cfl = 1 / np.sqrt(2)
    N = 128
    Nt = 10

    _, E = sol(N, Nt, cfl=cfl, mx=m, my=m, store_data=-1)
    assert E < 1e-12


@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 10])
def test_wave2d_neumann(m: int) -> None:
    sol = Wave2D_Neumann()

    cfl = 1 / np.sqrt(2)
    N = 128
    Nt = 10

    _, E = sol(N, Nt, cfl=cfl, mx=m, my=m, store_data=-1)
    assert E < 1e-12
