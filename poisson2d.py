import numpy as np
import sympy as sp
from scipy import sparse
from scipy.sparse import diags_array, eye_array, kron
from scipy.sparse.linalg import spsolve

x, y = sp.symbols("x,y")


class Poisson2D:
    r"""Solve Poisson's equation in 2D::

        \nabla^2 u(x, y) = f(x, y), in [0, L]^2

    where L is the length of the domain in both x and y directions.
    Dirichlet boundary conditions are used for the entire boundary.
    The Dirichlet values depend on the chosen manufactured solution.

    """

    h: float
    N: int

    def __init__(self, L: float, ue: sp.Function) -> None:
        """Initialize Poisson solver for the method of manufactured solutions

        Parameters
        ----------
        L : number
            The length of the domain in both x and y directions
        ue : Sympy function
            The analytical solution used with the method of manufactured solutions.
            ue is used to compute the right hand side function f.
        """
        self.L = L
        self.ue = ue
        self.f = sp.diff(self.ue, x, 2) + sp.diff(self.ue, y, 2)

    def create_mesh(self, N: int) -> None:
        """Create 2D mesh and store in self.xij and self.yij

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        """
        xi = np.linspace(0, self.L, N + 1)
        yj = np.linspace(0, self.L, N + 1)

        self.xij, self.yij = np.meshgrid(xi, yj, indexing="ij", sparse=True)
        self.h = self.L / N
        self.N = N

    def D2(self) -> sparse.dia_array:
        """Return second order differentiation matrix.

        Returns
        -------
        The differentiation matrix as a sparse DIA matrix
        """
        D = diags_array(
            [1, -2, 1],
            offsets=[-1, 0, 1],
            shape=(self.N + 1, self.N + 1),
            format="lil",
        )
        D[0, :4] = [2, -5, 4, -1]
        D[-1, -4:] = [-1, 4, -5, 2]

        return D

    def laplace(self) -> sparse.csr_array:
        "Return vectorized Laplace operator"
        identity = eye_array(self.N + 1)
        D2 = self.D2()

        D2x = 1.0 / self.h**2 * kron(D2, identity)
        D2y = 1.0 / self.h**2 * kron(identity, D2)

        laplace = D2x + D2y

        return laplace.tocsr()

    def get_boundary_indices(self) -> np.ndarray:
        """Return indices of vectorized matrix that belongs to the boundary"""

        toprow = np.arange(self.N + 1)
        bottomrow = np.arange(self.N + 1) + self.N * (self.N + 1)

        leftcol = np.arange(0, self.N**2 + 1, self.N + 1)
        rightcol = np.arange(self.N, self.N**2 + 1, self.N + 1)

        bnds = np.concatenate([toprow, bottomrow, leftcol, rightcol])

        return bnds

    def assemble(self) -> tuple[sparse.csr_matrix, np.ndarray]:
        """Return assembled matrix A and right hand side vector b"""
        A = self.laplace().tolil()
        bnds = self.get_boundary_indices()
        A[bnds] = 0
        A[bnds, bnds] = 1
        A = A.tocsr()

        F = sp.lambdify((x, y), self.f)(self.xij, self.yij)
        b = F.ravel()
        b[bnds] = 0

        return A, b

    def l2_error(self, u):
        """Return l2-error norm"""
        raise NotImplementedError

    def __call__(self, N: int) -> np.ndarray:
        """Solve Poisson's equation.

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction

        Returns
        -------
        The solution as a Numpy array

        """
        self.create_mesh(N)
        A, b = self.assemble()
        self.U = spsolve(A, b).reshape((N + 1, N + 1))

        return self.U

    def convergence_rates(
        self, m: int = 6
    ) -> tuple[list[float], np.ndarray, np.ndarray]:
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretization levels to use

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
            u = self(N0)
            E.append(self.l2_error(u))
            h.append(self.h)
            N0 *= 2
        r = [
            np.log(E[i - 1] / E[i]) / np.log(h[i - 1] / h[i])
            for i in range(1, m + 1, 1)
        ]
        return r, np.array(E), np.array(h)

    def eval(self, x: float, y: float):
        """Return u(x, y)

        Parameters
        ----------
        x, y : numbers
            The coordinates for evaluation

        Returns
        -------
        The value of u(x, y)

        """
        raise NotImplementedError


def test_convergence_poisson2d() -> None:
    # This exact solution is NOT zero on the entire boundary
    ue = sp.exp(sp.cos(4 * sp.pi * x) * sp.sin(2 * sp.pi * y))
    sol = Poisson2D(1, ue)
    r, *_ = sol.convergence_rates()
    assert abs(r[-1] - 2) < 1e-2


def test_interpolation() -> None:
    ue = sp.exp(sp.cos(4 * sp.pi * x) * sp.sin(2 * sp.pi * y))
    sol = Poisson2D(1, ue)
    sol(100)
    assert abs(sol.eval(0.52, 0.63) - ue.subs({x: 0.52, y: 0.63}).n()) < 1e-3
    assert (
        abs(
            sol.eval(sol.h / 2, 1 - sol.h / 2)
            - ue.subs({x: sol.h, y: 1 - sol.h / 2}).n()
        )
        < 1e-3
    )
