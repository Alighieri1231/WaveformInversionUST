import jax.numpy as jnp
from jax.scipy.linalg import solve
from jax import jit
from jax.experimental.sparse.linalg import spsolve
from jax.experimental import sparse as jsparse


def solve_helmholtz(x, y, vel, src, f, a0, L_PML, adjoint):
    sign_convention = -1
    h = jnp.mean(jnp.diff(x))
    gh = jnp.mean(jnp.diff(y))
    g = gh / h
    Nx, Ny = x.size, y.size
    k = 2 * jnp.pi * f / vel

    # PML definitions
    xmin, xmax = x[0], x[-1]
    ymin, ymax = y[0], y[-1]
    xe = jnp.linspace(xmin, xmax, 2 * (Nx - 1) + 1)
    ye = jnp.linspace(ymin, ymax, 2 * (Ny - 1) + 1)
    Xe, Ye = jnp.meshgrid(xe, ye, indexing="xy")

    xctr, xspan = (xmin + xmax) / 2, (xmax - xmin) / 2
    yctr, yspan = (ymin + ymax) / 2, (ymax - ymin) / 2

    sx = (
        2
        * jnp.pi
        * a0
        * f
        * ((jnp.maximum(jnp.abs(Xe - xctr) - xspan + L_PML, 0) / L_PML) ** 2)
    )
    sy = (
        2
        * jnp.pi
        * a0
        * f
        * ((jnp.maximum(jnp.abs(Ye - yctr) - yspan + L_PML, 0) / L_PML) ** 2)
    )

    ex = 1 + 1j * sx * jnp.sign(sign_convention) / (2 * jnp.pi * f)
    ey = 1 + 1j * sy * jnp.sign(sign_convention) / (2 * jnp.pi * f)

    A = (ey / ex)[::2, 1::2]
    B = (ex / ey)[1::2, ::2]
    C = (ex * ey)[::2, ::2]

    def lin_idx(x_idx, y_idx):
        return y_idx + Ny * x_idx

    b, d, e = stencil_opt_params(jnp.min(vel), jnp.max(vel), f, h, g)

    rows, cols, vals = [], [], []

    for x_idx in range(1, Nx - 1):
        for y_idx in range(1, Ny - 1):
            idx = lin_idx(x_idx, y_idx)

            # Up
            idx_up = lin_idx(x_idx, y_idx + 1)
            A_up = A[y_idx + 1, x_idx]
            A_up_left = A[y_idx + 1, x_idx - 1]
            C_up = C[y_idx + 1, x_idx]
            k_up = k[y_idx + 1, x_idx]
            B_up = B[y_idx, x_idx]
            val_up = (b * B_up / (g**2) - ((1 - b) / 2) * (A_up + A_up_left)) / (
                h**2
            ) + (d / 4) * C_up * (k_up**2)
            rows.append(idx)
            cols.append(idx_up)
            vals.append(val_up)

            # Top Left
            idx_tl = lin_idx(x_idx - 1, y_idx + 1)
            A_tl = A[y_idx + 1, x_idx - 1]
            B_tl = B[y_idx, x_idx - 1]
            C_tl = C[y_idx + 1, x_idx - 1]
            k_tl = k[y_idx + 1, x_idx - 1]
            val_tl = ((1 - b) / 2) * (A_tl + B_tl / (g**2)) / (h**2) + (
                e / 4
            ) * C_tl * (k_tl**2)
            rows.append(idx)
            cols.append(idx_tl)
            vals.append(val_tl)

            # Top Right
            idx_tr = lin_idx(x_idx + 1, y_idx + 1)
            A_tr = A[y_idx + 1, x_idx]
            B_tr = B[y_idx, x_idx + 1]
            C_tr = C[y_idx + 1, x_idx + 1]
            k_tr = k[y_idx + 1, x_idx + 1]
            val_tr = ((1 - b) / 2) * (A_tr + B_tr / (g**2)) / (h**2) + (
                e / 4
            ) * C_tr * (k_tr**2)
            rows.append(idx)
            cols.append(idx_tr)
            vals.append(val_tr)

    H = jsparse.BCOO(
        (jnp.array(vals), jnp.stack([jnp.array(rows), jnp.array(cols)])),
        shape=(Nx * Ny, Nx * Ny),
    )
    rhs = src.reshape(Nx * Ny)
    sol = spsolve(H.T.conj() if adjoint else H, rhs)
    return sol.reshape(Ny, Nx)


def stencil_opt_params(vmin, vmax, f, h, g):
    """
    Optimal parameters for the 9-point stencil Helmholtz discretization.

    Based on: Chen/Cheng/Feng/Wu 2013.

    Parameters:
        vmin: minimum wave velocity [L/T]
        vmax: maximum wave velocity [L/T]
        f: frequency [1/T]
        h: grid spacing in x-direction [L]
        g: dy/h (i.e., anisotropy ratio)

    Returns:
        b, d, e: optimal parameters for the 9-point stencil
    """
    l = 100  # angular resolution
    r = 10  # range resolution
    Gmin = vmin / (f * h)
    Gmax = vmax / (f * h)

    m = jnp.arange(1, l + 1)
    n = jnp.arange(1, r + 1)
    theta = (m - 1) * jnp.pi / (4 * (l - 1))
    G = 1 / (1 / Gmax + (n[:, None] - 1) / (r - 1) * (1 / Gmin - 1 / Gmax))

    TH, GG = jnp.meshgrid(theta, G)

    P = jnp.cos(g * 2 * jnp.pi * jnp.cos(TH) / GG)
    Q = jnp.cos(2 * jnp.pi * jnp.sin(TH) / GG)

    S1 = (1 + 1 / g**2) * GG**2 * (1 - P - Q + P * Q)
    S2 = jnp.pi**2 * (2 - P - Q)
    S3 = 2 * jnp.pi**2 * (1 - P * Q)
    S4 = 2 * jnp.pi**2 + GG**2 * ((1 + 1 / g**2) * P * Q - P - Q / g**2)

    fix_b = True
    if fix_b:
        b = 5 / 6
        A = jnp.stack([S2.ravel(), S3.ravel()], axis=1)
        y = S4.ravel() - b * S1.ravel()
        params = solve(A.T @ A, A.T @ y)
        d, e = params[0], params[1]
    else:
        A = jnp.stack([S1.ravel(), S2.ravel(), S3.ravel()], axis=1)
        y = S4.ravel()
        params = solve(A.T @ A, A.T @ y)
        b, d, e = params[0], params[1], params[2]

    return b, d, e
