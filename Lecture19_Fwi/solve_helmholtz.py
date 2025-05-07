import jax.numpy as jnp
from jax.scipy.linalg import solve
from jax import jit
import jax
import numpy as np

from jax.experimental.sparse.linalg import spsolve

from scipy.sparse.linalg import spsolve as spsolve_cpu
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from jax.experimental import sparse as jsparse

# time analysis with tic toc
import time

from jax.experimental import host_callback


def scipy_solve(data, indices, indptr, rhs_np, shape):
    NxNy = shape[0]
    mat = csr_matrix((data, indices, indptr), shape=(NxNy, NxNy))
    return spsolve_cpu(mat, rhs_np)


# @jit
# @partial(jax.jit, static_argnames=["assemble_Helmholtz"])
@jit
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

    b, d, e = stencil_opt_params(jnp.min(vel), jnp.max(vel), f, h, g)
    # show

    H_bcoo = assemble_Helmholtz(Nx, Ny, g, b, d, e, h, A, B, C, k)

    H_use = jax.lax.cond(
        adjoint,
        lambda H: jsparse.BCOO(
            (jnp.conj(H.transpose().data), H.transpose().indices), shape=H.shape
        ),
        lambda H: H,
        H_bcoo,
    )

    # 2) Ahora convertimos UNA SOLA VEZ ese BCOO (ya transpuesto/conjugado si tocaba)
    H_use = jsparse.BCSR.from_bcoo(H_use)

    # reshape src to Nx*Ny and -1 to get the right shape
    rhs = jnp.reshape(src, (Nx * Ny, -1))
    rhs = jnp.array(rhs, dtype=jnp.complex64)

    data, indices, indptr = H_use.data, H_use.indices, H_use.indptr

    start = time.time()

    # sol = jnp.stack(
    #     [spsolve(data, indices, indptr, rhs[:, i]) for i in range(rhs.shape[1])], axis=1
    # )
    sol = jax.pure_callback(
        scipy_solve,
        jax.ShapeDtypeStruct((Nx * Ny, rhs.shape[1]), dtype=jnp.complex64),
        data,
        indices,
        indptr,
        rhs,
        (Nx * Ny, Nx * Ny),
    )
    # sol = spsolve(data, indices, indptr, rhs)
    # sol = spsolve(data, indices, indptr, rhs)
    # A = csc_matrix((data, indices, indptr), shape=(Nx * Ny, Nx * Ny))
    # sol = spsolve(H_use, rhs)
    end = time.time()
    print("Time taken to solve system:", end - start)

    return sol.reshape(Ny, Nx, -1)


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
    # G is (10,1) squeeze to enter meshgrid
    G = jnp.squeeze(G)
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


# @jax.jit
def assemble_Helmholtz(Nx, Ny, g, b, d, e, h, A, B, C, k):
    """
    Fully-vectorized assembly of a sparse Helmholtz matrix using a 9-point stencil
    with Dirichlet boundaries, in JAX. Returns a BCOO sparse matrix of shape (Nx*Ny, Nx*Ny).
    Assumes row-major ordering: index = y*Nx + x.
    """

    # Linear index mapping
    def lin_idx(x, y):
        return y * Nx + x

    # 1) Interior grid coordinates
    xs = jnp.arange(1, Nx - 1)
    ys = jnp.arange(1, Ny - 1)
    X, Y = jnp.meshgrid(xs, ys, indexing="xy")  # shape (Ny-2, Nx-2)
    Xf = X.ravel()  # shape (M,)
    Yf = Y.ravel()  # M = (Nx-2)*(Ny-2)

    # 2) Neighbor coordinates for stencil
    Xc, Yc = Xf, Yf
    Xl, Yl = Xf - 1, Yf
    Xr, Yr = Xf + 1, Yf
    Xd, Yd = Xf, Yf - 1
    Xu, Yu = Xf, Yf + 1
    Xdl, Ydl = Xf - 1, Yf - 1
    Xdr, Ydr = Xf + 1, Yf - 1
    Xul, Yul = Xf - 1, Yf + 1
    Xur, Yur = Xf + 1, Yf + 1

    # 3) Linear indices for rows and cols (interior)
    idx_c = lin_idx(Xc, Yc)
    idx_l = lin_idx(Xl, Yl)
    idx_r = lin_idx(Xr, Yr)
    idx_d = lin_idx(Xd, Yd)
    idx_u = lin_idx(Xu, Yu)
    idx_dl = lin_idx(Xdl, Ydl)
    idx_dr = lin_idx(Xdr, Ydr)
    idx_ul = lin_idx(Xul, Yul)
    idx_ur = lin_idx(Xur, Yur)
    rows_int = jnp.repeat(idx_c, 9)
    cols_int = jnp.stack(
        [idx_c, idx_l, idx_r, idx_d, idx_u, idx_dl, idx_dr, idx_ul, idx_ur], axis=1
    ).ravel()

    # 4) Gather neighbor values from A, B, C, k
    A_c = A[Yc, Xc]
    A_l = A[Yl, Xl]
    A_r = A[Yr, Xr]
    A_d = A[Yd, Xd]
    A_u = A[Yu, Xu]
    B_c = B[Yc, Xc]
    B_l = B[Yl, Xl]
    B_r = B[Yr, Xr]
    B_d = B[Yd, Xd]
    B_u = B[Yu, Xu]
    C_c = C[Yc, Xc]
    C_l = C[Yl, Xl]
    C_r = C[Yr, Xr]
    C_d = C[Yd, Xd]
    C_u = C[Yu, Xu]
    k_c = k[Yc, Xc]
    k_l = k[Yl, Xl]
    k_r = k[Yr, Xr]
    k_d = k[Yd, Xd]
    k_u = k[Yu, Xu]

    C_dl = C[Ydl, Xdl]
    k_dl = k[Ydl, Xdl]
    B_dl = B[Ydl, Xdl]
    A_dl = A[Ydl, Xdl]
    C_dr = C[Ydr, Xdr]
    k_dr = k[Ydr, Xdr]
    B_dr = B[Ydr, Xdr]
    A_dr = A[Ydr, Xdr]
    C_ul = C[Yul, Xul]
    k_ul = k[Yul, Xul]
    B_ul = B[Yul, Xul]
    A_ul = A[Yul, Xul]
    C_ur = C[Yur, Xur]
    k_ur = k[Yur, Xur]
    B_ur = B[Yur, Xur]
    A_ur = A[Yur, Xur]

    # 5) Compute stencil values
    val_c = (1 - d - e) * C_c * k_c**2 - b * (
        A_c + A_l + B_c / (g**2) + B_d / (g**2)
    ) / h**2
    val_l = (b * A_l - ((1 - b) / 2) * (B_l / (g**2) + B_dl / (g**2))) / h**2 + (
        d / 4
    ) * C_l * k_l**2
    val_r = (b * A_c - ((1 - b) / 2) * (B_r / (g**2) + B_dr / (g**2))) / h**2 + (
        d / 4
    ) * C_r * k_r**2
    val_d = (b * B_d / (g**2) - ((1 - b) / 2) * (A_d + A_dl)) / h**2 + (
        d / 4
    ) * C_d * k_d**2
    val_u = (b * B_c / (g**2) - ((1 - b) / 2) * (A_u + A_ul)) / h**2 + (
        d / 4
    ) * C_u * k_u**2
    val_dl = ((1 - b) / 2) * (A_dl + B_dl / (g**2)) / h**2 + (e / 4) * C_dl * k_dl**2
    val_dr = ((1 - b) / 2) * (A_dr + B_dr / (g**2)) / h**2 + (e / 4) * C_dr * k_dr**2
    val_ul = ((1 - b) / 2) * (A_ul + B_ul / (g**2)) / h**2 + (e / 4) * C_ul * k_ul**2
    val_ur = ((1 - b) / 2) * (A_ur + B_ur / (g**2)) / h**2 + (e / 4) * C_ur * k_ur**2

    vals_int = jnp.stack(
        [val_c, val_l, val_r, val_d, val_u, val_dl, val_dr, val_ul, val_ur], axis=1
    ).ravel()

    # 6) Dirichlet boundary (value = 1)
    x0 = jnp.arange(Nx)
    y0 = jnp.arange(Ny)
    idx_top = lin_idx(x0, 0)
    idx_bot = lin_idx(x0, Ny - 1)
    idx_left = lin_idx(0, y0[1:-1])
    idx_right = lin_idx(Nx - 1, y0[1:-1])
    idx_bdr = jnp.concatenate([idx_top, idx_bot, idx_left, idx_right])
    rows_bdr = idx_bdr
    cols_bdr = idx_bdr
    vals_bdr = jnp.ones_like(idx_bdr, dtype=A.dtype)

    # 7) Combine interior + boundary
    rows = jnp.concatenate([rows_int, rows_bdr], axis=0)
    cols = jnp.concatenate([cols_int, cols_bdr], axis=0)
    vals = jnp.concatenate([vals_int, vals_bdr], axis=0)
    # change vals type to float32
    # vals = jnp.array(vals, dtype=jnp.float32)
    # check type
    # print("rows cols vals", rows.dtype, cols.dtype, vals.dtype)

    # 8) Assemble sparse COO
    H = jsparse.BCOO((vals, jnp.stack([rows, cols], axis=1)), shape=(Nx * Ny, Nx * Ny))

    return H


# Example usage:
# mat = assemble_Helmholtz(Nx, Ny, g, b, d, e, h, A, B, C, k)


def for_hemholtz(Nx, Ny, g, b, d, e, h, A, B, C, k):
    def lin_idx(x_idx, y_idx):
        return y_idx + Ny * x_idx

    rows = []
    cols = []
    vals = []

    for x in range(Nx):
        for y in range(Ny):
            idx = lin_idx(x, y)

            # --- Dirichlet boundary: M[idx,idx] = 1
            if x == 0 or x == Nx - 1 or y == 0 or y == Ny - 1:
                rows.append(idx)
                cols.append(idx)
                vals.append(1.0)
                continue

            # --- interior: 9‐point stencil
            # center
            A_c = A[y, x]
            A_l = A[y, x - 1]
            B_c = B[y, x]
            B_l = B[y - 1, x]
            C_c = C[y, x]
            k_c = k[y, x]

            val_center = (1 - d - e) * C_c * (k_c**2) - b * (
                A_c + A_l + B_c / (g**2) + B_l / (g**2)
            ) / (h**2)
            rows.append(idx)
            cols.append(idx)
            vals.append(val_center)

            # left
            idx_l = lin_idx(x - 1, y)
            val_l = (
                b * A_l
                - ((1 - b) / 2) * (B[y, x - 1] / (g**2) + B[y - 1, x - 1] / (g**2))
            ) / (h**2) + (d / 4) * C[y, x - 1] * (k[y, x - 1] ** 2)
            rows.append(idx)
            cols.append(idx_l)
            vals.append(val_l)

            # right
            idx_r = lin_idx(x + 1, y)
            val_r = (
                b * A_c
                - ((1 - b) / 2) * (B[y, x + 1] / (g**2) + B[y - 1, x + 1] / (g**2))
            ) / (h**2) + (d / 4) * C[y, x + 1] * (k[y, x + 1] ** 2)
            rows.append(idx)
            cols.append(idx_r)
            vals.append(val_r)

            # down
            idx_d = lin_idx(x, y - 1)
            val_d = (
                b * B[y - 1, x] / (g**2)
                - ((1 - b) / 2) * (A[y - 1, x] + A[y - 1, x - 1])
            ) / (h**2) + (d / 4) * C[y - 1, x] * (k[y - 1, x] ** 2)
            rows.append(idx)
            cols.append(idx_d)
            vals.append(val_d)

            # up
            idx_u = lin_idx(x, y + 1)
            val_u = (
                b * B_c / (g**2) - ((1 - b) / 2) * (A[y + 1, x] + A[y + 1, x - 1])
            ) / (h**2) + (d / 4) * C[y + 1, x] * (k[y + 1, x] ** 2)
            rows.append(idx)
            cols.append(idx_u)
            vals.append(val_u)

            # down-left
            idx_bl = lin_idx(x - 1, y - 1)
            val_bl = ((1 - b) / 2) * (A[y - 1, x - 1] + B[y - 1, x - 1] / (g**2)) / (
                h**2
            ) + (e / 4) * C[y - 1, x - 1] * (k[y - 1, x - 1] ** 2)
            rows.append(idx)
            cols.append(idx_bl)
            vals.append(val_bl)

            # down-right
            idx_br = lin_idx(x + 1, y - 1)
            val_br = ((1 - b) / 2) * (A[y - 1, x] + B[y - 1, x + 1] / (g**2)) / (
                h**2
            ) + (e / 4) * C[y - 1, x + 1] * (k[y - 1, x + 1] ** 2)
            rows.append(idx)
            cols.append(idx_br)
            vals.append(val_br)

            # up-left
            idx_ul = lin_idx(x - 1, y + 1)
            val_ul = ((1 - b) / 2) * (A[y + 1, x - 1] + B[y, x - 1] / (g**2)) / (
                h**2
            ) + (e / 4) * C[y + 1, x - 1] * (k[y + 1, x - 1] ** 2)
            rows.append(idx)
            cols.append(idx_ul)
            vals.append(val_ul)

            # up-right
            idx_ur = lin_idx(x + 1, y + 1)
            val_ur = ((1 - b) / 2) * (A[y + 1, x] + B[y, x + 1] / (g**2)) / (h**2) + (
                e / 4
            ) * C[y + 1, x + 1] * (k[y + 1, x + 1] ** 2)
            rows.append(idx)
            cols.append(idx_ur)
            vals.append(val_ur)

    # turn into JAX arrays
    rows = jnp.array(rows, dtype=jnp.int32)
    cols = jnp.array(cols, dtype=jnp.int32)
    vals = jnp.array(vals)
    print("rows, cols, vals", len(rows), len(cols), len(vals))
    print(Nx, Ny)

    # build a sparse COO:
    H = jsparse.BCOO((vals, jnp.stack([rows, cols], axis=0)), shape=(Nx * Ny, Nx * Ny))

    return H
