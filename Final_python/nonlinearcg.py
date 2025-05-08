import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from scipy.io import loadmat
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import numpy as np
from solve_helmholtz import solve_helmholtz
from jax import vmap
import time


@jit
def estimate_src_strength(REC_SIM, REC):
    return jnp.vdot(REC_SIM.ravel(order="F"), REC.ravel(order="F")) / (
        jnp.vdot(REC_SIM.ravel(order="F"), REC_SIM.ravel(order="F"))
    )

estimate_src_strength_batched = vmap(estimate_src_strength, in_axes=(0, 0))

@jit
def compute_step_size(dREC, REC_DATA, REC_SIM, SLOW, sd_new):
    num = jnp.real(
        jnp.vdot(dREC.ravel(order="F"), (REC_DATA - REC_SIM).ravel(order="F"))
    )
    den = jnp.real(jnp.vdot(dREC.ravel(order="F"), dREC.ravel(order="F")))
    step = num / den  # + 1e-12)
    SLOW_new = SLOW + step * sd_new
    VEL_new = 1.0 / SLOW_new

    return VEL_new, SLOW_new



# -----------------------------------------------------------------------------
# 1) Nonlinear Conjugate Gradient (FWI) with correct Fortran‐ordering indexing
# -----------------------------------------------------------------------------


def nonlinear_conjugate_gradient(
    xi,
    yi,
    numElements,
    REC_DATA,
    SRC,
    tx_include,
    ind_matlab,  # precomputed x_idx*Nyi + y_idx
    c_init,
    f,
    Niter,
    a0,
    L_PML,
    mask_indices,
):
    Nyi, Nxi = yi.size, xi.size
    t5 = time.time()
    # Initialize state
    VEL = c_init * jnp.ones((Nyi, Nxi))
    SLOW = 1.0 / VEL
    sd = jnp.zeros((Nyi, Nxi))
    gprev = jnp.zeros((Nyi, Nxi))
    ADJ_WV = jnp.zeros((Nyi, Nxi, len(tx_include)), dtype=jnp.complex64)
    WV = jnp.zeros((Nyi, Nxi, len(tx_include)), dtype=jnp.complex64)
    t6 = time.time() - t5
    jax.debug.print(
        "Initialization time: {t:.2f} seconds",
        t=t6,
    )

    def body_fun(state, it):
        VEL, SLOW, sd, gprev, ADJ_WV, WV = state
        t3 = time.time()

        # 1a) forward Helmholtz
        WV = solve_helmholtz(xi, yi, VEL, SRC, f, a0, L_PML, False)

        # 1b) estimate source strengths
        SRC_EST = jnp.zeros((len(tx_include),), dtype=jnp.complex64)
        for t in range(len(tx_include)):
            W = WV[:, :, t]

            flat = W.ravel(order="F")
            mask = mask_indices[t]  # array de 193 índices 0-based
            REC_SIM = flat[ind_matlab[mask]]  # == REC_SIM(:)
            REC = REC_DATA[t, mask]  # == REC(:)
            SRC_EST = SRC_EST.at[t].set(estimate_src_strength(REC_SIM, REC))

        WV = WV * SRC_EST[jnp.newaxis, jnp.newaxis, :]

        # 1c) build adjoint sources
        # Pre-allocate exactly like MATLAB
        ADJ_SRC = jnp.zeros((Nyi, Nxi, len(tx_include)), dtype=jnp.complex64)
        REC_SIM = jnp.zeros((len(tx_include), numElements), dtype=jnp.complex64)

        for t in range(len(tx_include)):
            # 1) Flatten in Fortran (column-major) order
            W = WV[:, :, t]
            flat_W = W.ravel(order="F")
            mask = mask_indices[t]  # 0-based receiver indices

            # 2) Gather simulated data into REC_SIM exactly like MATLAB
            REC_SIM = REC_SIM.at[t, mask].set(flat_W[ind_matlab[mask]])

            # 3) Compute difference
            diff = REC_SIM[t, mask] - REC_DATA[t, mask]

            # 4) Build adj_src_elmt and use the same 'ind' indexing inside
            #    just like MATLAB’s
            adj_src_elmt = jnp.zeros((Nyi, Nxi), dtype=jnp.complex64)
            flat_adj = adj_src_elmt.ravel(order="F")
            flat_adj = flat_adj.at[ind_matlab[mask]].set(diff)
            adj_src_elmt = flat_adj.reshape((Nyi, Nxi), order="F")

            # 5) Store into the 3D ADJ_SRC volume
            ADJ_SRC = ADJ_SRC.at[:, :, t].set(adj_src_elmt)

        # ------------------------------------------------
        # 1d) virtual source
        VIRT = (2 * (2 * jnp.pi * f) ** 2) * SLOW[:, :, None] * WV

        # 1e) backpropagate

        # because of the error the element is in other part
        ADJ_WV = solve_helmholtz(xi, yi, VEL, ADJ_SRC, f, a0, L_PML, True)
        BACK = -jnp.real(jnp.conj(VIRT) * ADJ_WV)
        grad = jnp.sum(BACK, axis=2)

        # 2) Conjugate‐gradient update (Hestenes‐Stiefel)
        dg = grad - gprev

        raw_beta = jnp.vdot(grad.ravel(order="F"), dg.ravel(order="F")) / (
            jnp.vdot(sd.ravel(order="F"), dg.ravel(order="F"))  # + 1e-12
        )

        beta = jax.lax.cond((it == 0), lambda _: 0.0, lambda _: raw_beta, operand=None)

        sd_new = beta * sd - grad

        # 3) forward project search direction
        PERT = solve_helmholtz(
            xi, yi, VEL, -VIRT * sd_new[:, :, None], f, a0, L_PML, False
        )

        # 4) line search
        dREC = jnp.zeros((len(tx_include), numElements), dtype=jnp.complex64)

        for t in range(len(tx_include)):
            # flatten in column-major order
            Wp = PERT[:, :, t].ravel(order="F")
            # restrict to the included receivers
            mask = mask_indices[t]  # 0-based indices of included elements
            vals = Wp[ind_matlab[mask]]  # simulated search-direction data

            # write only into those positions
            dREC = dREC.at[t, mask].set(vals)

        # step = compute_step_size(dREC, REC_DATA, REC_SIM)
        VEL_new, SLOW_new = compute_step_size(dREC, REC_DATA, REC_SIM, SLOW, sd_new)

        t4 = time.time() - t3

        jax.debug.print(
            "Iteration {it}: {t:.2f} seconds",
            it=it,
            t=t4,
        )

        return (VEL_new, SLOW_new, sd_new, grad, ADJ_WV, WV), None

    t7 = time.time()
    (VEL_F, _, sd_F, grad_F, ADJ_WV, WV), _ = jax.lax.scan(
        body_fun, (VEL, SLOW, sd, gprev, ADJ_WV, WV), jnp.arange(Niter)
    )
    t8 = time.time() - t7
    jax.debug.print(
        "Compile: {t:.2f} seconds",
        t=t8,
    )
    return VEL_F, sd_F, grad_F, ADJ_WV, WV


# @jit
def nonlinear_conjugate_gradient_vectorized(
    xi,
    yi,
    numElements,
    REC_DATA,
    SRC,
    tx_include,
    ind_matlab,  # precomputed x_idx*Nyi + y_idx
    c_init,
    f,
    Niter,
    a0,
    L_PML,
    mask_indices,
):
    Nyi, Nxi = yi.size, xi.size

    # Initialize state
    VEL = c_init * jnp.ones((Nyi, Nxi))
    SLOW = 1.0 / VEL
    sd = jnp.zeros((Nyi, Nxi))
    gprev = jnp.zeros((Nyi, Nxi))
    ADJ_WV = jnp.zeros((Nyi, Nxi, len(tx_include)), dtype=jnp.complex64)
    WV = jnp.zeros((Nyi, Nxi, len(tx_include)), dtype=jnp.complex64)

    def body_fun(state, it):
        VEL, SLOW, sd, gprev, ADJ_WV, WV = state

        # 1a) forward Helmholtz
        WV = solve_helmholtz(xi, yi, VEL, SRC, f, a0, L_PML, False)

        # --- Vectorization ---
        N1, N2, Nt = WV.shape
        Nflat = N1 * N2

        # 1b) estimate source strengths
        flat_WV = jnp.reshape(jnp.transpose(WV, (1, 0, 2)), (N1 * N2, Nt))
        global_inds = jnp.take(ind_matlab, mask_indices)
        rec_sim = jnp.take_along_axis(flat_WV.T, global_inds, axis=1)
        rec = jnp.take_along_axis(REC_DATA, mask_indices, axis=1)
        SRC_EST = estimate_src_strength_batched(rec_sim, rec)  # shape (Nt,)

        # 5) Update WV with the estimated source strengths
        WV = WV * SRC_EST[None, None, :]

        # 1) Flatten en Fortran-order (column-major) todas las W[:, :, t]
        flat_WV = jnp.reshape(
            jnp.transpose(WV, (1, 0, 2)),  # (Nxi, Nyi, Nt)
            (Nflat, Nt),  # (Nflat, Nt)
        )

        # 2) Índices globales de receptores por tiro: shape (Nt, Nmask)
        global_inds = jnp.take(ind_matlab, mask_indices)

        # 3) Gather simultáneo de simulados y observados: (Nt, Nmask)
        rec_sim = jnp.take_along_axis(flat_WV.T, global_inds, axis=1)
        rec_obs = jnp.take_along_axis(REC_DATA, mask_indices, axis=1)

        # 4) Construcción vectorizada de REC_SIM completo (Nt × numElements)
        REC_SIM = jnp.zeros((Nt, numElements), dtype=jnp.complex64)
        batch_idx = jnp.arange(Nt)[:, None]  # (Nt,1)
        REC_SIM = REC_SIM.at[batch_idx, mask_indices].set(rec_sim)

        # 5) Residual y scatter en flat_adj
        diff = rec_sim - rec_obs  # (Nt, Nmask)
        flat_adj = jnp.zeros((Nt, Nflat), dtype=jnp.complex64)
        flat_adj = flat_adj.at[batch_idx, global_inds].set(diff)

        # 6) Reconstrucción de ADJ_SRC en (Nyi, Nxi, Nt)
        adj_tmp = flat_adj.reshape((Nt, Nxi, Nyi))  # (Nt, Nxi, Nyi)
        ADJ_SRC = jnp.transpose(adj_tmp, (2, 1, 0))  # (Nyi, Nxi, Nt)

        # ------------------------------------------------
        # 1d) virtual source
        VIRT = (2 * (2 * jnp.pi * f) ** 2) * SLOW[:, :, None] * WV

        # 1e) backpropagate

        # because of the error the element is in other part
        ADJ_WV = solve_helmholtz(xi, yi, VEL, ADJ_SRC, f, a0, L_PML, True)
        BACK = -jnp.real(jnp.conj(VIRT) * ADJ_WV)
        grad = jnp.sum(BACK, axis=2)

        # 2) Conjugate‐gradient update (Hestenes‐Stiefel)
        dg = grad - gprev

        raw_beta = jnp.vdot(grad.ravel(order="F"), dg.ravel(order="F")) / (
            jnp.vdot(sd.ravel(order="F"), dg.ravel(order="F"))  # + 1e-12
        )

        beta = jax.lax.cond((it == 0), lambda _: 0.0, lambda _: raw_beta, operand=None)

        sd_new = beta * sd - grad

        # 3) forward project search direction
        PERT = solve_helmholtz(
            xi, yi, VEL, -VIRT * sd_new[:, :, None], f, a0, L_PML, False
        )

        # # 4) line search
        flat_PERT = jnp.reshape(
            jnp.transpose(PERT, (1, 0, 2)),  # (Nxi, Nyi, Nt)
            (Nflat, Nt),  # (Nflat, Nt)
        )

        # 3) Índices globales de receptores por tiro (shape: Nt × Nmask)
        global_inds = jnp.take(ind_matlab, mask_indices)

        # 4) Extrae de una vez los valores simulados de búsqueda (Nt × Nmask)
        rec_vals = jnp.take_along_axis(flat_PERT.T, global_inds, axis=1)

        # 5) Scatter de rec_vals en dREC (Nt × numElements)
        batch_idx = jnp.arange(Nt)[:, None]  # (Nt,1)
        dREC = jnp.zeros((Nt, numElements), dtype=jnp.complex64)
        dREC = dREC.at[batch_idx, mask_indices].set(rec_vals)

        # step = compute_step_size(dREC, REC_DATA, REC_SIM)
        VEL_new, SLOW_new = compute_step_size(dREC, REC_DATA, REC_SIM, SLOW, sd_new)

        return (VEL_new, SLOW_new, sd_new, grad, ADJ_WV, WV), None

    (VEL_F, _, sd_F, grad_F, ADJ_WV, WV), _ = jax.lax.scan(
        body_fun, (VEL, SLOW, sd, gprev, ADJ_WV, WV), jnp.arange(Niter)
    )
    return VEL_F, sd_F, grad_F, ADJ_WV, WV
