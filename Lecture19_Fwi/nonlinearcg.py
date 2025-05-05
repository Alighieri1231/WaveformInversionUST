import jax
import jax.numpy as jnp
from functools import partial
from scipy.io import loadmat
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import numpy as np
from solve_helmholtz import solve_helmholtz


# -----------------------------------------------------------------------------
# 1) Nonlinear Conjugate Gradient (FWI) with correct Fortran‐ordering indexing
# -----------------------------------------------------------------------------
# @partial(jax.jit, static_argnums=(2, 8, 9, 10, 11))
# @partial(jax.jit, static_argnums=(2, 6, 10, 12, 13))  # actualiza según corresponda
def nonlinear_conjugate_gradient(
    xi,
    yi,
    numElements,
    REC_DATA,
    SRC,
    elemInclude,
    tx_include,
    ind_matlab,  # precomputed x_idx*Nyi + y_idx
    c_init,
    f,
    Niter,
    a0,
    L_PML,
    explicit_indices,
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

        jax.debug.print("iter={i} ||WV||={n:.3e}", i=it, n=jnp.linalg.norm(WV))

        ###Error here

        # 1b) estimate source strengths
        SRC_EST = jnp.zeros((len(tx_include),), dtype=jnp.complex64)
        for t in range(len(tx_include)):
            W = WV[:, :, t]
            # flat = W.flatten(order="F")
            flat = W.ravel(order="F")
            # values at each receiver element
            vals = flat[ind_matlab]
            meas = REC_DATA[t, :]
            SRC_EST = SRC_EST.at[t].set(
                jnp.vdot(vals, meas) / (jnp.vdot(vals, vals))  # + 1e-12)
            )
        WV = WV * SRC_EST[jnp.newaxis, jnp.newaxis, :]

        jax.debug.print("iter={i} ||WV src||={n:.3e}", i=it, n=jnp.linalg.norm(WV))

        # 1c) build adjoint sources
        ADJ = jnp.zeros((Nyi * Nxi, len(tx_include)), dtype=jnp.complex64)
        REC_SIM = jnp.zeros((len(tx_include), numElements), dtype=jnp.complex64)
        for t in range(len(tx_include)):
            W = WV[:, :, t]
            # flat = W.flatten(order="F")
            # flat = W.s(order="F")
            flat = W.flatten()
            # simulated rec at included receivers
            grid_inds = ind_matlab[mask_indices[t]]
            sim_vals = flat[grid_inds]
            REC_SIM = REC_SIM.at[t, mask_indices[t]].set(sim_vals)
            diff = sim_vals - REC_DATA[t, mask_indices[t]]
            ADJ = ADJ.at[grid_inds, t].set(diff)
        ADJ_SRC = ADJ.reshape((Nyi, Nxi, len(tx_include)))

        # 1d) virtual source
        VIRT = (2 * (2 * jnp.pi * f) ** 2) * SLOW[:, :, None] * WV

        # 1e) backpropagate

        # because of the error the element is in other part
        ADJ_WV = solve_helmholtz(xi, yi, VEL, ADJ_SRC, f, a0, L_PML, True)

        jax.debug.print("iter={i} ||adj wv||={n:.3e}", i=it, n=jnp.linalg.norm(ADJ_WV))

        BACK = -jnp.real(jnp.conj(VIRT) * ADJ_WV)
        grad = jnp.sum(BACK, axis=2)
        jax.debug.print("iter={i} ||grad||={n:.3e}", i=it, n=jnp.linalg.norm(grad))

        # 2) Conjugate‐gradient update (Hestenes‐Stiefel)
        dg = grad - gprev
        raw_beta = jnp.vdot(grad.conj().T.ravel(order="F"), dg.ravel(order="F")) / (
            jnp.vdot(sd.conj().T.ravel(order="F"), dg.ravel(order="F"))  # + 1e-12
        )

        beta = jax.lax.cond((it == 0), lambda _: 0.0, lambda _: raw_beta, operand=None)

        jax.debug.print(
            "iter={i} grad={g} sd.sum={s:.3e}",
            i=it,
            g=jnp.linalg.norm(grad),
            s=jnp.linalg.norm(sd),
        )

        # print beta sd and grad shape
        jax.debug.print(
            "iter={i} beta={b:.3e} sd.shape={s} grad.shape={g}",
            i=it,
            b=beta,
            s=sd.shape,
            g=grad.shape,
        )

        # check if grad is zero and sd is zero
        # jax.debug.print(jnp.linalg.norm(grad))
        # jax.debug.print(jnp.linalg.norm(sd))
        jax.debug.print(
            "iter={i} grad={g} sd.sum={s:.3e}", i=it, g=jnp.sum(grad), s=jnp.sum(sd)
        )

        sd_new = beta * sd - grad
        # show the sum of sd_new
        jax.debug.print("iter={i} sd_new.sum={s:.3e}", i=it, s=jnp.sum(sd_new))

        # 3) forward project search direction
        PERT = solve_helmholtz(
            xi, yi, VEL, -VIRT * sd_new[:, :, None], f, a0, L_PML, False
        )

        # 4) line search
        dREC = jnp.zeros((len(tx_include), numElements), dtype=jnp.complex64)
        for t in range(len(tx_include)):
            Wp = PERT[:, :, t].flatten()
            vals = Wp[ind_matlab]
            dREC = dREC.at[t, :].set(vals)
        num = jnp.real(
            jnp.vdot(
                dREC.conj().T.ravel(order="F"), (REC_DATA - REC_SIM).ravel(order="F")
            )
        )
        den = jnp.real(jnp.vdot(dREC.conj().T.ravel(order="F"), dREC.ravel(order="F")))
        # num = -(jnp.vdot(grad.conj().T.ravel(order="F"), sd_new.ravel(order="F")))
        step = num / den  # + 1e-12)
        jax.debug.print("num={n:.3e} den={d:.3e}", n=num, d=den)
        jax.debug.print("iter={i} stepSize={s:.3e}", i=it, s=step)

        # 5) update
        SLOW_new = SLOW + step * sd_new
        VEL_new = 1.0 / SLOW_new
        jax.debug.print(
            "iter={i} Vmin/Vmax={vmin:.3e}/{vmax:.3e}",
            i=it,
            vmin=jnp.min(VEL_new),
            vmax=jnp.max(VEL_new),
        )

        return (VEL_new, SLOW_new, sd_new, grad, ADJ_WV, WV), None

    (VEL_F, _, sd_F, grad_F, ADJ_WV, WV), _ = jax.lax.scan(
        body_fun, (VEL, SLOW, sd, gprev, ADJ_WV, WV), jnp.arange(Niter)
    )
    return VEL_F, sd_F, grad_F, ADJ_WV, WV
