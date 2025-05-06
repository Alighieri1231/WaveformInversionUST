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

        # 1b) estimate source strengths
        SRC_EST = jnp.zeros((len(tx_include),), dtype=jnp.complex64)
        for t in range(len(tx_include)):
            W = WV[:, :, t]

            flat = W.ravel(order="F")
            mask = mask_indices[t]  # array de 193 índices 0-based

            REC_SIM = flat[ind_matlab[mask]]  # == REC_SIM(:)

            REC = REC_DATA[t, mask]  # == REC(:)

            # SRC_EST = SRC_EST.at[t].set(
            #     jnp.vdot(REC_SIM.conj().T.ravel(order="F"), REC.ravel(order="F"))
            #     / (jnp.vdot(REC_SIM.conj().T.ravel(order="F"), REC_SIM.ravel(order="F")))  # + 1e-12)
            # )
            SRC_EST = SRC_EST.at[t].set(
                jnp.vdot(REC_SIM.ravel(order="F"), REC.ravel(order="F"))
                / (
                    jnp.vdot(REC_SIM.ravel(order="F"), REC_SIM.ravel(order="F"))
                )  # + 1e-12)
            )
        # print norm of SRC_EST
        jax.debug.print(
            "iter={i} ||SRC_EST||={n:.3e}", i=it, n=jnp.linalg.norm(SRC_EST)
        )

        WV = WV * SRC_EST[jnp.newaxis, jnp.newaxis, :]

        jax.debug.print("iter={i} ||WV src||={n:.3e}", i=it, n=jnp.linalg.norm(WV))

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

        jax.debug.print(
            "iter={i} REC shape={r} REC_SIM shape={s}",
            i=it,
            r=REC.shape,
            s=REC_SIM.shape,
        )

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
        # raw_beta = jnp.vdot(grad.conj().T.ravel(order="F"), dg.ravel(order="F")) / (
        #     jnp.vdot(sd.conj().T.ravel(order="F"), dg.ravel(order="F"))  # + 1e-12
        # )
        raw_beta = jnp.vdot(grad.ravel(order="F"), dg.ravel(order="F")) / (
            jnp.vdot(sd.ravel(order="F"), dg.ravel(order="F"))  # + 1e-12
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
        jax.debug.print(
            "iter={i} sd_new={s:.3e}", i=it, s=jnp.linalg.norm(sd_new.flatten())
        )

        # 3) forward project search direction
        PERT = solve_helmholtz(
            xi, yi, VEL, -VIRT * sd_new[:, :, None], f, a0, L_PML, False
        )

        jax.debug.print("iter={i} ||PERT||={n:.3e}", i=it, n=jnp.linalg.norm(PERT))

        # 4) line search
        dREC = jnp.zeros((len(tx_include), numElements), dtype=jnp.complex64)
        # for t in range(len(tx_include)):
        #     # Wp = PERT[:, :, t].flatten()
        #     Wp = PERT[:, :, t].ravel(order="F")
        #     vals = Wp[ind_matlab]
        #     dREC = dREC.at[t, :].set(vals)
        for t in range(len(tx_include)):
            # flatten in column-major order
            Wp = PERT[:, :, t].ravel(order="F")

            # restrict to the included receivers
            mask = mask_indices[t]  # 0-based indices of included elements
            vals = Wp[ind_matlab[mask]]  # simulated search-direction data

            # write only into those positions
            dREC = dREC.at[t, mask].set(vals)

        # num = jnp.real(
        #     jnp.vdot(
        #         dREC.conj().T.ravel(order="F"), (REC_DATA - REC_SIM).ravel(order="F")
        #     )
        # )
        # den = jnp.real(jnp.vdot(dREC.conj().T.ravel(order="F"), dREC.ravel(order="F")))
        num = jnp.real(
            jnp.vdot(dREC.ravel(order="F"), (REC_DATA - REC_SIM).ravel(order="F"))
        )
        den = jnp.real(jnp.vdot(dREC.ravel(order="F"), dREC.ravel(order="F")))
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
