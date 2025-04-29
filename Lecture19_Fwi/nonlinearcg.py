import jax
import jax.numpy as jnp
from functools import partial
from solve_helmholtz import solve_helmholtz  # Assuming this is the correct import

# Assume solveHelmholtz is already defined and JAX-compatible
# solveHelmholtz(xi, yi, VEL_ESTIM, SRC, f, a0, L_PML, adjoint)


# @partial(jax.jit, static_argnums=(8, 9))
# jit full no partial
# @jax.jit
@partial(jax.jit, static_argnums=(2, 8, 9, 10, 11))
def nonlinear_conjugate_gradient(
    xi,  # 0
    yi,  # 1
    numElements,  # 2
    REC_DATA,  # 3
    SRC,  # 4
    elemInclude,  # 5
    tx_include,  # 6
    ind,  # 7
    c_init,  # 8 -> STATIC!
    f,  # 9 -> STATIC!
    Niter,  # 10 -> STATIC! [important for jnp.arange()]
    a0,  # 11
    L_PML,  # 12
    explicit_indices,  # 13
    mask_indices,  # 14
):
    Nyi, Nxi = yi.size, xi.size

    search_dir = jnp.zeros((Nyi, Nxi))
    gradient_img_prev = jnp.zeros((Nyi, Nxi))
    VEL_ESTIM = c_init * jnp.ones((Nyi, Nxi))
    SLOW_ESTIM = 1.0 / VEL_ESTIM

    def body_fun(state, iter_idx):
        VEL_ESTIM, SLOW_ESTIM, search_dir, gradient_img_prev = state
        print("Iteration:", iter_idx)

        # Step 1a: Gradient Calculation )
        WVFIELD = solve_helmholtz(xi, yi, VEL_ESTIM, SRC, f, a0, L_PML, False)

        # Step 1b: Estimate forward sources and adjust simulated fields
        SRC_ESTIM = jnp.zeros((len(tx_include),))
        for tx_elmt_idx in range(len(tx_include)):
            # logic 2
            WVFIELD_elmt = WVFIELD[:, :, tx_elmt_idx]
            REC_SIM = jnp.take(WVFIELD_elmt.reshape(-1), explicit_indices[tx_elmt_idx])
            REC = jnp.take(REC_DATA[tx_elmt_idx, :], mask_indices[tx_elmt_idx])

            # original from matlab
            # WVFIELD_elmt = WVFIELD[:, :, tx_elmt_idx]
            # REC_SIM = WVFIELD_elmt[ind(elemInclude[tx_include[tx_elmt_idx], :])]
            # REC = REC_DATA[tx_elmt_idx, elemInclude[tx_include[tx_elmt_idx], :]]
            SRC_ESTIM = SRC_ESTIM.at[tx_elmt_idx].set(
                jnp.vdot(REC_SIM, REC) / jnp.vdot(REC_SIM, REC_SIM)
            )

        WVFIELD = WVFIELD * SRC_ESTIM.reshape(1, 1, -1)

        # Step 1c: Build adjoint sources based on errors
        ADJ_SRC = jnp.zeros((Nyi, Nxi, len(tx_include)))
        REC_SIM = jnp.zeros((len(tx_include), numElements))

        for tx_elmt_idx in range(len(tx_include)):
            WVFIELD_elmt = WVFIELD[:, :, tx_elmt_idx]

            # original
            # REC_SIM = REC_SIM.at[
            #     tx_elmt_idx, elemInclude[tx_include[tx_elmt_idx], :]
            # ].set(WVFIELD_elmt[ind(elemInclude[tx_include[tx_elmt_idx], :])])
            # diff = (
            #     REC_SIM[tx_elmt_idx, elemInclude[tx_include[tx_elmt_idx], :]]
            #     - REC_DATA[tx_elmt_idx, elemInclude[tx_include[tx_elmt_idx], :]]
            # )

            # adj_src_update = (
            #     jnp.zeros((Nyi, Nxi))
            #     .at[ind(elemInclude[tx_include[tx_elmt_idx], :])]
            #     .set(diff)
            # )

            # changed
            REC_SIM = REC_SIM.at[tx_elmt_idx, explicit_indices[tx_elmt_idx]].set(
                jnp.take(WVFIELD_elmt.reshape(-1), explicit_indices[tx_elmt_idx])
            )

            diff = REC_SIM[tx_elmt_idx, explicit_indices[tx_elmt_idx]] - jnp.take(
                REC_DATA[tx_elmt_idx, :], mask_indices[tx_elmt_idx]
            )

            adj_src_update = (
                jnp.zeros((Nyi * Nxi,)).at[explicit_indices[tx_elmt_idx]].set(diff)
            )
            adj_src_update = adj_src_update.reshape((Nyi, Nxi))

            ADJ_SRC = ADJ_SRC.at[:, :, tx_elmt_idx].set(adj_src_update)

        # Step 1d: Calculate virtual sources
        # VIRT_SRC = (2 * (2 * jnp.pi * f) ** 2) * SLOW_ESTIM * WVFIELD
        VIRT_SRC = (2 * (2 * jnp.pi * f) ** 2) * SLOW_ESTIM[:, :, None] * WVFIELD

        # Step 1e: Backproject errors to get gradient
        ADJ_WVFIELD = solve_helmholtz(xi, yi, VEL_ESTIM, ADJ_SRC, f, a0, L_PML, True)
        BACKPROJ = -jnp.real(jnp.conj(VIRT_SRC) * ADJ_WVFIELD)
        gradient_img = jnp.sum(BACKPROJ, axis=2)

        # Step 2a: Hestenes-Stiefel Conjugate Gradient Beta
        diff_grad = gradient_img - gradient_img_prev
        beta_num = jnp.vdot(gradient_img.ravel(), diff_grad.ravel())
        beta_den = jnp.vdot(search_dir.ravel(), diff_grad.ravel())
        beta = beta_num / (beta_den + 1e-10)

        # Step 2b: Update Search Direction
        search_dir_new = beta * search_dir - gradient_img

        # Step 3: Forward Projection of Search Direction
        # PERTURBED_WVFIELD = solve_helmholtz(
        #     xi, yi, VEL_ESTIM, -VIRT_SRC * search_dir_new, f, a0, L_PML, False
        # )
        PERTURBED_WVFIELD = solve_helmholtz(
            xi,
            yi,
            VEL_ESTIM,
            -VIRT_SRC * search_dir_new[:, :, None],
            f,
            a0,
            L_PML,
            False,
        )

        dREC_SIM = jnp.zeros((len(tx_include), numElements))

        for tx_elmt_idx in range(len(tx_include)):
            PERTURBED_WVFIELD_elmt = PERTURBED_WVFIELD[:, :, tx_elmt_idx]
            # dREC_SIM = dREC_SIM.at[
            #     tx_elmt_idx, elemInclude[tx_include[tx_elmt_idx], :]
            # ].set(PERTURBED_WVFIELD_elmt[ind(elemInclude[tx_include[tx_elmt_idx], :])])
            dREC_SIM = dREC_SIM.at[tx_elmt_idx, explicit_indices[tx_elmt_idx]].set(
                jnp.take(
                    PERTURBED_WVFIELD_elmt.reshape(-1), explicit_indices[tx_elmt_idx]
                )
            )

        # Step 5: Line Search
        # step option 1
        stepSize = jnp.real(
            jnp.vdot(dREC_SIM.ravel(), (REC_DATA - REC_SIM).ravel())
        ) / (jnp.vdot(dREC_SIM.ravel(), dREC_SIM.ravel()) + 1e-10)

        # Step 6: Update Estimates
        SLOW_ESTIM_new = SLOW_ESTIM + stepSize * search_dir_new
        VEL_ESTIM_new = 1.0 / jnp.real(SLOW_ESTIM_new)

        return (VEL_ESTIM_new, SLOW_ESTIM_new, search_dir_new, gradient_img), None

    init_state = (VEL_ESTIM, SLOW_ESTIM, search_dir, gradient_img_prev)
    final_state, _ = jax.lax.scan(body_fun, init_state, jnp.arange(Niter))

    VEL_ESTIM_final, _, search_dir_final, gradient_img_final = final_state
    return VEL_ESTIM_final, search_dir_final, gradient_img_final
