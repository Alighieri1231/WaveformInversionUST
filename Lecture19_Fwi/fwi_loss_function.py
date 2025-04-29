import jax.numpy as jnp
from solve_helmholtz import solve_helmholtz


def fwi_loss_function(
    slow, xi, yi, SRC, f, a0, L_PML, REC_DATA, ind, elemInclude, tx_include, numElements
):
    """
    Computes the FWI data fidelity loss: L = 0.5 * ||REC_SIM - REC_DATA||^2

    Args:
        slow: Slowness image (1/velocity), shape (Nyi, Nxi)
        xi, yi: Spatial grid coordinates
        SRC: Source matrix of shape (Nyi, Nxi, num_tx)
        f: Frequency [Hz]
        a0, L_PML: PML parameters
        REC_DATA: Measured data matrix, shape (num_tx, num_elements)
        ind: Flattened indices of receivers on the grid
        elemInclude: Boolean matrix [numElements x numElements]
        tx_include: Indices of transmitters to include
        numElements: Total number of elements

    Returns:
        Scalar loss value
    """
    VEL_ESTIM = 1.0 / slow
    # (1A) Forward wavefield
    WVFIELD = solve_helmholtz(xi, yi, VEL_ESTIM, SRC, f, a0, L_PML, adjoint=False)
    print("WVFIELD shape:", WVFIELD.shape)
    print("WVFIELD:type", WVFIELD.dtype)
    # (1B) Estimate sources (per tx)
    REC_SIM = jnp.zeros((tx_include.size, numElements))
    SRC_ESTIM = jnp.zeros(tx_include.size)

    for i, tx_idx in enumerate(tx_include):
        u = WVFIELD[:, :, i]
        rcv_idx = elemInclude[tx_idx, :]
        rec_sim = u.flatten()[ind[rcv_idx]]
        rec_data = REC_DATA[i, rcv_idx]
        src_scale = jnp.vdot(rec_sim, rec_data) / jnp.vdot(rec_sim, rec_sim)
        SRC_ESTIM = SRC_ESTIM.at[i].set(src_scale)
        REC_SIM = REC_SIM.at[i, rcv_idx].set(src_scale * rec_sim)

    # (2) Data fidelity loss
    loss = 0.5 * jnp.sum((REC_SIM - REC_DATA) ** 2)
    print("Loss dtype: ", loss.shape)
    print("Loss shape: ", loss.shape)

    print("REC_SIM shape:", REC_SIM.shape)
    print("REC_SIM dtype:", REC_SIM.dtype)

    print("REC_DATA shape:", REC_DATA.shape)
    print("REC_DATA dtype:", REC_DATA.dtype)
    print("SRC_ESTIM shape:", SRC_ESTIM.shape)
    print("SRC_ESTIM dtype:", SRC_ESTIM.dtype)

    return loss
