import jax
import jax.numpy as jnp
from jaxopt import LBFGS
from functools import partial
import mat73
import matplotlib.pyplot as plt
from solve_helmholtz import solve_helmholtz
import time
from jax import lax

import jax
import jax.numpy as jnp
from functools import partial
from solve_helmholtz import solve_helmholtz


# your original single-shot estimator
@jax.jit
def estimate_src_strength(REC_SIM, REC):
    return jnp.vdot(REC_SIM.ravel(order="F"), REC.ravel(order="F")) / (
        jnp.vdot(REC_SIM.ravel(order="F"), REC_SIM.ravel(order="F"))
    )


# batch it over the first axis (transmitters)
estimate_src_strength_batched = jax.vmap(estimate_src_strength, in_axes=(0, 0))


def fwi_loss_function(
    params,
    xi,
    yi,
    REC_DATA,  # shape (Nt, num_elements)
    SRC,
    f,
    a0,
    L_PML,
    tx_include,
    ind_matlab,  # shape (num_elements,)
    mask_indices,  # **must** be an integer array of shape (Nt, Nmask)
    num_elements,
):
    # grid sizes
    Nyi, Nxi = yi.size, xi.size
    Nt = tx_include.size
    Nflat = Nyi * Nxi

    # build slowness and velocity
    SLOW = params.reshape((Nyi, Nxi))
    VEL = 1.0 / SLOW

    # one-shot forward
    WV = solve_helmholtz(xi, yi, VEL, SRC, f, a0, L_PML, False)
    # WV.shape == (Nyi, Nxi, Nt)

    # flatten each 2D field into a (Nflat,) vector, then stack into (Nt, Nflat)
    flat_WV = (
        jnp.transpose(WV, (2, 0, 1)).reshape(Nt, Nflat)  # (Nt, Nyi, Nxi)  # (Nt, Nflat)
    )

    # --- Vectorización de la estimación de fuentes ---
    N1, N2, Nt = WV.shape
    Nflat = N1 * N2

    # 1) Aplanamos WV en Fortran-order por tiro
    #    Trasponemos ejes para simular ravel(order='F') con reshape
    flat_WV = jnp.reshape(jnp.transpose(WV, (1, 0, 2)), (N1 * N2, Nt))

    # 2) Construimos un array de índices globales shape (Nt, Nmask)
    #    Aquí mask_indices ya debe ser un jnp.ndarray de enteros shape (Nt, Nmask)
    global_inds = jnp.take(ind_matlab, mask_indices)

    # 3) Extraemos simultáneamente REC_SIM y REC real (ambos shape (Nt, Nmask))
    rec_sim = jnp.take_along_axis(flat_WV.T, global_inds, axis=1)
    rec = jnp.take_along_axis(REC_DATA, mask_indices, axis=1)

    # 4) Estimamos todas las fuentes en paralelo
    SRC_EST = estimate_src_strength_batched(rec_sim, rec)  # shape (Nt,)

    # 5) Actualizamos WV usando broadcasting
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
    REC_SIM = jnp.zeros((Nt, num_elements), dtype=jnp.complex64)
    batch_idx = jnp.arange(Nt)[:, None]  # (Nt,1)
    REC_SIM = REC_SIM.at[batch_idx, mask_indices].set(rec_sim)

    # final L2 loss over everything
    loss = 0.5 * jnp.sum(jnp.abs(rec_sim - rec_obs) ** 2)
    return loss


def run_lbfgs_fwi(
    xi, yi, REC_DATA, SRC, tx_include, ind_matlab, c_init, f, a0, L_PML, mask_indices
):
    num_elements = SRC.shape[2]
    init_vel = c_init * jnp.ones((yi.size, xi.size))
    init_params = 1.0 / init_vel

    loss_fn = partial(
        fwi_loss_function,
        xi=xi,
        yi=yi,
        REC_DATA=REC_DATA,
        SRC=SRC,
        f=f,
        a0=a0,
        L_PML=L_PML,
        tx_include=tx_include,
        ind_matlab=ind_matlab,
        mask_indices=mask_indices,
        num_elements=num_elements,
    )

    solver = LBFGS(fun=loss_fn, maxiter=1, tol=1e-5)
    opt_params, state = solver.run(init_params)
    final_slow = opt_params.reshape((yi.size, xi.size))
    final_vel = 1.0 / final_slow
    return final_vel


def main():
    print("Loading data...")
    data = mat73.loadmat("RecordedData.mat", use_attrdict=True)
    x = jnp.array(data["x"], dtype=jnp.float32)
    y = jnp.array(data["y"], dtype=jnp.float32)
    C = jnp.array(data["C"], dtype=jnp.float32)
    x_circ = jnp.array(data["x_circ"], dtype=jnp.float32)
    y_circ = jnp.array(data["y_circ"], dtype=jnp.float32)
    f = jnp.array(data["f"], dtype=jnp.float32)
    REC_DATA = jnp.array(data["REC_DATA"], dtype=jnp.complex64)

    dwnsmp = 1
    num_elements = x_circ.size
    a0 = 10.0
    L_PML = 9.0e-3
    tx_include = jnp.arange(0, num_elements, dwnsmp)
    REC_DATA = REC_DATA[tx_include, :]

    numElemLR = 31
    arangeLR = jnp.arange(-numElemLR, numElemLR + 1)
    elemInclude = jnp.ones((num_elements, num_elements), dtype=bool)
    for tx in range(num_elements):
        excl = (arangeLR + tx) % num_elements
        elemInclude = elemInclude.at[tx, excl].set(False)

    dxi = 0.8e-3
    xmax = 120e-3
    xi = jnp.arange(-xmax, xmax + dxi, dxi)
    yi = xi.copy()
    Nyi, Nxi = yi.size, xi.size

    tree_x = jnp.abs(xi[None, :] - x_circ[:, None])
    x_idx = jnp.argmin(tree_x, axis=1)
    tree_y = jnp.abs(yi[None, :] - y_circ[:, None])
    y_idx = jnp.argmin(tree_y, axis=1)
    ind_matlab = x_idx * Nxi + y_idx

    SRC = jnp.zeros((Nyi, Nxi, tx_include.size), dtype=jnp.complex64)
    for i, t in enumerate(tx_include):
        SRC = SRC.at[y_idx[t], x_idx[t], i].set(1.0)

    explicit_indices = []
    mask_indices = []
    for t in tx_include:
        mask = elemInclude[t, :].nonzero()[0]
        explicit_indices.append(mask)
        mask_indices.append(mask)
    mask_indices = jnp.stack([jnp.array(m, dtype=int) for m in mask_indices], axis=0)

    c_init = 1480.0
    VEL_F = run_lbfgs_fwi(
        xi,
        yi,
        REC_DATA,
        SRC,
        tx_include,
        ind_matlab,
        c_init,
        f,
        a0,
        L_PML,
        mask_indices,
    )

    crange = [1400, 1600]
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].imshow(
        C,
        extent=[x.min(), x.max(), y.max(), y.min()],
        vmin=crange[0],
        vmax=crange[1],
        cmap="gray",
        origin="upper",
    )
    axs[0].set_title("True Sound Speed [m/s]")
    axs[0].axis("image")
    plt.colorbar(axs[0].images[0], ax=axs[0])

    axs[1].imshow(
        VEL_F,
        extent=[xi.min(), xi.max(), yi.max(), yi.min()],
        vmin=crange[0],
        vmax=crange[1],
        cmap="gray",
        origin="upper",
    )
    axs[1].set_title("Estimated Speed [m/s] (LBFGS)")
    axs[1].axis("image")
    plt.colorbar(axs[1].images[0], ax=axs[1])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
