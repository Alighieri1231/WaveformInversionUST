import jax
import jax.numpy as jnp
from functools import partial
from scipy.io import loadmat
from scipy.spatial import cKDTree
from nonlinearcg import nonlinear_conjugate_gradient
import matplotlib.pyplot as plt
from HelperFunctions import imagesc
import numpy as np


def main():
    # -------------------------
    # 1) Load recorded data
    # -------------------------
    print("Loading data...")
    data = loadmat("RecordedData1.mat", squeeze_me=True, struct_as_record=False)
    x = data["x"]
    y = data["y"]
    C = data["C"]
    x_circ = data["x_circ"]
    y_circ = data["y_circ"]
    f = float(data["f"])
    REC_DATA = data["REC_DATA"].astype(np.complex64)

    # -------------------------
    # 2) Preprocessing
    # -------------------------
    dwnsmp = 1
    num_elements = x_circ.size
    a0 = 10.0
    L_PML = 9.0e-3
    tx_include = jnp.arange(0, num_elements, dwnsmp)
    REC_DATA = REC_DATA[tx_include, :]

    # build mask elemInclude
    numElemLR = 31
    arangeLR = jnp.arange(-numElemLR, numElemLR + 1)
    elemInclude = jnp.ones((num_elements, num_elements), dtype=bool)
    for tx in range(num_elements):
        excl = (arangeLR + tx) % num_elements
        elemInclude = elemInclude.at[tx, excl].set(False)

    # grid
    dxi = 0.8e-3
    xmax = 120e-3
    xi = jnp.arange(-xmax, xmax + dxi, dxi)
    yi = xi.copy()
    Nyi, Nxi = yi.size, xi.size

    # nearest‐neighbor search for element positions
    tree_x = cKDTree(xi.reshape(-1, 1))
    x_idx = tree_x.query(x_circ.reshape(-1, 1))[1]
    tree_y = cKDTree(yi.reshape(-1, 1))
    y_idx = tree_y.query(y_circ.reshape(-1, 1))[1]

    # MATLAB‐style linear index (column‐major, zero‐based)
    ind_matlab = x_idx * Nyi + y_idx

    # build source array (one hot per tx)
    SRC = jnp.zeros((Nyi, Nxi, tx_include.size), dtype=jnp.complex64)
    for i, t in enumerate(tx_include):
        SRC = SRC.at[y_idx[t], x_idx[t], i].set(1.0)

    print(f"SRC shape: {SRC.shape}, dtype: {SRC.dtype}")

    # -------------------------
    # 3) Build explicit_indices
    # -------------------------
    explicit_indices = []
    mask_indices = []
    for t in tx_include:
        mask = elemInclude[t, :].nonzero()[0]
        explicit_indices.append(mask)
        mask_indices.append(mask)

    # debug print first 5
    for i, idx in enumerate(mask_indices[:5]):
        print(f"tx={i}: #included = {len(idx)}")

    # check row‐ vs col‐major
    for k in range(5):
        yi_k, xi_k = int(y_idx[k]), int(x_idx[k])
        row_ind = yi_k * Nxi + xi_k
        col_ind = xi_k * Nyi + yi_k
        print(f"k={k}: (y,x)=({yi_k},{xi_k}), row={row_ind}, col={col_ind}")

    # -------------------------
    # 4) FWI
    # -------------------------
    c_init = 1480.0
    Niter = 2  # prueba rápida

    print("Running Nonlinear Conjugate Gradient...")
    VEL_F, SD_F, GRAD_F = nonlinear_conjugate_gradient(
        xi,
        yi,
        num_elements,
        REC_DATA,
        SRC,
        elemInclude,
        tx_include,
        ind_matlab,  # ahora pasamos el índice col-major
        c_init,
        f,
        Niter,
        a0,
        L_PML,
        explicit_indices,
        mask_indices,
    )

    # -------------------------
    # 5) Visualization
    # -------------------------
    crange = [1400, 1600]
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # True speed
    axs[0, 0].imshow(
        C,
        extent=[x.min(), x.max(), y.max(), y.min()],
        vmin=crange[0],
        vmax=crange[1],
        cmap="gray",
        origin="upper",
    )
    axs[0, 0].set_title("True Sound Speed [m/s]")
    axs[0, 0].axis("image")
    plt.colorbar(axs[0, 0].images[0], ax=axs[0, 0])

    # Estimated speed
    axs[0, 1].imshow(
        VEL_F,
        extent=[xi.min(), xi.max(), yi.max(), yi.min()],
        vmin=crange[0],
        vmax=crange[1],
        cmap="gray",
        origin="upper",
    )
    axs[0, 1].set_title("Estimated Speed Iter  " + str(Niter))
    axs[0, 1].axis("image")
    plt.colorbar(axs[0, 1].images[0], ax=axs[0, 1])

    # Search direction
    axs[1, 0].imshow(
        SD_F,
        extent=[xi.min(), xi.max(), yi.max(), yi.min()],
        cmap="gray",
        origin="upper",
    )
    axs[1, 0].set_title("Search Direction")
    axs[1, 0].axis("image")
    plt.colorbar(axs[1, 0].images[0], ax=axs[1, 0])

    # Negative gradient
    axs[1, 1].imshow(
        -GRAD_F,
        extent=[xi.min(), xi.max(), yi.max(), yi.min()],
        cmap="gray",
        origin="upper",
    )
    axs[1, 1].set_title("Negative Gradient")
    axs[1, 1].axis("image")
    plt.colorbar(axs[1, 1].images[0], ax=axs[1, 1])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
