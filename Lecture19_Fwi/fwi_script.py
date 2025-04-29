import jax
import jax.numpy as jnp
from functools import partial
from scipy.io import loadmat
from scipy.spatial import cKDTree
from nonlinearcg import nonlinear_conjugate_gradient
import matplotlib.pyplot as plt

# Load recorded data
print("Loading data...")
data = loadmat("RecordedData1.mat")
x = data["x"].squeeze()
y = data["y"].squeeze()
C = data["C"]
x_circ = data["x_circ"].squeeze()
y_circ = data["y_circ"].squeeze()
f = data["f"].item()
REC_DATA = data["REC_DATA"]

# Data preprocessing
dwnsmp = 1
num_elements = x_circ.size
a0 = 10
L_PML = 9.0e-3
tx_include = jnp.arange(0, num_elements, dwnsmp)
REC_DATA = REC_DATA[tx_include, :]

numElemLeftRightExcl = 31
elemLeftRightExcl = jnp.arange(-numElemLeftRightExcl, numElemLeftRightExcl + 1)
elemInclude = jnp.ones((num_elements, num_elements), dtype=bool)

for tx_element in range(num_elements):
    elemLeftRightExclCurrent = (elemLeftRightExcl + tx_element) % num_elements
    elemInclude = elemInclude.at[tx_element, elemLeftRightExclCurrent].set(False)

dxi = 0.8e-3
xmax = 120e-3
xi = jnp.arange(-xmax, xmax + dxi, dxi)
yi = xi
Xi, Yi = jnp.meshgrid(xi, yi, indexing="xy")

xi_tree = cKDTree(xi[:, None])
x_idx = xi_tree.query(x_circ[:, None])[1]

yi_tree = cKDTree(yi[:, None])
y_idx = yi_tree.query(y_circ[:, None])[1]

Nyi, Nxi = yi.size, xi.size
ind = y_idx * Nxi + x_idx

SRC = jnp.zeros((Nyi, Nxi, tx_include.size))
for i, tx_elmt_idx in enumerate(tx_include):
    x_idx_src = x_idx[tx_elmt_idx]
    y_idx_src = y_idx[tx_elmt_idx]
    SRC = SRC.at[y_idx_src, x_idx_src, i].set(1)

print("SRC shape:", SRC.shape)
print("SRC type:", SRC.dtype)

# FWI initial guess
c_init = 1480.0
Niter = 2

explicit_indices = []
mask_indices = []

for tx in tx_include:
    mask = elemInclude[tx, :]
    idx = jnp.where(mask)[0]
    explicit_indices.append(idx)
    mask_indices.append(idx)  # Same here, because you need for REC_DATA also


# Run nonlinear conjugate gradient for FWI
print("Running Nonlinear Conjugate Gradient...")
VEL_ESTIM_final, search_dir_final, gradient_img_final = nonlinear_conjugate_gradient(
    xi,
    yi,
    num_elements,
    REC_DATA,
    SRC,
    elemInclude,
    tx_include,
    ind,
    c_init,
    f,
    Niter,
    a0,
    L_PML,
    explicit_indices,
    mask_indices,
)

print("FWI complete.")


import matplotlib.pyplot as plt

# Final visualization after FWI
crange = [1400, 1600]  # Display range for velocity

fig, axs = plt.subplots(2, 2, figsize=(14, 12))

# Subplot 1: True Sound Speed
axs[0, 0].imshow(
    C,
    extent=[x.min(), x.max(), y.max(), y.min()],
    vmin=crange[0],
    vmax=crange[1],
    cmap="gray",
)
axs[0, 0].set_title("True Sound Speed [m/s]")
axs[0, 0].set_xlabel("Lateral [m]")
axs[0, 0].set_ylabel("Axial [m]")
axs[0, 0].axis("image")
fig.colorbar(axs[0, 0].images[0], ax=axs[0, 0])

# Subplot 2: Estimated Sound Speed
axs[0, 1].imshow(
    VEL_ESTIM_final,
    extent=[xi.min(), xi.max(), yi.max(), yi.min()],
    vmin=crange[0],
    vmax=crange[1],
    cmap="gray",
)
axs[0, 1].set_title("Estimated Sound Speed [m/s]")
axs[0, 1].set_xlabel("Lateral [m]")
axs[0, 1].set_ylabel("Axial [m]")
axs[0, 1].axis("image")
fig.colorbar(axs[0, 1].images[0], ax=axs[0, 1])

# Subplot 3: Search Direction
axs[1, 0].imshow(
    search_dir_final, extent=[xi.min(), xi.max(), yi.max(), yi.min()], cmap="gray"
)
axs[1, 0].set_title("Search Direction Final Iteration")
axs[1, 0].set_xlabel("Lateral [m]")
axs[1, 0].set_ylabel("Axial [m]")
axs[1, 0].axis("image")
fig.colorbar(axs[1, 0].images[0], ax=axs[1, 0])

# Subplot 4: Negative Gradient
axs[1, 1].imshow(
    -gradient_img_final, extent=[xi.min(), xi.max(), yi.max(), yi.min()], cmap="gray"
)
axs[1, 1].set_title("Negative Gradient Final Iteration")
axs[1, 1].set_xlabel("Lateral [m]")
axs[1, 1].set_ylabel("Axial [m]")
axs[1, 1].axis("image")
fig.colorbar(axs[1, 1].images[0], ax=axs[1, 1])

plt.tight_layout()
plt.show()

plt.tight_layout()
plt.show()
