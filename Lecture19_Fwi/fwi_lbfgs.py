import jax  # Import JAX Before Importing Any Submodules
import jax.numpy as jnp
import numpy as np
from jax.scipy.interpolate import RegularGridInterpolator
from PIL import Image
from HelperFunctions import *
from functools import partial
import jaxopt
import pdb


# Projection for Ring-Based CT System
# @partial(jax.jit, static_argnames=['Npts'])
@jax.jit
def projectRing(x, y, obj, xr, yr, Npts=1001):
    """PROJECTRING Generate a Sinogram of the Image
    USAGE:
        sg = projectRing(x, y, obj, xr, yr, Npts)
    INPUTS:
        x -- x coordinates of object: numel(x) must be equal to size(object,2)
        y -- y coordinates of object: numel(y) must be equal to size(object,1)
        obj -- intensity of object as a function of x and y
        xr -- vector of x-positions [m] of all elements
        yr -- vector of y-positions [m] of all elements
        Npts -- number of points used in line integral"""
    # Create meshgrid of x and y coordinates
    interpolator = RegularGridInterpolator((y, x), obj, method="linear", fill_value=0)
    # Set up coordinate t for the line integral (between 0 and 1)
    t = (jnp.arange(0.5, Npts + 0.5)) / Npts
    # All Transmit-Receive Combinations
    xr_tx, xr_rx = jnp.meshgrid(xr, xr)  # Avoid for loops using meshgrid to vectorize
    yr_tx, yr_rx = jnp.meshgrid(yr, yr)
    # Compute Points Along the Line
    xline = xr_tx + t[:, None, None] * (
        xr_rx - xr_tx
    )  # Add new axis to match broadcasting
    yline = yr_tx + t[:, None, None] * (yr_rx - yr_tx)
    # Compute the ds (spacing) for the integral
    ds = jnp.sqrt((xr_rx - xr_tx) ** 2 + (yr_rx - yr_tx) ** 2) / (Npts - 1)
    # Perform 2D interpolation along the line (using interp2d)
    coords = (yline.flatten(), xline.flatten())  # Interpolate the points along the line
    integrand = interpolator(coords).reshape(Npts, Nelem, Nelem)
    # Sum the integrand over the line and multiply by ds
    sg = jnp.sum(integrand, axis=0) * ds
    return sg


# Load the TIFF image using Pillow
orig_object = jnp.mean(
    jnp.array(Image.open("cholangioca.jpg")).astype(jnp.float32), axis=2
)
img = jnp.pad(
    orig_object[14:-16, :], ((160, 160), (0, 0)), "constant", constant_values=(0, 0)
)[::2, ::2]
Ny, Nx = img.shape
# Now lets assign coordinates to points in the image
dx = 1e-3
dy = 1e-3  # spacing of points in image [m]
# assume center of image is at (0,0)
x = jnp.arange(-(Nx - 1) / 2, 1 + (Nx - 1) / 2) * dx
y = jnp.arange(-(Ny - 1) / 2, 1 + (Ny - 1) / 2) * dy
# CT Image
plt.subplot(1, 3, 1)
imagesc(x, y, img)
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("CT Image")
# Ring-Array Geometry
radius = 20e-2  # Ring radius [m]
Nelem = 256  # Number of source and detector positions
theta = 2 * jnp.pi * jnp.arange(Nelem) / Nelem  # Projection angles [rad]
xring = radius * jnp.cos(theta)
yring = radius * jnp.sin(theta)  # Element Positions [m]
plt.plot(xring, yring, "r.")
# Extract Subset of Times within Acceptance Angle
numElemLeftRightExcl = 31
elemLeftRightExcl = jnp.arange(-numElemLeftRightExcl, 1 + numElemLeftRightExcl)
elemInclude = jnp.ones((Nelem, Nelem), dtype=bool)
for tx_element in jnp.arange(Nelem):
    elemLeftRightExclCurrent = elemLeftRightExcl + tx_element
    elemLeftRightExclCurrent = jnp.mod(elemLeftRightExclCurrent, Nelem)
    elemInclude = elemInclude.at[tx_element, elemLeftRightExclCurrent].set(False)

# Simulate Sinogram and Extract the Specific Elements Used
sinogram = projectRing(x, y, img, xring, yring)
elemInclude_flat = elemInclude.flatten()
d = sinogram.flatten()[elemInclude_flat]  # Extract only the elements used
d = d + (1e-3) * np.random.randn(d.size)  # Add noise to the data

# Plot Sinogram and Elements Used
plt.subplot(1, 3, 2)
imagesc(np.arange(Nelem), np.arange(Nelem), sinogram, numticks=(0, 0))
plt.xlabel("Transmitter")
plt.ylabel("Receiver")
plt.title("Sinogram")
plt.subplot(1, 3, 3)
imagesc(jnp.arange(Nelem), jnp.arange(Nelem), elemInclude, numticks=(0, 0))
plt.xlabel("Transmitter")
plt.ylabel("Receiver")
plt.title("Elements to Include")
plt.show()


# Total variation regularization with periodic boundary conditions
@jax.jit
def total_variation(x, eps=1e-8):
    periodic_diff = lambda x, axis: jnp.roll(x, shift=-1, axis=axis) - x
    dx = periodic_diff(x, axis=0)  # Gradient along rows (vertical)
    dy = periodic_diff(x, axis=1)  # Gradient along columns (horizontal)
    return jnp.sum(jnp.sqrt(jnp.abs(dx) ** 2 + jnp.abs(dy) ** 2 + eps))


# Objective function using both data fidelity and total variation terms
@jax.jit
def loss_function(image, data, lambda_tv):
    sg = projectRing(x, y, image, xring, yring)
    data_sim = sg.flatten()[elemInclude_flat]
    data_fidelity = jnp.sum((data_sim - data) ** 2)
    tv_regularization = total_variation(image)
    return data_fidelity + lambda_tv * tv_regularization


# return data_fidelity;

# Parameters for CT Reconstruction
max_iterations = 20
lambda_tv = 1e-6
# Optimization loop
plt.ion()
solver = jaxopt.LBFGS(fun=loss_function)
image_estim = jnp.zeros((Ny, Nx))
state = solver.init_state(image_estim, data=d, lambda_tv=lambda_tv)
for iteration in range(max_iterations):
    # Solver Gradient Update
    image_grad = jax.grad(loss_function, argnums=0)(image_estim, d, lambda_tv)
    image_estim, state = solver.update(image_estim, state, data=d, lambda_tv=lambda_tv)
    # Plot Image
    plt.subplot(2, 2, 1)
    imagesc(x, y, img, numticks=(0, 0))
    plt.title("Original Image")
    plt.subplot(2, 2, 2)
    imagesc(x, y, image_estim, numticks=(0, 0))
    plt.title("Estimated Image")
    plt.subplot(2, 2, 3)
    imagesc(x, y, image_estim - img, numticks=(0, 0))
    plt.title("Error Image " + str(iteration))
    plt.subplot(2, 2, 4)
    imagesc(x, y, image_grad, numticks=(0, 0))
    plt.title("Gradient Image " + str(iteration))
    plt.show()
    plt.pause(1)
