import h5py
import jax.numpy as jnp

# Path to your MATLAB v7.3 file
mat_path = 'RecordedData.mat'

with h5py.File(mat_path, 'r') as mat:
    # read each dataset into a NumPy array, then convert to JAX
    x        = jnp.array(mat['x'][:])
    y        = jnp.array(mat['y'][:])
    C        = jnp.array(mat['C'][:])
    x_circ   = jnp.array(mat['x_circ'][:])
    y_circ   = jnp.array(mat['y_circ'][:])
    f_data   = jnp.array(mat['f'][:])         # renamed to avoid shadowing
    REC_DATA = jnp.array(mat['REC_DATA'][:])

# now x, y, C, x_circ, y_circ, f_data, REC_DATA are JAX arrays
print(type(x), x.shape)
