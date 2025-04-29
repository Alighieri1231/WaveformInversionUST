import jax # Import JAX Before Importing Any Submodules
import jax.numpy as jnp

# Python-Equivalent Command for IMAGESC in MATLAB
import matplotlib.pyplot as plt
def imagesc(x, y, img, rng=None, cmap='gray', numticks=(3, 3), aspect='equal'):
    if rng == None:
        rng = [jnp.min(img), jnp.max(img)];
    exts = (jnp.min(x)-jnp.mean(jnp.diff(x)), jnp.max(x)+jnp.mean(jnp.diff(x)), \
        jnp.min(y)-jnp.mean(jnp.diff(y)), jnp.max(y)+jnp.mean(jnp.diff(y)));
    plt.imshow(jnp.flipud(img), cmap=cmap, extent=exts, vmin=rng[0], vmax=rng[1], aspect=aspect);
    plt.xticks(jnp.linspace(jnp.min(x), jnp.max(x), numticks[0]));
    plt.yticks(jnp.linspace(jnp.min(y), jnp.max(y), numticks[1]));
    plt.gca().invert_yaxis();


