import jax # Import JAX Before Importing Any Submodules
import jax.numpy as jnp
import jaxopt
from jaxopt.tree_util import tree_add_scalar_mul
from jaxopt.tree_util import tree_div
from jaxopt.tree_util import tree_l2_norm
from jaxopt.tree_util import tree_scalar_mul
from jaxopt.tree_util import tree_sub
from jaxopt.tree_util import tree_vdot_real
from jaxopt.tree_util import tree_conj

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

## Hessian-Vector Product - Forward-Over-Reverse
def hvp(f, primals, tangents):
    return jax.jvp(jax.grad(f), primals, tangents)[1]

## Linearized CG with Closed-Form Step Size Calculation
## Conjugate Gradient Method (with Exact Line Search Step Size Calculation)
class LinearizedCG:
    def __init__(self, fun, maxiter=500, method="polak-ribiere"):
        self.fun = fun; 
        self.maxiter = maxiter; 
        self.method = method; 
        self.iter = 0; 
    def init_state(self, params, *args, **kwargs):
        # Initial input params
        #pdb.set_trace();
        self.params = params; 
        # Objective Function with *args, **kwargs 
        self.obj_fun = lambda params: self.fun(params, *args, **kwargs); 
    def update(self):
        # Compute Gradient
        new_grad = jax.grad(self.obj_fun)(self.params); 
        if self.iter == 0:
            # Calculate Search Direction
            descent_direction = tree_scalar_mul(-1, new_grad)
        else:
            # Load Previous Gradient and Descent Direction
            grad = self.grad; 
            descent_direction = self.descent_direction; 
            # Calculate Momentum
            eps = 1e-16
            if self.method == "polak-ribiere":
                # See Numerical Optimization, second edition, equation (5.44).
                gTg = tree_vdot_real(grad, grad)
                gTg = jnp.where(gTg >= eps, gTg, eps)
                momentum = tree_vdot_real(
                    tree_conj(tree_sub(new_grad, grad)), tree_conj(new_grad)) / gTg
                momentum = jax.nn.relu(momentum)
            elif self.method == "fletcher-reeves":
                # See Numerical Optimization, second edition, equation (5.41a).
                gTg = tree_vdot_real(grad, grad)
                gTg = jnp.where(gTg >= eps, gTg, eps)
                momentum = tree_vdot_real(new_grad, new_grad) / gTg
                momentum = jax.nn.relu(momentum)
            elif self.method == "hestenes-stiefel":
                # See Numerical Optimization, second edition, equation (5.45).
                grad_diff = tree_sub(new_grad, grad)
                dTg = tree_vdot_real(tree_conj(grad_diff), descent_direction)
                dTg = jnp.where(dTg >= eps, dTg, eps)
                momentum = tree_vdot_real(
                    tree_conj(grad_diff), tree_conj(new_grad)) / dTg
                momentum = jax.nn.relu(momentum)
            else:
                raise ValueError("method argument should be either 'polak-ribiere', "
                            "'fletcher-reeves', or 'hestenes-stiefel'.")
            # Calculate Search Direction
            descent_direction = tree_sub(tree_scalar_mul(momentum, descent_direction), new_grad);
        # Step Size
        stepsize = (tree_vdot_real(new_grad, descent_direction) / \
            tree_vdot_real(descent_direction, hvp(self.obj_fun, (self.params,), (descent_direction,))))
        # 6) Update
        self.params = tree_add_scalar_mul(self.params, -stepsize, descent_direction)
        # Record Previous Gradient
        self.grad = new_grad
        self.descent_direction = descent_direction
        self.iter += 1
        return self.params
    def run(self, params, *args, **kwargs):
        # Initialize Run State
        self.init_state(params, *args, **kwargs); 
        # Loop Until Max Iterations Hit
        while self.iter < self.maxiter:
            self.update(); 
        return self.params; 