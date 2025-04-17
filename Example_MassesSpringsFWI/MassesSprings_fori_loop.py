import jax # Import JAX Before Importing Any Submodules
import jax.numpy as jnp
import numpy as np
from jax.scipy.signal import fftconvolve
from PIL import Image
from HelperFunctions import *
import jaxopt
import pdb

## Differential Equations

# Forward Differential Equation: Forward-Time Propagation
#   F_i - k_i(u_i - u_{i-1}) - k_{i+1}(u_i - u_{i+1}) = m_i(d2u_i)
@jax.jit
def simulate(t, F, mass, springs):
    # Inputs
    Nt = t.size; # Number of Time Points
    dt = jnp.mean(jnp.diff(t)); # Time Step
    N = springs.size-1; # Number of Masses
    # Displacement of Each Mass (Walls Clamped to Zero)
    u = jnp.zeros((N+2, Nt)); # Displacements vs Time
    # Body of For Loop with Conditional Statement (JAX can't work with if statements)
    def compute_unext(t_idx, u):
        force_term = (F[:,t_idx] - springs[:-1]*(u[1:-1,t_idx]-u[:-2,t_idx]) - 
            springs[1:]*(u[1:-1,t_idx]-u[2:,t_idx])); 
        # jax.lax.cond(condition, ifTrueFunction, ifFalseFunction, *inputsToFunction)
        return jax.lax.cond(t_idx == 0, 
            lambda: 2*u[1:-1,t_idx] + (dt**2) * force_term / mass, 
            lambda: 2*u[1:-1,t_idx] - u[1:-1,t_idx-1] + (dt**2) * force_term / mass); 
    def update_u(t_idx, u):
        unext = compute_unext(t_idx, u); 
        return u.at[1:-1,t_idx+1].set(unext); 
    # Time Advancement Loop
    u = jax.lax.fori_loop(0, Nt-1, update_u, u)
    # Remove Walls
    return u[1:-1,:]; 

## Simulate Series of Masses and Springs

# Create series of masses and springs
N = 19; # Number of masses
x = jnp.arange(N+2)/(N+1); # Resting Locations of Masses (and Walls at the Ends)
masses = jnp.array(np.random.rand(N)+10); # Random unknown masses
springs = jnp.array(np.random.rand(N+1)+10); # springs between masses and walls

# Create Time Axis
dt = 0.25; # Time Step (should be less than sqrt(m/k))
Nt = 401; # Number of Time Points
t = jnp.arange(Nt)*dt; # Time Axis

# Random Force Input
force = jnp.zeros((N, Nt));
force = force.at[:,1].set(jnp.hstack((1,jnp.zeros(N-1))));

# Forward Simulation
u = simulate(t, force, masses, springs);
plt.ion()
for t_idx in np.arange(Nt):
    plt.clf(); 
    plt.stem(x, jnp.hstack((0,u[:,t_idx],0))); 
    plt.ylim(np.array([-1,1])*jnp.mean(jnp.diff(x))/2); 
    plt.xlabel('Location of Mass (Fixed Ends)'); 
    plt.ylabel('Horizontal Displacement of Mass'); 
    plt.show(); plt.pause(0.001); 

# Only Measure the Displacement of One Mass
meas_idx = np.arange(N);
umeas = u[meas_idx, :];

## Inverse-Problem: What were the masses and springs in the simulation?

# Initial Guess for Masses and Springs
masses_estim = 10*jnp.ones(N);
springs_estim = 10*jnp.ones(N+1);

# Forward Model and Objective Function for Nonlinear Least-Squares CG
forward_model = lambda params: simulate(t,force,params[0],params[1])[meas_idx,:];
loss_function = lambda params, data: jnp.sum((forward_model(params)-data)**2)/2;

# Run LBFGS Solver
maxiterLBFGS = 300; tolLBFGS = 1e-6;
solver = jaxopt.LBFGS(fun=loss_function, maxiter=maxiterLBFGS, tol=tolLBFGS);
params, state = solver.run((masses_estim, springs_estim), data=umeas); 
masses_estim, springs_estim = params; 

# Plot Solution
plt.clf(); plt.ioff(); 
plt.subplot(2,2,1); 
plt.plot(x[1:-1], masses, 'k', \
    x[1:-1], masses_estim, 'r', linewidth=2)
plt.xlabel('Location'); plt.ylabel('Masses'); 
plt.title('L-BFGS Solution'); 
plt.subplot(2,2,2); 
plt.plot((x[:-1]+x[1:])/2, springs, 'k', \
    (x[:-1]+x[1:])/2, springs_estim, 'r', linewidth=2)
plt.xlabel('Location'); plt.ylabel('Spring Constants'); 
plt.title('L-BFGS Solution'); 
plt.subplot(2,2,3); 
imagesc(np.linspace(0,1,Nt), np.linspace(0,1/2,meas_idx.size), umeas, numticks=(0,0)); 
plt.title('Measured Signal'); plt.xlabel('Time'); plt.ylabel('Location'); 
plt.subplot(2,2,4); 
imagesc(np.linspace(0,1,Nt), np.linspace(0,1/2,meas_idx.size), \
    forward_model((masses_estim, springs_estim)), numticks=(0,0)); 
plt.title('Simulated Signal'); plt.xlabel('Time'); plt.ylabel('Location'); 
plt.show(); 

# Conjugate Gradient Method (with Exact Line Search Step Size Calculation)
plt.ion(); 
# Parameters for Conjugate Gradient Reconstruction
maxiterCG = 300; # Number of Iterations
# Initial Guess for Masses and Springs
masses_estim = 10*jnp.ones(N);
springs_estim = 10*jnp.ones(N+1);
estim = (masses_estim, springs_estim);
# Iterative Nonlinear Conjugate Gradient Algorithm
for iteration in range(maxiterCG):
    # 1) Compute Gradient
    grad = jax.grad(loss_function,argnums=0)(estim, umeas); 
    if iteration == 0:
        # 3) Calculate Search Direction
        search_dir = tuple(-grad[i] for i in range(len(grad)))
    else:
        # 2) Calculate Momentum (Fletcher-Reeves Formula)
        momentum = jnp.vdot(jnp.hstack(grad).flatten(), jnp.hstack(grad).flatten()) / \
            jnp.vdot(jnp.hstack(grad_prev).flatten(), jnp.hstack(grad_prev).flatten());
        # 3) Calculate Search Direction
        search_dir = tuple(momentum * search_dir[i] - grad[i] for i in range(len(search_dir)))
    # 4) Forward Project the Search Direction
    dres = jax.jvp(forward_model,(estim,),(search_dir,))[1];
    # 5) Step Size
    stepsize = -jnp.vdot(jnp.hstack(grad).flatten(), jnp.hstack(search_dir).flatten()) / \
        jnp.vdot(jnp.hstack(dres).flatten(), jnp.hstack(dres).flatten());
    # 6) Update
    estim = tuple(estim[i] + stepsize * search_dir[i] for i in range(len(search_dir)))
    masses_estim, springs_estim = estim;
    # Record Previous Gradient
    grad_prev = grad;
    # Plot Image
    plt.clf(); 
    plt.subplot(2,2,1); 
    plt.plot(x[1:-1], masses, 'k', \
        x[1:-1], masses_estim, 'r', linewidth=2)
    plt.xlabel('Location'); plt.ylabel('Masses'); 
    plt.title('CG Iteration '+str(iteration)); 
    plt.subplot(2,2,2); 
    plt.plot((x[:-1]+x[1:])/2, springs, 'k', \
        (x[:-1]+x[1:])/2, springs_estim, 'r', linewidth=2)
    plt.xlabel('Location'); plt.ylabel('Spring Constants'); 
    plt.title('CG Iteration '+str(iteration)); 
    plt.subplot(2,2,3); 
    imagesc(np.linspace(0,1,Nt), np.linspace(0,1/2,meas_idx.size), umeas, numticks=(0,0)); 
    plt.title('Measured Signal'); plt.xlabel('Time'); plt.ylabel('Location'); 
    plt.subplot(2,2,4); 
    imagesc(np.linspace(0,1,Nt), np.linspace(0,1/2,meas_idx.size), \
        forward_model((masses_estim, springs_estim)), numticks=(0,0)); 
    plt.title('Simulated Signal'); plt.xlabel('Time'); plt.ylabel('Location'); 
    plt.show(); plt.pause(0.01); 
