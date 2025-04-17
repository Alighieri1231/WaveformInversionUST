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
    def update_u(u, t_idx):
        unext = compute_unext(t_idx, u); 
        return u.at[1:-1,t_idx+1].set(unext), None; 
    # Time Advancement Loop
    u, _ = jax.lax.scan(update_u, u, jnp.arange(Nt-1))
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

## Useful Functions for JVPs and VJPs
# Adjoint Differential Equation: Reverse-Time Propagation
def adjoint_sim(t, F, mass, springs):
    return simulate(t, F[:,::-1], mass, springs)[:,::-1];
# Virtual Source for Mass Estimation
def virtualSourceMasses(u, t, mass):
    # Inputs
    Nt = t.size; # Number of Time Points
    dt = jnp.mean(jnp.diff(t)); # Time Step
    N = mass.size; # Number of Masses
    # Displacement of Each Mass (Walls Clamped to Zero)
    virtSrc = jnp.zeros((N,Nt));
    virtSrc = virtSrc.at[:,0].set((-2*u[:,0]+u[:,1]) / (dt**2));
    virtSrc = virtSrc.at[:,1:Nt-1].set((u[:,0:Nt-2]-2*u[:,1:Nt-1]+u[:,2:Nt]) / (dt**2));
    virtSrc = virtSrc.at[:,Nt-1].set((u[:,Nt-2]-2*u[:,Nt-1]) / (dt**2));
    return virtSrc; 
# Virtual Source for Spring Estimation
def virtualSourceSprings(u, t, springs):
    # Inputs
    Nt = t.size; # Number of Time Points
    dt = jnp.mean(jnp.diff(t)); # Time Step
    N = springs.size-1; # Number of Masses
    # Displacement of Each Mass (Walls Clamped to Zero)
    virtSrc = jnp.zeros((N,Nt,N+1)); 
    virtSrc = virtSrc.at[0,:,0].set(u[0,:]); 
    for mass_idx in np.arange(N):
        if mass_idx == 0:
            virtSrc = virtSrc.at[mass_idx,:,mass_idx].set(u[mass_idx,:]);
        else:
            virtSrc = virtSrc.at[mass_idx,:,mass_idx].set(u[mass_idx,:]-u[mass_idx-1,:]);
        if mass_idx == N-1:
            virtSrc = virtSrc.at[mass_idx,:,mass_idx+1].set(u[mass_idx,:]);
        else:
            virtSrc = virtSrc.at[mass_idx,:,mass_idx+1].set(u[mass_idx,:]-u[mass_idx+1,:]);
    return virtSrc; 


# Forward Model With Custom JVP (Jacobian Vector Product) and VJP (Vector Jacobian Product)
@jax.custom_vjp
def forward_model(params): 
    return simulate(t,force,params[0],params[1])[meas_idx,:];
# Returns primal output and residuals to be used in backward pass by model_bwd.
# "residual" here (in JAX terms) just means extra data to be used in the backward pass
def model_fwd(params):
    # Full Forward Model Pass
    masses_in, springs_in = params; 
    usim_full = simulate(t,force,masses_in,springs_in) # Full wavefield
    primal_out = usim_full[meas_idx,:]; 
    # Virtual Sources
    virtSrcMasses = virtualSourceMasses(usim_full, t, masses_in);
    virtSrcSprings = virtualSourceSprings(usim_full, t, springs_in);
    residuals_out = (virtSrcMasses, virtSrcSprings, masses_in, springs_in);
    # Returns primal output and residuals to be used in backward pass by model_bwd.
    return primal_out, residuals_out
# Backpropagates gradient g at output backward through the model
# "res" here (in JAX terms) just means extra data to be used in the backward pass
# but "g" is the gradient propagated back to the forward model (presumably from the loss function)
# since loss function is (1/2)*(error)^2 -- "g" is actually the error (what we call residual...confusing right?)
def model_bwd(res, g):
    # Gets residuals computed in model_fwd
    virtSrcMasses, virtSrcSprings, masses_in, springs_in = res 
    # Build Adjoint Sources
    adj_force = jnp.zeros((N, Nt)); 
    adj_force = adj_force.at[meas_idx,:].set(g);
    # Solve Adjoint Wave Equation
    adj_sim_full = adjoint_sim(t, adj_force, masses_in, springs_in);
    # Calculate Gradient/Backprojection
    grad_masses = -jnp.sum(jnp.conj(virtSrcMasses)*adj_sim_full,axis=1);
    grad_springs = -jnp.sum(jnp.conj(virtSrcSprings)*adj_sim_full[:,:,None],axis=(0,1));
    grad = (grad_masses, grad_springs);
    return (grad,)
forward_model.defvjp(model_fwd, model_bwd)



# Objective Function for Nonlinear Least-Squares
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

# Run CG Solver
maxiterCG = 200; 
solverCG = LinearizedCG(fun=loss_function, maxiter=maxiterCG); # method="fletcher-reeves"
masses_estim = 10*jnp.ones(N); springs_estim = 10*jnp.ones(N+1); # Initial Guess for Masses and Springs
masses_estim, springs_estim = solverCG.run((masses_estim, springs_estim), data=umeas); 

# Plot Solution
plt.clf(); plt.ioff(); 
plt.subplot(2,2,1); 
plt.plot(x[1:-1], masses, 'k', \
    x[1:-1], masses_estim, 'r', linewidth=2)
plt.xlabel('Location'); plt.ylabel('Masses'); 
plt.title('CG Solution'); 
plt.subplot(2,2,2); 
plt.plot((x[:-1]+x[1:])/2, springs, 'k', \
    (x[:-1]+x[1:])/2, springs_estim, 'r', linewidth=2)
plt.xlabel('Location'); plt.ylabel('Spring Constants'); 
plt.title('CG Solution'); 
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
solverCG = LinearizedCG(fun=loss_function); 
masses_estim = 10*jnp.ones(N); springs_estim = 10*jnp.ones(N+1); # Initial Guess for Masses and Springs
solverCG.init_state((masses_estim, springs_estim), data=umeas); 
# Iterative Nonlinear Conjugate Gradient Algorithm
for iteration in range(maxiterCG):
    # Compute Updates
    masses_estim, springs_estim = solverCG.update(); 
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