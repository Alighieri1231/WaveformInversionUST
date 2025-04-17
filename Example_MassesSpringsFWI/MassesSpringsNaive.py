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
def simulate(t, F, mass, springs):
    # Inputs
    Nt = t.size; # Number of Time Points
    dt = jnp.mean(jnp.diff(t)); # Time Step
    N = springs.size-1; # Number of Masses
    # Displacement of Each Mass (Walls Clamped to Zero)
    u = jnp.zeros((N+2, Nt)); # Displacements vs Time
    # Time Advancement Loop
    for t_idx in jnp.arange(Nt-1):
        force_term = (F[:,t_idx] - springs[:-1]*(u[1:-1,t_idx]-u[:-2,t_idx]) - 
            springs[1:]*(u[1:-1,t_idx]-u[2:,t_idx])); 
        if t_idx == 0:
            unext = 2*u[1:-1,t_idx] + (dt**2) * force_term / mass; 
        else:
            unext = 2*u[1:-1,t_idx] - u[1:-1,t_idx-1] + (dt**2) * force_term / mass; 
        u = u.at[1:-1,t_idx+1].set(unext); 
    # Remove Walls
    u = u[1:-1,:]; 
    return u

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