import mat73
import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO, BCSR
from jax.experimental.sparse.linalg import spsolve
from jax.experimental import host_callback as hcb
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import time

#jax.config.update("jax_enable_x64", True)

def stencilOptParams(vmin,vmax,f,h,g):
#STENCILOPTPARAMS Optimal Params for 9-Point Stencil 
#   INPUTS:
#       vmin = minimum wave velocity [L/T]
#       vmax = maximum wave velocity [L/T]
#       f = frequency [1/T]
#       h = grid spacing in X [L]
#       g = (grid spacing in Y [L])/(grid spacing in X [L])
#   OUTPUTS:
#       b, d, e = optimal params according to Chen/Cheng/Feng/Wu 2013 Paper

    l = 100
    r = 10

    Gmin = vmin / (f * h)
    Gmax = vmax / (f * h)

    # MATLAB m = 1:l → Python 1,2,…,l
    m = jnp.arange(1, l+1)
    # MATLAB n = 1:r → Python 1,2,…,r
    n = jnp.arange(1, r+1)

    # theta = (m-1)*pi/(4*(l-1))
    theta = (m - 1) * jnp.pi / (4 * (l - 1))

    # G = 1./(1/Gmax + ((n-1)/(r-1))*(1/Gmin-1/Gmax));
    G = 1.0 / (
        1.0 / Gmax +
        ((n - 1) / (r - 1)) * (1.0 / Gmin - 1.0 / Gmax)
    )

    # replicated exactly as [TH,GG]=meshgrid(theta,G)
    TH, GG = jnp.meshgrid(theta, G, indexing='xy')

    # the four stencil-estimator summands
    P = jnp.cos(g * 2 * jnp.pi * jnp.cos(TH) / GG)
    Q = jnp.cos(2 * jnp.pi * jnp.sin(TH) / GG)

    S1 = (1 + 1/(g**2)) * (GG**2) * (1 - P - Q + P*Q)
    S2 = (jnp.pi**2)    * (2 - P - Q)
    S3 = (2*jnp.pi**2)  * (1 - P*Q)
    S4 = 2*jnp.pi**2 + (GG**2) * (
         (1 + 1/(g**2))*P*Q - P - Q/(g**2)
    )

    fixB = True
    if fixB:
        b = 5/6
        A = jnp.stack([S2.ravel(), S3.ravel()], axis=1)     # (M,2)
        y = S4.ravel() - b * S1.ravel()                     # (M,)
        # solve (AᵀA)·params = Aᵀy
        ATA    = A.T @ A
        ATy    = A.T @ y
        params = jnp.linalg.solve(ATA, ATy)                 # (2,)
        d, e = params[0], params[1]
    else:
        A = jnp.stack([S1.ravel(), S2.ravel(), S3.ravel()], axis=1)  # (M,3)
        y = S4.ravel()
        ATA    = A.T @ A
        ATy    = A.T @ y
        params = jnp.linalg.solve(ATA, ATy)                          # (3,)
        b, d, e = params[0], params[1], params[2]

    return b, d, e

def solveHelmholtz(x, y, vel, src, f, a0, L_PML, adjoint):
    """
    Solve 2D Helmholtz with a 9-point optimized stencil & PML in JAX.
    Inputs:
      x:        (Nx,) 1D array of grid-x
      y:        (Ny,) 1D array of grid-y
      vel:      (Ny,Nx) wave velocity map
      src:      (Ny,Nx,S) source array (S shots)
      f:        scalar frequency
      a0:       PML strength
      L_PML:    PML thickness
      adjoint:  bool, True→solve H^H, False→solve H
    Returns:
      wvfield:  (Ny,Nx,S) solved wavefields
    """
    sign_convention = -1
    h = jnp.mean(jnp.diff(x))
    gh = jnp.mean(jnp.diff(y))
    g = gh / h
    Nx, Ny = x.size, y.size
    k = 2 * jnp.pi * f / vel

    xmin, xmax = x[0], x[-1]
    ymin, ymax = y[0], y[-1]
    xe = jnp.linspace(xmin, xmax, 2 * (Nx - 1) + 1)
    ye = jnp.linspace(ymin, ymax, 2 * (Ny - 1) + 1)
    Xe, Ye = jnp.meshgrid(xe, ye, indexing="xy")

    xctr, xspan = (xmin + xmax) / 2, (xmax - xmin) / 2
    yctr, yspan = (ymin + ymax) / 2, (ymax - ymin) / 2

    sx = 2 * jnp.pi * a0 * f * ((jnp.maximum(jnp.abs(Xe - xctr) - xspan + L_PML, 0) / L_PML) ** 2)
    sy = 2 * jnp.pi * a0 * f * ((jnp.maximum(jnp.abs(Ye - yctr) - yspan + L_PML, 0) / L_PML) ** 2)

    ex = 1 + 1j * sx * jnp.sign(sign_convention) / (2 * jnp.pi * f)
    ey = 1 + 1j * sy * jnp.sign(sign_convention) / (2 * jnp.pi * f)

    A = (ey / ex)[::2, 1::2]
    B = (ex / ey)[1::2, ::2]
    C = (ex * ey)[::2, ::2]

    b, d, e = stencilOptParams(jnp.min(vel), jnp.max(vel), f, h, g)

    def lin_idx(x, y):
        return y * Nx + x

    xs = jnp.arange(1, Nx - 1)
    ys = jnp.arange(1, Ny - 1)
    X, Y = jnp.meshgrid(xs, ys, indexing="xy")
    Xf, Yf = X.ravel(), Y.ravel()

    Xc, Yc = Xf, Yf
    Xl, Yl = Xf - 1, Yf
    Xr, Yr = Xf + 1, Yf
    Xd, Yd = Xf, Yf - 1
    Xu, Yu = Xf, Yf + 1
    Xdl, Ydl = Xf - 1, Yf - 1
    Xdr, Ydr = Xf + 1, Yf - 1
    Xul, Yul = Xf - 1, Yf + 1
    Xur, Yur = Xf + 1, Yf + 1

    idx_c = lin_idx(Xc, Yc)
    idx_l = lin_idx(Xl, Yl)
    idx_r = lin_idx(Xr, Yr)
    idx_d = lin_idx(Xd, Yd)
    idx_u = lin_idx(Xu, Yu)
    idx_dl = lin_idx(Xdl, Ydl)
    idx_dr = lin_idx(Xdr, Ydr)
    idx_ul = lin_idx(Xul, Yul)
    idx_ur = lin_idx(Xur, Yur)
    rows_int = jnp.repeat(idx_c, 9)
    cols_int = jnp.stack([idx_c, idx_l, idx_r, idx_d, idx_u, idx_dl, idx_dr, idx_ul, idx_ur], axis=1).ravel()

    def gath(A, Y, X): return A[Y, X]
    def sq(val): return val ** 2

    val_c = (1 - d - e) * gath(C, Yc, Xc) * sq(gath(k, Yc, Xc)) - b * (gath(A, Yc, Xc) + gath(A, Yl, Xl) + gath(B, Yc, Xc) / g ** 2 + gath(B, Yd, Xd) / g ** 2) / h ** 2
    val_l = (b * gath(A, Yl, Xl) - ((1 - b) / 2) * (gath(B, Yl, Xl) / g ** 2 + gath(B, Ydl, Xdl) / g ** 2)) / h ** 2 + (d / 4) * gath(C, Yl, Xl) * sq(gath(k, Yl, Xl))
    val_r = (b * gath(A, Yc, Xc) - ((1 - b) / 2) * (gath(B, Yr, Xr) / g ** 2 + gath(B, Ydr, Xdr) / g ** 2)) / h ** 2 + (d / 4) * gath(C, Yr, Xr) * sq(gath(k, Yr, Xr))
    val_d = (b * gath(B, Yd, Xd) / g ** 2 - ((1 - b) / 2) * (gath(A, Yd, Xd) + gath(A, Ydl, Xdl))) / h ** 2 + (d / 4) * gath(C, Yd, Xd) * sq(gath(k, Yd, Xd))
    val_u = (b * gath(B, Yc, Xc) / g ** 2 - ((1 - b) / 2) * (gath(A, Yu, Xu) + gath(A, Yul, Xul))) / h ** 2 + (d / 4) * gath(C, Yu, Xu) * sq(gath(k, Yu, Xu))
    val_dl = ((1 - b) / 2) * (gath(A, Ydl, Xdl) + gath(B, Ydl, Xdl) / g ** 2) / h ** 2 + (e / 4) * gath(C, Ydl, Xdl) * sq(gath(k, Ydl, Xdl))
    val_dr = ((1 - b) / 2) * (gath(A, Ydr, Xdr) + gath(B, Ydr, Xdr) / g ** 2) / h ** 2 + (e / 4) * gath(C, Ydr, Xdr) * sq(gath(k, Ydr, Xdr))
    val_ul = ((1 - b) / 2) * (gath(A, Yul, Xul) + gath(B, Yul, Xul) / g ** 2) / h ** 2 + (e / 4) * gath(C, Yul, Xul) * sq(gath(k, Yul, Xul))
    val_ur = ((1 - b) / 2) * (gath(A, Yur, Xur) + gath(B, Yur, Xur) / g ** 2) / h ** 2 + (e / 4) * gath(C, Yur, Xur) * sq(gath(k, Yur, Xur))

    vals_int = jnp.stack([val_c, val_l, val_r, val_d, val_u, val_dl, val_dr, val_ul, val_ur], axis=1).ravel()

    x0 = jnp.arange(Nx)
    y0 = jnp.arange(Ny)
    idx_top = lin_idx(x0, 0)
    idx_bot = lin_idx(x0, Ny - 1)
    idx_left = lin_idx(0, y0[1:-1])
    idx_right = lin_idx(Nx - 1, y0[1:-1])
    idx_bdr = jnp.concatenate([idx_top, idx_bot, idx_left, idx_right])
    rows_bdr = idx_bdr
    cols_bdr = idx_bdr
    vals_bdr = jnp.ones_like(idx_bdr, dtype=A.dtype)

    rows = jnp.concatenate([rows_int, rows_bdr], axis=0)
    cols = jnp.concatenate([cols_int, cols_bdr], axis=0)
    vals = jnp.concatenate([vals_int, vals_bdr], axis=0)

    H_bcoo = BCOO((vals, jnp.stack([rows, cols], axis=1)), shape=(Nx * Ny, Nx * Ny))

    H_use = jax.lax.cond(
        adjoint,
        lambda H: BCOO((jnp.conj(H.transpose().data), H.transpose().indices), shape=H.shape),
        lambda H: H,
        H_bcoo,
    )

    H_use = BCSR.from_bcoo(H_use)
    rhs = jnp.reshape(src, (Nx * Ny, -1))
    rhs = jnp.array(rhs, dtype=jnp.complex64)
    data, indices, indptr = H_use.data, H_use.indices, H_use.indptr
    print('Solving start ...')
    sol = jnp.stack(
        [spsolve(data, indices, indptr, rhs[:, i]) for i in range(rhs.shape[1])], axis=1
    )
    # S = rhs.shape[1]
    # sol_cols = []

    # for i in range(S):
    #     t0 = time.perf_counter()
    #     col_sol = spsolve(data, indices, indptr, rhs[:, i], tol=1e-6, reorder=2)
    #     dt = time.perf_counter() - t0

    #     print(f"Shot {i:3d} solve time: {dt*1000:.2f} ms")
    #     sol_cols.append(col_sol)

    # # now stack into final solution
    # sol = jnp.stack(sol_cols, axis=1)
    print('Solving end')
    return sol.reshape(Ny, Nx, -1)


#Load data problem
data = mat73.loadmat('RecordedData_test.mat', use_attrdict=True)
x      = jnp.array(data['x'],      dtype=jnp.float32)
y        = jnp.array(data['y'],      dtype=jnp.float32)
C        = jnp.array(data['C'],      dtype=jnp.float32)
x_circ   = jnp.array(data['x_circ'], dtype=jnp.float32)
y_circ   = jnp.array(data['y_circ'], dtype=jnp.float32)
f_data   = jnp.array(data['f'],      dtype=jnp.float32)

REC_DATA = data['REC_DATA']
#print(REC_DATA)

numElements = x_circ.size
assert numElements == y_circ.size, \
       "x_circ and y_circ must have the same length"

# Which subset of transmit to use
dwnsmp = 1
tx_include = jnp.arange(0,numElements,dwnsmp)
REC_DATA = REC_DATA[tx_include,:]

# Extract Subset of Signals within Acceptance Angle
numElemLeftRightExcl = 3
elemLeftRightExcl    = jnp.arange(-numElemLeftRightExcl,numElemLeftRightExcl + 1)
elem_include         = jnp.ones((numElements, numElements),dtype=bool)

for tx_element in range(numElements):
    elemLeftRightExclCurrent = (elemLeftRightExcl + tx_element)
    elem_include = elem_include.at[tx_element, elemLeftRightExclCurrent].set(False)

#Parameters for Conjugate Gradient Reconstruction
Niter = 1 #Number of Iterations
momentumFormula = 4 #Momentum Formula for Conjugate Gradient
                    # 0 -- No Momentum (Gradient Descent)
                    # 1 -- Fletcher-Reeves (FR)
                    # 2 -- Polak-Ribiere (PR)
                    # 3 -- Combined FR + PR
                    # 4 -- Hestenes-Stiefel (HS)
stepSizeCalculation = 1 #Which Step Size Calculation:
                        # 1 -- Not Involving Gradient Nor Search Direction
                        # 2 -- Involving Gradient BUT NOT Search Direction
                        # 3 -- Involving Gradient AND Search Direction
c_init = 1480 # Initial Homogeneous Sound Speed [m/s] Guess

#Computational Grid (and Element Placement on Grid) for Reconstruction
dxi = 0.8e-3
xmax = 120e-3
xi = jnp.arange(-xmax,xmax+dxi,dxi) 
yi = xi.copy()
Nxi = xi.size
Nyi = yi.size
[Xi, Yi] = jnp.meshgrid(xi, yi)

xc = x_circ.ravel()   # shape (M,)
yc = y_circ.ravel()   # shape (M,)

x_idx = jnp.argmin(jnp.abs(xi[None, :] - xc[:, None]), axis=1)
y_idx = jnp.argmin(jnp.abs(yi[None, :] - yc[:, None]), axis=1)

ind = y_idx * Nxi + x_idx #Row majo

#Solver Options for Helmholtz Equation
a0 = 10.0 #PML Constant
L_PML = 9.0e-3 #Thickness of PML  

#Generate Sources
SRC = jnp.zeros((Nyi, Nxi, tx_include.size), dtype=jnp.float32)

for tx_elmt_idx in range(tx_include.size):
    #Single Element Source
    x_idx_src = x_idx[tx_include[tx_elmt_idx]]
    y_idx_src = y_idx[tx_include[tx_elmt_idx]] 
    SRC = SRC.at[y_idx_src, x_idx_src, tx_elmt_idx].set(1)

#(Nonlinear) Conjugate Gradient
search_dir = jnp.zeros((Nyi,Nxi)) # Conjugate Gradient Direction
gradient_img_prev = jnp.zeros((Nyi,Nxi)) # Previous Gradient Image
VEL_ESTIM = c_init*jnp.ones((Nyi,Nxi)) # Initial Sound Speed Image [m/s]
SLOW_ESTIM = 1./VEL_ESTIM # Initial Slowness Image [s/m]
crange = jnp.array([1400, 1600]) # For reconstruction display [m/s]

for iter in range(Niter):
    # (1A) Solve forward Helmholtz
    t0 = time.time()
    WVFIELD = solveHelmholtz(xi, yi, VEL_ESTIM, SRC, f_data, a0, L_PML, False)
    
    # (1B) Estimate forward sources
    SRC_ESTIM = jnp.zeros((tx_include.size,), dtype=jnp.complex64)
    for tx_elmt_idx in range(tx_include.size):
        # extract the single‐shot wavefield and flatten
        wv = WVFIELD[..., tx_elmt_idx].ravel()                    # length = Nyi*Nxi
        mask = elem_include[tx_include[tx_elmt_idx], :]           # length = numElements
        rec_sim = wv[ind[mask]]                                   # simulated rec for included elems
        rec     = REC_DATA[tx_elmt_idx, mask]                     # measured rec
        # source estimate = (rec_sim' * rec) / (rec_sim' * rec_sim)
        num   = jnp.vdot(rec_sim, rec)
        denom = jnp.vdot(rec_sim, rec_sim) + 1e-12
        SRC_ESTIM = SRC_ESTIM.at[tx_elmt_idx].set((num/denom))
    # scale the forward wavefield by the estimated source amplitudes
    WVFIELD = WVFIELD * SRC_ESTIM[None, None, :]

    # (1C) Build adjoint sources (data‐error)
    ADJ_SRC  = jnp.zeros_like(WVFIELD)
    REC_SIM2 = jnp.zeros((tx_include.size, numElements), dtype=jnp.complex64)
    for tx_elmt_idx in range(tx_include.size):
        wv = WVFIELD[..., tx_elmt_idx].ravel()
        mask = elem_include[tx_include[tx_elmt_idx], :]
        vals = wv[ind[mask]]                         # forward‐projected rec
        REC_SIM2 = REC_SIM2.at[tx_elmt_idx, mask].set(vals)
        err = vals - REC_DATA[tx_elmt_idx, mask]
        # scatter error back into grid
        flat_adj = jnp.zeros((Nyi*Nxi,), dtype=err.dtype)
        flat_adj = flat_adj.at[ind[mask]].set(err)
        ADJ_SRC = ADJ_SRC.at[..., tx_elmt_idx].set(
            flat_adj.reshape((Nyi, Nxi))
        )

    # (1D) Virtual source
    VIRT_SRC = (2*(2*jnp.pi*f_data)**2) * SLOW_ESTIM[..., None] * WVFIELD

    # (1E) Backproject error → gradient
    ADJ_WVFIELD = solveHelmholtz(xi, yi, VEL_ESTIM, ADJ_SRC, f_data, a0, L_PML, True)
    BACKPROJ    = -jnp.real(jnp.conj(VIRT_SRC) * ADJ_WVFIELD)
    gradient_img = jnp.sum(BACKPROJ, axis=2)

    # (2A) Conjugate‐gradient momentum β
    if iter == 1 or momentumFormula == 0:
        beta = 0.0
    else:
        gdotg   = jnp.vdot(gradient_img, gradient_img)
        gprev2  = jnp.vdot(gradient_img_prev, gradient_img_prev) + 1e-12
        if momentumFormula == 1:           # Fletcher‐Reeves
            beta = gdotg / gprev2
        elif momentumFormula == 2:         # Polak‐Ribiere
            diff = gradient_img - gradient_img_prev
            beta = jnp.vdot(gradient_img, diff) / gprev2
        elif momentumFormula == 3:         # combined
            diff = gradient_img - gradient_img_prev
            betaPR = jnp.vdot(gradient_img, diff) / gprev2
            betaFR = gdotg / gprev2
            beta   = jnp.clip(betaPR, 0, betaFR)
        else:                              # Hestenes‐Stiefel
            diff = gradient_img - gradient_img_prev
            beta = jnp.vdot(gradient_img, diff) / (jnp.vdot(search_dir, diff) + 1e-12)

    # (2B) Update search direction
    search_dir        = beta * search_dir - gradient_img
    gradient_img_prev = gradient_img

    # (3) Forward project search direction
    PERT_SRC = -VIRT_SRC * search_dir[None, ..., None]
    PERT_WV  = solveHelmholtz(xi, yi, VEL_ESTIM, PERT_SRC, f_data, a0, L_PML, False)

    dREC_SIM = jnp.zeros((tx_include.size, numElements), dtype=jnp.complex64)
    for tx_elmt_idx in range(tx_include.size):
        wv = PERT_WV[..., tx_elmt_idx].ravel()
        mask = elem_include[tx_include[tx_elmt_idx], :]
        vals = wv[ind[mask]]
        dREC_SIM = dREC_SIM.at[tx_elmt_idx, mask].set(vals)

    # (4) Step‐size via line search
    REC_SIM_flat = REC_SIM2.ravel()
    dREC_flat    = dREC_SIM.ravel()
    g_flat       = gradient_img.ravel()
    sd_flat      = search_dir.ravel()

    if stepSizeCalculation == 1:
        stepSize = jnp.real((dREC_flat @ (REC_DATA.ravel() - REC_SIM_flat))
                            / (dREC_flat @ dREC_flat))
    elif stepSizeCalculation == 2:
        stepSize = (g_flat @ g_flat) / (dREC_flat @ dREC_flat)
    else:  # involving search direction
        stepSize = -(g_flat @ sd_flat) / (dREC_flat @ sd_flat)

    # update slowness & velocity
    SLOW_ESTIM = SLOW_ESTIM + stepSize * search_dir
    VEL_ESTIM  = 1.0 / jnp.real(SLOW_ESTIM)

    # visualize (2×2)
    plt.subplot(2,2,1)
    plt.imshow(C,       extent=[xi.min(), xi.max(), yi.max(), yi.min()], 
               cmap='gray', vmin=crange[0], vmax=crange[1])
    plt.title('True Sound Speed'); plt.axis('equal')

    plt.subplot(2,2,2)
    plt.imshow(VEL_ESTIM, extent=[xi.min(), xi.max(), yi.max(), yi.min()],
               cmap='gray', vmin=crange[0], vmax=crange[1])
    plt.title(f'Estimate (iter {iter})'); plt.axis('equal')

    plt.subplot(2,2,3)
    plt.imshow(search_dir, extent=[xi.min(), xi.max(), yi.max(), yi.min()],
               cmap='gray'); plt.title('Search Dir'); plt.axis('equal')

    plt.subplot(2,2,4)
    plt.imshow(-gradient_img, extent=[xi.min(), xi.max(), yi.max(), yi.min()],
               cmap='gray'); plt.title('Gradient'); plt.axis('equal')

    plt.suptitle(f'Iteration {iter} (t={time.time()-t0:.2f}s)')
    plt.draw(); plt.pause(1e-3)
    plt.show()