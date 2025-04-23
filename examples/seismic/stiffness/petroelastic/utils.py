# ------------------------------ #

# ----------- IMPORT ----------- #

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import segyio
import cv2

from examples.seismic.stiffness.utils import C_Matrix, D, S, vec
from examples.seismic import Model, TimeAxis, RickerSource, Receiver

from devito import TimeFunction, VectorTimeFunction, TensorTimeFunction, Eq, solve, Operator, NODE
from devito.finite_differences.operators import div, grad

# ----------- MODELS ----------- #

def load_marmousi(dir='./marmousi_elastic', down_scale=16, origin=(0,0), space_order=8, nbl=50, PCS=False):
    marmousi = {}

    for param in ['vp', 'vs', 'rho']:
        file = segyio.open(f'{dir}/marmousi_{param}.segy')
        
        shape = (len(file.samples), len(file.ilines))
        arr = np.zeros(shape)

        down_shape = (np.asarray(shape) / down_scale).astype('int16')

        for i, trace in enumerate(file.trace):
            arr[:, i] = trace

        arr_down = cv2.resize(arr, (down_shape[1], down_shape[0]), interpolation=cv2.INTER_AREA)

        marmousi[param] = arr_down
        
    nx, nz = down_shape[1], down_shape[0]
    dx, dz = int(1.25 * down_scale), int(1.25 * down_scale)

    marmousi['vp'] /= 1000
    marmousi['vs'] /= 1000

    if PCS == True:
        print('Generating PCS model...')
        cc = 0.1 + 0.01 * marmousi['rho']

        Phi1 = (marmousi['vp'] - 5.59 + 2.18 * cc) / -6.93
        Phi2 = (marmousi['vs'] - 3.52 + 1.89 * cc) / -4.91
        Phi = (Phi1 + Phi2)/2

        Sw = np.ones_like(Phi1)

        model = Model(Phi=Phi.T, cc=cc.T, Sw=Sw.T, origin=origin, shape=(nx,nz), spacing=(dx,dz), space_order=space_order, nbl=nbl, bcs='damp')

    elif PCS == 'Han':
        print('Generating PCS (Han) model...')
        a1, a2, a3, b1, b2, b3 = 5.5, 6.9, 2.2, 3.4, 4.7, 1.8
        rhoq, rhoc, rhow, rhoh = 2.65, 2.55, 1., 0.1

        cc = (a2 * (marmousi['vs'] - b1) + b2 * (a1 - marmousi['vp'])) / (a3 * b2 - a2 * b3)
        Phi = (a1 - a3 * cc - marmousi['vp']) / a2
        Sw = (((marmousi['rho'] - (cc * (rhoc - rhoq) + rhoq) * (1 - Phi)) / Phi) - rhoh) / (rhow - rhoh)

        model = Model(Phi=Phi.T, cc=cc.T, Sw=Sw.T, origin=origin, shape=(nx,nz), spacing=(dx,dz), space_order=space_order, nbl=nbl, bcs='damp')

    else:
        print('Loading elastic model...')
        model = Model(vp=marmousi['vp'].T, vs=marmousi['vs'].T, b=1/marmousi['rho'].T, origin=origin, shape=(nx,nz), spacing=(dx,dz), space_order=space_order, nbl=nbl, bcs='damp')

    return model

# ---------- OPERATOR ---------- #

def elastic_forward(model, src, rec, param='lam-mu', src_direction=None, save=True):
    dt = src.time_range.step
    s = model.grid.stepping_dim.spacing
    damp = 1 - model.damp
    save = src.time_range.num if save else None

    if param.startswith('PCS'):
        rho_c, rho_q, rho_w, rho_h = 2.55, 2.65, 1, 0.1
        rho_m = model.cc * (rho_c - rho_q) + rho_q
        rho_f = model.Sw * (rho_w - rho_h) + rho_h
        rho = model.Phi * (rho_f - rho_m) + rho_m
    else:
        rho = 1 / model.b

    C = C_Matrix(model, param)
    V = VectorTimeFunction(name='V', grid=model.grid, time_order=1, space_order=model.space_order, save=save)
    tau = TensorTimeFunction(name='tau', grid=model.grid, time_order=1, space_order=model.space_order, save=save)
    tau = vec(tau)

    pde0 = V.dt * rho - D(tau)
    stencil0 = Eq(V.forward, damp * solve(pde0, V.forward))

    pde1 = tau.dt - C * S(V.forward)
    stencil1 = Eq(tau.forward, damp * solve(pde1, tau.forward))

    # Injetando a fonte
    if src_direction == None:
        src_xx = src.inject(field=tau[0].forward, expr=src * s)
        src_zz = src.inject(field=tau[1].forward, expr=src * s)
        src_xz = src.inject(field=tau[2].forward, expr=src * s)
        src_term = src_xx + src_zz
    elif src_direction == 'x':
        src_xx = src.inject(field=tau[0].forward, expr=src * s)
        src_term = src_xx
    elif src_direction == 'z':
        src_zz = src.inject(field=tau[1].forward, expr=src * s)
        src_term = src_zz
    elif src_direction == 'c':
        src_xz = src.inject(field=tau[2].forward, expr=src * s)
        src_term = src_xz

    # Interpolando receptores
    rec_term_vx = rec[0].interpolate(expr=V[0])
    rec_term_vz = rec[1].interpolate(expr=V[1])
    rec_term_tau = rec[2].interpolate(expr=tau[0] + tau[1])
    rec_expr = rec_term_vx + rec_term_vz + rec_term_tau

    op = Operator([stencil0, stencil1] + src_term + rec_expr, subs=model.spacing_map)

    op(dt=dt)

    return V, tau, rec

# ---------- GEOMETRY ---------- #

def load_setup(model, tn, dt=None, f0=0.010, ns=1, ng=None, s_pos=None, g_pos=None, order=1):
    '''

    '''
    dt = model.critical_dt if dt == None else dt
    ng = int(model.shape[0]) if ng == None else ng

    if s_pos == None:
        s_pos = np.zeros((ns,2))
        s_pos[:, 0] = model.domain_size[0] * 0.5 # Posição dos receptores (x)
        s_pos[:, 1] = model.origin[-1] + model.spacing[-1]

    if g_pos == None:
        g_pos = np.zeros((ng,2)) # Posição dos receptores
        g_pos[:, 0] = np.linspace(model.origin[0], model.domain_size[0], ng) # Posição dos receptores (x)
        g_pos[:, 1] = model.origin[-1] + 2 * model.spacing[-1]

    # Definindo a fonte (Ricker)
    time_range = TimeAxis(start=0, stop=tn, step=dt)
    src = RickerSource(name='src', grid=model.grid, f0=f0, time_range=time_range)
    src.coordinates.data[:,:] = s_pos

    # Definindo os receptores
    if order == 1:
        rec_vx = Receiver(name='rec_vx', grid=model.grid, npoint=ng, time_range=time_range)
        rec_vz = Receiver(name='rec_vz', grid=model.grid, npoint=ng, time_range=time_range)
        rec_tau = Receiver(name='rec_tau', grid=model.grid, npoint=ng, time_range=time_range)

        rec = [rec_vx, rec_vz, rec_tau]

        for rec_ in rec:
            rec_.coordinates.data[:,:] = g_pos

    return src, rec

# ---------- PLOTTING ---------- #

def plot_model(model, params=['vp','vs','rho'], figsize=None, save_path=None):
    cols = len(params)
    figsize = figsize if figsize != None else (cols * 3, 2.5)

    fig, axes = plt.subplots(1, cols, figsize=figsize, sharey=True, sharex=True)

    nbl_size = (model.grid.extent[0] - model.domain_size[0]) // 2
    extent = [-nbl_size, model.domain_size[0] + nbl_size, model.domain_size[1] + nbl_size, -nbl_size]

    for i, (param, ax) in enumerate(zip(params, axes)):
        img = ax.imshow(getattr(model, param).data.T, extent=extent, cmap='nipy_spectral', aspect='auto')

        ax.set_title(param)
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Depth (m)') if i == 0 else None

        cbar = fig.colorbar(img)

        rect = patches.Rectangle([0, 0], model.domain_size[0], model.domain_size[1], linestyle='--', linewidth=1, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

    fig.tight_layout()
    plt.show()

    fig.savefig(save_path, dpi=300) if save_path != None else None

def plot_setup(model, src, rec, rec_step=4, param='vp'):
    nx, nz = model.shape
    dx, dz = model.spacing
    dt = src.time_range.step

    plot_options = {'extent':[0, nx * dx, nz * dz, 0], 'cmap':'jet'}

    fig, axes = plt.subplots(1, 2, figsize=(9,4), gridspec_kw={'width_ratios':[4,1]})

    img = axes[0].imshow(getattr(model, param).data.T, **plot_options)
    axes[0].scatter(rec.coordinates.data[::rec_step,0], rec.coordinates.data[::rec_step,1], c='green', marker='v')
    axes[0].scatter(src.coordinates.data[0,0], src.coordinates.data[0,1], c='red', marker='x')
    axes[0].set_title('Geometria de Aquisição')
    axes[0].set_xlabel('Distância (m)')
    axes[0].set_ylabel('Profundidade (m)')
    cbar = fig.colorbar(img)

    axes[1].plot(src.data, src.time_values)
    axes[1].set_xlabel('Amplitude')
    axes[1].set_ylabel('Tempo (ms)')
    axes[1].set_title('Wavelet')
    axes[1].invert_yaxis()

    fig.tight_layout()
    plt.show()

# ------------------------------ #