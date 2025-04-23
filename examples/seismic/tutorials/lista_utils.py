# ----------------------------------- #

import matplotlib.pyplot as plt
import numpy as np
import matplotlib

from examples.seismic.stiffness.utils import C_Matrix, D, S, vec
from examples.seismic import Model, plot_velocity, TimeAxis, RickerSource, Receiver
from devito import TimeFunction, VectorTimeFunction, TensorTimeFunction, Eq, solve, Operator, NODE
from devito.finite_differences.operators import div, grad

from matplotlib.animation import FuncAnimation

# Definindo funções

def get_src_rec(model, t0, tn, dt, f0, ns, ng, s_pos, g_pos, order=1):
    # Definindo a fonte (Ricker)
    time_range = TimeAxis(start=t0, stop=tn, step=dt)
    src = RickerSource(name='src', grid=model.grid, f0=f0, time_range=time_range)
    src.coordinates.data[:,:] = s_pos

    # Definindo os receptores
    if order == 1:
        rec_vx = Receiver(name='rec_vx', grid=model.grid, npoint=ng, time_range=time_range)
        rec_vz = Receiver(name='rec_vz', grid=model.grid, npoint=ng, time_range=time_range)
        rec_sigma = Receiver(name='rec_sigma', grid=model.grid, npoint=ng, time_range=time_range)

        rec = [rec_vx, rec_vz, rec_sigma]

        for rec_ in rec:
            rec_.coordinates.data[:,:] = g_pos

    elif order == 2:
        rec = Receiver(name='rec', grid=model.grid, npoint=ng, time_range=time_range)
        rec.coordinates.data[:,:] = g_pos

    return src, rec

def acoustic_forward(model, src, rec, order=2):
    dt = src.time_range.step

    if order == 2:        
        # Definindo equação acústica de 2a ordem
        P = TimeFunction(name="P", grid=model.grid, time_order=2, space_order=model.space_order, save=src.time_range.num) # Campo de onda

        if len(np.unique(model.b.data)) == 1:    
            pde = model.m * P.dt2 - P.laplace + model.damp * P.dt # Equação Diferencial Parcial
            stencil = Eq(P.forward, solve(pde, P.forward)) # Solução por Diferenças Finitas
        else:
            rho = 1 / model.b
            kappa = rho * model.vp**2
            pde = P.dt2 - kappa * div(model.b * grad(P, shift=0.5), shift=-0.5) + model.damp * P.dt # Equação Diferencial Parcial
            stencil = Eq(P.forward, solve(pde, P.forward)) # Solução por Diferenças Finitas

        # Injetando a fonte no campo de pressão e interpolando os receptores
        src_term = src.inject(field=P.forward, expr=src * dt**2 / model.m) # Injeção da fonte no campo P(x,z,t+1)
        rec_term = rec.interpolate(expr=P.forward) # Interpolação do campo P(x,z,t+1) para os receptores

        op = Operator([stencil] + src_term + rec_term, subs=model.spacing_map)
        op(dt=dt)

        return P, rec

    elif order == 1:
        # Definindo a equação acústica de 1a ordem
        V = VectorTimeFunction(name='V', grid=model.grid, time_order=1, space_order=model.space_order, save=src.time_range.num) # Campo vetorial de velocidade
        P = TimeFunction(name="P", grid=model.grid, time_order=1, space_order=model.space_order, save=src.time_range.num, staggered=NODE) # Campo de pressão

        damp = 1 - model.damp
        rho = 1 / model.b
        kappa = rho * model.vp**2

        pde0 = V.dt + model.b *  grad(P) # Equação Diferencial Parcial do campo de velocidade
        pde1 = P.dt + kappa * div(V.forward) # Equação Diferencial Parcial do campo de pressão

        # Reescrevendo a função discretizada para resolver iterativamente para o termo P(x,z,t+1)
        stencil0 = Eq(V.forward, damp * solve(pde0, V.forward))
        stencil1 = Eq(P.forward, damp * solve(pde1, P.forward))

        # Injetando a fonte no e interpolando os receptores
        src_term = src.inject(field=P.forward, expr=src * model.grid.stepping_dim.spacing) # Injeção da fonte no campo P(x,z,t+1)

        rec_term_vx = rec[0].interpolate(expr=V[0])
        rec_term_vz = rec[1].interpolate(expr=V[1])
        rec_term_p = rec[2].interpolate(expr=P)

        rec_expr = rec_term_vx + rec_term_vz + rec_term_p

        # Definindo o operador Devito para resolver a equação e executando
        op = Operator([stencil0, stencil1] + src_term + rec_expr, subs=model.spacing_map)
        op(dt=dt)

        return V, P, rec

def elastic_forward(model, src, rec, order=1, param='lam-mu', src_direction=None, save=True):
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

    if order == 1:
        C = C_Matrix(model, param)
        V = VectorTimeFunction(name='V', grid=model.grid, time_order=1, space_order=model.space_order, save=save)
        sigma = TensorTimeFunction(name='sigma', grid=model.grid, time_order=1, space_order=model.space_order, save=save)
        sigma = vec(sigma)

        pde0 = V.dt * rho - D(sigma)
        stencil0 = Eq(V.forward, damp * solve(pde0, V.forward))

        pde1 = sigma.dt - C * S(V.forward)
        stencil1 = Eq(sigma.forward, damp * solve(pde1, sigma.forward))

        # Injetando a fonte
        if src_direction == None:
            src_xx = src.inject(field=sigma[0].forward, expr=src * s)
            src_zz = src.inject(field=sigma[1].forward, expr=src * s)
            src_xz = src.inject(field=sigma[2].forward, expr=src * s)
            src_term = src_xx + src_zz
        elif src_direction == 'x':
            src_xx = src.inject(field=sigma[0].forward, expr=src * s)
            src_term = src_xx
        elif src_direction == 'z':
            src_zz = src.inject(field=sigma[1].forward, expr=src * s)
            src_term = src_zz
        elif src_direction == 'c':
            src_xz = src.inject(field=sigma[2].forward, expr=src * s)
            src_term = src_xz

        # Interpolando receptores
        rec_term_vx = rec[0].interpolate(expr=V[0])
        rec_term_vz = rec[1].interpolate(expr=V[1])
        rec_term_sigma = rec[2].interpolate(expr=sigma[0] + sigma[1])
        rec_expr = rec_term_vx + rec_term_vz + rec_term_sigma

        op = Operator([stencil0, stencil1] + src_term + rec_expr, subs=model.spacing_map)

        op(dt=dt)

        return V, sigma, rec

def plot_aquisition_setup(model, src, rec, rec_step=4, param='vp'):
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

def plot_snaps(P, model, src, plot_t0, plot_tn, plot_dt, cols=5, vmin_max=None, factor=1):
    nx, nz = model.shape
    dx, dz = model.spacing
    nbl = model.nbl

    t0 = src.time_range.start
    dt = src.time_range.step

    plot_nt = int((plot_tn - plot_t0) / plot_dt)
    rows = int(plot_nt / cols) + 1

    if vmin_max == None:
        vmin = -max([abs(P.data.min()), P.data.min()]) / factor
        vmax = -vmin
    else:
        vmin = vmin_max[0]
        vmax = vmin_max[1]

    plot_options = {'extent':[-nbl * dx, (nx + nbl) * dx, (nz + nbl) * dz, -nbl * dz], 'cmap':'seismic', 'vmin':vmin, 'vmax':vmax}

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))

    i = int((plot_t0 - t0) / dt)
    for ax in axes.flatten():
        try:
            img = ax.imshow(P.data[i].T, **plot_options)
            ax.set_title(f'Tempo: {round(src.time_values[i], 2)}ms')
            ax.set_xlabel('Distância (m)')
            ax.set_ylabel('Profundidade (m)')
            fig.colorbar(img)

            i += int(plot_dt / dt)
        except:
            pass

    fig.tight_layout()
    plt.show()

def plot_video(P, rec, model, frames=None, interval=10, factor=1):
    nx, nz = model.shape
    dx, dz = model.spacing
    nbl = model.nbl
    t0, tn = rec.time_range.start, rec.time_range.stop
    dt = rec.time_range.step

    matplotlib.rcParams['animation.embed_limit'] = 2**128

    rec_mask = np.zeros_like(rec.data)

    vmin = -max([abs(P.data.min()), P.data.min()]) / factor
    vmax = -vmin
    plot_options0 = {'extent':[-nbl * dx, (nx + nbl) * dx, (nz + nbl) * dz, -nbl * dz], 'cmap':'seismic', 'vmin':vmin, 'vmax':vmax}

    vmin = -max([abs(rec.data.min()), rec.data.min()])
    vmax = -vmin
    plot_options1 = {'extent':[-nbl * dx, (nx + nbl) * dx, t0, tn], 'aspect':'auto', 'cmap':'gray', 'vmin':vmin, 'vmax':vmax}

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 7))

    def update(frame):
        rec_mask[ : frame + 1] = 1

        ax0.clear()
        img = ax0.imshow(P.data[frame].T, **plot_options0)
        ax0.set_title(f'Campo de Pressão')
        ax0.set_ylabel(f'Time: {round(frame * dt, 2)}ms \n\nProfundidade (m)')
        ax0.set_xlabel('Distãncia (m)')

        ax1.clear()
        ax1.imshow(rec.data * rec_mask, **plot_options1)
        ax1.set_title(f'Sismograma')
        ax1.set_ylabel(f'Tempo (ms)')
        ax1.set_xlabel('Distãncia (m)')
        return img,

    fig.tight_layout()

    frames = P.data.shape[0] if frames == None else frames

    ani = FuncAnimation(fig, update, frames=frames, blit=True, interval=interval)

    from IPython.display import HTML
    output = HTML(ani.to_jshtml())

    plt.close(fig)

    return output

def plot_model(model, figsize=None):
    import matplotlib.patches as patches

    params = model.physical_params()
    nparams = len(params)
    nbl = model.nbl

    cols = 5
    rows = nparams // cols if nparams % cols == 0 else nparams // cols + 1
    figsize = figsize if figsize != None else (cols * 3, 2.5 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, sharey=True, sharex=True)

    nbl_size = (model.grid.extent[0] - model.domain_size[0]) // 2
    extent = [-nbl_size, model.domain_size[0] + nbl_size, model.domain_size[1] + nbl_size, -nbl_size]

    for (i, ax), (name, param) in zip(enumerate(axes.flatten()), params.items()):
        img = ax.imshow(param.data.T, extent=extent, cmap='nipy_spectral', aspect='auto')

        ax.set_title(name)
        if i >= cols * (rows - 1):
            ax.set_xlabel('Distance (m)')
        if i % cols == 0:
            ax.set_ylabel('Depth (m)')

        cbar = fig.colorbar(img)

        rect = patches.Rectangle([0, 0], model.domain_size[0], model.domain_size[1], linestyle='--', linewidth=1, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

    fig.tight_layout()
    plt.show()

# ----------------------------------- #