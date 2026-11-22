
import numpy as np 
import matplotlib.pyplot as plt
import math as math 
import cmath as cmath
from scipy import linalg as linalg
from scipy import integrate as integ
from matplotlib import cm
from matplotlib import rcParams
from matplotlib import colors
from fractions import Fraction as Fraction
import warnings
import matplotlib.animation as animation
import matplotlib.lines as mlines
from matplotlib.patches import Ellipse
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D           # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def Select_Mode(mode, r_JK, sigma_JK):
    """Slice a single bosonic mode out of multi-mode GCS arrays.

    For canonical ordering (x₁,p₁, x₂,p₂, …, xₙ,pₙ) mode `m` lives at
    phase-space indices [2m, 2m+1].  Works on both time-series and single
    snapshots because it only touches the last phase-space axes.

    Parameters
    ----------
    mode     : int — 0-indexed mode to extract
    r_JK     : (..., 2n, 1) array of first moments
    sigma_JK : (..., 2n, 2n) array of covariance matrices

    Returns
    -------
    r_m     : (..., 2, 1)   — first moments for mode `m`
    sigma_m : (..., 2, 2)   — covariance matrix for mode `m`

    Examples
    --------
    # prepare mode 1 of a 3-mode state before any plot call
    r_1, s_1 = Select_Mode(1, GCS.r_JK_t, GCS.sigma_JK_t)
    Plot_Trajectories_Single(r_1)
    """
    idx = [2 * mode, 2 * mode + 1]
    r_m     = r_JK[..., idx, :]
    sigma_m = sigma_JK[..., idx, :][..., idx]
    return r_m, sigma_m


def format_latex_numbers(arr):
    formatted = []
    for num in arr:
        frac = Fraction(num).limit_denominator()
        if frac.denominator == 1:
            formatted.append(f'${frac.numerator}$')
        else:
            formatted.append(f'${frac.numerator}/{frac.denominator}$')
    return formatted

def Plot_Phase_Space_First_QRDM_Func(GCS, leged_lables, array_pi, save, mode=0):

    rcParams['mathtext.fontset'] = 'cm'
    rcParams['font.family'] = 'STIXGeneral'

    x_idx = 2 * mode
    p_idx = 2 * mode + 1

    N2 = 2 ** GCS.N_qubits
    if N2 == 2:
        _colors = ['tab:blue', 'tomato']
    else:
        _cmap = plt.cm.get_cmap('tab10')
        _colors = [_cmap(i % 10) for i in range(N2)]

    fig, axes = plt.subplots(2, 3,figsize=(6,4), dpi = 350)
    fig.subplots_adjust(wspace=0.1, hspace=0.2)


    max_x = np.max(np.array([np.real(GCS.r_JK_t[:,:,:,x_idx]),np.imag(GCS.r_JK_t[:,:,:,x_idx])]))*1.1
    min_x = np.min(np.array([np.real(GCS.r_JK_t[:,:,:,x_idx]),np.imag(GCS.r_JK_t[:,:,:,x_idx])]))*1.1
    max_p = np.max(np.array([np.real(GCS.r_JK_t[:,:,:,p_idx]),np.imag(GCS.r_JK_t[:,:,:,p_idx])]))*1.1
    min_p = np.min(np.array([np.real(GCS.r_JK_t[:,:,:,p_idx]),np.imag(GCS.r_JK_t[:,:,:,p_idx])]))*1.1


    lables_pi = format_latex_numbers(array_pi)

    for i in range(2):
        for j in range(3):
            axes[i,j].set_xlim(-GCS.t_array[-1]/(20*np.pi), GCS.t_array[-1]*21/(20*np.pi))
            axes[i,j].set_xticks(array_pi)
            axes[i,j].grid()
            axes[i,j].axvline(x=0, color='black',linewidth=1.2, alpha=0.6)
            axes[i,j].axhline(y=0, color='black',linewidth=1.2, alpha=0.6)
            for axis in ['top','bottom','left','right']:
                axes[i,j].spines[axis].set_linewidth(0.5)
            if i == 0:
                axes[i,j].xaxis.set_label_position("top")
                axes[i,j].set_xticklabels([])
                axes[i,j].xaxis.tick_top()  # Moves the ticks to the top
                axes[i,j].xaxis.set_label_position('top')
                #axes[i,j].set_xlabel(r'$\tau$',labelpad=2, fontsize=9)
                if j == 2:
                    pass
                else:
                    axes[i,j].set_ylim(min_x, max_x)
            
            else:
                axes[i,j].set_xlabel(r'Time $(\tau/\pi)$',labelpad=2, fontsize=13)
                axes[i,j].set_xticklabels(lables_pi)
                if j == 2:
                    pass
                else:
                    axes[i,j].set_ylim(min_p, max_p)

            if j == 2:
                axes[i,j].yaxis.set_label_position("right")
                axes[i,j].yaxis.tick_right()
                #axes[i,j].invert_xaxis()
            elif j == 1:
                axes[i,j].set_yticklabels([])
    
    #################################### On Diagonal ####################################

    axes[0,0].set_title('On Diagonal', fontsize=15)

    for j in range(N2):
        axes[0,0].plot(GCS.t_array/(np.pi), GCS.r_JK_t[:,j,j,x_idx], color=_colors[j])
        axes[1,0].plot(GCS.t_array/(np.pi), GCS.r_JK_t[:,j,j,p_idx], color=_colors[j])

    axes[0,0].set_ylabel(r'Position ($x = X/x_0$)',labelpad=2, fontsize=13)
    axes[1,0].set_ylabel(r'Momentum ($p= P/p_0$)',labelpad=2, fontsize=13)


    #################################### Off Diagonal ####################################

    axes[0,1].set_title('Off Diagonal', fontsize=15)

    for j in range(N2):
        for k in range(j + 1, N2):
            axes[0,1].plot(GCS.t_array/(np.pi), np.real(GCS.r_JK_t[:,j,k,x_idx]), color=_colors[j])
            axes[0,1].plot(GCS.t_array/(np.pi), np.real(GCS.r_JK_t[:,k,j,x_idx]), color=_colors[k])
            axes[0,1].plot(GCS.t_array/(np.pi), np.imag(GCS.r_JK_t[:,j,k,x_idx]), '--', color=_colors[j])
            axes[0,1].plot(GCS.t_array/(np.pi), np.imag(GCS.r_JK_t[:,k,j,x_idx]), '--', color=_colors[k])
            axes[1,1].plot(GCS.t_array/(np.pi), np.real(GCS.r_JK_t[:,j,k,p_idx]), color=_colors[j])
            axes[1,1].plot(GCS.t_array/(np.pi), np.real(GCS.r_JK_t[:,k,j,p_idx]), color=_colors[k])
            axes[1,1].plot(GCS.t_array/(np.pi), np.imag(GCS.r_JK_t[:,j,k,p_idx]), '--', color=_colors[j])
            axes[1,1].plot(GCS.t_array/(np.pi), np.imag(GCS.r_JK_t[:,k,j,p_idx]), '--', color=_colors[k])
    

#################################### QRDM ####################################\

    axes[0,2].set_title(r'QRDM $(\hat{\rho}^q)$', fontsize=15)

    axes[0,2].axvline(x=0, color='black',linewidth=0.5, alpha=0.4)
    axes[0,2].axhline(y=0, color='black',linewidth=0.5, alpha=0.4)

    for j in range(N2):
        for k in range(j + 1, N2):
            axes[0,2].plot(GCS.t_array/(np.pi), GCS.C_JK_t[:,j,k], color=_colors[j])
            axes[0,2].plot(GCS.t_array/(np.pi), GCS.C_JK_t[:,k,j], ':', color=_colors[k])
    axes[0,2].set_ylabel(r'Contrasts $(C)$',labelpad=15, fontsize=13,rotation=-90)

    #################################### Time Evolution Phases ####################################
    axes[1,2].axvline(x=0, color='black',linewidth=0.5, alpha=0.4)
    axes[1,2].axhline(y=0, color='black',linewidth=0.5, alpha=0.4)

    for j in range(N2):
        for k in range(j + 1, N2):
            axes[1,2].plot(GCS.t_array/(np.pi), np.sin(GCS.phi_JK_t[:,j,k]), color=_colors[j])
            axes[1,2].plot(GCS.t_array/(np.pi), np.sin(GCS.phi_JK_t[:,k,j]), color=_colors[k])

    axes[1,2].set_ylabel(r'$\sin(\phi)$',labelpad=15, fontsize=13, rotation=-90)

    
    for axis in ['top','bottom','left','right']:
        axes[0,1].spines[axis].set_linewidth(0.5)
        axes[1,1].spines[axis].set_linewidth(0.5)
        axes[1,0].spines[axis].set_linewidth(0.5)

    #################################### Plot Design ####################################
    
    if N2 == 2:
        _branch_labels = ['Up', 'Down']
    else:
        _branch_labels = [f'$J_{{{j}}}$' for j in range(N2)]
    for j in range(N2):
        axes[1,1].plot(np.array([-1000, -1001]),np.array([-1000, -1001]), color=_colors[j], label=_branch_labels[j])
    axes[1,1].plot(np.array([-1000, -1001]),np.array([-1000, -1001]),  color='gray', label='Real')
    axes[1,1].plot(np.array([-1000, -1001]),np.array([-1000, -1001]), '--', color='gray', label='Complex')

    for i in range(len(leged_lables)):
        axes[1,1].scatter(-1000,-1000, color='black', label=leged_lables[i])

    leg = fig.legend(loc='upper center', bbox_to_anchor=(0.50, 0.51),
                     ncol=N2+2+len(leged_lables), borderpad=-0.5, fontsize=11,
                     labelspacing=0.2, handlelength=1.4)
    leg.get_frame().set_edgecolor('w')
    for line in leg.get_lines():
        line.set_linewidth(1.1)
    leg.get_frame().set_linewidth(0.4)

    
    if save == False:
        pass
    else:
        plt.savefig(save, bbox_inches='tight', transparent=True)
    return 

def Plot_Wigner_Diag(wigner,  x_array, p_array, bar_limits, array_tick, title, save):
    
    rcParams['mathtext.fontset'] = 'cm'
    rcParams['font.family'] = 'STIXGeneral'
    
    fig, axes = plt.subplots(1, 1, figsize=(3.5,3.5),  dpi = 300)
    fig.set_tight_layout(True)

    lables_tick = format_latex_numbers(array_tick)
    
    
    axes.axvline(x=0, color='black',linewidth=0.5, alpha=0.4)
    axes.axhline(y=0, color='black',linewidth=0.5, alpha=0.4)

    axes.set_ylabel(r'Momentum, $p$',labelpad=2, fontsize=11)
    axes.set_xlabel(r'Position, $x$',labelpad=2, fontsize=11)

    axes.set_ylim(-np.max(x_array), np.max(x_array))
    axes.set_xlim(-np.max(x_array), np.max(x_array))
    axes.set_xticks(array_tick)
    axes.set_xticklabels(lables_tick)
    axes.set_yticks(array_tick)
    axes.set_yticklabels(lables_tick)




    for axis in ['top','bottom','left','right']:
        axes.spines[axis].set_linewidth(0.5)


    #title_box = dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2', linewidth=0.4)
    #axes[1,1].text(0.97, 5.8, r'$W^+(x,p)$', transform=ax.transAxes, fontsize=10,verticalalignment='top', bbox=title_box)
    # 'RdBu'
    
    im = axes.imshow(wigner, cmap='Blues',  interpolation='bilinear', 
                        #alpha=np.abs(wigner)/np.max(wigner), 
                        origin='lower', vmin=-bar_limits[0], vmax=bar_limits[1],
                        aspect=len(x_array)/len(p_array),
                        extent = [np.min(x_array), np.max(x_array), np.min(p_array)*len(p_array)/len(x_array), np.max(p_array)*len(p_array)/len(x_array)])


    
    #cbar_ax = fig.add_axes([0.95, 0.15, 0.03, 0.7])   cax=cbar_ax , 
    bar = fig.colorbar(im, fraction=0.04, orientation='vertical', alpha=np.linspace(0,1,100))
    bar.set_ticks([bar_limits[0],0,bar_limits[1]])
    bar.set_ticklabels(['{:.1f}'.format(x) for x in bar.get_ticks()])
    bar.set_label(r'$ W(\tilde{r})$',fontsize=10,labelpad=-5, y = 0.4,ha='right', rotation=-90)
    
    if title == False:
        axes.set_title(r'Wigner Function $ W(\bar{r})$', fontsize=13)
    else:
        axes.set_title(title, fontsize=13)
        
    
    if save == False:
        pass
    else:
        plt.savefig(save, bbox_inches='tight', transparent=True)
    return fig, im

def Plot_Wigner_4_Times(wigner_array, x_array, p_array, save):
    
    fig, axes = plt.subplots(2, 2, figsize=(3,3), dpi = 500)#,layout="compressed")
    
    colors = [(1, 1, 1, 0)  , (171/255, 219/255, 227/255, 1),      (236/255, 112/255, 49/255, 1)]
    
    ax = axes.flatten()
    
    rcParams['mathtext.fontset'] = 'cm'
    rcParams['font.family'] = 'STIXGeneral'

    titles = [r'$t=0,t_f$', r'$t_+$',r'$t_+ +t_-$', r'$t_{max}$']
    print([np.min(x_array), np.max(x_array), np.min(p_array), np.max(p_array)])
    
    # Create the heatmap
    for i in range(len(ax)):
        im = ax[i].matshow(wigner_array[i], cmap='Blues', zorder=5, interpolation='bilinear', alpha=wigner_array[i]/np.max(wigner_array[i]),origin='lower',
                          aspect=len(x_array)/len(p_array),extent = [np.min(x_array), np.max(x_array), np.min(p_array)*len(p_array)/len(x_array), np.max(p_array)*len(p_array)/len(x_array)])

        
        #ax[i].grid(linewidth=0.2, zorder=0)
        
        title_box = dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2', linewidth=0.4)
        ax[i].text(0.08, 0.95, titles[i], transform=ax[i].transAxes, fontsize=8,
                verticalalignment='top', bbox=title_box)
        
        ax[i].set_xticks([-30,-15,0,15,30], minor=False)
        ax[i].set_yticks([-10,0,10], minor=False)
        ax[i].xaxis.tick_bottom()
        
       
        if i >1:
            ax[i].set_xlabel(r'$x $',labelpad=-2)
            ax[i].set_xticklabels([r'$-30$',r'$-15$',r'$0$',r'$15$',r'$30$'], fontsize=6.5)
        else:
            ax[i].set_xticklabels([])
            
            
        if i == 0 or i == 2:
            ax[i].set_ylabel(r'$ p$',labelpad=-4)
            ax[i].set_yticklabels([r'$-10$',r'$0$',r'$10$'], fontsize=6.5)
        else:
            ax[i].set_yticklabels([], minor=False)
        
        ax[i].set_axisbelow(True)

        ax[i].xaxis.set_tick_params(width=0.5,direction="in",length=3)
        ax[i].yaxis.set_tick_params(width=0.5,direction="in",length=3)
            # change all spines
        for axis in ['top','bottom','left','right']:
            ax[i].spines[axis].set_linewidth(0.5)
            
    cbar_ax = fig.add_axes([0.93, 0.18, 0.03, 0.63])
    
    bar = fig.colorbar(im, cax=cbar_ax , orientation='vertical',alpha=np.linspace(0,1,100))
    bar.set_ticks([])
    bar.set_label(r'$ W(r,t)$',fontsize=8.5,labelpad=12, y=0.42, ha='right', rotation=-90)
    #ax.set_xlim(3*np.pi,9/2*np.pi)
    #ax.set_ylim(-0.05, 0.55)

    plt.subplots_adjust(wspace=0.1, hspace=-0.27)
    #fig.suptitle('(c)',x=0.06,y=0.83, fontsize=12)
    
    #a
    if save == False:
        return 
    else: 
        plt.savefig(save, bbox_inches='tight')
        return

def Plot_Wigner_Fringes(wigner_PMS,  x_array, p_array, bar_limits, array_tick,title, save):

    
    rcParams['mathtext.fontset'] = 'cm'
    rcParams['font.family'] = 'STIXGeneral'
    
    fig, axes = plt.subplots(1, 2, figsize=(4.5,3.3),  dpi = 400)
    fig.subplots_adjust(wspace=0.1)

    #fig.suptitle('Wigner Functions of Post Measurament States')

        #title_box = dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2', linewidth=0.4)
        #axes[1,1].text(0.97, 5.8, r'$W^+(x,p)$', transform=ax.transAxes, fontsize=10,verticalalignment='top', bbox=title_box)
        # 'RdBu'
    
    im1 = axes[0].imshow(wigner_PMS[0], cmap='RdBu',  interpolation='bilinear', 
                        #alpha=np.abs(wigner_array)/np.max(wigner_array), 
                        origin='lower', vmin=bar_limits[0], vmax=bar_limits[1],
                        aspect=len(x_array)/len(p_array),
                        extent = [np.min(x_array), np.max(x_array), np.min(p_array)*len(p_array)/len(x_array), np.max(p_array)*len(p_array)/len(x_array)])


    im2 = axes[1].imshow(wigner_PMS[1], cmap='RdBu',  interpolation='bilinear', 
                        #alpha=np.abs(wigner_array)/np.max(wigner_array), 
                        origin='lower', vmin=bar_limits[0], vmax=bar_limits[1],
                        aspect=len(x_array)/len(p_array),
                        extent = [np.min(x_array), np.max(x_array), np.min(p_array)*len(p_array)/len(x_array), np.max(p_array)*len(p_array)/len(x_array)])


    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.84, 0.23, 0.03, 0.5])
    bar = fig.colorbar(im2, cax=cbar_ax , orientation='vertical')#, alpha=np.linspace(0,1,100))
    bar.set_ticks([bar_limits[0],0,bar_limits[1]])
    bar.set_ticklabels(['{:.1f}'.format(x) for x in bar.get_ticks()])
    bar.set_label(r'$ W(\tilde{r})$',fontsize=10,labelpad=10, y=0.45, ha='right', rotation=-90)

    lables_tick = format_latex_numbers(array_tick)

    for i in range(2):
        
        axes[i].set_ylim(-np.max(x_array), np.max(x_array))
        axes[i].set_xlim(-np.max(x_array), np.max(x_array))
        
        axes[i].set_xticks(array_tick)
        axes[i].set_xticklabels(lables_tick)
        axes[i].set_yticks(array_tick)
        
        axes[i].axvline(x=0, color='black',linewidth=0.5, alpha=0.4)
        axes[i].axhline(y=0, color='black',linewidth=0.5, alpha=0.4)

        axes[i].set_xlabel(r'Position ($x$)',labelpad=2, fontsize=11)


        for axis in ['top','bottom','left','right']:
            axes[i].spines[axis].set_linewidth(0.5)
    
    axes[1].set_yticklabels([])
    axes[0].set_yticklabels(lables_tick)

    axes[0].set_title(r'$ W_+(\tilde{r})$')
    axes[1].set_title(r'$ W_-(\tilde{r})$')
    axes[0].set_ylabel(r'Momentum ($p$)',labelpad=0, fontsize=11)


    if title == False:
        fig.suptitle('(b) Wigner Functions of Post-Measurement States', y=0.85)
    else:
        fig.suptitle(title, y=0.85)
    
    if save == False:
        pass
    else:
        plt.savefig(save, bbox_inches='tight', transparent=True)
    return fig, im1, im2

def Animate_Diagonal(wigner_t,  x_array, p_array, bar_limits, array_tick, n_frames, title, save):

    fig, im = Plot_Wigner_Diag(wigner_t[0],  x_array, p_array, bar_limits, array_tick,title, False)

    def update(frame):
        """Update function for the animation."""
        # Update colormap data
        im.set_array(wigner_t[frame])

        return [im]
    
    fig.tight_layout()
    
    
    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=50)
    if save == False:
        pass
    else:
        ani.save(filename=save,writer="pillow")
    return

def Animate_Fringes(wigner_PMS,  x_array, p_array, bar_limits, array_tick, n_frames,title, save):

    fig, im1,im2 = Plot_Wigner_Fringes(wigner_PMS[0],  x_array, p_array, bar_limits, array_tick,title, False)

    def update(frame):
        """Update function for the animation."""
        # Update colormap data
        im1.set_array(wigner_PMS[frame,0])
        im2.set_array(wigner_PMS[frame,1])

        return [im1, im2]
    
    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=50)
    if save == False:
        pass
    else:
        ani.save(filename=save,writer="pillow")
    return

def Plot_Wigner_GCS_Fringes(W_c, W_q, x_array, p_array, array_tick, title, save):
    """Side-by-side plot of the classical W_c and quantum-interference W_q Wigner functions.

    W_c = W_+ + W_-  (sum of diagonal branches, always ≥ 0  → Blues colormap)
    W_q = 2 Re[W^off] (quantum fringes, can be negative       → RdBu colormap)

    Parameters
    ----------
    W_c, W_q   : (steps, steps) real arrays
    x_array, p_array : 1-D coordinate arrays
    array_tick : tick positions (shared for both axes)
    title      : suptitle string (False / '' to omit)
    save       : filepath string to save, or False

    Returns
    -------
    fig, im_c, im_q
    """
    rcParams['mathtext.fontset'] = 'cm'
    rcParams['font.family'] = 'STIXGeneral'

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.3), dpi=350)
    fig.subplots_adjust(wspace=0.08)

    aspect = len(x_array) / len(p_array)
    extent = [np.min(x_array), np.max(x_array),
              np.min(p_array) * len(p_array) / len(x_array),
              np.max(p_array) * len(p_array) / len(x_array)]

    lim          = np.max(np.abs(x_array))
    lim_p        = lim * len(p_array) / len(x_array)
    lables_tick  = format_latex_numbers(array_tick)

    bar_c = float(np.max(W_c)) if np.max(W_c) > 0 else 1.0
    bar_q = float(np.max(np.abs(W_q))) if np.max(np.abs(W_q)) > 0 else 1.0

    im_c = axes[0].imshow(W_c, cmap='Blues', interpolation='bilinear',
                          origin='lower', vmin=0, vmax=bar_c,
                          aspect=aspect, extent=extent)

    im_q = axes[1].imshow(W_q, cmap='RdBu', interpolation='bilinear',
                          origin='lower', vmin=-bar_q, vmax=bar_q,
                          aspect=aspect, extent=extent)

    for i, (ax, im, lbl) in enumerate([(axes[0], im_c, r'$\mathcal{W}_c$'),
                                        (axes[1], im_q, r'$\mathcal{W}_q$')]):
        ax.axvline(x=0, color='black', linewidth=0.5, alpha=0.4)
        ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.4)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim_p, lim_p)
        ax.set_xticks(array_tick)
        ax.set_xticklabels(lables_tick)
        ax.set_yticks(array_tick)
        ax.set_xlabel(r'Position, $x$', labelpad=2, fontsize=11)
        ax.set_title(lbl, fontsize=13)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

    axes[0].set_ylabel(r'Momentum, $p$', labelpad=2, fontsize=11)
    axes[0].set_yticklabels(lables_tick)
    axes[1].set_yticklabels([])

    cbar_c = fig.colorbar(im_c, ax=axes[0], fraction=0.046, pad=0.03, orientation='vertical')
    cbar_c.set_ticks([0, bar_c])
    cbar_c.set_ticklabels([r'$0$', f'{bar_c:.2f}'], fontsize=8)
    cbar_c.set_label(r'$\mathcal{W}_c$', fontsize=9, labelpad=4, rotation=-90)

    cbar_q = fig.colorbar(im_q, ax=axes[1], fraction=0.046, pad=0.03, orientation='vertical')
    cbar_q.set_ticks([-bar_q, 0, bar_q])
    cbar_q.set_ticklabels([f'{-bar_q:.2f}', r'$0$', f'{bar_q:.2f}'], fontsize=8)
    cbar_q.set_label(r'$\mathcal{W}_q$', fontsize=9, labelpad=4, rotation=-90)

    if title:
        fig.suptitle(title, fontsize=12, y=1.01)

    if save:
        plt.savefig(save, bbox_inches='tight', transparent=True)

    return fig, im_c, im_q


def Animate_GCS_Fringes(W_c_t, W_q_t, x_array, p_array, array_tick, n_frames, title, save):
    """Animate W_c and W_q side by side over time.

    Color limits are fixed to the global extrema across all frames so the
    fringes are comparable between time steps.

    Parameters
    ----------
    W_c_t, W_q_t : (T, steps, steps) real arrays
    x_array, p_array : 1-D coordinate arrays
    array_tick   : tick positions
    n_frames     : int — number of animation frames
    title        : suptitle string (False / '' to omit)
    save         : filepath to save .gif, or False

    Returns
    -------
    ani : FuncAnimation
    """
    bar_c = float(np.max(W_c_t)) if np.max(W_c_t) > 0 else 1.0
    bar_q = float(np.max(np.abs(W_q_t))) if np.max(np.abs(W_q_t)) > 0 else 1.0

    fig, im_c, im_q = Plot_Wigner_GCS_Fringes(
        W_c_t[0], W_q_t[0], x_array, p_array, array_tick, title, False)

    im_c.set_clim(vmin=0, vmax=bar_c)
    im_q.set_clim(vmin=-bar_q, vmax=bar_q)

    def update(frame):
        im_c.set_array(W_c_t[frame])
        im_q.set_array(W_q_t[frame])
        return [im_c, im_q]

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=50)
    if save:
        ani.save(filename=save, writer='pillow')
    return ani


def Plot_Wigner_GCS_Total(W_total, x_array, p_array, array_tick, title, save):
    """Plot the full GCS Wigner function W_total = W_c + W_q.

    Uses a diverging RdBu colormap symmetric around zero so negative (quantum)
    regions are immediately visible.

    Parameters
    ----------
    W_total    : (steps, steps) real array — W_c + W_q
    x_array, p_array : 1-D coordinate arrays
    array_tick : tick positions
    title      : axes title string (False / '' to omit)
    save       : filepath to save, or False

    Returns
    -------
    fig, im
    """
    rcParams['mathtext.fontset'] = 'cm'
    rcParams['font.family'] = 'STIXGeneral'

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5), dpi=300)
    fig.set_tight_layout(True)

    lables_tick = format_latex_numbers(array_tick)
    lim   = np.max(np.abs(x_array))
    lim_p = lim * len(p_array) / len(x_array)
    bar   = float(np.max(np.abs(W_total))) if np.max(np.abs(W_total)) > 0 else 1.0

    ax.axvline(x=0, color='black', linewidth=0.5, alpha=0.4)
    ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.4)
    ax.set_ylabel(r'Momentum, $p$', labelpad=2, fontsize=11)
    ax.set_xlabel(r'Position, $x$', labelpad=2, fontsize=11)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim_p, lim_p)
    ax.set_xticks(array_tick)
    ax.set_xticklabels(lables_tick)
    ax.set_yticks(array_tick)
    ax.set_yticklabels(lables_tick)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    extent = [np.min(x_array), np.max(x_array),
              np.min(p_array) * len(p_array) / len(x_array),
              np.max(p_array) * len(p_array) / len(x_array)]

    im = ax.imshow(W_total, cmap='RdBu', interpolation='bilinear',
                   origin='lower', vmin=-bar, vmax=bar,
                   aspect=len(x_array) / len(p_array), extent=extent)

    cbar = fig.colorbar(im, fraction=0.046, pad=0.04, orientation='vertical')
    cbar.set_ticks([-bar, 0, bar])
    cbar.set_ticklabels([f'{-bar:.2f}', r'$0$', f'{bar:.2f}'], fontsize=8)
    cbar.set_label(r'$\mathcal{W}(\tilde{r})$', fontsize=10, labelpad=4, rotation=-90)

    if title:
        ax.set_title(title, fontsize=13)
    if save:
        plt.savefig(save, bbox_inches='tight', transparent=True)
    return fig, im


def Animate_GCS_Total(W_total_t, x_array, p_array, array_tick, n_frames, title, save):
    """Animate W_total = W_c + W_q over time.

    Color limits are fixed to the global extremum across all frames.

    Parameters
    ----------
    W_total_t  : (T, steps, steps) real array
    x_array, p_array : 1-D coordinate arrays
    array_tick : tick positions
    n_frames   : int
    title      : figure title string (False / '' to omit)
    save       : filepath to save .gif, or False

    Returns
    -------
    ani : FuncAnimation
    """
    bar = float(np.max(np.abs(W_total_t))) if np.max(np.abs(W_total_t)) > 0 else 1.0

    fig, im = Plot_Wigner_GCS_Total(W_total_t[0], x_array, p_array, array_tick, title, False)
    im.set_clim(vmin=-bar, vmax=bar)

    def update(frame):
        im.set_array(W_total_t[frame])
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=50)
    if save:
        ani.save(filename=save, writer='pillow')
    return ani


def Evaluate_Wigner_On_Grid(W, x_array, p_array):
    """Evaluate a Wigner function on the phase-space grid used by the plotting routines.

    `W` is either a callable `W(x, p)` -- evaluated once on the whole meshgrid, so it
    must broadcast -- or an already-computed array of shape (len(p_array), len(x_array)).
    Row index runs over p and column index over x, matching the grid built by
    Wigner_Functions.Create_r_tilde_array and the origin='lower' imshow used elsewhere.

    Parameters
    ----------
    W        : callable W(x, p) -> array, or (len(p_array), len(x_array)) array
    x_array, p_array : 1-D coordinate arrays

    Returns
    -------
    (len(p_array), len(x_array)) real array
    """
    if callable(W):
        XX, PP = np.meshgrid(x_array, p_array)
        W = W(XX, PP)
    W = np.asarray(W)
    if W.shape != (len(p_array), len(x_array)):
        raise ValueError(f"Wigner grid has shape {W.shape}, expected "
                         f"{(len(p_array), len(x_array))} = (len(p_array), len(x_array))")
    if not np.all(np.isfinite(W)):
        # a NaN would otherwise propagate into vmin/vmax and render a blank figure
        raise ValueError(f"Wigner grid contains {np.count_nonzero(~np.isfinite(W))} "
                         f"non-finite values")
    return W


def Plot_Wigner_Generic(W, x_array, p_array, array_tick, title, save,
                        bar=None, cbar_label=r'$\mathcal{W}(\tilde{r})$'):
    """Plot an arbitrary Wigner function in the same style as Plot_Wigner_GCS_Total.

    Diverging RdBu colormap symmetric about zero, so negativity is immediately visible.

    Parameters
    ----------
    W          : callable W(x, p), or (len(p_array), len(x_array)) real array
    x_array, p_array : 1-D coordinate arrays
    array_tick : tick positions
    title      : axes title string (False / '' to omit)
    save       : filepath to save, or False
    bar        : colour-scale extremum; None -> max|W|
    cbar_label : colourbar label

    Returns
    -------
    fig, im, W_grid
    """
    rcParams['mathtext.fontset'] = 'cm'
    rcParams['font.family'] = 'STIXGeneral'

    W_grid = Evaluate_Wigner_On_Grid(W, x_array, p_array)

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5), dpi=300)
    fig.set_tight_layout(True)

    lables_tick = format_latex_numbers(array_tick)
    x_lo, x_hi = float(np.min(x_array)), float(np.max(x_array))
    p_lo, p_hi = float(np.min(p_array)), float(np.max(p_array))
    if bar is None:
        bar = float(np.max(np.abs(W_grid))) if np.max(np.abs(W_grid)) > 0 else 1.0

    ax.axvline(x=0, color='black', linewidth=0.5, alpha=0.4)
    ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.4)
    ax.set_ylabel(r'Momentum, $p$', labelpad=2, fontsize=11)
    ax.set_xlabel(r'Position, $x$', labelpad=2, fontsize=11)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(p_lo, p_hi)
    ax.set_xticks(array_tick)
    ax.set_xticklabels(lables_tick)
    ax.set_yticks(array_tick)
    ax.set_yticklabels(lables_tick)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    # True data extents, and an aspect that keeps the axes box square for any grid.
    # (The other Plot_Wigner_* helpers fold len(p)/len(x) into both, which agrees with
    # this only when the grid is square with equal x and p ranges.)
    im = ax.imshow(W_grid, cmap='RdBu', interpolation='bilinear',
                   origin='lower', vmin=-bar, vmax=bar,
                   aspect=(x_hi - x_lo) / (p_hi - p_lo),
                   extent=[x_lo, x_hi, p_lo, p_hi])

    cbar = fig.colorbar(im, fraction=0.046, pad=0.04, orientation='vertical')
    cbar.set_ticks([-bar, 0, bar])
    cbar.set_ticklabels([f'{-bar:.2g}', r'$0$', f'{bar:.2g}'], fontsize=8)
    cbar.set_label(cbar_label, fontsize=10, labelpad=4, rotation=-90)

    if title:
        ax.set_title(title, fontsize=13)
    if save:
        plt.savefig(save, bbox_inches='tight', transparent=True)
    return fig, im, W_grid


def Plot_Wigner_GCS_Weighted(W_c, W_q, W_g, p_c, p_q, p_g,
                             x_array, p_array, array_tick, title, save, atol=1e-8):
    """Plot the weighted superposition p_c*W_c + p_q*W_q + p_g*W_g.

    W_c and W_q are the classical and quantum GCS components built from Eq.(wigner_jk)
    (they are defined in the surrounding prose of the paper, not in a numbered equation);
    W_g is any further Wigner component, e.g. the DV-decay term W_Gamma of Appendix I B.  The
    weights are free -- nothing is renormalised -- but a warning is printed when they
    do not sum to one, since the total is then not a trace-one Wigner function.

    Parameters
    ----------
    W_c, W_q   : (len(p_array), len(x_array)) real arrays
    W_g        : callable W(x, p), or array of the same shape
    p_c, p_q, p_g : scalar weights
    x_array, p_array : 1-D coordinate arrays
    array_tick : tick positions
    title      : axes title string (False / '' to omit)
    save       : filepath to save, or False
    atol       : tolerance on the weight sum

    Returns
    -------
    fig, im, W_sum
    """
    W_g_grid = Evaluate_Wigner_On_Grid(W_g, x_array, p_array)
    W_sum = p_c*np.asarray(W_c) + p_q*np.asarray(W_q) + p_g*W_g_grid

    total = p_c + p_q + p_g
    if abs(total - 1.0) > atol:
        warnings.warn(f"weights sum to {total:.6g}, not 1; the plotted total is not "
                      f"normalised", RuntimeWarning, stacklevel=2)

    fig, im = Plot_Wigner_GCS_Total(W_sum, x_array, p_array, array_tick, title, save)
    return fig, im, W_sum


def generate_pi_ticks(maximum, step):
    # Generate numerical array
    x_vals = np.arange(0, (maximum + 1) * step * np.pi, step * np.pi)
    x_vals = np.round(x_vals, decimals=10)  # Avoid floating point issues

    labels = []
    for i in range(len(x_vals)):
        coeff = i
        if step == 1:
            label = r'$' + (f'{coeff}' if coeff > 1 else '') + r'\pi$' if coeff > 0 else r'$0$'
        elif step > 1:
            step_int = int(step) if step.is_integer() else step
            label = r'$' + (f'{coeff}' if coeff > 1 else '') + f'{step_int}\\pi$' if coeff > 0 else r'$0$'
        else:  # step < 1
            denom = int(1 / step) if (1 / step).is_integer() else f'1/{step}'
            if coeff == 0:
                label = r'$0$'
            else:
                label = r'$' + (f'{coeff}' if coeff > 1 else '') + r'\pi/' + f'{int(1/step)}$'
    # Special case: for fractional steps, show e.g. r'$2\pi/3$' rather than r'$2\pi/3$' as is.
                label = r'$' + (f'{coeff}\\pi/{int(1/step)}$' if coeff > 0 else '0')
        labels.append(label)

    return x_vals, labels

def Plot_Phase_Space_Diagonal_Total(t_array, quantum_state):

    rcParams['mathtext.fontset'] = 'cm'
    rcParams['font.family'] = 'STIXGeneral'

    fig = plt.figure(figsize=(4.5,4.5), dpi = 400)
    
    subfigs = fig.subfigures(1, 2, width_ratios=[2, 1]) 



    axes = plt.subplots(2, 2, figsize=(4.5,4.5), height_ratios=[1,2.5],width_ratios=[1, 2.5], dpi = 400)


    
    axes[0,0].axis('off')
    
    #axes[0,0].set_xticks([], minor=False)
    #axes[0,0].set_xticklabels([], fontsize=8)
    axes[0,0].set_xlim(0.1,0.2)
    axes[0,0].set_ylim(0.1,0.2)
    for i in range(len(leged_lables)):
        axes[0,0].scatter(1,1,s=1, color='black', label=leged_lables[i])
    
    axes[0,0].legend(loc='center',fontsize = 10,handletextpad=0.1)
    
    axes[1,0].axvline(x=0, color='black',linewidth=0.5, alpha=0.4)
    axes[1,0].axhline(y=0, color='black',linewidth=0.5, alpha=0.4)
    axes[1,0].set_yticks([], minor=False)
    #axes[0,0].set_yticks([-delta_x,-delta_x/2,0,delta_x/2,delta_x], minor=False)
    #axes[0,0].set_yticklabels([r'$-\delta x$',r'$- \frac{\delta x}{2}$',r'$0$',r'$\frac{\delta x}{2}$',r'$\delta x$'], fontsize=8)
    #axes[0,0].set_xlabel(r'$ p $',labelpad=2, fontsize=9)
    axes[1,0].set_xlabel(r'$x$' ,labelpad=6, fontsize=10)
    axes[1,0].set_xticks([0,0.1,0.2], minor=False)
    axes[1,0].set_xticklabels([r'$0$',r'$0.1$',r'$0.2$'], fontsize=8)
    axes[1,0].set_xlim(-0.05,0.25)
    axes[1,0].xaxis.set_label_position("top")
    axes[1,0].invert_xaxis()

    axes[1,0].axvline(x=0, color='black',linewidth=0.5, alpha=0.4)
    axes[1,0].axhline(y=0, color='black',linewidth=0.5, alpha=0.4)
    axes[0,1].set_xticks([], minor=False)
    #axes[1,1].set_xticks([-delta_x,-delta_x/2,0,delta_x/2,delta_x], minor=False)
    #axes[1,1].set_xticklabels([r'$-\delta x$',r'$- \frac{\delta x}{2}$',r'$0$',r'$\frac{\delta x}{2}$',r'$\delta x$'], fontsize=8)
    #axes[1,1].set_ylabel(r'$ p $',labelpad=2, fontsize=9)
    axes[0,1].set_ylabel(r'$ | \psi(x)|^2 $' ,labelpad=14, fontsize=10,rotation=-90)
    axes[0,1].set_yticks([0,0.1,0.2], minor=False)
    axes[0,1].set_yticklabels([r'$0$',r'$0.1$',r'$0.2$'], fontsize=8)
    axes[0,1].set_ylim(-0.05,0.25)
    axes[0,1].yaxis.set_label_position("right")
    
    axes[1,1].axvline(x=0, color='black',linewidth=0.5, alpha=0.4)
    axes[1,1].axhline(y=0, color='black',linewidth=0.5, alpha=0.4)
    
    axes[1,1].set_xticks([-delta_x,-delta_x/2,0,delta_x/2,delta_x], minor=False)
    axes[1,1].set_xticklabels([r'$-\delta x$',r'$- \frac{\delta x}{2}$',r'$0$',r'$\frac{\delta x}{2}$',r'$\delta x$'], fontsize=9)
    

    axes[1,1].set_yticks([-delta_x,-delta_x/2,0,delta_x/2,delta_x], minor=False)
    axes[1,1].set_yticklabels([r'$-\delta x$',r'$- \frac{\delta x}{2}$',r'$0$',r'$\frac{\delta x}{2}$',r'$\delta x$'], fontsize=9)
    
    axes[1,1].set_ylabel(r'$ p $' ,labelpad=-10, fontsize=11)
    axes[1,1].set_xlabel(r'$ x $',labelpad=7, fontsize=11)
    axes[1,1].xaxis.set_label_position("top")

    
    #ax.set_title(title, fontsize= 10)
    #x[i].set_ylabel(r'$ p$',labelpad=-4)
    #ax[i].set_yticklabels([r'$-10$',r'$0$',r'$10$'], fontsize=6.5)
    for axis in ['top','bottom','left','right']:
        axes[0,1].spines[axis].set_linewidth(0.5)
        axes[1,1].spines[axis].set_linewidth(0.5)
        axes[1,0].spines[axis].set_linewidth(0.5)
            
    axes[0,1].axvline(x=0, color='black',linewidth=0.5, alpha=0.4)
    axes[0,1].axhline(y=0, color='black',linewidth=0.5, alpha=0.4)

    #title_box = dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2', linewidth=0.4)
    #axes[1,1].text(0.97, 5.8, r'$W^+(x,p)$', transform=ax.transAxes, fontsize=10,verticalalignment='top', bbox=title_box)

   
    line_x = axes[0,1].plot(x_array, psi_x, color='tab:blue',linewidth=1.5)
    line_p = axes[1,0].plot(psi_p, x_array, color='tomato',linewidth=1.5)
    im = axes[1,1].imshow(wigner_array, cmap='RdBu',  interpolation='bilinear', 
                        #alpha=np.abs(wigner_array)/np.max(wigner_array), 
                        origin='lower', vmin=-0.5, vmax=0.5,
                        aspect=len(x_array)/len(p_array),
                        extent = [np.min(x_array), np.max(x_array), np.min(p_array)*len(p_array)/len(x_array), np.max(p_array)*len(p_array)/len(x_array)])


    cbar_ax = fig.add_axes([0.869, 0.13, 0.03, 0.45])
    bar = fig.colorbar(im, cax=cbar_ax , orientation='vertical', alpha=np.linspace(0,1,100))
    bar.set_ticks([-0.5,0,0.5])
    bar.set_ticklabels([r'$-0.5$',r'$0$',r'$0.5$'],fontsize=9)
    bar.set_label(r'$ W(r,t)$',fontsize=10,labelpad=1, y=0.42, ha='right', rotation=-90)
    
    return fig, im, line_x, line_p

    return 

def plot_paper_1st(rt, t_array, t_swich, t_swich_indx, steps, save):
    
    fig, ax = plt.subplots(1,1, figsize=(3,1.7), dpi = 500)
    
    colors_green = ['lightgreen', 'limegreen' , 'forestgreen', 'green', 'darkgreen']
    colors_blue = ['lightskyblue','cornflowerblue', 'royalblue', 'blue','darkblue']
    colors_red = ['salmon','tomato','orangered','red','darkred']

    colors_from_wigner = [ (8/255, 49/255, 121/255), (39/255, 116/255, 184/255)  ]
    
    
    rcParams['mathtext.fontset'] = 'cm'
    rcParams['font.family'] = 'STIXGeneral'

    markers = ["o",".", "s","^", "v","D","P"]
    labels = ["0", r"$t_0$",r"$t_0 + t_+$", r"$t_0 + t_+ + t_-$",r"$t_0 + 2 t_+ + t_-$",r"$t_0 +2  t_+ +2  t_-$",r"$2 (t_0 + t_+ + t_-)$"]
    
    labels_hamiltonians = [r"$H_+$",r"$H_-$",r"$H_+$",r"$H_-$"]

    ax.plot(rt[t_swich_indx[0]:t_swich_indx[1],0], rt[t_swich_indx[0]:t_swich_indx[1],1],'--', color=colors_from_wigner[1],linewidth=1.1)
    ax.plot(-rt[t_swich_indx[0]:t_swich_indx[1],0], -rt[t_swich_indx[0]:t_swich_indx[1],1],'--', color=colors_red[3],linewidth=1.1)

    ax.plot(rt[t_swich_indx[1]:t_swich_indx[2],0], rt[t_swich_indx[1]:t_swich_indx[2],1], color=colors_from_wigner[1],linewidth=1.1)
    ax.plot(-rt[t_swich_indx[1]:t_swich_indx[2],0], -rt[t_swich_indx[1]:t_swich_indx[2],1], color=colors_red[3],linewidth=1.1)

    ax.plot(rt[t_swich_indx[2]:t_swich_indx[3],0], rt[t_swich_indx[2]:t_swich_indx[3],1], '--', color=colors_from_wigner[1],linewidth=1.1)
    ax.plot(-rt[t_swich_indx[2]:t_swich_indx[3],0], -rt[t_swich_indx[2]:t_swich_indx[3],1], '--', color=colors_red[3],linewidth=1.1)

    ax.plot(rt[t_swich_indx[3]+1:t_swich_indx[4],0], rt[t_swich_indx[3]+1:t_swich_indx[4],1], color=colors_from_wigner[1],linewidth=1.1)
    ax.plot(-rt[t_swich_indx[3]+1:t_swich_indx[4],0], -rt[t_swich_indx[3]+1:t_swich_indx[4],1], color=colors_red[3],linewidth=1.1)
    
    ax.plot(-100,-100,color=colors_from_wigner[1], label=r"$r_+$" )      
    ax.plot(-100,-100,color=colors_red[3], label=r"$r_+$" )   
    ax.plot(-100,-100,'--', color="grey",label=r"$H_+$" )      
    ax.plot(-100,-100,color="grey", label=r"$H_+$" )      
       
            #ax.scatter((-1)**j*x_t[i,0],(-1)**j*p_t[i,0], marker= markers[i], label = labels[i], color="white" ,edgecolors='black',zorder=10)
            #ax.scatter((-1)**j*x_t[i,0], (-1)**j*p_t[i,0], marker= markers[i])
        

    ax.set_axisbelow(True)

    ax.xaxis.set_tick_params(width=0.5,length=3)
    ax.yaxis.set_tick_params(width=0.5,length=3)
    ax.grid(zorder=0, linewidth=0.6)

    scale = 1.1

    ax.set_xlim(-np.max(-rt[:,0])*scale, np.max(-rt[:,0])*scale)
    ax.set_ylim(-np.max(rt[:,1])*scale, np.max(rt[:,1])*scale)
                       
    ax.set_ylabel(r'$p $', fontsize=10,  labelpad=-6)
    ax.set_xlabel(r'$x$', fontsize=10,  labelpad=-1)
    ax.set_axisbelow(True)

    ax.set_xticks([-15,-10,-5,0,5,10,15], minor=False)
    ax.set_yticks([-10,-5,0,5,10], minor=False)
    ax.xaxis.tick_bottom()
    ax.set_xlabel(r'$x $')
    ax.set_xticklabels([r'$-15$',r'$-10$',r'$-5$', r'$0$',r'$5$',r'$10$',r'$15$'], fontsize=6.5)
    ax.set_yticklabels([r'$-10$',r'$-5$', r'$0$',r'$5$',r'$10$'], fontsize=6.5)

    for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(0.5)

    
    leg = fig.legend( loc='upper center', bbox_to_anchor=(0.995, 0.77), fontsize=8.5,labelspacing=0.7,handlelength=1.4)
    for line in leg.get_lines():
        line.set_linewidth(1.1)
    leg.get_frame().set_linewidth(0.4)
    #fig.suptitle('(b)',x=0.06,y=0.98, fontsize=12)
    
    if save == False:
        pass
    else:
        plt.savefig(save, bbox_inches='tight')
    return 

def plot_frame_presentation(wigner_array, psi_x, psi_p, delta_x, x_array, p_array, save):
    
    
    #im = plot_wigner_paper_1(axes[0,0], wigner_array_on, delta_x, x_array, p_array, 'On Diagonal')
    im = plot_wigner_paper_1(axes[1,1], wigner_array, delta_x, x_array, p_array, 'Off Diagonal')
    
    axes[1,1].plot(x_array, psi_x, color='tab:blue',linewidth=1.5)

    
    #ax.set_xlim(3*np.pi,9/2*np.pi)
    #ax.set_ylim(-0.05, 0.55)
    
    cbar_ax = fig.add_axes([0.92, 0.4, 0.03, 0.45])
    bar = fig.colorbar(im, cax=cbar_ax , orientation='vertical', alpha=np.linspace(0,1,100))
    bar.set_ticks([-0.5,0,0.5])
    bar.set_label(r'$ W(r,t)$',fontsize=8.5,labelpad=8, y=0.42, ha='right', rotation=-90)
    #plt.subplots_adjust(wspace=0.1, hspace=-0.27)

    if save == False:
        return 
    else: 
        plt.savefig(save, bbox_inches='tight')
        return
         
def plot_wigner_paper_1(ax, wigner_array, delta_x, x_array, p_array, title):

    colors1 = ["lightgreen", "limegreen", "forestgreen","darkgreen"]
    colors2 = [ "tomato", "red", "darkred", "black"]

    im = ax.imshow(wigner_array, cmap='RdBu',  interpolation='bilinear', 
                    #alpha=np.abs(wigner_array)/np.max(wigner_array), 
                    origin='lower', vmin=-0.5, vmax=0.5,
                    aspect=len(x_array)/len(p_array),
                    extent = [np.min(x_array), np.max(x_array), np.min(p_array)*len(p_array)/len(x_array), np.max(p_array)*len(p_array)/len(x_array)])

        
        #ax[i].grid(linewidth=0.2, zorder=0)

    
    #ax.grid(zorder=5)
    #fig.suptitle('(c)',x=0.06,y=0.83, fontsize=12)
    
    return im

def set_up_figure(wigner_array, psi_x, psi_p, delta_x, x_array, p_array, Nphonons, squeezing_r):

    rcParams['mathtext.fontset'] = 'cm'
    rcParams['font.family'] = 'STIXGeneral'
    
    fig, axes = plt.subplots(2, 2, figsize=(4.5,4.5), height_ratios=[1,2.5],width_ratios=[1, 2.5], dpi = 400)
    
    axes[0,0].axis('off')
    
    #axes[0,0].set_xticks([], minor=False)
    #axes[0,0].set_xticklabels([], fontsize=8)
    axes[0,0].set_xlim(0.1,0.2)
    axes[0,0].set_ylim(0.1,0.2)
    axes[0,0].scatter(1,1,s=1, color='black',label=r'$\delta x = {}$'.format(delta_x))
    axes[0,0].scatter(1,1,s=1, color='black',label=r'$N_p = {}$'.format(Nphonons))
    axes[0,0].scatter(1,1,s=1, color='black',label=r'$r_s = {}$'.format(squeezing_r))
    
    axes[0,0].legend(loc='center',fontsize = 10,handletextpad=0.1)
    
    axes[1,0].axvline(x=0, color='black',linewidth=0.5, alpha=0.4)
    axes[1,0].axhline(y=0, color='black',linewidth=0.5, alpha=0.4)
    axes[1,0].set_yticks([], minor=False)
    #axes[0,0].set_yticks([-delta_x,-delta_x/2,0,delta_x/2,delta_x], minor=False)
    #axes[0,0].set_yticklabels([r'$-\delta x$',r'$- \frac{\delta x}{2}$',r'$0$',r'$\frac{\delta x}{2}$',r'$\delta x$'], fontsize=8)
    #axes[0,0].set_xlabel(r'$ p $',labelpad=2, fontsize=9)
    axes[1,0].set_xlabel(r'$ | \psi(p)|^2 $' ,labelpad=6, fontsize=10)
    axes[1,0].set_xticks([0,0.1,0.2], minor=False)
    axes[1,0].set_xticklabels([r'$0$',r'$0.1$',r'$0.2$'], fontsize=8)
    axes[1,0].set_xlim(-0.05,0.25)
    axes[1,0].xaxis.set_label_position("top")
    axes[1,0].invert_xaxis()

    axes[1,0].axvline(x=0, color='black',linewidth=0.5, alpha=0.4)
    axes[1,0].axhline(y=0, color='black',linewidth=0.5, alpha=0.4)
    axes[0,1].set_xticks([], minor=False)
    #axes[1,1].set_xticks([-delta_x,-delta_x/2,0,delta_x/2,delta_x], minor=False)
    #axes[1,1].set_xticklabels([r'$-\delta x$',r'$- \frac{\delta x}{2}$',r'$0$',r'$\frac{\delta x}{2}$',r'$\delta x$'], fontsize=8)
    #axes[1,1].set_ylabel(r'$ p $',labelpad=2, fontsize=9)
    axes[0,1].set_ylabel(r'$ | \psi(x)|^2 $' ,labelpad=14, fontsize=10,rotation=-90)
    axes[0,1].set_yticks([0,0.1,0.2], minor=False)
    axes[0,1].set_yticklabels([r'$0$',r'$0.1$',r'$0.2$'], fontsize=8)
    axes[0,1].set_ylim(-0.05,0.25)
    axes[0,1].yaxis.set_label_position("right")
    
    axes[1,1].axvline(x=0, color='black',linewidth=0.5, alpha=0.4)
    axes[1,1].axhline(y=0, color='black',linewidth=0.5, alpha=0.4)
    
    axes[1,1].set_xticks([-delta_x,-delta_x/2,0,delta_x/2,delta_x], minor=False)
    axes[1,1].set_xticklabels([r'$-\delta x$',r'$- \frac{\delta x}{2}$',r'$0$',r'$\frac{\delta x}{2}$',r'$\delta x$'], fontsize=9)
    

    axes[1,1].set_yticks([-delta_x,-delta_x/2,0,delta_x/2,delta_x], minor=False)
    axes[1,1].set_yticklabels([r'$-\delta x$',r'$- \frac{\delta x}{2}$',r'$0$',r'$\frac{\delta x}{2}$',r'$\delta x$'], fontsize=9)
    
    axes[1,1].set_ylabel(r'$ p $' ,labelpad=-10, fontsize=11)
    axes[1,1].set_xlabel(r'$ x $',labelpad=7, fontsize=11)
    axes[1,1].xaxis.set_label_position("top")

    
    #ax.set_title(title, fontsize= 10)
    #x[i].set_ylabel(r'$ p$',labelpad=-4)
    #ax[i].set_yticklabels([r'$-10$',r'$0$',r'$10$'], fontsize=6.5)
    for axis in ['top','bottom','left','right']:
        axes[0,1].spines[axis].set_linewidth(0.5)
        axes[1,1].spines[axis].set_linewidth(0.5)
        axes[1,0].spines[axis].set_linewidth(0.5)
            
    axes[0,1].axvline(x=0, color='black',linewidth=0.5, alpha=0.4)
    axes[0,1].axhline(y=0, color='black',linewidth=0.5, alpha=0.4)

    #title_box = dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2', linewidth=0.4)
    #axes[1,1].text(0.97, 5.8, r'$W^+(x,p)$', transform=ax.transAxes, fontsize=10,verticalalignment='top', bbox=title_box)

   
    line_x = axes[0,1].plot(x_array, psi_x, color='tab:blue',linewidth=1.5)
    line_p = axes[1,0].plot(psi_p, x_array, color='tomato',linewidth=1.5)
    im = axes[1,1].imshow(wigner_array, cmap='RdBu',  interpolation='bilinear', 
                        #alpha=np.abs(wigner_array)/np.max(wigner_array), 
                        origin='lower', vmin=-0.5, vmax=0.5,
                        aspect=len(x_array)/len(p_array),
                        extent = [np.min(x_array), np.max(x_array), np.min(p_array)*len(p_array)/len(x_array), np.max(p_array)*len(p_array)/len(x_array)])


    cbar_ax = fig.add_axes([0.869, 0.13, 0.03, 0.45])
    bar = fig.colorbar(im, cax=cbar_ax , orientation='vertical', alpha=np.linspace(0,1,100))
    bar.set_ticks([-0.5,0,0.5])
    bar.set_ticklabels([r'$-0.5$',r'$0$',r'$0.5$'],fontsize=9)
    bar.set_label(r'$ W(r,t)$',fontsize=10,labelpad=1, y=0.42, ha='right', rotation=-90)
    
    return fig, im, line_x, line_p


# ══════════════════════════════════════════════════════════════════════════════
#  3-D Phase-Space Plots  (Plot_Trajectories_Single, Plot_Vectors_Single,
#                          Plot_Vectors_Ellipsoids_Single)
# ══════════════════════════════════════════════════════════════════════════════

# ── default color palettes ────────────────────────────────────────────────────
_COLORS_BLUE_3D  = ['lightskyblue', 'cornflowerblue', 'royalblue',  'blue',   'darkblue']
_COLORS_RED_3D   = ['salmon',       'tomato',          'orangered',  'red',    'darkred']

_INDEX_MAP_3D = [
    ((1, 1), r"$r^{\mathrm{on}}_{+}$",  'real+'),
    ((0, 0), r"$r^{\mathrm{on}}_{-}$",  'real-'),
    ((0, 1), r"$r^{\mathrm{off}}$",      'off'),
]

_DEFAULT_COLOR_MAP_3D = {
    'real+': _COLORS_BLUE_3D[1],
    'real-': _COLORS_BLUE_3D[3],
    'off':   _COLORS_RED_3D[1],
}

# Pre-computed unit-sphere grids (module-level, reused for every ellipsoid)
_N_LON_3D, _N_LAT_3D = 12, 5
_u_sph_3d = np.linspace(0, 2*np.pi, 80)
_v_sph_3d = np.linspace(0,   np.pi, 80)
_LAT_V_3D = np.linspace(0, np.pi, _N_LAT_3D + 2)[1:-1]   # exclude poles
_LON_U_3D = np.linspace(0, 2*np.pi, _N_LON_3D, endpoint=False)


# ─────────────────────────────────────────────────────────────────────────────
#  PRIVATE DRAWING PRIMITIVES
# ─────────────────────────────────────────────────────────────────────────────

def _triangle_arrowhead_3d(ax, tip, direction, head_len, head_width, color, alpha=1.0):
    """Flat filled 2-D triangle arrowhead embedded in 3-D space."""
    base = tip - direction * head_len
    ref  = np.array([0., 0., 1.]) if abs(direction[2]) < 0.9 else np.array([1., 0., 0.])
    perp = np.cross(direction, ref)
    perp /= np.linalg.norm(perp)
    v0, v1, v2 = tip, base + perp*(head_width/2), base - perp*(head_width/2)
    ax.add_collection3d(Poly3DCollection([[v0, v1, v2]],
                        facecolor=color, edgecolor=color, alpha=alpha,
                        linewidth=0.5, zorder=15))


def _draw_axis_arrow_3d(ax, neg_end, pos_end, label, tick_vals,
                        lw=1.2, color='black', fontsize=13,
                        label_offset=0.12, tick_size=0.06):
    """Shaft + triangle head + tick marks + axis label for one coordinate axis."""
    direction = pos_end - neg_end
    total_len = np.linalg.norm(direction)
    direction = direction / total_len
    head_len  = total_len * 0.08
    shaft_tip = pos_end - direction * head_len

    ax.plot(*zip(neg_end, shaft_tip), color=color, linewidth=lw, zorder=10)
    _triangle_arrowhead_3d(ax, pos_end, direction, head_len, head_len*0.7, color)

    label_pos = pos_end + direction * label_offset
    ax.text(*label_pos, label, color=color, fontsize=fontsize,
            ha='center', va='center', zorder=11)

    ref  = np.array([0., 0., 1.]) if abs(direction[2]) < 0.9 else np.array([1., 0., 0.])
    perp = np.cross(direction, ref);  perp /= np.linalg.norm(perp)
    for tv in tick_vals:
        tc = direction * tv
        ax.plot(*zip(tc - perp*tick_size, tc + perp*tick_size),
                color=color, linewidth=0.8, zorder=10)
        ax.text(*(tc + perp*(tick_size*6.0)), str(int(round(tv))),
                color=color, fontsize=fontsize-2, ha='center', va='center', zorder=11)


def _setup_axes_decoration_3d(ax, sym_lim, xlabel, ylabel, zlabel,
                               fontsize=13, axis_extend=0.18):
    """Turn off Matplotlib default decorations; draw custom cross-at-origin axes
    with tick marks and grid lines on the z=0 plane.
    """
    ax.set_axis_off()
    lo, hi = sym_lim
    ticks       = np.arange(int(np.ceil(lo)), int(np.floor(hi)) + 1, dtype=float)
    label_ticks = ticks[ticks != 0]
    ext         = (hi - lo) * axis_extend

    axes_def = [
        (np.array([lo, 0., 0.]), np.array([hi+ext, 0., 0.]), xlabel),
        (np.array([0., lo, 0.]), np.array([0., hi+ext, 0.]), ylabel),
        (np.array([0., 0., lo]), np.array([0., 0., hi+ext]), zlabel),
    ]
    for neg_end, pos_end, lbl in axes_def:
        _draw_axis_arrow_3d(ax, neg_end, pos_end, lbl, label_ticks,
                            fontsize=fontsize, label_offset=0.10 + ext*0.3)

    ax.set_xlim(lo, hi+ext)
    ax.set_ylim(lo, hi+ext)
    ax.set_zlim(lo, hi+ext)

    for tv in label_ticks:
        ax.plot([tv, tv], [lo, hi], [0, 0], color='0.65', lw=0.6, alpha=0.45, zorder=2)
        ax.plot([lo, hi], [tv, tv], [0, 0], color='0.65', lw=0.6, alpha=0.45, zorder=2)


def _safe_sqrt_3d(val, label=""):
    """sqrt with clipping; warns if value was negative."""
    if val < 0:
        print(f"  Warning: negative value {val:.4g} under sqrt ({label}), clipped to 0.")
    return np.sqrt(max(float(val), 0.0))


def _draw_vector_3d(ax, tip, color, lw=1.8, head_frac=0.12, alpha=1.0):
    """Draw a single arrow from the origin to tip: shaft + flat triangle head."""
    tip    = np.asarray(tip, dtype=float)
    origin = np.zeros(3)
    mag    = np.linalg.norm(tip)
    if mag < 1e-12:
        return

    direction  = tip / mag
    head_len   = mag * head_frac
    head_width = head_len * 0.7
    shaft_tip  = tip - direction * head_len

    ax.plot([origin[0], shaft_tip[0]],
            [origin[1], shaft_tip[1]],
            [origin[2], shaft_tip[2]],
            color=color, linewidth=lw, alpha=alpha, zorder=10)

    _triangle_arrowhead_3d(ax, tip, direction, head_len, head_width, color, alpha=alpha)


def _draw_ellipsoid_3d(ax, cx, cy, cz, ax_x, ax_y, ax_z, color,
                       surf_alpha=0.18, line_alpha=0.20, lw=0.7):
    """Draw an axis-aligned ellipsoid at (cx,cy,cz) with geodesic lines."""
    U, V = np.meshgrid(np.linspace(0, 2*np.pi, 40),
                       np.linspace(0,     np.pi, 20))
    ax.plot_surface(cx + ax_x*np.cos(U)*np.sin(V),
                    cy + ax_y*np.sin(U)*np.sin(V),
                    cz + ax_z*np.cos(V),
                    color=color, alpha=surf_alpha, linewidth=0, antialiased=True)

    for v0 in _LAT_V_3D:
        ax.plot(cx + ax_x*np.cos(_u_sph_3d)*np.sin(v0),
                cy + ax_y*np.sin(_u_sph_3d)*np.sin(v0),
                cz + ax_z*np.cos(v0)*np.ones_like(_u_sph_3d),
                color=color, linewidth=lw, alpha=line_alpha)

    for u0 in _LON_U_3D:
        ax.plot(cx + ax_x*np.cos(u0)*np.sin(_v_sph_3d),
                cy + ax_y*np.sin(u0)*np.sin(_v_sph_3d),
                cz + ax_z*np.cos(_v_sph_3d),
                color=color, linewidth=lw, alpha=line_alpha)

    # Outer contour: equatorial ring (widest cross-section, always at z=cz)
    ax.plot(cx + ax_x*np.cos(_u_sph_3d),
            cy + ax_y*np.sin(_u_sph_3d),
            cz + np.zeros_like(_u_sph_3d),
            color=color, linewidth=1.4, alpha=min(line_alpha * 4.0, 0.85), zorder=7)


# ─────────────────────────────────────────────────────────────────────────────
#  PRIVATE SCENE BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def _make_scene_3d(sym_lim, z_comp=0, figsize=(10, 10), elev=22, azim=-55):
    """Create a figure with a single 3-D axes, expanded to fill the canvas."""
    fig = plt.figure(figsize=figsize)
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_position([-0.05, -0.05, 1.10, 1.05])
    ax.view_init(elev=elev, azim=azim)
    return fig, ax


def _finalise_3d(ax, fig, sym_lim, z_comp,
                 title="", legend_handles=None,
                 fontsize=13, FS_TITLE=15, FS_LEG=12):
    """Must be called LAST: draws axis decorations, title, and legend."""
    zlabel = r"$\mathrm{Im}[x]$" if z_comp == 0 else r"$\mathrm{Im}[p]$"
    _setup_axes_decoration_3d(ax, sym_lim,
                               xlabel=r"$\mathrm{Re}[x]$",
                               ylabel=r"$\mathrm{Re}[p]$",
                               zlabel=zlabel,
                               fontsize=fontsize)

    if title:
        ax.set_title(title, fontsize=FS_TITLE, y=0.85)

    if legend_handles is None:
        handles, labels = ax.get_legend_handles_labels()
        legend_handles  = handles
    if legend_handles:
        fig.legend(legend_handles, [h.get_label() for h in legend_handles],
                   loc='lower center', ncol=4,
                   fontsize=FS_LEG, framealpha=0.7,
                   bbox_to_anchor=(0.5, 0.3))

    fig.subplots_adjust(top=0.98, bottom=0.08)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC DRAWERS  (animation-friendly — figure-independent)
# ─────────────────────────────────────────────────────────────────────────────

def Draw_XY_Plane(ax, sym_lim):
    """Draw the z=0 reference plane as a lightly shaded gray surface."""
    lo, hi = sym_lim
    xx, yy = np.meshgrid([lo, hi], [lo, hi])
    ax.plot_surface(xx, yy, np.zeros_like(xx),
                    color='gray', alpha=0.08,
                    linewidth=0, antialiased=False, zorder=1)


def Draw_Curves(ax, r_JK_t, z_comp, color_map=None):
    """Draw trajectory curves from r_JK_t with directional arrowheads.

    Parameters
    ----------
    ax        : Axes3D returned by _make_scene_3d
    r_JK_t    : ndarray (T, 2, 2, 2, 1) complex
    z_comp    : 0 -> Im[x] as z,  1 -> Im[p] as z
    color_map : dict role->colour (uses _DEFAULT_COLOR_MAP_3D if None)
    """
    if color_map is None:
        color_map = _DEFAULT_COLOR_MAP_3D

    T = r_JK_t.shape[0]

    for (j, k), label, role in _INDEX_MAP_3D:
        color = color_map[role]
        vec   = r_JK_t[:, j, k, :, 0]
        rx    = vec[:, 0].real
        rp    = vec[:, 1].real
        z     = vec[:, z_comp].imag

        ax.plot(rx, rp, z, color=color, linewidth=1.4, alpha=0.6, label=label)

        for frac in (1/3, 2/3):
            i = max(1, min(int(round(frac * (T - 1))), T - 2))
            d = np.array([rx[i]-rx[i-1], rp[i]-rp[i-1], z[i]-z[i-1]])
            norm = np.linalg.norm(d)
            if norm < 1e-12:
                continue
            _triangle_arrowhead_3d(ax, np.array([rx[i], rp[i], z[i]]),
                                   d/norm, head_len=0.3, head_width=0.3,
                                   color=color)


def Draw_Vectors(ax, r_JK_t, t_idx, z_comp, color_map=None, alpha=1.0, add_legend_entry=True):
    """Draw first-moment vectors as arrows from origin at time t_idx.

    Parameters
    ----------
    ax               : Axes3D returned by _make_scene_3d
    r_JK_t           : ndarray (T, 2, 2, 2, 1) complex
    t_idx            : int — time index to snapshot
    z_comp           : 0 -> tip z = Im[x],  1 -> tip z = Im[p]
    color_map        : dict role->colour (uses _DEFAULT_COLOR_MAP_3D if None)
    alpha            : float — opacity of arrows (default 1.0)
    add_legend_entry : bool — whether to add dummy legend handles (default True)
    """
    if color_map is None:
        color_map = _DEFAULT_COLOR_MAP_3D

    for (j, k), label, role in _INDEX_MAP_3D:
        color = color_map[role]
        vec   = r_JK_t[t_idx, j, k, :, 0]
        rx    = vec[0].real
        rp    = vec[1].real
        z     = vec[z_comp].imag

        if add_legend_entry:
            ax.plot([], [], [], color=color, linewidth=1.8, label=label)
        _draw_vector_3d(ax, np.array([rx, rp, z]), color, alpha=alpha)


def Draw_Ellipsoids(ax, r_JK_t, sigma_JK_t, t_idx, z_comp, color_map=None, alpha=1.0):
    """Draw one flat, axis-aligned ellipse per on-diagonal (j,j) branch at time t_idx.

    Off-diagonal entries are skipped here; draw them with Draw_Tilted_Pancake_3D.
    sigma_JJ and r_JJ are physically real, so the ellipse lies in the z=0 plane
    with semi-axes ax_x=sqrt(Re[s00]), ax_y=sqrt(Re[s11]), ax_z=0.

    Parameters
    ----------
    ax           : Axes3D returned by _make_scene_3d
    r_JK_t       : ndarray (T, 2, 2, 2, 1) complex
    sigma_JK_t   : ndarray (T, 2, 2, 2, 2) complex
    t_idx        : int — time index
    z_comp       : unused; kept for signature parity with the other Draw_* helpers
    color_map    : dict role->colour (uses _DEFAULT_COLOR_MAP_3D if None)
    alpha        : float in (0, 1] — overall opacity scale (default 1.0)
    """
    if color_map is None:
        color_map = _DEFAULT_COLOR_MAP_3D

    for (j, k), label, role in _INDEX_MAP_3D:
        if j != k:
            continue
        color = color_map[role]
        vec   = r_JK_t[t_idx, j, k, :, 0]
        sig_src = sigma_JK_t[t_idx, j, k]

        # On-diagonal covariance and first moment are physically real; take real
        # parts explicitly so ODE numerical noise doesn't produce phantom z-tilt.
        cx = float(np.real(vec[0]))
        cy = float(np.real(vec[1]))
        cz = 0.0

        s00 = float(np.real(sig_src[0, 0]))
        s11 = float(np.real(sig_src[1, 1]))

        ax_x = _safe_sqrt_3d(s00, f"Re[sigma_00] ({role})")
        ax_y = _safe_sqrt_3d(s11, f"Re[sigma_11] ({role})")
        ax_z = 0.0

        _draw_ellipsoid_3d(ax, cx, cy, cz, ax_x, ax_y, ax_z, color,
                           surf_alpha=0.18*alpha, line_alpha=0.25*alpha)


# ─────────────────────────────────────────────────────────────────────────────
#  OFF-DIAGONAL DECOMPOSITION AND TILTED PANCAKE
# ─────────────────────────────────────────────────────────────────────────────

def decompose_sigma_off_diagonal(sigma_off, atol=1e-10):
    """Try to write sigma_off = U Sigma U^T with real orthogonal U and complex diagonal Sigma.

    Valid only for n=1 (2×2 matrices). Uses the commutativity criterion: [Re(σ), Im(σ)] = 0
    is necessary and sufficient for simultaneous real diagonalization.

    Returns
    -------
    U         : (2, 2) real orthogonal — rotation matrix
    Sigma     : (2, 2) complex diagonal — diagonal covariance in rotated frame
    magnitudes: (2,) float — |Sigma_ii|
    phases    : (2,) float — arg(Sigma_ii) in radians

    Raises
    ------
    ValueError if sigma_off cannot be decomposed with a real orthogonal U.
    """
    A = np.real(sigma_off)
    B = np.imag(sigma_off)

    if not np.allclose(A, A.T, atol=atol):
        raise ValueError("Re(sigma_off) is not symmetric — cannot decompose with real orthogonal U.")
    if not np.allclose(B, B.T, atol=atol):
        raise ValueError("Im(sigma_off) is not symmetric — cannot decompose with real orthogonal U.")

    comm = A @ B - B @ A
    if np.max(np.abs(comm)) > atol:
        raise ValueError(
            f"Re(sigma_off) and Im(sigma_off) do not commute "
            f"(max |[A,B]| = {np.max(np.abs(comm)):.2e}); "
            "sigma_off cannot be written as U Sigma U^T with real orthogonal U."
        )

    _, U = np.linalg.eigh(A)
    U = np.real(U)

    # eigh returns columns in ascending eigenvalue order.
    # Re-order so that the eigenvector with the larger |x-component| (U[0,j]) comes first,
    # ensuring j=0 always corresponds to the x-like axis and j=1 to the p-like axis —
    # consistent with canonical (x,p) ordering and the z_comp=0 tilting convention.
    if abs(U[0, 0]) < abs(U[0, 1]):
        U = U[:, ::-1]

    B_rot = U.T @ B @ U
    off_diag = B_rot - np.diag(np.diag(B_rot))
    if np.max(np.abs(off_diag)) > atol:
        raise ValueError(
            f"U^T Im(sigma_off) U is not diagonal "
            f"(max off-diag = {np.max(np.abs(off_diag)):.2e}); "
            "sigma_off cannot be written as U Sigma U^T with real orthogonal U."
        )

    Sigma = U.T @ sigma_off @ U
    diag_entries = np.diag(Sigma)
    magnitudes = np.abs(diag_entries)
    phases = np.angle(diag_entries)

    return U, Sigma, magnitudes, phases


def transform_to_diagonal_frame(sigma_off, sigma_on_plus, sigma_on_minus,
                                 r_off, r_on_plus, r_on_minus, atol=1e-10):
    """Rotate all GCS phase-space quantities to the diagonal frame of sigma_off.

    Decomposes sigma_off = U Sigma U^T (n=1, 2×2 only) and applies U^T to every
    covariance matrix and first-moment vector.

    Parameters
    ----------
    sigma_off      : (2, 2) complex — off-diagonal covariance
    sigma_on_plus  : (2, 2) complex — on-diagonal (+,+) covariance
    sigma_on_minus : (2, 2) complex — on-diagonal (-,-) covariance
    r_off          : (2, 1) complex — off-diagonal first moment
    r_on_plus      : (2, 1) complex — on-diagonal (+,+) first moment
    r_on_minus     : (2, 1) complex — on-diagonal (-,-) first moment
    atol           : float — tolerance passed to decompose_sigma_off_diagonal

    Returns
    -------
    dict with keys U, Sigma, magnitudes, phases, sigma_off_diag,
    sigma_on_plus_rot, sigma_on_minus_rot, r_off_rot, r_on_plus_rot, r_on_minus_rot;
    or None if sigma_off is not real-diagonalizable (error printed).
    """
    try:
        U, Sigma, magnitudes, phases = decompose_sigma_off_diagonal(sigma_off, atol=atol)
    except ValueError as e:
        print(f"[transform_to_diagonal_frame] Not plottable: {e}")
        return None

    def _rot_sigma(s):
        return U.T @ s @ U

    def _rot_r(r):
        return U.T @ r

    return {
        'U': U,
        'Sigma': Sigma,
        'magnitudes': magnitudes,
        'phases': phases,
        'sigma_off_diag': Sigma,
        'sigma_on_plus_rot': _rot_sigma(sigma_on_plus),
        'sigma_on_minus_rot': _rot_sigma(sigma_on_minus),
        'r_off_rot': _rot_r(r_off),
        'r_on_plus_rot': _rot_r(r_on_plus),
        'r_on_minus_rot': _rot_r(r_on_minus),
    }


def Draw_Tilted_Pancake_3D(ax, center, sigma_off, z_comp=0, color='tomato',
                            alpha_fill=0.22, alpha_edge=0.8, n_pts=300,
                            label=None, n_meridians=6, n_parallels=2):
    """DEPRECATED -- superseded by Draw_Off_Diagonal_Tilt_3D / Plot_Off_Diagonal_3D.

    Kept intact while the new path is validated; do not build on it.  It differs from the
    new path in two substantive ways: (a) its semi-axes go as sqrt(|Sigma_jj|), whereas the
    exact construction needs a radial coordinate LINEAR in the matrix entry, so that the
    xp-shadow is Re(b_j) and the tips are Im(b_j); (b) it lifts only the axis matching
    z_comp, leaving the other in-plane, whereas both axes carry their own phase.

    Draw the off-diagonal covariance as a tilted pancake (flat 2-D ellipse) in 3-D phase space.

    sigma_off must be a 2×2 complex diagonal matrix (n=1 only). Magnitudes and phases
    are read directly from the diagonal: Sigma_jj = |Sigma_jj| e^{i phi_j}.

    Semi-axis vectors (axes aligned with canonical x/p since sigma_off is diagonal):
        j == z_comp: e = sqrt(|S_jj|) * [cos(phi_j), 0, sin(phi_j)]  (x tilted toward Im)
        j != z_comp: e = sqrt(|S_jj|) * [0, 1, 0]                    (p stays in-plane)

    Parameters
    ----------
    ax           : Axes3D
    center       : array-like (3,) — [cx, cy, cz] in (Re[x], Re[p], Im[x_z_comp]) coordinates
    sigma_off    : (2, 2) complex diagonal — off-diagonal covariance (n=1 only)
    z_comp       : 0 -> Im[x] on z-axis,  1 -> Im[p] on z-axis
    color        : face and edge colour
    alpha_fill   : face transparency (0–1)
    alpha_edge   : outline opacity (0–1)
    n_pts        : number of perimeter sample points
    label        : legend label string (None = no legend entry)
    n_meridians  : number of diameter lines drawn through the center (world-map meridians)
    n_parallels  : number of concentric scaled ellipses inside the pancake (latitudes)
    """
    diag = np.array([sigma_off[0, 0], sigma_off[1, 1]])
    magnitudes = np.abs(diag)
    phases = np.angle(diag)

    cx, cy, cz = np.asarray(center, dtype=float)
    theta = np.linspace(0, 2 * np.pi, n_pts)

    # U = I since sigma_off is diagonal: x-axis is j=0 (ux=1,up=0), p-axis is j=1 (ux=0,up=1)
    axes_3d = []
    for j in range(2):
        amp = np.sqrt(magnitudes[j])
        ux = 1.0 if j == 0 else 0.0
        up = 0.0 if j == 0 else 1.0
        phi_j = phases[j]
        if j == z_comp:
            e = amp * np.array([ux * np.cos(phi_j), up * np.cos(phi_j), np.sin(phi_j)])
        else:
            e = amp * np.array([ux, up, 0.0])
        axes_3d.append(e)

    e1, e2 = axes_3d
    xs = cx + e1[0]*np.cos(theta) + e2[0]*np.sin(theta)
    ys = cy + e1[1]*np.cos(theta) + e2[1]*np.sin(theta)
    zs = cz + e1[2]*np.cos(theta) + e2[2]*np.sin(theta)

    verts = [list(zip(xs, ys, zs))]
    poly = Poly3DCollection(verts, alpha=alpha_fill,
                            facecolor=color, edgecolor='none', zorder=5)
    ax.add_collection3d(poly)

    grid_alpha = 0.35 * alpha_edge
    grid_lw = 0.7

    # Meridians: n_meridians diameter lines through the center
    for k in range(n_meridians):
        alpha_m = k * np.pi / n_meridians
        p1 = np.array([cx + e1[0]*np.cos(alpha_m) + e2[0]*np.sin(alpha_m),
                       cy + e1[1]*np.cos(alpha_m) + e2[1]*np.sin(alpha_m),
                       cz + e1[2]*np.cos(alpha_m) + e2[2]*np.sin(alpha_m)])
        p2 = np.array([cx - e1[0]*np.cos(alpha_m) - e2[0]*np.sin(alpha_m),
                       cy - e1[1]*np.cos(alpha_m) - e2[1]*np.sin(alpha_m),
                       cz - e1[2]*np.cos(alpha_m) - e2[2]*np.sin(alpha_m)])
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                color=color, linewidth=grid_lw, alpha=grid_alpha, zorder=5)

    # Parallels: n_parallels concentric scaled ellipses
    for f in np.linspace(1.0 / (n_parallels + 1), n_parallels / (n_parallels + 1), n_parallels):
        xs_p = cx + f * e1[0] * np.cos(theta) + f * e2[0] * np.sin(theta)
        ys_p = cy + f * e1[1] * np.cos(theta) + f * e2[1] * np.sin(theta)
        zs_p = cz + f * e1[2] * np.cos(theta) + f * e2[2] * np.sin(theta)
        ax.plot(xs_p, ys_p, zs_p, color=color, linewidth=grid_lw, alpha=grid_alpha, zorder=5)

    lbl = label if label is not None else '_nolegend_'
    ax.plot(xs, ys, zs, color=color, linewidth=1.4, alpha=alpha_edge, label=lbl, zorder=6)

    ax.plot(xs, ys, np.full_like(zs, cz), '--',
            color=color, linewidth=0.8, alpha=0.4, zorder=4)


# ─────────────────────────────────────────────────────────────────────────────
#  COVARIANCE CONTOURS (2-D) AND THE OFF-DIAGONAL TILT (3-D)
# ─────────────────────────────────────────────────────────────────────────────

def Covariance_Geometry(sigma, c=1.0, radial='sqrt', iso_tol=1e-12):
    """Principal geometry of the level set xi^T sigma^-1 xi = c^2.

    `sigma` is the covariance matrix itself, not its inverse, so more noise gives a
    bigger contour.  With eigenvalues mu_plus >= mu_minus the semi-axes are c*sqrt(mu)
    (radial='sqrt') and theta is measured from +x counter-clockwise to the MAJOR axis.

    The paper fixes no contour level: it never draws a phase-space contour (no level-set
    convention appears anywhere in it).  c is therefore a plotting choice, defaulting to
    1, which makes the vacuum (sigma = 1, Sec. II) the unit circle.  The normalisation of
    Eq.(wigner_general_gaussian) -- which integrates to 2^n, not 1 -- scales the height of
    W_g and not this geometry, since sigma^-1 sits in the exponent either way.

    Parameters
    ----------
    sigma   : (2, 2) real symmetric positive-definite
    c       : contour level (see above)
    radial  : 'sqrt'   -> semi-axes c*sqrt(mu), the standard quantum-optics contour;
                          semi-axes are standard deviations, aspect ratio exp(2r).
              'linear' -> semi-axes c*mu, the image of the unit circle under sigma.
                          Radially commensurate with Draw_Off_Diagonal_Tilt_3D, but NOT
                          a contour of the Wigner function: the aspect ratio becomes
                          exp(4r), so a 10 dB state draws at 100 rather than 10, and the
                          semi-axes are no longer standard deviations.
    iso_tol : relative eigenvalue split below which the contour is called a circle

    Returns
    -------
    dict with mu_plus, mu_minus, theta, semi_major, semi_minor, radial, is_isotropic
    """
    sigma = np.asarray(sigma, dtype=float)
    T = (sigma[0, 0] + sigma[1, 1])/2
    D = np.hypot((sigma[0, 0] - sigma[1, 1])/2, sigma[0, 1])
    mu_plus = T + D
    if mu_plus <= 0:
        raise ValueError(f"sigma is not positive definite: largest eigenvalue {mu_plus}")
    # via the determinant, not T - D, which cancels catastrophically past ~10 dB
    mu_minus = float(np.linalg.det(sigma))/mu_plus

    is_isotropic = (mu_plus - mu_minus)/(mu_plus + mu_minus) < iso_tol
    # 0.5*atan2 already lands in (-pi/2, pi/2]; arctan would be ambiguous by pi/2 and
    # would silently swap major and minor
    theta = 0.0 if is_isotropic else 0.5*np.arctan2(2*sigma[0, 1], sigma[0, 0] - sigma[1, 1])

    if radial == 'sqrt':
        a, b = c*np.sqrt(mu_plus), c*np.sqrt(mu_minus)
    elif radial == 'linear':
        a, b = c*mu_plus, c*mu_minus
    else:
        raise ValueError(f"radial must be 'sqrt' or 'linear', got {radial!r}")

    return dict(mu_plus=mu_plus, mu_minus=mu_minus, theta=float(theta),
                semi_major=float(a), semi_minor=float(b),
                radial=radial, c=c, is_isotropic=bool(is_isotropic))


def Draw_Covariance_Contour(ax, sigma, centre=(0.0, 0.0), c=1.0, radial='sqrt',
                            color='cornflowerblue', label=None, lw=1.6, alpha=1.0,
                            fill_alpha=0.18, show_axes=False, show_vacuum=False,
                            n_pts=400):
    """Draw the phase-space contour of a real symmetric covariance matrix into `ax`.

    Geometry and conventions are those of Covariance_Geometry, whose metadata dict is
    returned so callers can assert on the geometry instead of re-deriving it.

    Parameters
    ----------
    ax          : Axes (equal aspect is enforced here)
    sigma       : (2, 2) real symmetric positive-definite
    centre      : contour centre; the first moments do not move it
    c, radial   : see Covariance_Geometry
    show_axes   : draw the principal axes as line segments
    show_vacuum : draw the dashed unit reference circle (vacuum, sigma = 1)
    fill_alpha  : opacity of the filled interior (0 to leave it hollow)

    Returns
    -------
    dict -- the Covariance_Geometry metadata
    """
    g = Covariance_Geometry(sigma, c=c, radial=radial)
    t = np.linspace(0, 2*np.pi, n_pts)
    ct, st = np.cos(g['theta']), np.sin(g['theta'])
    xs = centre[0] + g['semi_major']*np.cos(t)*ct - g['semi_minor']*np.sin(t)*st
    ys = centre[1] + g['semi_major']*np.cos(t)*st + g['semi_minor']*np.sin(t)*ct

    if fill_alpha > 0:
        ax.fill(xs, ys, color=color, alpha=fill_alpha, linewidth=0, zorder=2)

    ax.plot(xs, ys, color=color, lw=lw, alpha=alpha,
            label=label if label is not None else '_nolegend_', zorder=3)

    if show_axes:
        for length, u in ((g['semi_major'], (ct, st)), (g['semi_minor'], (-st, ct))):
            ax.plot([centre[0] - length*u[0], centre[0] + length*u[0]],
                    [centre[1] - length*u[1], centre[1] + length*u[1]],
                    color=color, lw=0.8, alpha=0.7*alpha, zorder=3)
    if show_vacuum:
        ax.plot(centre[0] + np.cos(t), centre[1] + np.sin(t),
                color='0.4', lw=0.8, ls='--', alpha=0.7, zorder=1,
                label='vacuum' if label is not None else '_nolegend_')

    ax.set_aspect('equal', adjustable='box')
    return g


def Covariance_Invariants(sigma, hbar=1.0):
    """Symplectic invariants of a genuine (on-diagonal) covariance matrix.

    The paper works in hbar = 1: the uncertainty relation is sigma + i Omega >= 0
    (Sec. II), the vacuum is sigma = 1 and a thermal state sigma = (1 + 2 N_p) 1.

    `physical` uses det(sigma) >= hbar^2, equivalent to sigma + i hbar Omega >= 0, and
    NOT Var_x Var_p >= hbar^2/4, which is strictly weaker and passes unphysical states
    once correlations are present -- exactly the conditional-rotation regime.

    Returns
    -------
    dict with nu, purity, n_bar, r, S_dB, physical, det_sigma
    """
    g = Covariance_Geometry(sigma)
    det = float(np.linalg.det(np.asarray(sigma, dtype=float)))
    nu = np.sqrt(abs(det))
    return dict(det_sigma=det, nu=float(nu), purity=float(hbar/nu),
                n_bar=float((nu/hbar - 1)/2),
                r=float(0.25*np.log(g['mu_plus']/g['mu_minus'])),
                S_dB=float(-10*np.log10(g['mu_minus']/hbar)),   # 10, not 20: variance is power-like
                physical=bool(det >= hbar**2*(1 - 1e-12)),   # relative: scales with hbar
                mu_plus=g['mu_plus'], mu_minus=g['mu_minus'], theta=g['theta'],
                is_isotropic=g['is_isotropic'])


def Plot_On_Diagonal_Contours(sigma_on_plus, sigma_on_minus, c=1.0, radial='sqrt', sym_lim=None,
                              r_on_plus=None, r_on_minus=None, hbar=1.0,
                              color_map=None, title="", annotate=True,
                              show_vacuum=True, show_axes=False, fill_alpha=0.18,
                              figsize=(5.2, 5.2), save=False):
    """Figure 1: the two on-diagonal covariance contours, overlaid at the origin.

    Overlaid in one axis rather than side by side, matching Plot_Diagonal_Multi_Time_2D
    and Plot_Phase_Space_Summary -- the existing 2-D views of the same two objects -- and
    reusing their colours from _DEFAULT_COLOR_MAP_3D.

    Both contours are centred at the origin.  Non-zero first moments are drawn as markers
    only and their offset is reported in the annotation; they do not move the contours.

    Physicality is reported as a warning carrying the number, not an exception: fed the
    vacuum it must give det(sigma) = hbar^2 exactly, so a stray factor of 2 or a
    sigma-vs-V mix-up shows up immediately.

    Returns
    -------
    fig, ax, info -- info['+'] and info['-'] hold Covariance_Invariants for each branch
    """
    rcParams['mathtext.fontset'] = 'cm'
    rcParams['font.family'] = 'STIXGeneral'
    if color_map is None:
        color_map = _DEFAULT_COLOR_MAP_3D

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=200)
    info, lim = {}, 0.0
    for key, sig, rvec, role, lbl in (
            ('+', sigma_on_plus,  r_on_plus,  'real+', r"$\sigma^{\mathrm{on}}_{+}$"),
            ('-', sigma_on_minus, r_on_minus, 'real-', r"$\sigma^{\mathrm{on}}_{-}$")):
        inv = Covariance_Invariants(sig, hbar=hbar)
        g = Draw_Covariance_Contour(ax, sig, c=c, radial=radial,
                                    color=color_map[role], label=lbl,
                                    show_axes=show_axes, fill_alpha=fill_alpha,
                                    show_vacuum=(show_vacuum and key == '+'))
        inv.update(g)
        if not inv['physical']:
            warnings.warn(f"branch {key}: det(sigma) = {inv['det_sigma']:.6g} < hbar^2 = "
                          f"{hbar**2:.6g}; state is unphysical", RuntimeWarning, stacklevel=2)
        if rvec is not None:
            rv = np.asarray(rvec, dtype=float).ravel()
            ax.plot(rv[0], rv[1], marker='o', ms=4, color=color_map[role], zorder=4)
            inv['r_offset'] = rv
            lim = max(lim, float(np.max(np.abs(rv))))
        info[key] = inv

        if sym_lim== None:
            sym_lim = max(lim, g['semi_major'])
            sym_lim *= 1.35
   
    
    ax.set_xlim(-sym_lim, sym_lim); ax.set_ylim(-sym_lim, sym_lim)

    ax.axhline(0, color='0.6', lw=0.5, zorder=0)
    ax.axvline(0, color='0.6', lw=0.5, zorder=0)
    ax.set_xlabel(r'Position, $x$', fontsize=11)
    ax.set_ylabel(r'Momentum, $p$', fontsize=11)
    if title:
        ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8, loc='upper right', framealpha=0.8)

    if annotate:
        for i, (key, inv) in enumerate(info.items()):
            txt = (rf"${key}$:  $({inv['semi_major']:.3f},\,{inv['semi_minor']:.3f})$, "
                   rf"$\theta={np.degrees(inv['theta']):.1f}^\circ$, "
                   rf"$\mu={inv['purity']:.3f}$, ${inv['S_dB']:.2f}$ dB")
            if 'r_offset' in inv:
                txt += rf", $\bar r=({inv['r_offset'][0]:.2f},{inv['r_offset'][1]:.2f})$"
            ax.text(0.02, 0.02 + 0.055*i, txt, transform=ax.transAxes, fontsize=7)
        ax.text(0.02, 0.02 + 0.055*len(info),
                rf"radial='{radial}', $c={c:g}$", transform=ax.transAxes, fontsize=7)

    fig.tight_layout()
    if save:
        plt.savefig(save, bbox_inches='tight', transparent=True)
    return fig, ax, info


def Off_Diagonal_Polar(sigma_off, tol=1e-10, use_inverse=True):
    """Polar decomposition of a diagonal complex off-diagonal block, b_j = zeta_j e^{i psi_j}.

    `use_inverse` selects which object is decomposed, and it matters: inverting a diagonal
    entry sends zeta -> 1/zeta and psi -> -psi, reversing every tilt.

      True  (default) -- decompose sigma_off^-1, consistent with the on-diagonal contours
                         of Draw_Covariance_Contour, whose level set xi^T sigma^-1 xi = c^2
                         is likewise built on the inverse, and with the paper's geometry
                         section, which works throughout with
                         sigma^{off -1} = sigma_R^-1 + i sigma_I^-1 (after Eq. wigner_jk).
      False           -- decompose sigma_off itself, i.e. the entries as they appear in
                         Eq.(cov_rotation).

    Returns None (after printing) if sigma_off is not diagonal to `tol`, measured as
    (|s01| + |s10|)/||sigma_off||_F.  Symmetry (s01 == s10) is checked separately: it is a
    different property from diagonality and a different bug if it fails.

    Normalisability of Eq.(wigner_jk) forces Re(sigma_off^-1) > 0.  With use_inverse=True
    that is directly cos(psi_j) > 0; either way psi_j lies in (-pi/2, pi/2), where the tilt
    of Draw_Off_Diagonal_Tilt_3D is injective, so no orientation marker or mod-pi
    disambiguation is needed.
    """
    sigma_off = np.asarray(sigma_off, dtype=complex)
    fro = np.linalg.norm(sigma_off, 'fro')

    # Symmetry is checked BEFORE the diagonality gate, and against a numerical-equality
    # tolerance rather than `tol`.  Ordered the other way it can never fire: passing the
    # gate means |s01| + |s10| <= tol*fro, which by the triangle inequality forces
    # |s01 - s10| <= tol*fro, so the condition is unsatisfiable exactly when it is reached.
    # It is a different property from diagonality -- a Hermitian block passed in place of a
    # complex-symmetric one has s10 = conj(s01) -- and a different bug.
    if abs(sigma_off[0, 1] - sigma_off[1, 0]) > 1e-12*max(fro, 1.0):
        warnings.warn("sigma_off is not symmetric (s01 != s10); it should be complex "
                      "symmetric, not Hermitian", RuntimeWarning, stacklevel=2)

    ratio = (abs(sigma_off[0, 1]) + abs(sigma_off[1, 0]))/fro if fro > 0 else 0.0
    if ratio > tol:
        print(f"sigma_off is not diagonal and cannot be computed (ratio = {ratio:.3e})")
        return None

    diag = np.array([sigma_off[0, 0], sigma_off[1, 1]])
    if use_inverse:
        if np.min(np.abs(diag)) == 0:
            print("sigma_off has a zero diagonal entry and cannot be inverted")
            return None
        diag = 1.0/diag                     # diagonal, so this is the matrix inverse
    zeta = np.abs(diag)
    psi = np.arctan2(diag.imag, diag.real)
    for j, p in enumerate(psi):
        if np.cos(p) <= 0:
            warnings.warn(f"psi_{'xp'[j]} = {np.degrees(p):.2f} deg has cos <= 0; "
                          f"Eq.(wigner_jk) is not normalisable there", RuntimeWarning,
                          stacklevel=2)
    return dict(zeta_x=float(zeta[0]), zeta_p=float(zeta[1]),
                psi_x=float(psi[0]), psi_p=float(psi[1]), ratio=float(ratio),
                use_inverse=bool(use_inverse), b_x=complex(diag[0]), b_p=complex(diag[1]))


def Draw_Off_Diagonal_Tilt_3D(ax, sigma_off, color='tomato', tol=1e-10,
                              n_pts=400, label=None, use_inverse=True,
                              alpha_fill=0.22, n_meridians=6, n_parallels=2,
                              show_plane_intersection=True, rim_lw=1.4):
    """Draw the diagonal off-diagonal block as two tilted semi-axis vectors in (x, p, z).

    The decomposed object is sigma_off^-1 by default (see Off_Diagonal_Polar); writing its
    diagonal entries as b_j = zeta_j e^{i psi_j}, the semi-axis vectors are

        u = zeta_x (cos psi_x, 0, sin psi_x)      # x semi-axis, lifted by psi_x
        v = zeta_p (0, cos psi_p, sin psi_p)      # p semi-axis, lifted by psi_p

    and the drawn curve is {u cos t + v sin t}.  The radial coordinate is LINEAR in the
    matrix entry -- zeta_j, not sqrt(zeta_j) -- which is what makes the picture exact:
    the shadow on the xp-plane has semi-axes zeta_j cos psi_j = Re(b_j) and the arrow tips
    sit at height zeta_j sin psi_j = Im(b_j).  Using sqrt(zeta_j) would break the first
    identity, since sqrt(zeta) cos(psi) != sqrt(zeta cos psi).

    u . v = zeta_x zeta_p sin(psi_x) sin(psi_p) vanishes only if a phase is zero, so with
    both non-zero the lift is a SHEAR, not a rotation: the ellipse's apparent proportions
    are an artefact and it is drawn only as a guide to the eye.  The arrows carry the
    numbers.  The disc is filled and ruled with meridians and parallels, in the style of
    Draw_Tilted_Pancake_3D, so that its orientation in space is readable.

    No xp-plane shadow is drawn.  The vertical projection of the curve does have semi-axes
    exactly Re(b_j) -- that identity still holds and is worth checking numerically -- but on
    screen the eye compares it against the camera silhouette, which is a different
    projection, so drawing it without projection lines only misleads.

    `show_plane_intersection` draws instead the disc's INTERSECTION with the xp-plane, as a
    dashed chord in the rim's own colour and width.  Unlike a projection this is exact and
    unambiguous: it is the set of disc points that genuinely have z = 0.  Along the rim
    z(t) = Im(b_x) cos t + Im(b_p) sin t, which vanishes at t0 = atan2(-Im b_x, Im b_p) and
    at t0 + pi; the disc is centred on the origin and the plane passes through that centre,
    so the intersection is the full diameter between those two antipodal rim points.  When
    both phases vanish the disc already lies in the plane and nothing is drawn.

    Dimensional honesty: with one vertical axis, non-zero psi_x and psi_p means a genuinely
    4-D object (Re x, Im x, Re p, Im p) is being projected into 3-D.

    Returns
    -------
    dict from Off_Diagonal_Polar, plus u, v -- or None if sigma_off is not diagonal.
    """
    pol = Off_Diagonal_Polar(sigma_off, tol=tol, use_inverse=use_inverse)
    if pol is None:
        return None

    u = pol['zeta_x']*np.array([np.cos(pol['psi_x']), 0.0, np.sin(pol['psi_x'])])
    v = pol['zeta_p']*np.array([0.0, np.cos(pol['psi_p']), np.sin(pol['psi_p'])])
    t = np.linspace(0, 2*np.pi, n_pts)
    curve = np.outer(np.cos(t), u) + np.outer(np.sin(t), v)

    # filled interior of the disc
    ax.add_collection3d(Poly3DCollection([curve], alpha=alpha_fill, facecolor=color,
                                         edgecolor='none', zorder=4))

    # meridians and parallels, as in Draw_Tilted_Pancake_3D, so the tilt is readable
    grid_alpha, grid_lw = 0.35, 0.7
    for k in range(n_meridians):
        a_m = k*np.pi/n_meridians
        p1, p2 = u*np.cos(a_m) + v*np.sin(a_m), -(u*np.cos(a_m) + v*np.sin(a_m))
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                color=color, linewidth=grid_lw, alpha=grid_alpha, zorder=5)
    for f in np.linspace(1.0/(n_parallels + 1), n_parallels/(n_parallels + 1), n_parallels):
        ring = f*curve
        ax.plot(ring[:, 0], ring[:, 1], ring[:, 2],
                color=color, linewidth=grid_lw, alpha=grid_alpha, zorder=5)

    # the rim: a guide to the eye only (see the shear note above)
    ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], color=color, lw=rim_lw, alpha=0.9,
            zorder=6, label=label if label is not None else '_nolegend_')

    # where the tilted disc actually crosses the xp-plane
    if show_plane_intersection:
        u_z, v_z = u[2], v[2]                       # = Im(b_x), Im(b_p)
        if max(abs(u_z), abs(v_z)) > tol:
            t0 = np.arctan2(-u_z, v_z)
            w = u*np.cos(t0) + v*np.sin(t0)
            ax.plot([-w[0], w[0]], [-w[1], w[1]], [-w[2], w[2]],
                    color=color, lw=rim_lw, ls='--', alpha=0.9, zorder=7)
            pol['plane_intersection'] = np.array([-w, w])

    for vec in (u, v):
        _draw_vector_3d(ax, vec, color, lw=2.0)

    pol['u'], pol['v'] = u, v
    return pol


def Plot_Off_Diagonal_3D(sigma_off, sym_lim=None, color=None, title="",
                         tol=1e-10, annotate=True, use_inverse=True,
                         alpha_fill=0.22, n_meridians=6, n_parallels=2,
                         show_plane_intersection=True,
                         figsize=(10, 10), elev=22, azim=-55, save=False):
    """Figure 2: the off-diagonal block in 3-D, in the style of Plot_3D_Diagonal_Sigma.

    Scene, axis decoration, plane shading and arrow style are those of
    Plot_3D_Diagonal_Sigma / _make_scene_3d / _setup_axes_decoration_3d.  The vertical
    axis carries the imaginary part of both semi-axes at once -- see the docstring of
    Draw_Off_Diagonal_Tilt_3D for what that projection costs.

    Returns
    -------
    fig, ax, info  (info is None if sigma_off is not diagonal to `tol`)
    """
    rcParams['mathtext.fontset'] = 'cm'
    rcParams['font.family'] = 'STIXGeneral'
    if color is None:
        color = _DEFAULT_COLOR_MAP_3D['off']

    pol = Off_Diagonal_Polar(sigma_off, tol=tol, use_inverse=use_inverse)
    if pol is None:
        return None, None, None

    if sym_lim is None:
        m = 1.45*max(pol['zeta_x'], pol['zeta_p'])
        sym_lim = (-m, m)

    fig, ax = _make_scene_3d(sym_lim, figsize=figsize, elev=elev, azim=azim)
    Draw_XY_Plane(ax, sym_lim)
    lbl = (r'$(\sigma^{\mathrm{off}})^{-1}$' if use_inverse
           else r'$\sigma^{\mathrm{off}}$')
    info = Draw_Off_Diagonal_Tilt_3D(ax, sigma_off, color=color, tol=tol,
                                     use_inverse=use_inverse, label=lbl,
                                     alpha_fill=alpha_fill, n_meridians=n_meridians,
                                     n_parallels=n_parallels,
                                     show_plane_intersection=show_plane_intersection)
    _setup_axes_decoration_3d(ax, sym_lim,
                              xlabel=r"$\mathrm{Re}[x]$", ylabel=r"$\mathrm{Re}[p]$",
                              zlabel=(r"$\mathrm{Im}[(\sigma^{\mathrm{off}})^{-1}]$"
                                      if use_inverse else r"$\mathrm{Im}[\sigma^{\mathrm{off}}]$"))
    if title:
        ax.set_title(title, fontsize=15, y=0.85)
    if annotate:
        ax.text2D(0.02, 0.02,
                  rf"$\zeta_x={info['zeta_x']:.4f}$,  $\zeta_p={info['zeta_p']:.4f}$,  "
                  rf"$\psi_x={np.degrees(info['psi_x']):.2f}^\circ$,  "
                  rf"$\psi_p={np.degrees(info['psi_p']):.2f}^\circ$",
                  transform=ax.transAxes, fontsize=11)
    if save:
        plt.savefig(save, bbox_inches='tight')
    return fig, ax, info


def Draw_Covariance_Contour_3D(ax, sigma, centre=(0.0, 0.0), z=0.0, c=1.0,
                               radial='sqrt', color='cornflowerblue', label=None,
                               lw=1.6, alpha=1.0, fill_alpha=0.16, n_pts=240):
    """Draw an on-diagonal covariance contour into a 3-D scene, in the plane Im = `z`.

    The 3-D companion of Draw_Covariance_Contour, sharing its geometry through
    Covariance_Geometry, so the two views of the same sigma agree by construction.

    Only the ON-DIAGONAL blocks belong here: sigma_JJ is real symmetric, so its contour is
    a genuine ellipse in the real phase plane and is drawn at z = 0 -- exactly where
    Draw_First_Moments_3D puts the matching on-diagonal arrow.  The off-diagonal block is
    complex and not Hermitian; its geometry is a tilted disc and is handled separately by
    Draw_Off_Diagonal_Tilt_3D.

    Parameters
    ----------
    ax          : 3-D Axes
    sigma       : (2, 2) real symmetric positive-definite
    centre      : (x, p) centre -- pass the branch first moment to sit the contour on the
                  tip of its arrow, or (0, 0) to centre it at the origin
    z           : height of the plane holding the contour
    c, radial   : see Covariance_Geometry
    fill_alpha  : opacity of the filled interior (0 to leave it hollow)

    Returns
    -------
    dict -- the Covariance_Geometry metadata
    """
    g = Covariance_Geometry(sigma, c=c, radial=radial)
    t = np.linspace(0, 2*np.pi, n_pts)
    ct, st = np.cos(g['theta']), np.sin(g['theta'])
    xs = centre[0] + g['semi_major']*np.cos(t)*ct - g['semi_minor']*np.sin(t)*st
    ys = centre[1] + g['semi_major']*np.cos(t)*st + g['semi_minor']*np.sin(t)*ct
    zs = np.full_like(xs, float(z))

    if fill_alpha > 0:
        ax.add_collection3d(Poly3DCollection(
            [list(zip(xs, ys, zs))], alpha=fill_alpha, facecolor=color,
            edgecolor='none', zorder=3))
    ax.plot(xs, ys, zs, color=color, lw=lw, alpha=alpha, zorder=6,
            label=label if label is not None else '_nolegend_')
    return g


def Draw_First_Moments_3D(ax, r_on_plus=None, r_on_minus=None, r_off=None, z_comp=0,
                          color_map=None, show_projections=True, lw=2.0,
                          sigma_on_plus=None, sigma_on_minus=None, c=1.0,
                          radial='sqrt', contour_fill_alpha=0.16):
    """Draw the first moments in (x, p, z), with z carrying one imaginary component.

    The on-diagonal moments are real, so each is one in-plane arrow to the phase-space
    point (Re r_x, Re r_p, 0).  Nothing is lost: that is exactly where the branch Gaussian
    sits.

    The off-diagonal moment is complex and is drawn as a SINGLE arrow to

        (Re r^off_x,  Re r^off_p,  Im r^off_[z_comp])

    the same convention as Draw_Vectors.  Its in-plane part is the paper's envelope centre
    r_R (Sec. 6, after Eq. wigner_jk).  One arrow can only carry one imaginary component,
    so the other is not shown -- `z_comp` selects which, exactly as elsewhere in this
    library, and Plot_First_Moments_3D reports the omitted one in its annotation.

    `show_projections` draws, for the off-diagonal arrow only: the drop from the tip to the
    xp-plane and the in-plane projection vector, and the line from the tip across to
    Im r^off_[z_comp] on the z axis.

    `sigma_on_plus` / `sigma_on_minus` optionally add the ON-DIAGONAL covariance contour of
    each branch, centred on the tip of that branch's arrow and lying in the z = 0 plane --
    the Gaussian each moment is the centre of.  Only the on-diagonal blocks are accepted:
    they are the real symmetric ones, so the contour is a true ellipse in the real phase
    plane.  Pass them in the SAME frame as the moments; if the moments were rotated into a
    driving frame by R, the covariances must be rotated as R sigma R^T.

    Returns
    -------
    dict of the drawn tip vectors, plus 'geom+'/'geom-' when contours were drawn
    """
    if color_map is None:
        color_map = _DEFAULT_COLOR_MAP_3D
    out = {}

    for key, vec, sig, role, lbl in (
            ('+', r_on_plus,  sigma_on_plus,  'real+', r"$\bar{r}^{\mathrm{on}}_{+}$"),
            ('-', r_on_minus, sigma_on_minus, 'real-', r"$\bar{r}^{\mathrm{on}}_{-}$")):
        if vec is None:
            continue
        v = np.asarray(vec, dtype=complex).ravel()
        tip = np.array([v[0].real, v[1].real, 0.0])
        ax.plot([], [], [], color=color_map[role], lw=lw, label=lbl)
        _draw_vector_3d(ax, tip, color_map[role], lw=lw)
        out[key] = tip
        if sig is not None:
            # centred on the arrow tip: the branch Gaussian sits at its own first moment
            out['geom' + key] = Draw_Covariance_Contour_3D(
                ax, sig, centre=tip[:2], z=0.0, c=c, radial=radial,
                color=color_map[role], lw=1.4, alpha=0.95,
                fill_alpha=contour_fill_alpha,
                label=rf"$\sigma^{{\mathrm{{on}}}}_{{{key}}}$")

    if r_off is not None:
        v = np.asarray(r_off, dtype=complex).ravel()
        col = color_map['off']
        tip = np.array([v[0].real, v[1].real, v[z_comp].imag])
        ax.plot([], [], [], color=col, lw=lw, label=r"$\bar{r}^{\mathrm{off}}$")
        _draw_vector_3d(ax, tip, col, lw=lw)
        out['off'] = tip

        if show_projections:
            foot = np.array([tip[0], tip[1], 0.0])          # xp-plane projection
            zpt = np.array([0.0, 0.0, tip[2]])              # imaginary part on the z axis
            for a, b in ((tip, foot), (tip, zpt)):
                ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                        color=col, lw=0.9, ls=':', alpha=0.75, zorder=8)
            ax.plot([0.0, foot[0]], [0.0, foot[1]], [0.0, 0.0],
                    color=col, lw=1.1, alpha=0.55, zorder=8)
            ax.plot([0.0, 0.0], [0.0, 0.0], [0.0, zpt[2]],
                    color=col, lw=1.1, alpha=0.55, zorder=8)
            out['proj_plane'], out['proj_z'] = foot, zpt
    return out


def Draw_First_Moment_Curves_3D(ax, r_on_plus_t=None, r_on_minus_t=None, r_off_t=None,
                                z_comp=0, color_map=None, lw=1.4, alpha=0.6,
                                arrowheads=True, mark_end=True):
    """Draw first-moment trajectories in (x, p, z), ending at the last sample.

    Same conventions as Draw_First_Moments_3D -- on-diagonal moments are real and lie in
    the plane, the off-diagonal one is a single curve whose height is Im r^off_[z_comp] --
    and the same directional-arrowhead style as Draw_Curves.

    Each argument is a (T, 2) complex time series.  The final point is where the arrows of
    Draw_First_Moments_3D sit, so a steady-state run ends exactly on them.

    Returns
    -------
    dict of the end points actually drawn
    """
    if color_map is None:
        color_map = _DEFAULT_COLOR_MAP_3D
    out = {}

    series = (('+', r_on_plus_t,  'real+', r"$\bar{r}^{\mathrm{on}}_{+}(\tau)$"),
              ('-', r_on_minus_t, 'real-', r"$\bar{r}^{\mathrm{on}}_{-}(\tau)$"),
              ('off', r_off_t,    'off',   r"$\bar{r}^{\mathrm{off}}(\tau)$"))
    for key, arr, role, lbl in series:
        if arr is None:
            continue
        a = np.asarray(arr, dtype=complex)
        rx, rp = a[:, 0].real, a[:, 1].real
        z = a[:, z_comp].imag if key == 'off' else np.zeros(len(a))
        col = color_map[role]
        ax.plot(rx, rp, z, color=col, linewidth=lw, alpha=alpha, label=lbl, zorder=6)

        if arrowheads and len(a) > 3:
            for frac in (1/3, 2/3):
                i = max(1, min(int(round(frac*(len(a) - 1))), len(a) - 2))
                d = np.array([rx[i] - rx[i-1], rp[i] - rp[i-1], z[i] - z[i-1]])
                n = np.linalg.norm(d)
                if n < 1e-12:
                    continue
                _triangle_arrowhead_3d(ax, np.array([rx[i], rp[i], z[i]]), d/n,
                                       head_len=0.3, head_width=0.3, color=col)
        end = np.array([rx[-1], rp[-1], z[-1]])
        if mark_end:
            ax.scatter([end[0]], [end[1]], [end[2]], color=col, s=18, marker='o',
                       depthshade=False, zorder=11)
        out[key] = end
    return out


def Plot_First_Moments_3D(r_on_plus=None, r_on_minus=None, r_off=None, z_comp=0,
                          sym_lim=None, color_map=None, title="", show_projections=True,
                          annotate=True, trajectories=None, reference=None,
                          reference_label=None, sigma_on_plus=None, sigma_on_minus=None,
                          c=1.0, radial='sqrt', contour_fill_alpha=0.16,
                          figsize=(10, 10), elev=22, azim=-55, save=False):
    """Figure: the first moments in 3-D, in the style of Plot_Off_Diagonal_3D.

    z_comp selects which imaginary component the vertical axis carries, 0 -> Im[x],
    1 -> Im[p], as in Draw_Vectors and _finalise_3d.  See Draw_First_Moments_3D.

    `trajectories` is an optional (r_on_plus_t, r_on_minus_t, r_off_t) of (T, 2) time
    series, drawn by Draw_First_Moment_Curves_3D; their last sample coincides with the
    arrows, so a run to the steady state ends exactly on them.

    `reference` is an optional in-plane 2-vector drawn as a thin dashed arrow -- e.g. the
    driving d, when the moments are expressed in a frame referenced to it.

    `sigma_on_plus` / `sigma_on_minus` add each branch's ON-DIAGONAL covariance contour,
    centred on the tip of its arrow in the z = 0 plane.  They must be given in the same
    frame as the moments -- in a driving frame that means R sigma R^T, not sigma.  See
    Draw_First_Moments_3D.
    """
    rcParams['mathtext.fontset'] = 'cm'
    rcParams['font.family'] = 'STIXGeneral'

    vecs = [np.asarray(v, dtype=complex).ravel()
            for v in (r_on_plus, r_on_minus, r_off) if v is not None]
    if trajectories is not None:
        vecs += [np.asarray(t, dtype=complex).ravel() for t in trajectories if t is not None]
    if reference is not None:
        vecs.append(np.asarray(reference, dtype=complex).ravel())
    if sym_lim is None:
        m = 1.45*max(float(np.max(np.abs(np.concatenate([v.real, v.imag])))) for v in vecs)
        # a contour drawn on an arrow tip reaches beyond it by its own semi-major axis
        for sig, ctr in ((sigma_on_plus, r_on_plus), (sigma_on_minus, r_on_minus)):
            if sig is None:
                continue
            g = Covariance_Geometry(sig, c=c, radial=radial)
            reach = float(np.max(np.abs(np.asarray(ctr, dtype=complex).ravel().real))) \
                if ctr is not None else 0.0
            m = max(m, 1.15*(reach + g['semi_major']))
        sym_lim = (-m, m)

    fig, ax = _make_scene_3d(sym_lim, z_comp=z_comp, figsize=figsize, elev=elev, azim=azim)
    Draw_XY_Plane(ax, sym_lim)
    if trajectories is not None:
        Draw_First_Moment_Curves_3D(ax, *trajectories, z_comp=z_comp, color_map=color_map)
    if reference is not None:
        ref = np.asarray(reference, dtype=float).ravel()
        ax.plot([0.0, ref[0]], [0.0, ref[1]], [0.0, 0.0], color='0.35', lw=1.4, ls='--',
                alpha=0.9, zorder=9,
                label=reference_label if reference_label is not None else '_nolegend_')
    info = Draw_First_Moments_3D(ax, r_on_plus, r_on_minus, r_off, z_comp=z_comp,
                                 color_map=color_map, show_projections=show_projections,
                                 sigma_on_plus=sigma_on_plus,
                                 sigma_on_minus=sigma_on_minus,
                                 c=c, radial=radial,
                                 contour_fill_alpha=contour_fill_alpha)
    _setup_axes_decoration_3d(ax, sym_lim,
                              xlabel=r"$\mathrm{Re}[x]$", ylabel=r"$\mathrm{Re}[p]$",
                              zlabel=(r"$\mathrm{Im}[x]$" if z_comp == 0
                                      else r"$\mathrm{Im}[p]$"))
    if title:
        ax.set_title(title, fontsize=15, y=0.85)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        # lower than _finalise_3d's default: with trajectories this legend carries up to
        # eight entries, which at y = 0.3 sits on top of the scene
        fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=12,
                   framealpha=0.7, bbox_to_anchor=(0.5, 0.12 if trajectories else 0.3))
    if annotate and r_off is not None:
        v = np.asarray(r_off, dtype=complex).ravel()
        other = 1 - z_comp
        ax.text2D(0.02, 0.02,
                  rf"$\bar{{r}}^{{\mathrm{{off}}}}=({v[0].real:.4f}{v[0].imag:+.4f}i,\ "
                  rf"{v[1].real:.4f}{v[1].imag:+.4f}i)$;  "
                  rf"not shown: $\mathrm{{Im}}[{'xp'[other]}]={v[other].imag:.4f}$",
                  transform=ax.transAxes, fontsize=11)
    if save:
        plt.savefig(save, bbox_inches='tight')
    return fig, ax, info


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def Sym_Lim_From_Data(*arrays, pad=1.08):
    """Compute a symmetric integer limit (-L, L) that covers all values.

    Example
    -------
    sl = Sym_Lim_From_Data(GCS.r_JK_t)
    sl = Sym_Lim_From_Data(GCS.r_JK_t, GCS.sigma_JK_t)
    """
    all_vals = [0.0]
    for a in arrays:
        if np.isscalar(a):
            all_vals.append(float(a))
        else:
            all_vals.extend(np.asarray(a).ravel().real)
            all_vals.extend(np.asarray(a).ravel().imag)
    L = int(np.ceil(max(abs(np.min(all_vals)), abs(np.max(all_vals))) * pad))
    return (-L, L)


def Make_Legend_Handle_3D(label, color, lw=2.0):
    """Build one Line2D legend entry for a 3-D phase-space plot.

    Example
    -------
    handles = [Make_Legend_Handle_3D(r'$r^{on}_{+}$', 'royalblue'),
               Make_Legend_Handle_3D(r'$r^{off}$',    'tomato')]
    """
    return mlines.Line2D([], [], color=color, linewidth=lw, label=label)


# ─────────────────────────────────────────────────────────────────────────────
#  CONVENIENCE WRAPPERS
# ─────────────────────────────────────────────────────────────────────────────

def Plot_Trajectories_Single(r_JK_t, z_comp=0,
                             title="Dynamics of the first moments",
                             legend_handles=None,
                             color_map=None,
                             sym_lim=None,
                             figsize=(10, 10), elev=22, azim=-55,
                             save=False):
    """One-call wrapper: trajectories in a single 3-D panel.

    Parameters
    ----------
    r_JK_t         : ndarray (T, 2, 2, 2, 1) complex — first moments
    z_comp         : 0 -> Im[x],  1 -> Im[p] on z-axis
    title          : figure title string
    legend_handles : explicit legend entries (None = auto, [] = none)
    color_map      : dict role->colour (None = defaults)
    sym_lim        : (lo, hi) — computed from r_JK_t if None
    """
    rcParams['mathtext.fontset'] = 'cm'
    rcParams['font.family'] = 'STIXGeneral' 

    if sym_lim is None:
        sym_lim = Sym_Lim_From_Data(r_JK_t)

    fig, ax = _make_scene_3d(sym_lim, z_comp=z_comp,
                             figsize=figsize, elev=elev, azim=azim)
    Draw_XY_Plane(ax, sym_lim)
    Draw_Curves(ax, r_JK_t, z_comp, color_map=color_map)
    _finalise_3d(ax, fig, sym_lim, z_comp,
                 title=title, legend_handles=legend_handles)
    if save:
        plt.savefig(save, bbox_inches='tight')
    return fig, ax


def Plot_Vectors_Single(r_JK_t, t_idx, z_comp=0,
                        title="First moments",
                        legend_handles=None,
                        color_map=None,
                        sym_lim=None,
                        figsize=(10, 10), elev=22, azim=-55,
                        save=False):
    """One-call wrapper: snapshot vectors in a single 3-D panel.

    Parameters
    ----------
    r_JK_t         : ndarray (T, 2, 2, 2, 1) complex — first moments
    t_idx          : int — time index to snapshot
    z_comp         : 0 -> Im[x],  1 -> Im[p] on z-axis
    title          : figure title string
    legend_handles : explicit legend entries (None = auto, [] = none)
    color_map      : dict role->colour (None = defaults)
    sym_lim        : (lo, hi) — computed from r_JK_t if None
    """
    rcParams['mathtext.fontset'] = 'cm'
    rcParams['font.family'] = 'STIXGeneral'
    
    if sym_lim is None:
        sym_lim = Sym_Lim_From_Data(r_JK_t)

    fig, ax = _make_scene_3d(sym_lim, z_comp=z_comp,
                             figsize=figsize, elev=elev, azim=azim)
    Draw_XY_Plane(ax, sym_lim)
    Draw_Vectors(ax, r_JK_t, t_idx, z_comp, color_map=color_map)
    _finalise_3d(ax, fig, sym_lim, z_comp,
                 title=title, legend_handles=legend_handles)
    if save:
        plt.savefig(save, bbox_inches='tight')
    return fig, ax


def Plot_Vectors_Ellipsoids_Single(r_JK_t, sigma_JK_t, t_idx, z_comp=0,
                                   title="First moments with uncertainty",
                                   legend_handles=None,
                                   color_map=None,
                                   sym_lim=None,
                                   figsize=(10, 10), elev=22, azim=-55,
                                   save=False):
    """One-call wrapper: vectors + uncertainty ellipsoids in a single 3-D panel.

    Parameters
    ----------
    r_JK_t       : ndarray (T, 2, 2, 2, 1) complex — first moments
    sigma_JK_t   : ndarray (T, 2, 2, 2, 2) complex — covariance matrices
    t_idx        : int — time index to snapshot
    z_comp       : 0 -> Im[x],  1 -> Im[p] on z-axis
    title        : figure title string
    legend_handles : explicit legend entries (None = auto, [] = none)
    color_map    : dict role->colour (None = defaults)
    sym_lim      : (lo, hi) — computed from r_JK_t if None
    """
    rcParams['mathtext.fontset'] = 'cm'
    rcParams['font.family'] = 'STIXGeneral'

    if sym_lim is None:
        sym_lim = Sym_Lim_From_Data(r_JK_t)

    fig, ax = _make_scene_3d(sym_lim, z_comp=z_comp,
                             figsize=figsize, elev=elev, azim=azim)
    Draw_XY_Plane(ax, sym_lim)
    Draw_Vectors(ax, r_JK_t, t_idx, z_comp, color_map=color_map)
    Draw_Ellipsoids(ax, r_JK_t, sigma_JK_t, t_idx, z_comp, color_map=color_map)
    _finalise_3d(ax, fig, sym_lim, z_comp,
                 title=title, legend_handles=legend_handles)
    if save:
        plt.savefig(save, bbox_inches='tight')
    return fig, ax


def Plot_3D_Diagonal_Sigma(r_JK_t, sigma_JK_t, t_idx, z_comp=0,
                           title="3-D phase space",
                           legend_handles=None, color_map=None,
                           sym_lim=None, pancake_color='tomato',
                           figsize=(10, 10), elev=22, azim=-55, save=False):
    """3-D phase-space snapshot: on-diagonal ellipsoids + off-diagonal tilted pancake.

    Draws first-moment vectors and uncertainty ellipsoids for the on-diagonal
    branches (j=k), then draws σ^off as a flat pancake whose semi-axes are tilted
    by the polar phases of its diagonal entries.

    sigma_JK_t[t_idx, 0, 1] must be a 2×2 complex diagonal matrix.

    Parameters
    ----------
    r_JK_t       : ndarray (T, 2, 2, 2, 1) complex — first moments
    sigma_JK_t   : ndarray (T, 2, 2, 2, 2) complex — covariance matrices
    t_idx        : int — time index to snapshot
    z_comp       : 0 → Im[x] on z-axis,  1 → Im[p] on z-axis
    title        : figure title string
    legend_handles : explicit legend entries (None = auto, [] = none)
    color_map    : dict role→colour (None = defaults)
    sym_lim      : (lo, hi) — computed from data if None
    pancake_color : colour for the off-diagonal pancake
    figsize, elev, azim : figure / view-angle controls
    save         : path to save as PNG/SVG, or False

    Returns
    -------
    fig, ax : Figure and Axes3D
    """
    rcParams['mathtext.fontset'] = 'cm'
    rcParams['font.family'] = 'STIXGeneral'

    if sym_lim is None:
        sym_lim = Sym_Lim_From_Data(r_JK_t[[t_idx]], sigma_JK_t[[t_idx]])

    fig, ax = _make_scene_3d(sym_lim, z_comp=z_comp, figsize=figsize, elev=elev, azim=azim)
    Draw_XY_Plane(ax, sym_lim)
    Draw_Vectors(ax, r_JK_t, t_idx, z_comp, color_map=color_map)
    Draw_Ellipsoids(ax, r_JK_t, sigma_JK_t, t_idx, z_comp, color_map=color_map)

    sigma_off = sigma_JK_t[t_idx, 0, 1]
    r_off     = r_JK_t[t_idx, 0, 1]
    cx = float(r_off[0, 0].real)
    cy = float(r_off[1, 0].real)
    cz = float(r_off[z_comp, 0].imag)
    Draw_Tilted_Pancake_3D(ax, [cx, cy, cz], sigma_off, z_comp=z_comp,
                            color=pancake_color,
                            label=r'$\sigma^{\mathrm{off}}$')

    _finalise_3d(ax, fig, sym_lim, z_comp,
                 title=title, legend_handles=legend_handles)
    if save:
        plt.savefig(save, bbox_inches='tight')
    return fig, ax


def Plot_Vectors_Ellipsoids_Multi_Time(r_JK_t, sigma_JK_t, t_idx_array, alpha_array,
                                       z_comp=0,
                                       title="First moments with uncertainty",
                                       legend_handles=None,
                                       color_map=None,
                                       sym_lim=None,
                                       figsize=(10, 10), elev=22, azim=-55,
                                       save=False):
    """Static 3-D plot: vectors + ellipsoids at multiple time snapshots.

    Colors are fixed per branch; opacity encodes time via alpha_array, letting
    the reader perceive dynamics without animation.  Legend entries are added
    once (from the first snapshot) so they appear fully opaque.

    Parameters
    ----------
    r_JK_t       : ndarray (T, 2, 2, 2, 1) complex — first moments
    sigma_JK_t   : ndarray (T, 2, 2, 2, 2) complex — covariance matrices
    t_idx_array  : sequence of int — time indices to render (e.g. [0, 10, 20, -1])
    alpha_array  : sequence of float — opacity for each time in t_idx_array
                   (e.g. [0.15, 0.35, 0.6, 1.0] — earlier = more transparent)
    z_comp       : 0 -> Im[x] on z-axis,  1 -> Im[p] on z-axis
    title        : figure title string
    legend_handles : explicit legend entries (None = auto from first snapshot)
    color_map    : dict role->colour (None = defaults)
    sym_lim      : (lo, hi) — computed from r_JK_t and sigma_JK_t if None

    Example
    -------
    t_indices = [0, 25, 50, 75, -1]
    alphas    = [0.10, 0.25, 0.45, 0.70, 1.0]
    fig, ax = Plot_Vectors_Ellipsoids_Multi_Time(r_m, s_m, t_indices, alphas)
    """
    rcParams['mathtext.fontset'] = 'cm'
    rcParams['font.family'] = 'STIXGeneral'

    if sym_lim is None:
        sym_lim = Sym_Lim_From_Data(r_JK_t, sigma_JK_t)

    fig, ax = _make_scene_3d(sym_lim, z_comp=z_comp,
                             figsize=figsize, elev=elev, azim=azim)
    Draw_XY_Plane(ax, sym_lim)

    for i, (t_idx, alpha) in enumerate(zip(t_idx_array, alpha_array)):
        add_entry = (i == 0)   # legend entries added only on first pass
        Draw_Vectors(ax, r_JK_t, t_idx, z_comp,
                     color_map=color_map, alpha=alpha, add_legend_entry=add_entry)
        Draw_Ellipsoids(ax, r_JK_t, sigma_JK_t, t_idx, z_comp,
                        color_map=color_map, alpha=alpha)

    _finalise_3d(ax, fig, sym_lim, z_comp,
                 title=title, legend_handles=legend_handles)
    if save:
        plt.savefig(save, bbox_inches='tight')
    return fig, ax


def Plot_Diagonal_Multi_Time_2D(r_JK_t, sigma_JK_t, t_idx_array, alpha_array,
                                 color_map=None, sym_lim=None,
                                 title="On-diagonal first moments",
                                 figsize=(6, 6), save=False):
    """2-D phase-space plot of on-diagonal first moments at multiple times.

    Draws one arrow (origin → (Re[x], Re[p])) and one uncertainty ellipse
    per on-diagonal branch (j=k) at each time in t_idx_array.  Color is
    fixed per branch; alpha encodes time so the reader perceives dynamics
    without animation.  The ellipse orientation comes from the full 2×2
    covariance eigendecomposition.

    Parameters
    ----------
    r_JK_t      : ndarray (T, N2, N2, 2, 1) — first moments (after Select_Mode)
    sigma_JK_t  : ndarray (T, N2, N2, 2, 2) — covariance matrices
    t_idx_array : sequence of int — time indices to render
    alpha_array : sequence of float — opacity per time step
                  (e.g. [0.15, 0.40, 0.70, 1.0] — earlier = more transparent)
    color_map   : dict role->colour (None = defaults; used only for N2==2)
    sym_lim     : (lo, hi) or float — axis limits; auto-computed if None
    title       : axes title string
    figsize     : figure size tuple

    Example
    -------
    t_indices = [0, 25, 50, -1]
    alphas    = [0.15, 0.40, 0.70, 1.0]
    fig, ax = Plot_Diagonal_Multi_Time_2D(r_m, s_m, t_indices, alphas)
    """
    rcParams['mathtext.fontset'] = 'cm'
    rcParams['font.family'] = 'STIXGeneral'

    if color_map is None:
        color_map = _DEFAULT_COLOR_MAP_3D

    N2 = r_JK_t.shape[1]
    if N2 == 2:
        diag_map = [((1, 1), r"$r^{\mathrm{on}}_{+}$", color_map['real+']),
                    ((0, 0), r"$r^{\mathrm{on}}_{-}$", color_map['real-'])]
    else:
        _tab = plt.cm.get_cmap('tab10')
        diag_map = [((j, j), f'$J_{{{j}}}$', _tab(j % 10)) for j in range(N2)]

    if sym_lim is None:
        vals = []
        for (j, k), _, _ in diag_map:
            v = r_JK_t[:, j, k, :, 0]
            vals.extend(v[:, 0].real)
            vals.extend(v[:, 1].real)
        L = float(np.max(np.abs(vals))) * 1.20
        sym_lim = (-L, L)
    elif np.isscalar(sym_lim):
        sym_lim = (-sym_lim, sym_lim)

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=300)
    ax.set_aspect('equal')
    ax.axvline(0, color='black', linewidth=0.6, alpha=0.4)
    ax.axhline(0, color='black', linewidth=0.6, alpha=0.4)
    ax.set_xlabel(r'Position, $x$', fontsize=12)
    ax.set_ylabel(r'Momentum, $p$', fontsize=12)
    ax.set_xlim(sym_lim)
    ax.set_ylim(sym_lim)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    legend_added = set()

    for i, (t_idx, alpha) in enumerate(zip(t_idx_array, alpha_array)):
        for (j, k), label, color in diag_map:
            vec = r_JK_t[t_idx, j, k, :, 0]
            cx  = float(vec[0].real)
            cy  = float(vec[1].real)
            sig_real = np.real(sigma_JK_t[t_idx, j, k])

            # Arrow from origin to (cx, cy)
            ax.quiver(0., 0., cx, cy, color=color, alpha=alpha,
                      angles='xy', scale_units='xy', scale=1,
                      headwidth=4, headlength=5, headaxislength=4.5,
                      width=0.005, zorder=5)

            # 2-D uncertainty ellipse via full covariance eigendecomposition
            eig_vals, eig_vecs = np.linalg.eigh(sig_real)
            eig_vals = np.maximum(eig_vals, 0.0)
            angle = np.degrees(np.arctan2(eig_vecs[1, 0], eig_vecs[0, 0]))
            ellipse = Ellipse(xy=(cx, cy),
                              width=2.0 * np.sqrt(eig_vals[0]),
                              height=2.0 * np.sqrt(eig_vals[1]),
                              angle=angle,
                              facecolor=color, alpha=0.15 * alpha,
                              edgecolor=color, linewidth=0.8,
                              linestyle='-', zorder=4)
            ax.add_patch(ellipse)

            if label not in legend_added:
                ax.plot([], [], color=color, linewidth=1.5, label=label)
                legend_added.add(label)

    ax.legend(loc='upper right', fontsize=10, framealpha=0.7)
    if title:
        ax.set_title(title, fontsize=13)
    fig.tight_layout()
    if save:
        plt.savefig(save, bbox_inches='tight')
    return fig, ax


def Plot_Phase_Space_Summary(r_JK_t, sigma_JK_t, t_idx_array, alpha_array,
                              color_map=None,
                              sym_lim_2d=None, sym_lim_3d=None,
                              title="",
                              figsize=(18, 6), elev=22, azim=-55,
                              save=False):
    """Three-panel static phase-space figure.

    Panel 1 — 2-D Re[x]–Re[p]: on-diagonal arrows + ellipses (same as
               Plot_Diagonal_Multi_Time_2D).
    Panel 2 — 3-D with Im[x] on z-axis: all branches, vectors + ellipsoids.
    Panel 3 — 3-D with Im[p] on z-axis: same, different z component.

    Color is fixed per branch across all panels; alpha encodes time.

    Parameters
    ----------
    r_JK_t      : ndarray (T, N2, N2, 2, 1) — first moments (after Select_Mode)
    sigma_JK_t  : ndarray (T, N2, N2, 2, 2) — covariance matrices
    t_idx_array : sequence of int — time indices to render
    alpha_array : sequence of float — opacity per time step
    color_map   : dict role->colour (None = defaults)
    sym_lim_2d  : (lo, hi) or float — axis limits for panel 1 (auto if None)
    sym_lim_3d  : (lo, hi) — axis limits for panels 2–3 (auto if None)
    title       : overall figure suptitle (omitted if empty)
    figsize     : figure size tuple
    elev, azim  : 3-D view angles

    Returns
    -------
    fig, (ax1, ax2, ax3)
    """
    rcParams['mathtext.fontset'] = 'cm'
    rcParams['font.family'] = 'STIXGeneral'

    if color_map is None:
        color_map = _DEFAULT_COLOR_MAP_3D

    # ── diagonal branch map (color + label) ───────────────────────────────────
    N2 = r_JK_t.shape[1]
    if N2 == 2:
        diag_map = [((1, 1), r"$r^{\mathrm{on}}_{+}$", color_map['real+']),
                    ((0, 0), r"$r^{\mathrm{on}}_{-}$", color_map['real-'])]
    else:
        _tab = plt.cm.get_cmap('tab10')
        diag_map = [((j, j), f'$J_{{{j}}}$', _tab(j % 10)) for j in range(N2)]

    # ── auto limits ───────────────────────────────────────────────────────────
    if sym_lim_2d is None:
        vals = []
        for (j, k), _, _ in diag_map:
            v = r_JK_t[:, j, k, :, 0]
            vals.extend(v[:, 0].real)
            vals.extend(v[:, 1].real)
        L2 = float(np.max(np.abs(vals))) * 1.20
        sym_lim_2d = (-L2, L2)
    elif np.isscalar(sym_lim_2d):
        sym_lim_2d = (-sym_lim_2d, sym_lim_2d)

    if sym_lim_3d is None:
        sym_lim_3d = Sym_Lim_From_Data(r_JK_t, sigma_JK_t)

    # ── figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax2.view_init(elev=elev, azim=azim)
    ax3.view_init(elev=elev, azim=azim)

    # ── Panel 1: 2-D Re[x]–Re[p] ─────────────────────────────────────────────
    ax1.set_aspect('equal')
    ax1.axvline(0, color='black', linewidth=0.6, alpha=0.4)
    ax1.axhline(0, color='black', linewidth=0.6, alpha=0.4)
    ax1.set_xlabel(r'$\mathrm{Re}[x]$', fontsize=12)
    ax1.set_ylabel(r'$\mathrm{Re}[p]$', fontsize=12)
    ax1.set_xlim(sym_lim_2d)
    ax1.set_ylim(sym_lim_2d)
    for spine in ax1.spines.values():
        spine.set_linewidth(0.5)
    ax1.set_title(r'$\mathrm{Re}[x]$–$\mathrm{Re}[p]$', fontsize=13)

    legend_added = set()
    for t_idx, alpha in zip(t_idx_array, alpha_array):
        for (j, k), label, color in diag_map:
            vec      = r_JK_t[t_idx, j, k, :, 0]
            cx, cy   = float(vec[0].real), float(vec[1].real)
            sig_real = np.real(sigma_JK_t[t_idx, j, k])

            ax1.quiver(0., 0., cx, cy, color=color, alpha=alpha,
                       angles='xy', scale_units='xy', scale=1,
                       headwidth=4, headlength=5, headaxislength=4.5,
                       width=0.005, zorder=5)

            eig_vals, eig_vecs = np.linalg.eigh(sig_real)
            eig_vals = np.maximum(eig_vals, 0.0)
            angle    = np.degrees(np.arctan2(eig_vecs[1, 0], eig_vecs[0, 0]))
            ax1.add_patch(Ellipse(xy=(cx, cy),
                                  width=2.0 * np.sqrt(eig_vals[0]),
                                  height=2.0 * np.sqrt(eig_vals[1]),
                                  angle=angle,
                                  facecolor=color, alpha=0.15 * alpha,
                                  edgecolor=color, linewidth=0.8,
                                  linestyle='-', zorder=4))

            if label not in legend_added:
                ax1.plot([], [], color=color, linewidth=1.5, label=label)
                legend_added.add(label)

    ax1.legend(loc='upper right', fontsize=10, framealpha=0.7)

    # ── Panels 2–3: 3-D ───────────────────────────────────────────────────────
    for ax_3d, z_comp in [(ax2, 0), (ax3, 1)]:
        Draw_XY_Plane(ax_3d, sym_lim_3d)
        for i, (t_idx, alpha) in enumerate(zip(t_idx_array, alpha_array)):
            Draw_Vectors(ax_3d, r_JK_t, t_idx, z_comp,
                         color_map=color_map, alpha=alpha, add_legend_entry=False)
            Draw_Ellipsoids(ax_3d, r_JK_t, sigma_JK_t, t_idx, z_comp,
                            color_map=color_map, alpha=alpha)

        zlabel = r"$\mathrm{Im}[x]$" if z_comp == 0 else r"$\mathrm{Im}[p]$"
        _setup_axes_decoration_3d(ax_3d, sym_lim_3d,
                                   xlabel=r"$\mathrm{Re}[x]$",
                                   ylabel=r"$\mathrm{Re}[p]$",
                                   zlabel=zlabel)
        ax_3d.set_title(zlabel, fontsize=13, y=0.85)

    if title:
        fig.suptitle(title, fontsize=14)

    fig.subplots_adjust(wspace=0.05)
    if save:
        plt.savefig(save, bbox_inches='tight')
    return fig, (ax1, ax2, ax3)


def Animate_Vectors_Single(r_JK_t, z_comp=0,
                           title="First moments",
                           legend_handles=None,
                           color_map=None,
                           sym_lim=None,
                           figsize=(10, 10), elev=22, azim=-55,
                           n_frames=None, interval=50, save=False):
    if sym_lim is None:
        sym_lim = Sym_Lim_From_Data(r_JK_t)
    if n_frames is None:
        n_frames = r_JK_t.shape[0]

    fig, ax = Plot_Vectors_Single(r_JK_t, 0, z_comp=z_comp, title=title,
                                  legend_handles=legend_handles,
                                  color_map=color_map, sym_lim=sym_lim,
                                  figsize=figsize, elev=elev, azim=azim)

    def update(frame):
        ax.clear()
        Draw_XY_Plane(ax, sym_lim)
        Draw_Vectors(ax, r_JK_t, frame, z_comp, color_map=color_map)
        _finalise_3d(ax, fig, sym_lim, z_comp,
                     title=title, legend_handles=[])
        return [ax]

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=interval)
    if save:
        ani.save(filename=save, writer="pillow")
    return ani


def Animate_Vectors_Ellipsoids_Single(r_JK_t, sigma_JK_t, z_comp=0,
                                      title="First moments with uncertainty",
                                      legend_handles=None,
                                      color_map=None,
                                      sym_lim=None,
                                      figsize=(10, 10), elev=22, azim=-55,
                                      n_frames=None, interval=50, save=False):
    if sym_lim is None:
        sym_lim = Sym_Lim_From_Data(r_JK_t)
    if n_frames is None:
        n_frames = r_JK_t.shape[0]

    fig, ax = Plot_Vectors_Ellipsoids_Single(
        r_JK_t, sigma_JK_t, 0, z_comp=z_comp, title=title,
        legend_handles=legend_handles, color_map=color_map,
        sym_lim=sym_lim, figsize=figsize, elev=elev, azim=azim)

    def update(frame):
        ax.clear()
        Draw_XY_Plane(ax, sym_lim)
        Draw_Vectors(ax, r_JK_t, frame, z_comp, color_map=color_map)
        Draw_Ellipsoids(ax, r_JK_t, sigma_JK_t, frame, z_comp, color_map=color_map)
        _finalise_3d(ax, fig, sym_lim, z_comp,
                     title=title, legend_handles=[])
        return [ax]

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=interval)
    if save:
        ani.save(filename=save, writer="pillow")
    return ani


def Animate_Gaussian_Trajectory(wigner_t, x_array, p_array, bar_limits, array_tick,
                                R_t, n_frames, title, save, trail=True, trail_length=None,
                                trail_color='navy', interval=50):
    """Animate a Gaussian Wigner function together with a trailing path of its mean.

    Same figure and writer as Animate_Diagonal (first frame from Plot_Wigner_Diag);
    the only addition is a Line2D following the first moment R_t.

    Parameters
    ----------
    wigner_t     : (T, steps, steps) Wigner function per frame
    x_array, p_array, bar_limits, array_tick, title : as in Plot_Wigner_Diag
    R_t          : (T, 2) or (T, 2, 1) first moment per frame
    n_frames     : number of frames to animate
    save         : filepath (GIF) or False
    trail        : draw the path of R_t up to the current frame
    trail_length : number of past frames kept in the trail (None = whole history)

    Returns
    -------
    ani
    """
    R_t = np.real(np.asarray(R_t)).reshape(len(R_t), -1)[:, :2]
    fig, im = Plot_Wigner_Diag(wigner_t[0], x_array, p_array, bar_limits, array_tick, title, False)
    ax = im.axes

    line, = ax.plot([], [], color=trail_color, lw=0.8, alpha=0.9)
    dot, = ax.plot([], [], 'o', color=trail_color, ms=2.5)

    def update(frame):
        im.set_array(wigner_t[frame])
        if trail:
            start = 0 if trail_length is None else max(0, frame - trail_length)
            line.set_data(R_t[start:frame + 1, 0], R_t[start:frame + 1, 1])
            dot.set_data([R_t[frame, 0]], [R_t[frame, 1]])
        return [im, line, dot]

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=interval)
    if save:
        ani.save(filename=save, writer="pillow")
    return ani


def Animate_Classical_Trajectory(r_t, x_array, array_tick, n_frames, title, save,
                                 trail=True, trail_length=None, color='tab:red', interval=50):
    """Animate a classical phase-space point (x_1, p_1) with an optional trailing path.

    Same axes, limits and writer as Animate_Gaussian_Trajectory, but without the
    Wigner function (a classical sample carries no uncertainty).

    Parameters
    ----------
    r_t          : (T, 2) classical trajectory, one row per frame
    x_array      : 1-D grid; sets the symmetric axis limits (pass the quantum grid to match)
    array_tick   : tick positions
    n_frames     : number of frames to animate
    title        : axes title (False for the default)
    save         : filepath (GIF) or False
    trail        : draw the path up to the current frame
    trail_length : number of past frames kept in the trail (None = whole history)

    Returns
    -------
    ani
    """
    rcParams['mathtext.fontset'] = 'cm'
    rcParams['font.family'] = 'STIXGeneral'

    r_t = np.real(np.asarray(r_t)).reshape(len(r_t), -1)[:, :2]
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5), dpi=300)
    fig.set_tight_layout(True)

    lables_tick = format_latex_numbers(array_tick)
    lim = np.max(np.abs(x_array))
    ax.axvline(x=0, color='black', linewidth=0.5, alpha=0.4)
    ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.4)
    ax.set_ylabel(r'Momentum, $p_1$', labelpad=2, fontsize=11)
    ax.set_xlabel(r'Position, $x_1$', labelpad=2, fontsize=11)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_xticks(array_tick)
    ax.set_xticklabels(lables_tick)
    ax.set_yticks(array_tick)
    ax.set_yticklabels(lables_tick)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.set_title(title if title else r'Classical phase space $(x_1, p_1)$', fontsize=13)

    line, = ax.plot([], [], color=color, lw=0.8, alpha=0.9)
    dot, = ax.plot([], [], 'o', color=color, ms=3.5)

    def update(frame):
        if trail:
            start = 0 if trail_length is None else max(0, frame - trail_length)
            line.set_data(r_t[start:frame + 1, 0], r_t[start:frame + 1, 1])
        dot.set_data([r_t[frame, 0]], [r_t[frame, 1]])
        return [line, dot]

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=interval)
    if save:
        ani.save(filename=save, writer="pillow")
    return ani


def Plot_Classical_Trajectory(t_array, r_t, sample_idx=0, title=False, save=False,
                              cmap='viridis', figsize=(7.5, 3.2)):
    """Classical stochastic trajectory: phase space (left) and quadratures vs time (right).

    Parameters
    ----------
    t_array    : (n_t,) time grid
    r_t        : (n_samples, n_t, 2) classical trajectories (x_1, p_1)
    sample_idx : int or list of ints, the samples to show
    title      : suptitle (False to omit)
    save       : filepath or False

    Returns
    -------
    fig, (ax1, ax2)
    """
    rcParams['mathtext.fontset'] = 'cm'
    rcParams['font.family'] = 'STIXGeneral'

    samples = np.atleast_1d(sample_idx)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=300,
                                   gridspec_kw={'width_ratios': [1, 1.4]})
    fig.set_tight_layout(True)

    norm = colors.Normalize(vmin=t_array[0], vmax=t_array[-1])
    for s in samples:
        pts = r_t[s].reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap=cmap, norm=norm, linewidth=0.6)
        lc.set_array(t_array[:-1])
        ax1.add_collection(lc)
    lim = 1.05*np.max(np.abs(r_t[samples]))
    ax1.set_xlim(-lim, lim)
    ax1.set_ylim(-lim, lim)
    ax1.set_aspect('equal')
    ax1.axvline(x=0, color='black', linewidth=0.5, alpha=0.4)
    ax1.axhline(y=0, color='black', linewidth=0.5, alpha=0.4)
    ax1.set_xlabel(r'Position, $x_1$', labelpad=2, fontsize=11)
    ax1.set_ylabel(r'Momentum, $p_1$', labelpad=2, fontsize=11)
    cbar = fig.colorbar(lc, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label(r'$t$', fontsize=10, rotation=0)

    line_styles = ['-', '--', ':', '-.']
    for n, s in enumerate(samples):
        ls = line_styles[n % len(line_styles)]
        ax2.plot(t_array, r_t[s, :, 0], ls, color='tab:blue', lw=0.8,
                 label=r'$x_1(t)$' if n == 0 else None)
        ax2.plot(t_array, r_t[s, :, 1], ls, color='tab:red', lw=0.8,
                 label=r'$p_1(t)$' if n == 0 else None)
    ax2.axhline(y=0, color='black', linewidth=0.5, alpha=0.4)
    ax2.set_xlim(t_array[0], t_array[-1])
    ax2.set_xlabel(r'Time, $t$', labelpad=2, fontsize=11)
    leg = ax2.legend(fontsize=9, frameon=True)
    leg.get_frame().set_linewidth(0.4)

    for ax in (ax1, ax2):
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
    if title:
        fig.suptitle(title, fontsize=13)
    if save:
        plt.savefig(save, bbox_inches='tight', transparent=True)
    return fig, (ax1, ax2)
