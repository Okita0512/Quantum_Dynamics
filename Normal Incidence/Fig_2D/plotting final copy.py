import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots()
from scipy import integrate
from matplotlib.pyplot import MultipleLocator, tick_params
# fig = plt.figure(figsize=(10,5),dpi=80)
import matplotlib.ticker as ticker
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams["font.family"] = "Helvetica"

lw = 3.0
axis_size = 28
unitlen = 6
legendsize = 48         # size for legend
font_legend = {'family':'Times New Roman',
        'style':'normal', # 'italic'
        'weight':'normal', #or 'blod'
        'size':28
        }

fig, ax = plt.subplots(1, 3, figsize=(3.0 * unitlen, 2.0 * unitlen), dpi = 512, sharey = 'row')
fig.subplots_adjust(hspace = 0.25, wspace = 0.0)

# ==============================================================================================
#                                       Global Parameters     
# ==============================================================================================
conv = 27.211397                            # 1 a.u. = 27.211397 eV
fs_to_au = 41.341                           # 1 fs = 41.341 a.u.
cm_to_au = 4.556335e-06                     # 1 cm^-1 = 4.556335e-06 a.u.
au_to_K = 3.1577464e+05                     # 1 au = 3.1577464e+05 K
kcal_to_au = 1.5936e-03                     # 1 kcal/mol = 1.5936e-3 a.u.

# ===== Cavity parameters =====
w0 = 1190 * cm_to_au

# ===== Auxiliary functions =====
def cot(x):
    return np.cos(x) / np.sin(x)

def omega_k(theta, wc):
    return wc * np.sqrt(1 + np.tan(theta)**2)

def brownian(w, alpha, wcav):  # the effective spectral density function
    return ((2 * wcav**2 * alpha * w) / ((wcav**2 - w**2)**2 + (alpha * w)**2))

# ==============================================================================================
#                                       Plotting Fig 1a     
# ==============================================================================================
plt.subplot(2,3,1)

wc = 1.0 * w0
a_ratio = 360 / (2 * np.pi)
theta_scan = np.linspace(- np.pi / 3, np.pi / 3, 100000)

plt.plot(theta_scan * a_ratio, omega_k(theta_scan, wc) / cm_to_au, '--', linewidth = lw, color = 'red', label = r'$\omega_{\mathbf{k}}$')
plt.plot(theta_scan * a_ratio, [w0 / cm_to_au] * len(theta_scan), '--', linewidth = lw, color = 'black', label = r'$\omega_0$')

# scale for major and minor locator
x_major_locator = MultipleLocator(20)
x_minor_locator = MultipleLocator(10)
y_major_locator = MultipleLocator(400)
y_minor_locator = MultipleLocator(80)

# x-axis and LHS y-axis
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.xaxis.set_minor_locator(x_minor_locator)
ax.yaxis.set_major_locator(y_major_locator)
ax.yaxis.set_minor_locator(y_minor_locator)
ax.tick_params(which = 'major', length = 15, labelsize = 30)
ax.tick_params(which = 'minor', length = 5)

x1_label = ax.get_xticklabels()
[x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
y1_label = ax.get_yticklabels()
[y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]

y1, y2 = 600, 2000
plt.tick_params(labelsize = 30, which = 'both', direction = 'in')
plt.xlim(- 55, 55)
plt.ylim(y1, y2)

# RHS y-axis
ax2 = ax.twinx()
ax2.yaxis.set_major_locator(y_major_locator)
ax2.yaxis.set_minor_locator(y_minor_locator)
ax2.tick_params(which = 'major', length = 15)
ax2.tick_params(which = 'minor', length = 5)
ax2.axes.yaxis.set_ticklabels([])

plt.tick_params(which = 'both', direction = 'in')
plt.ylim(y1, y2)

# name of x, y axis and the panel
ax.set_xlabel(r'Incident Angle $\theta~ (^\circ)$', size = axis_size)
ax.set_ylabel(r'$\omega\ (\mathrm{cm}^{-1})$', size = axis_size)
ax.legend(loc = 'upper center', frameon = False, prop = font_legend)
plt.legend(title = '(a)', frameon = False, title_fontsize = legendsize)

# ==============================================================================================
#                                       Plotting Fig 1b     
# ==============================================================================================

plt.subplot(2,3,2)

font_legend = {'family':'Times New Roman',
        'style':'normal', # 'italic'
        'weight':'normal', #or 'blod'
        'size':14 # 28
        }

tau_c = 200 * fs_to_au
temp = 300 / au_to_K

def DOS(tau_0, w):

    Gamma_perpendicular = (wc / w) / tau_c
    Gamma_parallel = (np.sqrt(w**2 - wc**2) / w) / tau_0
    
    return w * Gamma_perpendicular * np.exp(- w / temp) / (Gamma_perpendicular + Gamma_parallel)

w_plot = np.linspace(wc, y2 * cm_to_au, 1000)

tau_0 = 1e3 * fs_to_au
plt.plot(DOS(tau_0, w_plot) * 1e5, w_plot / cm_to_au, '-', linewidth = lw, color = 'navy', label = r'$L / \mathrm{c}$ = 1000 fs')

tau_0 = 1e1 * fs_to_au
plt.plot(DOS(tau_0, w_plot) * 1e5, w_plot / cm_to_au, '-', linewidth = lw, color = 'deepskyblue', label = r'10 fs')

tau_0 = 1e-1 * fs_to_au
plt.plot(DOS(tau_0, w_plot) * 1e5, w_plot / cm_to_au, '-', linewidth = lw, color = 'green', label = r'0.1 fs')

tau_0 = 1e-2 * fs_to_au
plt.plot(DOS(tau_0, w_plot) * 1e5, w_plot / cm_to_au, '-', linewidth = lw, color = 'greenyellow', label = r'0.01 fs')

tau_0 = 1e-3 * fs_to_au
plt.plot(DOS(tau_0, w_plot) * 1e5, w_plot / cm_to_au, '-', linewidth = lw, color = 'red', label = r'0.001 fs')

plt.plot(np.linspace(-1, 100, 1000), [w0 / cm_to_au] * 1000, '--', linewidth = lw, color = 'black')

# x-axis and LHS y-axis
x_major_locator = MultipleLocator(0.5)
x_minor_locator = MultipleLocator(0.1)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.xaxis.set_minor_locator(x_minor_locator)
ax.yaxis.set_major_locator(y_major_locator)
ax.yaxis.set_minor_locator(y_minor_locator)
ax.tick_params(which = 'major', length = 15, labelsize = 30)
ax.tick_params(which = 'minor', length = 5)

plt.axvline(0.0, y1, y2, linewidth = lw, color = 'black')

x1_label = ax.get_xticklabels()
[x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
y1_label = ax.get_yticklabels()
[y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]

plt.tick_params('x', labelsize = 30, which = 'both', direction = 'in')
plt.tick_params('y', labelsize = 0, which = 'both', direction = 'in')
plt.xlim(0.0, 1.8)
plt.ylim(y1, y2)

# RHS y-axis
ax2 = ax.twinx()
ax2.yaxis.set_major_locator(y_major_locator)
ax2.yaxis.set_minor_locator(y_minor_locator)
ax2.tick_params(which = 'major', length = 15)
ax2.tick_params(which = 'minor', length = 5)
ax2.axes.yaxis.set_ticklabels([])

plt.tick_params(which = 'both', direction = 'in')
plt.ylim(y1, y2)

# name of x, y axis and the panel
ax.set_xlabel(r'Intensity ($\times 10^{-5}$ a.u.)', size = axis_size)
# ax.set_ylabel(r'$\omega\ (\mathrm{cm}^{-1})$', size = 36)
ax.legend(loc = 'upper center', frameon = False, prop = font_legend)
plt.legend(title = '(b)', frameon = False, title_fontsize = legendsize)

# ==============================================================================================
#                                       Plotting Fig 1c     
# ==============================================================================================

plt.subplot(2,3,3)

font_legend = {'family':'Times New Roman',
        'style':'normal', # 'italic'
        'weight':'normal', #or 'blod'
        'size':28
        }

lamc_mu_0 = 1e0
w_scan = np.linspace(0.5 * w0, 1.8 * w0, 100)    # scan the w0 value
w_int = np.linspace(wc, 2 * wc, int(1e5))
# w_int = np.logspace(np.log(wc), np.log(2 * wc), int(1e6), base = np.e)

rescaling = 0.5 * 0.001

spd = []
tau_0 = 1e-3 * fs_to_au
def integrant_2(w):
    jw_2 = DOS(tau_0, w)
    return jw_2
y_2 = integrant_2(w_int)
norm = integrate.trapz(y_2, w_int)
print(norm)

for z in w_scan:
    def integrant(w):
        Gamma_perpendicular = (wc / w) / tau_c
        Gamma_parallel = (np.sqrt(w**2 - wc**2) / w) / tau_0
        jw = ((np.sqrt(w**2 - wc**2) / w)**2 / (1 + (wc / w)**2)) * brownian(z, 1. / tau_c, w) * w * np.exp(- w0 / temp) * rescaling
        return jw
    y = integrant(w_int)
    Jz = integrate.trapz(y, w_int)
    
    spd.append(2 * Jz * lamc_mu_0**2 / norm)

plt.plot(spd, w_scan / cm_to_au, '-', linewidth = lw, color = 'red', label = r'$A(\omega)$')

spd = []
tau_0 = 1e-2 * fs_to_au
def integrant_2(w):
    jw_2 = DOS(tau_0, w)
    return jw_2
y_2 = integrant_2(w_int)
norm = integrate.trapz(y_2, w_int)

for z in w_scan:
    def integrant(w):
        Gamma_perpendicular = (wc / w) / tau_c
        Gamma_parallel = (np.sqrt(w**2 - wc**2) / w) / tau_0
        jw = brownian(z, Gamma_perpendicular + Gamma_parallel, w) * w * np.exp(- w0 / temp) * rescaling
        return jw
    y = integrant(w_int)
    Jz = integrate.trapz(y, w_int)
    spd.append(2 * Jz * lamc_mu_0**2 / norm)

plt.plot(spd, w_scan / cm_to_au, '-', linewidth = lw, color = 'greenyellow') #, label = r'$J_\mathrm{eff}(\omega)$')

spd = []
tau_0 = 1e-1 * fs_to_au
def integrant_2(w):
    jw_2 = DOS(tau_0, w)
    return jw_2
y_2 = integrant_2(w_int)
norm = integrate.trapz(y_2, w_int)

for z in w_scan:
    def integrant(w):
        Gamma_perpendicular = (wc / w) / tau_c
        Gamma_parallel = (np.sqrt(w**2 - wc**2) / w) / tau_0
        jw = brownian(z, Gamma_perpendicular + Gamma_parallel, w) * w * np.exp(- w0 / temp) * rescaling
        return jw
    y = integrant(w_int)
    Jz = integrate.trapz(y, w_int)
    spd.append(2 * Jz * lamc_mu_0**2 / norm)

plt.plot(spd, w_scan / cm_to_au, '-', linewidth = lw, color = 'green') # , label = r'$J_\mathrm{eff}(\omega)$')

spd = []
tau_0 = 1e1 * fs_to_au
def integrant_2(w):
    jw_2 = DOS(tau_0, w)
    return jw_2
y_2 = integrant_2(w_int)
norm = integrate.trapz(y_2, w_int)

for z in w_scan:
    def integrant(w):
        Gamma_perpendicular = (wc / w) / tau_c
        Gamma_parallel = (np.sqrt(w**2 - wc**2) / w) / tau_0
        jw = brownian(z, Gamma_perpendicular + Gamma_parallel, w) * w * np.exp(- w0 / temp) * rescaling
        return jw
    y = integrant(w_int)
    Jz = integrate.trapz(y, w_int)
    spd.append(2 * Jz * lamc_mu_0**2 / norm)

plt.plot(spd, w_scan / cm_to_au, '-', linewidth = lw, color = 'deepskyblue') # , label = r'$J_\mathrm{eff}(\omega)$')

spd = []
tau_0 = 1e3 * fs_to_au
def integrant_2(w):
    jw_2 = DOS(tau_0, w)
    return jw_2
y_2 = integrant_2(w_int)
norm = integrate.trapz(y_2, w_int)

for z in w_scan:
    def integrant(w):
        Gamma_perpendicular = (wc / w) / tau_c
        Gamma_parallel = (np.sqrt(w**2 - wc**2) / w) / tau_0
        jw = brownian(z, Gamma_perpendicular + Gamma_parallel, w) * w * np.exp(- w0 / temp) * rescaling
        return jw
    y = integrant(w_int)
    Jz = integrate.trapz(y, w_int)
    spd.append(2 * Jz * lamc_mu_0**2 / norm)

plt.plot(spd, w_scan / cm_to_au, '-', linewidth = lw, color = 'navy') # , label = r'$J_\mathrm{eff}(\omega)$')


plt.plot(np.linspace(0, 0.3, 1000), [w0 / cm_to_au] * 1000, '--', linewidth = lw, color = 'black')

# x-axis and LHS y-axis
x_major_locator = MultipleLocator(0.1)
x_minor_locator = MultipleLocator(0.02)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.xaxis.set_minor_locator(x_minor_locator)
ax.yaxis.set_major_locator(y_major_locator)
ax.yaxis.set_minor_locator(y_minor_locator)
ax.tick_params(which = 'major', length = 15, labelsize = 30)
ax.tick_params(which = 'minor', length = 5)

x1_label = ax.get_xticklabels()
[x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
y1_label = ax.get_yticklabels()
[y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]

plt.tick_params('x', labelsize = 30, which = 'both', direction = 'in')
plt.tick_params('y', labelsize = 0, which = 'both', direction = 'in')
plt.xlim(0, 0.3)
plt.ylim(y1, y2)

# RHS y-axis
ax2 = ax.twinx()
ax2.yaxis.set_major_locator(y_major_locator)
ax2.yaxis.set_minor_locator(y_minor_locator)
ax2.tick_params(which = 'major', length = 15)
ax2.tick_params(which = 'minor', length = 5)
ax2.axes.yaxis.set_ticklabels([])

plt.tick_params(which = 'both', direction = 'in')
plt.ylim(y1, y2)

# name of x, y axis and the panel
ax.set_xlabel(r'Intensity ($\mathrm{a.u.}$)', size = axis_size)
# ax.set_ylabel(r'$\omega\ (\mathrm{cm}^{-1})$', size = 36)
ax.legend(loc = 'upper center', frameon = False, prop = font_legend)
plt.legend(title = '(c)', frameon = False, title_fontsize = legendsize)






# ==============================================================================================
#                                       Plotting Fig 1d     
# ==============================================================================================
plt.subplot(2,3,4)

wc = 0.85 * w0
a_ratio = 360 / (2 * np.pi)
theta_scan = np.linspace(- np.pi / 3, np.pi / 3, 100000)

plt.plot(theta_scan * a_ratio, omega_k(theta_scan, wc) / cm_to_au, '--', linewidth = lw, color = 'red', label = r'$\omega_{\mathbf{k}}$')
plt.plot(theta_scan * a_ratio, [w0 / cm_to_au] * len(theta_scan), '--', linewidth = lw, color = 'black', label = r'$\omega_0$')

# scale for major and minor locator
x_major_locator = MultipleLocator(20)
x_minor_locator = MultipleLocator(10)
y_major_locator = MultipleLocator(400)
y_minor_locator = MultipleLocator(80)

# x-axis and LHS y-axis
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.xaxis.set_minor_locator(x_minor_locator)
ax.yaxis.set_major_locator(y_major_locator)
ax.yaxis.set_minor_locator(y_minor_locator)
ax.tick_params(which = 'major', length = 15, labelsize = 30)
ax.tick_params(which = 'minor', length = 5)

x1_label = ax.get_xticklabels()
[x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
y1_label = ax.get_yticklabels()
[y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]

y1, y2 = 600, 2000
plt.tick_params(labelsize = 30, which = 'both', direction = 'in')
plt.xlim(- 55, 55)
plt.ylim(y1, y2)

# RHS y-axis
ax2 = ax.twinx()
ax2.yaxis.set_major_locator(y_major_locator)
ax2.yaxis.set_minor_locator(y_minor_locator)
ax2.tick_params(which = 'major', length = 15)
ax2.tick_params(which = 'minor', length = 5)
ax2.axes.yaxis.set_ticklabels([])

plt.tick_params(which = 'both', direction = 'in')
plt.ylim(y1, y2)

# name of x, y axis and the panel
ax.set_xlabel(r'Incident Angle $\theta~ (^\circ)$', size = axis_size)
ax.set_ylabel(r'$\omega\ (\mathrm{cm}^{-1})$', size = axis_size)
ax.legend(loc = 'upper center', frameon = False, prop = font_legend)
plt.legend(title = '(a)', frameon = False, title_fontsize = legendsize)

# ==============================================================================================
#                                       Plotting Fig 1e     
# ==============================================================================================

plt.subplot(2,3,5)

font_legend = {'family':'Times New Roman',
        'style':'normal', # 'italic'
        'weight':'normal', #or 'blod'
        'size':14 # 28
        }

tau_c = 200 * fs_to_au
Gamma_perpendicular = 1.0 / tau_c
temp = 300 / au_to_K

def DOS(tau_0, w):

    Gamma_parallel = (np.sqrt(w**2 - wc**2) / w) / tau_0
    
    return w * Gamma_perpendicular * np.exp(- w / temp) / (Gamma_perpendicular + Gamma_parallel)

w_plot = np.linspace(wc, y2 * cm_to_au, 1000)

tau_0 = 1e3 * fs_to_au
plt.plot(DOS(tau_0, w_plot) * 1e5, w_plot / cm_to_au, '-', linewidth = lw, color = 'navy', label = r'$L / \mathrm{c}$ = 1000 fs')

tau_0 = 1e1 * fs_to_au
plt.plot(DOS(tau_0, w_plot) * 1e5, w_plot / cm_to_au, '-', linewidth = lw, color = 'deepskyblue', label = r'10 fs')

tau_0 = 1e-1 * fs_to_au
plt.plot(DOS(tau_0, w_plot) * 1e5, w_plot / cm_to_au, '-', linewidth = lw, color = 'green', label = r'0.1 fs')

tau_0 = 1e-2 * fs_to_au
plt.plot(DOS(tau_0, w_plot) * 1e5, w_plot / cm_to_au, '-', linewidth = lw, color = 'greenyellow', label = r'0.01 fs')

tau_0 = 1e-3 * fs_to_au
plt.plot(DOS(tau_0, w_plot) * 1e5, w_plot / cm_to_au, '-', linewidth = lw, color = 'red', label = r'0.001 fs')

plt.plot(np.linspace(-1, 100, 1000), [w0 / cm_to_au] * 1000, '--', linewidth = lw, color = 'black')

# x-axis and LHS y-axis
x_major_locator = MultipleLocator(0.5)
x_minor_locator = MultipleLocator(0.1)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.xaxis.set_minor_locator(x_minor_locator)
ax.yaxis.set_major_locator(y_major_locator)
ax.yaxis.set_minor_locator(y_minor_locator)
ax.tick_params(which = 'major', length = 15, labelsize = 30)
ax.tick_params(which = 'minor', length = 5)

plt.axvline(0.0, y1, y2, linewidth = lw, color = 'black')

x1_label = ax.get_xticklabels()
[x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
y1_label = ax.get_yticklabels()
[y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]

plt.tick_params('x', labelsize = 30, which = 'both', direction = 'in')
plt.tick_params('y', labelsize = 0, which = 'both', direction = 'in')
plt.xlim(0.0, 1.8)
plt.ylim(y1, y2)

# RHS y-axis
ax2 = ax.twinx()
ax2.yaxis.set_major_locator(y_major_locator)
ax2.yaxis.set_minor_locator(y_minor_locator)
ax2.tick_params(which = 'major', length = 15)
ax2.tick_params(which = 'minor', length = 5)
ax2.axes.yaxis.set_ticklabels([])

plt.tick_params(which = 'both', direction = 'in')
plt.ylim(y1, y2)

# name of x, y axis and the panel
ax.set_xlabel(r'Intensity ($\times 10^{-5}$ a.u.)', size = axis_size)
# ax.set_ylabel(r'$\omega\ (\mathrm{cm}^{-1})$', size = 36)
ax.legend(loc = 'upper center', frameon = False, prop = font_legend)
plt.legend(title = '(e)', frameon = False, title_fontsize = legendsize)

# ==============================================================================================
#                                       Plotting Fig 1f     
# ==============================================================================================

plt.subplot(2,3,6)

font_legend = {'family':'Times New Roman',
        'style':'normal', # 'italic'
        'weight':'normal', #or 'blod'
        'size':28
        }

lamc_mu_0 = 1e0
w_scan = np.linspace(0.5 * w0, 1.8 * w0, 100)    # scan the w0 value
w_int = np.linspace(wc, 2 * wc, int(1e5))

rescaling = 0.5 * np.exp(- w0 / temp)

spd = []
tau_0 = 1e-3 * fs_to_au
def integrant_2(w):
    jw_2 = DOS(tau_0, w)
    return jw_2
y_2 = integrant_2(w_int)
norm = integrate.trapz(y_2, w_int)

for z in w_scan:
    def integrant(w):
        Gamma_perpendicular = (wc / w) / tau_c
        jw = brownian(z, Gamma_perpendicular, w) * DOS(tau_0, w) * rescaling
        return jw
    y = integrant(w_int)
    Jz = integrate.trapz(y, w_int)
    
    spd.append(2 * Jz * lamc_mu_0**2 / norm)

plt.plot(spd, w_scan / cm_to_au, '-', linewidth = lw, color = 'red', label = r'$A(\omega)$')

spd = []
tau_0 = 1e-2 * fs_to_au
def integrant_2(w):
    jw_2 = DOS(tau_0, w)
    return jw_2
y_2 = integrant_2(w_int)
norm = integrate.trapz(y_2, w_int)

for z in w_scan:
    def integrant(w):
        Gamma_perpendicular = (wc / w) / tau_c
        jw = brownian(z, Gamma_perpendicular, w) * DOS(tau_0, w) * rescaling
        return jw
    y = integrant(w_int)
    Jz = integrate.trapz(y, w_int)
    spd.append(2 * Jz * lamc_mu_0**2 / norm)

plt.plot(spd, w_scan / cm_to_au, '-', linewidth = lw, color = 'greenyellow') #, label = r'$J_\mathrm{eff}(\omega)$')

spd = []
tau_0 = 1e-1 * fs_to_au
def integrant_2(w):
    jw_2 = DOS(tau_0, w)
    return jw_2
y_2 = integrant_2(w_int)
norm = integrate.trapz(y_2, w_int)

for z in w_scan:
    def integrant(w):
        Gamma_perpendicular = (wc / w) / tau_c
        jw = brownian(z, Gamma_perpendicular, w) * DOS(tau_0, w) * rescaling
        return jw
    y = integrant(w_int)
    Jz = integrate.trapz(y, w_int)
    spd.append(2 * Jz * lamc_mu_0**2 / norm)

plt.plot(spd, w_scan / cm_to_au, '-', linewidth = lw, color = 'green') # , label = r'$J_\mathrm{eff}(\omega)$')

spd = []
tau_0 = 1e1 * fs_to_au
def integrant_2(w):
    jw_2 = DOS(tau_0, w)
    return jw_2
y_2 = integrant_2(w_int)
norm = integrate.trapz(y_2, w_int)

for z in w_scan:
    def integrant(w):
        Gamma_perpendicular = (wc / w) / tau_c
        jw = brownian(z, Gamma_perpendicular, w) * DOS(tau_0, w) * rescaling
        return jw
    y = integrant(w_int)
    Jz = integrate.trapz(y, w_int)
    spd.append(2 * Jz * lamc_mu_0**2 / norm)

plt.plot(spd, w_scan / cm_to_au, '-', linewidth = lw, color = 'deepskyblue') # , label = r'$J_\mathrm{eff}(\omega)$')

spd = []
tau_0 = 1e3 * fs_to_au
def integrant_2(w):
    jw_2 = DOS(tau_0, w)
    return jw_2
y_2 = integrant_2(w_int)
norm = integrate.trapz(y_2, w_int)

for z in w_scan:
    def integrant(w):
        Gamma_perpendicular = (wc / w) / tau_c
        jw = brownian(z, Gamma_perpendicular, w) * DOS(tau_0, w) * rescaling
        return jw
    y = integrant(w_int)
    Jz = integrate.trapz(y, w_int)
    spd.append(2 * Jz * lamc_mu_0**2 / norm)

plt.plot(spd, w_scan / cm_to_au, '-', linewidth = lw, color = 'navy') # , label = r'$J_\mathrm{eff}(\omega)$')


plt.plot(np.linspace(0, 0.3, 1000), [w0 / cm_to_au] * 1000, '--', linewidth = lw, color = 'black')

# x-axis and LHS y-axis
x_major_locator = MultipleLocator(0.1)
x_minor_locator = MultipleLocator(0.02)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.xaxis.set_minor_locator(x_minor_locator)
ax.yaxis.set_major_locator(y_major_locator)
ax.yaxis.set_minor_locator(y_minor_locator)
ax.tick_params(which = 'major', length = 15, labelsize = 30)
ax.tick_params(which = 'minor', length = 5)

x1_label = ax.get_xticklabels()
[x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
y1_label = ax.get_yticklabels()
[y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]

plt.tick_params('x', labelsize = 30, which = 'both', direction = 'in')
plt.tick_params('y', labelsize = 0, which = 'both', direction = 'in')
plt.xlim(0, 0.3)
plt.ylim(y1, y2)

# RHS y-axis
ax2 = ax.twinx()
ax2.yaxis.set_major_locator(y_major_locator)
ax2.yaxis.set_minor_locator(y_minor_locator)
ax2.tick_params(which = 'major', length = 15)
ax2.tick_params(which = 'minor', length = 5)
ax2.axes.yaxis.set_ticklabels([])

plt.tick_params(which = 'both', direction = 'in')
plt.ylim(y1, y2)

# name of x, y axis and the panel
ax.set_xlabel(r'Intensity ($\mathrm{a.u.}$)', size = axis_size)
# ax.set_ylabel(r'$\omega\ (\mathrm{cm}^{-1})$', size = 36)
ax.legend(loc = 'upper center', frameon = False, prop = font_legend)
plt.legend(title = '(f)', frameon = False, title_fontsize = legendsize)














plt.savefig("figure_DOS_3.pdf", bbox_inches='tight')