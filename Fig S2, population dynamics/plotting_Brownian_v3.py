import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator, tick_params
import matplotlib.ticker as ticker
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams["font.family"] = "Helvetica"
from gen_input import parameters

# ================= global ====================

conv = 27.211397                            # 1 a.u. = 27.211397 eV
fs_to_au = 41.341                           # 1 fs = 41.341 a.u.
ps_to_fs = 1000                             # 1 ps = 1000 fs
cm_to_au = 4.556335e-06                     # 1 cm^-1 = 4.556335e-06 a.u.
au_to_K = 3.1577464e+05                     # 1 au = 3.1577464e+05 K
kcal_to_au = 1.5936e-03                     # 1 kcal/mol = 1.5936e-3 a.u.

# ==============================================================================================
# linewidth and lineshape
lw = 3.0
# size for legend
legendsize = 48        
font_legend = {'family':'Times New Roman', 'weight': 'roman', 'size': 24}
# axis label size
lsize = 30
txtsize = 24
# tick length
lmajortick = 15
lminortick = 5

color3 = '#33d572' # green, |3>
color2 = '#e7513e' # red, |2>
color1 = '#f1a51e' # orange, |1>
color0 = '#378ada' # blue, |0>

# ==============================================================================================

# specify each number of DOFs
n_vib = parameters.n_vib
n_ph = parameters.n_ph
nfock = n_vib * n_ph
eigenvec = parameters.eigenvec

def get_overlap(state1, state2):

    P = 0.0
    center = int(len(eigenvec[:, 0])/2) # actually center - 1, since Ngrid is an odd number

    for k in range(0, center):
        P += eigenvec[k, state1].conjugate() * eigenvec[k, state2]
    
    P += eigenvec[center, state1].conjugate() * eigenvec[center, state2] / 2.0

    return P

def delta(m, n):
    return 1 if m == n else 0

# ==============================================================================================
#                                      plotting functions     
# ==============================================================================================

# ====== data reading ======
# data1 = np.loadtxt("1170.dat", dtype = float)
data2 = np.loadtxt("1000fs.dat", dtype = float)
# data2 = np.loadtxt("0.02.dat", dtype = float)

def plot_pop_rate():

    """
    computing the product total population:

    P_P(t) = 1 - Tr[(1 - h)rho(t)]
           = sum_ij rho_ij (t) * overlap(i,j)

    where overlap(i,j) is defined as the integral from 0 to +\infty for psi^*_i(R) and psi_j(R)
    """

    PRt = 0

    for i in range(n_vib):
        for j in range(n_vib):
            for m in range(n_ph):
                for n in range(n_ph):

                    PRt += get_overlap(i, j) * data2[:, (nfock * (n_vib * m + i) + (n_vib * n + j)) + 1] * delta(m, n)

    """
    computing the rate constant, using the expression

    k = lim_(t -> + \infty) [ (d P_P(t) / dt) / (1 - 2 * P_P(t)) ]

    """

    PRt = np.real(PRt)
    dt = (data2[1,0] - data2[0,0]) # / fs_to_au

    rate = np.zeros((len(PRt) - 1), dtype=float)
    for n in range(1, len(rate)):
        rate[n] = ((PRt[n] - PRt[n - 1]) / dt) / (1.0 - PRt[n] / 0.5)

    # plot the whole product state population
    plt.plot(data2[:,0] / fs_to_au, 1 - PRt, "-", color = 'blue', label = "P_prod(t)_HEOM")
    plt.legend()
    plt.show()

    # plot the time-dependent rate constant, unit = fs^-1
    time = data2[:,0]
    plt.semilogx(time[0:-1] / fs_to_au, rate, "-", color = 'black', label = "rate constant")
    plt.legend()
    plt.show()

    return 0

plot_pop_rate()

# ==============================================================================================
#                                      Fig S2a-c     
# ==============================================================================================

unitlen = 6
fig = plt.figure(figsize=(2 * unitlen, 2.6 * unitlen), dpi = 128)
fig.subplots_adjust(hspace = 0.0, wspace = 0.3)

"""
plotting the population of hybrid light-matter states of zero photons:

"""
iph = 0

# vL, vR states
ivib = 0
jvib = 1
pop0 = data2[:, (nfock * (n_vib * iph + ivib) + (n_vib * iph + ivib)) + 1]
coh01 = data2[:, (nfock * (n_vib * iph + ivib) + (n_vib * iph + jvib)) + 1]
coh10 = data2[:, (nfock * (n_vib * iph + jvib) + (n_vib * iph + ivib)) + 1]
pop1 = data2[:, (nfock * (n_vib * iph + jvib) + (n_vib * iph + jvib)) + 1]

pop_vL1 = 0.5 * np.real(pop0 - coh01 - coh10 + pop1)
pop_vR1 = 0.5 * np.real(pop0 + coh01 + coh10 + pop1)

# v'L, v'R states
ivib = 2
jvib = 3
pop0 = data2[:, (nfock * (n_vib * iph + ivib) + (n_vib * iph + ivib)) + 1]
coh01 = data2[:, (nfock * (n_vib * iph + ivib) + (n_vib * iph + jvib)) + 1]
coh10 = data2[:, (nfock * (n_vib * iph + jvib) + (n_vib * iph + ivib)) + 1]
pop1 = data2[:, (nfock * (n_vib * iph + jvib) + (n_vib * iph + jvib)) + 1]

pop_vL2 = 0.5 * np.real(pop0 + coh01 + coh10 + pop1)
pop_vR2 = 0.5 * np.real(pop0 - coh01 - coh10 + pop1)

nph_0 = pop_vL1 + pop_vR1 + pop_vL2 + pop_vR2

# ==============================================================================================

plt.subplot(3,2,1)

plt.plot(data2[:,0] / (ps_to_fs * fs_to_au), pop_vL1, "-", linewidth = lw, color = color0, label = r"$|\nu_\mathrm{L}, %s \rangle$" % iph)

# scale for major and minor locator
x_major_locator = MultipleLocator(5)
x_minor_locator = MultipleLocator(1)
y_major_locator = MultipleLocator(0.05)
y_minor_locator = MultipleLocator(0.01)

# x-axis and LHS y-axis
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.xaxis.set_minor_locator(x_minor_locator)
ax.yaxis.set_major_locator(y_major_locator)
ax.yaxis.set_minor_locator(y_minor_locator)
ax.tick_params(which = 'major', length = lmajortick, labelsize = 10, pad = 10)
ax.tick_params(which = 'minor', length = lminortick)

x1_label = ax.get_xticklabels()
[x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
y1_label = ax.get_yticklabels()
[y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]

x1, x2 = 0, 20
y1, y2 = 0.86, 1.0

plt.tick_params(axis = 'x', labelsize = 0, which = 'both', direction = 'in')
plt.tick_params(axis = 'y', labelsize = lsize, which = 'both', direction = 'in')
plt.xlim(x1, x2)
plt.ylim(y1, y2)

# RHS y-axis
ax2 = ax.twinx()
ax2.yaxis.set_major_locator(y_major_locator)
ax2.yaxis.set_minor_locator(y_minor_locator)
ax2.tick_params(which = 'major', length = lmajortick)
ax2.tick_params(which = 'minor', length = lminortick)
ax2.axes.yaxis.set_ticklabels([])

plt.tick_params(which = 'both', direction = 'in')
plt.ylim(y1, y2)

ax.set_ylabel(r'Population', size = txtsize)
ax.legend(frameon = False, loc = 'lower left', prop = font_legend, markerscale = 1)
plt.legend(title = '(a)', frameon = False, title_fontsize = legendsize)

# ==============================================================================================

plt.subplot(3,2,3)

plt.plot(data2[:,0] / (ps_to_fs * fs_to_au), pop_vR1, "-", linewidth = lw, color = color1, label = r"$|\nu_\mathrm{R}, %s \rangle$" % iph)

# scale for major and minor locator
y_major_locator = MultipleLocator(0.05)
y_minor_locator = MultipleLocator(0.01)

# x-axis and LHS y-axis
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.xaxis.set_minor_locator(x_minor_locator)
ax.yaxis.set_major_locator(y_major_locator)
ax.yaxis.set_minor_locator(y_minor_locator)
ax.tick_params(which = 'major', length = lmajortick, labelsize = 10, pad = 10)
ax.tick_params(which = 'minor', length = lminortick)

x1_label = ax.get_xticklabels()
[x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
y1_label = ax.get_yticklabels()
[y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]

y1, y2 = -0.01, 0.13

plt.tick_params(axis = 'x', labelsize = 0, which = 'both', direction = 'in')
plt.tick_params(axis = 'y', labelsize = lsize, which = 'both', direction = 'in')
plt.xlim(x1, x2)
plt.ylim(y1, y2)

# RHS y-axis
ax2 = ax.twinx()
ax2.yaxis.set_major_locator(y_major_locator)
ax2.yaxis.set_minor_locator(y_minor_locator)
ax2.tick_params(which = 'major', length = lmajortick)
ax2.tick_params(which = 'minor', length = lminortick)
ax2.axes.yaxis.set_ticklabels([])

plt.tick_params(which = 'both', direction = 'in')
plt.ylim(y1, y2)

ax.set_ylabel(r'Population', size = txtsize)
ax.legend(frameon = False, loc = 'upper left', prop = font_legend, markerscale = 1)
plt.legend(title = '(b)', frameon = False, title_fontsize = legendsize)

# ==============================================================================================

plt.subplot(3,2,5)

"""
plotting the population of |n = iph > Fock states

"""

plt.plot(data2[:,0] / (ps_to_fs * fs_to_au), nph_0, "-", linewidth = lw, color = 'violet', label = r"$|\mathrm{n}_\mathrm{ph} = %s \rangle$" % iph, alpha = .7)

# scale for major and minor locator
y_major_locator = MultipleLocator(0.01)
y_minor_locator = MultipleLocator(0.002)

# x-axis and LHS y-axis
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.xaxis.set_minor_locator(x_minor_locator)
ax.yaxis.set_major_locator(y_major_locator)
ax.yaxis.set_minor_locator(y_minor_locator)
ax.tick_params(which = 'major', length = lmajortick, labelsize = 10, pad = 10)
ax.tick_params(which = 'minor', length = lminortick)

x1_label = ax.get_xticklabels()
[x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
y1_label = ax.get_yticklabels()
[y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]

y1, y2 = 0.98, 0.999

plt.tick_params(labelsize = lsize, which = 'both', direction = 'in')
plt.xlim(x1, x2)
plt.ylim(y1, y2)

# RHS y-axis
ax2 = ax.twinx()
ax2.yaxis.set_major_locator(y_major_locator)
ax2.yaxis.set_minor_locator(y_minor_locator)
ax2.tick_params(which = 'major', length = lmajortick)
ax2.tick_params(which = 'minor', length = lminortick)
ax2.axes.yaxis.set_ticklabels([])

plt.tick_params(which = 'both', direction = 'in')
plt.ylim(y1, y2)

ax.set_xlabel(r'time (ps)', size = txtsize)
ax.set_ylabel(r'Population', size = txtsize)
ax.legend(frameon = False, loc = 'upper left', prop = font_legend, markerscale = 1)
plt.legend(title = '(c)', frameon = False, title_fontsize = legendsize)

# ==============================================================================================
#                                      Fig S2d-f     
# ==============================================================================================

"""
plotting the population of hybrid light-matter states of 1 photons:

"""

plt.subplot(3,2,2)

plt.plot(data2[:,0] / (ps_to_fs * fs_to_au), pop_vL2, "-", linewidth = lw, color = color2, label = r"$|\nu'_\mathrm{L}, %s \rangle$" % iph)
plt.plot(data2[:,0] / (ps_to_fs * fs_to_au), pop_vR2, "-", linewidth = lw, color = color3, label = r"$|\nu'_\mathrm{R}, %s \rangle$" % iph)

iph = 1

# vL, vR states
ivib = 0
jvib = 1
pop0 = data2[:, (nfock * (n_vib * iph + ivib) + (n_vib * iph + ivib)) + 1]
coh01 = data2[:, (nfock * (n_vib * iph + ivib) + (n_vib * iph + jvib)) + 1]
coh10 = data2[:, (nfock * (n_vib * iph + jvib) + (n_vib * iph + ivib)) + 1]
pop1 = data2[:, (nfock * (n_vib * iph + jvib) + (n_vib * iph + jvib)) + 1]

pop_vL1 = 0.5 * np.real(pop0 - coh01 - coh10 + pop1)
pop_vR1 = 0.5 * np.real(pop0 + coh01 + coh10 + pop1)

# v'L, v'R states
ivib = 2
jvib = 3
pop0 = data2[:, (nfock * (n_vib * iph + ivib) + (n_vib * iph + ivib)) + 1]
coh01 = data2[:, (nfock * (n_vib * iph + ivib) + (n_vib * iph + jvib)) + 1]
coh10 = data2[:, (nfock * (n_vib * iph + jvib) + (n_vib * iph + ivib)) + 1]
pop1 = data2[:, (nfock * (n_vib * iph + jvib) + (n_vib * iph + jvib)) + 1]

pop_vL2 = 0.5 * np.real(pop0 + coh01 + coh10 + pop1)
pop_vR2 = 0.5 * np.real(pop0 - coh01 - coh10 + pop1)

nph_1 = pop_vL1 + pop_vR1 + pop_vL2 + pop_vR2

# ==============================================================================================

plt.plot(data2[:,0] / (ps_to_fs * fs_to_au), pop_vL1, "-", linewidth = lw, color = "magenta", label = r"$|\nu_\mathrm{L}, %s \rangle$" % iph, alpha = 1)
plt.plot(data2[:,0] / (ps_to_fs * fs_to_au), pop_vR1, "-", linewidth = lw, color = "greenyellow", label = r"$|\nu_\mathrm{R}, %s \rangle$" % iph, alpha = 1)

# scale for major and minor locator
y_major_locator = MultipleLocator(0.01)
y_minor_locator = MultipleLocator(0.005)

# x-axis and LHS y-axis
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.xaxis.set_minor_locator(x_minor_locator)
ax.yaxis.set_major_locator(y_major_locator)
ax.yaxis.set_minor_locator(y_minor_locator)
ax.tick_params(which = 'major', length = lmajortick, labelsize = 10, pad = 10)
ax.tick_params(which = 'minor', length = lminortick)

x1_label = ax.get_xticklabels()
[x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
y1_label = ax.get_yticklabels()
[y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]

y1, y2 = -0.001, 0.028

plt.tick_params(axis = 'x', labelsize = 0, which = 'both', direction = 'in')
plt.tick_params(axis = 'y', labelsize = lsize, which = 'both', direction = 'in')
plt.xlim(x1, x2)
plt.ylim(y1, y2)

# RHS y-axis
ax2 = ax.twinx()
ax2.yaxis.set_major_locator(y_major_locator)
ax2.yaxis.set_minor_locator(y_minor_locator)
ax2.tick_params(which = 'major', length = lmajortick)
ax2.tick_params(which = 'minor', length = lminortick)
ax2.axes.yaxis.set_ticklabels([])

plt.tick_params(which = 'both', direction = 'in')
plt.ylim(y1, y2)

ax.legend(frameon = False, loc = 'upper left', prop = font_legend, markerscale = 1)
plt.legend(title = '(d)', frameon = False, title_fontsize = legendsize)

# ==============================================================================================

"""
plotting the population of hybrid light-matter states of 2 photons:

"""

plt.subplot(3,2,4)

plt.plot(data2[:,0] / (ps_to_fs * fs_to_au), pop_vL2 * 1e4, "-", linewidth = lw, color = color2, label = r"$|\nu'_\mathrm{L}, %s \rangle$" % iph, alpha = .7)
plt.plot(data2[:,0] / (ps_to_fs * fs_to_au), pop_vR2 * 1e4, "-", linewidth = lw, color = color3, label = r"$|\nu'_\mathrm{R}, %s \rangle$" % iph, alpha = .7)

iph = 2

# vL, vR states
ivib = 0
jvib = 1
pop0 = data2[:, (nfock * (n_vib * iph + ivib) + (n_vib * iph + ivib)) + 1]
coh01 = data2[:, (nfock * (n_vib * iph + ivib) + (n_vib * iph + jvib)) + 1]
coh10 = data2[:, (nfock * (n_vib * iph + jvib) + (n_vib * iph + ivib)) + 1]
pop1 = data2[:, (nfock * (n_vib * iph + jvib) + (n_vib * iph + jvib)) + 1]

pop_vL1 = 0.5 * np.real(pop0 - coh01 - coh10 + pop1)
pop_vR1 = 0.5 * np.real(pop0 + coh01 + coh10 + pop1)

# v'L, v'R states
ivib = 2
jvib = 3
pop0 = data2[:, (nfock * (n_vib * iph + ivib) + (n_vib * iph + ivib)) + 1]
coh01 = data2[:, (nfock * (n_vib * iph + ivib) + (n_vib * iph + jvib)) + 1]
coh10 = data2[:, (nfock * (n_vib * iph + jvib) + (n_vib * iph + ivib)) + 1]
pop1 = data2[:, (nfock * (n_vib * iph + jvib) + (n_vib * iph + jvib)) + 1]

pop_vL2 = 0.5 * np.real(pop0 + coh01 + coh10 + pop1)
pop_vR2 = 0.5 * np.real(pop0 - coh01 - coh10 + pop1)

nph_2 = pop_vL1 + pop_vR1 + pop_vL2 + pop_vR2

# ==============================================================================================

plt.plot(data2[:,0] / (ps_to_fs * fs_to_au), pop_vL1 * 1e4, "-", linewidth = lw, color = color0, label = r"$|\nu_\mathrm{L}, %s \rangle$" % iph, alpha = .4)
plt.plot(data2[:,0] / (ps_to_fs * fs_to_au), pop_vR1 * 1e4, "-", linewidth = lw, color = color1, label = r"$|\nu_\mathrm{R}, %s \rangle$" % iph, alpha = .4)
# plt.plot(data2[:,0] / (ps_to_fs * fs_to_au), pop_vL2 * 1e4, "-", linewidth = lw, color = color2, label = r"$|\nu'_\mathrm{L}, %s \rangle$" % iph, alpha = .4)
# plt.plot(data2[:,0] / (ps_to_fs * fs_to_au), pop_vR2 * 1e4, "-", linewidth = lw, color = color3, label = r"$|\nu'_\mathrm{R}, %s \rangle$" % iph, alpha = .4)

# scale for major and minor locator
y_major_locator = MultipleLocator(1.0)
y_minor_locator = MultipleLocator(0.2)

# x-axis and LHS y-axis
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.xaxis.set_minor_locator(x_minor_locator)
ax.yaxis.set_major_locator(y_major_locator)
ax.yaxis.set_minor_locator(y_minor_locator)
ax.tick_params(which = 'major', length = lmajortick, labelsize = 10, pad = 10)
ax.tick_params(which = 'minor', length = lminortick)

x1_label = ax.get_xticklabels()
[x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
y1_label = ax.get_yticklabels()
[y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]

y1, y2 = -0.1, 2.8

plt.tick_params(axis = 'x', labelsize = 0, which = 'both', direction = 'in')
plt.tick_params(axis = 'y', labelsize = lsize, which = 'both', direction = 'in')
plt.xlim(x1, x2)
plt.ylim(y1, y2)

# RHS y-axis
ax2 = ax.twinx()
ax2.yaxis.set_major_locator(y_major_locator)
ax2.yaxis.set_minor_locator(y_minor_locator)
ax2.tick_params(which = 'major', length = lmajortick)
ax2.tick_params(which = 'minor', length = lminortick)
ax2.axes.yaxis.set_ticklabels([])

plt.tick_params(which = 'both', direction = 'in')
plt.ylim(y1, y2)

ax.set_ylabel(r'($\times 10^{-4}$)', size = txtsize, labelpad = 20)
ax.legend(frameon = False, loc = 'upper left', prop = font_legend, markerscale = 1)
plt.legend(title = '(e)', frameon = False, title_fontsize = legendsize)

# ==============================================================================================

plt.subplot(3,2,6)

"""
plotting the population of |n = iph > Fock states

"""

plt.plot(data2[:,0] / (ps_to_fs * fs_to_au), nph_1, "-", linewidth = lw, color = 'm', label = r"$|\mathrm{n}_\mathrm{ph} = 1 \rangle$", alpha = .7)
plt.plot(data2[:,0] / (ps_to_fs * fs_to_au), nph_2, "-", linewidth = lw, color = 'purple', label = r"$|\mathrm{n}_\mathrm{ph} = 2 \rangle$", alpha = .7)

# scale for major and minor locator
y_major_locator = MultipleLocator(0.005)
y_minor_locator = MultipleLocator(0.001)

# x-axis and LHS y-axis
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.xaxis.set_minor_locator(x_minor_locator)
ax.yaxis.set_major_locator(y_major_locator)
ax.yaxis.set_minor_locator(y_minor_locator)
ax.tick_params(which = 'major', length = lmajortick, labelsize = 10, pad = 10)
ax.tick_params(which = 'minor', length = lminortick)

x1_label = ax.get_xticklabels()
[x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
y1_label = ax.get_yticklabels()
[y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]

y1, y2 = -0.001, 0.014

plt.tick_params(labelsize = lsize, which = 'both', direction = 'in')
plt.xlim(x1, x2)
plt.ylim(y1, y2)

# RHS y-axis
ax2 = ax.twinx()
ax2.yaxis.set_major_locator(y_major_locator)
ax2.yaxis.set_minor_locator(y_minor_locator)
ax2.tick_params(which = 'major', length = lmajortick)
ax2.tick_params(which = 'minor', length = lminortick)
ax2.axes.yaxis.set_ticklabels([])

plt.tick_params(which = 'both', direction = 'in')
plt.ylim(y1, y2)

ax.set_xlabel(r'time (ps)', size = txtsize)
ax.legend(frameon = False, loc = 'upper left', prop = font_legend, markerscale = 1)
plt.legend(title = '(f)', frameon = False, title_fontsize = legendsize)



plt.savefig("Fig_populations.pdf", bbox_inches='tight')
