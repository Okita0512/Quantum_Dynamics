import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots()
from matplotlib.pyplot import MultipleLocator, tick_params
from gen_input import parameters
import matplotlib.ticker as ticker
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams["font.family"] = "Helvetica"

# ================= global ====================

conv = 27.211397                            # 1 a.u. = 27.211397 eV
fs_to_au = 41.341                           # 1 fs = 41.341 a.u.
cm_to_au = 4.556335e-06                     # 1 cm^-1 = 4.556335e-06 a.u.
au_to_K = 3.1577464e+05                     # 1 au = 3.1577464e+05 K
kcal_to_au = 1.5936e-03                     # 1 kcal/mol = 1.5936e-3 a.u.

def get_overlap(state1, state2):

    P = 0.0
    center = int(len(eigenvec[:, 0])/2) # actually center - 1, since Ngrid is an odd number

    for k in range(0, center):
        P += eigenvec[k, state1].conjugate() * eigenvec[k, state2]
    
    P += eigenvec[center, state1].conjugate() * eigenvec[center, state2] / 2.0

    return P

def delta(m, n):
    return 1 if m == n else 0

NStates = parameters.NStates
eigenvec = parameters.eigenvec

# ==============================================================================================

lw = 3.0
legendsize = 48         # size for legend
font_legend = {'family':'Times New Roman', 'weight': 'roman', 'size': 16}

unitlen = 8
fig = plt.figure(figsize=(3.0 * unitlen, 0.83 * unitlen), dpi = 512)
fig.subplots_adjust(wspace = 0.2)

# ==============================================================================================
#                                      Fig 2a: time dependent rate  
# ==============================================================================================

plt.subplot(1,3,1)

data1 = np.loadtxt("./PSD_es=0.1_ec=0.05_tauc=200/900.dat", dtype = float)
data2 = np.loadtxt("./PSD_es=0.1_ec=0.05_tauc=200/1000.dat", dtype = float)
data3 = np.loadtxt("./PSD_es=0.1_ec=0.05_tauc=200/1100.dat", dtype = float)
data4 = np.loadtxt("./PSD_es=0.1_ec=0.05_tauc=200/1170.dat", dtype = float)
data5 = np.loadtxt("./PSD_es=0.1_ec=0.05_tauc=200/1200.dat", dtype = float)
data6 = np.loadtxt("./PSD_es=0.1_ec=0.05_tauc=200/1300.dat", dtype = float)
data7 = np.loadtxt("./PSD_es=0.1_ec=0.05_tauc=200/1400.dat", dtype = float)

PRt_1 = 0
PRt_2 = 0
PRt_3 = 0
PRt_4 = 0
PRt_5 = 0
PRt_6 = 0
PRt_7 = 0

for i in range(NStates):
    for j in range(NStates):
        PRt_1 += get_overlap(i, j) * data1[:, NStates * i + j + 1]
        PRt_2 += get_overlap(i, j) * data2[:, NStates * i + j + 1]
        PRt_3 += get_overlap(i, j) * data3[:, NStates * i + j + 1]
        PRt_4 += get_overlap(i, j) * data4[:, NStates * i + j + 1]
        PRt_5 += get_overlap(i, j) * data5[:, NStates * i + j + 1]
        PRt_6 += get_overlap(i, j) * data6[:, NStates * i + j + 1]
        PRt_7 += get_overlap(i, j) * data7[:, NStates * i + j + 1]

"""
computing the rate constant, using the expression

k = lim_(t -> + \infty) [ (d P_P(t) / dt) / (1 - 2 * P_P(t)) ]

"""

dt = (data2[1,0] - data2[0,0]) / fs_to_au

rate_1 = np.zeros((len(PRt_1) - 1), dtype=float)
rate_2 = np.zeros((len(PRt_2) - 1), dtype=float)
rate_3 = np.zeros((len(PRt_3) - 1), dtype=float)
rate_4 = np.zeros((len(PRt_4) - 1), dtype=float)
rate_5 = np.zeros((len(PRt_5) - 1), dtype=float)
rate_6 = np.zeros((len(PRt_6) - 1), dtype=float)
rate_7 = np.zeros((len(PRt_7) - 1), dtype=float)

for n in range(1, len(rate_1)):
    rate_1[n] = np.real( ((PRt_1[n] - PRt_1[n - 1]) / dt) / (1.0 - PRt_1[n] / 0.5) )

for n in range(1, len(rate_2)):
    rate_2[n] = np.real( ((PRt_2[n] - PRt_2[n - 1]) / dt) / (1.0 - PRt_2[n] / 0.5) )

for n in range(1, len(rate_3)):
    rate_3[n] = np.real( ((PRt_3[n] - PRt_3[n - 1]) / dt) / (1.0 - PRt_3[n] / 0.5) )

for n in range(1, len(rate_4)):
    rate_4[n] = np.real( ((PRt_4[n] - PRt_4[n - 1]) / dt) / (1.0 - PRt_4[n] / 0.5) )

for n in range(1, len(rate_5)):
    rate_5[n] = np.real( ((PRt_5[n] - PRt_5[n - 1]) / dt) / (1.0 - PRt_5[n] / 0.5) )

for n in range(1, len(rate_6)):
    rate_6[n] = np.real( ((PRt_6[n] - PRt_6[n - 1]) / dt) / (1.0 - PRt_6[n] / 0.5) )

for n in range(1, len(rate_7)):
    rate_7[n] = np.real( ((PRt_7[n] - PRt_7[n - 1]) / dt) / (1.0 - PRt_7[n] / 0.5) )

time_1 = data1[:,0]
time_2 = data2[:,0]
time_3 = data3[:,0]
time_4 = data4[:,0]
time_5 = data5[:,0]
time_6 = data6[:,0]
time_7 = data7[:,0]

# plot the time-dependent rate constant, unit = fs^-1
ratio = 1e-5
plt.semilogx(time_1[0:-1] / fs_to_au, rate_1 / ratio, "-", linewidth = lw, color = 'violet', label = r"$\omega_\mathrm{c} = 900\ \mathrm{cm}^{-1}$")
plt.semilogx(time_2[0:-1] / fs_to_au, rate_2 / ratio, "-", linewidth = lw, color = 'green', label = r'$\omega_\mathrm{c} = 1000\ \mathrm{cm}^{-1}$')
plt.semilogx(time_3[0:-1] / fs_to_au, rate_3 / ratio, "-", linewidth = lw, color = 'orange', label = r'$\omega_\mathrm{c} = 1100\ \mathrm{cm}^{-1}$')
plt.semilogx(time_4[0:-1] / fs_to_au, rate_4 / ratio, "-", linewidth = lw, color = 'red', label = r"$\omega_\mathrm{c} = 1172\ \mathrm{cm}^{-1}$")
plt.semilogx(time_5[0:-1] / fs_to_au, rate_5 / ratio, "-", linewidth = lw, color = 'gold', label = r'$\omega_\mathrm{c} = 1200\ \mathrm{cm}^{-1}$')
plt.semilogx(time_6[0:-1] / fs_to_au, rate_6 / ratio, "-", linewidth = lw, color = 'cyan', label = r'$\omega_\mathrm{c} = 1300\ \mathrm{cm}^{-1}$')
plt.semilogx(time_7[0:-1] / fs_to_au, rate_7 / ratio, "-", linewidth = lw, color = 'navy', label = r'$\omega_\mathrm{c} = 1400\ \mathrm{cm}^{-1}$')

# x and y range of plotting 
time = 10000
y1, y2 = - 0.6, 1.2     # y-axis range: (y1, y2)

# scale for major and minor locator
# x_major_locator = MultipleLocator(300)
# x_minor_locator = MultipleLocator(60)
y_major_locator = MultipleLocator(1)
y_minor_locator = MultipleLocator(0.5)

# x-axis and LHS y-axis
ax = plt.gca()
ax.yaxis.set_major_locator(y_major_locator)
ax.yaxis.set_minor_locator(y_minor_locator)
ax.tick_params(which = 'major', length = 15, labelsize = 30, pad = 10)
ax.tick_params(which = 'minor', length = 5)

x1_label = ax.get_xticklabels()
[x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
y1_label = ax.get_yticklabels()
[y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]

plt.tick_params(labelsize = 30, which = 'both', direction = 'in')
plt.xlim(100, time)
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

ax.set_xlabel(r'time (fs)', size = 24)
ax.set_ylabel(r'$k\ (\times 10^{-5}\ \mathrm{fs}^{-1})$', size = 24)
ax.legend(frameon = False, loc = 'lower right', prop = font_legend, markerscale = 1)
plt.legend(title = '(a)', frameon = False, title_fontsize = legendsize)

# ==============================================================================================
#                                      Fig 2b: time dependent rate  
# ==============================================================================================

plt.subplot(1,3,2)

data1 = np.loadtxt("./PSD_es=0.1_ec=0.05_tauc=1/1170.dat", dtype = float)
data2 = np.loadtxt("./PSD_es=0.1_ec=0.05_tauc=10/1170.dat", dtype = float)
data3 = np.loadtxt("./PSD_es=0.1_ec=0.05_tauc=100/1170.dat", dtype = float)
data4 = np.loadtxt("./PSD_es=0.1_ec=0.05_tauc=200/1170.dat", dtype = float)
data5 = np.loadtxt("./PSD_es=0.1_ec=0.05_tauc=1000/1170.dat", dtype = float)
data6 = np.loadtxt("./PSD_es=0.1_ec=0.05_tauc=2000/1170.dat", dtype = float)

PRt_1 = 0
PRt_2 = 0
PRt_3 = 0
PRt_4 = 0
PRt_5 = 0
PRt_6 = 0

for i in range(NStates):
    for j in range(NStates):
        PRt_1 += get_overlap(i, j) * data1[:, NStates * i + j + 1]
        PRt_2 += get_overlap(i, j) * data2[:, NStates * i + j + 1]
        PRt_3 += get_overlap(i, j) * data3[:, NStates * i + j + 1]
        PRt_4 += get_overlap(i, j) * data4[:, NStates * i + j + 1]
        PRt_5 += get_overlap(i, j) * data5[:, NStates * i + j + 1]
        PRt_6 += get_overlap(i, j) * data6[:, NStates * i + j + 1]

"""
computing the rate constant, using the expression

k = lim_(t -> + \infty) [ (d P_P(t) / dt) / (1 - 2 * P_P(t)) ]

"""

dt = (data2[1,0] - data2[0,0]) / fs_to_au

rate_1 = np.zeros((len(PRt_1) - 1), dtype=float)
rate_2 = np.zeros((len(PRt_2) - 1), dtype=float)
rate_3 = np.zeros((len(PRt_3) - 1), dtype=float)
rate_4 = np.zeros((len(PRt_4) - 1), dtype=float)
rate_5 = np.zeros((len(PRt_5) - 1), dtype=float)
rate_6 = np.zeros((len(PRt_6) - 1), dtype=float)

for n in range(1, len(rate_1)):
    rate_1[n] = np.real( ((PRt_1[n] - PRt_1[n - 1]) / dt) / (1.0 - PRt_1[n] / 0.5) )

for n in range(1, len(rate_2)):
    rate_2[n] = np.real( ((PRt_2[n] - PRt_2[n - 1]) / dt) / (1.0 - PRt_2[n] / 0.5) )

for n in range(1, len(rate_3)):
    rate_3[n] = np.real( ((PRt_3[n] - PRt_3[n - 1]) / dt) / (1.0 - PRt_3[n] / 0.5) )

for n in range(1, len(rate_4)):
    rate_4[n] = np.real( ((PRt_4[n] - PRt_4[n - 1]) / dt) / (1.0 - PRt_4[n] / 0.5) )

for n in range(1, len(rate_5)):
    rate_5[n] = np.real( ((PRt_5[n] - PRt_5[n - 1]) / dt) / (1.0 - PRt_5[n] / 0.5) )

for n in range(1, len(rate_6)):
    rate_6[n] = np.real( ((PRt_6[n] - PRt_6[n - 1]) / dt) / (1.0 - PRt_6[n] / 0.5) )

time_1 = data1[:,0]
time_2 = data2[:,0]
time_3 = data3[:,0]
time_4 = data4[:,0]
time_5 = data5[:,0]
time_6 = data6[:,0]

# plot the time-dependent rate constant, unit = fs^-1
ratio = 1e-5
line1, = plt.semilogx(time_6[0:-1] / fs_to_au, rate_6 / ratio, "-", linewidth = lw, color = 'red', label = r'$\tau_\mathrm{c} = 2000\ \mathrm{fs}$')
line2, = plt.semilogx(time_5[0:-1] / fs_to_au, rate_5 / ratio, "-", linewidth = lw, color = 'orange', label = r'$\tau_\mathrm{c} = 1000\ \mathrm{fs}$')
line3, = plt.semilogx(time_4[0:-1] / fs_to_au, rate_4 / ratio, "-", linewidth = lw, color = 'green', label = r"$\tau_\mathrm{c} = 200\ \mathrm{fs}$")
line4, = plt.semilogx(time_3[0:-1] / fs_to_au, rate_3 / ratio, "-", linewidth = lw, color = 'cyan', label = r'$\tau_\mathrm{c} = 100\ \mathrm{fs}$')
line5, = plt.semilogx(time_2[0:-1] / fs_to_au, rate_2 / ratio, "-", linewidth = lw, color = 'violet', label = r'$\tau_\mathrm{c} = 10\ \mathrm{fs}$')
line6, = plt.semilogx(time_1[0:-1] / fs_to_au, rate_1 / ratio, "-", linewidth = lw, color = 'black', label = r"$\tau_\mathrm{c}\ \to\ 0\ \mathrm{fs}$")

# x and y range of plotting 
time = 10000
y1, y2 = - 0.6, 1.2     # y-axis range: (y1, y2)

# scale for major and minor locator
# x_major_locator = MultipleLocator(300)
# x_minor_locator = MultipleLocator(60)
y_major_locator = MultipleLocator(1)
y_minor_locator = MultipleLocator(0.5)

# x-axis and LHS y-axis
ax = plt.gca()
ax.yaxis.set_major_locator(y_major_locator)
ax.yaxis.set_minor_locator(y_minor_locator)
ax.tick_params(which = 'major', length = 15, labelsize = 30, pad = 10)
ax.tick_params(which = 'minor', length = 5)

x1_label = ax.get_xticklabels()
[x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
y1_label = ax.get_yticklabels()
[y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]

plt.tick_params(labelsize = 30, which = 'both', direction = 'in')
plt.xlim(100, time)
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

ax.set_xlabel(r'time (fs)', size = 24)
ax.set_ylabel(r'$k\ (\times 10^{-5}\ \mathrm{fs}^{-1})$', size = 24)
label_legend = [r"$\tau_\mathrm{c}\ \to\ 0\ \mathrm{fs}$", r'$\tau_\mathrm{c} = 10\ \mathrm{fs}$', r'$\tau_\mathrm{c} = 100\ \mathrm{fs}$', r"$\tau_\mathrm{c} = 200\ \mathrm{fs}$", r'$\tau_\mathrm{c} = 1000\ \mathrm{fs}$', r'$\tau_\mathrm{c} = 2000\ \mathrm{fs}$']
ax.legend([line6, line5, line4, line3, line2, line1], label_legend, loc = 'lower right', frameon = False, prop = font_legend)
# ax.legend(frameon = False, loc = 'lower right', prop = font_legend, markerscale = 1)
plt.legend(title = '(b)', frameon = False, title_fontsize = legendsize)

# ==============================================================================================
#                                      Fig 2c: time dependent rate  
# ==============================================================================================

plt.subplot(1,3,3)

data1 = np.loadtxt("molecule.dat", dtype = float)
data2 = np.loadtxt("Discrete_es=0.1_ec=0.05_tauc=inf/1170.dat", dtype = float)

# ==============================================================================================

PRt = 0

for i in range(NStates):
    for j in range(NStates):

        PRt += get_overlap(i, j) * data2[:, NStates * i + j + 1]

PRt_0 = 0

for i in range(NStates):
    for j in range(NStates):

        PRt_0 += get_overlap(i, j) * data1[:, NStates * i + j + 1]

# ==============================================================================================

dt = (data2[1,0] - data2[0,0]) / fs_to_au

rate = np.zeros((len(PRt) - 1), dtype=float)
rate_0 = np.zeros((len(PRt_0) - 1), dtype=float)
for n in range(1, len(rate)):
    rate[n] = ((PRt[n] - PRt[n - 1]) / dt) / (1.0 - 2.0 * PRt[n])
for n in range(1, len(rate_0)):
    rate_0[n] = ((PRt_0[n] - PRt_0[n - 1]) / dt) / (1.0 - 2.0 * PRt_0[n])

time_0 = data1[:,0]
time = data2[:,0]

# plot the time-dependent rate constant, unit = fs^-1
plt.semilogx(time[0:-1] / fs_to_au, rate / ratio, "-", linewidth = lw, color = 'blue', label = "Inside resonant cavity")
plt.semilogx(time_0[0:-1] / fs_to_au, rate_0 / ratio, "-", linewidth = lw, color = 'black', label = "Bare molecule")
# x and y range of plotting 
time = 50000
y1, y2 = - 0.6, 1.2     # y-axis range: (y1, y2)

# scale for major and minor locator
# x_major_locator = MultipleLocator(300)
# x_minor_locator = MultipleLocator(60)
y_major_locator = MultipleLocator(1)
y_minor_locator = MultipleLocator(0.5)

# x-axis and LHS y-axis
ax = plt.gca()
ax.yaxis.set_major_locator(y_major_locator)
ax.yaxis.set_minor_locator(y_minor_locator)
ax.tick_params(which = 'major', length = 15, labelsize = 30, pad = 10)
ax.tick_params(which = 'minor', length = 5)

x1_label = ax.get_xticklabels()
[x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
y1_label = ax.get_yticklabels()
[y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]

plt.tick_params(labelsize = 30, which = 'both', direction = 'in')
plt.xlim(100, time)
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

ax.set_xlabel(r'time (fs)', size = 24)
ax.set_ylabel(r'$k\ (\times 10^{-5}\ \mathrm{fs}^{-1})$', size = 24)
ax.legend(frameon = False, loc = 'lower right', prop = font_legend, markerscale = 1)
plt.legend(title = '(c)', frameon = False, title_fontsize = legendsize)


# plt.show()
plt.savefig("figure_flux_side_additional.pdf", bbox_inches='tight')

