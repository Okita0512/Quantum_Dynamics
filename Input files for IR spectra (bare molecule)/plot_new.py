import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as sc
fig, ax = plt.subplots()
from gen_input import parameters

# ================= global ====================

conv = 27.211397                            # 1 a.u. = 27.211397 eV
fs_to_au = 41.341374575751                  # 1 fs = 41.341 a.u.
cm_to_au = 4.556335e-06                     # 1 cm^-1 = 4.556335e-06 a.u.
au_to_K = 3.1577464e+05                     # 1 au = 3.1577464e+05 K
kcal_to_au = 1.5936e-03                     # 1 kcal/mol = 1.5936e-3 a.u.

# ==============================================================================================

data = np.loadtxt("resp1st.w1", dtype = float)
data2 = np.loadtxt("resp1st_im.w", dtype = float)

plt.plot(data / cm_to_au, data2, '-')
plt.vlines([1172], 0, 400, colors = ['k'], linestyles = ["dashed"])

plt.xlim(600, 1800)
# plt.ylim(0.0, 1.2e6)

plt.savefig("IR.png")

