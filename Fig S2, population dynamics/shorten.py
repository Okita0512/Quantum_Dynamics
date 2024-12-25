import numpy as np

data2 = np.loadtxt("prop-rho.dat", dtype = float)
data = np.zeros((int(len(data2[:, 0]) / 10), len(data2[0, :])), dtype = float)

count = 0
for i in range(len(data2)):
    if (i%10 == 0):
        data[count, :] = data2[i, :]
        count += 1

np.savetxt("prop-rho2.dat", data)