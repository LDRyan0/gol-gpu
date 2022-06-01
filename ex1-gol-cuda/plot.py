# Plots GOL results from performance.txt with following csv format:

# DATA
#   0,1,2    ,3          ,4          ,5      ,6
#   n,m,steps,cpu_time_ms,gpu_time_ms,speedup,kernel_percent

import numpy as np
import matplotlib.pyplot as plt

# Read data
data = np.genfromtxt("performance.txt", delimiter=",", skip_header=1)
n = data[:,0]
m = data[:,1]
steps = data[:,2]
cpu_times_s = data[:,3] / 1000 # convert ms to s
gpu_times_s = data[:,4] / 1000 # convert ms to s
speedups = data[:,5]
kernel_percents = data[:,6]

# Plots CPU and GPU times
fig, ax = plt.subplots()
ax.loglog(n,cpu_times_s, label="CPU", color="blue")
ax.loglog(n,gpu_times_s, label="GPU", color="lime")
ax.legend()
ax.set_xlabel("size of grid")
ax.set_xticks(n)
ax.set_ylabel("time (s)")
plt.savefig("cpu_gpu_times.png")

# Plot speedups
fig, ax = plt.subplots()
ax.set_xticks(n)
ax.semilogx(n,speedups)
ax.set_xlabel("size of grid")
ax.set_ylabel("speedup")
plt.savefig("speedups.png")

# Plot kernel/total execution time percentages
fig, ax = plt.subplots()
ax.set_xticks(n)
ax.semilogx(n,kernel_percents)
ax.set_xlabel("size of grid")
ax.set_ylabel("kernel execution time (%)")
ax.set_ylim([0, 100])
plt.savefig("kernel_percents.png")
