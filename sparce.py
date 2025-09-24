import scipy
import matplotlib.pyplot as plt
import math
import numpy as np

def cappend(i, j, d):
    l = i * nx + j
    row.append(len(rhs))
    col.append(l)
    data.append(d)

row = []
col = []
rhs = []
data = []
nx = 50    
nt = 10
L = 1.0
T = 10.0
dt = T / nt
alpha = 0.01
sigma  = 0.2

dx = 2*L / (nx - 1)
dt = T / nt

for i in range(nt):
    for j in range(nx):
        cappend(0, j, 1)
