import numpy as np
import matplotlib.pyplot as plt
from Radial_solitons import r, R


R = np.append(R, np.zeros(2))
r = np.append(r, np.linspace(r[-1], r[-1]+10, 2))

p = np.linspace(0, 2*np.pi, 50)

Rmesh , pmesh = np.meshgrid(R, p)
rmesh, pmesh = np.meshgrid(r, p)
  
X, Y  = rmesh*np.cos(pmesh), rmesh*np.sin(pmesh)


fig = plt.figure(figsize=(9, 10))
ax = plt.axes(projection ='3d')
ax.plot_surface(X, Y, Rmesh, cmap ='viridis', edgecolor ='black')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("solution profile for t=0")
ax.set_title("Solution in 2D")


x, y = np.linspace(-20, 20, 100), np.linspace(-20, 20, 100)

X, Y = np.meshgrid(x, y)

def analytic_2D(x, y, w): return 2*np.sqrt(w)*(np.cosh(np.sqrt(w)*(x+y)))**(-1)

Z = analytic_2D(X, Y, 1)

fig2 = plt.figure(figsize=(9, 10))
ax = plt.axes(projection ='3d')
ax.plot_surface(X, Y, Z, cmap ='viridis', edgecolor ='black')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("analytical solution for t=0")
ax.set_title("Solution in 2D")



