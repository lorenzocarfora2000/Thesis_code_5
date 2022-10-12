import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve as solve

def U(N):
    base  = np.zeros([N, N])
    for x in range(N):
        if x+1 in range(N):
            base[x, x+1] = 1
    return base
 
    
def D(N):
    base  = np.zeros([N, N])
    for x in range(N):
        if x+1 in range(N):
            base[x+1, x] = 1
    return base

N = 301

x = np.linspace(-10, 10, N)


def initial(x):
    return np.exp(-abs(x)**2)/np.pi

def potential(x):
    return x**2

def CN(x0, fx0, tf, V):
    """

    Parameters
    ----------
    x0 : the spatial array, describing which points in space are evaluated
    fx0 : the function describing the profile along x for t = 0
    tf : the final time, for which the routine will calculate the function
    V : function describing the potential

    Returns
    -------
    t_arr : array of time.
    X : space-time matrix describing the solution
    """
    
    
    N = len(x0)     #points to evaluate, as deduced from x0 
    Psi0 = fx0(x0)  #the initial conditions are calculated at said points
    
    I = np.identity(N)
    dx = dt = x0[1]-x0[0]   #dx and dt deduced from x0, imposing dt = dx
    
    Psi0[0], Psi0[-1] = 0, 0 #boundary conditions
    
    
    #Hamiltonian calculation, along with the needed functions for CN routine
    H = (2*I- U(N) - D(N))/(dx**2) + np.diag(V(x0))
    M1 = I- 0.5j*dt*H
    M2 = np.linalg.inv(I+0.5j*dt*H)
    #boundary conditions:
        
    M1[0], M1[-1] = np.zeros(N), np.zeros(N)
    M1[0, 0], M1[-1, -1] = 1, 1
    
    M2[0], M2[-1] = np.zeros(N), np.zeros(N)
    M2[0, 0], M2[-1, -1] = 1, 1 
    
    #building the return terms
    X = np.matrix(Psi0)
    t_arr = np.linspace(0, tf, N)
    
    for t in t_arr[1:] :
        
        Psi_x = M1*(X[-1, :].T)
        Psi_new = (M2*Psi_x).T
        
        Psi_new[0, 0], Psi_new[0, -1] = 0, 0
        
        X = np.vstack([X, Psi_new])

    return t_arr, X.T
    

sol = CN(x, initial, 1, potential)

t, psi  = sol[0], sol[1][:, :]


tmesh, xmesh = np.meshgrid(t, x)
fig = plt.figure(figsize=(9, 10))
 
# syntax for 3-D plotting
ax = plt.axes(projection ='3d')
 
# syntax for plotting
ax.plot_surface(tmesh, xmesh, abs(psi)**2, cmap ='viridis', edgecolor ='green')
ax.set_title("shrodinger equation")
ax.set(ylabel = "x space", xlabel= "time", zlabel = "wavefunction")
plt.show()

"""
def NLS(x0, fx0, tf, V):
    

    Parameters
    ----------
    x0 : the spatial array, describing which points in space are evaluated
    fx0 : the function describing the profile along x for t = 0
    tf : the final time, for which the routine will calculate the function
    V : function describing the potential

    Returns
    -------
    t_arr : array of time.
    X : space-time matrix describing the solution
    
    
    
    N = len(x0)     #points to evaluate, as deduced from x0 
    Psi0 = fx0(x0)  #the initial conditions are calculated at said points
    
    I = np.identity(N)
    dx = dt = x0[1]-x0[0]   #dx and dt deduced from x0, imposing dt = dx
    
    Psi0[0], Psi0[-1] = 0, 0 #boundary conditions
    
    
    #Hamiltonian calculation, along with the needed functions for CN routine
    H = (2*I- U(N) - D(N))/(dx**2) + np.diag(V(x0))
    M1 = I- 0.5j*dt*H
    M2 = (I+0.5j*dt*H)
    #boundary conditions:
        
    M1[0], M1[-1] = np.zeros(N), np.zeros(N)
    M1[0, 0], M1[-1, -1] = 1, 1
    
    M2[0], M2[-1] = np.zeros(N), np.zeros(N)
    M2[0, 0], M2[-1, -1] = 1, 1 
    
    #building the return terms
    X = np.matrix(Psi0)
    t_arr = np.linspace(0, tf, N)
    
    for t in t_arr[1:] :
        
        Psi_x = X[-1, :]
        
        f = lambda x: (M2+ I*0.5j*dt*abs(x)**2)*(x.T) - (M1 - I*0.5j*dt*abs(x)**2)*(Psi_x.T)
        
        Psi_x  = solve(f, ) 
        
        Psi_new = (M2*(M1*(Psi_x.T))).T
        Psi_new[0, 0], Psi_new[0, -1] = 0, 0
        
        X = np.vstack([X, Psi_new])

    return t_arr, X.T

"""
        
        
        
        
    
    
    
    
    
    








    






