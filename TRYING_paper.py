import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from time import time as clock
from numba import njit, jit
from scipy import fft

k = 0.3   #wavenumber
ac = 1     #cubic nonlinearity parameter

e = 0.1   #amplitude (0< e << 1)

#NLS parameters:
w = np.sqrt(k**2 + 1)    #frequency 
c = k/w                  #group velocity

v1, v2, v3 = 2*w, 1-c**2,  (9*ac - 2*(k**4) )/3
gamma = 0.5
A, B = np.sqrt(2*gamma*v1/v3) , np.sqrt(gamma*v1/v2) 

### Functions: 

Exp = lambda x, t : np.exp(1j*(k*x - w*t))
sech = lambda x : 1/np.cosh(x)

#the envelope follows the NLS equation
def envelope(x, t, e):
    X = e*(x- c*t)
    T = t*(e**2)
    return A*sech(B*X)*np.exp(1j * gamma * T)

#approximation of function and its derivative
def NLS_approx(x, t, e):
    env = envelope(x, t, e)
    f = env*Exp(x, t)
    return f + np.conj(f) 

def NLS_approx_dt(x, t, e):
    env = envelope(x, t, e)
    X = e*(x - c*t)
    
    env_x = -B*env*np.tanh(B*X )
    env_xx = (B**2)*env*(np.tanh(B*X)**2 - sech(B*X)**2)
    
    U = Exp(x, t)*( -1j*w*env - e*c*env_x + 1j*(v2*env_xx + v3*env*abs(env)**2)*(e**2)/v1  )
    
    return U + np.conj(U) 

## array
Lx = 1000
dx = 0.02

Nx =int(2*Lx/dx)

x = 2*Lx*np.arange(-int(Nx/2), int(Nx/2))/Nx
dx = x[1]-x[0]

y  =  2*np.real(e*envelope(x, 0, e))
u0 = e*NLS_approx(x, 0, e)
v0 = e*NLS_approx_dt(x, 0, e)


plt.figure(0)
plt.title("initial condition, based on NLS solution")
plt.plot(x, np.real(y), label = "envelope")
plt.plot(x, np.real(u0), label = "NLS approximation")
plt.plot(x, np.real(v0), label = "NLS initial velocity")
plt.xlim(-200, 200)
plt.legend(loc = "best")
plt.grid(True)
plt.show()


#####################
#TIME EVOLUTION SIMULATOR (RK4)
#####################
"""
t0, tf = 0., 0.5   #initial and final time
dt = 0.025          #desired timestep

#the following funciton works as a delta operator, where the array x is an
#array for a function in space.

dx = x[1]-x[0]

@njit
def Del(u): return (np.roll(u, -1) -2*u + np.roll(u, 1))/dx**2

#the following function returns the second time derivative of the system
@njit
def dvdt(u):
    dvdt = Del(u) - u +0.5*Del(u**2) + ac*(u**3) #quasilinear
    #dvdt = Del(u) - u - u**3
    return dvdt

t, u, v = t0, u0, v0
        
plotnum = 1      #plot number
plotgap = 300   #distance between plots


#biggest error achieved by the approximation
superior_error = 0

n = 1 

@njit
def RK4(t, u, v):
    #RK4 routine
    k1_v = dt*dvdt(u)
    k1_x = dt*v
    k2_v = dt*dvdt(u+k1_x/2)
    k2_x = dt*(v+k1_v/2)
    k3_v = dt*dvdt(u +k2_x/2)
    k3_x = dt*(v + k2_v/2)
    k4_v = dt*dvdt(u + k3_x)
    k4_x = dt*(v + k3_v)

    vnew = v + k1_v/6 + k2_v/3 + k3_v/3 + k4_v/6 #new v and pos vals
    unew = u +k1_x/6 + k2_x/3 + k3_x/3 + k4_x/6
    tnew = t+dt
    return tnew, unew, vnew


ti = clock()
while t < tf:
    
    t, un, v = RK4(t, u, v)

    u_approx = e*NLS_approx(x, t, e)
    
    error = max(abs(u-u_approx))

    if error> superior_error: superior_error = error 

    if (n%plotgap)==0:
        
        #uexact= np.real(e*envelope(x, t, e))
        
        plt.figure(plotnum)
        
        plt.plot(x, np.real(u),'r-', label = 'numerical')
        plt.plot(x, np.real(u_approx), "g--", label = 'nls-approx')
        #plt.plot(x, uexact, 'b--', label="envelope")
        #plt.plot(x, -uexact, 'b--')
        plt.legend(loc='best')
        plt.title("time = {}".format(t))
        plt.xlabel("x")
        plt.ylabel("u") 
        plt.xlim(c*t-200, c*t + 200)
        plt.show()

        
        plotnum+= 1
    n += 1

print(superior_error)
print(superior_error/e**2)

tf = clock()

print("time taken = {} s".format(tf-ti))
"""
#######
# ERROR CONSISTENCY   
########  

t0, T = 0., 0.5
dt = 0.025

dx = x[1]-x[0]
@njit
def Del(u): return (np.roll(u, -1) -2*u + np.roll(u, 1))/dx**2


@njit
def dvdt(u, v):
    dvdt = Del(u) - u +0.5*Del(u**2) + ac*u**3
    #dvdt = Del(u) - u - u**3
    return dvdt

e_list = np.linspace(0.04, 0.1, 3)
sup_err_list = []



def RK4(t, u, v):
    #RK4 routine
    k1_v = dt*dvdt(u)
    k1_x = dt*v
    k2_v = dt*dvdt(u+k1_x/2)
    k2_x = dt*(v+k1_v/2)
    k3_v = dt*dvdt(u +k2_x/2)
    k3_x = dt*(v + k2_v/2)
    k4_v = dt*dvdt(u + k3_x)
    k4_x = dt*(v + k3_v)

    vnew = v + k1_v/6 + k2_v/3 + k3_v/3 + k4_v/6 #new v and pos vals
    unew = u +k1_x/6 + k2_x/3 + k3_x/3 + k4_x/6
    tnew = t+dt
    return tnew, unew, vnew


for e in e_list:
    u0 = e*NLS_approx(x, 0, e)
    v0 = e*NLS_approx_dt(x, 0, e)
    
    t, u, v = t0, u0, v0
    tf = T/(e**2)

    superior_error = 0
    
    start = clock()
    while t < tf:
        
        t, u, v = RK4(t, u, v)
        
        u_approx = e*NLS_approx(x, t, e)
    
        error = max(abs(u-u_approx))
    
        if error> superior_error: 
            superior_error = error 
            #print("new error = {}".format(superior_error))
            #print("time = {}".format(t))
        
    end = clock()
    
    sup_err_list.append(superior_error)
    print("completed for e = {} in {} seconds".format(e, end-start))
    

e_list = np.append(0, e_list)
sup_err_list = np.append(0, sup_err_list)

fitter = lambda x, a, b: a*x**b

popt, pcov = curve_fit(fitter, e_list, sup_err_list)
x_fit = np.linspace(0, e_list[-1], 100)
y_fit = fitter(x_fit, *popt)

C, b = np.round(popt, 2)

plt.figure(3)
plt.plot(e_list, sup_err_list, ".", label="numerical")
plt.xlabel("epsilon, 0 < e <<1")
plt.ylabel("sup s(t)")
plt.title("demonstration for dt ={}, T_0 = {}".format(dt, T)) 
plt.plot(x_fit, y_fit, "--", label = "fit = C*e^b, with C = {}, b = {}".format(C, b) )
plt.legend(loc = "best")