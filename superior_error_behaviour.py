import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from time import time as clock
import multiprocessing as mp
from scipy import fft
from scipy.integrate import trapz
import the_functions as funcs

#calculation done for the soliton solution

k = 0.3   #wavenumber
ac = 1     #cubic nonlinearity parameter

#NLS parameters:
w = np.sqrt(k**2 + 1)    #frequency 
c = k/w                  #group velocity

v1, v2, v3 = 2*w, 1-c**2 ,  3*ac
gamma = 0.5
A, B = np.sqrt(2*gamma*v1/v3) , np.sqrt(gamma*v1/v2) 


## spatial array
Lx = 1000
dx = 0.02

Nx =int(2*Lx/dx)

x = 2*Lx*np.arange(-int(Nx/2), int(Nx/2))/Nx
dx = x[1]-x[0]

T = 1.
dt = 0.015

dx = x[1]-x[0]


vals_rk4 = [dx, ac]
def sup_err_calc(e):
    
    vals = A, B, gamma, e, k, w, c, v1, v2, v3
    vals_NLS = vals[:7]
    
    u0 =    e*funcs.NLS_approx(x, 0, vals_NLS, funcs.soliton)
    v0 =    e*funcs.NLS_approx_dt(x, 0, vals, funcs.soliton)
    
    t, u, v, tf = 0., u0, v0, T/(e**2)
    
    superior_error = 0
    superior_H_error = 0
    
    while t < tf:

        t, u, v = funcs.RK4_NLKG(t, u, v, dt, vals_rk4)

        u_approx = e*funcs.NLS_approx(x, t, vals_NLS, funcs.soliton)
        
        error = max(abs(u-u_approx))
        H_error = funcs.Hp_norm(u-u_approx, 1, x) #sobolev norm error
    
        if error> superior_error: superior_error = error 
        if H_error> superior_H_error: superior_H_error = H_error 
        
    print(f"done with {e}")
    return superior_error, superior_H_error


e_list = np.linspace(0.04, 0.1, 4)

if __name__ == '__main__':
    
    start = clock()
    
    p = mp.Pool()
    sup_err_set = p.map(sup_err_calc,  e_list)
    p.close()
    p.join()
    end = clock()
    
    sup_err_list, sup_H_err_list = np.array(sup_err_set).T
    
    print(f"time taken: {end-start} seconds")
    
    e_list = np.append(0, e_list)
    sup_err_list = np.append(0, sup_err_list)
    sup_H_err_list = np.append(0, sup_H_err_list)

    fitter = lambda x, a, b: a*x**b

    popt, pcov = curve_fit(fitter, e_list, sup_err_list)
    x_fit = np.linspace(0, e_list[-1], 100)
    y_fit = fitter(x_fit, *popt)

    C, b = np.round(popt, 2)

    plt.figure(3)
    plt.plot(e_list, sup_err_list, ".", label="numerical")
    plt.xlabel("epsilon, 0 < e <<1")
    plt.ylabel("sup s(t)")
    plt.title("error consistency for cubic NLKG, T0 ={}".format(T)) 
    plt.plot(x_fit, y_fit, "--", label = "fit = C*e^b, with C = {}, b = {}".format(C, b) )
    plt.legend(loc = "best")
    
    poptH, pcovH = curve_fit(fitter, e_list, sup_H_err_list)
    y_fit = fitter(x_fit, *poptH)

    CH, bH = np.round(poptH, 2)

    plt.figure(4)
    plt.plot(e_list, sup_H_err_list, ".", label="numerical")
    plt.xlabel("epsilon, 0 < e <<1")
    plt.ylabel("sup s(t), with sobolev norm H1")
    plt.title("error consistency for cubic NLKG, T0 ={}".format(T)) 
    plt.plot(x_fit, y_fit, "--", label = "fit = C*e^b, with C = {}, b = {}".format(CH, bH) )
    plt.legend(loc = "best")