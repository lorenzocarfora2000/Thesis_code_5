import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from time import time as clock
import multiprocessing as mp
from itertools import product
from scipy import fft
from scipy.stats import linregress as lin_fit
from scipy.integrate import trapz
import the_functions as funcs

k = 0.3   #wavenumber
ac = 1     #cubic nonlinearity parameter

#NLS parameters:
w = np.sqrt(k**2 + 1)    #frequency 
c = k/w                  #group velocity

v1, v2, v3 = 2*w, 1-c**2 ,  3*ac
gamma = 0.5
A, B = np.sqrt(2*gamma*v1/v3) , np.sqrt(gamma*v1/v2) 

###########
#CONSISTENCY FOR dx
###########

## spatial array

Lx = 1000
dt = 0.01
t0, T = 0., 1

e_list = np.linspace(0.04, 0.1, 4)
e_list0 = np.append(0, e_list)

#returns the superior error for a given dx and e value
def dx_consistency(vals):
    dx, e = vals
    Nx =int(2*Lx/dx)

    x = 2*Lx*np.arange(-int(Nx/2), int(Nx/2))/Nx
    dx = x[1]-x[0]
    
    #dt = dx+ 0.005
    
    vals = A, B, gamma, e, k, w, c, v1, v2, v3
    vals_NLS = vals[:7]
    
    u0 =    e*funcs.NLS_approx(x, 0, vals_NLS, funcs.soliton)
    v0 =    e*funcs.NLS_approx_dt(x, 0, vals, funcs.soliton)
 
    t, tf, u, v = 0., (T/e**2), u0, v0
     
    superior_error = 0
    superior_H_error = 0
     
    vals_rk4 = [dx, ac]
    while t < tf:
    
        t, u, v = funcs.RK4_NLKG(t, u, v, dt, vals_rk4)

        u_approx = e*funcs.NLS_approx(x, t, vals_NLS, funcs.soliton)
        
        error = max(abs(u-u_approx))
        H_error = funcs.Hp_norm(u-u_approx, 1, x) #sobolev norm error
    
        if error> superior_error: superior_error = error 
        if H_error> superior_H_error: superior_H_error = H_error 
    
    print(f"done with {dx, e}")
    return superior_error, superior_H_error

#superior error returned based on dx and e 


#returns the fitting parameters for a given list 
#of superior values over epsilon
def fitter(sup_err_list):
    sup_err_list0 = np.append(0, sup_err_list)

    fitter = lambda x, C, b: C*x**b

    popt, pcov = curve_fit(fitter, e_list0, sup_err_list0)
    C , b = popt
    return C, b

if __name__ == '__main__':
    
    dx_n = 3
    dx_list = np.linspace(0.02, 0.05, dx_n)
    
    couple = list(product(dx_list, e_list))
    
    i = clock()
    
    p = mp.Pool()
    result = p.map(dx_consistency, couple)
    p.close()
    p.join()
    
    result = np.array(result)
    
    sup_err_res, sup_H_err_res = np.array(result).T
    
    sup_err_lists_set = np.split(sup_err_res, dx_n)
    sup_H_err_lists_set = np.split(sup_H_err_res, dx_n)
    
    f = clock()
    
    print(f-i)
    
    
    #calculate fitting parameters for each set
    
    p = mp.Pool()
    CB_parameters = p.map(fitter, sup_err_lists_set)
    p.close()
    p.join()

    C_list, b_list = np.array(CB_parameters).T
    
    Ch = np.mean(C_list).round(2)    
    bh = np.mean(b_list).round(2)
    
    #only intercept in value at dx = 0
    ip = lin_fit(dx_list, b_list)[1]
    
    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.set_title("stability of fit parameters")
    ax1.plot(dx_list, C_list, "b.-", label = "C = {}".format(Ch))
    ax1.set_ylim(Ch - 0.1, Ch + 0.1)
    ax1.set(ylabel= "C")
    ax1.legend()
    ax2.plot(dx_list, b_list, "r.-", label = "b = {}, b = {} for dx=0".format(bh, round(ip, 2)))
    ax2.set_ylim(bh - 0.1, bh + 0.1)
    ax2.set(xlabel = "dx", ylabel= "b")
    ax2.legend()
    
    #repeat procedure, but for H1 sobolev norm
    p = mp.Pool()
    CB_parameters = p.map(fitter, sup_H_err_lists_set)
    p.close()
    p.join()

    C_list, b_list = np.array(CB_parameters).T
    
    Ch = np.mean(C_list).round(2)    
    bh = np.mean(b_list).round(2)
    
    #only intercept in value at dx = 0
    ip = lin_fit(dx_list, b_list)[1]
    
    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.set_title("stability of fit parameters, H1 Sobolev norm")
    ax1.plot(dx_list, C_list, "b.-", label = "C = {}".format(Ch))
    ax1.set_ylim(Ch - 0.1, Ch + 0.1)
    ax1.set(ylabel= "C")
    ax1.legend()
    ax2.plot(dx_list, b_list, "r.-", label = "b = {}, b = {} for dx=0".format(bh, round(ip, 2)))
    ax2.set_ylim(bh - 0.1, bh + 0.1)
    ax2.set(xlabel = "dx", ylabel= "b")
    ax2.legend()
