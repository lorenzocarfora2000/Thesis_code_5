import numpy as np
import matplotlib.pyplot as plt
from time import time as clock
from scipy import fft
from scipy.integrate import trapz
import the_functions as funcs


k = 0.3    #wavenumber
ac = 1     #cubic nonlinearity parameter

e = 0.1    #amplitude (0< e << 1)


#NLS parameters:
w = np.sqrt(k**2 + 1)    #frequency 
c = k/w                  #group velocity

#NLS parameters
v1, v2, v3 = 2*w, 1-c**2 ,  3*ac
gamma = 0.5
A, B = np.sqrt(2*gamma*v1/v3) , np.sqrt(gamma*v1/v2) 

vals = A, B, gamma, e, k, w, c, v1, v2, v3

vals_env = vals[:3]
vals_NLS = vals[:7]

## spatial array
Lx = 1000
dx = 0.02

Nx =int(2*Lx/dx)

x = 2*Lx*np.arange(-int(Nx/2), int(Nx/2))/Nx
dx = x[1]-x[0]


#####################
#TIME EVOLUTION SIMULATOR (RK4)
#####################

t0, tf = 0., 100      #initial and final time
dt = 0.025             #desired timestep

#initial conditions
env0  = 2*e*funcs.soliton(e*x, 0, vals_env)
u0 =    e*funcs.NLS_approx(x, 0, vals_NLS, funcs.soliton)
v0 =    e*funcs.NLS_approx_dt(x, 0, vals, funcs.soliton)

#fourier k-space
kx = 2*np.pi*fft.fftfreq(Nx, d=dx)

plotnum = 1       #plot number
plotgap = 200     #distance between plots


#calculation of spatial derivative done via spectral differentiation
u0_hat = fft.fft(u0) 
u0x    = fft.ifft(1j*kx*u0_hat) 

#Kinetic, potential and strain energies of NLKG. 
#The total energy is their sum

KE0     = 0.5*abs(v0)**2
STRAIN0 = 0.5*abs(u0x)**2 + 0.5*abs(u0)**2
POT0    = 0.25*abs(u0)**4  
             
KE0     = trapz(KE0, x)                   #fft.fft(KE0)[0]
POT0    = trapz(POT0, x)                  #fft.fft(POT0)[0]
STRAIN0 = trapz(STRAIN0, x)               #fft.fft(STRAIN0)[0]
E_TOT0  = KE0 + POT0 + STRAIN0

KE_list     = [KE0]
POT_list    = [POT0]
STRAIN_list = [STRAIN0]
E_TOT_list  = [E_TOT0]


#### NLS approx energy
E_NLS0 = (v2*abs(env0)**2 + ac*v3*0.5*abs(env0)**4)/v1
E_NLS0 = trapz(E_NLS0, e*x)                 #fft.fft(E_NLS0)[0]
E_NLS_list = [E_NLS0]

#H5 norm check
H5_norm_list = [funcs.Hp_norm(env0, 5, e*x)]

time = [0]


#biggest error achieved by the approximation
superior_error = 0

#start conditions
t, u, v = t0, u0, v0
n = 0

ti = clock()


vals_rk4 = [dx, ac]

while t < tf:
    t, u, v = funcs.RK4_NLKG(t, u, v, dt, vals_rk4)

    u_approx = e*funcs.NLS_approx(x, t, vals_NLS, funcs.soliton)
    
    error = max(abs(u-u_approx))

    if error> superior_error: superior_error = error 

    if (n%plotgap)==0:
        
        uexact= np.real(2*e*A*funcs.sech(B*e*(x-c*t)))
        
        plt.figure(plotnum)
        
        plt.plot(x, np.real(u),'r-', label = 'numerical')
        plt.plot(x, np.real(u_approx), "g--", label = 'nls-approx')
        plt.plot(x, uexact, 'b--', label="envelope")
        plt.plot(x, -uexact, 'b--')
        plt.legend(loc='best')
        plt.title("cubic NLKG, e={}, time = {}".format(e, t))
        plt.xlabel("x")
        plt.ylabel("u") 
        plt.xlim(c*t-200, c*t + 200)
        plt.show()
        
        u_hat = fft.fft(u) 
        ux    = fft.ifft(1j*kx*u_hat) 
        
        KE     = 0.5*abs(v)**2
        STRAIN = 0.5*abs(ux)**2 + 0.5*abs(u)**2
        POT    = 0.25*abs(u)**4  
             
        KE     = trapz(KE, x)          #fft.fft(KE0)[0]
        POT    = trapz(POT, x)         #fft.fft(POT0)[0]
        STRAIN = trapz(STRAIN, x)      #fft.fft(STRAIN0)[0]
        E_TOT  = KE + POT + STRAIN
        
        KE_list.append(KE)     
        POT_list.append(POT)       
        STRAIN_list.append(STRAIN)   
        E_TOT_list.append(E_TOT)    
        time.append(t)
        
        
        T = t*(e)**2
        X = e*(x - c*t)
        
        env = 2*e*funcs.soliton(X, T, vals_env)
        
        E_NLS = (v2*abs(env)**2 + ac*v3*0.5*abs(env)**4)/v1
        E_NLS = trapz(E_NLS, X)              #fft.fft(E_NLS)[0]
        E_NLS_list.append(E_NLS)
        
        H5_norm_list.append(funcs.Hp_norm(env, 5, X))
        
        plotnum+= 1
    n += 1

tf = clock()

print("time taken = {} s".format(tf-ti))

print(superior_error)

KE_list = np.array(KE_list).real    
POT_list = np.array(POT_list).real   
STRAIN_list = np.array(STRAIN_list).real     
E_TOT_list = np.array(E_TOT_list).real      

#energy plotting
plt.figure(plotnum)
plt.title("cubic NLKG Energy conservation over time")
plt.plot(time, KE_list, "b.-",       label = "kinetic" )
plt.plot(time, POT_list , "b--",     label = "potential")
plt.plot(time, STRAIN_list, "g-.",   label = "strain" )
plt.plot(time, E_TOT_list, "r-",    label = "total" )
plt.legend(loc = "best")
plt.xlabel("time t")
plt.ylabel("Energy")
plt.grid(True)
plt.show()

E_change = np.log(abs(1-E_TOT_list[1:]/E_TOT0))

plt.figure(plotnum+1)
plt.title("cubic NLKG Total energy change")
plt.plot(time[1:], E_change)
plt.grid(True)
plt.xlabel("time")
plt.ylabel("Energy change")

time_slow = np.array(time)*(e**2)

plt.figure(plotnum+2)
plt.title("NLS energy, stored in the envelope function")
plt.plot(time_slow, E_NLS_list, label = "NLS energy")
plt.xlabel("slow time T")
plt.ylabel("Energy")
plt.legend(loc= "best")
plt.grid(True)

plt.figure(plotnum+3)
plt.title("H^5 Sobolev norm consistency of envelope")
plt.plot(time_slow, H5_norm_list)
plt.grid(True)
plt.xlabel("slow time T")
plt.ylabel("H^5 norm")








