import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

k = 0.3   #wavenumber
ac = 1     #cubic nonlinearity parameter

e = 0.1   #amplitude (0< e << 1)


#NLS parameters:
w = np.sqrt(k**2 + 1)    #frequency 
c = k/w                  #group velocity

v1, v2, v3 = 2*w, 1-c**2, (9*ac - 2*(k**4) )/3
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
    return np.real(f + np.conj(f))*0.5

def NLS_approx_dt(x, t, e):
    env = envelope(x, t, e)
    X = e*(x - c*t)
    
    env_x = -B*env*np.tanh(B*X )
    env_xx = (B**2)*env*(np.tanh(B*X)**2 - sech(B*X)**2)
    
    """
    Ahat = np.fft.fft(A)
    k = 2*np.pi*np.fft.fftfreq(Nx, (x[1]-x[0]))
    k2 = k**2
    
    dAhat, d2Ahat = 1j*k*Ahat, -k2*Ahat
    Ax = e*np.real(np.fft.ifft(dAhat))
    Axx =  (e**2)*np.real(np.fft.ifft(d2Ahat))
    """
    U = Exp(x, t)*( -1j*w*env - e*c*env_x + 1j*(v2*env_xx + v3*env*abs(env)**2)*(e**2)/v1  )
    
    
    return np.real(U + np.conj(U) )*0.5


## array
Nx = 2000 #int(Lx/dx)  #number of points in x space
Lx = 200


x = 2*np.arange(-int(Nx/2), int(Nx/2))*Lx/Nx

y  =  np.real(e*envelope(x, 0, e))
u0 = e*NLS_approx(x, 0, e)
v0 = e*NLS_approx_dt(x, 0, e)
plt.figure(0)
plt.title("initial condition, based on NLS solution")
plt.plot(x, y, label = "envelope")
plt.plot(x, u0, label = "NLS approximation")
plt.plot(x, v0, label = "NLS initial velocity")
plt.legend(loc = "best")
plt.grid(True)

#####################
#TIME EVOLUTION SIMULATOR (RK4)
#####################

t0, tf = 0., 10#initial and final time
dt = 0.005       #desired timestep


#the following funciton works as a delta operator, where the array x is an
#array for a function in space.

dx = x[1]-x[0]
Del = lambda u:  (np.roll(u , -1) -2*u + np.roll(u, 1))/dx**2

#the following function returns the second time derivative of the system
def dvdt(u, v):
    dvdt = Del(u) - u +0.5*Del(u**2) + ac*(u**3) #quasilinear
    return dvdt

t, u, v = t0, u0, v0
        
plotnum = 1      #plot number
plotgap = 100   #distance between plots

#initial energy

#calculation of spatial derivative is done via spectral differentiation
kx = np.fft.fftfreq(Nx)

u0_hat = np.fft.fft(u0) 
u0x= np.real(np.fft.ifft(kx*u0_hat)) 

u02_hat = np.fft.fft(u0**3) 
u02x= np.real(np.fft.ifft(kx*kx*u02_hat)) 

#Kinetic, potential and strain energies. The total energy is their sum
KE0 = 0.5*abs(v0)**2
POT0 = 0.5*abs(u0)**2 - 0.25*abs(u0)**4  
STRAIN0= 0.5*abs(u0x)**2 - 0.25*abs(u02x)          
          
KE0= np.fft.fft(KE0)[0]
POT0 = np.fft.fft(POT0)[0]
STRAIN0 = np.fft.fft(STRAIN0)[0]
E_TOT0 = KE0 +  POT0 + STRAIN0

Momentum0 = np.fft.fft(u0x*v0)[0]

KE_list     = [KE0]
POT_list    = [POT0]
STRAIN_list = [STRAIN0]
E_TOT_list  = [E_TOT0]
Momentum_list = [Momentum0]

time = [0]

#biggest error achieved by the approximation
superior_error = 0

n = 1 

while t < tf:
    #RK4 routine
    k1_v = dt*dvdt(u, v)
    k1_x = dt*v
    k2_v = dt*dvdt(u+k1_x/2, v+k1_v/2 )
    k2_x = dt*(v+k1_v/2)
    k3_v = dt*dvdt(u +k2_x/2, v+k2_v/2 )
    k3_x = dt*(v + k2_v/2)
    k4_v = dt*dvdt(u + k3_x, v + k3_v )
    k4_x = dt*(v + k3_v)
    
    v = v + k1_v/6 + k2_v/3 + k3_v/3 + k4_v/6 #new v and pos vals
    u = u +k1_x/6 + k2_x/3 + k3_x/3 + k4_x/6
    t = t+dt

    u_approx = e*NLS_approx(x, t, e)
    
    error = max(abs(u-u_approx))

    if error> superior_error: superior_error = error 

    if (n%plotgap)==0:
        
        uexact= e*envelope(x, t, e)
        
        plt.figure(plotnum)
        
        plt.plot(x, np.real(u),'r-', label = 'numerical')
        plt.plot(x, u_approx, "g--", label = 'nls-approx')
        plt.plot(x, np.real(uexact), 'b--', label="envelope")
        plt.plot(x, -np.real(uexact), 'b--')
        plt.legend(loc='best')
        plt.title("time = {}".format(t))
        plt.xlabel("x")
        plt.ylabel("u") 
        plt.show()
        
        #energy calculation
        u_hat = np.fft.fft(u) 
        ux= np.real(np.fft.ifft(kx*u_hat)) 

        u2_hat = np.fft.fft(u**3) 
        u2x= np.real(np.fft.ifft(kx*kx*u2_hat)) 

        #Kinetic, potential and strain energies. The total energy is their sum
        KE = 0.5*abs(v)**2
        POT = 0.5*abs(u)**2 - 0.25*abs(u)**4  
        STRAIN= 0.5*abs(ux)**2 - 0.25*abs(u2x)            
                  
        KE      = np.fft.fft(KE)[0]
        POT     = np.fft.fft(POT)[0]
        STRAIN  = np.fft.fft(STRAIN)[0]
        E_TOT   = KE +  POT + STRAIN
        
        Momentum = np.fft.fft(ux*v)[0]
        
        KE_list.append(KE)     
        POT_list.append(POT)       
        STRAIN_list.append(STRAIN)   
        E_TOT_list.append(E_TOT)    
        time.append(t)
        Momentum_list.append(Momentum)
        
        plotnum+= 1
    n += 1

print(superior_error)

KE_list = np.array(KE_list).real    
POT_list = np.array(POT_list).real   
STRAIN_list = np.array(STRAIN_list).real     
E_TOT_list = np.array(E_TOT_list).real      

#energy plotting
plt.figure(plotnum+1)
plt.plot(time, KE_list, "b.-",       label = "kinetic" )
plt.plot(time, POT_list , "b--",     label = "potential")
plt.plot(time, STRAIN_list, "g-.",   label = "strain" )
plt.plot(time, E_TOT_list, "r-",    label = "total" )
plt.legend(loc = "best")
plt.xlabel("time t")
plt.ylabel("Energy")
plt.grid(True)
plt.title("Energy conservation over time")

#energy fluctuations are under O(4)
#Enchange =np.log(abs(1-E_TOT_list/E_TOT0))[1:]

#plt.figure(plotnum+2)
#plt.plot(time[1:], Enchange, "r-")
#plt.xlabel("time t")
#plt.ylabel("Energy change")

#plt.plot(time , Momentum_list); plt.ylim(2e-17, -2e-17)

#######
# ERROR CONSISTENCY   
########  
"""
t0, tf = 0., 20
dt = 0.005

dx = x[1]-x[0]
Del = lambda u: (np.roll(u , -1) -2*u + np.roll(u, 1))/dx**2


def dvdt(u, v):
    dvdt = Del(u) - u +0.5*Del(u**2) + ac*u**3
    #dvdt = Del(u) - u - u**3
    return dvdt

e_list = np.linspace(0.03, 0.1, 3)
sup_err_list = []


for e in e_list:
    u0 = e*NLS_approx(x, 0, e)
    v0 = e*NLS_approx_dt(x, 0, e)
    
    t, u, v = t0, u0, v0

    superior_error = 0

    while t < tf:

        k1_v = dt*dvdt(u, v)
        k1_x = dt*v
        k2_v = dt*dvdt(u+k1_x/2, v+k1_v/2 )
        k2_x = dt*(v+k1_v/2)
        k3_v = dt*dvdt(u +k2_x/2, v+k2_v/2 )
        k3_x = dt*(v + k2_v/2)
        k4_v = dt*dvdt(u + k3_x, v + k3_v )
        k4_x = dt*(v + k3_v)
    
        v = v + k1_v/6 + k2_v/3 + k3_v/3 + k4_v/6 #new v and pos vals
        u = u +k1_x/6 + k2_x/3 + k3_x/3 + k4_x/6
        t = t+dt
    
        u_approx = e*NLS_approx(x, t, e)
    
        error = max(abs(u-u_approx))
    
        if error> superior_error: superior_error = error 
    
    sup_err_list.append(superior_error)
    print("completed for e = {}".format(e))
    

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
plt.title("demonstration for dt ={}".format(dt)) 
plt.plot(x_fit, y_fit, "--", label = "fit = C*e^b, with C = {}, b = {}".format(C, b) )
plt.legend(loc = "best")
"""
###########
#CONSISTENCY FOR h
###########
"""
t0, tf = 0., 10
dt_list = np.linspace(0.005, 0.04, 8)

dx = x[1]-x[0]
Del = lambda u: (np.roll(u , -1) -2*u + np.roll(u, 1))/dx**2

def dvdt(u, v):
    dvdt = Del(u) - u +0.5*Del(u**2) + a*u**3
    return dvdt

e_list = np.linspace(0.03, 0.1, 5)
C_list = []
b_list = []

for dt in dt_list: 
    sup_err_list = []

    for e in e_list:
        u0 = e*NLS_approx(x, 0, e)
        v0 = e*NLS_approx_dt(x, 0, e)
    
        t, u, v = t0, u0, v0

        superior_error = 0

        while t < tf:

            k1_v = dt*dvdt(u, v)
            k1_x = dt*v
            k2_v = dt*dvdt(u+k1_x/2, v+k1_v/2 )
            k2_x = dt*(v+k1_v/2)
            k3_v = dt*dvdt(u +k2_x/2, v+k2_v/2 )
            k3_x = dt*(v + k2_v/2)
            k4_v = dt*dvdt(u + k3_x, v + k3_v )
            k4_x = dt*(v + k3_v)
    
            v = v + k1_v/6 + k2_v/3 + k3_v/3 + k4_v/6 #new v and pos vals
            u = u +k1_x/6 + k2_x/3 + k3_x/3 + k4_x/6
            t = t+dt
    
            u_approx = e*NLS_approx(x, t, e)
    
            error = max(abs(u-u_approx))
    
            if error> superior_error: superior_error = error 
    
        sup_err_list.append(superior_error)
        print("completed for e = {}".format(e))
    

    e_list0 = np.append(0, e_list)
    sup_err_list0 = np.append(0, sup_err_list)

    fitter = lambda x, C, b: C*x**b

    popt, pcov = curve_fit(fitter, e_list0, sup_err_list0)
    C , b = popt
    
    C_list.append(C) ; b_list.append(b)
    
C_mean = np.mean(C_list)    
b_mean = np.mean(b_list)

Ch, bh = np.round([C_mean, b_mean], 2)

fig, (ax1, ax2) = plt.subplots(2, 1)

ax1.set_title("stability of fit parameters")
ax1.plot(dt_list, C_list, "b.-", label = "C = {}".format(Ch))
ax1.set_ylim(C_mean - 0.1, C_mean + 0.1)
ax1.legend()
ax2.plot(dt_list, b_list, "r.-", label = "b = {}".format(bh))
ax2.set_ylim(b_mean - 0.1, b_mean + 0.1)
ax2.set(xlabel = "h")
ax2.legend()

"""
   