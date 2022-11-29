import multiprocessing as mp
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
from scipy.integrate import trapz


k = 0.3   #wavenumber
ac = 1     #cubic nonlinearity parameter

e = 0.1   #amplitude (0< e << 1)

#NLS parameters:
w = np.sqrt(k**2 + 1)    #frequency 
c = k/w                  #group velocity

v1, v2, v3 = 2*w, 1-c**2 ,  3*ac
gamma = 0.5
A, B = np.sqrt(2*gamma*v1/v3) , np.sqrt(gamma*v1/v2) 

### Functions: 
sech = lambda x : 1/np.cosh(x)

#the envelope of the equation at the beginning:
def envelope(X, T, e):
    return A*sech(B*X)*np.exp(1j * gamma * T)

def envelope2(X, e):
    return 0.25*A*(1-np.tanh(B*(X-11*0.5))*np.tanh(B*(X+11*0.5)))


#setting up the x array:
Lx = 1000           #length
dx = 0.02          #space discretisation

Nx =int(2*Lx/dx)   #number of points

x = 2*Lx*np.arange(-int(Nx/2), int(Nx/2))/Nx    #x array
dx = x[1]-x[0]                                  #actual space discretisation


k = 2*np.pi*fft.fftfreq(Nx, d=dx)
k2= k**2

def Strang_splitting(u, dt):
    
    u_hat = fft.fft(u)
    w_hat = np.exp(1j*(k2*(v2/v1)-c*k)*0.5*dt)*u_hat 
    
    #w_hat = np.exp(1j*k2*(v2/v1)*0.5*dt)*u_hat
    w     = fft.ifft(w_hat)
    
    wnew = np.exp(-1j*(v3/v1)*dt*(abs(w)*e)**2)*w
    
    wnew_hat = fft.fft(wnew)
    unew_hat = np.exp(1j*(k2*(v2/v1)-c*k)*0.5*dt)*wnew_hat
    
    #unew_hat = np.exp(1j*k2*(v2/v1)*0.5*dt)*wnew_hat
    unew     = fft.ifft(unew_hat) 
    
    return unew


tf = 12
dt= 0.01

t = 0
u = envelope2(x, e)
n = 0

while t < tf:
    
    u = Strang_splitting(u, dt)
    t +=dt
    
    if n%200 == 0:
        plt.figure(n)
        plt.ylim(-1, 1)
        plt.xlim(-20+c*t, 20+c*t)
        plt.plot(x, np.real(u) )
        plt.title(f"{t}")
        plt.show()
    n += 1 





x= np.linspace(-10, 10, 500)

Nx, dx = len(x), x[1]-x[0]

  
  
"""  
def gauss(x):
    return np.exp(-x**2)

k = 2*np.pi*fft.fftfreq(Nx, d=dx)
f_hat = fft.fft(gauss(x))
df_hat = 1j*k*f_hat
df = np.real(fft.ifft(df_hat))


plt.plot(x, gauss(x), x, df)   

a = fft.fft(gauss(x))[0]
print(a)


I = trapz(gauss(x), x)
I2 = trapz(df, x)
print(I)
print(I2)


"""







    
        
        
        
    
    
    
