import numpy as np
from scipy import fft
from scipy.integrate import trapz


#exponential function
Exp = lambda x, t, k, w : np.exp(1j*(k*x - w*t))

#hyperbolic secant function
sech = lambda x : 1/np.cosh(x)


#soliton analytical solution of the NLS
def soliton(X, T, vals):
    A, B, gamma = vals
    return A*sech(B*X)*np.exp(1j * gamma * T)

#NLKG equation approximation using the soliton solution:
def NLS_approx(x, t, vals, envelope):
    #amplitude and group velocity
    A, B, gamma, e, k, w, c = vals
    
    #defining slow space and time coords
    X = e*(x- c*t)
    T = t*(e**2)
    
    env = envelope(X, T, [A, B, gamma])
    f = env*Exp(x, t, k, w)
    
    return f + np.conj(f) 

#time derivative of the NLKG approximation using soliton solution
def NLS_approx_dt(x, t, vals, envelope):
    A, B, gamma, e, k, w, c, v1, v2, v3 = vals
    
    #slow space and time coords
    X = e*(x- c*t)
    T = t*(e**2)
    
    ### first and second derivative of soliton envelope
    """
    env = envelope(X, T, [A, B, gamma])
    env_x = -B*env*np.tanh(B*X)
    env_xx = (B**2)*env*(np.tanh(B*X)**2 - sech(B*X)**2)
    """
    ### first and second derivative of envelope via fourier transform
    env = envelope(X, T, [A, B, gamma])
    env_hat = np.fft.fft(env)
    
    k1 = 2*np.pi*np.fft.fftfreq(len(x), (x[1]-x[0]))
    k2 = k**2
    
    env_x_hat, env_xx_hat = 1j*k1*env_hat, -k2*env_hat
    env_x = e*np.real(np.fft.ifft(env_x_hat))
    env_xx =  (e**2)*np.real(np.fft.ifft(env_xx_hat))
    
    
    U = Exp(x, t, k, w)*(-1j*w*env - e*c*env_x + 1j*(v2*env_xx + v3*env*abs(env)**2)*(e**2)/v1  )
    
    return U + np.conj(U) 

# L_p norm, requires function u, degree p and xarray x
def Lp_norm(u, p, x): return trapz( abs(u)**p , x)**(1/p)
    
# Sobolev H_p norm
def Hp_norm(u, p, x):
    Nx, dx = len(x), x[1]-x[0] 
    
    u_hat = fft.fft(u)
    kappa = 2*np.pi*fft.fftfreq(Nx, d=dx)
    
    I = 0
    for n in range(p+1):
        
        if n == 0: I += Lp_norm(u, 2, x)**2
        
        else:
            du_hat = ((1j*kappa)**p)*u_hat
            du = fft.ifft(du_hat)
            I += Lp_norm(du, 2, x)**2
            
    return np.sqrt(I)   

#second spatial derivative function, using the discretisation in physical space
def Del(u, dx): 
    return (np.roll(u, -1) -2*u + np.roll(u, 1))/dx**2

#nonlinear cubic Klein gordon equation:
def NLKG(u, vals):
    dx, ac = vals
    dvdt = Del(u, dx) - u + ac*u**3
    return dvdt

#Rk4 routine for the NLKG equation
def RK4_NLKG(t, u, v, dt, vals):
    
    #RK4 routine
    k1_v = dt*NLKG(u, vals)
    k1_x = dt*v
    k2_v = dt*NLKG(u+k1_x/2, vals)
    k2_x = dt*(v+k1_v/2)
    k3_v = dt*NLKG(u +k2_x/2, vals)
    k3_x = dt*(v + k2_v/2)
    k4_v = dt*NLKG(u + k3_x, vals)
    k4_x = dt*(v + k3_v)

    vnew = v + k1_v/6 + k2_v/3 + k3_v/3 + k4_v/6 #new v and pos vals
    unew = u +k1_x/6 + k2_x/3 + k3_x/3 + k4_x/6
    tnew = t+dt
    return tnew, unew, vnew



#for the NLS equation calculation
def Strang_splitting(u, dt, vals):
    
    e, c, v1, v2, v3, k1, k2 = vals
    """
    u_hat = fft.fft(u)
    w_hat = np.exp(1j*(k2*(v2/v1)-c*k1)*0.5*dt)*u_hat 
    
    #w_hat = np.exp(1j*k2*(v2/v1)*0.5*dt)*u_hat
    w  = fft.ifft(w_hat)
    
    wnew = np.exp(-1j*(v3/v1)*dt*(e*abs(w))**2)*w
    
    wnew_hat = fft.fft(wnew)
    unew_hat = np.exp(1j*(k2*(v2/v1)-c*k1)*0.5*dt)*wnew_hat
    
    #unew_hat = np.exp(1j*k2*(v2/v1)*0.5*dt)*wnew_hat
    unew     = fft.ifft(unew_hat) 
    """
    #######
    
    for _ in range(4):
        
        
        u_hat = fft.fft(u)
        w_hat = np.exp(1j*(k2*(v2/v1)-c*k1)*0.25*dt)*u_hat 
        
        #w_hat = np.exp(1j*k2*(v2/v1)*0.5*dt)*u_hat
        w  = fft.ifft(w_hat)
        
        u = np.exp(-1j*(v3/v1)*0.25*dt*(e*abs(w))**2)*w
    
    return u




