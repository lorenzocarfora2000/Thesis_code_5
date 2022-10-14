import numpy as np
import matplotlib.pyplot as plt


#Butcher tableau data:
rkf4 = np.array([25/216, 0., 1408/2565, 2197/4104, -1/5, 0.])
rkf5 = np.array([16/135 , 0. ,6656/12825 ,28561/56430 ,-9/50 ,2/55])
beta = np.array([[1/4, 0., 0., 0., 0., 0.],
                  [3/32, 9/32, 0., 0., 0., 0.], 
                  [1932/2197, -7200/2197, 7296/2197, 0., 0., 0.],
                  [439/216 ,-8., 3680/513, -845/4104, 0., 0.],
                  [-8/27, 2., -3544/2565, 1859/4104, -11/40, 0.]])
diff = rkf4 - rkf5

def RKF45(t0, tf, vec_0, values, dvdt):  
    
    """
    The store parameter dictates how long would the time array and vecs array
    supposed to be. It is here set to be 20000.
    
    the time array contains all the retrieved values of time. First element is 
    set to be the initial time t0.
    
    vecs is a matrix containing all the coordinates of the system over time. 
    The shape of the matrix is length x store, where length is the number of 
    coordinates evaluated in the system, and store the number of data points 
    collected over time. The length is imposed by referring to the initial 
    condition of the system at t0, vec_0. 
    """
    store = 20000
    
    time = np.zeros(store)
    time[0] = t0
    
    length = len(vec_0)
    vecs = np.zeros((length , store), dtype = complex)
    vecs[:, 0] = vec_0
    
    #variables of time and coordinates
    t0 , vec_0 = t0, vec_0
    
    #initial imposed stepsize & error tolerance
    h, epsilon_0 =  1e-2, 1e-8
    
    """
    pos takes count of the number of iterations calculated in the array.
    cut takes account of the number of times the data points have to be reduced 
    to be less than the imposed store.
    """
    pos = 1
    cut = 1
    while t0 < tf:
        
        #matrix to be filled with the k values:
        k = np.zeros((length, 6), dtype=complex)
        k[:, 0] =  h*dvdt(vec_0, values)
        
        for i in range(len(diff)-1):
            #successive k vals are calculated by using the data 
            #in the Butcher Tableau:
            sums = vec_0+ np.dot(k , beta[i])
            k[:, i+1]= h*dvdt(sums, values)
            
        #truncation error:
        eps = max([abs(np.dot(diff, k[i])) for i in range(len(k))])

        if eps > epsilon_0:    
            #new step calculated and procedure repeated
            h = 0.87*h*(epsilon_0/eps)**(1/5) 
            continue
        
        #when the truncation error is acceptable, derive successive point:
        vec_0 = vec_0 + np.dot(k, rkf5)
        t0 = t0+h
        
        #next timestep is derived
        h = 0.87*h*(epsilon_0/eps)**(1/5)
        
        #attach newly derived values to time and vecs, update pos
        time[pos] += t0
        vecs[:, pos] += vec_0
        pos += 1
        
        """
        The following if condition is started when pos becomes greater than 
        store, hence when the number of iterations is greater than the accepted
        number of data points. the time array and vecs matrix are enlarged, and
        the cut parameter is updated.
        """
        if pos%store == 0:
            time = np.append(time, np.zeros(store))
            vecs = np.hstack((vecs, np.zeros((length, store)) ))
            cut += 1
        
    #the "empty" spaces in the time array and vecs matrix are deleted:    
    time = time[:pos]
    vecs = vecs[:, :pos]
    
    #correcting overshooting, so the computed final time matches with tf:
    ta, tb, va, vb = time[-2], time[-1], vecs[:, -2], vecs[:, -1]
    vecs_cut, t_cut  = (va*(tf-tb) - vb*(tf - ta))/(ta-tb), tf
    
    #data cutting, the number of datat points is reduced by cut:
    time = time[::cut]
    vecs = vecs[:, ::cut]
    
    """
    the final coords and time calculated with the overshootinf correction are
    finally imposed, substituting the last computed data point.
    """
    time[-1] = t_cut
    vecs[:, -1] = vecs_cut
    
    return time, vecs

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

def Delta_D(N, dx, d):
    I = np.identity(N)
    Delta = np.matrix((2*I - U(N) - D(N))/(dx**2))+0j
    
    for i in range(d-1):
        Delta = np.kron(Delta, I) + np.kron(I, Delta)
    
    return Delta

def NLS(X, Delta):
    
    NL = np.diag( X*np.conj(X) )
    dPdt = np.matrix(1j*(-Delta+NL))*np.matrix(X).T
    
    return np.asarray(dPdt.T)[0]     #np.array(dPdt.T)[0]

    
d = 1 
w = 1
   
x = np.linspace(-7, 7, 70)
Psi0 = np.sqrt(2*w)/np.cosh(np.sqrt(w)*x)

N, dx = len(x), x[1]-x[0]


Delta = Delta_D(N, dx, d)

sol = RKF45(0, 10, Psi0, Delta, NLS)

t = sol[0]
Psi = sol[1]

Psi_squared = np.real(Psi)
tmesh, xmesh = np.meshgrid(t, x)

fig = plt.figure(figsize=(9, 10))
 
# syntax for 3-D plotting
ax = plt.axes(projection ='3d')
 
# syntax for plotting
ax.plot_surface(tmesh, xmesh, Psi_squared, cmap ='viridis', edgecolor ='green')
ax.set_title("1D Nonlinear Schrodinger Equation via Range-Kutta")
ax.set(ylabel = "x space", xlabel= "time", zlabel = "wavefunction (Real Part)")
plt.show()


def true_Psi(x, t, w): return (np.sqrt(2*w)/np.cosh(np.sqrt(w)*x))*np.exp(w*1j*t)

true = np.real(true_Psi(xmesh, tmesh, w))

fig2 = plt.figure(figsize=(9, 10))
ax2 = plt.axes(projection ='3d')
  
# syntax for plotting
ax2.plot_surface(tmesh, xmesh, true, cmap ='viridis', edgecolor ='green')
ax2.set_title("Analytical solution for NLSE ")
ax2.set(ylabel = "x space", xlabel= "time", zlabel = "wavefunction (Real Part)")
plt.show()


