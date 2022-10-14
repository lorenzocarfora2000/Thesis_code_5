import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Butcher tableau data:
rkf4 = np.array([25/216, 0., 1408/2565, 2197/4104, -1/5, 0.])
rkf5 = np.array([16/135 , 0. ,6656/12825 ,28561/56430 ,-9/50 ,2/55])
alfa = np.array([1/4, 3/8, 12/13, 1, 1/2])
beta = np.matrix([[1/4, 0., 0., 0., 0., 0],
                  [3/32, 9/32, 0, 0, 0, 0],
                  [1932/2197, -7200/2197, 7296/2197, 0., 0., 0.],
                  [439/216 ,-8, 3680/513, -845/4104, 0., 0.],
                  [-8/27, 2, -3544/2565, 1859/4104, -11/40, 0.]])
diff = rkf4 - rkf5

def RKF45(t0, tf, x0, v0, dvdt, values):
    
    doc = open("RKF45_text.txt", "w+")
    #ititial conditions written on document
    doc.write(str(t0)+" "+str(x0)+" "+str(v0)+"\n")
    
    t0 , p0 = t0, np.array([x0, v0]) #initial time and p (point, position-velocity vector)
    
    h = 0.1 #initial imposed stepsize
    epsilon_0 = 1e-14 #error tolerance
    length = 1 #number of iterations taken to calculate all needed points
    
    while t0 < tf:
        
        #matrix to be filled with the k values, row 0 is for x, row 1 is for v:
        k = np.zeros((2, len(alfa)+1))
        k[:, 0] = [h*p0[1], h*dvdt(t0, *p0, values)] #attach k1 derived from position data
        
        for i in range(len(alfa)):
            
            #successive ks are calculated by using the data in the Butcher Tableau:
            terms = np.array(beta[i])[0]
            sums = p0+np.dot(k , terms ) #first element is x, second one is v
            k[:, i+1]= [h*(sums[1]), h*dvdt(t0+h*alfa[i], *sums, values)]
            
        #truncation error calculated:
        #(where k[0] is an array made up of the k values for position)
        epsilon = abs(np.dot(diff , k[0]))
        
        if epsilon > epsilon_0:
            #new step calculated and procedure repeated
            h = 0.87*h*(epsilon_0/epsilon)**(1/5)
            continue
        
        #when the truncation error is acceptable, the successive point is derived:
        p0 = p0 + np.dot(k, rkf5)
        t0 = t0+h
        
        #next timestep is derived
        h = 0.87*h*(epsilon_0/epsilon)**(1/5)
        
        doc.write(str(t0)+" "+str(p0[0])+" "+str(p0[1])+"\n")
        
        length = length+1
        
    doc.close()

    #matrix for time, space and veloctiy, to be filled with the derived data points
    tsv = np.zeros((3, length))
    doc = open("RKF45_text.txt", "r", buffering=1048576)
    #writing from document to matrix:
    i = 0
    for line in doc:
        point = line.split()
        tsv[:, i] = [float(point[0]), float(point[1]), float(point[2])]
        i = i+1
    doc.close()

    return tsv

def radial_sol(r, R, dR, array):
    d, sigma  = array
    if r == 0:
        return R*(1-abs(R)**(2*sigma) )/d #R
    else: 
        return R - R*abs(R)**(2*sigma) - (d-1)*dR/r 


def D1_soliton(r, sigma):
    return ((1+sigma)**(1/(2*sigma)))/(np.cosh(r)**(1/sigma))

d, sigma = 2, 1    #dimensions and sigma parameters
r_max = 17   #propagation radius limit
E = 5.4254

#2.2061914429335863
#E(0), the on-axis amplitude
#3.33198890015326034
#5.4255 
#4.15008
#4.827 
#(1+sigma)**(1/(2*sigma))


val = np.array([d, sigma])
sol = RKF45(0., r_max, E, 0., radial_sol, val)

r = sol[0]
R = sol[1]
dR = sol[2]

fig, (ax1, ax2) = plt.subplots(2, sharex = True, figsize=(5, 6))
ax1.plot(r, R)
ax1.plot(r, D1_soliton(r, 1) , "--")
#ax1.set_xlabel("radius r")
ax1.set_ylabel("Amplitude R(r)")
ax1.set_title("NLS soliton solution R(r)*e^{iwt}" )
ax1.grid(True)

ax2.plot(r, dR)
ax2.set_xlabel("radius r")
ax2.set_ylabel("amplitude velocity R'(r)")
ax2.grid(True)

fig.tight_layout()


def count_sols(R):
    sol = 0
    for i in range(len(R)-1):
        if (R[i]>0 and R[i+1] < 0) or (R[i]<0 and R[i+1] > 0):
            sol += 1
    return sol
    
dr = r[1:] - r[:-1]
In = r*(R**2)

I = 0
for i in range(len(dr)):
    I += dr[i]*(In[i]+In[i+1])/2

C = count_sols(R)
print("radial power = {}".format(I) )
print("number of solutions = {}".format(C))


R = np.append(R, np.zeros(2))
r = np.append(r, np.linspace(r[-1], r[-1]+10, 2))

p = np.linspace(0, 2*np.pi, 50)

Rmesh , pmesh = np.meshgrid(R, p)
rmesh, pmesh = np.meshgrid(r, p)
  
X, Y  = rmesh*np.cos(pmesh), rmesh*np.sin(pmesh)

fig = plt.figure(figsize=(9, 10))
ax = plt.axes(projection ='3d')
ax.plot_surface(X, Y, Rmesh, cmap ='viridis', edgecolor ='yellow')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("solution profile for t=0")
ax.set_title("Solution for t=0 in 2D")
fig.legend(title ="R^({}), noticeable from the number of radial solutions = {}".format(C, C))






