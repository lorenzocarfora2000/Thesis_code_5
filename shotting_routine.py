import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

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

def count_sols(R):
    sol = 0
    for i in range(len(R)-1):
        if (R[i]>0 and R[i+1] < 0) or (R[i]<0 and R[i+1] > 0):
            sol += 1
    return sol

    
def radial_sol(r, R, dR, array):
    d, sigma  = array
    if r == 0:
        return R*(1-abs(R)**(2*sigma) )/d 
    else: 
        return R - R*abs(R)**(2*sigma) - (d-1)*dR/r 


def objective1(R0, vals):
    val, N = vals[:2], vals[2]
    R0 = R0[0]
    sol = RKF45(0., r_max, R0, 0., radial_sol, val)
    
    n = count_sols(sol[1])-N
    return n

def objective2(R0, vals):
    val = vals[:2]
    d = vals[0]
    N = vals[2]
    R0 = R0[0]
    sol = RKF45(0., r_max, R0, 0., radial_sol, val)
    
    r, R, dR = sol
    
    dr = r[1:] - r[:-1]
    In1, In2  = r**(d-1)*(R)**2, (dR**2)*r**(d-1)

    I1, I2 = 0, 0
    for i in range(len(dr)):
        I1 += dr[i]*(In1[i]+In1[i+1])/2
        I2 += dr[i]*(In2[i]+In2[i+1])/2
    
    Val = I1/I2 - (2-sigma*(d-2))/(sigma*d)

    sol = RKF45(0., r_max, R0, 0., radial_sol, val)
    
    n = count_sols(sol[1])-N
    
    print(Val)
    return [Val, n]
    

R0 = 1.5
N = 1 
d , sigma = 2, 1


val = np.array([d, sigma, N])

r_max = 15

#guess1 = np.array([R0])
#R0, = fsolve(objective1, guess1, args=(val) )
#print(R0)

guess2 = np.array([R0, 0])
R0 = fsolve(objective2, guess2, args=(val) )[0]

R0 = abs(R0)
print(R0)

sol = RKF45(0., r_max, R0, 0., radial_sol, val[:-1])

r = sol[0]
R = sol[1]
dR = sol[2]

plt.figure(1)
plt.plot(r, R)
#plt.plot(r, r**((1-d)/2)*np.exp(-r))
plt.xlabel("radius r")
plt.ylabel("R")
plt.title("solution $R({})$ for d = {}, $\sigma$={}".format(N, d, sigma))
plt.grid(True)




