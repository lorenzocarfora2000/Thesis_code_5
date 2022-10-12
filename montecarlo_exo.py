import numpy as np

#all the functions used in Lab 2

def montecarlo(start, end, n, function):
    """
    first pass-in element is "start", hence all the numbers at the 
    bottom of the integrals, while "end" is all the numbers at the top
    of the integrals. "n" is the number of iteration and "function" is the 
    function to integrate.
    
    "per_row" is the number of "integrals" and hence the number of variables.
    The append is used for 1D cases, as it puts an integer element into an 
    array.
    """
    
    a  = []
    a = np.append(a, start)
    per_row = len(a)
    
    """
    Each column i of the random_sample is made up of "n" random elements
    chosen between "start_i" and "end_i". The if-else condition
    is applied to ease the writing of 1D integrals, as otherwise "random_vals"
    would be written as a matrix, each element corresponding to a row.
    """
    
    if per_row == 1:
        random_vals = np.random.uniform(start, end, n)
    else:
        random_vals = np.random.uniform(start, end, (n, per_row))
    
    """
    calculation of the average of the function average "f_avg" and the average
    of the squared function "f_squared_avg".
    """
    
    f_avg, f_squared_avg = 0, 0
    
    for x in random_vals:
        value = function(x)
        f_avg += value
        f_squared_avg += value**2
    
    f_avg /= n
    f_squared_avg /= n
    
    """
    The integral solution and its corresponding Root Mean Squared (RMS) are
    calculated and returned as solutions of the function.
    """
    
    interval = end - start
    
    Integral = np.prod(interval)*f_avg
    
    variance = abs(f_squared_avg - f_avg**2)/n
    RMS = np.sqrt(variance)
    return Integral, RMS

def inside(Y):
    r = np.cos(Y[0]+Y[1])
    return r


x = y = np.linspace(0, 10, 100)
h = x[1]-x[0]

I_array = np.zeros((len(x), len(y)))


for dx in range(len(x)-1):
    
    for dy in range(len(y)-1):
        V = np.array([ x[dx], y[dy] ])
        
        I = montecarlo(V, V+h, 300, inside)[0]
        
        I_array[dx+1, dy+1] += I
        
        V_pre = V

def summer(x):
    c = np.copy(x)
    for i in range(len(x)):
        for j in range(len(x[i])):
            c[i, j] = np.sum(x[:(i+1),:(j+1)])
    return c

solution = summer(I_array)
    
import matplotlib.pyplot as plt
xmesh, ymesh = np.meshgrid(x, y)
fig = plt.figure(figsize=(9, 10))
 
# syntax for 3-D plotting
ax = plt.axes(projection ='3d')
 
# syntax for plotting
ax.plot_surface(xmesh, ymesh, solution, cmap ='viridis', edgecolor ='green')
ax.set_title("provando")
ax.set(ylabel = "y", xlabel= "x", zlabel = "solution")
plt.show()

