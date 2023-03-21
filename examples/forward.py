from scipy.integrate import odeint 
import matplotlib.pyplot as plt
import numpy as np

# Initialization 
tstart = 0 
tstop = 10 
increment = 0.1

# Initial condition 
x_init = [0,0]
t = np.arange(tstart,tstop+1,increment)

# Function that returns dx/dt 
def mydiff(x, t):
    c = 4 # Damping constant
    k = 2 # Stiffness of the spring 
    m = 20 # Mass
    F=np.sin(np.pi*t)
    dx1dt = x[1]
    dx2dt = (F - c*x[1] - k*x[0])/m
    dxdt = [dx1dt, dx2dt] 
    return dxdt

def plotting(t, xi, x2):
    plt.plot(t,x1)
    plt.plot(t,x2)
    plt.title('Simulation of Mass-Spring-Damper System') 
    plt.xlabel('t')
    plt.ylabel('x(t)') 
    plt.legend(["x1", "x2"]) 
    plt.grid()
    plt.show()

# Solve ODE
x = odeint(mydiff, x_init, t)
x1 = x[:,0]
x2 = x[:,1]

print(len(x[:, 0]))
