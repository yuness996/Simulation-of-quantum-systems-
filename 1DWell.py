import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import PillowWriter

Nx = 301 # number of steps in x
Nt = 100000 # number of steps in time
dx = 1/(Nx-1) #x step size
dt=1e-7 #time step size
x = np.linspace(0, 1, Nx) #x grid
sig = 1/100
k = 400 #wave number of the wave packet
a = 0.2 # initial position of the peak
A = 1/(np.pi**(1/4)*np.sqrt(sig)) #Amplitude of the wave packet
psi0 = A * np.exp(1j*k*(x-a) - ((x-a)**2)/(sig**2))
#initial condition of psi(gaussian wave packet)   

normal = np.sum(np.absolute(psi0)**2)*dx
psi0 = psi0/normal

V = np.zeros(301)

for i in range (140,160):
    V[i] = 0.007
V[0] = V[-1] = 1e7
#potential perturbation


psi = np.zeros([Nt,Nx],'complex_')
psi[0] = psi0


# Matrices for the Crank-Nicolson calculus.
def Mat1(Nx):
    a1 = np.zeros(Nx,'complex_')
    a2 = np.zeros(Nx-1,'complex_')
    a3 = np.zeros(Nx-1,'complex_')
    for i in range (0, Nx-2):
        a1[i] = 1 + 1j*dt/(2*dx**2) + 1j*V[i]/2
        a2[i] = -1j*dt/(4*dx**2)
        a3[i] = a2[i]
    a1[Nx-1] = 1 + 1j*dt/(2*dx**2) + 1j*V[Nx-1]/2
    A = np.diag(a1, k=0) + np.diag(a2, k=1) + np.diag(a3, k=-1)
   
    return A
   


def Mat2(Nx):
    a1 = np.zeros(Nx,'complex_')
    a2 = np.zeros(Nx-1,'complex_')
    a3 = np.zeros(Nx-1,'complex_')
    for i in range (0, Nx-2):
        a1[i] = 1 - 1j*dt/(2*dx**2) - 1j*V[i]/2
        a2[i] = 1j*dt/(4*dx**2)
        a3[i] = a2[i]
    a1[Nx-1] = 1 + 1j*dt/(2*dx**2) + 1j*V[Nx-1]/2
    A = np.diag(a1, k=0) + np.diag(a2, k=1) + np.diag(a3, k=-1)
   
    return A
   

#Solve the A·x[n+1] = B·x[n] system for each time step.
def compute_psi(psi):
    A = Mat1(Nx)
    B = Mat2(Nx)
    C = np.matmul(np.linalg.inv(A), B)
    for t in range(0, Nt-1):
        psi[t+1] = np.matmul(C, psi[t])
       
    return psi
    

psi1 = compute_psi(psi)
#plt.plot(x, np.absolute(psi[1])**2)

#Animate

def animate(i):
    ln1.set_data(x, np.absolute(psi1[100*i])**2)
   
fig, ax = plt.subplots(1,1, figsize=(8,4))
#ax.grid()
ln1, = plt.plot([], [], 'g-', lw=2, markersize=8, label='Wave function')
#returns a tuple with one element
ax.set_ylim(-1, 60)
ax.set_xlim(-0.5,1.5)

ax.set_ylabel('$|\psi(x)|^2$', fontsize=20)
ax.set_xlabel('$x/L$', fontsize=20)

ax.legend(loc='upper left')
plt.plot(x,V)
plt.tight_layout()
ani = animation.FuncAnimation(fig, animate, frames=1000, interval=50)
#call our main animation function
ani.save('pip1.gif',writer='pillow',fps=50,dpi=100)
