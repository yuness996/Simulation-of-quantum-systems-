import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from matplotlib.animation import PillowWriter
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve



#Initial condition of psi
def psi0(x, y, x0, y0, sigma=0.5, k=15*np.pi):
    return np.exp(-1/2*((x-x0)**2 + (y-y0)**2)/sigma**2)*np.exp(1j*k*(x-x0)+1j*k*(y-y0))


# Parameters
L = 8 # Well of width L. Shafts from 0 to +L.
Dy = 0.05 # Spatial step size.
Dt = Dy**2/4 # Temporal step size.
Nx = int(L/Dy) + 1 # Number of points on the x axis.
Ny = int(L/Dy) + 1 # Number of points on the y axis.
Nt = 500 # Number of time steps.
rx = -Dt/(2j*Dy**2)
ry = -Dt/(2j*Dy**2) # Constants to simplify expressions.

x = np.linspace(0, 1, Ny) #x grid
y = np.linspace(0, 1, Ny) #y grid
# Initial position of the center of the Gaussian wave function.
x0 = L/5
y0 = L/5


#step potential perturbation
v = np.zeros((Ny,Ny), complex)
x1 = int(Ny*0.4)
x2 = int(Ny*0.6)
y1 = int(Ny*0.4)
y2 = int(Ny*0.6)
for ii in range(x1,x2):
    for jj in range(y1, y2):
        v[ii][jj] = 1e3
        #potential perturbation
v[0,:] = v[-1,:] = v[:,0] = v[:,-1] = 1e5
# Potential.

Ni = (Nx-2)*(Ny-2)  # Number of unknown factors v[i,j], i = 1,...,Nx-2, j = 1,...,Ny-2


# Matrices for the Crank-Nicolson calculus. The problem A·x[n+1] = b = M·x[n] will be solved at each time step.
A = np.zeros((Ni,Ni), complex)
M = np.zeros((Ni,Ni), complex)

# We fill the A and M matrices.
for k in range(Ni):     
    
    # k = (i-1)*(Ny-2) + (j-1)
    i = 1 + k//(Ny-2)
    j = 1 + k%(Ny-2)
    
    # Main central diagonal.
    A[k,k] = 1 + 2*rx + 2*ry + 1j*Dt/2*v[i,j]
    M[k,k] = 1 - 2*rx - 2*ry - 1j*Dt/2*v[i,j]
    
    if i != 1: # Lower lone diagonal.
        A[k,(i-2)*(Ny-2)+j-1] = -ry 
        M[k,(i-2)*(Ny-2)+j-1] = ry
        
    if i != Nx-2: # Upper lone diagonal.
        A[k,i*(Ny-2)+j-1] = -ry
        M[k,i*(Ny-2)+j-1] = ry
    
    if j != 1: # Lower main diagonal.
        A[k,k-1] = -rx 
        M[k,k-1] = rx 

    if j != Ny-2: # Upper main diagonal.
        A[k,k+1] = -rx
        M[k,k+1] = rx


Asp = csc_matrix(A)

x = np.linspace(0, L, Ny-2) # Array of spatial points.
y = np.linspace(0, L, Ny-2) # Array of spatial points.
x, y = np.meshgrid(x, y)
psis = [] # To store the wave function at each time step.

psi = psi0(x, y, x0, y0) # We initialise the wave function with the Gaussian.
psi[0,:] = psi[-1,:] = psi[:,0] = psi[:,-1] = 0 # The wave function equals 0 at the edges of the simulation box (infinite potential well).
psis.append(np.copy(psi)) # We store the wave function of this time step.

# We solve the matrix system at each time step in order to obtain the wave function.
for i in range(1,Nt):
    psi_vect = psi.reshape((Ni)) # We adjust the shape of the array to generate the matrix b of independent terms.
    b = np.matmul(M,psi_vect) # We calculate the array of independent terms.
    psi_vect = spsolve(Asp,b) # Resolvemos el sistema para este paso temporal.
    psi = psi_vect.reshape((Nx-2,Ny-2)) # Recuperamos la forma del array de la función de onda.
    psis.append(np.copy(psi)) # Save the result.

# We calculate the modulus of the wave function at each time step.
mod_psis = [] # For storing the modulus of the wave function at each time step.
for wavefunc in psis:
    re = np.real(wavefunc) # Real part.
    im = np.imag(wavefunc) # Imaginary part.
    mod = np.sqrt(re**2 + im**2) # We calculate the modulus.
    mod_psis.append(mod) # We save the calculated modulus.
    

# animation.

fig = plt.figure() # We create the figure.
ax = fig.add_subplot(111, xlim=(0,L), ylim=(0,L)) # We add the subplot to the figure.

img = ax.imshow(mod_psis[0], extent=[0,L,0,L], cmap=plt.get_cmap("hot"), vmin=0, vmax=np.max(mod_psis), zorder=1, interpolation="none") # Here the modulus of the 2D wave function shall be represented.


#draw the center square
rectangle1 = plt.Rectangle((0.4*L, 0.4*L), 0.1, 0.2*L, fc='b')
rectangle2 = plt.Rectangle((0.4*L, 0.4*L), 0.2*L, 0.1, fc='b')
rectangle3 = plt.Rectangle((0.4*L, 0.6*L), 0.2*L, 0.1, fc='b')
rectangle4 = plt.Rectangle((0.6*L-0.1, 0.4*L), 0.1, 0.2*L, fc='b')

plt.gca().add_patch(rectangle1)
plt.gca().add_patch(rectangle2)
plt.gca().add_patch(rectangle3)
plt.gca().add_patch(rectangle4)

# We define the animation function for FuncAnimation.

def animate(i):
    
    """
    Animation function. Paints each frame. Function for Matplotlib's 
    FuncAnimation.
    """
    
    img.set_data(mod_psis[i]) # Fill img with the modulus data of the wave function.
    img.set_zorder(1)
    
    return img, # We return the result ready to use with blit=True.


anim = FuncAnimation(fig, animate, interval=1, frames=np.arange(0,Nt,2), repeat=False, blit=0) # We generate the animation.

cbar = fig.colorbar(img)
#plt.show() # We show the animation finally.

anim.save('2Dtest3.gif',writer='pillow',fps=50,dpi=100)

