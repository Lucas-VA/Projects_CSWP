import numpy as np
import matplotlib.pyplot as plt

# Discretisationtion:
nx = 300                                    # number of cells in x direction (without PML)
ny = 300                                    # number of cells in y direction (without PML)
N = 20                                      # number of cells per d (must be even)

PML = 40                                    # number of cells of PML
nx += 2*PML
ny += 2*PML

c = 340                                     # wave speed [m/s]
d = 1                                       # [m]
           
dx = d/N                                    # spatial discretisation step (x)
dy = dx                                     # spatial discretisation step (x)

CFL = 0.8                                   # Courant number
dt = CFL/(c*np.sqrt((1/dx**2)+(1/dy**2)))   # time step
nt = int(nx//CFL)                          # number of time steps

### Implementing source and receivers: ###
# location of source
S = 5*N
x_bron = PML + S
y_bron = PML + S

# source pulse information
A = 10
fc = c/10/dx
t0 = 1E-2
sigma = 5E-6

"""
plt.figure()
plt.plot([A*np.exp(-(((it-1)*dt-t0)**2)/(sigma)) for it in range(0, nt)])
plt.grid()
plt.title('Pressure source')
plt.xlabel('timestep')
plt.show()
"""

# location of receivers
x_rec1 = int(x_bron + N*np.sqrt(13)/2*np.sin(np.arctan(2/3) - 20/180*np.pi))
y_rec1 = int(y_bron + N*np.sqrt(13)/2*np.cos(np.arctan(2/3) - 20/180*np.pi))

x_rec2 = int(x_bron + N*5/2*np.sin(np.arctan(4/3) - 20/180*np.pi))
y_rec2 = int(y_bron + N*5/2*np.cos(np.arctan(4/3) - 20/180*np.pi))

x_rec3 = int(x_bron + N*3*np.sqrt(5)/2*np.sin(np.arctan(2) - 20/180*np.pi))
y_rec3 = int(y_bron + N*3*np.sqrt(5)/2*np.cos(np.arctan(2) - 20/180*np.pi))

x_hoek = int(x_bron + N*np.sqrt(2)*np.sin((45-20)/180*np.pi))
y_hoek = int(y_bron + N*np.sqrt(2)*np.cos((45-20)/180*np.pi))

# PML:
k = 1000
D = [i**3 for i in np.linspace(0,1,2*PML+1)]
D_o = np.array(D[::-2]+(nx-1-2*PML)*[0]+D[::2])
D_p = np.array(D[-2::-2]+(nx-2*PML)*[0]+D[1::2])

#Impedance:
Z = 2*c

# create wedge in matrix
wedge = np.zeros((ny, nx))
wedge[y_hoek, x_hoek] = 1
j = 0
vwedgepoints = []
hwedgepoints = []
for i in range(y_hoek+1):
    xshift = i*dy*np.tan(20/180*np.pi)
    if xshift > (j+1)*dx:
        j += 1
    wedge[y_hoek-i, x_hoek+j] = 1
    vwedgepoints.append((y_hoek-i, x_hoek+j))
print(j)
j = 0
for i in range(nx-x_hoek):
    yshift = i*dx*np.tan(20/180*np.pi)
    if yshift > (j+1)*dx:
        j += 1
    wedge[y_hoek+j, x_hoek+i] = 1
    hwedgepoints.append((y_hoek+j, x_hoek+i))
print(j, xshift, yshift)

plt.scatter(x_bron, y_bron, marker='x')
plt.scatter(x_rec1, y_rec1, marker='x')
plt.scatter(x_rec2, y_rec2, marker='x')
plt.scatter(x_rec3, y_rec3, marker='x')
# plt.scatter(x_hoek, y_hoek)
for point in vwedgepoints+hwedgepoints:
    plt.scatter(point[1], point[0], c='k')
# plt.xlim(0, nx)
# plt.ylim(0, ny)
plt.xlim(140, 200)
plt.ylim(140, 200)
# plt.axis('equal')

plt.grid()
plt.show()

#initialisation time series receivers
rec1 = np.zeros((nt,1))
ref1 = np.zeros((nt,1))

rec2 = np.zeros((nt,1))
ref2 = np.zeros((nt,1))

rec3 = np.zeros((nt,1))
ref3 = np.zeros((nt,1))

### time iteration (FDTD): ###
bron = 0

ox = np.zeros((ny, nx+1))
ox_ref = np.zeros((ny, nx+1))

oy = np.zeros((ny+1, nx))
oy_ref = np.zeros((ny+1, nx))

px = np.zeros((ny, nx))
px_ref = np.zeros((ny, nx))

py = np.zeros((ny, nx))
py_ref = np.zeros((ny, nx))

p = px + py
p_ref = px_ref + py_ref

for it in range(0, nt):
    t = (it-1)*dt
    
    # updating and adding source term to propagation
    bron = A*np.sin(2*np.pi*fc*(t-t0))*np.exp(-((t-t0)**2)/(sigma))
    #bron = A*np.exp(-((t-t0)**2)/(sigma))
    
    p[x_bron,y_bron] = p[x_bron,y_bron] + bron 
    p_ref[x_bron,y_bron] = p_ref[x_bron,y_bron] + bron
    
    
    oxold = ox.copy()
    oyold = oy.copy()
    
    ox[:,1:-1] = oxold[:, 1:-1] + (-dt/dx)*(p[:,1:]-p[:,:-1])-dt*k*(oxold*D_o)[:,1:-1]
    oy[1:-1, :] = oyold[1:-1, :] + (-dt/dy)*(p[1:,:]-p[:-1,:])-dt*k*(oyold.transpose()*D_o).transpose()[1:-1,:]
    
#     # Perfectly reflecting boundary
#     xprev = vwedgepoints[0][1]
#     for wy, wx in vwedgepoints:
#         ox[wy, wx+1] = 0
#         if wx != xprev:
#             oy[wy+1, wx] = 0
#         xprev = wx
    
#     yprev = hwedgepoints[0][0]
#     for wy, wx in hwedgepoints[1:]:
#         oy[wy+1, wx] = 0
#         if wy != yprev:
#             ox[wy, wx] = 0
#         yprev = wy
        
    # Impedance boundary
    xprev = vwedgepoints[0][1]
    for wy, wx in vwedgepoints:
        ox[wy, wx+1] = (1/(1+Z*dt/dx))*((1-Z*dt/dx)*oxold[wy, wx+1]+(2*dt/dx)*(p[wy,wx]))
        ox[wy, wx+2] = 0
        oy[wy, wx+1] = 0
        if wx != xprev:
            oy[wy+1, wx] = (1/(1+Z*dt/dy))*((1-Z*dt/dy)*oy[wy+1,wx]+(2*dt/dy)*(p[wy,wx]))-dt*k*(oyold.transpose()*D_o).transpose()[wy+1,wx]
            oy[wy+2, wx] = 0
            oy[wy+1, wx+1] = 0
        xprev = wx
    
    yprev = hwedgepoints[0][0]
    for wy, wx in hwedgepoints[1:]:
        oy[wy+1, wx] = (1/(1+Z*dt/dy))*((1-Z*dt/dy)*oyold[wy+1,wx]-(2*dt/dy)*(p[wy+1,wx]))
        oy[wy, wx] = 0
        ox[wy, wx+1] = 0
        if wy != yprev:
            ox[wy, wx] = (1/(1+Z*dt/dx))*((1-Z*dt/dx)*oxold[wy, wx]+(2*dt/dx)*(p[wy, wx-1]))-dt*k*(oxold*D_o)[wy, wx]
            ox[wy, wx+1] = 0
            ox[wy-1, wx] = 0
        yprev = wy
    
    ox_ref[:,1:-1] += (-dt/dx)*(p_ref[:,1:]-p_ref[:,:-1])-dt*k*(ox_ref*D_o)[:,1:-1]
    
    oy_ref[1:-1,:] += -dt/dx*(p_ref[1:,:]-p_ref[:-1,:])-dt*k*(oy_ref.transpose()*D_o).transpose()[1:-1,:]
    
    px += (-c**2*dt)*((1/dx)*(ox[:,1:]-ox[:,:-1]))-dt*k*px*D_p
    px_ref += (-c**2*dt)*((1/dx)*(ox_ref[:,1:]-ox_ref[:,:-1]))-dt*k*px_ref*D_p
    
    py += (-c**2*dt)*((1/dy)*(oy[1:,:]-oy[:-1,:]))-dt*k*(py.transpose()*D_p).transpose()
    py_ref += (-c**2*dt)*((1/dy)*(oy_ref[1:,:]-oy_ref[:-1,:]))-dt*k*(py_ref.transpose()*D_p).transpose()
    p = px + py
    p_ref = px_ref + py_ref
    
    # store p field at receiver locations
    rec1[it] = p[y_rec1, x_rec1]
    ref1[it] = p_ref[y_rec1, x_rec1]
    
    rec2[it] = p[y_rec2, x_rec2]
    ref2[it] = p_ref[y_rec2, x_rec2]
    
    rec3[it] = p[y_rec3, x_rec3]
    ref3[it] = p_ref[y_rec3, x_rec3]

plt.figure()
plt.matshow(p[::-1], vmin=-0.01, vmax=0.01) #, vmin=-1, vmax=1
# plt.ylim(225, 218)
# plt.xlim(150, 300)
plt.colorbar()
plt.show()
print(PML+N+S, nx, ny)

# Recorded data:
plt.figure()
plt.plot(range(0, nt), rec1, label='recorder 1')
plt.plot(range(0, nt), rec2, label='recorder 2')
plt.plot(range(0, nt), rec3, label='recorder 3')
plt.legend()
plt.ylabel('p-field')
plt.xlabel('timestep')
plt.title('Pressure at reciever locations (wedge present)')
plt.grid()
plt.show()

plt.figure()
plt.plot(range(0, nt), ref1, label='recorder 1')
plt.plot(range(0, nt), ref2, label='recorder 2')
plt.plot(range(0, nt), ref3, label='recorder 3')
plt.legend()
plt.ylabel('p-field')
plt.xlabel('timestep')
plt.title('Pressure at reciever locations (wedge absent)')
plt.grid()
plt.show()
