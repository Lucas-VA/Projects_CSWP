import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq

c = 340
d = 1
dx = d/40
dy = dx
CFL = 0.5
dt = CFL/(c*(1/dx**2 + 1/dy**2)**(1/2))
nt = 1000
PML = 30
dPML = PML*dx
N = 20+PML

nx = int(3*d/dx)+2*N
ny = int(2*d/dy)+2*N


x_source = N
y_source = int(3/2*d/dx)+N
x_rec1 = int(d/dx)+N
y_rec1 = N
x_rec2 = int(2*d/dx)+N
y_rec2 = N
x_rec3 = int(3*d/dx)+N
y_rec3 = N

dist1 = ((d)**2 + (3/2*d)**2)**(1/2)
dist2 = ((2*d)**2 + (3/2*d)**2)**(1/2)
dist3 = ((3*d)**2 + (3/2*d)**2)**(1/2)

kapmax = 1500
Do = [(i/dPML)**3 for i in np.linspace(dx, dPML, PML)]
Dp = [(i/dPML)**3 for i in np.linspace(1/2*dx, dPML-1/2*dx, PML)]
kox = kapmax*np.array(Do[::-1] + (nx + 1 - 2*PML)*[0] + Do)
koy = kapmax*np.array(Do[::-1] + (ny + 1 - 2*PML)*[0] + Do)
kpx = kapmax*np.array(Dp[::-1] + (nx - 2*PML)*[0] + Dp)
kpy = kapmax*np.array(Dp[::-1] + (ny - 2*PML)*[0] + Dp)
print(len(Do))

def source(t):
    A = 10
    fc = 0.5*c/10/dx
    t0 = 1E-2
    sigma = 5E-6
    return A*np.sin(2*np.pi*fc*(t-t0))*np.exp(-((t-t0)**2)/(sigma))

ox = np.zeros((nx+1, ny))
oy = np.zeros((nx, ny+1))
px = np.zeros((nx, ny))
py = np.zeros((nx, ny))
p = px + py

rec1 = np.zeros((nt, 1))
rec2 = np.zeros((nt, 1))
rec3 = np.zeros((nt, 1))

for it in range(nt):
    
    # bron 
    p[x_source, y_source] = p[x_source, y_source] + source(it*dt)
    
#     p, px, py, ox, oy = step_spat(p, px, py, ox, oy)
    
    ox[1:-1, :int(d/2 *1/dy+N)] = ox[1:-1, :int(d/2 *1/dy+N)] - dt/dx * (p[1:, :int(d/2 *1/dy+N)] - p[:-1, :int(d/2 *1/dy+N)]) - dt*(ox.T * kox).T[1:-1, :int(d/2 *1/dy+N)]
    oy[:N+int(d/dx), 1:-1] = oy[:N+int(d/dx), 1:-1] - dt/dy * (p[:N+int(d/dx), 1:] - p[:N+int(d/dx), :-1]) - dt*(oy*koy)[:N+int(d/dx), 1:-1]
    
    ox[1:N+int(d/dx), int(d/2 *1/dy+N):] = ox[1:N+int(d/dx), int(d/2 *1/dy+N):] - dt/dx * (p[1:N+int(d/dx), int(d/2 *1/dy+N):] - p[:N+int(d/dx)-1, int(d/2 *1/dy+N):]) - dt*(ox.T*kox).T[1:N+int(d/dx), int(d/2 *1/dy+N):]
    oy[N+int(d/dx):, 1:int(d/2 *1/dy+N)] = oy[N+int(d/dx):, 1:int(d/2 *1/dy+N)] - dt/dy * (p[N+int(d/dx):, 1:int(d/2 *1/dy+N)] - p[N+int(d/dx):, :int(d/2 *1/dy+N)-1]) - dt*(oy*koy)[N+int(d/dx):, 1:int(d/2 *1/dy+N)]
    
    px = px - dt * c**2 * (1/dx * (ox[1:, :] - ox[:-1, :])) - dt*(px.T*kpx).T
    py = py - dt * c**2 * (1/dy * (oy[:, 1:] - oy[:, :-1])) - dt*(py*kpy)
    
    p = px + py
    
    rec1[it] = p[x_rec1, y_rec1]
    rec2[it] = p[x_rec2, y_rec2]
    rec3[it] = p[x_rec3, y_rec3]

plt.figure()
plt.title('source in time')
plt.plot(source(np.arange(0, nt*dt, dt)))
plt.show()

rec1_free = source(np.arange(0, nt*dt, dt)-dist1/c)
rec2_free = source(np.arange(0, nt*dt, dt)-dist2/c)
rec3_free = source(np.arange(0, nt*dt, dt)-dist3/c)

plt.figure()
plt.plot(rec1, label='recorder 1')
plt.plot(rec2, label='recorder 2')
plt.plot(rec3, label='recorder 3')
# plt.plot(rec1_free, label='recorder 1 in free space')
# plt.plot(rec2_free, label='recorder 2 in free space')
# plt.plot(rec3_free, label='recorder 3 in free space')
plt.legend()
plt.show()

# # Fourier transformations werkt nog niet goed
# ft_rec1 = fft(rec1)[:nt//2]
# ft_rec1_free = fft(rec1_free)[:nt//2]
# omega = 2 * np.pi * fftfreq(nt, dt)[:nt//2]
# plt.figure()
# # plt.plot(omega, np.abs(ft_rec1))
# # plt.plot(omega, np.abs(ft_rec1_free))
# plt.plot(omega, np.abs(ft_rec1)/np.abs(ft_rec1_free))
# # plt.xlim(2.5e11, 2.6e11)
# # plt.ylim(100000, 105000)
# plt.show()

plt.figure()   
plt.matshow(p.T) #, extent=(dx, 3*d+2*N*dx, dy, 2*d + 2*N*dy)
plt.scatter(N, (int(3/2*d/dx)+N), color='r')
# shading flat
plt.colorbar()
plt.title('pressure p at t = nt*dt')
plt.show()
print(nx, ny)