import numpy as np
np.set_printoptions(threshold=np.inf)
from matplotlib import cm, pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


nx=41
ny=41
nt=100
nit=50
dx=2/(nx-1)
dy=2/(ny-1)
x=np.linspace(0,2,nx)
y=np.linspace(0,2,ny)
X,Y=np.meshgrid(x,y)

rho=1
nu=0.1
dt=.001

u=np.zeros((ny,nx))
v=np.zeros((ny,nx))
p=np.zeros((ny,nx))

# =============================================================================
def pressure(u,v,p,dx,dy,dt,rho,nit):
    
    pn=p.copy()
    
    for i in range(nit):
    	
        pn=p.copy()
        p[1:-1, 1:-1] = ((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2+ \
                         (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /\
                         (2 * (dx**2 + dy**2))-\
                        dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * \
                         (rho * (1 / dt*((u[1:-1, 2:] - u[1:-1, 0:-2]) / \
                     (2 * dx) + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -\
                        ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -\
                        2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *\
                           (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx))-\
                          ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))
                      
        
        p[:, -1] = p[:, -2] # dp/dx = 0 at x = 2
        p[0, :] = p[1, :]   # dp/dy = 0 at y = 0
        p[:, 0] = p[:, 1]   # dp/dx = 0 at x = 0
        p[-1, :] = 0
    return p
def cavitation(u,v,p,dx,dy,dt,nt,nu,rho,nit):
    
    for i in range(nt):
        
        p=pressure(u,v,p,dx,dy,dt,rho,nit)
        un=u.copy()
        vn=v.copy()
        
        u[1:-1, 1:-1] = (un[1:-1, 1:-1]-
                         un[1:-1, 1:-1] * dt / dx *
                        (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                         vn[1:-1, 1:-1] * dt / dy *
                        (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                         dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                         nu * (dt / dx**2 *
                        (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                         dt / dy**2 *
                        (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))

        v[1:-1,1:-1] = (vn[1:-1, 1:-1] -
                        un[1:-1, 1:-1] * dt / dx *
                       (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                        vn[1:-1, 1:-1] * dt / dy *
                       (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                        dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                        nu * (dt / dx**2 *
                       (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                        dt / dy**2 *
                       (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))        
             
        u[0, :]  = 0
        u[:, 0]  = 0
        u[:, -1] = 0
        u[-1, :] = 1    # set velocity on cavity lid equal to 1
        v[0, :]  = 0
        v[-1, :] = 0
        v[:, 0]  = 0
        v[:, -1] = 0
    return u,v,p

def Cplot(X,Y,p):
    plt.figure(figsize=(11,7),dpi=100)
    
    plt.contourf(X,Y,p, alpha=.5, cmap=cm.viridis)
    plt.colorbar()
    
    plt.contour(X,Y,p, cmap=cm.viridis)
    
    plt.quiver(X[::2, ::2],Y[::2, ::2],u[::2,::2],v[::2,::2])
    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def Splot(X,Y,p):
        plt.figure(figsize=(11,7),dpi=100)
    
        plt.contourf(X,Y,p, alpha=.5, cmap=cm.viridis)
        plt.colorbar()
        
        plt.contour(X,Y,p, cmap=cm.viridis)
        
        plt.streamplot(X, Y, u, v)
        
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()


# =============================================================================

u,v,p=cavitation(u,v,p,dx,dy,dt,nt,nu,rho,nit)
Splot(X,Y,p)
Cplot(X,Y,p)