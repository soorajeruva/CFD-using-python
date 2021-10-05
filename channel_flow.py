import numpy as np
np.set_printoptions(threshold=np.inf)
from matplotlib import cm, pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


nx=41
ny=41
nt=10
nit=50
dx=2/(nx-1)
dy=2/(ny-1)

x=np.linspace(0,2,nx)
y=np.linspace(0,2,ny)
X,Y=np.meshgrid(x,y)

rho=1
nu=0.1
F=1
dt=.01

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
                      
        p[:,0]=p[:,-1]
        p[0,:]=p[1,:]
        p[-1,:]=p[-2,:]

    return p
def cavitation(u,v,p,dx,dy,dt,nt,nu,rho,nit):
    undiff=1
    count=0
    while undiff>0.001:
        
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
                        (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))+F*dt

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
# at x=2
 
        u[1:-1, -1] = (un[1:-1, -1]-
                         un[1:-1,-1] * dt / dx *
                        (un[1:-1,-1] - un[1:-1,-2]) -
                         vn[1:-1,-1] * dt / dy *
                        (un[1:-1,-1] - un[0:-2,-1]) -
                         dt / (2 * rho * dx) * (p[1:-1,0] - p[1:-1, -2]) +
                         nu * (dt / dx**2 *
                        (un[1:-1, 0] - 2 * un[1:-1, -1] + un[1:-1,-2]) +
                         dt / dy**2 *
                        (un[2:,-1] - 2 * un[1:-1,-1] + un[0:-2,-1])))+F*dt   
        
        v[1:-1,1:-1] = (vn[1:-1,-1] -
                        un[1:-1,-1] * dt / dx *
                       (vn[1:-1,-1] - vn[1:-1,-2]) -
                        vn[1:-1,-1] * dt / dy *
                       (vn[1:-1,-1] - vn[0:-2,-1]) -
                        dt / (2 * rho * dy) * (p[2:,-1] - p[0:-2,-1]) +
                        nu * (dt / dx**2 *
                       (vn[1:-1, 0] - 2 * vn[1:-1,-1] + vn[1:-1,-2]) +
                        dt / dy**2 *
                       (vn[2:,-1] - 2 * vn[1:-1,-1] + vn[0:-2,-1])))         
#at x=0
        u[1:-1,0] = (un[1:-1, 0]-
                         un[1:-1, 0] * dt / dx *
                        (un[1:-1, 0] - un[1:-1, -1]) -
                         vn[1:-1, 0] * dt / dy *
                        (un[1:-1, 0] - un[0:-2, 0]) -
                         dt / (2 * rho * dx) * (p[1:-1, 1] - p[1:-1, -1]) +
                         nu * (dt / dx**2 *
                        (un[1:-1, 1] - 2 * un[1:-1, 0] + un[1:-1, -1]) +
                         dt / dy**2 *
                        (un[2:, 0] - 2 * un[1:-1, 0] + un[0:-2, 0])))+F*dt

        v[1:-1,1:-1] = (vn[1:-1, 0] -
                        un[1:-1, 0] * dt / dx *
                       (vn[1:-1, 0] - vn[1:-1, -1]) -
                        vn[1:-1, 0] * dt / dy *
                       (vn[1:-1, 0] - vn[0:-2, 0]) -
                        dt / (2 * rho * dy) * (p[2:, 0] - p[0:-2, 0]) +
                        nu * (dt / dx**2 *
                       (vn[1:-1, 1] - 2 * vn[1:-1, 0] + vn[1:-1, -1]) +
                        dt / dy**2 *
                       (vn[2:, 0] - 2 * vn[1:-1, 0] + vn[0:-2, 0])))
        u[0,:]=0
        u[-1,:]=0
        v[0,:]=0
        v[-1,:]=0
    
        undiff=(np.sum(u)-np.sum(un))/np.sum(un)
        count+=1
    return u,v,p,count

def Cplot(X,Y,u,v):
    plt.figure(figsize=(11,7),dpi=100)
    plt.quiver(X[::3, ::3],Y[::3, ::3],u[::3,::3],v[::3,::3])
    #plt.quiver(X, Y, u, v)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
# =============================================================================

u,v,p,count=cavitation(u,v,p,dx,dy,dt,nt,nu,rho,nit)
Cplot(X,Y,u,v)
print(count)