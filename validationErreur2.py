from fem2d import *

N = 30
A=1.
K1=1.
K2=2.
w1=np.pi/2.
w2=np.pi/2.
xDiri = np.linspace(0.,1.,N+1)
T1=A*np.sin(K1*w1*xDiri)
T3=A*np.sin(K1*w1*xDiri)*np.cos(K2*w2)
he = 0.
yNeumann=np.linspace(0.,1.,N+1)
Tinf2 = A*np.sin(K1*w1)*np.cos(K2*w2*yNeumann)
Tinf4 = 0.*yNeumann
Cond = TestCase(N)

# lamb=Cond.getLamb([0.5],[1.,100.])
lamb = Cond.getLamb([0.5],[1.,1.])
plotter(array2d(lamb),'lamb')

x = np.linspace(0., 1., N)
X, Y = np.meshgrid(x, x)
f = A*( ((K1*w1)**2.)*np.cos(K2*w2*Y)*np.sin(K1*w1*X) + ((K2*w2)**2.)*np.sin(K1*w1*X)*np.cos(K2*w2*Y) )
f = f.reshape(-1)
# f=Cond.getF([0.5],[-4.,-400.])
# f = Cond.getF([0.5],[-4.,-4.])
plotter(array2d(f),'f')

M = FEM(N,T1,T3,Tinf2,Tinf4,allDiri=True,he=he,f=f,lamb=lamb,verbose=0)

M.computeBoundaryCond()
M.compute()


# plotter(array2d(M.x),'x')

Cond.setSolTheorique(A*np.sin(K1*w1*Cond.X)*np.cos(K2*w2*(1.-Cond.Y)))

diff=Cond.sol-array2d(M.x)
print("|\xce\x94|_2="+str(np.linalg.norm(diff)/(diff.shape[0]*diff.shape[1])))
print("|\xce\x94|_inf="+str(np.max(np.abs(diff))))

plotter(array2d(M.x), 'x', Case=Cond)
