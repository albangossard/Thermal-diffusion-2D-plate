from fem2d import *

N = 30*0+200
xDiri = np.linspace(0.,1.,N+1)
T1 = (xDiri)**2.
T3 = (xDiri)**2.+1.
he = 0.
yNeumann=np.linspace(0.,1.,N+1)
Tinf2 = (yNeumann)**2.+1.
Tinf4 = (yNeumann)**2.
Cond = TestCase(N)

lamb=Cond.getLamb([0.5],[1.,100.])
# lamb = Cond.getLamb([0.5],[1.,1.])
plotter(array2d(lamb),'lamb')

f=Cond.getF([0.5],[-4.,-400.])
# f = Cond.getF([0.5],[-4.,-4.])
plotter(array2d(f),'f')

M = FEM(N,T1,T3,Tinf2,Tinf4,allDiri=True,he=he,f=f,lamb=lamb,verbose=0)

M.computeBoundaryCond()
M.compute()


# plotter(array2d(M.x),'x')

Cond.setSolTheorique((Cond.X)**2.+(1.-Cond.Y)**2.)

diff=Cond.sol-array2d(M.x)
print("|\xce\x94|_2="+str(np.linalg.norm(diff)/(diff.shape[0]*diff.shape[1])))
print("|\xce\x94|_inf="+str(np.max(np.abs(diff))))

plotter(array2d(M.x), 'x', Case=Cond)



Cond.setSolTheorique((Cond.X)**2.+(Cond.Y)**2.)
diff = Cond.sol.reshape(-1)-M.x
plotter(array2d(diff),'diff2')
E = FEError(N, diff)
l2Error, h1Error = E.L2H1()
print("L2 error="+str(l2Error))
print("H1 error="+str(h1Error))
