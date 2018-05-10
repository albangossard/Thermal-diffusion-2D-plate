from fem2d import *

N = 100
T1 = 200.
T3 = 10.

# he = 0.
# he = 2.
he = 200.

Tinf2 = 10.
Tinf4 = 10.
Cond = TestCase(N)

lamb=Cond.getLamb([0.5],[1.,100.])
plotter(array2d(lamb),'lamb')

f=Cond.getF([0.5],[-4.,-400.])*0.
plotter(array2d(f),'f')

M = FEM(N, T1, T3, Tinf2, Tinf4, allDiri=False, he=he, f=f, lamb=lamb, verbose=0, parallel=1)

M.computeBoundaryCond()
M.compute()


plotter(array2d(M.x),'x')

list_xSection=np.linspace(0.1,0.65,3)
section(M.x,list_xSection,'y')