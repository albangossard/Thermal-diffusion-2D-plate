from fem2d import *
from sklearn import linear_model

# list_N=[6,10,20,30,50,80,100]
# list_N=[6,10,20,30,50]
list_N=np.arange(2,21,2)

exactNorm = False

list_errL2=[]
list_errH1=[]
for N in list_N:
    print("\n{:#^70s}".format("N="+str(N)))
    xDiri = np.linspace(0.,1.,N+1)
    T1 = (xDiri)**2.
    T3 = (xDiri)**2.+1.
    yNeumann=np.linspace(0.,1.,N+1)
    Tinf2 = (yNeumann)**2.+1.
    Tinf4 = (yNeumann)**2.
    Cond = TestCase(N)

    lamb=Cond.getLamb([0.5],[1.,100.])

    f=Cond.getF([0.5],[-4.,-400.])

    M = FEM(N,T1,T3,Tinf2,Tinf4,allDiri=True,f=f,lamb=lamb,verbose=0)
    M.computeBoundaryCond()
    M.compute()


    if exactNorm:
        Cond.setSolTheorique((Cond.X)**2.+(Cond.Y)**2.)
        diff = Cond.sol.reshape(-1)-M.x
        # plotter(array2d(diff),'diff2')
        E = FEError(N, diff)
        l2Error, h1Error = E.L2H1()
        print("L2 error="+str(l2Error))
        print("H1 error="+str(h1Error))
        list_errL2.append(l2Error)
        list_errH1.append(h1Error)
    else:
        Cond.setSolTheorique((Cond.X)**2.+(1.-Cond.Y)**2.)
        diff=Cond.sol-array2d(M.x)
        print("L2 norm")
        print("|\xce\x94|_2="+str(np.linalg.norm(diff)/(diff.shape[0]*diff.shape[1])))
        print("|\xce\x94|_inf="+str(np.max(np.abs(diff))))
        print("H1 norm")
        print("|\xce\x94|="+str(np.linalg.norm(diff)/(diff.shape[0]*diff.shape[1])+np.linalg.norm(np.gradient(diff))/(diff.shape[0]*diff.shape[1])))
        # plotter(array2d(M.x), 'x', Case=Cond)
        list_errL2.append(np.linalg.norm(diff)/(diff.shape[0]*diff.shape[1]))
        list_errH1.append(np.linalg.norm(diff)/(diff.shape[0]*diff.shape[1])+np.linalg.norm(np.gradient(diff))/(diff.shape[0]*diff.shape[1]))

list_NRegr=np.log(np.array(list_N)).reshape((-1,1))
list_errL2Regr=np.log(np.array(list_errL2)).reshape((-1,1))
list_errH1Regr=np.log(np.array(list_errH1)).reshape((-1,1))
print("L2")
regrL2 = linear_model.LinearRegression()
regrL2.fit(list_NRegr, list_errL2Regr)
print('slope', regrL2.coef_)
print('intercept', regrL2.intercept_)
print('score', regrL2.score(list_NRegr, list_errL2Regr))
print("H1")
regrH1 = linear_model.LinearRegression()
regrH1.fit(list_NRegr, list_errH1Regr)
print('slope', regrH1.coef_)
print('intercept', regrH1.intercept_)
print('score', regrH1.score(list_NRegr, list_errH1Regr))



# fig = plt.figure()
ax = plt.gca()
ax.plot(list_N, list_errL2, 'o', label='L2', markeredgecolor='none')
ax.plot(list_N, list_errH1, 'o', label='H1', markeredgecolor='none')
ax.set_yscale('log')
ax.set_xscale('log')
plt.grid(True, which="both")
plt.xlabel('N')
plt.ylabel('Error')
plt.legend()
plt.show()