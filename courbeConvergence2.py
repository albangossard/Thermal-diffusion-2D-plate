from fem2d import *
if not os.path.isfile("NOPLOT"):
	from sklearn import linear_model


list_N=[6,10,20,30,50,80,100,150,200]
# list_N=[6,10,20,30,50]


exactNorm = False

parallel = 1


list_errL2=[]
list_errH1=[]
for N in list_N:
    print("\n{:#^70s}".format("N="+str(N)))
    A=1.
    K1=1.
    K2=2.
    w1=np.pi/2.
    w2=np.pi/2.
    xDiri = np.linspace(0.,1.,N+1)
    T1=A*np.sin(K1*w1*xDiri)
    T3=A*np.sin(K1*w1*xDiri)*np.cos(K2*w2)
    yNeumann=np.linspace(0.,1.,N+1)
    Tinf2 = A*np.sin(K1*w1)*np.cos(K2*w2*yNeumann)
    Tinf4 = 0.*yNeumann
    Cond = TestCase(N)

    lamb=Cond.getLamb([0.5],[1.,1.])

    x = np.linspace(0., 1., N)
    X, Y = np.meshgrid(x, x)
    f = A*( ((K1*w1)**2.)*np.cos(K2*w2*Y)*np.sin(K1*w1*X) + ((K2*w2)**2.)*np.sin(K1*w1*X)*np.cos(K2*w2*Y) )
    f = f.reshape(-1)

    M = FEM(N, T1, T3, Tinf2, Tinf4, allDiri=True, f=f, lamb=lamb, verbose=0, parallel=parallel)
    M.computeBoundaryCond()
    M.compute()

    Cond.setSolTheorique((Cond.X)**2.+(1.-Cond.Y)**2.)
    diff=Cond.sol-array2d(M.x)
    print("L2 norm")
    print("|\xce\x94|_2="+str(np.linalg.norm(diff)/(diff.shape[0]*diff.shape[1])))
    print("|\xce\x94|_inf="+str(np.max(np.abs(diff))))
    print("H1 norm")
    print("|\xce\x94|="+str(np.linalg.norm(diff)/(diff.shape[0]*diff.shape[1])+np.linalg.norm(np.gradient(diff))/(diff.shape[0]*diff.shape[1])))
    list_errL2.append(np.linalg.norm(diff)/(diff.shape[0]*diff.shape[1]))
    list_errH1.append(np.linalg.norm(diff)/(diff.shape[0]*diff.shape[1])+np.linalg.norm(np.gradient(diff))/(diff.shape[0]*diff.shape[1]))

np.savetxt('data/courbeCV2_list_N.txt',list_N)
np.savetxt('data/courbeCV2_list_errL2.txt',list_errL2)
np.savetxt('data/courbeCV2_list_errH1.txt',list_errH1)
if not os.path.isfile("NOPLOT"):
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