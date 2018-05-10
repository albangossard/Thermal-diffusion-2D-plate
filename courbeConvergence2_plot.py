from fem2d import *
if not os.path.isfile("NOPLOT"):
	from sklearn import linear_model

list_N = np.loadtxt('data/courbeCV2_list_N.txt')
list_errL2 = np.loadtxt('data/courbeCV2_list_errL2.txt')
list_errH1 = np.loadtxt('data/courbeCV2_list_errH1.txt')


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
plt.savefig('plots/courbeCV2.png', dpi=200)
plt.show()