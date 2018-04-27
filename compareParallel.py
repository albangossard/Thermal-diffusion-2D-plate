from fem2d import *
if not os.path.isfile("NOPLOT"):
	from sklearn import linear_model

parallel=True

list_N = np.loadtxt('courbeCV1_list_N.txt')
list_assemblingTime = np.loadtxt('courbeCV1_parallel=1_list_assemblingTime.txt')
list_computingTime_parallel = np.loadtxt('courbeCV1_parallel=1_list_computingTime.txt')
list_computingTime_parallel2 = np.loadtxt('courbeCV1_parallel=2_list_computingTime.txt')
list_computingTime_notParallel = np.loadtxt('courbeCV1_parallel=0_list_computingTime.txt')



ax = plt.gca()
ax.plot(list_N, list_computingTime_parallel, 'o', label='parallel 1', markeredgecolor='none')
ax.plot(list_N, list_computingTime_parallel2, 'o', label='parallel 2', markeredgecolor='none')
ax.plot(list_N, list_computingTime_notParallel, 'o', label='not parallel', markeredgecolor='none')
ax.set_yscale('log')
ax.set_xscale('log')
plt.grid(True, which="both")
plt.xlabel('N')
plt.ylabel('Computing time (s)')
plt.legend(loc=2)
plt.show()



ax = plt.gca()
ax.plot(list_N, list_assemblingTime, 'o', label='assembling time ', markeredgecolor='none')
ax.plot(list_N, list_computingTime_parallel, 'o', label='computing time ', markeredgecolor='none')
ax.set_yscale('log')
ax.set_xscale('log')
plt.grid(True, which="both")
plt.xlabel('N')
plt.ylabel('Time (s)')
plt.legend(loc=2)
plt.show()