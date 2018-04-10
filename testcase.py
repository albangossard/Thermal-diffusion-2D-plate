# -*-coding:Latin-1 -*
import numpy as np

class TestCase:
	# Une classe pour manipuler facilement les parametres d'entree du modele
	def __init__(self, N):
		self.N = N
		x = np.linspace(0., 1., self.N)
		self.XCell, self.YCell = np.meshgrid(x, x)
		x = np.linspace(0., 1., self.N+1)
		self.X, self.Y = np.meshgrid(x, x)
		self.lamb = np.ones(self.N**2)
	def getSourceGaussian(self, coeff=1000.):
		# Terme source en gaussienne
		return np.exp(-(self.XCell-0.5)**2.-(self.YCell-0.5)**2.).reshape(-1)*coeff
	def getF(self, listX=[0.5], listF=[1., 100.]):
		# Calcul de la matrice f à partir de délimiteurs en x et de valeur à donner sur des plages rectangulaires
		if len(listX)+1 != len(listF):
			raise ValueError('listX and listF lengths do not match')
		dx = 1./self.N
		listX = [0.]+listX+[1.]
		f = np.ones(self.N**2)
		for i in range(self.N):
			for j in range(self.N):
				# Calcul du numero de cellule et du x
				idCell = i+self.N*j
				# idCell=i+self.N*(self.N-1-j)
				x = j*dx
				# Trouve le plus grand delimiteur plus petit que x
				idF = (next(l[0] for l in enumerate(listX) if l[1] > x)-1)
				# idF=(next(l[0] for l in enumerate(listX) if l[1] < x))
				f[idCell] = listF[idF]
		return f
	def getLamb(self, listX=[0.5], listLamb=[1., 20.]):
		# Calcul de la matrice lambda à partir de délimiteurs en x et de valeur à donner sur des plages rectangulaires
		if len(listX)+1 != len(listLamb):
			raise ValueError('listX and listLamb lengths do not match')
		dx = 1./self.N
		listX = [0.]+listX+[1.]
		for i in range(self.N):
			for j in range(self.N):
				# Calcul du numero de cellule et du x
				idCell = i+self.N*j
				# idCell=i+self.N*(self.N-1-j)
				x = j*dx
				# Trouve le plus grand delimiteur plus petit que x
				idLamb = (next(l[0] for l in enumerate(listX) if l[1] > x)-1)
				# idLamb=(next(l[0] for l in enumerate(listX) if l[1] < x))
				self.lamb[idCell] = listLamb[idLamb]
		return self.lamb
	def setSolTheorique(self, sol):
		self.sol = sol
