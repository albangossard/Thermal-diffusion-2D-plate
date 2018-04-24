# -*-coding:Latin-1 -*
from __future__ import print_function
import time
import numpy as np
import scipy.sparse as scp
from scipy.sparse.linalg import spsolve
from testcase import *
import os
if not os.path.isfile("NOPLOT"):
	from plotter import *

from joblib import Parallel, delayed
import multiprocessing



"""
cells :
30 31 32 33 34 35
24 25 26 27 28 29
18 19 20 21 22 23 N
12 13 14 15 16 17
6  7  8  9  10 11
0  1  2  3  4  5
       N

nodes :
42----43----44----45----46----47----48
|  30 |  31 |  32 |  33 |  34 |  35 |
35----36----37----38----39----40----41
|  24 |  25 |  26 |  27 |  28 |  29 |
28----29----30----31----32----33----34
|  18 |  19 |  20 |  21 |  22 |  23 |
21----22----23----24----25----26----27
|  12 |  13 |  14 |  15 |  16 |  17 |
14----15----16----17----18----19----20
|  6  |  7  |  8  |  9  |  10 |  11 |
7-----8-----9-----10----11----12----13
|  0  |  1  |  2  |  3  |  4  |  5  |
0-----1-----2-----3-----4-----5-----6

orientation :
^ y
|
|
|
+----> x

localNotation :
3-----2
| idC |
0-----1

"""
def array2d(x):
	N = int(np.sqrt(x.shape[0]))
	x2 = np.zeros((N, N))
	for i in range(N):
		x2[N-i-1, :] = x[i*N:(i+1)*N]
	return x2

def progress(iteration, total, prefix='', suffix='', decimals=1, length=40, fill='#'):
	# Barre de progression pour éviter des affichages trop importants
	percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
	filledLength = int(length * iteration // total)
	bar = fill * filledLength + '-' * (length - filledLength)
	print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
	if iteration == total:
		print()


class Polynome:
	#Classe pour représenter un polynôme
	def __init__(self, a, b, c, d):
		self.a = a
		self.b = b
		self.c = c
		self.d = d
	def evaluateX(self, x):
		#Méthode pour construire un polynôme d'une variable à partir du polynôme courant à 2 variables
		Ret = Polynome(0., self.b+self.c*x, 0., self.d+self.a*x)
		return Ret
def ftimesg(pol1, pol2, x, y):
	#Evaluation d'un produit de polynômes en un point donné
	res = 0.
	res += pol1.a*pol2.a*x**2.   + pol1.a*pol2.b*x*y     + pol1.a*pol2.c*y*x**2.   + pol1.a*pol2.d*x
	res += pol1.b*pol2.a*x*y     + pol1.b*pol2.b*y**2.   + pol1.b*pol2.c*x*y**2.   + pol1.b*pol2.d*y
	res += pol1.c*pol2.a*y*x**2. + pol1.c*pol2.b*x*y**2. + pol1.c*pol2.c*(x*y)**2. + pol1.c*pol2.d*x*y
	res += pol1.d*pol2.a*x       + pol1.d*pol2.b*y       + pol1.d*pol2.c*x*y       + pol1.d*pol2.d
	return res
def gradftimesgradg(pol1, pol2, x, y):
	#Evaluation d'un produit scalaire de gradient de polynômes en un point donné
	return (pol1.a+pol1.c*y)*(pol2.a+pol2.c*y) + (pol1.b+pol1.c*x)*(pol2.b+pol2.c*x)
def ftimescte(pol1, pol2, x, y):
	#Evaluation d'un produit entre un polynôme et une constante
	return pol2.d*(pol1.a*x + pol1.b*y + pol1.c*x*y + pol1.d)
def oppNode(idNode):
	#Renvoie le numéro de noeud opposé à un noeud donné dans une cellule quelconque en numérotation locale
	return (idNode+2)%4




class FEComputing:
	def __init__(self, N, verbose=0):
		# verbose:
		#	- 0 : none
		#	- 1 : basic information
		#	- 2 : loop information
		#	- 3 : node choice information
		self.verbose = verbose
		self.N = N
		self.dx = 1./self.N

		#Poids et points de quadrature
		self.quadWeight = np.array([5./18., 8./18., 5./18.])
		self.quadPts = np.array([0.112701665379258311482073460022, 0.5, 0.887298334620741688517926539978])
	def getNeighbors(self, idCell):
		#Renvoie les ID des noeuds délimitant une cellule
		#Calcul des identifiants en x et en y
		yCell = idCell/self.N
		xCell = idCell-self.N*yCell
		# print("   xCell="+str(xCell)+"   yCell="+str(yCell))
		listNeighbors = [0, 0, 0, 0]
		#Formules se déterminant facilement avec un dessin
		listNeighbors[0] = xCell+(self.N+1)*yCell
		listNeighbors[1] = xCell+1+(self.N+1)*yCell
		listNeighbors[2] = xCell+1+(self.N+1)*(yCell+1)
		listNeighbors[3] = xCell+(self.N+1)*(yCell+1)
		return listNeighbors
	def getCoordNode(self, idNode):
		#Renvoie les coordonnées spatiales d'un noeud
		yNode = idNode/(self.N+1)
		xNode = idNode-(self.N+1)*yNode
		return xNode*self.dx, yNode*self.dx
	def phi(self, alphaX, alphaY, betaX, betaY, x_tilde, y_tilde):
		#Fonction de changement de variable, renvoie :
		#	x=alphaX+betaX*x_tilde
		#	y=alphaY+betaY*y_tilde
		return alphaX+betaX*x_tilde, alphaY+betaY*y_tilde
	def quadIntegration(self, multipFct, fct1, fct2, alphaX, alphaY, betaX, betaY):
		#Méthode réalisant l'intégration par quadrature de Gauss sur un carré
		#On prend en paramètre le type d'évaluation du produit de polynôme, les fonctions passées dans ce produit et le changement de variable
		res = 0.
		for i in range(3):
			for j in range(3):
				w_i = self.quadWeight[i]*self.quadWeight[j]
				x_i = self.quadPts[i]
				y_i = self.quadPts[j]
				x_reel, y_reel = self.phi(alphaX, alphaY, betaX, betaY, x_i, y_i)
				res += w_i*multipFct(fct1, fct2, x_reel, y_reel)
		return res
	def getGlobalNodeNb(self, i, idNode):
		#Méthode donnant l'ID d'un noeud à partir du noeud de référence et de la numérotation locale
		if i == 0:
			return idNode
		elif i == 1:
			return idNode+1
		elif i == 2:
			return idNode+1+(self.N+1)
		else:
			return idNode+(self.N+1)
	def constructPolyNode(self, i1, idNode, idNode0):
		iOpp = oppNode(i1)
		idNodeOpp = self.getGlobalNodeNb(iOpp, idNode0)
		if self.verbose >= 3:
			print("-----------")
			print("idNode="+str(idNode)+"  idNodeOpp="+str(idNodeOpp))
			print("i1="+str(idNode)+"  iOpp="+str(iOpp))
		xi, yi = self.getCoordNode(idNode)
		# print("xi="+str(xi)+"  yi="+str(yi))
		xj, yj = self.getCoordNode(idNodeOpp)
		# print("xj="+str(xj)+"  yj="+str(yj))
		norm = (xi-xj)*(yi-yj)
		poly = Polynome(-yj/norm, -xj/norm, 1./norm, xj*yj/norm)
		return poly



def computeErrorOnCell(self2, idCell, derivative):
	res = 0.
	resDerivative = 0.
	#Récupère les noeuds délimitant la cellule
	idNodes = self2.getNeighbors(idCell)
	#Calcul des positions de ces noeuds
	x0, y0 = self2.getCoordNode(idNodes[0])
	x1, _ = self2.getCoordNode(idNodes[1])
	_, y3 = self2.getCoordNode(idNodes[3])
	#Calcul du changement de variable :
	#	x=alphaX+betaX*x_tilde
	#	y=alphaY+betaY*y_tilde
	alphaX = x0
	betaX = x1-x0
	alphaY = y0
	betaY = y3-y0
	#Récupère le noeud de référence de la cellule (en bas à gauche)
	idNode0 = self2.getNeighbors(idCell)[0]
	#Boucle sur les noeuds associés à la cellule
	for i1, idNode1 in enumerate(self2.getNeighbors(idCell)):
		#Boucle sur les noeuds associés à la cellule
		for i2, idNode2 in enumerate(self2.getNeighbors(idCell)):
			#Construction des 2 fonctions à intégrer sur la cellule
			fct1 = self2.constructPolyNode(i1, idNode1, idNode0)
			fct2 = self2.constructPolyNode(i2, idNode2, idNode0)
			#Calcul du terme intégrale(phi_i*phi_j)
			res += self2.f[idNode1]*self2.f[idNode2]*np.abs(betaX*betaY)*self2.quadIntegration(ftimesg, fct1, fct2, alphaX, alphaY, betaX, betaY)
			if derivative > 0:
				#Calcul du terme intégrale(Grad(phi_i)*Grad(phi_j))
				resDerivative += self2.f[idNode1]*self2.f[idNode2]*np.abs(betaX*betaY)*self2.quadIntegration(gradftimesgradg, fct1, fct2, alphaX, alphaY, betaX, betaY)
	return (res, resDerivative)



class FEError(FEComputing):
	def __init__(self, N, f, verbose=0):
		FEComputing.__init__(self, N, verbose=verbose)
		self.f = f
	def compute(self, derivative):
		# derivate :
		#	- 0 : compute L2 norm only
		#	- 1 : compute H1 norm only
		#	- 2 : compute both L2 and H1 norms

		#Ce code est parallélisable car il ne comporte pas de section critique
		#On calcule donc l'intégrale sur chaque cellule et on somme le tout ensuite
		num_cores = multiprocessing.cpu_count()
		#Boucle sur les cellules
		results = Parallel(n_jobs=num_cores)(delayed(computeErrorOnCell)(self, idCell, derivative) for idCell in range(self.N**2))
		res = 0.
		resDerivative = 0.
		for e in results:
			res += e[0]
			resDerivative += e[1]
			
		if derivative == 0:
			return res
		elif derivative == 1:
			return res+resDerivative
		else:
			return res, resDerivative
	def L2(self):
		return self.compute(0)
	def H1(self):
		return self.compute(1)
	def L2H1(self):
		return self.compute(2)



class FEM(FEComputing):
	def __init__(self, N, T1, T3, Tinf2, Tinf4, allDiri=False, he=0., f=None, lamb=None, verbose=0):
		FEComputing.__init__(self, N, verbose=verbose)
		print("N="+str(self.N))
		# self.A = np.zeros(((self.N+1)**2, (self.N+1)**2))
		self.A = scp.lil_matrix(((self.N+1)**2, (self.N+1)**2))
		self.b = np.zeros((self.N+1)**2)
		self.allDiri = allDiri
		if not self.allDiri:
			self.he = he
		else:
			self.he = 0.
		# self.Tinf=Tinf
		if lamb is None:
			lamb = np.ones(self.N**2)
		self.lamb = lamb
		self.c = np.ones(self.N**2)*0.
		if f is None:
			f = np.zeros(self.N**2)
		self.f = f
		#Impose les conditions de Dirichlet, permet de passer sous forme de flottant (condition uniforme sur un bord) ou sous forme d'array
		if hasattr(T1, "__len__"):
			self.T1Uniform = False
			self.T1 = T1
		else:
			self.T1Uniform = True
			self.T1 = T1
		if hasattr(T3, "__len__"):
			self.T3Uniform = False
			self.T3 = T3
		else:
			self.T3Uniform = True
			self.T3 = T3
		#Idem pour Tinf de Neumann
		if hasattr(Tinf2, "__len__"):
			self.T2Uniform = False
			self.Tinf2 = Tinf2
		else:
			self.T2Uniform = True
			self.Tinf2 = Tinf2
		if hasattr(Tinf4, "__len__"):
			self.T4Uniform = False
			self.Tinf4 = Tinf4
		else:
			self.T4Uniform = True
			self.Tinf4 = Tinf4

		self.listNodeDiri = []
		self.listNodeNeumann = []
	def computeBoundaryCond(self):
		#Calcul des noeuds qui sont sur les bords de Dirichlet
		for idNode in range((self.N+1)**2):
			xNode, yNode = self.getCoordNode(idNode)
			if abs(yNode-0.) <= self.dx/100. or abs(yNode-1.) <= self.dx/100.:
				self.listNodeDiri.append(idNode)
			elif abs(xNode-0.) <= self.dx/100. or abs(xNode-1.) <= self.dx/100.:
				if self.allDiri:
					self.listNodeDiri.append(idNode)
				else:
					self.listNodeNeumann.append(idNode)
		if self.verbose >= 1:
			print("Nodes on Dirichlet border : ")
			print("\t"+str(self.listNodeDiri))
			print("Nodes on Neumann border : ")
			print("\t"+str(self.listNodeNeumann))
	def getDiriCond(self, idNode):
		#Renvoie le terme de bord de Dirichlet
		xNode, yNode = self.getCoordNode(idNode)
		#On procède de la sorte pour tester si un noeud est sur un bord de Dirichlet car pour un grand nombre de noeud, il y a des erreurs d'arrondi
		if abs(yNode-0.) <= self.dx/100.:
			if self.T1Uniform:
				return self.T1
			else:
				yNode2 = idNode/(self.N+1)
				xNode2 = idNode-(self.N+1)*yNode2
				return self.T1[xNode2]
		elif abs(yNode-1.) <= self.dx/100.:
			if self.T3Uniform:
				return self.T3
			else:
				yNode2 = idNode/(self.N+1)
				xNode2 = idNode-(self.N+1)*yNode2
				return self.T3[xNode2]
		else:
			if not self.allDiri:
				raise ValueError('Node not on Dirichlet border')
			else:
				if abs(xNode-1.) <= self.dx/100.:
					if self.T2Uniform:
						return self.Tinf2
					else:
						yNode2 = idNode/(self.N+1)
						xNode2 = idNode-(self.N+1)*yNode2
						return self.Tinf2[yNode2]
				elif abs(xNode-0.) <= self.dx/100.:
					if self.T4Uniform:
						return self.Tinf4
					else:
						yNode2 = idNode/(self.N+1)
						xNode2 = idNode-(self.N+1)*yNode2
						return self.Tinf4[yNode2]
	def getNeumannCond(self, idNode):
		#Renvoie le terme de bord de Neumann Tinf
		xNode, yNode = self.getCoordNode(idNode)
		if not self.allDiri:
			#On procède de la sorte pour tester si un noeud est sur un bord de Neumann car pour un grand nombre de noeud, il y a des erreurs d'arrondi
			if abs(xNode-0.) <= self.dx/100.:
				if self.T2Uniform:
					return self.Tinf2
				else:
					yNode2 = idNode/(self.N+1)
					return self.Tinf2[yNode2]
			elif abs(xNode-1.) <= self.dx/100.:
				if self.T4Uniform:
					return self.Tinf4
				else:
					yNode2 = idNode/(self.N+1)
					return self.Tinf4[yNode2]
			else:
				raise ValueError('Node not on Neumann border')
		else:
			raise ValueError('Node not on Neumann border')
	def quadIntegration1D(self, multipFct, fct1, fct2, alphaX, alphaY, betaX, betaY):
		#Méthode réalisant l'intégration par quadrature de Gauss sur un segment vertical
		#On prend en paramètre le type d'évaluation du produit de polynôme, les fonctions passées dans ce produit et le changement de variable
		res = 0.
		for i in range(3):
			w_i = self.quadWeight[i]
			y_i = self.quadPts[i]
			#On se fiche de la coordonnée en x car on l'a éliminée par évaluation partielle du polynôme
			x_reel, y_reel = self.phi(alphaX, alphaY, betaX, betaY, 0., y_i)
			res += w_i*multipFct(fct1, fct2, x_reel, y_reel)
		return res
	def compute(self):
		# Méthode calculant la solution
		start = time.time()


		#Ce code est parallélisable car il ne comporte pas de section critique
		#On calcule donc l'intégrale sur chaque et on remplit la matrice ensuite
		num_cores = multiprocessing.cpu_count()
		#Boucle sur les cellules
		results = Parallel(n_jobs=num_cores)(delayed(computeOnCell)(self, idCell) for idCell in range(self.N**2))
		progress(self.N**2, self.N**2, prefix='Filling matrix', suffix='', decimals=1, length=40, fill='#')
		for tab in results:
			elemA=tab[0]
			elemb=tab[1]
			for e in elemA:
				self.A[e[0], e[1]] += e[2]
			for e in elemb:
				self.b[e[0]] += e[1]


		#Traitement des noeuds situés sur le bord de Dirichlet
		for idNodeDiri in self.listNodeDiri:
			#On impose la valeur de la condition limite en mettant toute la ligne de A à 0 sauf le coefficient (i,i) que l'on met à 1
			#La résolution du système linéaire imposera donc que l'intégrale vaut le terme de droite
			self.A[idNodeDiri, idNodeDiri] = 1.
			self.b[idNodeDiri] = self.getDiriCond(idNodeDiri)
			# print(str(self.getCoordNode(idNodeDiri))+" -> "+str(self.b[idNodeDiri]))
		progress(self.N**2, self.N**2, prefix='Solving system', suffix='', decimals=1, length=40, fill='#')
		#Résolution du système linéaire
		# self.x = np.linalg.solve(self.A, self.b)
		print("nonZeroRate="+str(float(self.A.nonzero()[0].shape[0])/(self.A.shape[0]*self.A.shape[1])))
		self.A = self.A.tocsr()
		self.x = spsolve(self.A, self.b)
		end = time.time()
		self.computingTime = end-start
		print("computingTime="+str(self.computingTime))

def computeOnCell(self2, idCell):
	# Tableaux de retour des données
	# Format d'un élément de A : [idNode1, idNode2, value]
	resA=[]
	# Format d'un élément de b : [idNode1, value]
	resb=[]

	# Affichage sympa
	if self2.verbose == 0 and idCell%20 == 0:
		progress(idCell, self2.N**2, prefix='Iteration '+str(idCell)+'/'+str(self2.N**2), suffix='', decimals=1, length=40, fill='#')
	if self2.verbose >= 2:
		print("\n{:#^70s}".format("LOOP idCell="+str(idCell)))
	#Récupère les noeuds délimitant la cellule
	idNodes = self2.getNeighbors(idCell)
	#Calcul des positions de ces noeuds
	x0, y0 = self2.getCoordNode(idNodes[0])
	x1, _ = self2.getCoordNode(idNodes[1])
	_, y3 = self2.getCoordNode(idNodes[3])
	#Calcul du changement de variable :
	#	x=alphaX+betaX*x_tilde
	#	y=alphaY+betaY*y_tilde
	alphaX = x0
	betaX = x1-x0
	alphaY = y0
	betaY = y3-y0
	#Récupère le noeud de référence de la cellule (en bas à gauche)
	idNode0 = self2.getNeighbors(idCell)[0]
	#Boucle sur les noeuds associés à la cellule
	for i1, idNode1 in enumerate(self2.getNeighbors(idCell)):
		if self2.verbose >= 2:
			print("\n{:#^55s}".format("LOOP idNode1="+str(idNode1)))
		#Boucle sur les noeuds associés à la cellule
		for i2, idNode2 in enumerate(self2.getNeighbors(idCell)):
			if self2.verbose >= 2:
				print("\n{:#^40s}".format("idNode1="+str(idNode1)+" idNode2="+str(idNode2)))
			#Construction des 2 fonctions à intégrer sur la cellule
			fct1 = self2.constructPolyNode(i1, idNode1, idNode0)
			fct2 = self2.constructPolyNode(i2, idNode2, idNode0)
			#Sur les bords de Dirichlet on impose les fonctions à zéro, on les exclue donc de la boucle pour les traiter à part
			if idNode1 not in self2.listNodeDiri:
				#Calcul du terme intégrale(c*phi_i*phi_j) -> notre algorithme permet de traiter un cas plus général
				# self2.A[idNode1, idNode2] += self2.c[idCell]*np.abs(betaX*betaY)*self2.quadIntegration(ftimesg, fct1, fct2, alphaX, alphaY, betaX, betaY)
				resA.append([idNode1, idNode2, self2.c[idCell]*np.abs(betaX*betaY)*self2.quadIntegration(ftimesg, fct1, fct2, alphaX, alphaY, betaX, betaY)])
				#Calcul du terme intégrale(lambda*Grad(phi_i)*Grad(phi_j))
				# self2.A[idNode1, idNode2] += self2.lamb[idCell]*np.abs(betaX*betaY)*self2.quadIntegration(gradftimesgradg, fct1, fct2, alphaX, alphaY, betaX, betaY)
				resA.append([idNode1, idNode2, self2.lamb[idCell]*np.abs(betaX*betaY)*self2.quadIntegration(gradftimesgradg, fct1, fct2, alphaX, alphaY, betaX, betaY)])
			#Terme d'intégration sur les bords de Neumann
			if idNode1 in self2.listNodeNeumann:
				# print("IN NEUMANN NODE")
				xNode, _ = self2.getCoordNode(idNode1)
				xNeumann = xNode
				# self2.A[idNode1, idNode2] += self2.he*np.abs(betaX*betaY)*self2.quadIntegration1D(ftimesg, fct1.evaluateX(xNeumann), fct2.evaluateX(xNeumann), alphaX, alphaY, betaX, betaY)
				resA.append([idNode1, idNode2, self2.he*np.abs(betaX*betaY)*self2.quadIntegration1D(ftimesg, fct1.evaluateX(xNeumann), fct2.evaluateX(xNeumann), alphaX, alphaY, betaX, betaY)])
		#Calcul de la constante du terme de droite
		cte = Polynome(0., 0., 0., self2.f[idCell])
		#Calcul du terme de droite par intégration
		# self2.b[idNode1] += np.abs(betaX*betaY)*self2.quadIntegration(ftimescte, fct1, cte, alphaX, alphaY, betaX, betaY)
		resb.append([idNode1, np.abs(betaX*betaY)*self2.quadIntegration(ftimescte, fct1, cte, alphaX, alphaY, betaX, betaY)])
		#Terme d'intégration sur les bords de Neumann
		if idNode1 in self2.listNodeNeumann:
			xNode, _ = self2.getCoordNode(idNode1)
			xNeumann = xNode
			cte = Polynome(0., 0., 0., self2.getNeumannCond(idNode1))
			# self2.b[idNode1] += self2.he*np.abs(betaX*betaY)*self2.quadIntegration1D(ftimescte, fct1.evaluateX(xNeumann), cte.evaluateX(xNeumann), alphaX, alphaY, betaX, betaY)
			resb.append([idNode1, self2.he*np.abs(betaX*betaY)*self2.quadIntegration1D(ftimescte, fct1.evaluateX(xNeumann), cte.evaluateX(xNeumann), alphaX, alphaY, betaX, betaY)])
	return [resA, resb]