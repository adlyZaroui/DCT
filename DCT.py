#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from math import sqrt, cos, pi

Q = np.array([[16,11,10,16,24,40,51,61],
             [12,12,13,19,26,58,60,55],
             [14,13,16,24,40,57,69,56],
             [14,17,22,29,51,87,80,62],
             [18,22,37,56,68,109,103,77],
             [24,35,55,64,81,104,113,92],
             [49,64,78,87,103,121,120,101],
             [72,92,95,98,112,100,103,99]])

#INITIALISATION

def imgToArray(imgLoc): #Prend le chemin d'une image et retourne la matrice m*p*3 associee
	img = mpimg.imread(imgLoc)
	return np.array(img)

def arrayToImg(A): #Prend matrice n*p*3 de pixels et affiche (sauvegarde) l'image
	A = np.array(A,np.uint8)
	plt.imshow(A)
	plt.show()

def separer(A): #Prend matrice M(n,p,3) et retourne la liste des 3 matrices M(n,p)
	n = np.shape(A)[0]
	p = np.shape(A)[1]
	R = np.zeros((n,p))
	G = np.zeros((n,p))
	B = np.zeros((n,p))
	for i in range(n):
		for j in range(p):
			R[i,j] = A[i,j][0]
			G[i,j] = A[i,j][1]
			B[i,j] = A[i,j][2]
	return [R,G,B]

def rassembler(A,B,C): #Prend 3 matrices n*p et retourne la matrice n*p*3 (reciprque de sepaer())
	return np.dstack((A,B,C))

def tronquer(A): #Prend matrice M(n,p) et la coupe tronque en multiple de 8 en n et p
	n = np.shape(A)[0]
	p = np.shape(A)[1]
	n1 = n - n%8
	p1 = p - p%8
	return A[:n1,:p1]

def formatPourcent(A): #Prend matrice M(n,p) de pixels et renvoi True si le format est entre 0 et 255, Faux si le format est entre 0 et 1
	n = np.shape(A)[0]
	p = np.shape(A)[1]
	for i in range(n):
		for j in range(p):
			if A[i,j] > 1 :
				return False
	return True

def reduire(A): #Prend matrice M(n,p) de pixels (quelque soit le format) et renvoi la matrice de pixels dont le format est reduit (0--255)
	n = np.shape(A)[0]
	p = np.shape(A)[1]
	if formatPourcent(A) == False :
		return A
	else:
		for i in range(n):
			for j in range(p):
				A[i,j] = int(255 * A[i,j])
	return A

def centrer(A): #Prend matrice M(n,p) de pixels au format 255  et renvoi la matrice des valeurs centrees entre -128 et 127
	n = np.shape(A)[0]
	p = np.shape(A)[1]
	for i in range(n):
		for j in range(p):
			A[i,j] = A[i,j] - 128
	return A


def initialisation(imgLoc): #Prend chemin d'une image et retourne la liste [R,G,B] des matrices dex pixels centrees reduits et tronquees
	A = imgToArray(imgLoc)
	[R,G,B] = separer(A)
	R = centrer(reduire(tronquer(R)))
	G = centrer(reduire(tronquer(G)))
	B = centrer(reduire(tronquer(B)))

	return [R,G,B]

#COMPRESSION


def passage(): #Ne Prend aucun argument et retourne la matrice de passage 8*8 constante de la DCT II
        P = np.zeros((8,8))
        for i in range(8):
                for j in range(8):
                        if i == 0 :
                                P[i,j] = 1/sqrt(8)
                        P[i,j] = 0.5 * cos(1/16 * (2*j+1) * i * pi)
        return P

def bloc(A,i,j): #Prend une matrice dont les lignes et colonnes sont multiples de 8 et renvoi la sous-matrice 8*8 correspondante au i-je bloc 8*8

	try:
		B = A[8*i:8*(i+1),8*j:8*(j+1)]
	except:
		print("pas de bloc correspondant")
	return B

def chgtBaseOrto(A): #Prend matrice A 8*8 et retourne P*A*Ptranspose avec P matrice de chgt base orthogonale
	return np.dot(np.dot(passage(), A), passage().T)

def chgtBaseInv(A): #Prend A 8*8 et retourne Ptransposee*A*P
	return np.dot(np.dot(passage().T, A),passage())

def quantifier(A): #Prend A 8*8 et retourne A./Q
	return np.floor_divide(A,Q)

def compression(A): #Prend une matrice A n*p de pixels et renvoi la liste la  matrice n*p compressee, et le nomrbre de coef non nuls
	n = np.shape(A)[0]
	p = np.shape(A)[1]
	C = np.zeros((n,p))
	nbCoefNonNul = 0

	for i in range(0,n,8): # On construit la matrice C n*p des blocs 8*8 compressees
		for j in range(0,p,8):
			C[i:i+8,j:j+8]=quantifier(chgtBaseOrto(A[i:i+8,j:j+8]))
			if C[i,j] != 0: nbCoefNonNul = nbCoefNonNul + 1

	return [C, nbCoefNonNul]

#DECOMPRESSION

def dequantification(A): #Prend A 8*8 et renvoi A .* Q
	return A*Q

def decompression(A): #Prend matrice A n*p compressee et renvoi la matrice n*p decompressee
	n = np.shape(A)[0]
	p = np.shape(A)[1]
	C = np.zeros((n,p))

	for i in range(0,n,8): #On construit la matrice C n*p des blocs 8*8 decompresses
		for j in range(0,p,8):
			#C[i:i+8,j:j+8] = ChgtBaseInv(dequantification(A[i:i+8,j:j+8]))
			C[i:i+8,j:j+8] = np.dot(np.dot(passage().T, dequantification(A[i:i+8,j:j+8])),passage())
	return C

#POST-PROCESSING

def post_proccessing(A,B): #Prend deux matrices de n*p de pixels centrees reduits, et retourne la liste composee dans cet ordre
                           #de la matrice A et B decentree, dereduits, et leurs normes avant transformation
	nA = np.shape(A)[0]
	pA = np.shape(A)[1]
	nB = np.shape(B)[0]
	pB = np.shape(B)[1]
	normA = np.linalg.norm(A)
	normB = np.linalg.norm(B)

	for i in range(n):
		for j in range(p):
			A[i,j] = (A[i,j] + 128)/255
			B[i,j] = (B[i,j] + 128)/255

	return [A,B,normA,normB]
