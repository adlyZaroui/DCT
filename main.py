#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from math import sqrt, cos, pi
from DCT import *

def main(imgLoc): #Prend chemin d'une image et renvoi la liste de la matrice compressee et la matrice decompressee

	A = initialisation(imgLoc)

	R = compression(A[0])[0]
	G = compression(A[1])[0]
	B = compression(A[2])[0]

	C = rassembler(R,G,B) #Matrice compressee

	Rd = decompression(R)
	Gd = decompression(R)
	Bd = decompression(B)

	D = rassembler(Rd,Gd,Bd) #Matrice decompressee

	C = np.array(C,np.uint8) #On force C et D a prendre des valeurs entieres : type = uint8
	D = np.array(D,np.uint8)

	return [C,D]
