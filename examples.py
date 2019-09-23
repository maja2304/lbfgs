from numpy import *
from lbfgs import *
from ctypes import *
import time

def fun0(x, grad):
	fx = 0
	for i in range(0, N, 2):
		x1 = 1.0 - x[i]
		x2 = 10.0 * (x[i+1] - x[i] * x[i])
		grad[i+1] = 20.0 * x2
		grad[i] = -2.0 + 2.0 * x[i] - 200.0 * ((x[i+1] - x[i] * x[i])) * 2.0 * x[i]
		fx += x1 * x1 + x2 * x2
	return fx

def fun1(x, grad):
	fx = 0
	x1 = x[0]
	x2 = x[1]
	grad[0] = -400 * x1 * (x2 - x1 * x1) - 2 * (1 - x1)
	grad[1] = 200 * (x2 - x1 * x1)
	fx = 100 * (x2 - x1 * x1) * (x2 - x1 * x1) + (1 - x1) * (1 - x1)
	return fx

def fun2(x, grad):
	fx = 0
	x1 = x[0]
	x2 = x[1]
	grad[0] = 2*(x1 + 2*x2 - 7.0) + 2*(2*x1 + x2 - 5.0)*2;
	grad[1] = 2*(x1 + 2*x2 - 7.0)*2 + 2*(2*x1 + x2 - 5.0);
	fx = (x1 + 2*x2 - 7.0) * (x1 + 2*x2 - 7.0) + (2*x1 + x2 - 5.0) * (2*x1 + x2 - 5.0);
	return fx

def fun3(x, grad):
	fx = 0
	x1 = x[0]
	x2 = x[1]
	grad[0] = 40 * x1
	grad[1] = 200 * x2
	fx = 20 * x1 * x1 + 100 * x2 * x2
	return fx

def fun4(x, grad):
	fx = 0
	x1 = x[0]
	x2 = x[1]
	grad[0] = 2 * x1 + x2
	grad[1] = 3 * x2 * x2 + x1
	fx = x1 * x1 + x2 * x2 * x2 + x1 * x2
	return fx

# Boje
BOLD = "\033[1m"
ENDC = '\033[0m'
OKGREEN = '\033[92m'
FAIL = '\033[91m'

# Fleg za testiranje [SAMO SA IZBILDOVANOM BIBLIOTEKOM - Pogledati README]
testing = 1
# Kreiranje podrazumevanih parametara
l = LbfgsClass()
#l.display = 1
# Prvi primer
print("1. Funkcija")
N = 100
# Alokacija i inicijalizacija nizova
x = array([0.0 for i in range(N)])
for i in range(0, N, 2):
	x[i] = -1.2
	x[i+1] = 1.0

start = time.monotonic()
status, fx = l.minimize(N, x, fun0)
end = time.monotonic()
if status != 0:
	print("Greska u pokusaju minimizacije funkcije")
else:
	print("Vreme izvrsavanja funkcije sa {} parametara: {} [ms]".format(N, (end - start) * 1000))
	if testing == 0:
		print("Rezultat LBFGS minimizacije = ", fx)

# Uporedjivanje sa referentnom C bibliotekom
if testing == 1:
	libCalc = CDLL("./libc_to_py.so")
	lbfgs_c = libCalc.c_lbfgs_0
	lbfgs_c.restype = c_float
	c_res = lbfgs_c()
	print ("Referentna c biblioteka LBFGS = ", c_res)
	print ("Python implementacija LBFGS = ", fx)
	rel_tol=1e-09
	if abs(c_res - fx) < rel_tol:
		print(OKGREEN, BOLD, "Tacnost potvrdjena sa C bibliotekom", ENDC)
	else :
		print(FAIL, BOLD,"Postoje neslaganje izmedju pajton i c racunice", ENDC)

if 0:
	print("--------------------------------------------------")
	print("2. Funkcija")
	# Drugi primer
	N = 2
	x = array([0.0 for i in range(N)])
	for i in range(0, N, 2):
		x[i] = -1.2
		x[i+1] = 1.0

	start = time.monotonic()
	status, fx = l.minimize(N, x, fun1)
	end = time.monotonic()
	if status != 0:
		print("Greska u pokusaju minimizacije funkcije")
	else:
		print("Vreme izvrsavanja funkcije: {} [ms]".format((end - start) * 1000))
		if testing == 0:
			print("Rezultat LBFGS minimizacije = ", fx)

	# Uporedjivanje sa referentnom C bibliotekom
	if testing == 1:
		libCalc = CDLL("./libc_to_py.so")
		lbfgs_c = libCalc.c_lbfgs_1
		lbfgs_c.restype = c_float
		c_res = lbfgs_c()
		print ("Referentna c biblioteka LBFGS = ", c_res)
		print ("Python implementacija LBFGS = ", fx)
		rel_tol=1e-09
		if abs(c_res - fx) < rel_tol:
			print(OKGREEN, BOLD, "Tacnost potvrdjena sa C bibliotekom", ENDC)
		else :
			print(FAIL, BOLD,"Postoje neslaganje izmedju pajton i c racunice", ENDC)

	print("--------------------------------------------------")
	print("3. Funkcija")
	# Treci primer
	N = 2
	x = array([0.0 for i in range(N)])
	for i in range(0, N, 2):
		x[i] = -1.2
		x[i+1] = 1.0

	start = time.monotonic()
	status, fx = l.minimize(N, x, fun2)
	end = time.monotonic()
	if status != 0:
		print("Greska u pokusaju minimizacije funkcije")
	else:
		print("Vreme izvrsavanja funkcije: {} [ms]".format((end - start) * 1000))
		if testing == 0:
			print("Rezultat LBFGS minimizacije = ", fx)

	# Uporedjivanje sa referentnom C bibliotekom
	if testing == 1:
		libCalc = CDLL("./libc_to_py.so")
		lbfgs_c = libCalc.c_lbfgs_2
		lbfgs_c.restype = c_float
		c_res = lbfgs_c()
		print ("Referentna c biblioteka LBFGS = ", c_res)
		print ("Python implementacija LBFGS = ", fx)
		rel_tol=1e-09
		if abs(c_res - fx) < rel_tol:
			print(OKGREEN, BOLD, "Tacnost potvrdjena sa C bibliotekom", ENDC)
		else:
			print(FAIL, BOLD,"Postoje neslaganje izmedju pajton i c racunice", ENDC)

	print("--------------------------------------------------")
	print("4. Funkcija")
	# Cetvrti primer
	N = 2
	x = array([0.0 for i in range(N)])
	for i in range(0, N, 2):
		x[i] = -1.2
		x[i+1] = 1.0

	start = time.monotonic()
	status, fx = l.minimize(N, x, fun3)
	end = time.monotonic()
	if status != 0:
		print("Greska u pokusaju minimizacije funkcije")
	else:
		print("Vreme izvrsavanja funkcije: {} [ms]".format((end - start) * 1000))
		if testing == 0:
			print("Rezultat LBFGS minimizacije = ", fx)

	# Uporedjivanje sa referentnom C bibliotekom
	if testing == 1:
		libCalc = CDLL("./libc_to_py.so")
		lbfgs_c = libCalc.c_lbfgs_3
		lbfgs_c.restype = c_float
		c_res = lbfgs_c()
		print ("Referentna c biblioteka LBFGS = ", c_res)
		print ("Python implementacija LBFGS = ", fx)
		rel_tol=1e-09
		if abs(c_res - fx) < rel_tol:
			print(OKGREEN, BOLD, "Tacnost potvrdjena sa C bibliotekom", ENDC)
		else :
			print(FAIL, BOLD,"Postoje neslaganje izmedju pajton i c racunice", ENDC)

	print("--------------------------------------------------")
	print("5. Funkcija")
	# Peti primer
	N = 2
	x = array([0.0 for i in range(N)])
	for i in range(0, N, 2):
		x[i] = -1.2
		x[i+1] = 1.0

	start = time.monotonic()
	status, fx = l.minimize(N, x, fun4)
	start = time.monotonic()
	if status != 0:
		print("Greska u pokusaju minimizacije funkcije")
	else:
		print("Vreme izvrsavanja funkcije: {} [ms]".format((end - start) * 1000))
		if testing == 0:
			print("Rezultat LBFGS minimizacije = ", fx)

	# Uporedjivanje sa referentnom C bibliotekom
	if testing == 1:
		libCalc = CDLL("./libc_to_py.so")
		lbfgs_c = libCalc.c_lbfgs_4
		lbfgs_c.restype = c_float
		c_res = lbfgs_c()
		print ("Referentna c biblioteka LBFGS = ", c_res)
		print ("Python implementacija LBFGS = ", fx)
		rel_tol=1e-09
		if abs(c_res - fx) < rel_tol:
			print(OKGREEN, BOLD, "Tacnost potvrdjena sa C bibliotekom", ENDC)
		else :
			print(FAIL, BOLD,"Postoje neslaganje izmedju pajton i c racunice", ENDC)