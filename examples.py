from numpy import *
from lbfgs import *
from ctypes import *

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

# Fleg za testiranje
testing = 0
# Kreiranje podrazumevanih parametara
l = LbfgsClass()
#l.display = 1
# Prvi primer
N = 100
# Alokacija i inicijalizacija nizova
x = array([0.0 for i in range(N)])
for i in range(0, N, 2):
	x[i] = -1.2
	x[i+1] = 1.0

status, fx = l.minimize(N, x, fun0)
if status == 0:
	print("Miniminizirana vrednost = ", fx)
else:
	print("Greska u pokusaju minimizacije funkcije")

# Uporedjivanje sa referentnom C bibliotekom
if testing == 1:
	libCalc = CDLL("./libc_to_py.so")
	lbfgs_c = libCalc.c_lbfgs_0
	lbfgs_c.restype = c_float
	c_res = lbfgs_c()
	print ("lbfgs_c = ", c_res)
	print ("lbfgs_py = ", fx)
	rel_tol=1e-09
	if abs(c_res - fx) < rel_tol:
		print("Tacnost potvrdjena sa C bibliotekom")
	else :
		print("Postoje neslaganje izmedju pajton i c racunice")

# Drugi primer
N = 2
x1 = array([0.0 for i in range(N)])
for i in range(0, N, 2):
	x1[i] = -1.2
	x1[i+1] = 1.0
status, fx = l.minimize(N, x1, fun1)
if status == 0:
	print("Miniminizirana vrednost = ", fx)
else:
	print("Greska u pokusaju minimizacije funkcije")

# Uporedjivanje sa referentnom C bibliotekom
if testing == 1:
	libCalc = CDLL("./libc_to_py.so")
	lbfgs_c = libCalc.c_lbfgs_1
	lbfgs_c.restype = c_float
	c_res = lbfgs_c()
	print ("lbfgs_c = ", c_res)
	print ("lbfgs_py = ", fx)
	rel_tol=1e-09
	if abs(c_res - fx) < rel_tol:
		print("Tacnost potvrdjena sa C bibliotekom")
	else :
		print("Postoje neslaganje izmedju pajton i c racunice")

# Treci primer
N = 2
x1 = array([0.0 for i in range(N)])
for i in range(0, N, 2):
	x1[i] = -1.2
	x1[i+1] = 1.0
status, fx = l.minimize(N, x1, fun2)
if status == 0:
	print("Miniminizirana vrednost = ", fx)
else:
	print("Greska u pokusaju minimizacije funkcije")

# Uporedjivanje sa referentnom C bibliotekom
if testing == 1:
	libCalc = CDLL("./libc_to_py.so")
	lbfgs_c = libCalc.c_lbfgs_2
	lbfgs_c.restype = c_float
	c_res = lbfgs_c()
	print ("lbfgs_c = ", c_res)
	print ("lbfgs_py = ", fx)
	rel_tol=1e-09
	if abs(c_res - fx) < rel_tol:
		print("Tacnost potvrdjena sa C bibliotekom")
	else :
		print("Postoje neslaganje izmedju pajton i c racunice")