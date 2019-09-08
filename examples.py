from numpy import *
from lbfgs import *

def fun(x, grad):
	fx = 0
	for i in range(0, N, 2):
		t1 = 1.0 - x[i]
		t2 = 10.0 * (x[i+1] - x[i] * x[i])
		grad[i+1] = 20.0 * t2
		grad[i] = -2.0 * (x[i] * grad[i+1] + t1)
		fx += t1 * t1 + t2 * t2;
	return fx

# Kreiranje podrazumevanih parametara
l = LbfgsClass()
l.display = 1
# Prvi primer
N = 100
# Alokacija i inicijalizacija nizova
x = array([0.0 for i in range(N)])
for i in range(0, N, 2):
	x[i] = -1.2
	x[i+1] = 1.0

status, fx = l.minimize(N, x, fun)
if status == 0:
	print("Miniminizirana vrednost = ", fx)
else:
	print("Greska u pokusaju minimizacije funkcije")