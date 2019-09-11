from numpy import *

class LbfgsIterData:
	# Klasa za cuvanje trenutnih podataka iteracije
	# Alpha vrednost trenutne iteracije
	alpha = 0
	# Razlika x(i+1) i x(i)
	s = []
	# Razlika g(i+1) i g(i)
	y = []
	# skalarni proizvod y i s
	ys = 0

	# Postavi alpha i ys
	def __init__(self, _alpha, _ys):
		self.alpha = _alpha
		self.ys = _ys

class LbfgsClass:
	# Klasa za parametre LBFGS minimizacije

	# Umesto inverzne Hesian Hk,
	# L-BFGS cuva istoriju prethodnih m promena pozicije x i gradijenta ∇f(x),
	# Generalno, m je mali broj (m < 10)
	m = 6

	# Epsilon je parametar za test konvergencije
	epsilon = 1e-5

	# Display je parametar za stampanje privremenih rezultata i informacija
	#0 - ne stampa, != 0 stampa
	display = 0

	# Maksimalni broj iteracija
	# Ukoliko je 0, minimizacija se nastavlja do kraja ili do greske
	# Ukoliko je postavljena vrednost veca od 0, minimizacija se zaustavlja pri dostizanju maksimalne iteracije
	max_iter = 0

	# Makimalni broj procena vrednosti funkcije i gradijenata po iteraciji
	max_linesearch = 40

	# Minimalna vrednost koraka za procenu vrednosti
	# Linesearch prekida ukoliko je dostigne ovu vrednost
	min_linesearch_step = 1e-20

	# Maksimalna vrednost koraka za procenu vrednosti
	# Linesearch prekida ukoliko je dostigne ovu vrednost
	max_linesearch_step = 1e20

	# Ftol kontrolise tacnost rutine procene vrednosti
	# Ovaj parametar bi trebao biti veco od 0 a manji od 0.5
	ftol = 1e-4

	# Koeficijent za Wolfe uslov.
	# Trebao bi biti veci od ftol a manji od 1.0
	wolfe = 0.9

	def linesearch(self, fun, x, grad, dirs, fx, step, x_prev):
	# Ovaj algoritam osigurava da su pravci pravilno skalirani
	# a samim tim je duzina koraka jedinice prihvacena u vecini iteracija
	# Wolfe algoritam procene je koriscen
	# Argumenti:
	# self - klasa
	# fun - funkcija za procenu vrednosti i gradijenta
	# x - trenutne vrednosti pozicija
	# grad - trenutne vrednosti gradijenata
	# dirs - trenutni pravci
	# fx - trenutna vrednost funkcije
	# step - trenutna duzina koraka
	# x_prev - prethodne vrednosti pozicija

		dec = 0.5
		inc = 2.1
		count = 0
		# Provera greske ulaznih parametara
		res = {'status':0,'fx':fx,'step':step,'x':x}
		if (step <= 0.):
			res['status'] = -1
			print('[GRESKA] Korak iteracije je manji od 0')
			return res

		# Racunanje inicijalnog gradijenta u proceni pravca
		dginit = dot(grad, dirs)
		if dginit > 0:
			print('[GRESKA] Inicijalni gradijent je veci od 0')
			res['status'] = -1
			return res

		# Inicijalna vrednost funkcije
		finit = fx
		dgtest = self.ftol * dginit

		while True:
			x = x_prev
			x = x + dirs * step;

			# Procena funkcije i gradijenta
			# Gradijent se menja
			fx = fun(x, grad)

			count = count + 1

			# Provera dovoljnog uslova za smanjenje (Armijo uslov)
			if fx > finit + (step * dgtest):
				width = dec
			else:
				# Provera Wolfe uslova
				# Sada je gradijent od f(xk + step * d)
				dg = dot(grad, dirs)
				if dg < self.wolfe * dginit:
					width = inc
				else:
					res = {'status':0, 'fx':fx, 'step':step, 'x':x}
					return res
			# Ako je trenutni korak manji od minimuma
			if step < self.min_linesearch_step:
				res['status'] = -1
				print('[GRESKA] Dostignut minimalni korak iteracije: step < min_linesearch_step')
				return res
			# Ako je trenutni korak veci od maksimuma
			if step > self.max_linesearch_step:
				res['status'] = -1
				print('[GRESKA] Dostignut maksimalni korak iteracije: step > max_linesearch_step')
				return res
			# Ako je broj iteracija dostigao maksimalnu vrednost
			if self.max_linesearch <= count:
				res = {'status':-1, 'fx':fx, 'step':step, 'x':x}
				print('[GRESKA] Dostignut maksimalni broj iteracija: cnt > max_linesearch')
				return res
			# Azuriranje koraka
			step = step * width

	def minimize(self, n, x, function):
	# Minimize je funkcija koja vrsi Limited-memory BFGS optimizaciju
	# Argumenti:
	# self - klasa sa parametrima
	# n - velicina niza
	# x - inicijalne vrednosti pozicija
	# function - funkcija za procenu vrednosti i gradijenata

		# Alokacija radnog prostora
		status = 0
		x_prev = array([0.0 for i in range(n)])
		grad = array([0.0 for i in range(n)])
		grad_prev = array([0.0 for i in range(n)])
		directions = array([0.0 for i in range(n)])
		fx = 0

		# Alokacija ogranicenog memorijskog prostora
		lim_mem = []
		for i in range(0, self.m):
			lim_mem.append(LbfgsIterData(0,0))
			lim_mem[i].s = array([0.0 for n in range(n)])
			lim_mem[i].y = array([0.0 for n in range(n)])

		# Procena vrednosti funkcije i gradijenata
		fx = function(x, grad)

		# Racunanje pravaca;
		# (Inicijalna hessian - I)
		# dk = -Hk*gk
		directions = -grad

		# Provera da li su inicijalne vrednosti zapravo konacne
		x_norm = sqrt(dot(x, x))
		grad_norm = sqrt(dot(grad, grad))

		if x_norm < 1.0:
			x_norm = 1.0

		if grad_norm / x_norm <= self.epsilon:
			if self.display != 0:
				print('[INFO] Vec je minimizirano')
			return  status, fx

		# Racunanje inicijalnog koraka
		step = 1.0 / sqrt(dot(directions, directions))


		k = 1
		end = 0
		i = 1
		while True:
			# Skladistenje trenutnih pozicija i gradijenata
			x_prev = x.copy()
			grad_prev = grad.copy()

			# Obezvedjivanje da su pravci pretrage dobro skalirani
			ls = self.linesearch(function, x, grad, directions, fx, step, x_prev)
			if ls['status'] < 0:
				status = ls['status']
				x = x_prev.copy()
				grad = grad_prev.copy()
				fx = ls['fx']
				if self.display != 0:
					print ('[GRESKA] Vracanje tacaka na prethodne')
				return status, fx


			fx = ls['fx']
			step = ls['step']
			x = ls['x']

			# Racunanje normi pozicija i gradijenata
			x_norm = sqrt(dot(x, x))
			g_norm = sqrt(dot(grad, grad))

			# Informacije o progresu
			if self.display != 0:
				print("[INFO] Iteracija", k,":")
				print("[INFO] fx = ", fx,", x[0] = ", x[0],", x[1] = ", x[1])
				print("[INFO] xnorm = ", x_norm, ", gnorm = ", g_norm,", step = ", step)
				print("\n")

			# Test konvergencije
			if x_norm < 1.0:
				x_norm = 1.0
			if g_norm / x_norm <= self.epsilon:
				if self.display != 0:
					print ('[INFO] Completed lbfgs')
				return status, fx

			# Provera dostizanja maksimalne vrednosti iteracija
			if self.max_iter != 0 and self.max_iter < (k+1):
				if self.display != 0:
					printf("[INFO] Maksimalna vrednost iteracije je dostignuta -> exit")
				status = -2
				return status, fx

			# Azuriranje vektora s i y:
			# s_{k+1} = x_{k+1} - x_{k} = \step * d_{k}.
			# y_{k+1} = g_{k+1} - g_{k}.
			data = lim_mem[end]
			data.s = x - x_prev
			data.y = grad - grad_prev

			# Racunanje skalara ys i yy:
			# ys = y^t \cdot s = 1 / \rho.
			# yy = y^t \cdot y.
			# yy se koristi za skaliranje hesian matrice H_0 (Cholesky faktor)
			ys = dot(data.y, data.s)
			yy = dot(data.y, data.y)
			data.ys = ys

			bound = (self.m <= k and [self.m] or [k])[0]
			k = k + 1
			# Dohvati sledeci u poslednjih m iteracione liste
			end = (end + 1) % self.m

			# Izracunavanje najstrmijeg pravca
			# Racunanje negativnog gradijenta
			directions = -grad
			j = end
			for i in range(0, bound):
				# Od kasnijeg ka prethodnom
				j = (j + self.m -1) % self.m
				data = lim_mem[j]
				data.alpha = dot(data.s, directions) / data.ys
				directions = directions + (data.y * (-data.alpha))

			directions = directions * (ys/yy)

			for i in range(0, bound):
				data = lim_mem[j]
				beta = dot(data.y, directions)
				beta = beta / data.ys
				directions = directions + (data.s * (data.alpha - beta))
				# Od prethodnog ka kasnijem
				j = (j + 1) % self.m
			# Sada su pravci pretrage spremni. Pokusavamo prvo za step = 1
			step = 1.0