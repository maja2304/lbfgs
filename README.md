# lbfgs

python3 examples.py


# Uporedjivanje sa referentnom c bibliotekom
1. Ukljuciti potrebni fleg u examples.py - testing = 1
2. cd c_lib
3. cc -fPIC -shared -o libc_to_py.so sample.c liblbfgs.a -lm
4. cp libc_to_py.so ..
5. cd ..
6. python3 examples.py