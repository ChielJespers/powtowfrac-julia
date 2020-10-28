all:
	nvcc -g -o powtowfrac main.cu -lgd -lm -ldl
shared:
	nvcc -Xcompiler -fPIC -shared -o powtowfrac.so parametrized.cu