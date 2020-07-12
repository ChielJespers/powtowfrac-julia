all:
	nvcc -g -o powtowfrac main.cu -lgd -lm -ldl