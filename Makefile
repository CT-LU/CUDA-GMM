all:
	nvcc main.cu -L/usr/lib -I./ `pkg-config --cflags opencv` `pkg-config --libs opencv` -o gmmWithcuda

clean:	
	rm -f gmmWithcuda *.o
