CXX=g++

## FIXME add back -fopenmp

INCLUDES=-I/Users/jonas/anaconda/envs/py36/include/eigen3/
CPPFLAGS:= -O3 -fPIC  -march=native -shared -std=c++0x -pedantic -Wall -Wshadow -Wpointer-arith -Wcast-qual -Wstrict-prototypes -Wmissing-prototypes -mavx -g3 -larmadillo -I/data/vaishaal/eigen/ $(INCLUDES)
RM= rm -f
LDFLAGS= -shared
.PHONY: all clean


all: fastfood.so

fastfood.so: fastfood.o
	$(CXX) -shared $< -o $@

clean:
	rm -rf *.so
	rm -rf *.o



