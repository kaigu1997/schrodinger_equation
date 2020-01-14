compiler = icpc
cpl_cmd := -mkl -std=c++17 -Wall -O3 -m64

dvr: matrix.o general.o main.o
	${compiler} main.o general.o matrix.o ${cpl_cmd} -o dvr
main.o: main.cpp matrix.h general.h
	${compiler} -c main.cpp ${cpl_cmd} -g -o main.o
general.o: general.cpp general.h matrix.h
	${compiler} -c general.cpp ${cpl_cmd} -g -o general.o
matrix.o: matrix.cpp matrix.h
	${compiler} -c matrix.cpp ${cpl_cmd} -g -o matrix.o

clean:
	rm *.o log output *.txt
