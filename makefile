Compiler := icpc
MakeFlags := -mkl -std=c++17 -Wall -Wextra -Wconversion -Wshadow -Werror -O3
Objects := matrix.o general.o pes.o main.o
HeaderFile := matrix.h general.h pes.h

all: dvr

dvr: ${Objects}
	${Compiler} ${Objects} ${MakeFlags} -o dvr
main.o: main.cpp ${HeaderFile}
	${Compiler} -c main.cpp ${MakeFlags} -g -o main.o
pes.o: pes.cpp ${HeaderFile}
	${Compiler} -c pes.cpp ${MakeFlags} -g -o pes.o
general.o: general.cpp ${HeaderFile}
	${Compiler} -c general.cpp ${MakeFlags} -g -o general.o
matrix.o: matrix.cpp matrix.h
	${Compiler} -c matrix.cpp ${MakeFlags} -g -o matrix.o

.PHONY: clean
clean:
	-rm *.o

.PHONY: clean_result
clean_result:
	-rm log output *.txt *.gif

.PHONY: git
git:
	git add *.h *.cpp makefile *.sh *.py .gitignore

