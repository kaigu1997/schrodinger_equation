#!/bin/bash
make
make clean
for (( i=1;i<=350;i=i+1 ))
do
    p=$(echo "scale=1;$i/10.0"|bc)
    sigmap=$(echo "scale=3;$p/20.0"|bc)
    cat > input << END_FILE
mass:
2000.0
x0:
-10.0
p0:
$p
sigma p:
$sigmap
Left boundary:
-20.0
Right boundary:
20.0
Absorb potential: (on, off)
off
Upper limit of dx:
1.0
Upper limit of dt:
0.1
Total time of evolution:
5000.0
Output period:
1.0
Phase space output period:
1000.0
Representation:(diabatic, adiabatic, force)
diabatic
END_FILE
    ./dvr >> output 2>>log
done

