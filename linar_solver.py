from scipy.optimize import fsolve
from scipy.optimize import leastsq
import numpy as np

#EXAMPLE https://www.youtube.com/watch?v=44pAWI7v5Zk&ab_channel=APMonitor.com
#equations
#3x - 9y = -42
#2x + 4y = 2
#
#  A    .  z  =   b
#|3 -9| . |x| = |-42|
#|2  4|   |y|   | 2 |
#
#z = A^-1 . b

#--------------------------- EXEMPLE
def example():
    print('example:')
    A = np.array([[3,-9], [2,4]])
    print("A: \n", A)
    b = np.array([-42, 2])
    print("b: \n", b)

    z = np.linalg.solve(A,b)
    print("result: ", z)

def exampleEquations(x):
    return [
        3*x[0] - 9*x[1] + 42,
        2*x[0] + 4*x[1] - 2
    ]

#example()
#solution = fsolve(exampleEquations, [0, 0])
#solution, info  = leastsq(exampleEquations, [0.0, 0.0])

#print("solution:", solution)
#print("Informações adicionais:", info)
#--------------------------- EXEMPLE

motionU = np.array([[-4.59611045, -8.26642521, -3.89229999], [-3.91950256, -8.08813409, -1.1451819 ], [-3.639234, -8.07157967, 0.08884564]])
motionV = np.array([[-9.16207003, 3.50904357, -0.23679604], [-9.26015295, 3.35513799, 0.18129132], [-9.26616895, 3.10488608, 0.10746241]])

# motionU = np.transpose(motionU)
# motionV = np.transpose(motionV)

print('motionU: ')
print(motionU)

print('motionV: ')
print(motionV)

u1 = motionU[0]
u2 = motionU[1]
u3 = motionU[2]
v1 = motionV[0]
v2 = motionV[1]
v3 = motionV[2]

x1 = motionU[0][0]
y1 = motionU[1][0]
z1 = motionU[2][0]
r1 = motionV[0][0]
s1 = motionV[1][0]
t1 = motionV[2][0]

x2 = motionU[0][1]
y2 = motionU[1][1]
z2 = motionU[2][1]
r2 = motionV[0][1]
s2 = motionV[1][1]
t2 = motionV[2][1]

x3 = motionU[0][2]
y3 = motionU[1][2]
z3 = motionU[2][2]
r3 = motionV[0][2]
s3 = motionV[1][2]
t3 = motionV[2][2]

def equations(q):
    return [
        (1*q[0]) + (1*q[1]) + (1*q[2]) + (1*q[3]) + (1*q[4]) + (1*q[5]) + (1*q[6]) + (1*q[7]) + (1*q[8]) - 1,
        (1*q[0]) + (1*q[1]) + (1*q[2]) + (1*q[3]) + (1*q[4]) + (1*q[5]) + (1*q[6]) + (1*q[7]) + (1*q[8]) - 1,
        (1*q[0]) + (1*q[1]) + (1*q[2]) + (1*q[3]) + (1*q[4]) + (1*q[5]) + (1*q[6]) + (1*q[7]) + (1*q[8]),
        (1*q[0]) + (1*q[1]) + (1*q[2]) + (1*q[3]) + (1*q[4]) + (1*q[5]) + (1*q[6]) + (1*q[7]) + (1*q[8]) - 1,
        (1*q[0]) + (1*q[1]) + (1*q[2]) + (1*q[3]) + (1*q[4]) + (1*q[5]) + (1*q[6]) + (1*q[7]) + (1*q[8]) - 1,
        (1*q[0]) + (1*q[1]) + (1*q[2]) + (1*q[3]) + (1*q[4]) + (1*q[5]) + (1*q[6]) + (1*q[7]) + (1*q[8]),
        (1*q[0]) + (1*q[1]) + (1*q[2]) + (1*q[3]) + (1*q[4]) + (1*q[5]) + (1*q[6]) + (1*q[7]) + (1*q[8]) - 1,
        (1*q[0]) + (1*q[1]) + (1*q[2]) + (1*q[3]) + (1*q[4]) + (1*q[5]) + (1*q[6]) + (1*q[7]) + (1*q[8]) - 1,
        (1*q[0]) + (1*q[1]) + (1*q[2]) + (1*q[3]) + (1*q[4]) + (1*q[5]) + (1*q[6]) + (1*q[7]) + (1*q[8])
    ]