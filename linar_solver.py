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

u1 = motionU[0]
u2 = motionU[1]
u3 = motionU[2]
v1 = motionV[0]
v2 = motionV[1]
v3 = motionV[2]

# print('motionU: ')
# print(motionU)

# print('motionV: ')
# print(motionV)

x1 = u1[0]
y1 = u1[1]
z1 = u1[2]
r1 = v1[0]
s1 = v1[1]
t1 = v1[2]

x2 = u2[0]
y2 = u2[1]
z2 = u2[2]
r2 = v2[0]
s2 = v2[1]
t2 = v2[2]

x3 = u3[0]
y3 = u3[1]
z3 = u3[2]
r3 = v3[0]
s3 = v3[1]
t3 = v3[2]

# u^-1(1x3) = (x, y, z)
# v^-1(1x3) = (r, s, t)

# Q = |a b c|
#     |d e f|
#     |g h i|

# u^-1 * Q * u - 1 = 
# a*x^2 + b*xy + c*xz + d*xy + e*y^2 + f*yz + g*xz + h*yz + i*z^2 - 1 = 
# q[0]*x*x + q[1]*x*y + q[2]*x*z + q[3]*x*y + q[4]*y*y + q[5]*y*z + q[6]*x*z + q[7]*y*z + q[8]*z*z - 1

# v^-1 * Q * v - 1 = 
# a*r^2 + b*rs + c*rt + d*rs + e*s^2 + f*st + g*rt + h*st + i*t^2 - 1 = 
# q[0]*r*r + q[1]*r*s + q[2]*r*t + q[3]*r*s + q[4]*s*s + q[5]*s*t + q[6]*r*t + q[7]*s*t + q[8]*t*t - 1

# u^-1 * Q * v = 
# a*rx + b*sx + c*tx + d*ry + e*sy + f*ty + g*rz + h*sz + i*tz = 
# q[0]*r*x + q[1]*s*x + q[2]*t*x + q[3]*r*y + q[4]*s*y + q[5]*t*y + q[6]*r*z + q[7]*s*z + q[8]*t*z

def equations(q): #case_up_this (q)
    return [
        q[0]*x1*x1 + q[1]*x1*y1 + q[2]*x1*z1 + q[3]*x1*y1 + q[4]*y1*y1 + q[5]*y1*z1 + q[6]*x1*z1 + q[7]*y1*z1 + q[8]*z1*z1 - 1,
        q[0]*r1*r1 + q[1]*r1*s1 + q[2]*r1*t1 + q[3]*r1*s1 + q[4]*s1*s1 + q[5]*s1*t1 + q[6]*r1*t1 + q[7]*s1*t1 + q[8]*t1*t1 - 1,
        q[0]*r1*x1 + q[1]*s1*x1 + q[2]*t1*x1 + q[3]*r1*y1 + q[4]*s1*y1 + q[5]*t1*y1 + q[6]*r1*z1 + q[7]*s1*z1 + q[8]*t1*z1,
        q[0]*x2*x2 + q[1]*x2*y2 + q[2]*x2*z2 + q[3]*x2*y2 + q[4]*y2*y2 + q[5]*y2*z2 + q[6]*x2*z2 + q[7]*y2*z2 + q[8]*z2*z2 - 1,
        q[0]*r2*r2 + q[1]*r2*s2 + q[2]*r2*t2 + q[3]*r2*s2 + q[4]*s2*s2 + q[5]*s2*t2 + q[6]*r2*t2 + q[7]*s2*t2 + q[8]*t2*t2 - 1,
        q[0]*r2*x2 + q[1]*s2*x2 + q[2]*t2*x2 + q[3]*r2*y2 + q[4]*s2*y2 + q[5]*t2*y2 + q[6]*r2*z2 + q[7]*s2*z2 + q[8]*t2*z2,
        q[0]*x3*x3 + q[1]*x3*y3 + q[2]*x3*z3 + q[3]*x3*y3 + q[4]*y3*y3 + q[5]*y3*z3 + q[6]*x3*z3 + q[7]*y3*z3 + q[8]*z3*z3 - 1,
        q[0]*r3*r3 + q[1]*r3*s3 + q[2]*r3*t3 + q[3]*r3*s3 + q[4]*s3*s3 + q[5]*s3*t3 + q[6]*r3*t3 + q[7]*s3*t3 + q[8]*t3*t3 - 1,
        q[0]*r3*x3 + q[1]*s3*x3 + q[2]*t3*x3 + q[3]*r3*y3 + q[4]*s3*y3 + q[5]*t3*y3 + q[6]*r3*z3 + q[7]*s3*z3 + q[8]*t3*z3
    ]

resultFormat = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

solution, info  = leastsq(equations, resultFormat)

Q = solution.reshape(3, 3)
print('solution:\n')
print(Q[0])
print(Q[1])
print(Q[2])

if np.all(np.linalg.eigvals(Q) > 0):
    print("Todos os autovalores de R são positivos. R é definida positiva.")
else:
    print("R não é definida positiva. A fatoração de Cholesky não é aplicável.")

Q = np.linalg.cholesky(Q)

print("Matriz Q:")
print(Q)