import numpy as np
A = np.array([[2.0,1.0,1.0],[4.0,3.0,3.0],[8.0,7.0,9.0]])
A_shared = A.copy()
print("Original Matrix: A ")
print(A_shared)

multiplier_21 = A_shared[1, 0] / A_shared[0, 0]  
A_shared[1, 0] = multiplier_21              
A_shared[1, 1:] -= multiplier_21 * A_shared[0, 1:]

multiplier_31 = A_shared[2,0] / A_shared[0,0]
A_shared[2,0] = multiplier_31
A_shared[2,1:] -=  multiplier_31* A_shared[0,1:]
print("Matrix Memory after column one")
print(A_shared)

multiplier_32 = A_shared[2, 1] / A_shared[1, 1]  
A_shared[2, 1] = multiplier_32                  
A_shared[2, 2:] -= multiplier_32 * A_shared[1, 2:] 

print("Final Packed Matrix (A_shared):")
print(A_shared)

U = np.triu(A_shared)
L = np.tril(A_shared, -1 ) + np.eye(3)
print("Extracted lower triangle Matrix")
print(L)
print("Extracted Upper triangle Matrix")
print(U)
print("Does L @ U equal the original Matrix A?")
print(np.allclose(L@U ,A))

b = np.array([4.0,10.0,26.0])
y = np.zeros(3)
for i in range(3):
 sum_ly = np.dot(L[i,:i],y[:i])
 y[i] = b[i] - sum_ly
print("Intermediate Vector:")
print(y)

x = np.zeros(3)
for i in reversed(range(3)):
    sum_ux = np.dot(U[i,i+1:],x[i+1:])
    x[i] = (y[i] -sum_ux)/ U[i,i]
print("Final SOlution Vector x : ")
print(x)
print("Does A@x , b")
print(np.allclose(A@x,b))

