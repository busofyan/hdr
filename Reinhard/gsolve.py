import numpy as N
import math
import cv2

def gsolve(Z,B,l,w):
    n = 256;
    A = N.zeros(N.size(Z) + n+1, n + N.size(Z, 1));
    b = N.zeros(N.size(A, 1), 1);

    # Include the data-fitting equations
    k = 1;
    for i in range(1, N.size(Z, 1)):
        for j in range(1, N.size(Z,2)):
            wij = w[Z(i, j) + 1];
            A[k, Z(i, j) + 1] = wij;
            A[k,n + i] = -wij;
            b[k,1] = wij * B(i,j);
            k=k+1;

    # Fix the curve by setting its middle value to 0
    A[k, 129] = 1;

    k = k + 1;
    
    # Include the smoothness equations
    for i in range(1, n-2):
        A[k, i] = l * w[i + 1];
        A[k, i + 1] = -2 * l * w[i + 1];
        A[k, i + 2]=l * w[i + 1];
        k = k + 1;
    
    # Solve the system using SVD
    x = A/b;
    g = x[1:n];
    lE = x[n+1:N.size(x)];

    return [g, lE];
