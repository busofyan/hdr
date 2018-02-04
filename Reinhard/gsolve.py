import numpy as N
import numpy.linalg as lin

def gsolve(Z,B,l,w):
    n = 256;
    A = N.zeros((Z.shape[0]*Z.shape[1]+n+1,n+Z.shape[0]));
    b = N.zeros((A.shape[0], 1));

    # Include the data-fitting equations
    k = 0;
    for i in range(0, Z.shape[0]):
        for j in range(0, Z.shape[1]):
            wij = w[(int(Z[i, j]))];
            A[k, (Z[i, j])] = wij;
            A[k,(n + i)] = - wij;
            b[k] = N.multiply(wij, B[i,j]);
            k = k + 1;

    # Fix the curve by setting its middle value to 0
    A[k, 128] = 1;

    k = k + 1;

    # Include the smoothness equations
    for i in range(0, n-3):
        A[k, i] = l * w[i + 1];
        A[k, i + 1] = -2 * l * w[i + 1];
        A[k, i + 2]=l * w[i + 1];
        k = k + 1;
    
    # Solve the system using SVD
    x = lin.lstsq(A,b)[0]
    g = x[1:n];
    lE = x[n+1:x.shape[0]];
    return [g, lE];
