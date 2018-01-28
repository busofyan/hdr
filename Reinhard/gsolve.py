import numpy as N
import numpy.linalg as lin

def gsolve(Z,B,l,w):
    n = 256;
    A = N.zeros((N.multiply(Z.shape[0], Z.shape[1])+n+1, n+Z.shape[0]));
    b = N.zeros((A.shape[0], 1));

    # Include the data-fitting equations
    k = 1;
    for i in range(1, Z.shape[0]+1):
        for j in range(1, Z.shape[1]+1):
            wij = w[(int(Z[i-1, j-1]))];
            A[k-1, (Z[i-1, j-1])] = wij;
            A[k-1,(n + i - 1)] = - wij;
            b[k-1,1-1] = N.multiply(wij, B[i-1,j-1]);
            k = k + 1;

    # Fix the curve by setting its middle value to 0
    A[k-1, 129-1] = 1;

    k = k + 1;
    
    # Include the smoothness equations
    for i in range(0, n-3):
        A[k-1, i-1] = l * w[i-1 + 1-1];
        A[k-1, i + 1 -1] = -2 * l * w[i - 1 + 1 - 1];
        A[k-1, i + 2 -1]=l * w[i -1 + 1 -1];
        k = k + 1;
    
    # Solve the system using SVD
    x = lin.lstsq(A,b)[0]
    g = x[1:n];
    lE = x[n+1:x.shape[0]];

    return [g, lE];
