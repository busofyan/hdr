import numpy as np
import numpy.linalg as lin


def gsolve(Z, B, l, w):
    n = 256;
    A = np.zeros((Z.shape[0] * Z.shape[1] + n + 1, n + Z.shape[0]));
    b = np.zeros((A.shape[0], 1));

    # Include the data-fitting equations
    k = 0;
    for i in range(0, Z.shape[0]):
        for j in range(0, Z.shape[1]):
            wij = w[(int(Z[i, j]))];
            A[k, (int(Z[i, j]))] = wij;
            A[k, (n + i)] = - wij;
            b[k] = np.multiply(wij, B[i, j]);
            k = k + 1;

    # Fix the curve by setting its middle value to 0
    A[k, 128] = 1;
    k = k + 1;

    # Include the smoothness equations
    for i in range(0, n - 2):
        A[k, i] = l * w[i + 1];
        A[k, i + 1] = -2 * l * w[i + 1];
        A[k, i + 2] = l * w[i + 1];
        k = k + 1;

    # Solve the system using SVD
    x = lin.lstsq(A, b)[0]
    g = x[0:n];
    lE = x[n:x.shape[0]];
    return [g, lE];
