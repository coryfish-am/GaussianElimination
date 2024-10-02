#----------------------------------------------------------------
# File:     gauss_solve.py
#----------------------------------------------------------------
#
# Author:   Marek Rychlik (rychlik@arizona.edu)
# Date:     Thu Sep 26 10:38:32 2024
# Copying:  (C) Marek Rychlik, 2020. All rights reserved.
# 
#----------------------------------------------------------------
# A Python wrapper module around the C library libgauss.so

import ctypes
import numpy as np

gauss_library_path = './libgauss.so'

def P(A):
    n = len(A)  # Assuming `a` is a square matrix
    identity = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    return identity
    
def unpack(A):
    """ Extract L and U parts from A, fill with 0's and 1's """
    n = len(A)
    
    # Create L and U matrices
    L = [[0 if j != i else 1 for j in range(n)] for i in range(n)]
    U = [[0 for j in range(n)] for i in range(n)]

    # Fill L with the lower triangular part of A (below diagonal)
    # Fill U with the upper triangular part of A (on and above diagonal)
    for i in range(n):
        for j in range(n):
            print(f"Processing: A[{i}][{j}] = {A[i][j]}")
            if i > j:
                L[i][j] = A[i][j]  # Values below diagonal in A are for L
            else:
                U[i][j] = A[i][j]  # Values on and above diagonal in A are for U

    return L, U


def lu_c(A):
    """ Accepts a list of lists A of floats and
    it returns (L, U) - the LU-decomposition as a tuple.
    """
    # Load the shared library
    lib = ctypes.CDLL(gauss_library_path)

    # Create a 2D array in Python and flatten it
    n = len(A)
    flat_array_2d = [item for row in A for item in row]

    # Convert to a ctypes array
    c_array_2d = (ctypes.c_double * len(flat_array_2d))(*flat_array_2d)

    # Define the function signature
    lib.lu_in_place.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double))

    # Modify the array in C (e.g., add 10 to each element)
    lib.lu_in_place(n, c_array_2d)

    # Convert back to a 2D Python list of lists
    modified_array_2d = [
        [c_array_2d[i * n + j] for j in range(n)]
        for i in range(n)
    ]

    # Extract L and U parts from A, fill with 0's and 1's
    return unpack(modified_array_2d)

def plu_c(A):
    """ Accepts a list of lists A of floats and
    it returns (L, U) - the LU-decomposition as a tuple.
    """
    # Load the shared library
    lib = ctypes.CDLL(gauss_library_path)

    # Create a 2D array in Python and flatten it
    n = len(A)
    flat_array_2d = [item for row in A for item in row]

    # Create the identity permutation array P as a 1D array
    P_array = [i for i in range(n)]

    # Convert to ctypes array
    c_array_2d = (ctypes.c_double * len(flat_array_2d))(*flat_array_2d)
    c_P_array = (ctypes.c_int * n)(*P_array)

    # Define the function signature (accepting n, A, and P)
    lib.plu.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int))

    # Call the C function (pass n, A, and P)
    lib.plu(n, c_array_2d, c_P_array)

    # Convert back to a 2D Python list of lists
    modified_array_2d = [
        [c_array_2d[i * n + j] for j in range(n)]
        for i in range(n)
    ]
    L,U = unpack(modified_array_2d)
    # Convert the 1D permutation array back to a permutation matrix
    permutation_matrix = [[1 if c_P_array[i] == j else 0 for j in range(n)] for i in range(n)]
    permutation_vector = [list(row).index(1) for row in permutation_matrix]
    # Extract L and U parts from A, fill with 0's and 1's
    return permutation_vector, L, U

def lu_python(A):
    n = len(A)
    for k in range(n):
        for i in range(k,n):
            for j in range(k):
                A[k][i] -= A[k][j] * A[j][i]
        for i in range(k+1, n):
            for j in range(k):
                A[i][k] -= A[i][j] * A[j][k]
            A[i][k] /= A[k][k]

    return unpack(A)

def swap_rows(matrix, row1, row2):
    """ Swap two rows in a NumPy array """
    temp = np.copy(matrix[row1, :])
    matrix[row1, :] = matrix[row2, :]
    matrix[row2, :] = temp
    
def plu_python(A):
    n = len(A)
    
    # Initialize P as an identity matrix
    P = np.eye(n)
    
    # Initialize U to a copy of A
    U = np.copy(A)
    
    # LU Decomposition with partial pivoting
    for k in range(n - 1):
        # Step 5: Find the pivot element (largest absolute value in column k)
        pivot_row = k
        max_val = abs(U[k][k])
        for i in range(k + 1, n):
            if abs(U[i][k]) > max_val:
                max_val = abs(U[i][k])
                pivot_row = i
        
        # Step 6: Swap rows in U and P if needed (use manual swap function)
        if pivot_row != k:
            swap_rows(U, k, pivot_row)
            swap_rows(P, k, pivot_row)
        
        # Step 9: Perform elimination for rows below the pivot row
        for i in range(k + 1, n):
            # Step 10: Compute the multiplier
            U[i][k] = U[i][k] / U[k][k]
            
            # Step 11-13: Update the matrix U
            U[i, k+1:n] = U[i, k+1:n] - U[i][k] * U[k, k+1:n]
    
    # Use the provided unpack function to extract L and U
    L, U = unpack(U.tolist())
    
    # Return the permutation matrix P (as a 2D matrix), and matrices L and U
    return P.tolist(), L, U


def lu(A, use_c=False):
    if use_c:
        return lu_c(A)
    else:
        return lu_python(A)

def plu(A, use_c=False):
    if use_c:
        return plu_c(A)
    else:
        return plu_python(A)



if __name__ == "__main__":

    def get_A():
        """ Make a test matrix """
        B = [[2.0, 3.0, -1.0],
             [4.0, 1.0, 2.0],
             [-2.0, 7.0, 2.0]]
        A = [[-2.0, 3.0, 5.0],
             [1.8, 10.0, 2.0],
             [-2.0, 1.5, 3.0]]
        D = [[2.0, 3.0, -1.0],
             [4.0, 1.0, 2.0],
             [-2.0, 7.0, 2.0]]
        C = [[2.0, 3.0, -1.0],
             [4.0, 1.0, 2.0],
             [-2.0, 7.0, 2.0]]
        return A

    A = get_A()

    L, U = lu(A, use_c = False)
    print(L)
    print(U)

    # Must re-initialize A as it was destroyed
    A = get_A()

    L, U = lu(A, use_c=True)
    print(L)
    print(U)
    
    A = get_A()
    P, L, U = plu(A, use_c=False)
    print(P)
    print(L)
    print(U)
    
    A = get_A()
    P, L, U = plu(A, use_c=True)
    print(P)
    print(L)
    print(U)