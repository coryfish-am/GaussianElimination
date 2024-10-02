import ctypes

gauss_library_path = './libgauss.so'
    
def unpack(A):
    #Extract L and U parts from A, fill with 0's and 1's
    n = len(A)
    
    # Create L and U matrices
    L = [[0 if j != i else 1 for j in range(n)] for i in range(n)]
    U = [[0 for j in range(n)] for i in range(n)]

    # Fill L with the lower triangular part of A (below diagonal)
    # Fill U with the upper triangular part of A (on and above diagonal)
    for i in range(n):
        for j in range(n):
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

def plu_python(A):
    # Perform PLU decomposition: PA = LU
    n = len(A)
    
    # Initialize permutation vector P and zero matrix L
    P = list(range(n))
    L = [[0.0] * n for _ in range(n)]
    
    # Copy of matrix A (which will become U)
    U = [row[:] for row in A]

    # LU Decomposition with partial pivoting
    for k in range(n - 1):
        # Pivot: Find the row with the largest value in column k
        pivot_row = k
        max_val = abs(U[k][k])
        
        # Searching for the pivot row
        for i in range(k + 1, n):
            if abs(U[i][k]) > max_val:
                max_val = abs(U[i][k])
                pivot_row = i

        # If pivot_row is different, swap the rows in U and P
        if pivot_row != k:
            U[k], U[pivot_row] = U[pivot_row], U[k]
            P[k], P[pivot_row] = P[pivot_row], P[k]

            # Also swap the rows in L for columns before the pivot
            for j in range(k):
                L[k][j], L[pivot_row][j] = L[pivot_row][j], L[k][j]

        # Perform elimination on rows below the pivot
        for i in range(k + 1, n):
            # Compute the multiplier and store it in L
            L[i][k] = U[i][k] / U[k][k]

            # Update the rest of the row in U
            for j in range(k, n):
                U[i][j] -= L[i][k] * U[k][j]

    # Set the diagonal elements of L to 1 (by convention)
    for i in range(n):
        L[i][i] = 1.0

    # Return the permutation vector P, the lower matrix L, and the upper matrix U
    return P, L, U



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
        A = [[2.0, 3.0, -1.0],
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