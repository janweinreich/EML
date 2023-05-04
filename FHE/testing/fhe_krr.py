import numpy as np
import concrete.numpy as cnp
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pdb


X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Select a subset of the data to make things faster
X_train = X_train[:5]
y_train = y_train[:5]
X_test = X_test[:2]
y_test = y_test[:2]

sigma = 1
alphas = np.ones_like(y_train, dtype=np.int64).reshape(-1, 1)
alphas = alphas.flatten()


def mul_tlu(A, B):
    apbsq = ((A + B).astype(np.float64) ** 2 // 4).astype(np.int64)
    ambsq = ((A - B).astype(np.float64) ** 2 // 4).astype(np.int64)
    return apbsq - ambsq

def dot_product(A, B):
    len_A = A.shape[0] if hasattr(A, 'shape') else len(A)
    len_B = B.shape[0] if hasattr(B, 'shape') else len(B)

    if len_A != len_B:
        raise ValueError("The input vectors must have the same length.")

    result = 0
    for i in range(len_A):
        a = A[i]
        b = B[i]
        result += mul_tlu(a, b)

    return result


def gaussian_matrix(X_train, X_test, inv_sigma_sq):
    n_rows = X_train.shape[0]
    n_cols = X_test.shape[0]
    K = []
    

    for i in range(n_rows):
        row = []
        for j in range(n_cols):
            diff = X_train[i] - X_test[j]
            diff_sq = dot_product(diff, diff)
            div_sigma = mul_tlu(diff_sq, inv_sigma_sq)
            c = np.exp(-div_sigma).astype(np.int64)
            row.append(c)
        K.append(row)

    K = cnp.array(K)
    return K

def gaussian_predict(X_train,X_test,sigma, alphas):
    sigma = sigma.astype(np.int64)
    inv_sigma_sq = (1 / sigma) ** 2
    K = gaussian_matrix(X_train, X_test, inv_sigma_sq)
    
    alphas = alphas.astype(np.int64)
    
    y_predicted = dot_product(K, alphas)
    return y_predicted.astype(np.int64)



compiler = cnp.Compiler(gaussian_predict, {"X_train" : "encrypted", "X_test" : "encrypted", "sigma" : "encrypted", "alphas" : "encrypted"})
                    
inputset = [(X_train, X_test, sigma, alphas)]
circuit = compiler.compile(inputset )
circuit.keygen()
encrypted_example = circuit.encrypt(*inputset)
encrypted_result = circuit.run(encrypted_example)
result = circuit.decrypt(encrypted_result)