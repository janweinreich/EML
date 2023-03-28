from sklearn.datasets import load_breast_cancer
import numpy as np
import concrete.numpy as cnp
from sklearn.model_selection import train_test_split
import pdb


X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Select a subset of the data to make things faster
X_train = X_train[:10]
y_train = y_train[:10]
X_test = X_test[:10]
y_test = y_test[:10]


# Instead of doing plaintext KRR we just create a dummy model
sigma = 1
alphas = np.ones_like(y_train).reshape(-1,1)

def gaussian_matrix(X_train,alphas,sigma, X_test):
    """Predict using a Gaussian kernel.
    Args:
        X_train: Training data
        alphas: Weights
        sigma: Kernel width
        X_test: Test data
    Returns:
        Predictions
    """
    with cnp.tag("gaussian_matrix"):
      K =  cnp.array([np.exp(-  np.dot( X_train[i]-X_test[j],X_train[i]-X_test[j])   /sigma**2) for i in range(X_train.shape[0]) for j in range(X_test.shape[0])])
      K =  K.reshape((X_train.shape[0],X_test.shape[0]))
    return K
    
def gaussian_predict(X_train,alphas,sigma, X_test):
    with cnp.tag("cnp_matmul_1"):
       K = gaussian_matrix(X_train,alphas,sigma, X_test)
    #make matrix multiplication with double loop not use np dot or cnp dot
    
    with cnp.tag("cnp_matmul_2"):
      n_rows = X_train.shape[0]
      n_cols = X_test.shape[0]

      # Initialize the result vector
      result = 0

      for i in range(n_rows):
        for j in range(n_cols):
            result += K[i][j] * alphas[i]

    return result

compiler = cnp.Compiler(gaussian_predict, {"X_train": "encrypted", "alphas": "encrypted", "sigma": "encrypted", "X_test": "encrypted"})


inputset = [(X_train, alphas,sigma, X_test)]
circuit = compiler.compile(inputset)
circuit.keygen()
encrypted_example = circuit.encrypt(*inputset)
encrypted_result = circuit.run(encrypted_example)
result = circuit.decrypt(encrypted_result)


"""
%1512 = %1205[6]                                 # EncryptedTensor<float64, shape=(10,)>           ∈ [0.0, 0.0]                                            @ cnp_matmul_2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ only integer operations are supported
                                                                                                                                                                          fhe_krr.py:53
%1513 = %1512[1]                                 # EncryptedScalar<float64>                        ∈ [0.0, 0.0]                                            @ cnp_matmul_2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""

