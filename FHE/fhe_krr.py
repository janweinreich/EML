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



sigma = 1
alphas = np.ones_like(y_train,dtype=np.int64).reshape(-1,1)
alphas = alphas.flatten()

def gaussian_matrix(X_train,sigma, X_test):
    """Predict using a Gaussian kernel.
    Args:
        X_train: Training data
        alphas: Weights
        sigma: Kernel width
        X_test: Test data
    Returns:
        Predictions


      currently np.dot is not supported between two encrypted tensors.
      Thus we create a double loop to compute the kernel matrix element by element.
      This is not efficient but should work for now.
    """
      
    K = []
    n_rows = X_train.shape[0]
    n_cols = X_test.shape[0]
    for i in range(n_rows):
        for j in range(n_cols):
            A = X_train[i]-X_test[j]
            B = X_train[i]-X_test[j]

            """
            Trick for computing the squared euclidean distance between two vectors
            without using np.dot
            """

            prod_A_B = np.sum(((A+B)**2 // 4 ).astype(np.int64) - ((A-B)**2//4).astype(np.int64))

            K.append(np.exp(- prod_A_B /sigma**2))
    
    K =  np.array(K)
    K =  K.reshape((X_train.shape[0],X_test.shape[0]))
    return K
    
def gaussian_predict(X_train,alphas,sigma, X_test):
    K = gaussian_matrix(X_train,sigma, X_test)
    #make matrix multiplication with double loop not use np dot or cnp dot

    n_rows = X_train.shape[0]
    n_cols = X_test.shape[0]   
    y_predicted = []
    
    for j in range(n_cols):
        summe = cnp.zero()
        for i in range(n_rows):

            A = K[i][j]
            B = alphas[i]
            summe += (A*B).astype(np.int64)
        
        
        y_predicted.append(summe)

    y_predicted = np.array(y_predicted).flatten()
    #flatten because we need to return a 1d array of encrypted scalars to use cnp.array
    #y_predicted = cnp.array(y_predicted)
    return y_predicted



def matmul_new(a, b):
    prod_a_b = ((a-b)**2//4).astype(np.int64)
    return prod_a_b


#def mul_tlu(A, B):
#    apbsq = ((A+B).astype(np.float64)**2 // 4).astype(np.int64)
##    ambsq = ((A-B).astype(np.float64)**2 // 4).astype(np.int64)
 #   return apbsq - ambsq


A = np.random.randint(0, 100, size=(10, 10))
B = np.random.randint(0, 100, size=(10, 10))


#https://community.zama.ai/t/valueerror-when-compiling/499/8
exit()
compiler = cnp.Compiler(gaussian_predict, {"X_train": "encrypted", "alphas": "encrypted", "sigma": "encrypted", "X_test": "encrypted"})
inputset = [(X_train, alphas,sigma, X_test)]
circuit = compiler.compile(inputset)
circuit.keygen()
encrypted_example = circuit.encrypt(*inputset)
encrypted_result = circuit.run(encrypted_example)
result = circuit.decrypt(encrypted_result)

print(result)
"""
%1915 = astype(%1914, dtype=int_)               # EncryptedTensor<uint1, shape=(30,)>
%1916 = subtract(%1906, %1909)                  # EncryptedTensor<float64, shape=(30,)>
......
%2274 = negative(%2273)                         # EncryptedScalar<int1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                        fhe_krr.py:53
%2275 = 2                                       # ClearScalar<uint2>
%2276 = power(%2, %2275)                        # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:53
%2277 = true_divide(%2274, %2276)               # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:53
%2278 = exp(%2277)                              # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:53
%2279 = %0[9]                                   # EncryptedTensor<float64, shape=(30,)>
%2280 = %3[1]                                   # EncryptedTensor<float64, shape=(30,)>
%2281 = subtract(%2279, %2280)                  # EncryptedTensor<float64, shape=(30,)>
%2282 = %0[9]                                   # EncryptedTensor<float64, shape=(30,)>
%2283 = %3[1]                                   # EncryptedTensor<float64, shape=(30,)>
%2284 = subtract(%2282, %2283)                  # EncryptedTensor<float64, shape=(30,)>
%2285 = add(%2281, %2284)                       # EncryptedTensor<float64, shape=(30,)>
%2286 = 2                                       # ClearScalar<uint2>
%2287 = power(%2285, %2286)                     # EncryptedTensor<float64, shape=(30,)>
%2288 = 4                                       # ClearScalar<uint3>
%2289 = floor_divide(%2287, %2288)              # EncryptedTensor<float64, shape=(30,)>
%2290 = astype(%2289, dtype=int_)               # EncryptedTensor<uint1, shape=(30,)>
%2291 = subtract(%2281, %2284)                  # EncryptedTensor<float64, shape=(30,)>
%2292 = 2                                       # ClearScalar<uint2>
%2293 = power(%2291, %2292)                     # EncryptedTensor<float64, shape=(30,)>
%2294 = 4                                       # ClearScalar<uint3>
%2295 = floor_divide(%2293, %2294)              # EncryptedTensor<float64, shape=(30,)>
%2296 = astype(%2295, dtype=int_)               # EncryptedTensor<uint1, shape=(30,)>
...
%2531 = astype(%2530, dtype=int_)               # EncryptedScalar<uint1>
%2532 = add(%2528, %2531)                       # EncryptedScalar<uint2>
%2533 = %1[7]                                   # EncryptedScalar<uint1>
%2534 = multiply(%1778, %2533)                  # EncryptedScalar<float64>
%2535 = astype(%2534, dtype=int_)               # EncryptedScalar<uint1>
%2536 = add(%2532, %2535)                       # EncryptedScalar<uint2>
%2537 = %1[8]                                   # EncryptedScalar<uint1>
%2538 = multiply(%2028, %2537)                  # EncryptedScalar<float64>
%2539 = astype(%2538, dtype=int_)               # EncryptedScalar<uint1>
%2540 = add(%2536, %2539)                       # EncryptedScalar<uint2>
%2541 = %1[9]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:72
%2542 = multiply(%2278, %2541)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:73
%2543 = astype(%2542, dtype=int_)               # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                         fhe_krr.py:73
%2544 = add(%2540, %2543)                       # EncryptedScalar<uint2>
%2545 = zeros()                                 # EncryptedScalar<uint1>
%2546 = %1[0]                                   # EncryptedScalar<uint1>
%2547 = multiply(%53, %2546)                    # EncryptedScalar<float64>
....
%2910 = %1[9]                                   # EncryptedScalar<uint1>
%2911 = multiply(%2503, %2910)                  # EncryptedScalar<float64>
%2912 = astype(%2911, dtype=int_)               # EncryptedScalar<uint1>
%2913 = add(%2909, %2912)                       # EncryptedScalar<uint2>
%2914 = array([%2544, %2 ... 72, %2913])        # EncryptedTensor<uint2, shape=(10,)>
return %2914
                                                                        
"""

