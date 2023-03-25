from sklearn.datasets import load_breast_cancer
import numpy as np
import concrete.numpy as cnp
from sklearn.model_selection import train_test_split

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

def gaussian_predict(X_train,alphas,sigma, X_test):
    """Predict using a Gaussian kernel.
    Args:
        X_train: Training data
        alphas: Weights
        sigma: Kernel width
        X_test: Test data
    Returns:
        Predictions
    """

    K =  np.asarray([np.exp(-  np.dot( X_train[i]-X_test[j],X_train[i]-X_test[j])   /sigma**2) for i in range(X_train.shape[0]) for j in range(X_test.shape[0])])
    K = K.reshape(X_train.shape[0],X_test.shape[0])    
    return np.dot(K,alphas)

compiler = cnp.Compiler(gaussian_predict, {"X_train": "encrypted", "alphas": "encrypted", "sigma": "encrypted", "X_test": "encrypted"})
inputset = [(X_train, alphas,sigma, X_test)]
circuit = compiler.compile(inputset)
circuit.keygen()
encrypted_example = circuit.encrypt(*inputset)
encrypted_result = circuit.run(encrypted_example)
result = circuit.decrypt(encrypted_result)


"""
np.dot produced error
(Pdb) np.dot(K,alphas)
*** ValueError: Constant array([[<concrete.numpy.tracing.tracer.Tracer object at 0x7f10f6600dc0>,
        <concrete.numpy.tracing.tracer.Tracer object at 0x7f10f6605df0>,
        <concrete.numpy.tracing.tracer.Tracer object at 0x7f10f658dd60>,
        <concrete.numpy.tracing.tracer.Tracer object at 0x7f10f6594760>,
        ......
        <concrete.numpy.tracing.tracer.Tracer object at 0x7f10f63ecbe0>]],
      dtype=object) is not supported
"""