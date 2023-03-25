from sklearn.datasets import load_breast_cancer
import numpy as np
import concrete.numpy as cnp
from sklearn.model_selection import train_test_split






class Fhe_krr_predictor:
    def __init__(self, X_train, alphas, sigma, kernel_type) -> None:
        self.X_train = X_train
        self.alphas = alphas
        self.sigma = sigma
        self.kernel_type = kernel_type

    
    def gaussian_predict(X_train,alphas,sigma, X_test):
        K = np.exp(-np.sum((X_test[:, None, :] - X_train[None, :, :]) ** 2, axis=-1) / sigma ** 2)
        return K @ alphas
    
    
    def laplacian_predict(X_train,alphas,sigma, X_test):
        K = np.exp(-np.sum(np.abs(X_test[:, None, :] - X_train[None, :, :]), axis=-1) / sigma)
        return K @ alphas
    
    def compile(self):
        if self.kernel_type == "gaussian":
            compiler = cnp.Compiler(self.gaussian_predict, {"X_train": "encrypted", "alphas": "encrypted", "sigma": "encrypted", "X_test": "encrypted"})
        elif self.kernel_type == "laplacian":
            compiler = cnp.Compiler(self.laplacian_predict, {"X_train": "encrypted", "alphas": "encrypted", "sigma": "encrypted", "X_test": "encrypted"})
        else:
            raise ValueError("kernel_type must be either gaussian or laplacian!")
        
        inputset = [self.X_train, self.alphas, self.sigma]
        print(f"Compiling...")
        self.circuit = compiler.compile(inputset)
        print(f"Compilation done!")

        print(f"Generating keys...")
        self.circuit.keygen()


    def predict(self, X_train, alphas, sigma, X_test, kernel_type):
        self.compile()
        if self.kernel_type == "gaussian":
            return self.gaussian_predict(X_train,alphas,sigma, X_test)
        elif self.kernel_type == "laplacian":
            return self.laplacian_predict(X_train,alphas,sigma, X_test)
        else:
            raise ValueError("kernel_type must be either gaussian or laplacian!")

    def execute(self):
        examples = [(3, 4), (1, 2), (7, 7), (0, 0)]
        for example in examples:
            encrypted_example = self.circuit.encrypt(*example)
            encrypted_result = self.circuit.run(encrypted_example)
            result = self.circuit.decrypt(encrypted_result)
            print(f"Evaluation of {' + '.join(map(str, example))} homomorphically = {result}")        



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

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "fhe_krr.py", line 39, in <module>
    circuit = compiler.compile(inputset)
  File "/home/jan/miniconda3/envs/fhe/lib/python3.8/site-packages/concrete/numpy/compilation/compiler.py", line 436, in compile
    self._evaluate("Compiling", inputset)
  File "/home/jan/miniconda3/envs/fhe/lib/python3.8/site-packages/concrete/numpy/compilation/compiler.py", line 277, in _evaluate
    self._trace(first_sample)
  File "/home/jan/miniconda3/envs/fhe/lib/python3.8/site-packages/concrete/numpy/compilation/compiler.py", line 206, in _trace
    self.graph = Tracer.trace(self.function, parameters)
  File "/home/jan/miniconda3/envs/fhe/lib/python3.8/site-packages/concrete/numpy/tracing/tracer.py", line 75, in trace
    output_tracers: Any = function(**arguments)
  File "fhe_krr.py", line 35, in gaussian_predict
    return np.dot(K,alphas)
  File "<__array_function__ internals>", line 180, in dot
  File "/home/jan/miniconda3/envs/fhe/lib/python3.8/site-packages/concrete/numpy/tracing/tracer.py", line 447, in __array_function__
    sanitized_args = [self.sanitize(arg) for arg in args]
  File "/home/jan/miniconda3/envs/fhe/lib/python3.8/site-packages/concrete/numpy/tracing/tracer.py", line 447, in <listcomp>
    sanitized_args = [self.sanitize(arg) for arg in args]
  File "/home/jan/miniconda3/envs/fhe/lib/python3.8/site-packages/concrete/numpy/tracing/tracer.py", line 187, in sanitize
    computation = Node.constant(value)
  File "/home/jan/miniconda3/envs/fhe/lib/python3.8/site-packages/concrete/numpy/representation/node.py", line 59, in constant
    raise ValueError(f"Constant {repr(constant)} is not supported") from error


(Pdb) np.dot(K,alphas)
*** ValueError: Constant array([[<concrete.numpy.tracing.tracer.Tracer object at 0x7f10f6600dc0>,
        <concrete.numpy.tracing.tracer.Tracer object at 0x7f10f6605df0>,
        <concrete.numpy.tracing.tracer.Tracer object at 0x7f10f658dd60>,
        <concrete.numpy.tracing.tracer.Tracer object at 0x7f10f6594760>,
        ......
        <concrete.numpy.tracing.tracer.Tracer object at 0x7f10f63ecbe0>]],
      dtype=object) is not supported
"""

    X_train, X_test, y_train, y_test = Data_preprocess(N_max = 5,rep_type="spahm", avg_hydrogens=False).run()
    alphas = np.ones_like(y_train).reshape(-1,1)

    sigma = 1
    def gaussian_predict(X_train,alphas,sigma, X_test):
        
        K =  np.asarray([np.exp(-  np.dot( X_train[i]-X_test[j],X_train[i]-X_test[j])   /sigma**2) for i in range(X_train.shape[0]) for j in range(X_test.shape[0])])
        K = K.reshape(X_train.shape[0],X_test.shape[0])
        pdb.set_trace()
        return np.dot(alphas, K)
    
    compiler = cnp.Compiler(gaussian_predict, {"X_train": "encrypted", "alphas": "encrypted", "sigma": "encrypted", "X_test": "encrypted"})
    inputset = [(X_train, alphas,sigma, X_test)]
    circuit = compiler.compile(inputset)
    circuit.keygen()
    encrypted_example = circuit.encrypt(*inputset)
    encrypted_result = circuit.run(encrypted_example)
    result = circuit.decrypt(encrypted_result)
