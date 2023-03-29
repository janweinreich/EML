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
            a = X_train[i]-X_test[j]
            b = X_train[i]-X_test[j]
            prod_a_b = np.sum(((a+b)**2 // 4 ).astype(np.int64) - ((a-b)**2//4).astype(np.int64))

            K.append(np.exp(- prod_a_b /sigma**2))
    
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
           
           summe += K[i][j] * alphas[i]
        
        y_predicted.append(summe)

    y_predicted = np.array(y_predicted).flatten()
    #flatten because we need to return a 1d array of encrypted scalars to use cnp.array
    y_predicted = cnp.array(y_predicted)

    return y_predicted

compiler = cnp.Compiler(gaussian_predict, {"X_train": "encrypted", "alphas": "encrypted", "sigma": "encrypted", "X_test": "encrypted"})
inputset = [(X_train, alphas,sigma, X_test)]
circuit = compiler.compile(inputset)
circuit.keygen()
encrypted_example = circuit.encrypt(*inputset)
encrypted_result = circuit.run(encrypted_example)
result = circuit.decrypt(encrypted_result)


"""
%2447 = subtract(%2440, %2446)                  # EncryptedTensor<uint1, shape=(30,)>
%2448 = sum(%2447)                              # EncryptedScalar<uint5>
%2449 = negative(%2448)                         # EncryptedScalar<int1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                        fhe_krr.py:48
%2450 = 2                                       # ClearScalar<uint2>
%2451 = power(%2, %2450)                        # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:48
%2452 = true_divide(%2449, %2451)               # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:48
%2453 = exp(%2452)                              # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:48
%2454 = %0[9]                                   # EncryptedTensor<float64, shape=(30,)>
%2455 = %3[8]                                   # EncryptedTensor<float64, shape=(30,)>
%2456 = subtract(%2454, %2455)                  # EncryptedTensor<float64, shape=(30,)>
%2457 = %0[9]                                   # EncryptedTensor<float64, shape=(30,)>
%2458 = %3[8]                                   # EncryptedTensor<float64, shape=(30,)>
%2459 = subtract(%2457, %2458)                  # EncryptedTensor<float64, shape=(30,)>
%2460 = add(%2456, %2459)                       # EncryptedTensor<float64, shape=(30,)>
%2461 = 2                                       # ClearScalar<uint2>
%2462 = power(%2460, %2461)                     # EncryptedTensor<float64, shape=(30,)>
%2463 = 4                                       # ClearScalar<uint3>
%2464 = floor_divide(%2462, %2463)              # EncryptedTensor<float64, shape=(30,)>
%2465 = astype(%2464, dtype=int_)               # EncryptedTensor<uint1, shape=(30,)>
%2466 = subtract(%2456, %2459)                  # EncryptedTensor<float64, shape=(30,)>
%2467 = 2                                       # ClearScalar<uint2>
%2468 = power(%2466, %2467)                     # EncryptedTensor<float64, shape=(30,)>
%2469 = 4                                       # ClearScalar<uint3>
%2470 = floor_divide(%2468, %2469)              # EncryptedTensor<float64, shape=(30,)>
%2471 = astype(%2470, dtype=int_)               # EncryptedTensor<uint1, shape=(30,)>
%2472 = subtract(%2465, %2471)                  # EncryptedTensor<uint1, shape=(30,)>
%2473 = sum(%2472)                              # EncryptedScalar<uint5>
%2474 = negative(%2473)                         # EncryptedScalar<int1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                        fhe_krr.py:48
%2475 = 2                                       # ClearScalar<uint2>
%2476 = power(%2, %2475)                        # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:48
%2477 = true_divide(%2474, %2476)               # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:48
%2478 = exp(%2477)                              # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:48
%2479 = %0[9]                                   # EncryptedTensor<float64, shape=(30,)>
%2480 = %3[9]                                   # EncryptedTensor<float64, shape=(30,)>
%2481 = subtract(%2479, %2480)                  # EncryptedTensor<float64, shape=(30,)>
%2482 = %0[9]                                   # EncryptedTensor<float64, shape=(30,)>
%2483 = %3[9]                                   # EncryptedTensor<float64, shape=(30,)>
%2484 = subtract(%2482, %2483)                  # EncryptedTensor<float64, shape=(30,)>
%2485 = add(%2481, %2484)                       # EncryptedTensor<float64, shape=(30,)>
%2486 = 2                                       # ClearScalar<uint2>
%2487 = power(%2485, %2486)                     # EncryptedTensor<float64, shape=(30,)>
%2488 = 4                                       # ClearScalar<uint3>
%2489 = floor_divide(%2487, %2488)              # EncryptedTensor<float64, shape=(30,)>
%2490 = astype(%2489, dtype=int_)               # EncryptedTensor<uint1, shape=(30,)>
%2491 = subtract(%2481, %2484)                  # EncryptedTensor<float64, shape=(30,)>
%2492 = 2                                       # ClearScalar<uint2>
%2493 = power(%2491, %2492)                     # EncryptedTensor<float64, shape=(30,)>
%2494 = 4                                       # ClearScalar<uint3>
%2495 = floor_divide(%2493, %2494)              # EncryptedTensor<float64, shape=(30,)>
%2496 = astype(%2495, dtype=int_)               # EncryptedTensor<uint1, shape=(30,)>
%2497 = subtract(%2490, %2496)                  # EncryptedTensor<uint1, shape=(30,)>
%2498 = sum(%2497)                              # EncryptedScalar<uint5>
%2499 = negative(%2498)                         # EncryptedScalar<int1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                        fhe_krr.py:48
%2500 = 2                                       # ClearScalar<uint2>
%2501 = power(%2, %2500)                        # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:48
%2502 = true_divide(%2499, %2501)               # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:48
%2503 = exp(%2502)                              # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:48
%2504 = zeros()                                 # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:63
%2505 = %1[0]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2506 = multiply(%28, %2505)                    # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2507 = add(%2504, %2506)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2508 = %1[1]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2509 = multiply(%278, %2508)                   # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2510 = add(%2507, %2509)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2511 = %1[2]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2512 = multiply(%528, %2511)                   # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2513 = add(%2510, %2512)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2514 = %1[3]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2515 = multiply(%778, %2514)                   # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2516 = add(%2513, %2515)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2517 = %1[4]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2518 = multiply(%1028, %2517)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2519 = add(%2516, %2518)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2520 = %1[5]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2521 = multiply(%1278, %2520)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2522 = add(%2519, %2521)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2523 = %1[6]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2524 = multiply(%1528, %2523)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2525 = add(%2522, %2524)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2526 = %1[7]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2527 = multiply(%1778, %2526)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2528 = add(%2525, %2527)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2529 = %1[8]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2530 = multiply(%2028, %2529)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2531 = add(%2528, %2530)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2532 = %1[9]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2533 = multiply(%2278, %2532)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2534 = add(%2531, %2533)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2535 = zeros()                                 # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:63
%2536 = %1[0]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2537 = multiply(%53, %2536)                    # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2538 = add(%2535, %2537)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2539 = %1[1]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2540 = multiply(%303, %2539)                   # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2541 = add(%2538, %2540)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2542 = %1[2]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2543 = multiply(%553, %2542)                   # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2544 = add(%2541, %2543)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2545 = %1[3]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2546 = multiply(%803, %2545)                   # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2547 = add(%2544, %2546)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2548 = %1[4]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2549 = multiply(%1053, %2548)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2550 = add(%2547, %2549)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2551 = %1[5]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2552 = multiply(%1303, %2551)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2553 = add(%2550, %2552)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2554 = %1[6]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2555 = multiply(%1553, %2554)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2556 = add(%2553, %2555)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2557 = %1[7]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2558 = multiply(%1803, %2557)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2559 = add(%2556, %2558)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2560 = %1[8]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2561 = multiply(%2053, %2560)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2562 = add(%2559, %2561)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2563 = %1[9]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2564 = multiply(%2303, %2563)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2565 = add(%2562, %2564)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2566 = zeros()                                 # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:63
%2567 = %1[0]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2568 = multiply(%78, %2567)                    # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2569 = add(%2566, %2568)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2570 = %1[1]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2571 = multiply(%328, %2570)                   # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2572 = add(%2569, %2571)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2573 = %1[2]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2574 = multiply(%578, %2573)                   # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2575 = add(%2572, %2574)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2576 = %1[3]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2577 = multiply(%828, %2576)                   # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2578 = add(%2575, %2577)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2579 = %1[4]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2580 = multiply(%1078, %2579)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2581 = add(%2578, %2580)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2582 = %1[5]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2583 = multiply(%1328, %2582)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2584 = add(%2581, %2583)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2585 = %1[6]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2586 = multiply(%1578, %2585)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2587 = add(%2584, %2586)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2588 = %1[7]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2589 = multiply(%1828, %2588)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2590 = add(%2587, %2589)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2591 = %1[8]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2592 = multiply(%2078, %2591)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2593 = add(%2590, %2592)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2594 = %1[9]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2595 = multiply(%2328, %2594)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2596 = add(%2593, %2595)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2597 = zeros()                                 # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:63
%2598 = %1[0]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2599 = multiply(%103, %2598)                   # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2600 = add(%2597, %2599)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2601 = %1[1]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2602 = multiply(%353, %2601)                   # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2603 = add(%2600, %2602)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2604 = %1[2]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2605 = multiply(%603, %2604)                   # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2606 = add(%2603, %2605)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2607 = %1[3]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2608 = multiply(%853, %2607)                   # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2609 = add(%2606, %2608)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2610 = %1[4]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2611 = multiply(%1103, %2610)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2612 = add(%2609, %2611)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2613 = %1[5]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2614 = multiply(%1353, %2613)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2615 = add(%2612, %2614)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2616 = %1[6]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2617 = multiply(%1603, %2616)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2618 = add(%2615, %2617)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2619 = %1[7]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2620 = multiply(%1853, %2619)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2621 = add(%2618, %2620)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2622 = %1[8]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2623 = multiply(%2103, %2622)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2624 = add(%2621, %2623)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2625 = %1[9]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2626 = multiply(%2353, %2625)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2627 = add(%2624, %2626)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2628 = zeros()                                 # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:63
%2629 = %1[0]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2630 = multiply(%128, %2629)                   # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2631 = add(%2628, %2630)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2632 = %1[1]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2633 = multiply(%378, %2632)                   # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2634 = add(%2631, %2633)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2635 = %1[2]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2636 = multiply(%628, %2635)                   # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2637 = add(%2634, %2636)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2638 = %1[3]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2639 = multiply(%878, %2638)                   # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2640 = add(%2637, %2639)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2641 = %1[4]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2642 = multiply(%1128, %2641)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2643 = add(%2640, %2642)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2644 = %1[5]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2645 = multiply(%1378, %2644)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2646 = add(%2643, %2645)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2647 = %1[6]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2648 = multiply(%1628, %2647)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2649 = add(%2646, %2648)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2650 = %1[7]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2651 = multiply(%1878, %2650)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2652 = add(%2649, %2651)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2653 = %1[8]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2654 = multiply(%2128, %2653)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2655 = add(%2652, %2654)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2656 = %1[9]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2657 = multiply(%2378, %2656)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2658 = add(%2655, %2657)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2659 = zeros()                                 # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:63
%2660 = %1[0]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2661 = multiply(%153, %2660)                   # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2662 = add(%2659, %2661)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2663 = %1[1]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2664 = multiply(%403, %2663)                   # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2665 = add(%2662, %2664)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2666 = %1[2]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2667 = multiply(%653, %2666)                   # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2668 = add(%2665, %2667)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2669 = %1[3]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2670 = multiply(%903, %2669)                   # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2671 = add(%2668, %2670)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2672 = %1[4]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2673 = multiply(%1153, %2672)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2674 = add(%2671, %2673)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2675 = %1[5]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2676 = multiply(%1403, %2675)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2677 = add(%2674, %2676)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2678 = %1[6]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2679 = multiply(%1653, %2678)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2680 = add(%2677, %2679)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2681 = %1[7]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2682 = multiply(%1903, %2681)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2683 = add(%2680, %2682)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2684 = %1[8]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2685 = multiply(%2153, %2684)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2686 = add(%2683, %2685)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2687 = %1[9]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2688 = multiply(%2403, %2687)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2689 = add(%2686, %2688)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2690 = zeros()                                 # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:63
%2691 = %1[0]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2692 = multiply(%178, %2691)                   # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2693 = add(%2690, %2692)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2694 = %1[1]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2695 = multiply(%428, %2694)                   # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2696 = add(%2693, %2695)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2697 = %1[2]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2698 = multiply(%678, %2697)                   # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2699 = add(%2696, %2698)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2700 = %1[3]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2701 = multiply(%928, %2700)                   # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2702 = add(%2699, %2701)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2703 = %1[4]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2704 = multiply(%1178, %2703)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2705 = add(%2702, %2704)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2706 = %1[5]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2707 = multiply(%1428, %2706)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2708 = add(%2705, %2707)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2709 = %1[6]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2710 = multiply(%1678, %2709)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2711 = add(%2708, %2710)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2712 = %1[7]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2713 = multiply(%1928, %2712)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2714 = add(%2711, %2713)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2715 = %1[8]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2716 = multiply(%2178, %2715)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2717 = add(%2714, %2716)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2718 = %1[9]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2719 = multiply(%2428, %2718)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2720 = add(%2717, %2719)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2721 = zeros()                                 # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:63
%2722 = %1[0]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2723 = multiply(%203, %2722)                   # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2724 = add(%2721, %2723)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2725 = %1[1]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2726 = multiply(%453, %2725)                   # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2727 = add(%2724, %2726)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2728 = %1[2]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2729 = multiply(%703, %2728)                   # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2730 = add(%2727, %2729)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2731 = %1[3]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2732 = multiply(%953, %2731)                   # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2733 = add(%2730, %2732)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2734 = %1[4]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2735 = multiply(%1203, %2734)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2736 = add(%2733, %2735)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2737 = %1[5]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2738 = multiply(%1453, %2737)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2739 = add(%2736, %2738)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2740 = %1[6]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2741 = multiply(%1703, %2740)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2742 = add(%2739, %2741)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2743 = %1[7]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2744 = multiply(%1953, %2743)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2745 = add(%2742, %2744)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2746 = %1[8]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2747 = multiply(%2203, %2746)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2748 = add(%2745, %2747)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2749 = %1[9]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2750 = multiply(%2453, %2749)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2751 = add(%2748, %2750)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2752 = zeros()                                 # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:63
%2753 = %1[0]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2754 = multiply(%228, %2753)                   # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2755 = add(%2752, %2754)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2756 = %1[1]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2757 = multiply(%478, %2756)                   # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2758 = add(%2755, %2757)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2759 = %1[2]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2760 = multiply(%728, %2759)                   # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2761 = add(%2758, %2760)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2762 = %1[3]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2763 = multiply(%978, %2762)                   # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2764 = add(%2761, %2763)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2765 = %1[4]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2766 = multiply(%1228, %2765)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2767 = add(%2764, %2766)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2768 = %1[5]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2769 = multiply(%1478, %2768)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2770 = add(%2767, %2769)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2771 = %1[6]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2772 = multiply(%1728, %2771)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2773 = add(%2770, %2772)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2774 = %1[7]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2775 = multiply(%1978, %2774)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2776 = add(%2773, %2775)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2777 = %1[8]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2778 = multiply(%2228, %2777)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2779 = add(%2776, %2778)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2780 = %1[9]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2781 = multiply(%2478, %2780)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2782 = add(%2779, %2781)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2783 = zeros()                                 # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:63
%2784 = %1[0]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2785 = multiply(%253, %2784)                   # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2786 = add(%2783, %2785)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2787 = %1[1]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2788 = multiply(%503, %2787)                   # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2789 = add(%2786, %2788)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2790 = %1[2]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2791 = multiply(%753, %2790)                   # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2792 = add(%2789, %2791)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2793 = %1[3]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2794 = multiply(%1003, %2793)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2795 = add(%2792, %2794)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2796 = %1[4]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2797 = multiply(%1253, %2796)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2798 = add(%2795, %2797)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2799 = %1[5]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2800 = multiply(%1503, %2799)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2801 = add(%2798, %2800)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2802 = %1[6]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2803 = multiply(%1753, %2802)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2804 = add(%2801, %2803)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2805 = %1[7]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2806 = multiply(%2003, %2805)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2807 = add(%2804, %2806)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2808 = %1[8]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2809 = multiply(%2253, %2808)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2810 = add(%2807, %2809)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2811 = %1[9]                                   # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this is one of the input nodes
                                                                         fhe_krr.py:66
%2812 = multiply(%2503, %2811)                  # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2813 = add(%2810, %2812)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                           fhe_krr.py:66
%2814 = array([%2534, %2 ... 82, %2813])        # EncryptedTensor<float64, shape=(10,)>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ within this subgraph
                                                                                        fhe_krr.py:71
return %2814
                                                                        
"""

