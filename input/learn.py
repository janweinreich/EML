import numpy as np
from qml.kernels import gaussian_kernel, gaussian_kernel_symmetric
from qml.math import cho_solve
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import lib_CM as fml
import numpy as np
random.seed(1337)
np.random.seed(1337)

new_data, new_hyper = True, False

if __name__ == "__main__":

    
    path_to_file = "/store/jan/datasets/qm9_data.npz"
    qm9 = np.load(path_to_file, allow_pickle=True)
    data = qm9
    Nmax = 30000  # len(data['coordinates'])
    coords = data['coordinates'][:Nmax]
    nuclear_charges = data['charges'][:Nmax]
    elements = data['elements'][:Nmax]
    energies = np.array(data['H_atomization'])[:Nmax]
    Cvs = np.array(data['Cv'])[:Nmax]

    #{'kernel_lambda': 1e-08, 'kernel_sigma': 73.86199822079365}
    #16384 0.01753798614051945 {'kernel_lambda': 1e-08, 'kernel_sigma': 73.86199822079365}
    sigma = 74 #73.86199822079365

    new_rep     = True
    new_hyper   = False

    if new_rep:
        X = []
        for q, r in tqdm(zip(nuclear_charges, coords), total=len(energies)):
            
            X.append(fml.get_single_CM(q, r))
            
        X, Q = np.array(X), nuclear_charges
        print(X.shape)
        exit()
        y = energies


        X_train, X_test, Qtrain, Qtest, y_train, y_test, ids_train, ids_test = train_test_split(X, Q, y, np.arange(len(X)), shuffle=True, test_size=20, random_state=1337)

    else:
        data = np.load("REP.npz", allow_pickle=True)
        X, X_train, y_train, X_test, y_test = data["X"], data["X_train"], data["y_train"], data["X_test"], data["y_test"]

    if new_hyper:
        #[32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
        # [2**i for i in range(5, 15)]

        param_grid = {'kernel_sigma': np.logspace(1, 2.1, 20),
                      'kernel_lambda': [1e-8]}

        # np.logspace(1.6, 2.5, num=2)
        lrn_crvs = []

        #train, test = indexes[:Nmax], indexes[Nmax:]
        #X_train, Q_train, X_test, Q_test, y_train, y_test = X[train], Q[train], X[test], Q[test], y[train], y[test]

        lrn_crv = []
        for n in [16384]:  

            X_curr,  y_curr = X_train[:n], y_train[:n]
            alphas, opt_p = fml.CV(X_curr, y_curr, param_grid)

            predictions   = fml.pred(X_curr, X_test,alphas, opt_p['kernel_sigma'])
            MAE = fml.mae(predictions, y_test)
            print(n, MAE, opt_p)
            lrn_crv.append(MAE)

        lrn_crvs.append(lrn_crv)
        exit()

    if new_data:
        
        np.savez_compressed("REP", X=X, X_train=X_train, y_train=y_train,
                            X_test=X_test, y_test=y_test, ids_train=ids_train, ids_test=ids_test)

        for ind, xt in enumerate(X_test):
            query = open("./data/test/X_QUERY_{}".format(ind), "w")
            np.savetxt(query, xt.flatten(), fmt='%1.20f', newline=" ")
            query.close()
    
    
    lrncrv = []
    ntp =[32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]  
    #np.array([2**i for i in range(0, 10, 1)])
    alphas =[]
    predictions = []

    for n in ntp:
    
        K = gaussian_kernel_symmetric(X_train[:n], sigma)
        K[np.diag_indices_from(K)] += 1e-8
        alpha = cho_solve(K, y_train[:n])
    
        if new_data:
            mod = open("./data/train/MODEL_{}".format(n), "a")
            for xtr, a in zip(X_train[:n], alpha):
                xtr = np.append(xtr, a)
                np.savetxt(mod, xtr.flatten(), fmt='%1.20f', newline=" ")
                mod.write("\n")

            mod.close()


        Ks = gaussian_kernel(X_test, X_train[:n], sigma)


        y_predicted  = np.dot(Ks, alpha)
        mae = np.mean(np.abs(y_predicted - y_test))
        print(n, mae)
        lrncrv.append(mae)
        alphas.append(alpha)
        predictions.append(y_predicted)


        if new_data:
            for ind, p in enumerate(y_predicted):
        
                predicted = open(
                    "./data/test/PRED_{}_{}".format(n, ind), "w")
                predicted.write("{} \n".format(p))
                predicted.close()

    lrncrv, alphas, predictions   = np.array(lrncrv), np.array(alphas), np.array(predictions)
    
    if new_data:
        np.savez("RESULTS", ntp=ntp, lrncrv=lrncrv, predictions=predictions, alphas=alphas)

    plt.plot(ntp, lrncrv, "+")
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig("learning_curve.png")
    plt.show()
