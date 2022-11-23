from sklearn.model_selection import ParameterGrid, KFold
import numpy as np
from tqdm import tqdm
import constants
from scipy.spatial.distance import pdist, squareform
import numpy as np
from qml.kernels import gaussian_kernel_symmetric, gaussian_kernel
from qml.representations import generate_coulomb_matrix

#BOTLZMANN CONSTANT IN KCAL/MOL
k_B = 0.001985875





def get_single_CM(charges, coordinates):
    x = generate_coulomb_matrix(charges, coordinates, size=26)
    return x    





def getcolumn(filename, column, mode='float'):
    dat = np.loadtxt(filename, usecols=column, dtype=mode)
    return dat

def convert_charges(c):
    converted = []
    for curr in c:
        converted.append(np.asarray(
            [constants.element2charge[v] for v in curr]))

    return np.array(converted)



def train(X, energies, sigma, LLAMBDA):
    Ktrain = gaussian_kernel_symmetric(X, sigma)

    for i in range(Ktrain.shape[0]):
        Ktrain[i, i] -= LLAMBDA

    alpha = np.linalg.solve(Ktrain, energies)

    return alpha


def pred(representations, representations_test, alphas, kernel_sigma):
    

    K_geo = gaussian_kernel(
        representations_test, representations,  kernel_sigma)

    predvals = np.dot(K_geo, alphas)
    
    return predvals

def mae(prediction, reference):
    return np.mean(np.abs(prediction - reference))



#def CV(representations, Q, y, param_grid={'kernel_sigma': np.logspace(-0.5, 0.5, num=20), 'kernel_lambda':  np.logspace(-7, -2, num=10)}, kfn=5, refit=True):
def CV(representations,y, param_grid, kfn=5, refit=True):
    
    """
    Perform as Cross Validation to obtain the optimal weight coefficients for
    each trainingset sizes. A grid search will be performed.

    input:
    representation, distances for training
    array of hyperparameters

    returns:
    weight coefficients and optimized hyperparameters
    """
    
    param_grid = list(ParameterGrid(param_grid))

    kf = KFold(n_splits=kfn, shuffle=True, random_state=42)

    all_gpr = []
    for gp in tqdm(param_grid):
        gp_errs = []
        for train_index, test_index in kf.split(representations):
            alphas = train(representations[train_index], y[train_index],gp['kernel_sigma'],gp['kernel_lambda'])

            predictions = pred(representations[train_index], representations[test_index],
                                  alphas, kernel_sigma=gp['kernel_sigma'])

            gp_errs.append(
                mae(predictions, y[test_index]))

        all_gpr.append(np.mean(np.array(gp_errs)))

    all_gpr = np.array(all_gpr)
    opt_p = param_grid[np.argmin(all_gpr)]
    print(opt_p)

    if refit:
        alphas_opt = train(representations,y,opt_p['kernel_sigma'], opt_p['kernel_lambda'])
        return alphas_opt, opt_p
