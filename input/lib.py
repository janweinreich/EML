from sklearn.model_selection import ParameterGrid, KFold
import numpy as np
from tqdm import tqdm
import numpy as np
import math
import gzip
from qml.kernels import gaussian_kernel_symmetric, gaussian_kernel
from qml.representations import generate_coulomb_matrix


element2charge    = {'H':1.0, 'C': 6.0, 'N':7.0, 'O':8.0, 'F':9.0,'Si':14.0,'P':15.0,'Pb':82.0,'S':16.0, 'Cl':17.0,'I':53.0,'B':35, 'Br':35.0,'Cs':55.0 } #!!!!! B not Br
element2effective = {'H': 1, 'C': 3.136, 'N': 3.834, 'O': 4.453 , 'Pb': 12.393, 'I': 11.612, 'Li':1.279}
charge2element    = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9:'F',11:'Na', 14: 'Si', 15:'P', 16: 'S',17: 'Cl',53:'I',35:'Br'}
bohr2angstrom = 0.52917720859
angstrom2bohr = 1.88972613386246
hartree2kcal = 627.5094706




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
            [element2charge[v] for v in curr]))

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



def loadXYZ(filename, ang2bohr=False):
    with open(filename, 'r') as f:
        lines = f.readlines()
        numAtoms = int(lines[0])
        positions = np.zeros((numAtoms, 3), dtype=np.double)
        elems = [None] * numAtoms
        comment = lines[1]
        for x in range (2, 2 + numAtoms):
            line_split = lines[x].rsplit()
            elems[x - 2] = line_split[0]
            
            line_split[1] = line_split[1].replace('*^', 'E')
            line_split[2] = line_split[2].replace('*^', 'E')
            line_split[3] = line_split[3].replace('*^', 'E')
            
            positions[x - 2][0] = np.double(line_split[1]) 
            positions[x - 2][1] = np.double(line_split[2]) 
            positions[x - 2][2] = np.double(line_split[3])
            if (ang2bohr):
                positions[x - 2][0] *= angstrom2bohr
                positions[x - 2][1] *= angstrom2bohr
                positions[x - 2][2] *= angstrom2bohr
                
    return np.asarray(elems), np.asarray(positions), comment


def writeXYZ(fileName, elems, positions, comment='', ang2bohr=False):
    with open (fileName, 'w') as f:
        f.write(str(len(elems)) + "\n")
        if (comment is not None):
            if ('\n' in comment):
                f.write(comment)
            else:
                f.write(comment + '\n')
        for x in range (0, len(elems)):
            if (ang2bohr):
                positions[x][0] *= angstrom2bohr
                positions[x][1] *= angstrom2bohr
                positions[x][2] *= angstrom2bohr
            f.write(elems[x] +" "+str(positions[x][0]) +  " " +    str(positions[x][1]) +" "+ str(positions[x][2]) + "\n")
    f.close()