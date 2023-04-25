import numpy as np
import glob
from qstack import compound, spahm
import pdb

#list of xyz files
xyz_files = glob.glob("*.xyz")
X = []
for mol_name in xyz_files:
    mol = compound.xyz_to_mol(mol_name, 'def2svp', charge=0, spin=0)
    X.append(spahm.compute_spahm.get_spahm_representation(mol, "lb")[0])

X = np.array(X)
max_size = max([len(subarray) for subarray in X])
X = np.array([np.pad(subarray, (0, max_size - len(subarray)), 'constant') for subarray in X])

pdb.set_trace()