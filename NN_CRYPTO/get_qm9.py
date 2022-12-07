import numpy as np
import qml
from tqdm import tqdm
import numpy as np
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle as pickle
import random
from qml.utils import NUCLEAR_CHARGE

random.seed(1337)
np.random.seed(1337)

def get_atomsizes(compounds):
    charge_to_element = {
        1: 'H',
        6: 'C',
        7: 'N',
        8: 'O',
        9: 'F',
        16: 'S',
        17: 'Cl',
        53: 'I',
        35: 'Br',
        15: 'P'}
    asize_dict = {'H': 0, 'C': 0, 'N': 0, 'O': 0, 'F': 0, 'Cl': 0, 'S':0, 'I':0, 'Br':0, 'P':0}
    for compound in compounds:
        nc, counts = np.unique(compound.nuclear_charges, return_counts=True)
        for z, count in zip(nc, counts):
            if asize_dict[charge_to_element[z]] < count:
                asize_dict[charge_to_element[z]] = count
    return asize_dict


def atomization_en(EN, ATOMS):
  import collections, numpy
  en_H = -0.500273 
  en_C = -37.846772
  en_N = -54.583861
  en_O = -75.064579 
  en_F = -99.718730
  COMP =  collections.Counter(ATOMS)
  
  ATOMIZATION = (EN - (COMP['H']*en_H + COMP['C']*en_C + COMP['N']*en_N +  COMP['O']*en_O +  COMP['F']*en_F))

  return ATOMIZATION
  
def get_atomsizes(compounds):
    charge_to_element = {
        1: 'H',
        6: 'C',
        7: 'N',
        8: 'O',
        9: 'F',
        16: 'S',
        17: 'Cl',
        53: 'I',
        35: 'Br',
        15: 'P'}
    asize_dict = {'H': 0, 'C': 0, 'N': 0, 'O': 0, 'F': 0, 'Cl': 0, 'S':0, 'I':0, 'Br':0, 'P':0}
    for compound in compounds:
        nc, counts = np.unique(compound.nuclear_charges, return_counts=True)
        for z, count in zip(nc, counts):
            if asize_dict[charge_to_element[z]] < count:
                asize_dict[charge_to_element[z]] = count
    return asize_dict


def get_bob(mol_data):
    all_compounds = []
    for mol in mol_data:
        cmp = qml.Compound()
        cmp.coordinates = mol[1]
        cmp.atomtypes = mol[0]
        cmp.nuclear_charges = np.array([NUCLEAR_CHARGE[ele] for ele in cmp.atomtypes]).flatten()     
        all_compounds.append(cmp)


    max_atoms = max([z.nuclear_charges.shape[0] for z in all_compounds])
    asize = get_atomsizes(all_compounds)

    for compound in tqdm(all_compounds, total=len(all_compounds)):
        compound.generate_bob(size=max_atoms, asize=asize)

    replist = np.array([compound.representation for compound in all_compounds])

    return replist

def read_xyz(path):
    """
    Reads the xyz files in the directory on 'path'
    Input
    path: the path to the folder to be read
    
    Output
    atoms: list with the characters representing the atoms of a molecule
    coordinates: list with the cartesian coordinates of each atom
    smile: list with the SMILE representation of a molecule
    prop: list with the scalar properties
    """
    atoms = []
    coordinates = []

    with open(path, 'r') as file:
        lines = file.readlines()
        n_atoms = int(lines[0])  # the number of atoms
        smile = lines[n_atoms + 3].split()[0]  # smiles string
        prop = lines[1].split()[2:]  # scalar properties
        
        # to retrieve each atmos and its cartesian coordenates
        for atom in lines[2:n_atoms + 2]:
            line = atom.split()
            # which atom
            atoms.append(line[0])

            # its coordinate
            # Some properties have '*^' indicading exponentiation 
            try:
                coordinates.append(
                    (float(line[1]),
                     float(line[2]),
                     float(line[3]))
                    )
            except:
                coordinates.append(
                    (float(line[1].replace('*^', 'e')),
                     float(line[2].replace('*^', 'e')),
                     float(line[3].replace('*^', 'e')))
                    )
                    
    return np.array(atoms), np.array(coordinates), smile, prop

def get_qm9(N):

    directory = '.'
    len(os.listdir(directory))

    file = os.listdir(directory)[0]
    with open(directory+file, 'r') as f:

        content = f.readlines()

    all_mol_paths = os.listdir(directory)
    np.random.shuffle(all_mol_paths)
    all_mol_paths = all_mol_paths[:N]

    data = []
    smiles = []
    properties = []
    for file in tqdm(all_mol_paths):
        path = os.path.join(directory, file)
        atoms, coordinates, smile, prop = read_xyz(path)
        data.append((atoms, coordinates))
        smiles.append(smile) 

        ATOMIZATION = atomization_en(float(prop[10]),atoms )
        prop += [ATOMIZATION]
        properties.append(prop) # The molecules properties


    


    properties_names = ['A', 'B', 'C', 'mu', 'alfa', 'homo', 'lumo', 'gap', 'R²', 'zpve', 'U0', 'U', 'H', 'G', 'Cv', 'atomization']
    df = pd.DataFrame(properties, columns = properties_names)
    df['smiles'] = smiles
    print(df.head())
    y  = df['atomization'].values


    data = np.array(data)
    X= get_bob(data)

    return X, y
