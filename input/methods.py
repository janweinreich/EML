'''
Created on Nov 4, 2015

@author: roethlisbergergroup
'''

import math
import numpy as np
import gzip
import constants as cnst

#===============================================================================
# 5
# 000001
# C    -0.0126981359     1.0858041578     0.0080009958    -0.535689
# H     0.002150416    -0.0060313176     0.0019761204     0.133921
# H     1.0117308433     1.4637511618     0.0002765748     0.133922
# H    -0.540815069     1.4475266138    -0.8766437152     0.133923
# H    -0.5238136345     1.4379326443     0.9063972942     0.133923
#===============================================================================


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
                positions[x - 2][0] *= cnst.angstrom2bohr
                positions[x - 2][1] *= cnst.angstrom2bohr
                positions[x - 2][2] *= cnst.angstrom2bohr
                
    return np.asarray(elems), np.asarray(positions), comment


def _loadXYZ(inputLines, ang2bohr=False):
    numAtoms = int(inputLines[0])
    positions = np.zeros((numAtoms, 3), dtype=np.double)
    elems = [None] * numAtoms
    comment = inputLines[1]
    for x in range (2, 2 + numAtoms):
        line_split = inputLines[x].rsplit()
        elems[x - 2] = line_split[0]
        positions[x - 2][0] = np.double(line_split[1]) 
        positions[x - 2][1] = np.double(line_split[2])
        positions[x - 2][2] = np.double(line_split[3])

        if (ang2bohr):
            positions[x - 2][0] *= cnst.angstrom2bohr
            positions[x - 2][1] *= cnst.angstrom2bohr
            positions[x - 2][2] *= cnst.angstrom2bohr
            
    return elems, positions, comment



def loadMNsol(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    #print(lines)
    numAtoms  = len(lines)-3
    
    positions = np.zeros((numAtoms, 3), dtype=np.double)
    #print(positions)
    elems     = [None] * numAtoms
    #print(elems)
    
    for i in range(3, len(lines)):
        line_split = lines[i].rsplit()
        #print(line_split)
        elems[i - 3] = int(line_split[0])
        positions[i - 3][0] = np.double(line_split[1]) 
        positions[i - 3][1] = np.double(line_split[2])
        positions[i - 3][2] = np.double(line_split[3])
        
    
    return elems, positions



def loadTrajectory(filename, ang2bohr=False, space_sep=True):
    all_elems = []
    all_positions = []
    all_comments = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        
        count = 0
        # print numAtoms
        # print len(lines)
        while count < len(lines):
            numAtoms = int(lines[count])
            subLines = lines[count: count + numAtoms + 2]
            elems, positions, comment = _loadXYZ(subLines, ang2bohr)
            all_elems.append(elems)
            all_positions.append(positions)
            all_comments.append(comment)
            if (space_sep):
                count += (numAtoms + 3)
            else:
                count += (numAtoms + 2)
                
    return np.asarray(all_elems), np.asarray(all_positions), all_comments


def loadTrajectoryGZ(filename, ang2bohr=False, space_sep=True):
    all_elems = []
    all_positions = []
    all_comments = []
    with gzip.open(filename, 'rt') as f:
        lines = f.readlines()

        count = 0
        # print numAtoms
        # print len(lines)
        while count < len(lines):
            numAtoms = int(lines[count])
            subLines = lines[count: count + numAtoms + 2]
            elems, positions, comment = _loadXYZ(subLines, ang2bohr)
            all_elems.append(elems)
            all_positions.append(positions)
            all_comments.append(comment)
            if (space_sep):
                count += (numAtoms + 3)
            else:
                count += (numAtoms + 2)

    return np.asarray(all_elems), np.asarray(all_positions), all_comments

def writeTrajectory(filename, all_elems, all_positions, all_comments=None, space_sep=False):

    with open(filename, 'w') as f:
    
        for i in range (0, len(all_elems)):
            f.write(str(len(all_elems[i])) + '\n')
            if (all_comments is not None):
                f.write(all_comments[i])
            else:
                f.write("None\n")
            for j in range (0, len(all_elems[i])):
                f.write('{:5s} {:20.10f} {:20.10f} {:20.10f}'.format(all_elems[i][j], all_positions[i][j][0], all_positions[i][j][1], all_positions[i][j][2]) + "\n")
            if (i != (len(all_elems) - 1) and space_sep == True):
                f.write('\n')


def loadTurbomolEnergy(filename):
      
    all_energies = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        
        for i in xrange(1, len(lines) - 1):
           all_energies.append(float(lines[i].split()[1]))
     
    return np.asarray(all_energies)

 
def loadTurbomolForces(filename, numatoms):
    
    all_forces = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        
        for i in xrange(2, len(lines), 2 * numatoms + 1):
            
            forces = []
            
            for j in range(0, numatoms):
            
                force_j = lines[i + numatoms + j]
                
                force_j = force_j.replace('D', 'E')
                
                forces.append(np.fromstring(force_j, sep=' '))
                
                # print force_j , 
                
            all_forces.append(forces)
    return np.asarray(all_forces)

                
def loadForces(filename, numatoms, cpmd_traj=False):
    all_forces = []
    all_steps = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        i = 1
        while (i < len(lines)):
            force = np.zeros((numatoms, 3), dtype=np.float32)
            step = []
            
            for j in xrange (0, numatoms):
                idx = i + j
                data = lines[idx].split()
                step.append(int(data[0]))
                if (cpmd_traj):
                    force[j] = np.asarray([float(v) for v in data[4:7]])
                else:
                    force[j] = np.asarray([float(v) for v in data[1:4]])
                
            i += numatoms
            all_forces.append(force)
            all_steps.append(step)
    return np.asarray(all_forces), np.asarray(all_steps)


def ElisaIsBeautiful(filename, numatoms):
    all_base_forces = []
    all_correction_forces = []
    all_coords = []
    
    all_steps = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        i = 0
        while (i < len(lines)):
            coords = np.zeros((numatoms, 3), dtype=np.float32)
            force_base = np.zeros((numatoms, 3), dtype=np.float32)
            force_correction = np.zeros((numatoms, 3), dtype=np.float32)
            step = []
            
            for j in xrange (0, numatoms):
                idx = i + j
                data = lines[idx].split()
                step.append(int(data[0]))
                
                coords[j] = np.asarray([float(v) for v in data[1:4]])
                force_base[j] = np.asarray([float(v) for v in data[4:7]])
                force_correction[j] = np.asarray([float(v) for v in data[7:10]])
                
            i += numatoms
            all_base_forces.append(force_base)
            all_correction_forces.append(force_correction)
            all_coords.append(coords)
    return np.asarray(all_coords), np.asarray(all_base_forces), np.asarray(all_correction_forces)
            

def loadMTSForces(filename, numatoms):
    all_forces_base = []
    all_forces_correction = []
    all_steps = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        i = 0
        while (i < len(lines)):
            force_base = np.zeros((numatoms, 3), dtype=np.float32)
            force_correction = np.zeros((numatoms, 3), dtype=np.float32)
            step = []
            
            for j in xrange (0, numatoms):
                idx = i + j
                data = lines[idx].split()
                step.append(int(data[0]))
                force_base[j] = np.asarray([float(v) for v in data[1:4]])
                force_correction[j] = np.asarray([float(v) for v in data[4:7]])
                
            i += numatoms
            all_forces_base.append(force_base)
            all_forces_correction.append(force_correction)
            all_steps.append(step)
    return np.asarray(all_forces_base), np.asarray(all_forces_correction), np.asarray(all_steps)


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
                positions[x][0] *= cnst.angstrom2bohr
                positions[x][1] *= cnst.angstrom2bohr
                positions[x][2] *= cnst.angstrom2bohr
            f.write(elems[x] +" "+str(positions[x][0]) +  " " +    str(positions[x][1]) +" "+ str(positions[x][2]) + "\n")
    f.close()


def createCoulombMatrix(elems, positions, cmsize, effectiveCharge=False, alpha=1.0):
    numAtoms = len(elems)
    cm = np.zeros((cmsize, cmsize), dtype=np.float32)
    for i in range(0, numAtoms):
        
        if (effectiveCharge):
            cm[i][i] = 0.5 * math.pow(cnst.element2effective[elems[i]], 2.4)
        else:
            cm[i][i] = 0.5 * math.pow(cnst.element2charge[elems[i]], 2.4)
        for j in range (i + 1, numAtoms):
            dR = positions[i] - positions[j]
            if (effectiveCharge):
                cm[i][j] = (cnst.element2charge[elems[i]] * cnst.element2effective[elems[j]]) / np.power(np.linalg.norm(dR), alpha)
            else:
                cm[i][j] = (cnst.element2charge[elems[i]] * cnst.element2charge[elems[j]]) / np.power(np.linalg.norm(dR), alpha)
            cm[j][i] = cm[i][j]
    return cm


def cm_derivative(elems, positions, n_max, effectiveCharge=False, alpha=1.0):
    num_atoms = len(positions)
    num_cm_entries = (n_max ** 2 - n_max) / 2
    indicies = np.triu_indices(num_atoms, 1)

    descriptor_deriv = np.zeros((num_cm_entries, n_max * 3), dtype=np.float32)
    index = 0
    for q in xrange (0, len(indicies[0])):
        i = indicies[0][q]
        j = indicies[1][q]
        
        if (effectiveCharge):
            qi = cnst.element2effective[elems[i]]
            qj = cnst.element2effective[elems[j]]
        else:
            qi = cnst.element2charge[elems[i]]
            qj = cnst.element2charge[elems[j]]
        
        xyz_i = positions[i]
        xyz_j = positions[j]
        
        normij = np.linalg.norm(xyz_i - xyz_j)
        
        prefactor = (-alpha * qi * qj) / np.power(normij, alpha + 1)

        for k in xrange (0, num_atoms):    
            if (k == i or k == j):
                descriptor_deriv[index][k * 3] = prefactor * ((xyz_i[0] - xyz_j[0]) / normij)
                descriptor_deriv[index][k * 3 + 1] = prefactor * ((xyz_i[1] - xyz_j[1]) / normij)
                descriptor_deriv[index][k * 3 + 2] = prefactor * ((xyz_i[2] - xyz_j[2]) / normij)
            if (k == j):
                descriptor_deriv[index][k * 3] *= -1 
                descriptor_deriv[index][k * 3 + 1] *= -1
                descriptor_deriv[index][k * 3 + 2] *= -1
                    
        index += 1
    
    return descriptor_deriv  


def createRIJMatrix(positions, cmsize):
    numAtoms = len(positions)
    cm = np.zeros((cmsize, cmsize), dtype=np.float32)
    for i in range(0, numAtoms):
        cm[i][i] = 0
        for j in range (i + 1, numAtoms):
            dR = positions[i] - positions[j]
            cm[i][j] = 1 / np.linalg.norm(dR)
            cm[j][i] = cm[i][j]
    return cm


def rij_derivative(positions, natoms):
    index = 0
    
    num_cm_entries = (natoms ** 2 - natoms) / 2
    
    descriptor_deriv = np.zeros((num_cm_entries, natoms * 3), dtype=np.float32)
    
    for i in xrange (0, natoms):
        xyz_i = positions[i]
        for j in xrange (i + 1, natoms):
            xyz_j = positions[j]
            norm3 = np.power(np.linalg.norm(xyz_i - xyz_j), 3)
            prefactor = -1 / norm3
            
            for k in xrange (0, natoms):
                if (k == i):
                    descriptor_deriv[index][k * 3] = prefactor * (xyz_i[0] - xyz_j[0]) 
                    descriptor_deriv[index][k * 3 + 1] = prefactor * (xyz_i[1] - xyz_j[1]) 
                    descriptor_deriv[index][k * 3 + 2] = prefactor * (xyz_i[2] - xyz_j[2]) 
                elif (k == j):
                    descriptor_deriv[index][k * 3] = -1 * prefactor * (xyz_i[0] - xyz_j[0]) 
                    descriptor_deriv[index][k * 3 + 1] = -1 * prefactor * (xyz_i[1] - xyz_j[1]) 
                    descriptor_deriv[index][k * 3 + 2] = -1 * prefactor * (xyz_i[2] - xyz_j[2]) 
            index += 1
            
    return descriptor_deriv

            
def distance_pbc(x0, x1, box_dimensions):
    delta = np.abs(x0 - x1)
    delta = np.where(delta > 0.5 * box_dimensions, delta - box_dimensions, delta)
    return np.sqrt((delta ** 2).sum(axis=-1))


def createLCM(elems, positions, cmsize, effectiveCharge=False, alpha=1.0):
    numAtoms = len(elems)
    rm = np.zeros((cmsize, cmsize), dtype=np.float32)
    
    pos_k = positions[0]
    
    for i in xrange(0, numAtoms):
        if (effectiveCharge):
            qi = cnst.element2effective[elems[i]]
        else:
            qi = cnst.element2charge[elems[i]]
            
        rm[i][i] = 0.5 * qi ** 2.4
        
        for j in xrange (i + 1, numAtoms):
            
            if (effectiveCharge):
                qj = cnst.element2effective[elems[j]]
            else:
                qj = cnst.element2charge[elems[j]]
                
            dRik = positions[i] - pos_k
            dRjk = positions[j] - pos_k
    
            rm[i][j] = (qi * qj) / np.power(np.linalg.norm(dRik) + np.linalg.norm(dRjk), alpha)
            rm[j][i] = rm[i][j]
    return rm


def lcm_derivative(elems, positions, n_max=0, effectiveCharge=False, alpha=1.0):
    
    num_atoms = len(positions)
    num_cm_entries = (num_atoms ** 2 - num_atoms) / 2
    indicies = np.triu_indices(num_atoms, 1)

    descriptor_deriv = np.zeros(((n_max ** 2 - n_max) / 2, n_max * 3), dtype=np.float32)

    xyz_k = positions[0]
    
    for q in xrange (0, len(indicies[0])):
        i = indicies[0][q]
        j = indicies[1][q]
        
        xyz_i = positions[i]
        xyz_j = positions[j]
        
        dRik = xyz_i - xyz_k
        dRjk = xyz_j - xyz_k
       
        if (effectiveCharge):
            qi = cnst.element2effective[elems[i]]
            qj = cnst.element2effective[elems[j]]
        else:
            qi = cnst.element2charge[elems[i]]
            qj = cnst.element2charge[elems[i]]
            
        normik = np.linalg.norm(dRik)
        normjk = np.linalg.norm(dRjk)
        
        normik += 1.e-15
        normjk += 1.e-15
        prefactor = -(alpha * qi * qj) / np.power(normik + normjk, alpha)

        for p in xrange (0, num_atoms):    
            if (p == 0  and  i != 0 or j != 0):
                descriptor_deriv[q][p * 3] = prefactor * -1 * ((dRik[0] / normik) + (dRjk[0] / normjk))
                descriptor_deriv[q][p * 3 + 1] = prefactor * -1 * ((dRik[1] / normik) + (dRjk[1] / normjk))
                descriptor_deriv[q][p * 3 + 2] = prefactor * -1 * ((dRik[2] / normik) + (dRjk[2] / normjk))  
            elif (p == i and i != 0):
                descriptor_deriv[q][p * 3] = prefactor * (dRik[0] / normik) 
                descriptor_deriv[q][p * 3 + 1] = prefactor * (dRik[1] / normik) 
                descriptor_deriv[q][p * 3 + 2] = prefactor * (dRik[2] / normik)
            elif (p == j and j != 0):
                descriptor_deriv[q][p * 3] = prefactor * (dRjk[0] / normjk)
                descriptor_deriv[q][p * 3 + 1] = prefactor * (dRjk[1] / normjk) 
                descriptor_deriv[q][p * 3 + 2] = prefactor * (dRjk[2] / normjk)
    
    return descriptor_deriv  

    
def createLRM(positions, cmsize, alpha=1.0):
    numAtoms = len(positions)
    rm = np.zeros((cmsize, cmsize), dtype=np.float32)
    
    pos_k = positions[0]
    
    for i in xrange(0, numAtoms):
        rm[i][i] = 0
        for j in xrange (i + 1, numAtoms):
            
            dRik = positions[i] - pos_k
            dRjk = positions[j] - pos_k
        
            rm[i][j] = 1 / np.power((np.linalg.norm(dRik) + np.linalg.norm(dRjk)), alpha)
            rm[j][i] = rm[i][j]
    return rm


def lrm_derivative(positions, n_max=0, alpha=1.0):
    # TODO
    num_atoms = len(positions)
    num_cm_entries = (num_atoms ** 2 - num_atoms) / 2
    indicies = np.triu_indices(num_atoms, 1)

    descriptor_deriv = np.zeros(((n_max ** 2 - n_max) / 2, n_max * 3), dtype=np.float32)

    xyz_k = positions[0]
    
    for q in xrange (0, len(indicies[0])):
        i = indicies[0][q]
        j = indicies[1][q]
        
        xyz_i = positions[i]
        xyz_j = positions[j]
        
        dRik = xyz_i - xyz_k
        dRjk = xyz_j - xyz_k
        dRij = xyz_i - xyz_j
       
        normik = np.linalg.norm(dRik) + 1.e-15
        normjk = np.linalg.norm(dRjk) + 1.e-15
        
        norm = np.power(normik + normjk, alpha)
        
        prefactor = -alpha / norm

        for p in xrange (0, num_atoms):    
            if (p == 0  and  i != 0 and j != 0):
                descriptor_deriv[q][p * 3] = prefactor * -1 * ((dRik[0] / normik) + (dRjk[0] / normjk))
                descriptor_deriv[q][p * 3 + 1] = prefactor * -1 * ((dRik[1] / normik) + (dRjk[1] / normjk))
                descriptor_deriv[q][p * 3 + 2] = prefactor * -1 * ((dRik[2] / normik) + (dRjk[2] / normjk))  
            elif (p == i and i != 0):
                descriptor_deriv[q][p * 3] = prefactor * (dRik[0] / normik)
                descriptor_deriv[q][p * 3 + 1] = prefactor * (dRik[1] / normik) 
                descriptor_deriv[q][p * 3 + 2] = prefactor * (dRik[2] / normik) 
            elif (p == j and j != 0):
                descriptor_deriv[q][p * 3] = prefactor * (dRjk[0] / normjk) 
                descriptor_deriv[q][p * 3 + 1] = prefactor * (dRjk[1] / normjk)
                descriptor_deriv[q][p * 3 + 2] = prefactor * (dRjk[2] / normjk)
    
    return descriptor_deriv  
    

def createLocalCM(elems, positions, cmsize):
    '''elems[0], positions[0] assumed to be local centre '''
    numAtoms = len(elems)
    cm = np.zeros((cmsize, cmsize), dtype=np.float32)
    
    elem_k = elems[0]
    pos_k = positions[0]
    
    for i in xrange(0, numAtoms):
        cm[i][i] = 0.5 * math.pow(cnst.element2charge[elem_k], 2.4)
        for j in xrange (i + 1, numAtoms):
            dRik = positions[i] - pos_k
            dRjk = positions[j] - pos_k
            dRij = positions[i] - positions[j]
            cm[i][j] = (cnst.element2charge[elems[[i]]] * cnst.element2charge[elems[[j]]]) / (np.linalg.norm(dRik) + np.linalg.norm(dRjk) + np.linalg.norm(dRij))
            cm[j][i] = cm[i][j]
    return cm
    

def CoM(elems, xyzs):
    v = np.zeros(3)
    M = 0
    for i in xrange (0, len(elems)):
        v = v + np.multiply(cnst.element2mass[elems[i]], xyzs[i])
        M = M + cnst.element2mass[elems[i]]
    v = v / M
    return v


def eucdistance (cm1, cm2):
    indices = np.triu_indices(len(cm1))
    cm1_1 = cm1[indices]
    cm2_1 = cm2[indices]
    return np.sqrt(np.sum(np.square(cm1_1 - cm2_1)))


def distance (cm1, cm2):
    indices = np.triu_indices(len(cm1))
    cm1_1 = cm1[indices]
    cm2_1 = cm2[indices]
    return np.sum(np.abs(cm1_1 - cm2_1))


def createSortedCoulombMatrix(elems, positions, cmsize):
    numAtoms = len(elems)
    cm = np.zeros((cmsize, cmsize), dtype=np.double)
    
    for i in xrange(0, numAtoms):
        cm[i][i] = 0.5 * math.pow(cnst.element2charge[elems[i]], 2.4)
        for j in xrange (i + 1, numAtoms):
            dR = (positions[i] - positions[j])
            cm[i][j] = (cnst.element2charge[elems[i]] * cnst.element2charge[elems[j]]) / np.sqrt(np.dot(dR, dR))
            cm[j][i] = cm[i][j]
            
    norms = np.zeros((cmsize), dtype=np.double)
    indexes = np.zeros((cmsize), dtype=int)
    for i in xrange (0, cmsize):
        norms[i] = np.sqrt(np.dot(cm[i], cm[i]))
        indexes[i] = i
      
    cm2 = np.zeros((cmsize, cmsize), dtype=np.double)
    for i in xrange (0, cmsize):
        cm2[i][i] = cm[i][i]
        for j in xrange (i + 1, cmsize):
            cm2[i][j] = cm[i][j]
            cm2[j][i] = cm2[i][j]
    
    for i in xrange (0, cmsize):
        for j in xrange (i, cmsize):
            if (norms[i] < norms [j]):
                i1 = indexes[i]
                indexes[i] = indexes[j]
                indexes[j] = i1
                
                dp1 = norms[i]
                norms[i] = norms[j]
                norms [j] = dp1
    
    for i in xrange (0, cmsize):
        for j in xrange (i, cmsize) :
            cm [i][j] = cm2[indexes[i]][indexes[j]]
            cm[j][i] = cm[i][j]
                                                                                                                                                                                                                  
    return cm, norms
