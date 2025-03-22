'''
Source code for the decomposition of MPS to quantum circuits from the following repository:
https://github.com/ToyotaCRDL/lchs-for-pde/blob/main/lib/utils.py

Author: Yuki Sato
GitHub: @yksat
'''

import numpy as np
from qiskit import QuantumCircuit
from qiskit.synthesis import TwoQubitBasisDecomposer, OneQubitEulerDecomposer
from qiskit.circuit.library import CXGate
import scikit_tt.tensor_train as tt
from scikit_tt import TT


def mps2circuit(mps: TT, D:int=1) -> QuantumCircuit:

    '''
    convert right canonical MPS to qiskit's Quantumcircuit object

    mps: TT object of scikit-tt whose cores must be 4-dimensional numpy.ndarray with the size of (a, i, 1, b),
           where a and b are the bond dimensions and i is the physical dimension.
           i must be 2.
    '''

    qc = QuantumCircuit(len(mps.cores))
    decomposer = TwoQubitBasisDecomposer(CXGate(), euler_basis='U')
    mps_l = mps.copy()

    for l in range(D):

        if max(mps_l.ranks) == 1:
            print("MPS has been disentangled. No additional layers are required.")
            break

        qc_tmp = QuantumCircuit(len(mps.cores))

        mps_l_trunc = mps_l.copy()
        mps_l_trunc.ortho(max_rank=2)
        cores = (1. / mps_l_trunc.norm() * mps_l_trunc).cores
        mpd_unitaries = []
        for i, core in enumerate(cores):
    
            if i == 0:
                unitary = np.random.rand(4, 4) + 1j*np.random.rand(4, 4)
                unitary[:, 0] = core.flatten()
    
                if np.abs(np.linalg.norm(unitary[:, 0]) - 1) > 1e-9:
                    raise ValueError('MPS is not in the canonical form.')
                
                tmp = unitary[:, 1] - np.vdot(unitary[:, 0], unitary[:, 1]) * unitary[:, 0]
                unitary[:, 1] = tmp / np.linalg.norm(tmp)
    
                tmp = unitary[:, 2] - np.vdot(unitary[:, 0], unitary[:, 2]) * unitary[:, 0] \
                                    - np.vdot(unitary[:, 1], unitary[:, 2]) * unitary[:, 1]
                unitary[:, 2] = tmp / np.linalg.norm(tmp)
    
                tmp = unitary[:, 3] - np.vdot(unitary[:, 0], unitary[:, 3]) * unitary[:, 0] \
                                    - np.vdot(unitary[:, 1], unitary[:, 3]) * unitary[:, 1] \
                                    - np.vdot(unitary[:, 2], unitary[:, 3]) * unitary[:, 2]
                unitary[:, 3] = tmp / np.linalg.norm(tmp)
            

                qc_tmp.compose(decomposer(unitary), qubits=[qc.num_qubits - 2, qc.num_qubits - 1], inplace=True)
                mpd_unitaries.append(unitary.transpose().conjugate())
            
            elif i < len(mps.cores) - 1:
    
                unitary = np.random.rand(4, 4) + 1j*np.random.rand(4, 4)
                unitary[:, 0] = core[0, :, :, :].flatten()
                unitary[:, 2] = core[1, :, :, :].flatten()
    
                if np.abs(np.vdot(unitary[:, 0], unitary[:, 2])) > 1e-9 or np.abs(np.linalg.norm(unitary[:, 0]) - 1) > 1e-9 or np.abs(np.linalg.norm(unitary[:, 2]) - 1) > 1e-9:
                    raise ValueError('MPS is not in the canonical form.')
                
                tmp = unitary[:, 1] - np.vdot(unitary[:, 0], unitary[:, 1]) * unitary[:, 0] \
                                    - np.vdot(unitary[:, 2], unitary[:, 1]) * unitary[:, 2]
                unitary[:, 1] = tmp / np.linalg.norm(tmp)
    
                tmp = unitary[:, 3] - np.vdot(unitary[:, 0], unitary[:, 3]) * unitary[:, 0] \
                                    - np.vdot(unitary[:, 1], unitary[:, 3]) * unitary[:, 1] \
                                    - np.vdot(unitary[:, 2], unitary[:, 3]) * unitary[:, 2]
                unitary[:, 3] = tmp / np.linalg.norm(tmp)

                qc_tmp.compose(decomposer(unitary), qubits=[qc.num_qubits - 2 - i, qc.num_qubits - 1 - i], inplace=True)
                mpd_unitaries.append(unitary.transpose().conjugate())
            
            else:
                unitary = core.reshape((2, 2)).transpose()
                theta, phi, lam = OneQubitEulerDecomposer().angles(unitary)
                qc_tmp.u(theta, phi, lam, 0) 
                mpd_unitaries.append(unitary.transpose().conjugate())

        qc.compose(qc_tmp, front=True, inplace=True)

        for i, mpd_unitary in enumerate(reversed(mpd_unitaries)):

            if i == 0:
                mpd = tt.eye([2]*(len(mps.cores)-1)).concatenate(TT(mpd_unitary.reshape((2, 2))))

            elif i == 1:
                mpd = tt.eye([2]*(len(mps.cores)-2)).concatenate(TT(mpd_unitary.reshape((2, 2, 2, 2))))

            elif i < len(mps.cores) - 1:
                mpd = tt.eye([2]*(len(mps.cores)-i-1)).concatenate(TT(mpd_unitary.reshape((2, 2, 2, 2)))).concatenate(tt.eye([2]*(i-1)))

            else:
                mpd = TT(mpd_unitary.reshape((2, 2, 2, 2))).concatenate(tt.eye([2]*(i-1)))
                
            mps_l = mpd.dot(mps_l)
            mps_l.ortho(threshold=1e-12)
                
    return qc

