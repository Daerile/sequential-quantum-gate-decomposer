# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 14:42:56 2020
Copyright (C) 2020 Peter Rakyta, Ph.D.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see http://www.gnu.org/licenses/.

@author: Peter Rakyta, Ph.D.
"""
## \file example.py
## \brief Simple example python code demonstrating the basic usage of the Python interface of the Quantum State Preparation package

## [import]
from qgd_python.state_preparation.qgd_N_Qubit_State_Preparation_adaptive import qgd_N_Qubit_State_Preparation_adaptive
## [import]     
## [import adaptive]

print('******************** Preparing 3 qubits to general state *******************************')


# cerate unitary q-bit matrix
from scipy.stats import unitary_group
import numpy as np

    
# the number of qubits
qbit_num = 3

# determine the size of the state vector
vector_size = int(2**qbit_num)
   
# creating a random state vector we are preparing to
vec = np.random.rand(vector_size,1) + np.random.rand(vector_size,1)*1j
Stvec = vec / np.linalg.norm(vec)
print(Stvec)

# creating a class
cPreparation = qgd_N_Qubit_State_Preparation_adaptive( Stvec, level_limit_max=5, level_limit_min=0 )

#start state preparation
cPreparation.Start_State_Preparation()

