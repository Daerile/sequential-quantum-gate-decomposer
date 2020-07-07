#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 15:44:26 2020

@author: rakytap
"""

from .Decomposition_Base import def_layer_num
from .Decomposition_Base import Decomposition_Base
from .Two_Qubit_Decomposition import Two_Qubit_Decomposition
from .Sub_Matrix_Decomposition import Sub_Matrix_Decomposition
import numpy as np
from numpy import linalg as LA
import time

##
# @brief A class for the decomposition of N-qubit unitaries into U3 and CNOT gates.

class N_Qubit_Decomposition(Decomposition_Base):
    
    
## 
# @brief Constructor of the class.
# @param Umtx The unitary matrix
# @param optimize_layer_num Optional logical value. If true, then the optimalization tries to determine the lowest number of the layers needed for the decomposition. If False (default), the optimalization is performed for the maximal number of layers.
# @param parallel Optional logical value. I true, parallelized optimalization id used in the decomposition. The parallelized optimalization is efficient if the number of blocks optimized in one shot (given by attribute @optimalization_block) is at least 10). For False (default) sequential optimalization is applied
# @param method Optional string value labeling the optimalization method used in the calculations. Deafult is L-BFGS-B. For details see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
# @param identical_blocks A dictionary {'n': integer} indicating that how many identical succesive blocks should be used in the disentanglement of the nth qubit from the others
# @param initial_guess String indicating the method to guess initial values for the optimalization. Possible values: 'zeros' (deafult),'random'
# @return An instance of the class
    def __init__( self, Umtx, optimize_layer_num=False, parallel= False, method='L-BFGS-B',  identical_blocks = dict(), initial_guess= 'zeros'):  
        
        Decomposition_Base.__init__( self, Umtx, parallel=parallel, method=method, initial_guess=initial_guess   ) 
            
                
        # logical value. Set true if finding the minimum number of operation layers is required (default), or false when the maximal number of CNOT gates is used (ideal for general unitaries).
        self.optimize_layer_num  = optimize_layer_num
        
        # number of iteratrion loops in the finale optimalization
        self.iteration_loops = 3
        
        # The maximal allowed error of the optimalization problem
        self.optimalization_tolerance = 1e-7
        
        # Maximal number of iterations in the optimalization process
        self.max_iterations = int(1e4)
    
        # number of operators in one sub-layer of the optimalization process
        self.optimalization_block = 1    
        if parallel:
            self.optimalization_block = 10  
        
        # The number of gate blocks used in the decomposition
        self.max_layer_num = def_layer_num.get(str(self.qbit_num), 200)
        
        # The number of successive identical blocks
        self.identical_blocks = identical_blocks  
        
        
                        
    
    
    
    
## start_decomposition
# @brief Start the disentanglig process of the least significant two qubit unitary
# @param finalize_decomposition Optional logical parameter. If true (default), the decoupled qubits are rotated into state |0> when the disentangling of the qubits is done. Set to False to omit this procedure
    def start_decomposition(self, finalize_decomposition=True):
        
        
            
        
        print('***************************************************************')
        print('Starting to disentangle ' + str(self.qbit_num) + '-qubit matrix')
        print('***************************************************************')
        
        #measure the time for the decompositin        
        start_time = time.time()
            
        # create an instance of class to disentangle the given qubit pair
        cSub_decomposition = Sub_Matrix_Decomposition(self.Umtx, optimize_layer_num=self.optimize_layer_num, parallel=self.parallel, identical_blocks=self.identical_blocks)   
        
        # The maximal error of the optimalization problem
        cSub_decomposition.optimalization_tolerance = self.optimalization_tolerance
        
        
        # ---------- setting the decomposition parameters  --------
        
        # setting the maximal number of iterations in the disentangling process
        cSub_decomposition.optimalization_block = self.optimalization_block
        
        # setting the number of operators in one sub-layer of the disentangling process
        cSub_decomposition.max_iterations = self.max_iterations
        
        # setting the number of parameters in one sub-layer of the disentangling process
        cSub_decomposition.max_layer_num = self.max_layer_num
            
        # start to disentangle the qubit pair
        cSub_decomposition.disentangle_submatrices()
                                
        if not cSub_decomposition.subdisentaglement_done:
            return
        
        
        # saving the subunitarization operations
        self.save_subdecomposition_results( cSub_decomposition )
        
        # decompose the qubits in the disentangled submatrices
        self.decompose_submatrix()
            
        if finalize_decomposition:
            # finalizing the decompostition
            self.finalize_decomposition()
            
            # simplify layers
            #self.simplify_layers()
            
            # final tuning of the decomposition parameters
            self.final_optimalization()
            
            matrix_new = self.get_transformed_matrix(self.optimized_parameters, self.operations )    

            # calculating the final error of the decomposition
            self.decomposition_error = LA.norm(matrix_new*np.exp(np.complex(0,-np.angle(matrix_new[0,0]))) - np.identity(len(matrix_new))*abs(matrix_new[0,0]), 2)
            
            # get the number of gates used in the decomposition
            gates_num = self.get_gate_nums()
            print( 'In the decomposition with error = ' + str(self.decomposition_error) + ' were used ' + str(self.layer_num) + ' layers with '  + str(gates_num['u3']) + ' U3 operations and ' + str(gates_num['cnot']) + ' CNOT gates.' )        
            
        
        print("--- In total %s seconds elapsed during the decomposition ---" % (time.time() - start_time))            
        
    
        
    
    
## save_subdecomposition_results
# @brief stores the calculated parameters and operations of the sub-decomposition processes
# @param cSub_decomposition An instance of class Sub_Two_Qubit_Decomposition used to disentangle qubit pairs from the others.
# @param qbits_reordered A permutation of qubits that was applied on the initial unitary in prior of the sub decomposition. (This is needed to restore the correct qubit indices.)
    def save_subdecomposition_results( self, cSub_decomposition ):
                
        # get the unitarization operations
        operations = cSub_decomposition.operations
        
        # get the unitarization parameters
        parameters = cSub_decomposition.optimized_parameters
        
        # create new operations in the original, not reordered qubit list
        for idx in range(len(operations)-1,-1,-1):
            operation = operations[idx]
            self.add_operation_to_front( operation )   
            
        self.optimized_parameters = np.concatenate((parameters, self.optimized_parameters))
            
            
    
    
## start_decomposition
# @brief Start the decompostion process of the two-qubit unitary
    def decompose_submatrix(self):
        
        if self.decomposition_finalized:
            print('Decomposition was already finalized')
            return
                       
        
        # obtaining the subdecomposed submatrices
        subdecomposed_mtx = self.get_transformed_matrix( self.optimized_parameters, self.operations )        
        
        # get the most unitary submatrix
        # get the number of 2qubit submatrices
        submatrices_num_row = 2#2^(self.qbit_num-2)
        
        # get the size of the submatrix
        submatrix_size = int(len(subdecomposed_mtx)/2)
        
        # fill up the submatrices
        unitary_error_min = None
        most_unitary_submatrix = []
        for idx in range(0,submatrices_num_row):
            for jdx in range(0,submatrices_num_row):
                submatrix = subdecomposed_mtx[ (idx*submatrix_size):((idx+1)*submatrix_size+1), (jdx*submatrix_size):((jdx+1)*submatrix_size+1) ]
                
                submatrix_prod = np.dot(submatrix, submatrix.conj().T)
                unitary_error = LA.norm( submatrix_prod - np.identity(len(submatrix_prod))*submatrix_prod[0,0] )
                if (unitary_error_min is None) or unitary_error < unitary_error_min:
                    unitary_error_min = unitary_error
                    most_unitary_submatrix = submatrix
                
            
        
        
        # if the qubit number in the submatirx is greater than 2 new N-qubit decomposition is started
        if len(most_unitary_submatrix) > 4:
            cdecomposition = N_Qubit_Decomposition(most_unitary_submatrix, optimize_layer_num=True, initial_guess=self.initial_guess, identical_blocks=self.identical_blocks)
            
            # setting operation layer
            if len(most_unitary_submatrix) <= 8:
                # for three qubits
                cdecomposition.set_optimalization_blocks(self.optimalization_block)
                cdecomposition.set_parallel( False )
            else:
                # for 4 or more qubits
                cdecomposition.set_optimalization_blocks( self.optimalization_block )
                cdecomposition.set_parallel( self.parallel )

            # starting the decomposition of the random unitary
            cdecomposition.start_decomposition(finalize_decomposition=False)
        else:

            # decompose the chosen 2-qubit unitary
            cdecomposition = Two_Qubit_Decomposition(most_unitary_submatrix, optimize_layer_num=True, initial_guess=self.initial_guess)
        
            # starting the decomposition of the random unitary
            cdecomposition.start_decomposition(finalize_decomposition=False)
        
                
        # saving the decomposition operations
        self.save_subdecomposition_results( cdecomposition )
        
        
     
    
    
    
##
# @brief final optimalization procedure improving the accuracy of the decompositin when all the qubits were already disentangled.
    def final_optimalization( self ):
        
        print(' ')
        print('Final fine tuning of the parameters')
        
        # setting the global minimum
        self.global_target_minimum = 0
        
        self.solve_optimalization_problem( optimalization_problem=self.final_optimalization_problem, solution_guess=self.optimized_parameters) 
        
        
    
    
    
    
## final_optimalization_problem
# @brief The optimalization problem to be solved in order to disentangle the two least significant qubits
# @param parameters An array of the free parameters to be optimized. (The number of teh free paramaters should be equal to the number of parameters in one sub-layer)
# @return Returns with the value representing the difference between the decomposed matrix and the unity.
    def final_optimalization_problem( self, parameters ):
               
        # get the transformed matrix with the operations in the list
        matrix_new = self.get_transformed_matrix( parameters, self.operations, initial_matrix=self.Umtx)
                
        matrix_new = matrix_new - np.identity(len(matrix_new))*matrix_new[0,0]
        
        cost_function = np.sum( np.multiply(matrix_new, matrix_new.conj() ) )
        #cost_function = np.linalg.norm( matrix_new, 2 )
        
        return np.real(cost_function)
        
       
        
      
##  
# @brief Call to simplify the gate structure in each layers if possible
    def simplify_layers(self):
        
        parameter_idx = 0
        
        # the number of parameters after the simplification
        parameter_num = 0
        
        operations = list()
        optimized_parameters = np.array()
        
        for block_idx in range(0,len(self.operations)):
            block = self.operations[block_idx]
             
            #number of perations in the block
            parameter_num_layer = len(block.parameters)
            
            # get the number of CNOT gates and store the block operations if the number of CNOT gates cannot be reduced
            gate_nums = block.get_gate_nums()
            if gate_nums.get('cnot',0) < 2:
                operations.append(block)
                optimized_parameters = np.concatenate(( optimized_parameters, self.optimized_parameters[parameter_idx:parameter_idx+parameter_num_layer] ))
                parameter_idx = parameter_idx + parameter_num_layer
                continue
            
            full_matrix = block.matrix( self.optimized_parameters[parameter_idx:parameter_idx+parameter_num_layer])
            
            # get the target bit
            target_qbit = None
            control_qbit = None
            for block_op_idx in range(0,len(block.operations)):
                operation = block.operations[block_op_idx]
                if operation.type == 'cnot':
                    target_qbit = operation.target_qbit
                    control_qbit = operation.control_qbit
                    break
            
            if target_qbit is None or control_qbit is None:
                continue
            
            # get the matrix of the two qubit space
            qbits_reordered = list()
            for qbit_idx in range(self.qbit_num-1,-1,-1):
                if qbit_idx != target_qbit and qbit_idx != control_qbit:
                    qbits_reordered.append(qbit_idx)
            
            qbits_reordered.append(control_qbit)
            qbits_reordered.append(target_qbit)            
            
            # contruct the permutation to get the basis for the reordered qbit list
            bases_reorder_indexes = self.get_basis_of_reordered_qubits( qbits_reordered )
            
            # reorder the matrix elements to get the submatrix representing the given 2-qubit matrix into the corner
            full_matrix_reordered = full_matrix[:, bases_reorder_indexes][bases_reorder_indexes]
            submatrix = full_matrix_reordered[0:4, 0:4]
            
            # decompose the chosen 2-qubit unitary
            cdecomposition = Two_Qubit_Decomposition(submatrix, optimize_layer_num=True, initial_guess=self.initial_guess)
            
            # set the maximal number of layers
            cdecomposition.max_layer_num = gate_nums['cnot']-1
        
            # starting the decomposition of the random unitary
            cdecomposition.start_decomposition(finalize_decomposition=True)
            
            # check whether simplification was succesfull
            if not cdecomposition.check_optimalization_solution():
                continue
            
            # replace the block with the simplified one
            for operation_idx in range(0,len(cdecomposition.operations)):
                operation = cdecomposition.operations[operation_idx]
                
                #reorder the qubits and increase the number of qubits
                operation.set_qbit_num( self.qbit_num )
                
                if operation.type == 'cnot':
                    operation.control_qbit = control_qbit
                    operation.target_qbit = target_qbit
                elif operation.type == 'u3':
                    pass
                
                
                
                operations.append(cdecomposition.operations[LLLLLLLLLLLL])
            
            
            print(parameter_num_layer)
    
    
    
    
    

