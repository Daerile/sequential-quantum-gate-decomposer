#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 16:29:39 2020
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

from operations.Operations import Operations
import numpy as np


##
# @brief A base class responsible for constructing matrices of C-NOT, U3
# gates acting on the N-qubit space

class operation_block(Operations):
     
##
# @brief Constructor of the class.
# @param qbit_num The number of qubits
# @return An instance of the class
    def __init__( self, qbit_num ):
        
        Operations.__init__( self, qbit_num ) 
        
        # labels of the free parameters
        self.parameters = list()
        
        # logical value. Set true if block is active, false otherwise
        self.active = True
        # The type of the operation
        self.type = 'block'            
     
    
    
    
##
# @brief Constructor of the class.
# @param parameters List of parameters to calculate the matrix of the operation block
# @return Returns with the matrix of the operation
    def matrix( self, parameters )  :
        
        operation_mtx_tot = np.identity(2**self.qbit_num )
        
        # return with identity if not active
        if not self.active:
            return operation_mtx_tot
         
        if len(parameters) != len(self.parameters):
            raise BaseException('Number of parameters shoould be ' + str(len(self.parameters)) + ' instead of ' + str(len(parameters)) )
         
                
        parameter_idx = len(parameters)
        
        
        
        for idx in range(len(self.operations)-1,-1,-1):
            
            operation = self.operations[idx]
            
            if operation.type == 'cnot':
                operation_mtx = operation.matrix
                
            elif operation.type == 'u3': 
                
                if len(operation.parameters) == 1:
                    operation_mtx = operation.matrix( parameters[parameter_idx-1] )
                    parameter_idx = parameter_idx - 1
                    
                elif len(operation.parameters) == 2:
                    operation_mtx = operation.matrix( parameters[parameter_idx-2:parameter_idx] )
                    parameter_idx = parameter_idx - 2
                    
                elif len(operation.parameters) == 3:
                    operation_mtx = operation.matrix( parameters[parameter_idx-3:parameter_idx] )
                    parameter_idx = parameter_idx - 3
                else:
                    raise BaseException('The U3 operation has wrong number of parameters')
                                 
            elif operation.type == 'general':
                operation_mtx = operation.matrix
             
        
            
            operation_mtx_tot = np.dot(operation_mtx,operation_mtx_tot)
            
        return operation_mtx_tot
        
         
    
    
## 
# @brief Set the number of qubits spanning the matrix of the operation stored in the block
# @param qbit_num The number of qubits spanning the matrix
    def set_qbit_num( self, qbit_num ):
        
        self.qbit_num = qbit_num;
        
        # setting the number of qubit in the operations
        for idx in range(0,len(self.operations)):
           self.operations[idx].set_qbit_num( qbit_num )
         
     
      

    
## add_operation_to_end 
# @brief App  an operation to the list of operations
# @param operation A class describing an operation
    def add_operation_to_end (self, operation ):
                
        # set the number of qubit in the operation
        operation.set_qbit_num( self.qbit_num )
        
        
        self.operations.append(operation)
        
        
        # increase the number of U3 gate parameters by the number of parameters
        self.parameter_num = self.parameter_num + len(operation.parameters)
        
        # increase the number of CNOT operations if necessary
        if operation.type == 'block':
            self.layer_num = self.layer_num + 1
         
        
        # adding parameter labels
        if len( operation.parameters ) > 0:
            self.parameters = self.parameters + operation.parameters
         
        
     
    
## add_operation_to_front
# @brief Add an operation to the front of the list of operations
# @param operation A class describing an operation.
    def add_operation_to_front(self, operation):
        
        
        # set the number of qubit in the operation
        operation.set_qbit_num( self.qbit_num )
        
        if len(self.operations) > 0:
            self.operations = [operation] + self.operations
        else:
            self.operations = [operation]
         
            
        # increase the number of U3 gate parameters by the number of parameters
        self.parameter_num = self.parameter_num + len(operation.parameters)   
        
        # increase the number of CNOT operations if necessary
        if operation.type == 'block':
            self.layer_num = self.layer_num + 1;
         
        
        # adding parameter labels    
        if len( operation.parameters ) > 0:
            self.parameters = operation.parameters + self.parameters
         
        
    
    
    
    
    
     
    
 

