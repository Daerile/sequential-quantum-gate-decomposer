import numpy as np
import random
from qgd_python.decomposition.qgd_N_Qubit_Decomposition_adaptive import qgd_N_Qubit_Decomposition_adaptive       
# number of adaptive levels
levels = 1

# set true to limit calculations to real numbers
real=False


##
# @brief Call to construct random parameter, with limited number of non-trivial adaptive layers
# @param num_of_parameters The number of parameters
def create_randomized_parameters( qbit_num, num_of_parameters, real=False ):


    parameters = np.zeros(num_of_parameters)

    # the number of adaptive layers in one level
    num_of_adaptive_layers = int(qbit_num*(qbit_num-1)/2 * levels)
    
    if (real):
        
        for idx in range(qbit_num):
            parameters[idx*3] = np.random.rand(1)*2*np.pi

    else:
        parameters[0:3*qbit_num] = np.random.rand(3*qbit_num)*np.pi
        pass

    nontrivial_adaptive_layers = np.zeros( (num_of_adaptive_layers ))
    
    for layer_idx in range(num_of_adaptive_layers) :

        nontrivial_adaptive_layer = random.randint(0,1)
        nontrivial_adaptive_layers[layer_idx] = nontrivial_adaptive_layer

        if (nontrivial_adaptive_layer) :
        
            # set the random parameters of the chosen adaptive layer
            start_idx = qbit_num*3 + layer_idx*7
            
            if (real):
                parameters[start_idx]   = np.random.rand(1)*2*np.pi
                parameters[start_idx+1] = np.random.rand(1)*2*np.pi
                parameters[start_idx+4] = np.random.rand(1)*2*np.pi
            else:
                end_idx = start_idx + 7
                parameters[start_idx:end_idx] = np.random.rand(7)*2*np.pi
         
        
    
    #print( parameters )
    return parameters, nontrivial_adaptive_layers

def perf_collection():
    import timeit
    objs = [] #avoid destructor calls by storing in a list
    for qbit_num in range(5, 10):
        matrix_size = 1 << qbit_num
        # creating a class to decompose the unitary
        cDecompose = qgd_N_Qubit_Decomposition_adaptive( np.eye(matrix_size), level_limit_max=levels, level_limit_min=0 )
        objs.append(cDecompose)
        # adding decomposing layers to the gat structure
        for idx in range(levels):
            cDecompose.add_Adaptive_Layers()
        
        cDecompose.add_Finalyzing_Layer_To_Gate_Structure()
        
        # get the number of free parameters
        num_of_parameters = cDecompose.get_Parameter_Num()
        
        # create randomized parameters
        result = [None]
        def cpuInvoke():
            result[0] = cDecompose.Optimization_Problem_Combined( parameters, True )
        parameters, nontrivial_adaptive_layers = create_randomized_parameters( qbit_num, num_of_parameters, real=real )            
        t = timeit.timeit(cpuInvoke, number=1)
        print(t, result[0])
        result = [None]
        def dfeInvoke():
            result[0] = cDecompose.Optimization_Problem_Combined( parameters, False )
        t = timeit.timeit(dfeInvoke, number=1)
        print(t, result[0])
    
perf_collection()
