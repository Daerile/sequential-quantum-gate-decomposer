import numpy as np
import random
from qgd_python.decomposition.qgd_N_Qubit_Decomposition_adaptive import qgd_N_Qubit_Decomposition_adaptive       
# number of adaptive levels
levels = 5

# set true to limit calculations to real numbers
real=False

isGroq = True


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
    import timeit, os, pickle
    params = {}
    if os.path.exists("squanderperf.pickle"):
        with open("squanderperf.pickle", 'rb') as f:
            results = pickle.load(f)
    else: results = {}
    alldev = (1, 2) if isGroq else (0, 1, 2, 3)    
    qbit_range = list(range(5, 10+1))
    if not isGroq in results: results[isGroq] = {}    
    for numdev in alldev:
        results[isGroq][numdev] = {}
        for qbit_num in qbit_range:
            matrix_size = 1 << qbit_num
            # creating a class to decompose the unitary
            result = [None]
            def initInvoke():
                result[0] = qgd_N_Qubit_Decomposition_adaptive( np.eye(matrix_size), level_limit_max=levels, level_limit_min=0, accelerator_num = numdev )
            inittime = timeit.timeit(initInvoke, number=1)
            cDecompose = result[0]
            def uploadInvoke():
                cDecompose.Upload_Umtx_to_DFE() #program and matrix load are currently both here
            uptime = timeit.timeit(uploadInvoke, number=1)
            uptime = timeit.timeit(uploadInvoke, number=1)
            # adding decomposing layers to the gat structure
            for idx in range(levels):
                cDecompose.add_Adaptive_Layers()
            
            cDecompose.add_Finalyzing_Layer_To_Gate_Structure()
            cDecompose.set_Cost_Function_Variant(2)
            
            # get the number of free parameters
            num_of_parameters = cDecompose.get_Parameter_Num()
            print("Qubits:", qbit_num, "Levels:", levels, "Parameters:", num_of_parameters) 
            # create randomized parameters            
            if numdev == 1:
                parameters, nontrivial_adaptive_layers = create_randomized_parameters( qbit_num, num_of_parameters, real=real )
                params[qbit_num] = parameters
            else: parameters = params[qbit_num] 
            result = [None]
            def dfeInvoke():
                result[0] = cDecompose.Optimization_Problem_Combined( parameters )
            t = timeit.timeit(dfeInvoke, number=1)
            print(numdev, qbit_num, inittime, uptime, t, result[0][0])
            results[isGroq][numdev][qbit_num] = (uptime, t)
    with open("squanderperf.pickle", 'wb') as f:
        pickle.dump(results, f)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(8.5, 4)
    ax[0].set_xticks(qbit_range)
    ax[1].set_xticks(qbit_range)
    ax[0].set_xlabel("# of qubits")
    ax[1].set_xlabel("# of qubits")
    ax[0].set_ylabel("Time (s)")
    ax[1].set_ylabel("Time (s)")
    ax[1].set_yscale('log', base=2)
    ax[0].set_title("Quantum Unitary Simulator Initialization")
    ax[1].set_title("Quantum Unitary Simulator Performance")
    for g in results:
        for numdev in results[g]:
            ax[0].plot(qbit_range, [results[g][numdev][x][0] for x in qbit_range], label="CPU" if numdev==0 else str(numdev) + " " + ("Groq" if g else "DFE"))
            ax[1].plot(qbit_range, [results[g][numdev][x][1] for x in qbit_range], label="CPU" if numdev==0 else str(numdev) + " " + ("Groq" if g else "DFE"))
    ax[0].legend()
    ax[1].legend()
    fig.savefig('squanderperf.svg', format='svg')
perf_collection()
