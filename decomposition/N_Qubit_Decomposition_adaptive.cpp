/*
Created on Fri Jun 26 14:13:26 2020
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
*/
/*! \file N_Qubit_Decomposition_adaptive.cpp
    \brief Base class to determine the decomposition of a unitary into a sequence of two-qubit and one-qubit gate gates.
    This class contains the non-template implementation of the decomposition class
*/

#include "N_Qubit_Decomposition_adaptive.h"
#include "N_Qubit_Decomposition_custom.h"
#include "N_Qubit_Decomposition_Cost_Function.h"
#include "Random_Orthogonal.h"

#include <time.h>
#include <stdlib.h>


Matrix_real create_random_paramaters( Gates_block* gate_structure ) {

    int parameter_num = gate_structure->get_parameter_num();

    Matrix_real parameters(1, parameter_num);

    for(int idx = 0; idx < parameter_num; idx++) {
         if ( idx % 5 == 0 ) {
             if ( rand() % 2 == 0 ) {
                 parameters[idx] = 0.0;
             }
             else {
                 parameters[idx] = M_PI;
             }
         }
         else {
             parameters[idx] = (2*double(rand())/double(RAND_MAX)-1)*2*M_PI;
         }
    }

    return parameters;


}

static int limit_max=20;

/**
@brief Nullary constructor of the class.
@return An instance of the class
*/
N_Qubit_Decomposition_adaptive::N_Qubit_Decomposition_adaptive() : N_Qubit_Decomposition_Base() {

    // initialize custom gate structure
    gate_structure = NULL;

    // set the level limit
    level_limit = 0;

    iter_max = 10000;
    gradient_threshold = 1e-8;

    srand(time(NULL));   // Initialization, should only be called once.
}

/**
@brief Constructor of the class.
@param Umtx_in The unitary matrix to be decomposed
@param qbit_num_in The number of qubits spanning the unitary Umtx
@param optimize_layer_num_in Optional logical value. If true, then the optimization tries to determine the lowest number of the layers needed for the decomposition. If False (default), the optimization is performed for the maximal number of layers.
@param initial_guess_in Enumeration element indicating the method to guess initial values for the optimization. Possible values: 'zeros=0' ,'random=1', 'close_to_zero=2'
@return An instance of the class
*/
N_Qubit_Decomposition_adaptive::N_Qubit_Decomposition_adaptive( Matrix Umtx_in, int qbit_num_in, int level_limit_in, int level_limit_min_in, guess_type initial_guess_in ) : N_Qubit_Decomposition_Base(Umtx_in, qbit_num_in, false, initial_guess_in) {


    // initialize custom gate structure
    gate_structure = NULL;

    // set the level limit
    level_limit = level_limit_in;
    level_limit_min = level_limit_min_in;

    // Maximal number of iteartions in the optimization process
    max_iterations = 4;

    iter_max = 10000;
    gradient_threshold = 1e-8;

    srand(time(NULL));   // Initialization, should only be called once.
}



/**
@brief Constructor of the class.
@param Umtx_in The unitary matrix to be decomposed
@param qbit_num_in The number of qubits spanning the unitary Umtx
@param optimize_layer_num_in Optional logical value. If true, then the optimization tries to determine the lowest number of the layers needed for the decomposition. If False (default), the optimization is performed for the maximal number of layers.
@param initial_guess_in Enumeration element indicating the method to guess initial values for the optimization. Possible values: 'zeros=0' ,'random=1', 'close_to_zero=2'
@return An instance of the class
*/
N_Qubit_Decomposition_adaptive::N_Qubit_Decomposition_adaptive( Matrix Umtx_in, int qbit_num_in, int level_limit_in, int level_limit_min_in, std::vector<matrix_base<int>> topology_in, guess_type initial_guess_in ) : N_Qubit_Decomposition_Base(Umtx_in, qbit_num_in, false, initial_guess_in) {


    // initialize custom gate structure
    gate_structure = NULL;

    // set the level limit
    level_limit = level_limit_in;
    level_limit_min = level_limit_min_in;

    // Maximal number of iteartions in the optimization process
    max_iterations = 4;

    // setting the topology
    topology = topology_in;

    srand(time(NULL));   // Initialization, should only be called once.
}



/**
@brief Destructor of the class
*/
N_Qubit_Decomposition_adaptive::~N_Qubit_Decomposition_adaptive() {


    if ( gate_structure != NULL ) {
        // release custom gate structure
        delete gate_structure;
        gate_structure = NULL;
    }

}



/**
@brief Start the disentanglig process of the unitary
@param finalize_decomp Optional logical parameter. If true (default), the decoupled qubits are rotated into state |0> when the disentangling of the qubits is done. Set to False to omit this procedure
@param prepare_export Logical parameter. Set true to prepare the list of gates to be exported, or false otherwise.
*/
void
N_Qubit_Decomposition_adaptive::start_decomposition(bool prepare_export) {



    if (verbose) {
        printf("***************************************************************\n");
        printf("Starting to disentangle %d-qubit matrix\n", qbit_num);
        printf("***************************************************************\n\n\n");
    }

    // temporarily turn off OpenMP parallelism
#if BLAS==0 // undefined BLAS
    num_threads = omp_get_max_threads();
    omp_set_num_threads(1);
#elif BLAS==1 // MKL
    num_threads = mkl_get_max_threads();
    MKL_Set_Num_Threads(1);
#elif BLAS==2 //OpenBLAS
    num_threads = openblas_get_num_threads();
    openblas_set_num_threads(1);
#endif

    //measure the time for the decompositin
    tbb::tick_count start_time = tbb::tick_count::now();

    if (level_limit == 0 ) {
        std::cout << "please increase level limit" << std::endl;
        exit(-1);
    }
/*
int qbit_num_loc = 2;
int target_qbit = 0;
int control_qbit = 1;

                        RX*   rx_gate_1   = new RX(qbit_num_loc, target_qbit);
                        Adaptive*   cz_gate     = new Adaptive(qbit_num_loc, target_qbit, control_qbit);
                        RZ*   rz_gate     = new RZ(qbit_num_loc, control_qbit);
                        RX*   rx_gate_2   = new RX(qbit_num_loc, target_qbit);


                        Gates_block* czr_gate = new Gates_block(qbit_num_loc);
                        czr_gate->add_gate(rx_gate_1);
                        czr_gate->add_gate(cz_gate);
                        czr_gate->add_gate(rz_gate);
                        czr_gate->add_gate(rx_gate_2);

Matrix_real params(1,4);
params[0] = -M_PI/2;
params[1] = -M_PI/2;
params[2] = M_PI;
params[3] = M_PI/2;

Matrix mtx = czr_gate->get_matrix( params );
mtx.print_matrix();
exit(-1);
*/
/*
optimized_parameters_mtx.print_matrix();
std::cout << gate_structure->get_parameter_num() << std::endl;
Matrix mtx = gate_structure->get_matrix( optimized_parameters_mtx );
mtx.print_matrix();

*/

//    combine( gate_structure );
//std::cout << optimization_problem(optimized_parameters_mtx.get_data()) << std::endl;


//list_gates( 0);

    // calculating the final error of the decomposition
 //   Matrix matrix_decomposed2 = get_transformed_matrix(optimized_parameters_mtx, gates.begin(), gates.size(), Umtx );
//matrix_decomposed2.print_matrix();

//exit(-1);


//iteration_loops[4] = 2;    
//Gates_block* gate_structure_loc = gate_structure->clone();
//insert_random_layers( gate_structure_loc, optimized_parameters_mtx );


    double optimization_tolerance_orig = optimization_tolerance;
    optimization_tolerance = 1e-4;

    Gates_block* gate_structure_loc = NULL;
    if ( gate_structure != NULL ) {
        std::cout << "Using imported gate structure for the decomposition." << std::endl;
        gate_structure_loc = optimize_imported_gate_structure(optimized_parameters_mtx);
    }
    else {
        std::cout << "Construct initial gate structure for the decomposition." << std::endl;
        gate_structure_loc = determine_initial_gate_structure(optimized_parameters_mtx);
    }




    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "**************************************************************" << std::endl;
    std::cout << "***************** Compressing Gate structure *****************" << std::endl;
    std::cout << "**************************************************************" << std::endl;

    int iter = 0;
    int uncompressed_iter_num = 0;
    while ( iter<25 || uncompressed_iter_num <= 5 ) {

        if ( current_minimum > 1e-2 ) {
             for (int idx=0; idx<optimized_parameters_mtx.size(); idx++ ) {
                 optimized_parameters_mtx[idx] = (2*double(rand())/double(RAND_MAX)-1)*2*M_PI;
             }
        }


        std::cout << "iteration " << iter+1 << ": ";
        Gates_block* gate_structure_compressed = compress_gate_structure( gate_structure_loc );

        if ( gate_structure_compressed->get_gate_num() < gate_structure_loc->get_gate_num() ) {
            uncompressed_iter_num = 0;
        }
        else {
            uncompressed_iter_num++;
        }

        if ( gate_structure_compressed != gate_structure_loc ) {
            delete( gate_structure_loc );
            gate_structure_loc = gate_structure_compressed;
            gate_structure_compressed = NULL;
        }

        iter++;

        if (uncompressed_iter_num>10) break;

    }


    std::cout << "**************************************************************" << std::endl;
    std::cout << "************ Final tuning of the Gate structure **************" << std::endl;
    std::cout << "**************************************************************" << std::endl;

    optimization_tolerance = optimization_tolerance_orig;

    // store the decomposing gate structure    
    combine( gate_structure_loc );
    optimization_block = get_gate_num();

std::cout << optimization_problem(optimized_parameters_mtx.get_data()) << std::endl;


    Gates_block* gate_structure_tmp = replace_trivial_CRY_gates( gate_structure_loc, optimized_parameters_mtx );
    Matrix_real optimized_parameters_save = optimized_parameters_mtx;

    release_gates();
    optimized_parameters_mtx = optimized_parameters_save;

    combine( gate_structure_tmp );
    delete( gate_structure_tmp );
    delete( gate_structure_loc );

std::cout << optimization_problem(optimized_parameters_mtx.get_data()) << std::endl;

    // reset the global minimum before final tuning
    current_minimum = DBL_MAX;

    // final tuning of the decomposition parameters
    final_optimization();



    // prepare gates to export
    if (prepare_export) {
        prepare_gates_to_export();
    }

    // calculating the final error of the decomposition
    Matrix matrix_decomposed = get_transformed_matrix(optimized_parameters_mtx, gates.begin(), gates.size(), Umtx );
    calc_decomposition_error( matrix_decomposed );


    // get the number of gates used in the decomposition
    gates_num gates_num = get_gate_nums();

    if (verbose) {
        std::cout << "In the decomposition with error = " << decomposition_error << " were used " << layer_num << " gates with:" << std::endl;
        if ( gates_num.u3>0 ) std::cout << gates_num.u3 << " U3 gates," << std::endl;
        if ( gates_num.rx>0 ) std::cout << gates_num.rx << " RX gates," << std::endl;
        if ( gates_num.ry>0 ) std::cout << gates_num.ry << " RY gates," << std::endl;
        if ( gates_num.rz>0 ) std::cout << gates_num.rz << " RZ gates," << std::endl;
        if ( gates_num.cnot>0 ) std::cout << gates_num.cnot << " CNOT gates," << std::endl;
        if ( gates_num.cz>0 ) std::cout << gates_num.cz << " CZ gates," << std::endl;
        if ( gates_num.ch>0 ) std::cout << gates_num.ch << " CH gates," << std::endl;
        if ( gates_num.x>0 ) std::cout << gates_num.x << " X gates," << std::endl;
        if ( gates_num.sx>0 ) std::cout << gates_num.sx << " SX gates," << std::endl;
        if ( gates_num.syc>0 ) std::cout << gates_num.syc << " Sycamore gates," << std::endl;
        if ( gates_num.un>0 ) std::cout << gates_num.un << " UN gates," << std::endl;
        if ( gates_num.cry>0 ) std::cout << gates_num.cry << " CRY gates," << std::endl;
        if ( gates_num.adap>0 ) std::cout << gates_num.adap << " Adaptive gates," << std::endl;
        std::cout << std::endl;
        tbb::tick_count current_time = tbb::tick_count::now();
        std::cout << "--- In total " << (current_time - start_time).seconds() << " seconds elapsed during the decomposition ---" << std::endl;
    }

/*
Matrix_real param0 = optimized_parameters_mtx;
std::cout << "gradient test "<< std::endl;
std::vector<Matrix>&& Umtx_deriv = apply_derivate_to( param0, Umtx );
std::cout << "parameter num: " << param0.size() << ", gradient num: " << Umtx_deriv.size() << std::endl;

int deriv_idx_test = 26;

Matrix mtx0 = Umtx.copy();
Matrix mtx_delta = Umtx.copy();

apply_to(param0, mtx0 );

Matrix_real param_delta =  param0.copy();
param_delta[deriv_idx_test] += 1e-8;
apply_to(param_delta, mtx_delta );
for (int idx=0; idx<Umtx.size(); idx++ ) {
    mtx_delta[idx].real = (mtx_delta[idx].real-mtx0[idx].real)/1e-8;
    mtx_delta[idx].imag = (mtx_delta[idx].imag-mtx0[idx].imag)/1e-8;
}

mtx_delta.print_matrix();
Umtx_deriv[deriv_idx_test].print_matrix();
*/



#if BLAS==0 // undefined BLAS
    omp_set_num_threads(num_threads);
#elif BLAS==1 //MKL
    MKL_Set_Num_Threads(num_threads);
#elif BLAS==2 //OpenBLAS
    openblas_set_num_threads(num_threads);
#endif

}


/**
@brief ??????????????
*/
Gates_block* 
N_Qubit_Decomposition_adaptive::optimize_imported_gate_structure(Matrix_real& optimized_parameters_mtx_loc) {


    Gates_block* gate_structure_loc = gate_structure->clone();

    //measure the time for the decompositin
    tbb::tick_count start_time_loc = tbb::tick_count::now();


    // solve the optimization problem
    N_Qubit_Decomposition_custom cDecomp_custom;
    // solve the optimization problem in isolated optimization process
    cDecomp_custom = N_Qubit_Decomposition_custom( Umtx.copy(), qbit_num, false, initial_guess);
    cDecomp_custom.set_custom_gate_structure( gate_structure_loc );
    cDecomp_custom.set_optimized_parameters( optimized_parameters_mtx_loc.get_data(), optimized_parameters_mtx_loc.size() );
    cDecomp_custom.set_optimization_blocks( gate_structure_loc->get_gate_num() );
    cDecomp_custom.set_max_iteration( max_iterations );
    cDecomp_custom.set_verbose(false);
    cDecomp_custom.set_iteration_loops( iteration_loops );
    cDecomp_custom.set_optimization_tolerance( optimization_tolerance );  
    cDecomp_custom.start_decomposition(true);
    //cDecomp_custom.list_gates(0);

    tbb::tick_count end_time_loc = tbb::tick_count::now();

    current_minimum = cDecomp_custom.get_current_minimum();
    optimized_parameters_mtx_loc = cDecomp_custom.get_optimized_parameters();



    if ( cDecomp_custom.get_current_minimum() < optimization_tolerance ) {
        std::cout << "Optimization problem solved with " << gate_structure_loc->get_gate_num() << " decomposing layers in " << (end_time_loc-start_time_loc).seconds() << " seconds." << std::endl;
    }   
    else {
        std::cout << "Optimization problem converged to " << cDecomp_custom.get_current_minimum() << " with " <<  gate_structure_loc->get_gate_num() << " decomposing layers in "   << (end_time_loc-start_time_loc).seconds() << " seconds." << std::endl;
    }

    if (current_minimum > optimization_tolerance) {
        std::cout << "Decomposition did not reached prescribed high numerical precision." << std::endl;        
        optimization_tolerance = 1.5*current_minimum < 1e-2 ? 1.5*current_minimum : 1e-2;
    }

    std::cout << "Continue with the compression of gate structure consisting of " << gate_structure_loc->get_gate_num() << " decomposing layers." << std::endl;

    return gate_structure_loc;



}

/**
@brief ??????????????
*/
Gates_block* 
N_Qubit_Decomposition_adaptive::determine_initial_gate_structure(Matrix_real& optimized_parameters_mtx_loc) {


    // strages to store the optimized minimums in case of different cirquit depths
    std::vector<double> minimum_vec;
    std::vector<Gates_block*> gate_structure_vec;
    std::vector<Matrix_real> optimized_parameters_vec;

    int level = level_limit_min;
    while ( current_minimum > optimization_tolerance && level <= level_limit) {

        // reset optimized parameters
        optimized_parameters_mtx_loc = Matrix_real(0,0);


        // create gate structure to be optimized
        Gates_block* gate_structure_loc = new Gates_block(qbit_num);
        for (int idx=0; idx<level; idx++) {

            // create the new decomposing layer
            Gates_block* layer = construct_gate_layer(0,0);
            gate_structure_loc->combine( layer );
        }
           
        // add finalyzing layer to the top of the gate structure
        add_finalyzing_layer( gate_structure_loc );

        //measure the time for the decompositin
        tbb::tick_count start_time_loc = tbb::tick_count::now();


        // solve the optimization problem
        N_Qubit_Decomposition_custom cDecomp_custom;
        // solve the optimization problem in isolated optimization process
        cDecomp_custom = N_Qubit_Decomposition_custom( Umtx.copy(), qbit_num, false, initial_guess);
        cDecomp_custom.set_custom_gate_structure( gate_structure_loc );
        cDecomp_custom.set_optimization_blocks( gate_structure_loc->get_gate_num() );
        cDecomp_custom.set_max_iteration( max_iterations );
        cDecomp_custom.set_verbose(false);
        cDecomp_custom.set_iteration_loops( iteration_loops );
        cDecomp_custom.set_optimization_tolerance( optimization_tolerance );  
        cDecomp_custom.start_decomposition(true);
        //cDecomp_custom.list_gates(0);

        tbb::tick_count end_time_loc = tbb::tick_count::now();

        minimum_vec.push_back(cDecomp_custom.get_current_minimum());
        gate_structure_vec.push_back(gate_structure_loc);
        optimized_parameters_vec.push_back(cDecomp_custom.get_optimized_parameters());



        if ( cDecomp_custom.get_current_minimum() < optimization_tolerance ) {
            std::cout << "Optimization problem solved with " << gate_structure_loc->get_gate_num() << " decomposing layers in " << (end_time_loc-start_time_loc).seconds() << " seconds." << std::endl;
            //cDecomp_custom.list_gates(0);
            break;
        }   
        else {
            std::cout << "Optimization problem converged to " << cDecomp_custom.get_current_minimum() << " with " <<  gate_structure_loc->get_gate_num() << " decomposing layers in "   << (end_time_loc-start_time_loc).seconds() << " seconds." << std::endl;
        }

        level++;
    }

//exit(-1);

    // find the best decomposition
    int idx_min = 0;
    double current_minimum = minimum_vec[0];
    for (int idx=1; idx<minimum_vec.size(); idx++) {
        if( current_minimum > minimum_vec[idx] ) {
            idx_min = idx;
            current_minimum = minimum_vec[idx];
        }
    }
     
    Gates_block* gate_structure_loc = gate_structure_vec[idx_min];
    optimized_parameters_mtx_loc = optimized_parameters_vec[idx_min];

    // release unnecesarry data
    for (int idx=0; idx<minimum_vec.size(); idx++) {
        if( idx == idx_min ) {
            continue;
        }
        delete( gate_structure_vec[idx] );
    }    
    minimum_vec.clear();
    gate_structure_vec.clear();
    optimized_parameters_vec.clear();
    

    if (current_minimum > optimization_tolerance) {
        std::cout << "Decomposition did not reached prescribed high numerical precision." << std::endl;        
        optimization_tolerance = 1.5*current_minimum < 1e-2 ? 1.5*current_minimum : 1e-2;
    }

    std::cout << "Continue with the compression of gate structure consisting of " << gate_structure_loc->get_gate_num() << " decomposing layers." << std::endl;

    return gate_structure_loc;

}



/**
@brief ???????????????
*/
Gates_block*
N_Qubit_Decomposition_adaptive::compress_gate_structure( Gates_block* gate_structure ) {

    int layer_num_max = 10;
    int layer_num_orig = gate_structure->get_gate_num()-1;

    // create a list of layers to be tested for removal.
    std::vector<int> layers_to_remove;
    layers_to_remove.reserve(layer_num_orig);
    for (int idx=0; idx<layer_num_orig; idx++ ) {
        layers_to_remove.push_back(idx+1);
    }   

    while ( layers_to_remove.size() > layer_num_max ) {
        int remove_idx = rand() % layers_to_remove.size();
        layers_to_remove.erase( layers_to_remove.begin() + remove_idx );
    }


    int panelties_num = layer_num_max < layer_num_orig ? layer_num_max : layer_num_orig;

    // preallocate panelties associated with the number of remaining two-qubit controlled gates
    matrix_base<int> panelties(1,panelties_num);
    std::vector<Gates_block*> gate_structures_vec(panelties_num, NULL);
    std::vector<Matrix_real> optimized_parameters_vec(panelties_num, Matrix_real(0,0));
    std::vector<double> current_minimum_vec(panelties_num, 0.0);



    tbb::parallel_for( 0, panelties_num, 1, [&](int idx_to_remove) {
    //for (int idx_to_remove=0; idx_to_remove<layer_num_orig; idx_to_remove++) {

        double current_minimum_loc = 0.0;

        Matrix_real optimized_parameters_loc = optimized_parameters_mtx.copy();
        Gates_block* gate_structure_reduced = compress_gate_structure( gate_structure, layers_to_remove[idx_to_remove], optimized_parameters_loc,  current_minimum_loc  );
 
        // remove further adaptive gates if possible
        Gates_block* gate_structure_tmp;
        if ( gate_structure_reduced->get_gate_num() ==  gate_structure->get_gate_num() ) {
            gate_structure_tmp = gate_structure_reduced->clone();
        }
        else {
            gate_structure_tmp = remove_trivial_gates( gate_structure_reduced, optimized_parameters_loc, current_minimum_loc );
        }
        panelties[idx_to_remove] = get_panelty(gate_structure_tmp, optimized_parameters_loc);
        gate_structures_vec[idx_to_remove] = gate_structure_tmp;
        optimized_parameters_vec[idx_to_remove] = optimized_parameters_loc;
        current_minimum_vec[idx_to_remove] = current_minimum_loc;

        

        delete(gate_structure_reduced);


    //}
    });    

//panelties.print_matrix();

    // determine the reduction with the lowest penalty
    int panelty_min = panelties[0];
    int idx_min = 0;
    for (int idx=0; idx<panelties.size(); idx++) {
        if ( panelty_min > panelties[idx] ) {
            panelty_min = panelties[idx];
            idx_min = idx;
        }
        else if ( panelty_min == panelties[idx] ) {

            // randomly choose the solution between identical penalties
            if ( (rand() % 2) == 1 ) {
                panelty_min = panelties[idx];
                idx_min = idx;
            }

        }
    }

    for (int idx=0; idx<panelties.size(); idx++) {
        if (idx==idx_min) continue;
        if ( gate_structures_vec[idx] == gate_structure) continue;
        delete( gate_structures_vec[idx] );
        gate_structures_vec[idx] = NULL;
    }


    // release the reduced gate structure, keep only the most efficient one
    gate_structure = gate_structures_vec[idx_min];
    optimized_parameters_mtx = optimized_parameters_vec[idx_min];
    current_minimum =  current_minimum_vec[idx_min];

    int layer_num = gate_structure->get_gate_num();

    if ( layer_num < layer_num_orig+1 ) {
        std::cout << "gate structure reduced from " << layer_num_orig+1 << " to " << layer_num << " decomposing layers" << std::endl;
    }
    else {
        std::cout << "gate structure kept at " << layer_num << " layers" << std::endl;
    }


    return gate_structure;



}

/**
@brief ???????????????
*/
Gates_block* 
N_Qubit_Decomposition_adaptive::compress_gate_structure( Gates_block* gate_structure, int layer_idx, Matrix_real& optimized_parameters, double& currnt_minimum_loc ) {

    // create reduced gate structure without layer indexed by layer_idx
    Gates_block* gate_structure_reduced = gate_structure->clone();
    gate_structure_reduced->release_gate( layer_idx );
        
    Matrix_real&& parameters_reduced = create_reduced_parameters( gate_structure_reduced, optimized_parameters, layer_idx );

    N_Qubit_Decomposition_custom cDecomp_custom;
       
    // solve the optimization problem in isolated optimization process
    cDecomp_custom = N_Qubit_Decomposition_custom( Umtx.copy(), qbit_num, false, initial_guess);
    cDecomp_custom.set_custom_gate_structure( gate_structure_reduced );
    cDecomp_custom.set_optimized_parameters( parameters_reduced.get_data(), parameters_reduced.size() );
    cDecomp_custom.set_verbose(false);
    cDecomp_custom.set_max_iteration( max_iterations );
    cDecomp_custom.set_iteration_loops( iteration_loops );
    cDecomp_custom.set_optimization_blocks( gate_structure_reduced->get_gate_num() ) ;
    cDecomp_custom.set_optimization_tolerance( optimization_tolerance );
    cDecomp_custom.start_decomposition(true);
    double current_minimum_loc = cDecomp_custom.get_current_minimum();

    if ( current_minimum_loc < optimization_tolerance ) {
        //cDecomp_custom.list_gates(0);
        optimized_parameters = cDecomp_custom.get_optimized_parameters();
        return gate_structure_reduced;
    }


    return gate_structure->clone();

}


/**
@brief ???????????????
*/
int 
N_Qubit_Decomposition_adaptive::get_panelty( Gates_block* gate_structure, Matrix_real& optimized_parameters ) {


    int panelty = 0;

    // iterate over the elements of tha parameter array
    int parameter_idx = 0;
    for ( int idx=0; idx<gate_structure->get_gate_num(); idx++) {

        double parameter = optimized_parameters[parameter_idx];
 
        
        if ( std::abs(std::sin(parameter/2)) < 0.99 && std::abs(std::cos(parameter/2)) < 1e-2 ) {
            // Condition of pure CNOT gate
            panelty += 2;
        }
        else if ( std::abs(std::sin(parameter/2)) < 1e-2 && std::abs(1-std::cos(parameter/2)) < 1e-2 ) {
            // Condition of pure Identity gate
            panelty++;
        }
        else {
            // Condition of controlled rotation gate
            panelty += 4;
        }

        Gate* gate = gate_structure->get_gate( idx );
        parameter_idx += gate->get_parameter_num();

    }

    return panelty;


}


/**
@brief ???????????????
*/
Gates_block* 
N_Qubit_Decomposition_adaptive::replace_trivial_CRY_gates( Gates_block* gate_structure, Matrix_real& optimized_parameters ) {


    Gates_block* gate_structure_ret = new Gates_block(qbit_num);

    int layer_num = gate_structure->get_gate_num();

    int parameter_idx = 0;
    for (int idx=0; idx<layer_num; idx++ ) {

        Gate* gate = gate_structure->get_gate(idx);

        if ( gate->get_type() != BLOCK_OPERATION ) {
            std::cout << "N_Qubit_Decomposition_adaptive::replace_trivial_adaptive_gates: Only block gates are accepted in this conversion." << std::endl;
            exit(-1);
        }

        Gates_block* block_op = static_cast<Gates_block*>(gate);
        //int param_num = gate->get_parameter_num();


        if (  true ) {//gate_structure->contains_adaptive_gate(idx) ) {

                Gates_block* layer = block_op->clone();

                for ( int jdx=0; jdx<layer->get_gate_num(); jdx++ ) {

                    Gate* gate_tmp = layer->get_gate(jdx);
                    int param_num = gate_tmp->get_parameter_num();


                    double parameter = optimized_parameters[parameter_idx];
                    parameter = activation_function(parameter, 1);//limit_max);

//std::cout << param[0] << " " << (gate_tmp->get_type() == ADAPTIVE_OPERATION) << " "  << std::abs(std::sin(param[0])) << " "  << 1+std::cos(param[0]) << std::endl;

                    if ( gate_tmp->get_type() == ADAPTIVE_OPERATION &&  std::abs(std::sin(parameter/2)) > 0.99 && std::abs(std::cos(parameter/2)) < 1e-2 ) {

                        // convert to CZ gate
                        int target_qbit = gate_tmp->get_target_qbit();
                        int control_qbit = gate_tmp->get_control_qbit();
                        layer->release_gate( jdx );

                        RX*   rx_gate_1   = new RX(qbit_num, target_qbit);
                        CZ*   cz_gate     = new CZ(qbit_num, target_qbit, control_qbit);
                        RX*   rx_gate_2   = new RX(qbit_num, target_qbit);
                        RZ*   rz_gate     = new RZ(qbit_num, control_qbit);

                        Gates_block* czr_gate = new Gates_block(qbit_num);
                        czr_gate->add_gate(rx_gate_1);
                        czr_gate->add_gate(cz_gate);
                        czr_gate->add_gate(rx_gate_2);
                        czr_gate->add_gate(rz_gate);

                        layer->insert_gate( (Gate*)czr_gate, jdx);

                        Matrix_real parameters_new(1, optimized_parameters.size()+2);
                        memcpy(parameters_new.get_data(), optimized_parameters.get_data(), parameter_idx*sizeof(double));
                        memcpy(parameters_new.get_data()+parameter_idx+3, optimized_parameters.get_data()+parameter_idx+1, (optimized_parameters.size()-parameter_idx-1)*sizeof(double));
                        optimized_parameters = parameters_new;
                        if ( std::sin(parameter/2) < 0 ) {
                            optimized_parameters[parameter_idx] = -M_PI/2; // rz parameter
                        }
                        else{
                            optimized_parameters[parameter_idx] = M_PI/2; // rz parameter
                        }
                        optimized_parameters[parameter_idx+1] = M_PI/2; // rx_2 parameter
                        optimized_parameters[parameter_idx+2] = -M_PI/2; // rx_1 parameter
                        parameter_idx += 3;


                    }
                    else if ( gate_tmp->get_type() == ADAPTIVE_OPERATION &&  std::abs(std::sin(parameter/2)) < 1e-2 && std::abs(1-std::cos(parameter/2)) < 1e-2  ) {
                        // release trivial gate  

                        layer->release_gate( jdx );
                        jdx--;
                        Matrix_real parameters_new(1, optimized_parameters.size()-1);
                        memcpy(parameters_new.get_data(), optimized_parameters.get_data(), parameter_idx*sizeof(double));
                        memcpy(parameters_new.get_data()+parameter_idx, optimized_parameters.get_data()+parameter_idx+1, (optimized_parameters.size()-parameter_idx-1)*sizeof(double));
                        optimized_parameters = parameters_new;


                    }
                    else if ( gate_tmp->get_type() == ADAPTIVE_OPERATION ) {
                        // controlled Z rotation decomposed into 2 CNOT gates

                        int target_qbit = gate_tmp->get_target_qbit();
                        int control_qbit = gate_tmp->get_control_qbit();
                        layer->release_gate( jdx );

                        RY*   ry_gate_1   = new RY(qbit_num, target_qbit);
                        CNOT* cnot_gate_1 = new CNOT(qbit_num, target_qbit, control_qbit);
                        RY*   ry_gate_2   = new RY(qbit_num, target_qbit);
                        CNOT* cnot_gate_2 = new CNOT(qbit_num, target_qbit, control_qbit);

                        Gates_block* czr_gate = new Gates_block(qbit_num);
                        czr_gate->add_gate(ry_gate_1);
                        czr_gate->add_gate(cnot_gate_1);
                        czr_gate->add_gate(ry_gate_2);
                        czr_gate->add_gate(cnot_gate_2);

                        layer->insert_gate( (Gate*)czr_gate, jdx);

                        Matrix_real parameters_new(1, optimized_parameters.size()+1);
                        memcpy(parameters_new.get_data(), optimized_parameters.get_data(), parameter_idx*sizeof(double));
                        memcpy(parameters_new.get_data()+parameter_idx+2, optimized_parameters.get_data()+parameter_idx+1, (optimized_parameters.size()-parameter_idx-1)*sizeof(double));
                        optimized_parameters = parameters_new;
                        optimized_parameters[parameter_idx] = -parameter/2; // ry_2 parameter
                        optimized_parameters[parameter_idx+1] = parameter/2; // ry_1 parameter
                        parameter_idx += 2;


                    }
                    else {
                        parameter_idx  += param_num;

                    }



                }

                gate_structure_ret->add_gate_to_end((Gate*)layer);

                            

        }

    }

    return gate_structure_ret;


}

/**
@brief ???????????????
*/
Gates_block*
N_Qubit_Decomposition_adaptive::remove_trivial_gates( Gates_block* gate_structure, Matrix_real& optimized_parameters, double& currnt_minimum_loc ) {


    int layer_num = gate_structure->get_gate_num();
    int parameter_idx = 0;

    Matrix_real&& optimized_parameters_loc = optimized_parameters.copy();

    Gates_block* gate_structure_loc = gate_structure->clone();

    int idx = 0;
    while (idx<layer_num ) {

        Gates_block* layer = static_cast<Gates_block*>( gate_structure_loc->get_gate(idx) );

        for ( int jdx=0; jdx<layer->get_gate_num(); jdx++ ) {

            Gate* gate_tmp = layer->get_gate(jdx);
            int param_num = gate_tmp->get_parameter_num();


            double parameter = optimized_parameters[parameter_idx];
            parameter = activation_function(parameter, 1);//limit_max);


            if ( gate_tmp->get_type() == ADAPTIVE_OPERATION &&  std::abs(std::sin(parameter/2)) < 1e-2 && std::abs(1-std::cos(parameter/2)) < 1e-2  ) {

               
                // remove gate from the structure
                Gates_block* gate_structure_tmp = compress_gate_structure( gate_structure_loc, idx, optimized_parameters_loc, currnt_minimum_loc );

                optimized_parameters = optimized_parameters_loc;
                delete( gate_structure_loc );
                gate_structure_loc = gate_structure_tmp;
                layer_num = gate_structure_loc->get_gate_num();   
                break;            

          
            }


            parameter_idx += param_num;
            

        }



        idx++;


    }
//std::cout << "N_Qubit_Decomposition_adaptive::remove_trivial_gates :" << gate_structure->get_gate_num() << " reduced to " << gate_structure_loc->get_gate_num() << std::endl;
    return gate_structure_loc;




}


/**
@brief ???????????????
*/
Matrix_real 
N_Qubit_Decomposition_adaptive::create_reduced_parameters( Gates_block* gate_structure, Matrix_real& optimized_parameters, int layer_idx ) {


    // determine the index of the parameter that is about to delete
    int parameter_idx = 0;
    for ( int idx=0; idx<layer_idx; idx++) {
        Gate* gate = gate_structure->get_gate( idx );
        parameter_idx += gate->get_parameter_num();
    }


    Gate* gate = gate_structure->get_gate( layer_idx );
    int param_num_removed = gate->get_parameter_num();

    Matrix_real reduced_parameters(1, optimized_parameters.size() - param_num_removed );
    memcpy( reduced_parameters.get_data(), optimized_parameters.get_data(), (parameter_idx)*sizeof(double));
    memcpy( reduced_parameters.get_data()+parameter_idx, optimized_parameters.get_data()+parameter_idx+param_num_removed, (optimized_parameters.size()-parameter_idx-param_num_removed) *sizeof(double));


    return reduced_parameters;
}












/**
@brief Call to add further layer to the gate structure used in the subdecomposition.
*/
Gates_block* 
N_Qubit_Decomposition_adaptive::construct_gate_layer( const int& _target_qbit, const int& _control_qbit) {


    // creating block of gates
    Gates_block* block = new Gates_block( qbit_num );

    int layer_num = (qbit_num*(qbit_num-1))/2;
    std::vector<Gates_block* > layers;


    if ( topology.size() > 0 ) {
        for ( std::vector<matrix_base<int>>::iterator it=topology.begin(); it!=topology.end(); it++) {

            if ( it->size() != 2 ) {
                std::cout << "The connectivity data should contains two qubits" << std::endl;
                it->print_matrix();
                exit(-1);
            }

            int control_qbit_loc = (*it)[0];
            int target_qbit_loc = (*it)[1];

            if ( control_qbit_loc >= qbit_num || target_qbit_loc >= qbit_num ) {
                std::cout << "Label of control/target qubit should be less than the number of qubits in the register." << std::endl;
                exit(-1);            
            }

            Gates_block* layer = new Gates_block( qbit_num );

            bool Theta = true;
            bool Phi = true;
            bool Lambda = true;
            layer->add_u3(target_qbit_loc, Theta, Phi, Lambda);
            layer->add_u3(control_qbit_loc, Theta, Phi, Lambda); 
            layer->add_adaptive(target_qbit_loc, control_qbit_loc);

            layers.push_back(layer);


        }
    }
    else {  
    
        // sequ
        for (int target_qbit_loc = 0; target_qbit_loc<qbit_num; target_qbit_loc++) {
            for (int control_qbit_loc = target_qbit_loc+1; control_qbit_loc<qbit_num; control_qbit_loc++) {

                Gates_block* layer = new Gates_block( qbit_num );

                bool Theta = true;
                bool Phi = true;
                bool Lambda = true;
                layer->add_u3(target_qbit_loc, Theta, Phi, Lambda);
                layer->add_u3(control_qbit_loc, Theta, Phi, Lambda); 
                layer->add_adaptive(target_qbit_loc, control_qbit_loc);

                layers.push_back(layer);
            }
        }

    }

/*
    for (int idx=0; idx<layers.size(); idx++) {
        Gates_block* layer = (Gates_block*)layers[idx];
        block->add_gate( layers[idx] );

    }

*/
    while (layers.size()>0) { 
        int idx = std::rand() % layers.size();
        Gates_block* layer = (Gates_block*)layers[idx];
        block->add_gate( layers[idx] );
        layers.erase( layers.begin() + idx );
    }


    return block;


}




/**
@brief ??????????????????
*/
void 
N_Qubit_Decomposition_adaptive::add_finalyzing_layer( Gates_block* gate_structure ) {


    // creating block of gates
    Gates_block* block = new Gates_block( qbit_num );
/*
    block->add_un();
    block->add_ry(qbit_num-1);
*/
    for (int idx=0; idx<qbit_num; idx++) {
            bool Theta = true;
            bool Phi = false;
            bool Lambda = true;
             block->add_u3(idx, Theta, Phi, Lambda);
//        block->add_ry(idx);
    }

    // adding the opeartion block to the gates
    gate_structure->add_gate( block );

}



/**
@brief Call to set custom layers to the gate structure that are intended to be used in the subdecomposition.
@param gate_structure An <int, Gates_block*> map containing the gate structure used in the individual subdecomposition (default is used, if a gate structure for specific subdecomposition is missing).
*/
void 
N_Qubit_Decomposition_adaptive::set_adaptive_gate_structure( Gates_block* gate_structure_in ) {

    gate_structure = gate_structure_in->clone();

}






