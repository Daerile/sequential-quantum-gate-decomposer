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
#include "Random_Unitary.h"

#include "X.h"

#include <time.h>
#include <stdlib.h>


extern "C" {

/**
 * \brief ???????????
 * 
 */
typedef struct {
  float real;
  float imag;
} Complex8;



/**
@brief ????????????
@return ??????????
*/
int calcqgdKernelDFE(size_t dim, DFEgate_kernel_type* gates, int gatesNum);

/**
@brief ????????????
@return ??????????
*/
int load2LMEM( Complex8* data, size_t dim );

/**
@brief ????????????
@return ??????????
*/
void releive_DFE();

/**
@brief ????????????
@return ??????????
*/
int initialize_DFE();

/**
 * \brief ???????????
 * 
 */
int downloadFromLMEM( Complex8** data, size_t dim );

}


/**
@brief ????????????
@return ??????????
*/
void uploadMatrix2DFE( Matrix& input ) {

    // first convert the input to float32
    matrix_base<Complex8> input32( input.rows, input.cols ); // number of columns needed to be made twice due to complex -> real tranformation

    size_t element_num = input.size();
    QGD_Complex16* input_data = input.get_data();
    Complex8* input32_data = input32.get_data();
    for ( size_t idx=0; idx<element_num; idx++) {
        input32_data[idx].real = (float)(input_data[idx].real);
        input32_data[idx].imag = (float)(input_data[idx].imag);
    }
    
std::cout << "size in bytes of uploading: " << element_num*sizeof(float) << std::endl;    

    // load the data to LMEM
    load2LMEM( input32_data, input.rows );

}



/**
@brief ????????????
@return ??????????
*/
void DownloadMatrixFromDFE( std::vector<Matrix>& output_vec ) {

    // first convert the input to float32
    size_t element_num = output_vec[0].size();

    std::vector<matrix_base<Complex8>> output32_vec;
    for( int idx=0; idx<4; idx++) {
        output32_vec.push_back(matrix_base<Complex8>( output_vec[0].rows, output_vec[0].cols ));
    }
    
    Complex8* output32_data[4];
    for( int idx=0; idx<4; idx++) {
        output32_data[idx] = output32_vec[idx].get_data();
    }    
    

    // load the data to LMEM
    downloadFromLMEM( output32_data, output_vec[0].rows );

    for( int idx=0; idx<4; idx++) {
	QGD_Complex16* output_data = output_vec[idx].get_data();    
	Complex8* output32_data_loc = output32_data[idx]; 	
        for ( size_t jdx=0; jdx<element_num; jdx++) {
            output_data[jdx].real = (double)(output32_data_loc[jdx].real);
            output_data[jdx].imag = (double)(output32_data_loc[jdx].imag);
        }
    }



}

/**
@brief Method to create random initial parameters for the optimization
@return 
*/
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
N_Qubit_Decomposition_adaptive::N_Qubit_Decomposition_adaptive( Matrix Umtx_in, int qbit_num_in, int level_limit_in, int level_limit_min_in ) : N_Qubit_Decomposition_Base(Umtx_in, qbit_num_in, false, RANDOM) {


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
N_Qubit_Decomposition_adaptive::N_Qubit_Decomposition_adaptive( Matrix Umtx_in, int qbit_num_in, int level_limit_in, int level_limit_min_in, std::vector<matrix_base<int>> topology_in ) : N_Qubit_Decomposition_Base(Umtx_in, qbit_num_in, false, RANDOM) {


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


    //The stringstream input to store the output messages.
    std::stringstream sstream;
    sstream << "***************************************************************" << std::endl;
    sstream << "Starting to disentangle " << qbit_num << "-qubit matrix" << std::endl;
    sstream << "***************************************************************" << std::endl << std::endl << std::endl;

    print(sstream, 1);   


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
        std::stringstream sstream;
	sstream << "please increase level limit" << std::endl;
        print(sstream, 0);	
        exit(-1);
    }

////////////////////////////////
    int num_of_qbits_loc = qbit_num;
    int dim_loc = 1 << num_of_qbits_loc;//1024*4;
/*
    Random_Unitary ru(dim_loc);
    Matrix test_Umtx = ru.Construct_Unitary_Matrix() ;
*/
//    Matrix test_Umtx = create_identity( dim_loc );


/*
    Matrix test_Umtx(dim_loc,dim_loc);
    for (size_t idx=0; idx<test_Umtx.size(); idx++) {
        test_Umtx[idx].real = (2*double(rand())/double(RAND_MAX)-1)*0.5;
        test_Umtx[idx].imag = (2*double(rand())/double(RAND_MAX)-1)*0.5;
    }
*/

    Matrix test_Umtx = Umtx.copy();
    uploadMatrix2DFE( test_Umtx );
/*
    std::vector<Matrix_real> parameters_vec;
    std::vector<int> target_qbit_vec;
    std::vector<int> control_qbit_vec;

    tbb::tick_count t0_DFE = tbb::tick_count::now(); /////////////////////////////
    
    int gatesNum = 6;
    for (int idx=0; idx<gatesNum; idx=idx+3 ) {

        int target_qbit_loc = rand() % num_of_qbits_loc;


        int control_qbit_loc = target_qbit_loc;

        while( control_qbit_loc == target_qbit_loc ) {
            control_qbit_loc = rand() % num_of_qbits_loc;
        }

        target_qbit_vec.push_back( target_qbit_loc );
        target_qbit_vec.push_back( control_qbit_loc );
        target_qbit_vec.push_back( target_qbit_loc );

        control_qbit_vec.push_back( -1 );
        control_qbit_vec.push_back( -1 );
        control_qbit_vec.push_back( control_qbit_loc );



        Matrix_real parameters(3,1);
        parameters[0] = (2*double(rand())/double(RAND_MAX)-1)*4*M_PI;
        parameters[1] = (2*double(rand())/double(RAND_MAX)-1)*2*M_PI;
        parameters[2] = (2*double(rand())/double(RAND_MAX)-1)*2*M_PI;

        parameters_vec.push_back( parameters );

        Matrix_real parameters2(3,1);
        parameters2[0] = (2*double(rand())/double(RAND_MAX)-1)*4*M_PI;
        parameters2[1] = (2*double(rand())/double(RAND_MAX)-1)*2*M_PI;
        parameters2[2] = (2*double(rand())/double(RAND_MAX)-1)*2*M_PI;

        parameters_vec.push_back( parameters2 );


	// CNOT gate 
        //Matrix_real parameters3(3,1);
        //parameters3[0] = M_PI;
        //parameters3[1] = 0.0;
        //parameters3[2] = M_PI;


	// CRY gate 
        Matrix_real parameters3(3,1);
        parameters3[0] = (2*double(rand())/double(RAND_MAX)-1)*4*M_PI;
        parameters3[1] = 0.0;
        parameters3[2] = 0.0;
        parameters_vec.push_back( parameters3 );

    }


    
    
    DFEgate_kernel_type* gates_loc = new DFEgate_kernel_type[gatesNum];
    for (int idx=0; idx<gatesNum; idx=idx+3 ) {

        Matrix_real parameters1 = parameters_vec[idx];
        gates_loc[idx].target_qbit = target_qbit_vec[idx];
        gates_loc[idx].control_qbit = -1;
        gates_loc[idx].gate_type = U3_OPERATION;
        gates_loc[idx].ThetaOver2 = (int32_t)(std::fmod( parameters1[0]/2, 2*M_PI)*(1<<25));
        gates_loc[idx].Phi = (int32_t)(std::fmod( parameters1[1], 2*M_PI)*(1<<25));
        gates_loc[idx].Lambda = (int32_t)(std::fmod( parameters1[2], 2*M_PI)*(1<<25)); 

        Matrix_real parameters2 = parameters_vec[idx+1];
        gates_loc[idx+1].target_qbit = target_qbit_vec[idx+1];
        gates_loc[idx+1].control_qbit = -1;
        gates_loc[idx+1].gate_type = U3_OPERATION;
        gates_loc[idx+1].ThetaOver2 = (int32_t)(std::fmod( parameters2[0]/2, 2*M_PI)*(1<<25));
        gates_loc[idx+1].Phi = (int32_t)(std::fmod( parameters2[1], 2*M_PI)*(1<<25));
        gates_loc[idx+1].Lambda = (int32_t)(std::fmod( parameters2[2], 2*M_PI)*(1<<25)); 

        Matrix_real parameters3 = parameters_vec[idx+2];
        gates_loc[idx+2].target_qbit = target_qbit_vec[idx+2];
        gates_loc[idx+2].control_qbit = control_qbit_vec[idx+2];
        gates_loc[idx+2].gate_type = CNOT_OPERATION;
        gates_loc[idx+2].ThetaOver2 = (int32_t)(std::fmod( parameters3[0]/2, 2*M_PI)*(1<<25));
        gates_loc[idx+2].Phi = (int32_t)(std::fmod( parameters3[1], 2*M_PI)*(1<<25));
        gates_loc[idx+2].Lambda = (int32_t)(std::fmod( parameters3[2], 2*M_PI)*(1<<25)); 

        // adjust parameters to calculate the derivate
        if (idx==0) {
            // set the most significant bit on target_qbit to indicate derivate
            gates_loc[idx+2].ThetaOver2 = (int32_t)(std::fmod( parameters3[0]/2 + M_PI/2, 2*M_PI)*(1<<25));
            gates_loc[idx+2].target_qbit = gates_loc[idx+2].target_qbit + (1 << 7);
        }


    }
    

    calcqgdKernelDFE( dim_loc, gates_loc, gatesNum );   
    delete[] gates_loc;
    tbb::tick_count t1_DFE = tbb::tick_count::now();/////////////////////////////////
    std::cout << "time elapsed DFE: " << (t1_DFE-t0_DFE).seconds() << ", expected time: " << ((double)(dim_loc*dim_loc*gatesNum/3 + 531))/350000000 + 0.001<< std::endl;

    // tranform the matrix on CPU
    
    //X X_gate(num_of_qbits_loc, target_qbit_loc);    
    tbb::tick_count t0_gate = tbb::tick_count::now();
    //X_gate.apply_to(test_Umtx);

    for (int idx=0; idx<gatesNum; idx=idx+3) {

        Matrix_real& parameters1 = parameters_vec[idx];
        U3 U3_gate(num_of_qbits_loc, target_qbit_vec[idx], true, true, true);
        U3_gate.apply_to(parameters1, test_Umtx);

        Matrix_real& parameters2 = parameters_vec[idx+1];
        U3 U3_gate2(num_of_qbits_loc, target_qbit_vec[idx+1], true, true, true);
        U3_gate2.apply_to(parameters2, test_Umtx);

        //CNOT CNOT_gate(num_of_qbits_loc, target_qbit_vec[idx+2], control_qbit_vec[idx+2]);
        //CNOT_gate.apply_to(test_Umtx);

        Matrix_real& parameters3 = parameters_vec[idx+2];
        CRY CRY_gate(num_of_qbits_loc, target_qbit_vec[idx+2], control_qbit_vec[idx+2]);
        if (idx==0) {
            std::vector<Matrix> res = CRY_gate.apply_derivate_to(parameters3, test_Umtx);
            test_Umtx = res[0];
        } 
        else {
            CRY_gate.apply_to(parameters3, test_Umtx);
        }

    }
    
    tbb::tick_count t1_gate = tbb::tick_count::now();
    std::cout << "time elapsed CPU: " << (t1_gate-t0_gate).seconds() << std::endl;
*/
    //releive_DFE(); 
    //return;
/*
///////////////////////////////////////////

    std::vector<Matrix> outputMtx_vec;
    for ( int idx=0; idx<4; idx++) {
	    outputMtx_vec.push_back( Matrix( test_Umtx.rows, test_Umtx.cols ) );
	    memset( outputMtx_vec[idx].get_data(), 0.0, outputMtx_vec[idx].size()*sizeof(QGD_Complex16) );
    }

    DownloadMatrixFromDFE( outputMtx_vec );

    for ( int kdx=0; kdx<4; kdx++) {
        Matrix& outputMtx = outputMtx_vec[kdx];
        for (size_t idx=0; idx<test_Umtx.size(); idx++) {
            double diff = (test_Umtx[idx].real-outputMtx[idx].real)*(test_Umtx[idx].real-outputMtx[idx].real*0.5) + (test_Umtx[idx].imag-outputMtx[idx].imag)*(test_Umtx[idx].imag-outputMtx[idx].imag*0.5);
            if ( diff > 1e-5 ) {
                std::cout << "input and output matrices differs at index " << idx << ":(" << test_Umtx[idx].real << "+ i*" << test_Umtx[idx].imag<< ") and :(" << outputMtx[idx].real << "+ i*" << outputMtx[idx].imag<<")" << std::endl;
            }
        }
    }

    double trace = 0.0;
    for( int idx=0; idx<test_Umtx.rows; idx++) {
        trace += test_Umtx[idx+test_Umtx.rows*idx].real;
    }
std::cout << "trace CPU: " << trace << std::endl;
*/
//outputMtx.print_matrix();
//test_Umtx.print_matrix();





////////////////////////////////////////////


///////////////////////////////

    double optimization_tolerance_orig = optimization_tolerance;
    optimization_tolerance = 1e-4;

    Gates_block* gate_structure_loc = NULL;
    if ( gate_structure != NULL ) {
        std::stringstream sstream;
	sstream << "Using imported gate structure for the decomposition." << std::endl;
        print(sstream, 1);	
	        
        gate_structure_loc = optimize_imported_gate_structure(optimized_parameters_mtx);
    }
    else {
        std::stringstream sstream;
	sstream << "Construct initial gate structure for the decomposition." << std::endl;
        print(sstream, 1);
        gate_structure_loc = determine_initial_gate_structure(optimized_parameters_mtx);
    }
releive_DFE();
return;
    sstream.str("");
    sstream << std::endl;
    sstream << std::endl;
    sstream << "**************************************************************" << std::endl;
    sstream << "***************** Compressing Gate structure *****************" << std::endl;
    sstream << "**************************************************************" << std::endl;
    print(sstream, 1);	    	
    

    int iter = 0;
    int uncompressed_iter_num = 0;
    while ( iter<25 || uncompressed_iter_num <= 5 ) {

        if ( current_minimum > 1e-2 ) {
             for (int idx=0; idx<optimized_parameters_mtx.size(); idx++ ) {
                 optimized_parameters_mtx[idx] = (2*double(rand())/double(RAND_MAX)-1)*2*M_PI;
             }
        }

    sstream.str("");
    sstream << "iteration " << iter+1 << ": ";
    print(sstream, 1);	

       
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

    sstream.str("");
    sstream << "**************************************************************" << std::endl;
    sstream << "************ Final tuning of the Gate structure **************" << std::endl;
    sstream << "**************************************************************" << std::endl;
    print(sstream, 1);	    	
    

    optimization_tolerance = optimization_tolerance_orig;

    // store the decomposing gate structure    
    combine( gate_structure_loc );
    optimization_block = get_gate_num();


    sstream.str("");
    sstream << optimization_problem(optimized_parameters_mtx.get_data()) << std::endl;
    print(sstream, 3);	
    	
    Gates_block* gate_structure_tmp = replace_trivial_CRY_gates( gate_structure_loc, optimized_parameters_mtx );
    Matrix_real optimized_parameters_save = optimized_parameters_mtx;

    release_gates();
    optimized_parameters_mtx = optimized_parameters_save;

    combine( gate_structure_tmp );
    delete( gate_structure_tmp );
    delete( gate_structure_loc );

    sstream.str("");
    sstream << optimization_problem(optimized_parameters_mtx.get_data()) << std::endl;
    print(sstream, 3);	
    	
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

    
    sstream.str("");
    sstream << "In the decomposition with error = " << decomposition_error << " were used " << layer_num << " gates with:" << std::endl;
      
        if ( gates_num.u3>0 ) sstream << gates_num.u3 << " U3 gates," << std::endl;
        if ( gates_num.rx>0 ) sstream << gates_num.rx << " RX gates," << std::endl;
        if ( gates_num.ry>0 ) sstream << gates_num.ry << " RY gates," << std::endl;
        if ( gates_num.rz>0 ) sstream << gates_num.rz << " RZ gates," << std::endl;
        if ( gates_num.cnot>0 ) sstream << gates_num.cnot << " CNOT gates," << std::endl;
        if ( gates_num.cz>0 ) sstream << gates_num.cz << " CZ gates," << std::endl;
        if ( gates_num.ch>0 ) sstream << gates_num.ch << " CH gates," << std::endl;
        if ( gates_num.x>0 ) sstream << gates_num.x << " X gates," << std::endl;
        if ( gates_num.sx>0 ) sstream << gates_num.sx << " SX gates," << std::endl; 
        if ( gates_num.syc>0 ) sstream << gates_num.syc << " Sycamore gates," << std::endl;   
        if ( gates_num.un>0 ) sstream << gates_num.un << " UN gates," << std::endl;
        if ( gates_num.cry>0 ) sstream << gates_num.cry << " CRY gates," << std::endl;  
        if ( gates_num.adap>0 ) sstream << gates_num.adap << " Adaptive gates," << std::endl;
    
        sstream << std::endl;
        tbb::tick_count current_time = tbb::tick_count::now();

	sstream << "--- In total " << (current_time - start_time).seconds() << " seconds elapsed during the decomposition ---" << std::endl;
    	print(sstream, 1);	    	
    	
            
               
    


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
    cDecomp_custom.set_verbose(0);
    cDecomp_custom.set_debugfile("");
    cDecomp_custom.set_iteration_loops( iteration_loops );
    cDecomp_custom.set_optimization_tolerance( optimization_tolerance );  
    cDecomp_custom.start_decomposition(true);
    //cDecomp_custom.list_gates(0);

    tbb::tick_count end_time_loc = tbb::tick_count::now();

    current_minimum = cDecomp_custom.get_current_minimum();
    optimized_parameters_mtx_loc = cDecomp_custom.get_optimized_parameters();



    if ( cDecomp_custom.get_current_minimum() < optimization_tolerance ) {
        std::stringstream sstream;
	sstream << "Optimization problem solved with " << gate_structure_loc->get_gate_num() << " decomposing layers in " << (end_time_loc-start_time_loc).seconds() << " seconds." << std::endl;
        print(sstream, 1);	
    }   
    else {
        std::stringstream sstream;
	sstream << "Optimization problem converged to " << cDecomp_custom.get_current_minimum() << " with " <<  gate_structure_loc->get_gate_num() << " decomposing layers in "   << (end_time_loc-start_time_loc).seconds() << " seconds." << std::endl;
        print(sstream, 1);       
    }

    if (current_minimum > optimization_tolerance) {
        std::stringstream sstream;
	sstream << "Decomposition did not reached prescribed high numerical precision." << std::endl; 
        print(sstream, 1);             
        optimization_tolerance = 1.5*current_minimum < 1e-2 ? 1.5*current_minimum : 1e-2;
    }

    std::stringstream sstream;
    sstream << "Continue with the compression of gate structure consisting of " << gate_structure_loc->get_gate_num() << " decomposing layers." << std::endl;
    print(sstream, 1);	
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


        N_Qubit_Decomposition_custom cDecomp_custom_random, cDecomp_custom_close_to_zero;
/*
        // try the decomposition withrandom and with close to zero initial values
        tbb::parallel_invoke(
            [&]{            
*/
                // solve the optimization problem in isolated optimization process
                cDecomp_custom_random = N_Qubit_Decomposition_custom( Umtx.copy(), qbit_num, false, RANDOM);
                cDecomp_custom_random.set_custom_gate_structure( gate_structure_loc );
                cDecomp_custom_random.set_optimization_blocks( gate_structure_loc->get_gate_num() );
                cDecomp_custom_random.set_max_iteration( max_iterations );
                cDecomp_custom_random.set_verbose(0);
                cDecomp_custom_random.set_debugfile("");
                cDecomp_custom_random.set_optimization_tolerance( optimization_tolerance );  
                cDecomp_custom_random.start_decomposition(true);
/*            },
            [&]{
                // solve the optimization problem in isolated optimization process
                cDecomp_custom_close_to_zero = N_Qubit_Decomposition_custom( Umtx.copy(), qbit_num, false, CLOSE_TO_ZERO);
                cDecomp_custom_close_to_zero.set_custom_gate_structure( gate_structure_loc );
                cDecomp_custom_close_to_zero.set_optimization_blocks( gate_structure_loc->get_gate_num() );    
                cDecomp_custom_close_to_zero.set_max_iteration( max_iterations );
                cDecomp_custom_close_to_zero.set_verbose(0);
                cDecomp_custom_close_to_zero.set_debugfile("");
                cDecomp_custom_close_to_zero.set_optimization_tolerance( optimization_tolerance );  
                cDecomp_custom_close_to_zero.start_decomposition(true);
               }
         );
*/
         tbb::tick_count end_time_loc = tbb::tick_count::now();
return NULL;
         double current_minimum_random         = cDecomp_custom_random.get_current_minimum();
         double current_minimum_close_to_zero = cDecomp_custom_close_to_zero.get_current_minimum();
         double current_minimum_loc;


         // select between the results obtained for different initial value strategy
         if ( current_minimum_random < optimization_tolerance && current_minimum_close_to_zero > optimization_tolerance ) {
             current_minimum_loc = current_minimum_random;
             optimized_parameters_mtx_loc = cDecomp_custom_random.get_optimized_parameters();
             initial_guess = RANDOM;
         }
         else if ( current_minimum_random > optimization_tolerance && current_minimum_close_to_zero < optimization_tolerance ) {
             current_minimum_loc = current_minimum_close_to_zero;
             optimized_parameters_mtx_loc = cDecomp_custom_close_to_zero.get_optimized_parameters();
             initial_guess = CLOSE_TO_ZERO;
         }
         else if ( current_minimum_random < optimization_tolerance && current_minimum_close_to_zero < optimization_tolerance ) {
             Matrix_real optimized_parameters_mtx_random = cDecomp_custom_random.get_optimized_parameters();
             Matrix_real optimized_parameters_mtx_close_to_zero = cDecomp_custom_close_to_zero.get_optimized_parameters();

             int panelty_random         = get_panelty(gate_structure_loc, optimized_parameters_mtx_random);
             int panelty_close_to_zero = get_panelty(gate_structure_loc, optimized_parameters_mtx_close_to_zero );

             if ( panelty_random < panelty_close_to_zero ) {
                 current_minimum_loc = current_minimum_random;
                 optimized_parameters_mtx_loc = cDecomp_custom_random.get_optimized_parameters();
                 initial_guess = RANDOM;
             }
             else {
                 current_minimum_loc = current_minimum_close_to_zero;
                 optimized_parameters_mtx_loc = cDecomp_custom_close_to_zero.get_optimized_parameters();
                 initial_guess = CLOSE_TO_ZERO;
             }

        }
        else {
           if ( current_minimum_random < current_minimum_close_to_zero ) {
                current_minimum_loc = current_minimum_random;
                optimized_parameters_mtx_loc = cDecomp_custom_random.get_optimized_parameters();
                initial_guess = RANDOM;
           }
           else {
                current_minimum_loc = current_minimum_close_to_zero;
                optimized_parameters_mtx_loc = cDecomp_custom_close_to_zero.get_optimized_parameters();
                initial_guess = CLOSE_TO_ZERO;
           }

        }

        minimum_vec.push_back(current_minimum_loc);
        gate_structure_vec.push_back(gate_structure_loc);
        optimized_parameters_vec.push_back(optimized_parameters_mtx_loc);



        if ( current_minimum_loc < optimization_tolerance ) {
	    std::stringstream sstream;
            sstream << "Optimization problem solved with " << gate_structure_loc->get_gate_num() << " decomposing layers in " << (end_time_loc-start_time_loc).seconds() << " seconds." << std::endl;
            print(sstream, 1);	       
            break;
        }   
        else {
            std::stringstream sstream;
            sstream << "Optimization problem converged to " << current_minimum_loc << " with " <<  gate_structure_loc->get_gate_num() << " decomposing layers in "   << (end_time_loc-start_time_loc).seconds() << " seconds." << std::endl;
            print(sstream, 1);  
        }

        level++;
    }

//exit(-1);

    // find the best decomposition
    int idx_min = 0;
    double current_minimum = minimum_vec[0];
    for (int idx=1; idx<(int)minimum_vec.size(); idx++) {
        if( current_minimum > minimum_vec[idx] ) {
            idx_min = idx;
            current_minimum = minimum_vec[idx];
        }
    }
     
    Gates_block* gate_structure_loc = gate_structure_vec[idx_min];
    optimized_parameters_mtx_loc = optimized_parameters_vec[idx_min];

    // release unnecesarry data
    for (int idx=0; idx<(int)minimum_vec.size(); idx++) {
        if( idx == idx_min ) {
            continue;
        }
        delete( gate_structure_vec[idx] );
    }    
    minimum_vec.clear();
    gate_structure_vec.clear();
    optimized_parameters_vec.clear();
    

    if (current_minimum > optimization_tolerance) {
       std::stringstream sstream;
       sstream << "Decomposition did not reached prescribed high numerical precision." << std::endl;
       print(sstream, 1);              
       optimization_tolerance = 1.5*current_minimum < 1e-2 ? 1.5*current_minimum : 1e-2;
    }
    
    std::stringstream sstream;
    sstream << "Continue with the compression of gate structure consisting of " << gate_structure_loc->get_gate_num() << " decomposing layers." << std::endl;
    print(sstream, 1);	
    return gate_structure_loc;
       
}



/**
@brief ???????????????
*/
Gates_block*
N_Qubit_Decomposition_adaptive::compress_gate_structure( Gates_block* gate_structure ) {


    int layer_num_max;
    int layer_num_orig = gate_structure->get_gate_num()-1;
    if ( layer_num_orig < 50 ) layer_num_max = 10;
    else if ( layer_num_orig < 60 ) layer_num_max = 4;
    else layer_num_max = 2;

    // create a list of layers to be tested for removal.
    std::vector<int> layers_to_remove;
    layers_to_remove.reserve(layer_num_orig);
    for (int idx=0; idx<layer_num_orig; idx++ ) {
        layers_to_remove.push_back(idx+1);
    }   

    while ( (int)layers_to_remove.size() > layer_num_max ) {
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
        Matrix_real optimized_parameters_loc;

        if ( current_minimum > 1e-2 ) {
             optimized_parameters_loc = Matrix_real(0, 0);
        }
        else {
            optimized_parameters_loc = optimized_parameters_mtx.copy();
        }

   //         optimized_parameters_loc = optimized_parameters_mtx.copy();

        Gates_block* gate_structure_reduced = compress_gate_structure( gate_structure, layers_to_remove[idx_to_remove], optimized_parameters_loc,  current_minimum_loc  );
        if ( optimized_parameters_loc.size() == 0 ) optimized_parameters_loc = optimized_parameters_mtx.copy();

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
       std::stringstream sstream;
       sstream << "gate structure reduced from " << layer_num_orig+1 << " to " << layer_num << " decomposing layers" << std::endl;
       print(sstream, 1);	
    }
    else {
       std::stringstream sstream;
       sstream << "gate structure kept at " << layer_num << " layers" << std::endl;
       print(sstream, 1);		            
    }


    return gate_structure;



}

/**
@brief ???????????????
*/
Gates_block* 
N_Qubit_Decomposition_adaptive::compress_gate_structure( Gates_block* gate_structure, int layer_idx, Matrix_real& optimized_parameters, double& current_minimum_loc ) {

    // create reduced gate structure without layer indexed by layer_idx
    Gates_block* gate_structure_reduced = gate_structure->clone();
    gate_structure_reduced->release_gate( layer_idx );

    Matrix_real parameters_reduced;
    if ( optimized_parameters.size() > 0 ) {
        parameters_reduced = create_reduced_parameters( gate_structure_reduced, optimized_parameters, layer_idx );
    }
    else {
        parameters_reduced = Matrix_real(0, 0);
    }



    N_Qubit_Decomposition_custom cDecomp_custom;
       
    // solve the optimization problem in isolated optimization process
    cDecomp_custom = N_Qubit_Decomposition_custom( Umtx.copy(), qbit_num, false, initial_guess);
    cDecomp_custom.set_custom_gate_structure( gate_structure_reduced );
    cDecomp_custom.set_optimized_parameters( parameters_reduced.get_data(), parameters_reduced.size() );
    cDecomp_custom.set_verbose(0);
    cDecomp_custom.set_debugfile("");
    cDecomp_custom.set_max_iteration( max_iterations );
    cDecomp_custom.set_iteration_loops( iteration_loops );
    cDecomp_custom.set_optimization_blocks( gate_structure_reduced->get_gate_num() ) ;
    cDecomp_custom.set_optimization_tolerance( optimization_tolerance );
    cDecomp_custom.start_decomposition(true);
    double current_minimum_tmp = cDecomp_custom.get_current_minimum();

    if ( current_minimum_tmp < optimization_tolerance ) {
        //cDecomp_custom.list_gates(0);
        optimized_parameters = cDecomp_custom.get_optimized_parameters();
        current_minimum_loc = current_minimum_tmp;
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
           std::stringstream sstream;
	   sstream << "N_Qubit_Decomposition_adaptive::replace_trivial_adaptive_gates: Only block gates are accepted in this conversion." << std::endl;
           print(sstream, 1);	
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
N_Qubit_Decomposition_adaptive::remove_trivial_gates( Gates_block* gate_structure, Matrix_real& optimized_parameters, double& current_minimum_loc ) {


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
                Gates_block* gate_structure_tmp = compress_gate_structure( gate_structure_loc, idx, optimized_parameters_loc, current_minimum_loc );

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


    //The stringstream input to store the output messages.
    std::stringstream sstream;

    // creating block of gates
    Gates_block* block = new Gates_block( qbit_num );

    std::vector<Gates_block* > layers;


    if ( topology.size() > 0 ) {
        for ( std::vector<matrix_base<int>>::iterator it=topology.begin(); it!=topology.end(); it++) {

            if ( it->size() != 2 ) {
                std::stringstream sstream;
	        sstream << "The connectivity data should contains two qubits" << std::endl;
	        print(sstream, 0);	
                it->print_matrix();
                exit(-1);
            }

            int control_qbit_loc = (*it)[0];
            int target_qbit_loc = (*it)[1];

            if ( control_qbit_loc >= qbit_num || target_qbit_loc >= qbit_num ) {
                std::stringstream sstream;
	        sstream << "Label of control/target qubit should be less than the number of qubits in the register." << std::endl;	        
                print(sstream, 0);
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
/*
    // creating block of gates
    Gates_block* block1 = new Gates_block( qbit_num );

block1->add_u3(3, true, false, true);
block1->add_u3(3, true, false, true);
block1->add_u3(3, true, false, true);
block1->add_u3(2, true, false, true);
block1->add_u3(0, true, false, true);
block1->add_u3(3, true, false, true);
*/
/*
    Gates_block* block2 = new Gates_block( qbit_num );

block2->add_u3(4, true, false, true);
block2->add_u3(2, true, false, true);
block2->add_u3(5, true, false, true);
block2->add_u3(2, true, false, true);
block2->add_u3(4, true, false, true);
block2->add_u3(3, true, false, true);
*/
    // adding the opeartion block to the gates
    gate_structure->add_gate( block );

    //gate_structure->add_gate( block1 );
    //gate_structure->add_gate( block2 );

}



/**
@brief Call to set custom layers to the gate structure that are intended to be used in the subdecomposition.
@param gate_structure An <int, Gates_block*> map containing the gate structure used in the individual subdecomposition (default is used, if a gate structure for specific subdecomposition is missing).
*/
void 
N_Qubit_Decomposition_adaptive::set_adaptive_gate_structure( Gates_block* gate_structure_in ) {

    gate_structure = gate_structure_in->clone();

}






