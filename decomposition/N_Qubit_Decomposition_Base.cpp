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
/*! \file N_Qubit_Decomposition.cpp
    \brief Base class to determine the decomposition of a unitary into a sequence of two-qubit and one-qubit gate gates.
    This class contains the non-template implementation of the decomposition class
*/

#include "N_Qubit_Decomposition_Base.h"
#include "N_Qubit_Decomposition_Cost_Function.h"

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

}




/**
@brief Nullary constructor of the class.
@return An instance of the class
*/
N_Qubit_Decomposition_Base::N_Qubit_Decomposition_Base() {

    // logical value. Set true if finding the minimum number of gate layers is required (default), or false when the maximal number of two-qubit gates is used (ideal for general unitaries).
    optimize_layer_num  = false;

    // A string describing the type of the class
    type = N_QUBIT_DECOMPOSITION_CLASS;

    // The global minimum of the optimization problem
    global_target_minimum = 0;

}

/**
@brief Constructor of the class.
@param Umtx_in The unitary matrix to be decomposed
@param qbit_num_in The number of qubits spanning the unitary Umtx
@param optimize_layer_num_in Optional logical value. If true, then the optimization tries to determine the lowest number of the layers needed for the decomposition. If False (default), the optimization is performed for the maximal number of layers.
@param initial_guess_in Enumeration element indicating the method to guess initial values for the optimization. Possible values: 'zeros=0' ,'random=1', 'close_to_zero=2'
@return An instance of the class
*/
N_Qubit_Decomposition_Base::N_Qubit_Decomposition_Base( Matrix Umtx_in, int qbit_num_in, bool optimize_layer_num_in, guess_type initial_guess_in= CLOSE_TO_ZERO ) : Decomposition_Base(Umtx_in, qbit_num_in, initial_guess_in) {

    // logical value. Set true if finding the minimum number of gate layers is required (default), or false when the maximal number of two-qubit gates is used (ideal for general unitaries).
    optimize_layer_num  = optimize_layer_num_in;

    // A string describing the type of the class
    type = N_QUBIT_DECOMPOSITION_CLASS;

    // The global minimum of the optimization problem
    global_target_minimum = 0;

    // number of iteratrion loops in the optimization
    iteration_loops[2] = 3;

    // filling in numbers that were not given in the input
    for ( std::map<int,int>::iterator it = max_layer_num_def.begin(); it!=max_layer_num_def.end(); it++) {
        if ( max_layer_num.count( it->first ) == 0 ) {
            max_layer_num.insert( std::pair<int, int>(it->first,  it->second) );
        }
    }


}



/**
@brief Destructor of the class
*/
N_Qubit_Decomposition_Base::~N_Qubit_Decomposition_Base() {


}


/**
@brief Call to add further layer to the gate structure used in the subdecomposition.
*/
void 
N_Qubit_Decomposition_Base::add_finalyzing_layer() {


    // creating block of gates
    Gates_block* block = new Gates_block( qbit_num );

    // adding U3 gate to the block
    bool Theta = true;
    bool Phi = false;
    bool Lambda = true;

    for (int qbit=0; qbit<qbit_num; qbit++) {
        block->add_u3(qbit, Theta, Phi, Lambda);
    }

    // adding the opeartion block to the gates
    add_gate( block );

}




/**
@brief Calculate the error of the decomposition according to the spectral norm of \f$ U-U_{approx} \f$, where \f$ U_{approx} \f$ is the unitary produced by the decomposing quantum cirquit.
@param decomposed_matrix The decomposed matrix, i.e. the result of the decomposing gate structure applied on the initial unitary.
@return Returns with the calculated spectral norm.
*/
void
N_Qubit_Decomposition_Base::calc_decomposition_error(Matrix& decomposed_matrix ) {

	// (U-U_{approx}) (U-U_{approx})^\dagger = 2*I - U*U_{approx}^\dagger - U_{approx}*U^\dagger
	// U*U_{approx}^\dagger = decomposed_matrix_copy
	
 	Matrix A(matrix_size, matrix_size);
	QGD_Complex16* A_data = A.get_data();
	QGD_Complex16* decomposed_data = decomposed_matrix.get_data();
	QGD_Complex16 phase;
	phase.real = decomposed_matrix[0].real/(std::sqrt(decomposed_matrix[0].real*decomposed_matrix[0].real + decomposed_matrix[0].imag*decomposed_matrix[0].imag));
	phase.imag = -decomposed_matrix[0].imag/(std::sqrt(decomposed_matrix[0].real*decomposed_matrix[0].real + decomposed_matrix[0].imag*decomposed_matrix[0].imag));

	for (int idx=0; idx<matrix_size; idx++ ) {
		for (int jdx=0; jdx<matrix_size; jdx++ ) {
			
			if (idx==jdx) {
				QGD_Complex16 mtx_val = mult(phase, decomposed_data[idx*matrix_size+jdx]);
				A_data[idx*matrix_size+jdx].real = 2.0 - 2*mtx_val.real;
				A_data[idx*matrix_size+jdx].imag = 0;
			}
			else {
				QGD_Complex16 mtx_val_ij = mult(phase, decomposed_data[idx*matrix_size+jdx]);
				QGD_Complex16 mtx_val_ji = mult(phase, decomposed_data[jdx*matrix_size+idx]);
				A_data[idx*matrix_size+jdx].real = - mtx_val_ij.real - mtx_val_ji.real;
				A_data[idx*matrix_size+jdx].imag = - mtx_val_ij.imag + mtx_val_ji.imag;
			}

		}
	}


	Matrix alpha(matrix_size, 1);
	Matrix beta(matrix_size, 1);
	Matrix B = create_identity(matrix_size);

	// solve the generalized eigenvalue problem of I- 1/2
	LAPACKE_zggev( CblasRowMajor, 'N', 'N',
                          matrix_size, A.get_data(), matrix_size, B.get_data(),
                          matrix_size, alpha.get_data(),
                          beta.get_data(), NULL, matrix_size, NULL,
                          matrix_size );

	// determine the largest eigenvalue
	double eigval_max = 0;
	for (int idx=0; idx<matrix_size; idx++) {
		double eigval_abs = std::sqrt((alpha[idx].real*alpha[idx].real + alpha[idx].imag*alpha[idx].imag) / (beta[idx].real*beta[idx].real + beta[idx].imag*beta[idx].imag));
		if ( eigval_max < eigval_abs ) eigval_max = eigval_abs;		
	}

	// the norm is the square root of the largest einegvalue.
	decomposition_error = std::sqrt(eigval_max);


}



/**
@brief final optimization procedure improving the accuracy of the decompositin when all the qubits were already disentangled.
*/
void  N_Qubit_Decomposition_Base::final_optimization() {

	//The stringstream input to store the output messages.
	std::stringstream sstream;
	sstream << "***************************************************************" << std::endl;
	sstream << "Final fine tuning of the parameters in the " << qbit_num << "-qubit decomposition" << std::endl;
	sstream << "***************************************************************" << std::endl;
	print(sstream, 1);	    	

         


        //# setting the global minimum
        global_target_minimum = 0;


        if ( optimized_parameters_mtx.size() == 0 ) {
            solve_optimization_problem(NULL, 0);
        }
        else {
            current_minimum = optimization_problem(optimized_parameters_mtx.get_data());
            if ( check_optimization_solution() ) return;

            solve_optimization_problem(optimized_parameters_mtx.get_data(), parameter_num);
        }
}



/**
// @brief Call to solve layer by layer the optimization problem. The optimalized parameters are stored in attribute optimized_parameters.
// @param num_of_parameters Number of parameters to be optimized
// @param solution_guess_gsl A GNU Scientific Library vector containing the solution guess.
*/
void N_Qubit_Decomposition_Base::solve_layer_optimization_problem( int num_of_parameters, gsl_vector *solution_guess_gsl) {

        if (gates.size() == 0 ) {
            return;
        }


        if (solution_guess_gsl == NULL) {
            solution_guess_gsl = gsl_vector_alloc(num_of_parameters);
        }


        if (optimized_parameters_mtx.size() == 0) {
            optimized_parameters_mtx = Matrix_real(1, num_of_parameters);
            memcpy(optimized_parameters_mtx.get_data(), solution_guess_gsl->data, num_of_parameters*sizeof(double) );
        }

        // maximal number of iteration loops
        int iteration_loops_max;
        try {
            iteration_loops_max = std::max(iteration_loops[qbit_num], 1);
        }
        catch (...) {
            iteration_loops_max = 1;
        }


        // do the optimization loops
        for (int idx=0; idx<iteration_loops_max; idx++) {

            int iter = 0;
            int status;

            const gsl_multimin_fdfminimizer_type *T;
            gsl_multimin_fdfminimizer *s;

            N_Qubit_Decomposition_Base* par = this;


            gsl_multimin_function_fdf my_func;


            my_func.n = num_of_parameters;
            my_func.f = optimization_problem;
            my_func.df = optimization_problem_grad;
            my_func.fdf = optimization_problem_combined;
            my_func.params = par;


            T = gsl_multimin_fdfminimizer_vector_bfgs2;
            s = gsl_multimin_fdfminimizer_alloc (T, num_of_parameters);

            gsl_multimin_fdfminimizer_set(s, &my_func, solution_guess_gsl, 0.01, 0.1);

            do {
                iter++;
                gsl_set_error_handler_off();
                status = gsl_multimin_fdfminimizer_iterate (s);

                if (status) {
                  break;
                }

                status = gsl_multimin_test_gradient (s->gradient, gradient_threshold);

            } while (status == GSL_CONTINUE && iter < iter_max);

            if (current_minimum > s->f) {
                current_minimum = s->f;
                memcpy( optimized_parameters_mtx.get_data(), s->x->data, num_of_parameters*sizeof(double) );
                gsl_multimin_fdfminimizer_free (s);

                for ( int jdx=0; jdx<num_of_parameters; jdx++) {
                    solution_guess_gsl->data[jdx] = solution_guess_gsl->data[jdx] + (2*double(rand())/double(RAND_MAX)-1)*2*M_PI/100;
                }
            }
            else {
                for ( int jdx=0; jdx<num_of_parameters; jdx++) {
                    solution_guess_gsl->data[jdx] = solution_guess_gsl->data[jdx] + (2*double(rand())/double(RAND_MAX)-1)*2*M_PI;
                }
                gsl_multimin_fdfminimizer_free (s);
            }



        }


}



/**
// @brief The optimization problem of the final optimization
// @param parameters An array of the free parameters to be optimized. (The number of teh free paramaters should be equal to the number of parameters in one sub-layer)
// @return Returns with the cost function. (zero if the qubits are desintangled.)
*/
double N_Qubit_Decomposition_Base::optimization_problem( double* parameters ) {

    // get the transformed matrix with the gates in the list
    Matrix_real parameters_mtx(parameters, 1, parameter_num );
    Matrix matrix_new = get_transformed_matrix( parameters_mtx, gates.begin(), gates.size(), Umtx );

    return get_cost_function(matrix_new);

}


/**
// @brief The optimization problem of the final optimization
// @param parameters An array of the free parameters to be optimized. (The number of teh free paramaters should be equal to the number of parameters in one sub-layer)
// @return Returns with the cost function. (zero if the qubits are desintangled.)
*/
double N_Qubit_Decomposition_Base::optimization_problem( Matrix_real& parameters ) {

    // get the transformed matrix with the gates in the list
    if ( parameters.size() != parameter_num ) {
        std::stringstream sstream;
	sstream << "Number of free paramaters should be " << parameter_num << ", but got " << parameters.size() << std::endl;
        print(sstream, 0);	  
        exit(-1);
    }


    Matrix matrix_new = get_transformed_matrix( parameters, gates.begin(), gates.size(), Umtx );

    return get_cost_function(matrix_new);

}


/**
// @brief The optimization problem of the final optimization
@param parameters A GNU Scientific Library containing the parameters to be optimized.
@param void_instance A void pointer pointing to the instance of the current class.
@return Returns with the cost function. (zero if the qubits are desintangled.)
*/
double N_Qubit_Decomposition_Base::optimization_problem( const gsl_vector* parameters, void* void_instance ) {

    N_Qubit_Decomposition_Base* instance = reinterpret_cast<N_Qubit_Decomposition_Base*>(void_instance);
    std::vector<Gate*> gates_loc = instance->get_gates();

    // get the transformed matrix with the gates in the list
    Matrix Umtx_loc = instance->get_Umtx();
    Matrix_real parameters_mtx(parameters->data, 1, instance->get_parameter_num() );
    Matrix matrix_new = instance->get_transformed_matrix( parameters_mtx, gates_loc.begin(), gates_loc.size(), Umtx_loc );

    return get_cost_function(matrix_new);
}


/**
@brief Calculate the approximate derivative (f-f0)/(x-x0) of the cost function with respect to the free parameters.
@param parameters A GNU Scientific Library vector containing the free parameters to be optimized.
@param void_instance A void pointer pointing to the instance of the current class.
@param grad A GNU Scientific Library vector containing the calculated gradient components.
*/

void N_Qubit_Decomposition_Base::optimization_problem_grad( const gsl_vector* parameters, void* void_instance, gsl_vector* grad ) {

    // The function value at x0
    double f0;

    // calculate the approximate gradient
    optimization_problem_combined( parameters, void_instance, &f0, grad);

}


/**
@brief Call to calculate both the cost function and the its gradient components.
@param parameters A GNU Scientific Library vector containing the free parameters to be optimized.
@param void_instance A void pointer pointing to the instance of the current class.
@param f0 The value of the cost function at x0.
@param grad A GNU Scientific Library vector containing the calculated gradient components.
*/

void N_Qubit_Decomposition_Base::optimization_problem_combined( const gsl_vector* parameters, void* void_instance, double* f0, gsl_vector* grad ) {

    N_Qubit_Decomposition_Base* instance = reinterpret_cast<N_Qubit_Decomposition_Base*>(void_instance);

    int parameter_num_loc = instance->get_parameter_num();


///////////////////////////////////////
tbb::tick_count t0_DFE = tbb::tick_count::now();/////////////////////////////////
    Matrix_real parameters_mtx(parameters->data, 1, parameters->size);
    Gates_block* gates_loc = new Gates_block( instance->get_qbit_num() );
    instance->extract_gates(gates_loc);

    gates_loc->release_gate( 0 ); // THE FIRST GATE IS A GENERAL GATE APPENDED IN THE BLOCK-WISE OPTIMISATION ROUTINE OF DECOMPOSITION_BASE -- need to be discarded

    int gatesNum;
    DFEgate_kernel_type* DFEgates = gates_loc->convert_to_DFE_gates( parameters_mtx, gatesNum );

    Matrix&& Umtx_loc = instance->get_Umtx();

//uploadMatrix2DFE( Umtx_loc );
    calcqgdKernelDFE( Umtx_loc.rows, DFEgates, gatesNum );


    delete gates_loc;
    delete[] DFEgates;
tbb::tick_count t1_DFE = tbb::tick_count::now();/////////////////////////////////
std::cout << "time elapsed DFE: " << (t1_DFE-t0_DFE).seconds() << ", expected time: " << ((double)(Umtx_loc.rows*Umtx_loc.rows*gatesNum/3 + 531))/350000000 + 0.001<< std::endl;
///////////////////////////////////////

    // vector containing gradients of the transformed matrix
    std::vector<Matrix> Umtx_deriv;

tbb::tick_count t0_CPU = tbb::tick_count::now();/////////////////////////////////
    tbb::parallel_invoke(
        [&]{*f0 = instance->optimization_problem(parameters, reinterpret_cast<void*>(instance)); },
        [&]{
            Matrix Umtx_loc = instance->get_Umtx();
            Matrix_real parameters_mtx(parameters->data, 1, parameters->size);
            Umtx_deriv = instance->apply_derivate_to( parameters_mtx, Umtx_loc );
        });

    tbb::parallel_for( tbb::blocked_range<int>(0,parameter_num_loc,2), [&](tbb::blocked_range<int> r) {
        for (int idx=r.begin(); idx<r.end(); ++idx) { 
            // This is approximate derivate giving a good approximation when f0->0. Helps to avoid higher barren plateaus
            //double grad_comp = (get_cost_function(Umtx_deriv[idx]) - 1.0)/(*f0);  

            //double f = get_cost_function(Umtx_deriv[idx]);
            //double grad_comp = (f*f - 1.0)/(*f0)/2;
            double grad_comp = (get_cost_function(Umtx_deriv[idx]) - 1.0);
            gsl_vector_set(grad, idx, grad_comp);
        }
    });

tbb::tick_count t1_CPU = tbb::tick_count::now();/////////////////////////////////
std::cout << "time elapsed CPU: " << (t1_CPU-t0_CPU).seconds() << " number of parameters: " << parameter_num_loc << std::endl;
std::cout << "cost function CPU: " << *f0 << std::endl;


std::cout << "N_Qubit_Decomposition_Base::optimization_problem_combined" << std::endl;
std::string error("N_Qubit_Decomposition_Base::optimization_problem_combined");
        throw error;



}

