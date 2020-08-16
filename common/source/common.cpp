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

//
// @brief A base class responsible for constructing matrices of C-NOT, U3
// gates acting on the N-qubit space

#include <common.h> 
#include <sstream>

using namespace std;

// Calculates the power of 2
int Power_of_2(int n) {
  if (n == 0) return 1;
  if (n == 1) return 2;

  return 2 * Power_of_2(n-1);
}


// print the matrix
void print_mtx( MKL_Complex16* matrix, int rows, int cols ) {

    for ( int row_idx=0; row_idx < rows; row_idx++ ) {
        for ( int col_idx=0; col_idx < cols; col_idx++ ) {
            int element_idx = row_idx*cols + col_idx;    
            printf("%f + i*%f,  ", matrix[element_idx].real, matrix[element_idx].imag);
        }
        printf("\n");
    }
    printf("\n\n\n");

}


// print a CNOT
void print_CNOT( MKL_Complex16* matrix, int size ) {

    for ( int row_idx=0; row_idx < size; row_idx++ ) {
        for ( int col_idx=0; col_idx < size; col_idx++ ) {
            int element_idx = row_idx*size+col_idx;    
            printf("%d,  ", int(matrix[element_idx].real));
        }
        printf("\n");
    }
    printf("\n\n\n");
}


// @brief Add an integer to an integer vector if the integer is not already an element of the vector. The sorted order is kept during the process
void add_unique_elelement( vector<int> &involved_qbits, int qbit ) {

    if ( involved_qbits.size() == 0 ) {
        involved_qbits.push_back( qbit );
    }

    for(std::vector<int>::iterator it = involved_qbits.begin(); it != involved_qbits.end(); ++it) {

        int current_val = *it;

        if (current_val == qbit) {
            return;
        } 
        else if (current_val > qbit) {
            involved_qbits.insert( it, qbit );
            return;
        }

    }

    // add the qbit to the end if neither of the conditions were satisfied
    involved_qbits.push_back( qbit );

    return;

}


// @brief Create an identity matrix
MKL_Complex16* create_identity( int matrix_size ) {

    MKL_Complex16* matrix = (MKL_Complex16*)mkl_calloc(matrix_size*matrix_size, sizeof(MKL_Complex16), 64);

    // setting the giagonal elelments to identity
    #pragma omp parallel for
    for(int idx = 0; idx < matrix_size; ++idx)
    {
        int element_index = idx*matrix_size + idx;
            matrix[element_index].real = 1;
    }

    return matrix;

}



// @brief Call to calculate the product of two matrices using cblas_zgemm3m
MKL_Complex16* zgemm3m_wrapper( MKL_Complex16* A, MKL_Complex16* B, int matrix_size) {

    // parameters alpha and beta for the cblas_zgemm3m function
    double alpha = 1;
    double beta = 0;

    // preallocate array for the result
    MKL_Complex16* C = (MKL_Complex16*)mkl_malloc(matrix_size*matrix_size*sizeof(MKL_Complex16), 64); 

    // calculate the product of A and B
    cblas_zgemm3m (CblasRowMajor, CblasNoTrans, CblasNoTrans, matrix_size, matrix_size, matrix_size, &alpha, A, matrix_size, B, matrix_size, &beta, C, matrix_size);

    return C;
}


// @brief Calculate the product of complex matrices stored in a vector of matrices
MKL_Complex16* reduce_zgemm( vector<MKL_Complex16*> mtxs, int matrix_size ) {
    

    if (mtxs.size() == 0 ) {
        return create_identity(matrix_size);
    }


    // parameters alpha and beta for the cblas_zgemm3m function
    double alpha = 1;
    double beta = 0;


    // pointers to matrices to be used in the multiplications
    MKL_Complex16* A = NULL;
    MKL_Complex16* B = NULL;

    // the iteration number
    int iteration = 0;


    std::vector<MKL_Complex16*>::iterator it = mtxs.begin();
    MKL_Complex16* C = *it;
    it++;   
 
    // calculate the product of complex matrices
    for(it; it != mtxs.end(); ++it) {

        iteration++;

        A = C;
        B = *it;

/*printf("The matrix A:\n");
print_mtx( A, matrix_size, matrix_size );
printf("\n");
printf("The matrix B:\n");
print_mtx( B, matrix_size, matrix_size );
printf("\n");*/

        // calculate the product of A and B
        C = zgemm3m_wrapper(A, B, matrix_size);

        // free the memory of previously allocated matrix A which is irrelevant for the further calculations
        if ( iteration>1 ) {
            mkl_free(A);
        }

/*printf("The product matrix C:\n");
print_mtx( C, matrix_size, matrix_size );
printf("\n");*/


    }


    return C;
}


// @brief subtract a scalar from the diagonal of a matrix
void subtract_diag( MKL_Complex16* & mtx,  int matrix_size, MKL_Complex16 scalar ) {

    #pragma omp parallel for
    for(int idx = 0; idx < matrix_size; idx++)   {
        int element_idx = idx*matrix_size+idx;
        mtx[element_idx].real = mtx[element_idx].real - scalar.real;
        mtx[element_idx].imag = mtx[element_idx].imag - scalar.imag;
    }

}



// calculate the cost funtion from the submatrices of the given matrix 
double get_submatrix_cost_function(MKL_Complex16* matrix, int matrix_size) {

    // ********************************
    // Calculate the submatrix products
    // ********************************

    // number ofcolumns in the submatrices
    int submatrix_size = matrix_size/2;
    // the number of rows in the matrix containing the products of submatrices
    int rows_submatrix_prods = 8*submatrix_size;
    // number of elements in the matrix of submatrix products
    int element_num = submatrix_size*rows_submatrix_prods;

    // preallocate memory for the submatrix products  
    MKL_Complex16* submatrix_prods = (MKL_Complex16*)mkl_malloc(element_num*sizeof(MKL_Complex16), 64);


    // calculate the elements of the submatirx products by row
    #pragma omp parallel for
    for ( int row_idx=0; row_idx<rows_submatrix_prods; row_idx++ ) {

        // offset idices to get the correct submatrix in the matrix 
        int row_offset_1;
        int col_offset_1;
        int row_offset_2;
        int col_offset_2;

        // element indices in the submatrix
        int row_idx_submtx;
        int col_idx_submtx;

        //determine the id of the submatrix product in the list
        int prod_idx = int(row_idx/submatrix_size);

        // get the ID-s of submatrices to be used in the product
        int first_submatrix = int(prod_idx/2);
        int second_submatrix = int(prod_idx % 2);

        // get the row and col offsets in the initial matrix
        if (first_submatrix == 0) {
            row_offset_1 = 0;
            col_offset_1 = 0;
        }
        else if (first_submatrix == 1) {
            row_offset_1 = 0;
            col_offset_1 = submatrix_size;
        }
        else if (first_submatrix == 2) {
            row_offset_1 = submatrix_size;
            col_offset_1 = 0;
        }
        else if (first_submatrix == 3) {
            row_offset_1 = submatrix_size;
            col_offset_1 = submatrix_size;
        }


        if (second_submatrix == 0) {
            row_offset_2 = 0;
            col_offset_2 = 0;
        }
        else if (second_submatrix == 1) {
            row_offset_2 = 0;
            col_offset_2 = submatrix_size;
        }

        // determine the row index in the submatrix
        row_idx_submtx = int(row_idx % submatrix_size);
        //col_idx = int(y % submatrix_size)
  

        // row index in the original matrix
        int row_idx_1 = row_idx_submtx+row_offset_1;

        // calculate the col-wise products of the elements
        for (int col_idx=0; col_idx<submatrix_size; col_idx++) {

            // col index in the original matrix --- col is converted to row due to the transpose
            int row_idx_2 = col_idx+row_offset_2;

            // variable storing the element products
            MKL_Complex16 tmp;
            tmp.real = 0;
            tmp.imag = 0;

            // calculate the product
            for (int idx=0; idx<submatrix_size; idx++) {            
                MKL_Complex16 a = matrix[row_idx_1*matrix_size + idx+col_offset_1];
                MKL_Complex16 b = matrix[row_idx_2*matrix_size + idx+col_offset_2];
                tmp.real = tmp.real + a.real*b.real + a.imag*b.imag;
                tmp.imag = tmp.imag + a.imag*b.real - a.real*b.imag;
            }

            // setting the element in the matrix conating the submatrix products
            submatrix_prods[row_idx*submatrix_size + col_idx] = tmp;
        }

    }

//printf("The matrix of the submatrix products:\n");
//print_mtx( submatrix_prods, rows_submatrix_prods, submatrix_size );
//printf("\n");

    // ********************************
    // Subtract the corner elements
    // ********************************

    // get corner elements
    MKL_Complex16* corner_elements = (MKL_Complex16*)mkl_malloc(8*sizeof(MKL_Complex16), 64);
    #pragma omp parallel for
    for ( int corner_idx=0; corner_idx<8; corner_idx++ ) {
        corner_elements[corner_idx] = submatrix_prods[corner_idx*submatrix_size*submatrix_size];
    }

    #pragma omp parallel for
    for ( int row_idx=0; row_idx<rows_submatrix_prods; row_idx++ ) {

        // identify the submatrix product corresponding to the given row index
        int submatrix_prod_id = int(row_idx/submatrix_size);

        // get the corner element
        MKL_Complex16 corner_element = corner_elements[submatrix_prod_id];
       
        // subtract the corner element from the diagonal elemnts
        int element_idx = row_idx*submatrix_size + (row_idx % submatrix_size);
        submatrix_prods[element_idx].real = submatrix_prods[element_idx].real - corner_element.real;
        submatrix_prods[element_idx].imag = submatrix_prods[element_idx].imag - corner_element.imag;

    }

    mkl_free( corner_elements );

//printf("The matrix of the submatrix products after subtracking corner elements:\n");
//print_mtx( submatrix_prods, rows_submatrix_prods, submatrix_size );
//printf("\n");


    // ********************************
    // Calculate the |x|^2 value of the elements of the submatrixproducts
    // ********************************
    #pragma omp parallel for
    for ( int idx=0; idx<element_num; idx++ ) {
        submatrix_prods[idx].real = submatrix_prods[idx].real*submatrix_prods[idx].real + submatrix_prods[idx].imag*submatrix_prods[idx].imag;
        // for performance reason we leave the imaginary part intact (we dont neet it anymore)
        //submatrix_prods[idx].imag = 0;
    }


    // ********************************
    // Calculate the final cost function
    // ********************************
    double cost_function = 0;
    #pragma omp parallel for reduction(+:cost_function)
    for ( int row_idx=0; row_idx<rows_submatrix_prods; row_idx++ ) {

        // calculate thesum for each row
        for (int col_idx=0; col_idx<submatrix_size; col_idx++) {

            cost_function = cost_function + submatrix_prods[row_idx*submatrix_size + col_idx].real;
        } 
    }

//printf("The cost function is: %f\n", cost_function);

    return cost_function;

}

