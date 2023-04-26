# include <math.h>
# include <interval.h>
# include <vector>
# include <stdio.h>
# include <iostream>
# include <QJsonObject>
# include <QFile>
# include <QTextStream>

#include <gsl/gsl_vector.h>
#include "N_Qubit_Decomposition_adaptive.h"
#include "Decomposition_Base.h"
#include "matrix_real.h"
#include <math.h>

//../bin/OptimusApp --filename=libsquander.so --opt_method=Pso --pso_particles=100 --pso_generations=10

using namespace std;

extern "C"
{
/*thread_local*/ N_Qubit_Decomposition_adaptive* decomp = NULL;
/*thread_local*/ int num_of_parameters = 0;
thread_local gsl_vector* grad_gsl = NULL;

void    init(QJsonObject data)
{
    printf("init\n");
    if (decomp == NULL) {
        int qbit_num, accelerator_num = 0, level_limit=5, level_limit_min=5;
        system("python ../../sequential-quantum-gate-decomposer/saveunitary.py ../../sequential-quantum-gate-decomposer/examples/vqe/19CNOT.qasm ./unitary.binary");
        std::string filename("unitary.binary");
        Matrix Umtx = Decomposition_Base().import_unitary_from_binary(filename);
        qbit_num = log2(Umtx.rows);
        //Matrix Umtx = create_identity( 1 << qbit_num);
        decomp = new N_Qubit_Decomposition_adaptive( Umtx, qbit_num, level_limit, level_limit_min, accelerator_num );
        Matrix_real optimized_parameters_mtx_loc;
        Gates_block* gate_structure_loc = decomp->determine_initial_gate_structure(optimized_parameters_mtx_loc);
        decomp->combine( gate_structure_loc );
        num_of_parameters = optimized_parameters_mtx_loc.rows * optimized_parameters_mtx_loc.cols;
        grad_gsl = gsl_vector_alloc(num_of_parameters);
    }
}

int	getdimension()
{
	return num_of_parameters;
}
void    getmargins(vector<Interval> &x)
{
        for(int i=0;i<x.size();i++)
                x[i]=Interval(0,2*M_PI);
}


//f(x)
double	funmin(vector<double> &x)
{
   //return decomp->optimization_problem(x.data());
   Matrix_real parameters_mtx(x.data(), 1, x.size());
   return decomp->optimization_problem(parameters_mtx);
}

//f'(x)
void    granal(vector<double> &x,vector<double> &g)
{
    gsl_vector gv = { x.size(), sizeof(double), x.data(), NULL, 0 };
    decomp->optimization_problem_grad(&gv, (void*)decomp, grad_gsl);
    for (size_t i = 0; i < num_of_parameters; i++) {
        g[i] = gsl_vector_get(grad_gsl, i);
    }
}

QJsonObject    done(vector<double> &x)
{
    //printf("done\n");
    /*gsl_vector_free(grad_gsl);
    delete decomp;
    return QJsonObject();*/
}
}
