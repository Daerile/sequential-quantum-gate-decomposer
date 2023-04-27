# include <math.h>
# include <interval.h>
# include <vector>
# include <stdio.h>
# include <iostream>
# include <QJsonDocument>
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
N_Qubit_Decomposition_adaptive* decomp = NULL;
vector<double> min_params;
double cur_min;

void    init(QJsonObject data)
{
    //printf("%s\n", QJsonDocument(data).toJson(QJsonDocument::Compact).toStdString().c_str());
    if (decomp == NULL) {
        string folder = data.take("folder").toString().toStdString();
        string qasm = data.take("qasm").toString().toStdString();        
        int qbit_num, accelerator_num = 0, level_limit=data.take("levels").toString().toInt(), level_limit_min=3;
        if (system((std::string("python ../../sequential-quantum-gate-decomposer/saveunitary.py ") + folder + qasm + ".qasm ./" + qasm + ".binary").c_str()) != 0) {
            exit(-1);
        }        
        std::string filename = qasm + ".binary";
        Matrix Umtx = Decomposition_Base().import_unitary_from_binary(filename);
        qbit_num = log2(Umtx.rows);
        //Matrix Umtx = create_identity( 1 << qbit_num);
        decomp = new N_Qubit_Decomposition_adaptive( Umtx, qbit_num, level_limit, level_limit_min, accelerator_num );
        //Matrix_real optimized_parameters_mtx_loc;
        //Gates_block* gate_structure_loc = decomp->determine_initial_gate_structure(optimized_parameters_mtx_loc);
        //decomp->combine( gate_structure_loc );
        for (int idx=0; idx<level_limit; idx++ ) {
              decomp->add_adaptive_layers();  
        }
        decomp->add_finalyzing_layer();
        min_params.resize(decomp->get_parameter_num());
        cur_min = 1.0;
    }
}

int	getdimension()
{
	return decomp->get_parameter_num();
}
void    getmargins(vector<Interval> &x)
{
        for(int i=0;i<x.size();i++)
                x[i]=Interval(0,2*M_PI);
}


//f(x)
double	funmin(vector<double> &x)
{
   return decomp->optimization_problem(x.data());
   //Matrix_real parameters_mtx(x.data(), 1, x.size());
   //return decomp->optimization_problem(parameters_mtx);
}

//f'(x)
void    granal(vector<double> &x,vector<double> &g)
{
    gsl_vector* grad_gsl = gsl_vector_alloc(decomp->get_parameter_num());
    gsl_vector gv = { x.size(), sizeof(double), x.data(), NULL, 0 };
    decomp->optimization_problem_grad(&gv, (void*)decomp, grad_gsl);
    for (size_t i = 0; i < decomp->get_parameter_num(); i++) {
        g[i] = gsl_vector_get(grad_gsl, i);
    }
    gsl_vector_free(grad_gsl);
}

QJsonObject    done(vector<double> &x)
{
   double min = decomp->optimization_problem(x.data());
   if (min < cur_min) {
       cur_min = min;
       min_params.clear();
       copy(x.begin(), x.end(), back_inserter(min_params));
       printf("New Min: %f\n", cur_min);
   }
    //printf("done\n");
    /*
    delete decomp;
    decomp = NULL;*/
    return QJsonObject();
}
}
