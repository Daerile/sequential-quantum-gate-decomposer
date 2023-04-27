/*
Created on Fri Jun 26 14:42:56 2020
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
/*
\file qgd_CNOT.cpp
\brief Python interface for the CNOT gate class
*/

#define PY_SSIZE_T_CLEAN


#include <Python.h>
#include "structmember.h"
#include "CNOT.h"
#include "numpy_interface.h"



/**
@brief Type definition of the  qgd_CNOT Python class of the  qgd_CNOT module
*/
typedef struct {
    PyObject_HEAD
    /// Pointer to the C++ class of the CNOT gate
    CNOT* gate;
} qgd_CNOT;


/**
@brief Creates an instance of class N_Qubit_Decomposition and return with a pointer pointing to the class instance (C++ linking is needed)
@param qbit_num Number of qubits spanning the unitary
@param target_qbit The Id (0<= target_qbit < qbit_num ) of the target qubit.
@param control_qbit The Id (0<= control_qbit < qbit_num ) of the control qubit.
@return Return with a void pointer pointing to an instance of N_Qubit_Decomposition class.
*/
CNOT* 
create_CNOT( int qbit_num, int target_qbit, int control_qbit ) {

    return new CNOT( qbit_num, target_qbit, control_qbit );
}


/**
@brief Call to deallocate an instance of N_Qubit_Decomposition class
@param ptr A pointer pointing to an instance of N_Qubit_Decomposition class.
*/
void
release_CNOT( CNOT*  instance ) {
    delete instance;
    return;
}




extern "C"
{


/**
@brief Method called when a python instance of the class  qgd_CNOT is destroyed
@param self A pointer pointing to an instance of class  qgd_CNOT.
*/
static void
 qgd_CNOT_dealloc(qgd_CNOT *self)
{
    release_CNOT( self->gate );

    Py_TYPE(self)->tp_free((PyObject *) self);
}


/**
@brief Method called when a python instance of the class  qgd_CNOT is allocated
@param type A pointer pointing to a structure describing the type of the class  qgd_CNOT.
*/
static PyObject *
 qgd_CNOT_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    qgd_CNOT *self;
    self = (qgd_CNOT *) type->tp_alloc(type, 0);
    if (self != NULL) {}
    return (PyObject *) self;
}


/**
@brief Method called when a python instance of the class  qgd_CNOT is initialized
@param self A pointer pointing to an instance of the class  qgd_CNOT.
@param args A tuple of the input arguments: qbit_num (int), target_qbit (int), control_qbit (int)
@param kwds A tuple of keywords
*/
static int
 qgd_CNOT_init(qgd_CNOT *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {(char*)"qbit_num", (char*)"target_qbit", (char*)"control_qbit", NULL};
    int  qbit_num = -1; 
    int target_qbit = -1;
    int control_qbit = -1;

    if (PyArray_API == NULL) {
        import_array();
    }

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|iii", kwlist,
                                     &qbit_num, &target_qbit, &control_qbit))
        return -1;

    if (qbit_num != -1 && target_qbit != -1) {
        self->gate = create_CNOT( qbit_num, target_qbit, control_qbit );
    }
    return 0;
}

/**
@brief Extract the optimized parameters
@param start_index The index of the first inverse gate
*/
static PyObject *
qgd_CNOT_get_Matrix( qgd_CNOT *self ) {

    Matrix CNOT_mtx = self->gate->get_matrix(  );
    
    // convert to numpy array
    CNOT_mtx.set_owner(false);
    PyObject *CNOT_py = matrix_to_numpy( CNOT_mtx );

    return CNOT_py;
}



/**
@brief Call to apply the gate operation on the inut matrix
*/
static PyObject *
qgd_CNOT_apply_to( qgd_CNOT *self, PyObject *args ) {

    PyObject * unitary_arg = NULL;



    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|O", &unitary_arg )) 
        return Py_BuildValue("i", -1);

    // convert python object array to numpy C API array
    if ( unitary_arg == NULL ) {
        PyErr_SetString(PyExc_Exception, "Input matrix was not given");
        return NULL;
    }


    if ( PyArray_Check(unitary_arg) && PyArray_IS_C_CONTIGUOUS(unitary_arg) ) {
        Py_INCREF(unitary_arg);
    }
    else {
        unitary_arg = PyArray_FROM_OTF(unitary_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    }

   PyObject* unitary = PyArray_FROM_OTF(unitary_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);

    // test C-style contiguous memory allocation of the array
    if ( !PyArray_IS_C_CONTIGUOUS(unitary) ) {
        PyErr_SetString(PyExc_Exception, "input mtrix is not memory contiguous");
        return NULL;
    }


    // create QGD version of the input matrix
    Matrix unitary_mtx = numpy2matrix(unitary);

    self->gate->apply_to( unitary_mtx );
    
    Py_DECREF(unitary);

    return Py_BuildValue("i", 0);
}


/**
@brief Calculate the matrix of a U3 gate gate corresponding to the given parameters acting on a single qbit space.
@param ThetaOver2 Real parameter standing for the parameter theta.
@param Phi Real parameter standing for the parameter phi.
@param Lambda Real parameter standing for the parameter lambda.
@return Returns with the matrix of the one-qubit matrix.
*/

static PyObject *
qgd_CNOT_get_Gate_Kernel( qgd_CNOT *self, PyObject *args ) {

    double ThetaOver2;
    double Phi; 
    double Lambda; 

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|ddd", &ThetaOver2, &Phi, &Lambda )) 
        return Py_BuildValue("i", -1);


    // create QGD version of the input matrix

    Matrix CNOT_1qbit_ = self->gate->calc_one_qubit_u3(ThetaOver2, Phi, Lambda );
    PyObject *CNOT_1qbit = matrix_to_numpy( CNOT_1qbit_ );

    return CNOT_1qbit;
}



/**
@brief Structure containing metadata about the members of class  qgd_CNOT.
*/
static PyMemberDef  qgd_CNOT_members[] = {
    {NULL}  /* Sentinel */
};


/**
@brief Structure containing metadata about the methods of class  qgd_CNOT.
*/
static PyMethodDef  qgd_CNOT_methods[] = {
    {"get_Matrix", (PyCFunction) qgd_CNOT_get_Matrix, METH_NOARGS,
     "Method to get the matrix of the operation."
    },
    {"apply_to", (PyCFunction) qgd_CNOT_apply_to, METH_VARARGS,
     "Call to apply the gate on the input matrix."
    },
    {"get_Gate_Kernel", (PyCFunction) qgd_CNOT_get_Gate_Kernel, METH_VARARGS,
     "Call to calculate the gate matrix acting on a single qbit space."
    },
    {NULL}  /* Sentinel */
};


/**
@brief A structure describing the type of the class  qgd_CNOT.
*/
static PyTypeObject  qgd_CNOT_Type = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "qgd_CNOT.qgd_CNOT", /*tp_name*/
  sizeof(qgd_CNOT), /*tp_basicsize*/
  0, /*tp_itemsize*/
  (destructor)  qgd_CNOT_dealloc, /*tp_dealloc*/
  #if PY_VERSION_HEX < 0x030800b4
  0, /*tp_print*/
  #endif
  #if PY_VERSION_HEX >= 0x030800b4
  0, /*tp_vectorcall_offset*/
  #endif
  0, /*tp_getattr*/
  0, /*tp_setattr*/
  #if PY_MAJOR_VERSION < 3
  0, /*tp_compare*/
  #endif
  #if PY_MAJOR_VERSION >= 3
  0, /*tp_as_async*/
  #endif
  0, /*tp_repr*/
  0, /*tp_as_number*/
  0, /*tp_as_sequence*/
  0, /*tp_as_mapping*/
  0, /*tp_hash*/
  0, /*tp_call*/
  0, /*tp_str*/
  0, /*tp_getattro*/
  0, /*tp_setattro*/
  0, /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
  "Object to represent a CNOT gate of the QGD package.", /*tp_doc*/
  0, /*tp_traverse*/
  0, /*tp_clear*/
  0, /*tp_richcompare*/
  0, /*tp_weaklistoffset*/
  0, /*tp_iter*/
  0, /*tp_iternext*/
   qgd_CNOT_methods, /*tp_methods*/
   qgd_CNOT_members, /*tp_members*/
  0, /*tp_getset*/
  0, /*tp_base*/
  0, /*tp_dict*/
  0, /*tp_descr_get*/
  0, /*tp_descr_set*/
  0, /*tp_dictoffset*/
  (initproc)  qgd_CNOT_init, /*tp_init*/
  0, /*tp_alloc*/
   qgd_CNOT_new, /*tp_new*/
  0, /*tp_free*/
  0, /*tp_is_gc*/
  0, /*tp_bases*/
  0, /*tp_mro*/
  0, /*tp_cache*/
  0, /*tp_subclasses*/
  0, /*tp_weaklist*/
  0, /*tp_del*/
  0, /*tp_version_tag*/
  #if PY_VERSION_HEX >= 0x030400a1
  0, /*tp_finalize*/
  #endif
  #if PY_VERSION_HEX >= 0x030800b1
  0, /*tp_vectorcall*/
  #endif
  #if PY_VERSION_HEX >= 0x030800b4 && PY_VERSION_HEX < 0x03090000
  0, /*tp_print*/
  #endif
};


/**
@brief Structure containing metadata about the module.
*/
static PyModuleDef  qgd_CNOT_Module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "qgd_CNOT",
    .m_doc = "Python binding for QGD CNOT gate",
    .m_size = -1,
};


/**
@brief Method called when the Python module is initialized
*/
PyMODINIT_FUNC
PyInit_qgd_CNOT(void)
{
    PyObject *m;
    if (PyType_Ready(& qgd_CNOT_Type) < 0)
        return NULL;

    m = PyModule_Create(& qgd_CNOT_Module);
    if (m == NULL)
        return NULL;

    Py_INCREF(& qgd_CNOT_Type);
    if (PyModule_AddObject(m, "qgd_CNOT", (PyObject *) & qgd_CNOT_Type) < 0) {
        Py_DECREF(& qgd_CNOT_Type);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}



}
