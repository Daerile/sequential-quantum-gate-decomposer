# -*- coding: utf-8 -*-
"""
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
"""
## \file example_get_circuit_unitary.py
## \brief Simple example python code demonstrating the basic usage of the Python interface of the Quantum Gate Decomposer package

## [import adaptive]
from qgd_python.decomposition.qgd_N_Qubit_Decomposition_adaptive import qgd_N_Qubit_Decomposition_adaptive       
## [import adaptive]

import numpy as np
import random


# The gate stucture created by the adaptive decomposition class reads as:
# [ || U3, U3, ..., U3, || CRY, U3, U3 || CRY, U3, U3 || CRY, U3, U3 || .... ]
# The individual blocks are separated by ||. Each U3 gate has 3 free parameters, the CRY gates have 1 free parameter
# The first qbit_num gates are U3 transformations on each qubit. 
# In the quantum circuit the operations on the unitary Umtx are performed in the following order:
# U3*U3*...*U3 * CRY*U3*U3 * CRY*U3*U3 * CRY*U3*U3 * ... * CRY*U3*U3 * Umtx


# the ratio of nontrivial 2-qubit building blocks
nontrivial_ratio = 0.5

# number of qubits
qbit_num = 3

# matrix size of the unitary
matrix_size = pow(2, qbit_num )

# number of adaptive levels
levels = 5


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

# creating a class to decompose the unitary
cDecompose = qgd_N_Qubit_Decomposition_adaptive( np.eye(matrix_size), level_limit_max=5, level_limit_min=0 )

# adding decomposing layers to the gat structure
for idx in range(levels):
    cDecompose.add_Adaptive_Layers()

cDecompose.add_Finalyzing_Layer_To_Gate_Structure()


# get the number of free parameters
num_of_parameters = cDecompose.get_Parameter_Num()

# create randomized parameters having number of nontrivial adaptive blocks determined by the parameter nontrivial_ratio
parameters, adaptive_layer_indices = create_randomized_parameters( qbit_num, num_of_parameters )

# getting the unitary corresponding to quantum circuit
unitary = cDecompose.get_Matrix( parameters )

cDecompose.set_Optimized_Parameters(parameters)

gates = cDecompose.List_Gates()
#print(gates)
#print( parameters )
#print( unitary )

def get_gate_structure(levels, qbit_num):
    target_qbits, control_qbits = [], []
    for _ in range(levels):        
        #adaptive blocks
        for target_qubit in range(qbit_num):
            for control_qubit in range(target_qubit+1, qbit_num, 1):
                target_qbits.extend([target_qubit, control_qubit, target_qubit])
                control_qbits.extend([target_qubit, control_qubit, control_qubit])
    # U3 gates
    for target_qubit in range(qbit_num):
        target_qbits.append(target_qubit)
        control_qbits.append(target_qubit)
    return np.array(target_qbits, dtype=np.uint8), np.array(control_qbits, dtype=np.uint8)

def transformer_model(num_parameters):
    import tensorflow as tf
    import numpy as np
    from tensorflow.keras.layers import LayerNormalization, Layer, Dense, ReLU, Dropout, Embedding
    from tensorflow import math, matmul, reshape, shape, transpose, cast, float32
    from tensorflow.keras.layers import Dense, Layer
    from keras.backend import softmax
    
    class PositionEmbeddingFixedWeights(Layer):
        def __init__(self, sequence_length, vocab_size, output_dim, **kwargs):
            super(PositionEmbeddingFixedWeights, self).__init__(**kwargs)
            word_embedding_matrix = self.get_position_encoding(vocab_size, output_dim)   
            position_embedding_matrix = self.get_position_encoding(sequence_length, output_dim)                                          
            self.word_embedding_layer = Embedding(
                input_dim=vocab_size, output_dim=output_dim,
                weights=[word_embedding_matrix],
                trainable=False
            )
            self.position_embedding_layer = Embedding(
                input_dim=sequence_length, output_dim=output_dim,
                weights=[position_embedding_matrix],
                trainable=False
            )
                 
        def get_position_encoding(self, seq_len, d, n=10000):
            P = np.zeros((seq_len, d))
            for k in range(seq_len):
                for i in np.arange(int(d/2)):
                    denominator = np.power(n, 2*i/d)
                    P[k, 2*i] = np.sin(k/denominator)
                    P[k, 2*i+1] = np.cos(k/denominator)
            return P
     
     
        def call(self, inputs):        
            position_indices = tf.range(tf.shape(inputs)[-1])
            embedded_words = self.word_embedding_layer(inputs)
            embedded_indices = self.position_embedding_layer(position_indices)
            return embedded_words + embedded_indices
     
    # Implementing the Scaled-Dot Product Attention
    class DotProductAttention(Layer):
        def __init__(self, **kwargs):
            super(DotProductAttention, self).__init__(**kwargs)
     
        def call(self, queries, keys, values, d_k, mask=None):
            # Scoring the queries against the keys after transposing the latter, and scaling
            scores = matmul(queries, keys, transpose_b=True) / math.sqrt(cast(d_k, float32))
     
            # Apply mask to the attention scores
            if mask is not None:
                scores += -1e9 * mask
     
            # Computing the weights by a softmax operation
            weights = softmax(scores)
     
            # Computing the attention by a weighted sum of the value vectors
            return matmul(weights, values)
     
    # Implementing the Multi-Head Attention
    class MultiHeadAttention(Layer):
        def __init__(self, h, d_k, d_v, d_model, **kwargs):
            super(MultiHeadAttention, self).__init__(**kwargs)
            self.attention = DotProductAttention()  # Scaled dot product attention
            self.heads = h  # Number of attention heads to use
            self.d_k = d_k  # Dimensionality of the linearly projected queries and keys
            self.d_v = d_v  # Dimensionality of the linearly projected values
            self.d_model = d_model  # Dimensionality of the model
            self.W_q = Dense(d_k)  # Learned projection matrix for the queries
            self.W_k = Dense(d_k)  # Learned projection matrix for the keys
            self.W_v = Dense(d_v)  # Learned projection matrix for the values
            self.W_o = Dense(d_model)  # Learned projection matrix for the multi-head output
     
        def reshape_tensor(self, x, heads, flag):
            if flag:
                # Tensor shape after reshaping and transposing: (batch_size, heads, seq_length, -1)
                x = reshape(x, shape=(shape(x)[0], shape(x)[1], heads, -1))
                x = transpose(x, perm=(0, 2, 1, 3))
            else:
                # Reverting the reshaping and transposing operations: (batch_size, seq_length, d_k)
                x = transpose(x, perm=(0, 2, 1, 3))
                x = reshape(x, shape=(shape(x)[0], shape(x)[1], self.d_k))
            return x
     
        def call(self, queries, keys, values, mask=None):
            # Rearrange the queries to be able to compute all heads in parallel
            q_reshaped = self.reshape_tensor(self.W_q(queries), self.heads, True)
            # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)
     
            # Rearrange the keys to be able to compute all heads in parallel
            k_reshaped = self.reshape_tensor(self.W_k(keys), self.heads, True)
            # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)
     
            # Rearrange the values to be able to compute all heads in parallel
            v_reshaped = self.reshape_tensor(self.W_v(values), self.heads, True)
            # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)
     
            # Compute the multi-head attention output using the reshaped queries, keys and values
            o_reshaped = self.attention(q_reshaped, k_reshaped, v_reshaped, self.d_k, mask)
            # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)
     
            # Rearrange back the output into concatenated form
            output = self.reshape_tensor(o_reshaped, self.heads, False)
            # Resulting tensor shape: (batch_size, input_seq_length, d_v)
     
            # Apply one final linear projection to the output to generate the multi-head attention
            # Resulting tensor shape: (batch_size, input_seq_length, d_model)
            return self.W_o(output)
     
    # Implementing the Add & Norm Layer
    class AddNormalization(Layer):
        def __init__(self, **kwargs):
            super(AddNormalization, self).__init__(**kwargs)
            self.layer_norm = LayerNormalization()  # Layer normalization layer
     
        def call(self, x, sublayer_x):
            # The sublayer input and output need to be of the same shape to be summed
            add = x + sublayer_x
     
            # Apply layer normalization to the sum
            return self.layer_norm(add)
     
    # Implementing the Feed-Forward Layer
    class FeedForward(Layer):
        def __init__(self, d_ff, d_model, **kwargs):
            super(FeedForward, self).__init__(**kwargs)
            self.fully_connected1 = Dense(d_ff)  # First fully connected layer
            self.fully_connected2 = Dense(d_model)  # Second fully connected layer
            self.activation = ReLU()  # ReLU activation layer
     
        def call(self, x):
            # The input is passed into the two fully-connected layers, with a ReLU in between
            x_fc1 = self.fully_connected1(x)
     
            return self.fully_connected2(self.activation(x_fc1))
     
    # Implementing the Encoder Layer
    class EncoderLayer(Layer):
        def __init__(self, h, d_k, d_v, d_model, d_ff, rate, **kwargs):
            super(EncoderLayer, self).__init__(**kwargs)
            self.multihead_attention = MultiHeadAttention(h, d_k, d_v, d_model)
            self.dropout1 = Dropout(rate)
            self.add_norm1 = AddNormalization()
            self.feed_forward = FeedForward(d_ff, d_model)
            self.dropout2 = Dropout(rate)
            self.add_norm2 = AddNormalization()
     
        def call(self, x, padding_mask, training):
            # Multi-head attention layer
            multihead_output = self.multihead_attention(x, x, x, padding_mask)
            # Expected output shape = (batch_size, sequence_length, d_model)
     
            # Add in a dropout layer
            multihead_output = self.dropout1(multihead_output, training=training)
     
            # Followed by an Add & Norm layer
            addnorm_output = self.add_norm1(x, multihead_output)
            # Expected output shape = (batch_size, sequence_length, d_model)
     
            # Followed by a fully connected layer
            feedforward_output = self.feed_forward(addnorm_output)
            # Expected output shape = (batch_size, sequence_length, d_model)
     
            # Add in another dropout layer
            feedforward_output = self.dropout2(feedforward_output, training=training)
     
            # Followed by another Add & Norm layer
            return self.add_norm2(addnorm_output, feedforward_output)
     
    # Implementing the Encoder
    class Encoder(Layer):
        def __init__(self, vocab_size, sequence_length, h, d_k, d_v, d_model, d_ff, n, rate, num_params, **kwargs):
            super(Encoder, self).__init__(**kwargs)
            self.pos_encoding = PositionEmbeddingFixedWeights(sequence_length, vocab_size, d_model)
            self.dropout = Dropout(rate)
            self.encoder_layer = [EncoderLayer(h, d_k, d_v, d_model, d_ff, rate) for _ in range(n)]
            self.output_layer = tf.keras.layers.Dense(num_params)
            self.flatten = Flatten()     
        def call(self, input_sentence, padding_mask, training):
            # Generate the positional encoding
            pos_encoding_output = self.pos_encoding(input_sentence)
            # Expected output shape = (batch_size, sequence_length, d_model)
     
            # Add in a dropout layer
            x = self.dropout(pos_encoding_output, training=training)
     
            # Pass on the positional encoded values to each encoder layer
            for i, layer in enumerate(self.encoder_layer):
                x = layer(x, padding_mask, training)
            x = self.flatten(x)
            x = self.output_layer(x)     
            return x
            
    from numpy import random
     
    enc_vocab_size = 20 # Vocabulary size for the encoder
    input_seq_length = 5  # Maximum length of the input sequence
    h = 8  # Number of self-attention heads
    d_k = 64  # Dimensionality of the linearly projected queries and keys
    d_v = 64  # Dimensionality of the linearly projected values
    d_ff = 2048  # Dimensionality of the inner fully connected layer
    d_model = 512  # Dimensionality of the model sub-layers' outputs
    n = 6  # Number of layers in the encoder stack
     
    batch_size = 64  # Batch size from the training process
    dropout_rate = 0.1  # Frequency of dropping the input units in the dropout layers
     
    #input_seq = random.random((batch_size, input_seq_length))
     
    encoder = Encoder(enc_vocab_size, input_seq_length, h, d_k, d_v, d_model, d_ff, n, dropout_rate, num_parameters)
    model = tf.keras.models.Sequential([encoder])
    #print(encoder(input_seq, None, True))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[tf.keras.metrics.RootMeanSquaredError(),'mae'])
    model.fit(x_train, {'out' + str(i): y_train[:,i] for i in real_params}, epochs=10)
def train_model(num_of_parameters, size):
    import tensorflow as tf #pip install tensorflow-cpu
    inputs = tf.keras.layers.Input(shape = unitary.shape, dtype=tf.complex64) # + (2,))
    #x = tf.keras.layers.Flatten(input_shape=unitary.shape)(inputs)
    #tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True), backward_layer=tf.keras.layers.LSTM(128, activation='relu', return_sequences=True, go_backwards=True)),
    #x = tf.keras.layers.Conv2D(16, 3, activation='relu', input_shape=inputs.shape)(inputs)
    #x = tf.keras.layers.MaxPooling2D()(x)
    class ReverseGateDenseLayer(tf.keras.layers.Layer):
      def __init__(self, num_outputs, target_qbit, control_qbit):
        super(ReverseGateDenseLayer, self).__init__()
        self.num_outputs = num_outputs
        self.target_qbit = target_qbit
        self.control_qbit = control_qbit
      def build(self, input_shape):
        self.pow2qb = int(input_shape[-1])
        self.num_qbits = qbit_num
        self.kernel = self.add_weight("kernel",
                                      shape=[1 if self.target_qbit != self.control_qbit else 3])#, self.num_outputs])
      def call(self, inputs):
        z = tf.constant([0], dtype=self.kernel.dtype)  
        if self.target_qbit != self.control_qbit:
            c = tf.complex(tf.math.cos(self.kernel), z)
            s = tf.complex(tf.math.sin(self.kernel), z) 
            g = tf.stack([tf.concat([c, tf.math.negative(s)], 0),            
                          tf.concat([s, c], 0)], 0)
        else:
            kcells = [tf.gather_nd(self.kernel, [0]), tf.gather_nd(self.kernel, [1]), tf.gather_nd(self.kernel, [2])]
            c = tf.complex(tf.math.cos(kcells[0]), z)
            s = tf.complex(tf.math.sin(kcells[0]), z)
            kcells[1] = tf.complex(z, kcells[1])
            kcells[2] = tf.complex(z, kcells[2])
            g = tf.stack([tf.concat([c, tf.math.negative(tf.multiply(tf.math.exp(kcells[2]), s))], 0),
                          tf.concat([tf.multiply(tf.math.exp(kcells[1]), s), tf.multiply(tf.math.exp(kcells[1]+kcells[2]), c)], 0)], 0)
        outputs = tf.identity(inputs)
        pow2qb = 1 << self.num_qbits
        t = np.roll(np.arange(self.num_qbits), self.target_qbit)
        idxs = np.arange(pow2qb).reshape(*([2]*self.num_qbits)).transpose(t).reshape(-1, 2)
        for pair in (idxs if self.control_qbit is None else idxs[(idxs[:,0] & (1<<self.control_qbit)) != 0,:]):
            tfpair = [[0, pair[0]], [0, pair[1]]]
            update = tf.matmul(g, tf.gather_nd(outputs, tfpair))
            tf.tensor_scatter_nd_update(outputs, tfpair, update)
        return outputs
    #x = tf.keras.layers.Reshape((-1, 2))(inputs)
    #x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=False, return_state=False))(x)
    x = inputs#x = tf.keras.layers.Dropout(0.2)(inputs)
    #x = tf.keras.layers.Flatten()(x)
    #x = tf.keras.layers.Dense(32, activation='tanh')(x)
    #x = tf.keras.layers.Dense(64, activation='tanh')(x)
    #x = tf.keras.layers.Dense(128, activation='tanh')(x)
    #x = tf.keras.layers.Dense(256, activation='tanh')(x)
    #x = tf.keras.layers.Dense(512, activation='tanh')(x)
    target_qbits, control_qbits = get_gate_structure(levels, qbit_num)
    for t, c in zip(reversed(target_qbits), reversed(control_qbits)):
        x = ReverseGateDenseLayer(1, t, c)(x) 
    x = tf.keras.layers.Dropout(0.2)(x)
    real_params = list(np.where(parameters != 0)[0])
    print(len(real_params), real_params)
    outputs = [tf.keras.layers.Dense(1, name='out' + str(i))(x) for i in real_params]
    model = tf.keras.Model(inputs = inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[tf.keras.metrics.RootMeanSquaredError(),'mae'])
    x_train, y_train = np.empty((size*4//5, *unitary.shape), dtype=unitary.dtype), np.empty((size*4//5, num_of_parameters), dtype=parameters.dtype)
    for i in range(size*4//5):
        params, _ = create_randomized_parameters(qbit_num, num_of_parameters)
        x_train[i,:] = cDecompose.get_Matrix(params)
        y_train[i,:] = params / (2*np.pi)
    #model.fit(x_train.view(np.float64).reshape(x_train.shape + (2,)), {'out' + str(i): y_train[:,i] for i in real_params}, epochs=10)
    model.fit(x_train, {'out' + str(i): y_train[:,i] for i in real_params}, epochs=10)
    x_test, y_test = np.empty((size//5, *unitary.shape), dtype=unitary.dtype), np.empty((size//5, num_of_parameters), dtype=parameters.dtype)
    for i in range(size//5):
        params, _ = create_randomized_parameters(qbit_num, num_of_parameters)
        x_test[i,:] = cDecompose.get_Matrix(params)
        y_test[i,:] = params / (2*np.pi)
    model.evaluate(x_test.view(np.float64).reshape(x_test.shape + (2,)), {'out' + str(i): y_test[:,i] for i in real_params})
    print(model.predict(x_test.view(np.float64).reshape(x_test.shape + (2,)))[:5], y_test[:5,real_params])
def backtracking_line_search(f, xn, delta, oldmn, m): #compute learning rate by backtracking line search
    c, tau, alpha = 0.5, 0.5, 1.0
    #assert m < 0
    t = -c * m
    while True:
        ftry = f(xn + delta * alpha)
        if oldmn - ftry >= alpha * t: return alpha, ftry        
        alpha *= tau
def ilb(xn, cost, f, ops, D, k):
    import bisect
    b = len(ops)
    for exp in range(1, D+1):
        breadthLimit = k + b**exp
        newOpen = [(cost, 0, xn)]; idx = 1
        for d in range(1, D+1):
            print(exp, d)
            opn, newOpen = newOpen, []
            for node in opn:
                for op in ops:
                    newx = op(node[2])
                    tryf = f(newx)
                    if tryf < cost:
                        return newx, tryf
                    else:
                        bisect.insort(newOpen, (tryf, idx, newx)); idx += 1
                        if len(newOpen) > breadthLimit: newOpen.pop()
def mbfgs(x, f, g):
    B = np.eye(x.shape[0], dtype=x.dtype)
    #B = np.random.random((x.shape[0], x.shape[0]))
    #B = B@B.T + x.shape[0]*np.eye(x.shape[0], dtype=x.dtype)
    phi, rho = 1e-4, 0.8
    k = 0
    cost, grad = f(x), g(x)
    initcost = cost
    while True:
        p = np.linalg.solve(B, -grad)
        lbda = 1.0
        crateoverall, crate = (initcost-cost)/max(1,k), np.linalg.norm(grad)
        if crateoverall < 10e-5: phi = max(0.99, phi*2)
        print("Cost: ", cost, "Convergence rate: ", crateoverall, crate)
        gradp = phi*np.dot(grad, p)
        while True: #backtracking line search
            s = lbda*p
            newx = x+s
            newcost = f(newx)
            if newcost <= cost + lbda*gradp:
                cost, x = newcost, newx % (2*np.pi)
                break
            lbda *= rho
        newgrad = g(x)
        gamma = newgrad-grad
        t = 1 + max(-np.dot(gamma, s)/np.square(np.linalg.norm(s)), 0)
        y = gamma + t*np.linalg.norm(grad)*s #y = gamma + C*s
        B = B - (B@np.outer(s, s)@B)/np.dot(s, B @ s) + np.outer(y, y)/np.dot(y, s)
        #assert np.all(np.abs(B-B.T) < 1e-8)
        grad = newgrad
        k += 1
    
def newton_method():
    from qiskit import QuantumCircuit, transpile
    from qgd_python.utils import get_unitary_from_qiskit_circuit
    import sys
    filename = "/home/morse/ibm_qx_mapping/examples/" + 'one-two-three-v0_98' + ".qasm" # + 'ham3_102' + ".qasm"
    qc_trial = QuantumCircuit.from_qasm_file( filename )
    qc_trial = transpile(qc_trial, optimization_level=3, basis_gates=['cz', 'cx', 'u3'], layout_method='sabre')
    Umtx_orig = get_unitary_from_qiskit_circuit( qc_trial )
    from qgd_python.decomposition.qgd_N_Qubit_Decomposition_adaptive import qgd_N_Qubit_Decomposition_adaptive
    cDecompose = qgd_N_Qubit_Decomposition_adaptive( Umtx_orig.conj().T, level_limit_max=5, level_limit_min=0 )
    cDecompose.set_Cost_Function_Variant(0)
    # adding decomposing layers to the gat structure
    levels = 5
    for idx in range(levels):
        cDecompose.add_Adaptive_Layers()
    cDecompose.add_Finalyzing_Layer_To_Gate_Structure()
    # get the number of free parameters
    batchsize = 100
    num_of_parameters = cDecompose.get_Parameter_Num()
    lbda = 0.1
    def sample():
        return np.random.random(num_of_parameters)*2*np.pi #np.ones(num_of_parameters, dtype=np.float64)*np.pi/2
    def fourier_coeffs_mc(f, domain, num_samples):
        # f: function to be approximated
        # k: list of tuples representing the frequencies for each dimension, e.g. [(1, 0), (0, 1)] for a 2D function with frequencies (1, 0) and (0, 1)
        # domain: list of tuples representing the lower and upper bounds for each dimension, e.g. [(0, 1), (0, 1)] for a 2D function with domain [0, 1] x [0, 1]
        # num_samples: number of random samples to generate
        k = np.fft.fftfreq(num_samples, d=2*np.pi / num_samples)
        d = len(domain)  # number of dimensions
        c = np.zeros(len(k), dtype=np.complex128)  # initialize Fourier coefficients
    
        # generate random samples with importance weights
        samples = np.zeros((num_samples, d))
        weights = np.zeros(num_samples)
        for i in range(num_samples):
            x = [np.random.uniform(low=domain[j][0], high=domain[j][1]) for j in range(d)]
            w = 1
            for j in range(d):
                w *= np.exp(1j * freq * x[j])
                #w *= np.exp(-2 * (freq[0]**2 + freq[1]**2) / num_samples)  # importance weight
            samples[i] = x
            weights[i] = np.abs(w)
    
        # normalize weights
        weights /= np.sum(weights)
    
        # evaluate function at each sample point
        values = np.array([f(x) for x in samples])
    
        # compute Fourier coefficients for each frequency
        for i, freq in enumerate(k):
            # compute Fourier coefficient using Monte Carlo integration
            c[i] = np.sum(values * np.exp(-1j * freq[0] * samples[:, 0]) * np.exp(-1j * freq[1] * samples[:, 1]) * weights) / num_samples    
        return c
    def fourier_series(coeffs, k, x):
        # coeffs: list of Fourier coefficients for each frequency
        # k: list of tuples representing the frequencies for each dimension, e.g. [(1, 0), (0, 1)] for a 2D function with frequencies (1, 0) and (0, 1)
        # x: point at which to evaluate the Fourier series
    
        d = len(x)  # number of dimensions
        result = 0
    
        # compute Fourier series by summing over all frequencies
        for i, freq in enumerate(k):
            # compute weight factor for this frequency
            w = 1
            for j in range(d):
                w *= np.exp(1j * freq[j] * x[j])
    
            # add contribution of this frequency to the result
            result += coeffs[i] * w
    
        return result        
    xn = sample()
    #batch = np.random.random((batchsize, num_of_parameters)) * 2 * np.pi
    #print(cDecompose.Optimization_Problem_Batch(batch).shape)
    def f(xn, *args):
        return cDecompose.Optimization_Problem(xn)
        #mat = cDecompose.get_Matrix(xn)
        #print(cDecompose.Optimization_Problem(xn),costfunc(mat)) 
        #return costfunc(mat)
    def grad(xn, *args):
        return cDecompose.Optimization_Problem_Grad(xn)
    def costfunc(mat):
        #return np.sum(np.abs(mat-np.eye(mat.shape[1], dtype=mat.dtype)))
        return np.sum(np.square((mat-np.eye(mat.shape[1], dtype=mat.dtype)).view(np.float64)))
        #return 1.0-np.real(np.trace(mat))/mat.shape[1]
    from scipy.optimize import least_squares, minimize
    def fun(xn):
        mat = cDecompose.get_Matrix(xn)
        #return np.real(mat - np.eye(mat.shape[1], dtype=mat.dtype)).flatten()
        return (mat - np.eye(mat.shape[1], dtype=mat.dtype)).view(np.float64).flatten()
    def jac(xn):
        mat, mat_deriv = cDecompose.Optimization_Problem_Combined_Unitary(xn)
        #return np.real(np.stack(mat_deriv, -1)).reshape(mat.flatten().shape[0], num_of_parameters)
        mat = mat.view(np.float64).flatten()
        return np.stack(mat_deriv, -1).view(np.float64).reshape(mat.shape[0]//2, num_of_parameters, 2).transpose(0,2,1).reshape(mat.shape[0], num_of_parameters)
    #print(fourier_coeffs_mc(f, [[0]*num_of_parameters, [2*np.pi]*num_of_parameters], 100))
    #f = A*cos(p1)*cos(p2) + B*sin(p1)*sin(p2) + C*sin(p1)*cos(p2) + D*cos(p1)*sin(p2) + E*cos(p1) + F*cos(p2) + G*sin(p1) + H*sin(p2) + I
    #C*sin(p1)*cos(p2)+D*cos(p1)*sin(p2)+G*sin(p1) + H*sin(p2) + I
    #sin(p1)(C*cos(p2)+G) + sin(p2)(D*cos(p1)+H)+I
    def twoparammin(xn): #f(x_n)=D+A*sin(x1+x2)+B*sin(x1-x2)+C*sin(x1+x2)*sin(x1-x2)
        pass #f'x1=A*cos(x1+x2)+B*cos(x1-x2)+C*sin(x1+x2)*cos(x1-x2)+C*cos(x1+x2)*sin(x1-x2)
        #f'x2=A*cos(x1+x2)-B*cos(x1-x2)-C*sin(x1+x2)*cos(x1-x2)+C*cos(x1+x2)*sin(x1-x2)
        #0=A*cos(x1+x2)+C*cos(x1+x2)*sin(x1-x2)
    def oneparammin(xn): #f(x)=C+A*sin(x+phi)
        import itertools
        cDecompose.set_Cost_Function_Variant(0)
        cost = cDecompose.Optimization_Problem(xn)
        while cost >= 1e-8:
            shift, costs = np.zeros((len(xn),)), np.zeros((len(xn),))
            xnspi2 = np.repeat(xn[np.newaxis,...], len(xn), axis=0)
            xnspi = np.repeat(xn[np.newaxis,...], len(xn), axis=0)
            xnspi2[np.diag_indices_from(xnspi2)] += np.pi/2
            xnspi[np.diag_indices_from(xnspi)] += np.pi
            cs = cDecompose.Optimization_Problem_Batch(np.concatenate([xnspi2, xnspi]))
            grad = []
            for d in range(len(xn)):
                #f(x)=C+A*sin(x+phi) and f(x+PI/2)=C+A*cos(x+phi)
                #(f(x)-C)/(f(x+PI/2)-C)=tan(x+phi)
                c1 = cs[d]
                c2 = cs[d+len(xn)]                
                C = (cost+c2)/2
                truephi = np.arctan2(cost-C, c1-C)               
                truephi = (truephi-xn[d]) % (2 * np.pi) #sin(x+phi)=-1 3PI/2-phi
                shift[d] = 3*np.pi/2-truephi
                origsin = np.sin(xn[d]+truephi)
                A = (cost - c2) / 2 / origsin
                costs[d] = cost-A*(origsin+1)
                #sin(x+y) = A sin(x+phi)cos(y) + A cos(x+phi)sin(y)
                #f(x)=C+Asin(x+phi)
                #f(x+PI)=C-Asin(x+phi)
                #determine: f(x+y)=C+Asin(x+y+phi) or f(x+y)=C-A sin(x+y+phi)=-1  phi=3*PI/2-x-y
                #f(x+PI/2) = C+Asin(x+PI/2 + phi) = C+Acos(x+phi)
                #f(x+PI) = C+Asin(x+PI + phi) = C-Asin(x+phi)
                #f(x+3PI/2)= C+Asin(x+3PI/2 + phi)= C-Acos(x+phi)
                #print(cost, c1, c2, c3, costs[d])
                grad.append(np.linalg.norm(cDecompose.Optimization_Problem_Grad(xn)))
            minidx = max(filter(lambda x: cost-costs[x[0]]>1e-6, enumerate(grad)),
                key=lambda x: x[1],
                default=min(enumerate(costs), key=lambda x: x[1]))[0]
            #def grad(x):
            #    val = xn[x[0]] 
            #    xn[x[0]] = shift[x[0]]
            #    g = np.linalg.norm(cDecompose.Optimization_Problem_Grad(xn))
            #    xn[x[0]] = val
            #    return g
            #minidx = max(list(sorted(enumerate(costs), key=lambda x: x[1]))[:len(xn)//5], key=grad)[0]
            #if cost - costs[minidx] <= 1e-3:
            #    print(cost, costs[minidx])
            #    return xn
            """
            costs2 = []
            shift2 = []
            for c in itertools.combinations(range(len(xn)), 2):
                val1, val2 = xn[c[0]], xn[c[1]]
                xn[c[0]], xn[c[1]] = shift[c[0]], shift[c[1]]
                costs2.append(cDecompose.Optimization_Problem(xn))
                shift2.append(c)
                xn[c[0]], xn[c[1]] = val1, val2
            minidx2 = min(enumerate(costs2), key=lambda x: x[1])[0]
            """
            #minidx = min(enumerate(costs), key=lambda x: x[1])[0]            
            #if minidx2 < minidx:
            #    cost = costs2[minidx2]
            #    xn[shift2[minidx2][0]] = shift[shift2[minidx2][0]]
            #    xn[shift2[minidx2][1]] = shift[shift2[minidx2][1]]
            #else:
            #cost = costs[minidx]
            #xn[minidx] = shift[minidx]
            #minidxs = [y[0] for y in sorted(enumerate(costs), key=lambda x: x[1])]
            #minidx = minidxs[0]
            xn[minidx] = shift[minidx]
            cost = costs[minidx]
            #for minidx in minidxs[1:]:
            #    val = xn[minidx]
            #    xn[minidx] = shift[minidx]
            #    ftry = cDecompose.Optimization_Problem(xn)         
            #    if ftry < cost:
            #        cost = ftry; continue
            #    else: xn[minidx] = val
            print(cost, grad[minidx])#, cDecompose.Optimization_Problem(xn))#, np.linalg.norm(cDecompose.Optimization_Problem_Grad(xn)))
            break
        return xn
    #ret = least_squares(fun, xn, jac, (-np.inf, np.inf), verbose=2)
    #ret = minimize(f, xn, 'BFGS', jac=grad, options={'maxiter': 10000, 'disp': True})
    #ret = least_squares(fun, xn, jac, method='lm', verbose=2)
    #ret = least_squares(f, xn, grad, (0, 2*np.pi), verbose=2)
    #ret = least_squares(fun, xn, jac, (0, 2*np.pi), verbose=2)
    #print(f(ret.x)); assert False
    #mbfgs(xn, f, grad)
    xn = oneparammin(xn)
    #mbfgs(xn, f, grad)
    #https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm
    cDecompose.set_Cost_Function_Variant(6)
    while True:
        mat, mat_deriv = cDecompose.Optimization_Problem_Combined_Unitary(xn)
        cost = f(xn)
        print("Cost: ", cost, 1.0-np.real(np.trace(mat))/mat.shape[1], lbda)
        if cost < 1e-8: break
        mat = (mat - np.eye(mat.shape[1], dtype=mat.dtype)).view(np.float64).flatten()
        J = np.stack(mat_deriv, -1).view(np.float64).reshape(mat.shape[0]//2, num_of_parameters, 2).transpose(0,2,1).reshape(mat.shape[0], num_of_parameters)
        #squareidx = np.random.choice(J.shape[0], J.shape[1])
        #delta = np.linalg.solve(J[squareidx,:], -mat[squareidx])
        #Jplus = np.linalg.inv(J.T @ J) @ J.T
        #xn = xn - Jplus @ mat
        #mat = (mat - np.eye(mat.shape[1], dtype=mat.dtype)).flatten()
        #mat = mat(np.eye(mat.shape[1], dtype=mat.dtype) - mat).flatten()
        #J = np.stack(mat_deriv, 2).reshape(mat.shape[0], num_of_parameters)
        grad = J.T @ mat
        genJ = J.T @ J
        #https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm
        #damping = lbda*np.eye(num_of_parameters, dtype=J.dtype)
        damping = lbda*np.diag(np.diag(genJ))
        try:
            delta = np.linalg.lstsq(genJ - damping, -grad, rcond=None)[0]
            #delta = np.linalg.solve(genJ, -grad)
        except np.linalg.LinAlgError:
            print("Singular matrix", np.linalg.matrix_rank(genJ), num_of_parameters)
            delta = np.linalg.lstsq(genJ, -grad, rcond=None)[0]
            #xn = np.random.random(num_of_parameters)*2*np.pi
            #continue
        drctn = np.dot(2*grad, delta)
        if drctn < 0: lr, newc = backtracking_line_search(f, xn, delta, cost, drctn)
        if drctn < 0 and newc < cost:
            xn = (xn + delta * lr)# % (2 * np.pi)
            xn = oneparammin(xn)
            cDecompose.set_Cost_Function_Variant(6)
            if lbda > 1e-7: lbda = lbda / 10
        else:
            #xn = oneparammin(xn)
            #xn = np.random.random(num_of_parameters)*2*np.pi
            #continue
            #https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume8/finkelstein98a-html/node6.html
            inc = 2*np.pi/num_of_parameters
            def makeop():
                idxes = np.random.choice(num_of_parameters, num_of_parameters//2)
                mask = np.arange(len(xn))==idxes
                adj = (2*np.random.randint(0, 2, len(xn))-1)*inc
                return lambda x: np.where(mask, x+adj, x)
            cDecompose.set_Cost_Function_Variant(0)
            xn, newc = ilb(xn, f(xn), f, [makeop() for _ in range(len(xn))], 5, 20)
            cDecompose.set_Cost_Function_Variant(6)            
            continue
            while True:
                amount = np.random.randint(1, num_of_parameters+1)
                idxes = np.random.choice(num_of_parameters, amount)
                mask = np.zeros(num_of_parameters); mask[idxes] = True
                rand = np.zeros(num_of_parameters); rand[idxes] = xn[idxes]+np.pi/2
                newx = np.where(mask, rand, xn)
                tryf = f(newx)
                print(cost, tryf)
                if tryf < cost:
                    xn = newx
                    break
            if lbda < 0.1: lbda = lbda * 10
        #xn = np.real(xn + np.linalg.solve(J.conjugate().T @ J, -J.conjugate().T @ mat)) % (2 * np.pi)      
        #Jplus = np.linalg.inv(J.conjugate().T @ J) @ J.conjugate().T
        #xn = xn - np.real(Jplus @ mat) % (2 * np.pi)

from numba import njit
from numba.np.unsafe.ndarray import to_fixed_tuple
from functools import lru_cache
@njit
def make_u3(parameters):
    costheta, sintheta = np.cos(parameters[0]*2/2), np.sin(parameters[0]*2/2)
    lambdaphi = parameters[1]+parameters[2]
    cosphi, sinphi = np.cos(parameters[1]), np.sin(parameters[1])
    coslambda, sinlambda = np.cos(parameters[2]), np.sin(parameters[2])
    cosphilambda, sinphilambda = np.cos(lambdaphi), np.sin(lambdaphi)
    return np.array(
        [[costheta, -(coslambda+sinlambda*1j)*sintheta],
         [(cosphi+sinphi*1j)*sintheta, (cosphilambda+sinphilambda*1j)*costheta]])
@njit
def make_ry(parameters):
    return make_u3([parameters[0], 0, 0])
    #return np.array(
    #    [[np.cos(parameters[0]*2/2), -np.sin(parameters[0]*2/2)],
    #     [np.sin(parameters[0]*2/2), np.cos(parameters[0]*2/2)]])
@njit
def make_controlled(gate):
    return np.block([[np.eye(2), np.zeros((2, 2))], [np.zeros((2, 2)), gate]]) #[np.ix_(*([[0,2,1,3]]*2))]
@njit
def make_cry(parameters):
    return make_ry(parameters) #make_controlled(make_ry(parameters))
@njit
def twoByTwoFloat(A, B):
    res = np.empty(B.shape, dtype=B.dtype)
    for j in range(2):
        for i in range(B.shape[1]):
            res[j,i] = (np.real(A[j,0])*np.real(B[0,i])-np.imag(A[j,0])*np.imag(B[0,i])) + (np.real(A[j,1])*np.real(B[1,i])-np.imag(A[j,1])*np.imag(B[1,i]))
            res[j,i] += ((np.real(A[j,0])*np.imag(B[0,i])+np.imag(A[j,0])*np.real(B[0,i])) + (np.real(A[j,1])*np.imag(B[1,i])+np.imag(A[j,1])*np.real(B[1,i]))) * 1j
            #((np.real(A[j,0])*np.imag(B[0,i])+np.real(A[j,1])*np.imag(B[1,i])) + (np.imag(A[j,0])*np.real(B[0,i])+np.imag(A[j,1])*np.real(B[1,i]))) * 1j
    return res
@lru_cache(128)
def make_apply_to_qbit_loop(num_qbits):
    twos = tuple([2]*num_qbits)
    @njit
    def apply_to_qbit_loop(unitary, _, target_qbit, control_qbit, gate):
        pow2qb = 1 << num_qbits
        t = np.roll(np.arange(num_qbits), target_qbit)
        idxs = np.arange(pow2qb).reshape(twos).transpose(to_fixed_tuple(t, num_qbits)).copy().reshape(-1, 2) #.reshape(*([2]*num_qbits)).transpose(t).reshape(-1, 2)
        for pair in (idxs if control_qbit is None else idxs[(idxs[:,0] & (1<<control_qbit)) != 0,:]):
            unitary[pair,:] = twoByTwoFloat(gate, unitary[pair,:])
            #unitary[pair,:] = gate @ unitary[pair,:]
        return unitary
    return apply_to_qbit_loop
def process_gates(unitary, num_qbits, parameters, target_qbits, control_qbits):
    if unitary.dtype == np.dtype(np.complex128): unitary = np.copy(unitary)
    return process_gates_loop(unitary, num_qbits, parameters, target_qbits, control_qbits, make_apply_to_qbit_loop(num_qbits)) #apply_to_qbit
@njit
def process_gates_loop(unitary, num_qbits, parameters, target_qbits, control_qbits, apply_to_qbit_func):
    for param, target_qbit, control_qbit in zip(parameters, target_qbits, control_qbits):
        unitary = apply_to_qbit_func(unitary, num_qbits, target_qbit, None if control_qbit == target_qbit else control_qbit, (make_u3(param) if control_qbit is None or control_qbit==target_qbit else make_cry(param)).astype(unitary.dtype))
    return unitary
def poly_cost_func():
    from qiskit import QuantumCircuit, transpile
    from qgd_python.utils import get_unitary_from_qiskit_circuit
    import sys
    filename = "/home/morse/ibm_qx_mapping/examples/" + 'ham3_102' + ".qasm"
    qc_trial = QuantumCircuit.from_qasm_file( filename )
    qc_trial = transpile(qc_trial, optimization_level=3, basis_gates=['cz', 'cx', 'u3'], layout_method='sabre')
    Umtx_orig = get_unitary_from_qiskit_circuit( qc_trial )
    #Umtx_orig = np.eye(1<<3, dtype=np.complex128)
    levels = 1
    from qgd_python.decomposition.qgd_N_Qubit_Decomposition_adaptive import qgd_N_Qubit_Decomposition_adaptive
    cDecompose = qgd_N_Qubit_Decomposition_adaptive( Umtx_orig.conj().T, level_limit_max=5, level_limit_min=0 )
    for idx in range(levels):
        cDecompose.add_Adaptive_Layers()
    cDecompose.add_Finalyzing_Layer_To_Gate_Structure()
    cDecompose.set_Cost_Function_Variant(0)
    num_of_parameters = cDecompose.get_Parameter_Num()
    xn = np.random.random(num_of_parameters)*2*np.pi #np.ones(num_of_parameters, dtype=np.float64)*np.pi/2
    cDecompose.set_Optimized_Parameters(xn)
    cDecompose.Prepare_Gates_To_Export()
    gates = cDecompose.get_Gates()
    #target_qbits, control_qbits = get_gate_structure(levels, qbit_num)
    target_qbits = np.array([x['target_qbit'] for x in reversed(gates)], dtype=np.uint8)
    control_qbits = np.array([x['control_qbit'] if 'control_qbit' in x else x['target_qbit'] for x in reversed(gates)], dtype=np.uint8)
    params = np.array([[x['Theta']/2, 0 if x['type'] == 'CRY' else x['Phi'], 0 if x['type'] == 'CRY' else x['Lambda']] for x in reversed(gates)], dtype=np.float64)
    result = process_gates(Umtx_orig.conj().T, Umtx_orig.shape[0].bit_length()-1, params, target_qbits, control_qbits)
    print(cDecompose.Optimization_Problem(xn), 1.0-np.trace(np.real(result))/Umtx_orig.shape[0])
import itertools, collections
sym_expand, use_tosinwave = True, False
class SymExpr:
    def makeSymExpr(newsums):
        newsums = list(itertools.chain.from_iterable([x.sums if isinstance(x, SymExpr) else [x] for x in filter(lambda x: not isinstance(x, SymConst) or x.c != 0.0, newsums)]))
        consts = {k: list(g) for k, g in itertools.groupby(sorted(newsums, key=lambda x: isinstance(x, SymConst)),
            key=lambda x: isinstance(x, SymConst))}
        if True in consts and len(consts[True])>1:
            newsums = [SymConst(np.sum([x.c for x in consts[True]]))] + (consts[False] if False in consts else [])
        #cos(x)=sin(x+pi/2), sin(x)=cos(x-pi/2)
        #combine like terms
        allsums = {}
        for i, sum in enumerate(newsums):
            if isinstance(sum, SymTerm):
                t = tuple(sorted(term for term in sum.var if isinstance(term, SymFunc) and term.func in ["cos", "sin"]))
                if len(t) != 0:
                    if not t in allsums: allsums[t] = set()
                    allsums[t].add(i)
            elif isinstance(sum, SymFunc) and sum.func in ["cos", "sin"]:
                if not sum in allsums: allsums[sum] = set()
                allsums[sum].add(i)
        for sum in allsums:
            if len(allsums[sum]) >= 2:
                idxs = list(allsums[sum])
                newsums[idxs[0]] = SymTerm.makeSymTerm([*sum, SymExpr.makeSymExpr([SymTerm.makeSymTerm([x for x in newsums[idx].var if not isinstance(x, SymFunc) or not x.func in ["cos", "sin"]]) for idx in idxs])])
                for idx in idxs[1:]: newsums[idx] = None
        newsums = list(filter(lambda x: not x is None, newsums))
        if False and sym_expand:
            allsums = {} #parameter ordering
            for i, sum in enumerate(newsums):
                if isinstance(sum, SymTerm):
                    for term in sum.var:
                        if isinstance(term, SymFunc) and term.func in ["cos", "sin"]:
                            if not term in allsums: allsums[term] = set()
                            allsums[term].add(i)
                elif isinstance(sum, SymFunc) and sum.func in ["cos", "sin"]:
                    if not sum in allsums: allsums[sum] = set()
                    allsums[sum].add(i)
            consumed = set()
            for sum in sorted(allsums):
                allsums[sum] -= consumed        
                if len(allsums[sum]) >= 2:
                    idxs = list(allsums[sum])
                    #print(sum, [newsums[idx] for idx in idxs])
                    newsums[idxs[0]] = SymTerm.makeSymTerm([sum, SymExpr.makeSymExpr([SymTerm.makeSymTerm([x for x in newsums[idx].var if not isinstance(x, SymFunc) or not x.func in ["cos", "sin"] or x != sum]) for idx in idxs])])
                    #print(newsums[idxs[0]])
                    for idx in idxs[1:]: newsums[idx] = None
                consumed |= allsums[sum]
            newsums = list(filter(lambda x: not x is None, newsums))
            if use_tosinwave:
                allsums = {}
                for i, sum in enumerate(newsums):
                    if isinstance(sum, SymTerm):
                        allsums[(tuple(term for term in sum.var if isinstance(term, SymFunc) and term.func in ["cos", "sin"]))] = i
                    elif isinstance(sum, SymFunc) and sum.func in ["cos", "sin"]:
                        allsums[((sum,))] = i
                #assert all([np.isclose(np.sqrt(a*a+b*b)*np.sin(z+np.arctan2(b,a)),a*np.sin(z)+b*np.cos(z)) for a, b, z in 2*np.pi*np.random.rand(1000,3)])
                for sum in allsums: #f(x)=a*sin(x)+b*cos(x)=A sin(x+phi) where A=sqrt(a*a+b*b), phi=arctan(b/a)
                    if newsums[allsums[sum]] is None: continue
                    for i, term in enumerate(sum):
                        test = (sum[:i] + (SymFunc("cos" if term.func == "sin" else "sin", term.sym),) + sum[i+1:])
                        if test in allsums and not newsums[allsums[test]] is None:
                            assert not isinstance(newsums[allsums[sum]], SymFunc) and not isinstance(newsums[allsums[test]], SymFunc)  
                            a = [t for t in newsums[allsums[test]].var if not isinstance(t, SymFunc) or not t.func in ["cos", "sin"]]
                            b = [t for t in newsums[allsums[sum]].var if not isinstance(t, SymFunc) or not t.func in ["cos", "sin"]]
                            a = (a[0] if len(a) == 1 else SymTerm.makeSymTerm(a)) if len(a) != 0 else SymConst(1.0)
                            b = (b[0] if len(b) == 1 else SymTerm.makeSymTerm(b)) if len(b) != 0 else SymConst(1.0)
                            if term.func == "sin": a, b = b, a
                            newsums[allsums[test]] = None
                            #newsums[allsums[sum]] = SymTerm.makeSymTerm(sum[:i] + (SymTerm.makeSymTerm([SymFunc("sqrt", SymExpr.makeSymExpr([SymTerm.makeSymTerm([a, a]), SymTerm.makeSymTerm([b, b])])), SymFunc("sin", SymExpr.makeSymExpr([term.sym, SymFunc("atan", [b, a])]))]),) + sum[i+1:])
                            newsums[allsums[sum]] = SymTerm.makeSymTerm(sum[:i] + (SymFunc('tosinwave', [term.sym, a, b]),) + sum[i+1:])
                            #print(newsums[allsums[sum]])
                newsums = list(filter(lambda x: not x is None, newsums))
        if len(newsums) == 0: return SymConst(0.0)
        return SymExpr(newsums) if len(newsums) > 1 else newsums[0]
    def linear(a, b): #ax+b=0, x=-b/a
        return [-b / a]
    def quadratic(a, b, c): #ax^2+bx+c=0
        discriminant = b*b - 4*a*c
        if discriminant < 0: return []
        if discriminant == 0: return [-b/(2*a)]
        return [(-b+np.sqrt(b*b-4*a*c))/(2*a), (-b-np.sqrt(b*b-4*a*c))/(2*a)]
    def cubic(a, b, c, d):
        discriminant = 18*a*b*c*d-4*b*b*b*d+b*b*c*c-4*a*c*c*c-27*a*a*d*d
        d0 = b*b-3*a*c
        d1 = 2*b*b*b-9*a*b*c+27*a*a*d
        print(d0, d1)
        if d1*d1-4*d0*d0*d0 < 0: return []
        if d0==0 and d1==0: return [-b/(3*a)]
        C = np.cbrt((d1+np.sqrt(d1*d1-4*d0*d0*d0))/2)
        if C == 0: C = np.cbrt((d1-np.sqrt(d1*d1-4*d0*d0*d0))/2)
        #C*(-1+np.sqrt(3)*1j)/2+2*d0/(-1+np.sqrt(3)*1j)/C
        #C*C*(-1+np.sqrt(3)*1j)/(2*C)+d0*(-1-np.sqrt(3)*1j)/(2*C)
        #(C*C*(-1+np.sqrt(3)*1j)+d0*(-1-np.sqrt(3)*1j))/(2*C)
        sol = [-(b+C+d0/C)/(3*a)]
        #if discriminant > 0: 3 real roots
        print(C)
        if C*C != d0: return sol
        return [*sol, -(b+(-C*C-d0)/(2*C))/(3*a)]
        print(d0, d1, C)
    def quartic(a, b, c, d, e):
        depa = -3*b*b/(8*a*a)+c/a
        depb = b*b*b/(8*a*a*a)-b*c/(2*a*a)+d/a
        depc = -3*b*b*b*b/(256*a*a*a*a)+c*b*b/(16*a*a*a)-b*d/(4*a*a)+e/a
        if depb == 0: #biquadratic u^4+au^2+c=0
            print(1, depa, depc)
            zs = SymExpr.quadratic(1, depa, depc)
            #if 0 returned, handle specially as odd number of solutions
            us = [np.sqrt(z) for z in zs if z>=0] + [-np.sqrt(z) for z in zs if z>=0]
            return [u - b/(4*a) for u in us]
        else: #u^4+au^2+bu+c=0
            pass
    def check_poly_funcs():
        assert SymExpr.linear(2, 3) == [-3/2]
        assert SymExpr.quadratic(1, 0, 4) == []
        assert SymExpr.quadratic(1, -6, 9) == [3]
        assert SymExpr.quadratic(1, -5, 6) == [3, 2]
        assert SymExpr.cubic(1, 2, 4, 8) == [-2]
        assert SymExpr.cubic(1, 3, 3, 1) == [-1]
        assert SymExpr.cubic(1, -4, 4, 0) == [0, 2]
        assert SymExpr.cubic(1, -6, 11, -6) == [1, 2, 3], SymExpr.cubic(1, -6, 11, -6)
        assert SymExpr.cubic(1, 0, 2, 3) == [(-2-2*np.cbrt(2/(677-15*np.sqrt(2037)))-np.cbrt(4*(677-15*np.sqrt(2037))))/15], SymExpr.cubic(1, 0, 2, 3) #depressed cubics with 0, 1, 2,3 real roots
        assert SymExpr.cubic(1, 0, -3, 2) == [-2, 1], SymExpr.cubic(1, 0, -3, 2)
        assert SymExpr.cubic(1, 0, 2, -2) == [-1.7693, 0.5344]
        assert SymExpr.cubic(1, 0, -7, 6) == [-3, 1, 2]
        assert SymExpr.quartic(1, 4, 6, 4, 1) == []
        assert SymExpr.quartic(1, -4, 6, -4, 1) == [0.5981]
        assert SymExpr.quartic(1, -3, 2, 3, -2) == [np.sqrt(2), -np.sqrt(2)]
        assert SymExpr.quartic(1, -5, 6, -5, 1) == [2 + np.sqrt(3), 2 - np.sqrt(3)]
        assert SymExpr.quartic(1, -8, 17, -10, 1) == [0.143, 0.305, 1.957]
        assert SymExpr.quartic(1, 0, -10, 0, 9) == [-3, -1, 1, 3]
        assert SymExpr.quartic(1, -7, 10, 7, -6) == [-1.966, 0.732, (5+np.sqrt(13))/2, (5-np.sqrt(13))/2]
        assert SymExpr.quartic(1, 0, 2, 0, 3) == [] #depresed quartics with 0, 1, 2, 3, 4 real roots
        assert SymExpr.quartic(1, 0, -4, 0, 4) == [np.sqrt(2), -np.sqrt(2)]
        assert SymExpr.quartic(1, 0, -3, 0, -4) == [2, -2]
        assert SymExpr.quartic(1, 0, 2, 0, -1) == [np.sqrt(np.sqrt(2)-1), -np.sqrt(np.sqrt(2)-1)]
        assert SymExpr.quartic(1, 0, -5, 0, 4) == [2, 1, -2, -1]
        
    def partial_deriv_solver(functions, symbols, costfunc): #assert len(functions) == len(symbols)
        #assumes the function is bounded and the extreme value theorem is satisfied
        print(functions)
        def partial_deriv_solver_inner(functions, symbols):
            b, a, bdbl, adbl = [], [], [], []
            #if len(functions) == 1 and isinstance(functions[0], SymTerm):
            #    #C*sin(x)cos(x)==C*sin(2x)/2==0 only when x=0, x=PI/2
            #    return [[0.0], [np.pi/2]]
            #A*sin(x)+B*cos(x)+C*sin(x)*sin(x)+D*sin(x)*cos(x)+E*cos(x)*sin(x)+F*cos(x)*cos(x)
            #A*sin(x)+B*sin(y) proof of conversion to product
            #A*sin(a+b)=A*sin(a)cos(b)+A*cos(a)sin(b)
            #+B*sin(a-b)=B*sin(a)cos(b)-B*cos(a)sin*b)
            #=A*sin(a+b)+B*sin(a+b)=(A+B)sin(a)cos(b)+(A-B)cos(a)sin(b)
            #x=a+b, y=a-b
            #=A*sin(x)+B*sin(y)=(A+B)sin((x+y)/2)cos((x-y)/2)+(A-B)cos((x+y)/2)sin((x-y)/2)
            #Q=(A+B)sin((x+y)/2), R=(A-B)cos((x+y)/2), P = (x-y)/2 
            #A*sin(x)+B*sin(y)=Q*cos(P)+R*sin(p)=sqrt(Q*Q+R*R)sin(P+arctan(Q/R))
            for func in functions: #(1+sin+cos)(sin+cos)
                allsums = {} #factor one parameter, earliest first            
                for i, sum in enumerate(func.sums):
                    if isinstance(sum, SymTerm):
                        t = tuple(term.func for term in sum.var if isinstance(term, SymFunc) and term.func in ["cos", "sin"] and term.sym.sym == symbols[0])
                        assert len(t) != 0
                        if not t in allsums: allsums[t] = set()
                        allsums[t].add(i)
                    elif isinstance(sum, SymFunc) and sum.func in ["cos", "sin"]:
                        if not (sum.func,) in allsums: allsums[(sum.func,)] = set()
                        allsums[(sum.func,)].add(i)
                    else: assert False, (sum, symbols)
                if ('sin',) in allsums and ('cos',) in allsums:
                    b.append(SymExpr.makeSymExpr([SymTerm.makeSymTerm([t for t in func.sums[k].var if not isinstance(t, SymFunc) or not t.func in ["cos", "sin"] or t.sym.sym != symbols[0]]) for k in allsums[('cos',)]]))
                    a.append(SymExpr.makeSymExpr([SymTerm.makeSymTerm([t for t in func.sums[k].var if not isinstance(t, SymFunc) or not t.func in ["cos", "sin"] or t.sym.sym != symbols[0]]) for k in allsums[('sin',)]]))
                elif 'sin' in allsums or 'cos' in allsums: assert False
                if ('sin', 'cos') in allsums and ('sin', 'sin') in allsums and ('cos', 'cos') in allsums: #sin(x)cos(x)=0.5*sin(2x), sin(x)sin(x)=-0.5*cos(2x), cos(x)cos(x)=0.5*cos(2x)
                    adbl.append(SymConst(0.5)*SymExpr.makeSymExpr([SymTerm.makeSymTerm([t for t in func.sums[k].var if not isinstance(t, SymFunc) or not t.func in ["cos", "sin"] or t.sym.sym != symbols[0]]) for k in allsums[('sin', 'cos',)]]))
                    bdbl.append((SymConst(0.5) *
                        (SymExpr.makeSymExpr([SymTerm.makeSymTerm([t for t in func.sums[k].var if not isinstance(t, SymFunc) or not t.func in ["cos", "sin"] or t.sym.sym != symbols[0]]) for k in allsums[('cos', 'cos',)]]) -  
                        SymExpr.makeSymExpr([SymTerm.makeSymTerm([t for t in func.sums[k].var if not isinstance(t, SymFunc) or not t.func in ["cos", "sin"] or t.sym.sym != symbols[0]]) for k in allsums[('sin', 'sin',)]]))))
                elif ('sin', 'cos') in allsums or ('sin', 'sin') in allsums or ('cos', 'cos') in allsums: assert False
            if len(symbols) == 1: #x=(-1)^k*arcsin(y)+PI*k if sin(x)=y
                if len(a) != 0 and len(adbl) != 0:
                    A, B = SymFunc.makeSymFunc('sqrt', a[0]*a[0]+b[0]*b[0]), SymFunc.makeSymFunc('sqrt', adbl[0]*adbl[0]+bdbl[0]*bdbl[0])
                    #x, y = SymSymbol(symbols[0]) + SymFunc.makeSymFunc('atan', [b[0], a[0]]), SymConst(2.0)*SymSymbol(symbols[0]) + SymFunc.makeSymFunc('atan', [bdbl[0], adbl[0]])
                    #Q, R, P = (A+B)*SymFunc.makeSymFunc('sin', SymConst(0.5)*(x+y)), (A-B)*SymFunc.makeSymFunc('cos', SymConst(0.5)*(x+y)), SymConst(0.5)*(x-y)
                    #Q^2+P^2=(A+B)^2sin^2(x)+(A-B)^2(1-sin^2(x))=(A-B)^2+sin^2(x)((A+B)^2-(A-B)^2)
                    #print(functions[0].apply_to({symbols[0]: 0}))
                    #print((SymFunc.makeSymFunc('sqrt', Q*Q+R*R)*SymFunc.makeSymFunc('sin', P+SymFunc.makeSymFunc('atan', [Q, R]))).apply_to({symbols[0]: 0}))
                    #print(SymFunc.makeSymFunc('sin', P+SymFunc.makeSymFunc('atan', [Q, R])), (SymFunc('sin', P+SymFunc.makeSymFunc('atan', [Q, R]))).apply_to({symbols[0]: 0}))
                    #print((P+SymFunc.makeSymFunc('atan', [Q, R])).apply_to({symbols[0]: 0}))
                    #tan(a+b)=(tan(a)+tan(b))/(1-tan(a)tan(b)), tan(a-b)=(tan(a)-tan(b))/(1+tan(a)tan(b))
                    #y=x/2
                    #phi1 = [phi1, phi1-SymConst(np.pi)]
                    K, phi1, phi2 = (A+B)/(A-B), SymConst(0.5)*(SymFunc.makeSymFunc('atan', [b[0], a[0]]) - SymFunc.makeSymFunc('atan', [bdbl[0], adbl[0]])), SymConst(0.5)*(SymFunc.makeSymFunc('atan', [b[0], a[0]]) + SymFunc.makeSymFunc('atan', [bdbl[0], adbl[0]])) 
                    #print(K, phi1, phi2)
                    #phi1 -= SymConst(np.pi) 
                    #print(SymFunc('sin', (K*SymFunc('tan', SymConst(1.5)*SymSymbol(symbols[0])+phi2)-SymFunc('tan', SymConst(0.5)*SymSymbol(symbols[0])-phi1))).apply_to({symbols[0]: 0}))
                    #tan(y-phi1)=Ktan(3y+phi2)
                    #Ktan(3y+phi2)-tan(y-phi1)
                    #C, D, z = SymFunc.makeSymFunc('tan', phi1), SymFunc.makeSymFunc('tan', phi2), SymFunc.makeSymFunc('tan', y)
                    #print(K*SymFunc.makeSymFunc('tan', SymConst(3.0)*y+phi2)-SymFunc.makeSymFunc('tan', y-phi1))
                    #tan3 = SymFunc.makeSymFunc('tan', SymConst(3.0)*y)
                    #print(K*(tan3+D)/(SymConst(1.0)-tan3*D)-(z-C)/(SymConst(1.0)+z*C))
                    #(tan(y)-tan(phi1))/(1+tan(y)tan(phi1))=K(tan(3y)+tan(phi2))/(1-tan(3y)tan(phi2))
                    #(tan(y)-tan(phi1))(1-tan(3y)tan(phi2))=K(tan(3y)+tan(phi2))(1+tan(y)tan(phi1))
                    #print(K*(tan3+D)*(SymConst(1.0)+z*C)-(z-C)*(SymConst(1.0)-tan3*D))
                    #(tan(y)-tan(phi1))(1-tan(phi2)(3tan(y)-tan^3(y))/(1-3tan^2(y)))=K((3tan(y)-tan^3(y))/(1-3tan^2(y))+tan(phi2))(1+tan(y)tan(phi1))
                    #z=tan(y), C=tan(phi1), D=tan(phi2)
                    #(z-C)(1-D(3z-z^3)/(1-3z^2))=K((3z-z^3)/(1-3z^2)+D)(1+z*C)
                    #(z-C)(1-3z^2-D(3z-z^3))=K((3z-z^3)+D(1-3z^2))(1+z*C)
                    C, D, z = SymFunc('tan', phi1), SymFunc('tan', phi2), SymFunc('tan', SymSymbol(symbols[0]))                    
                    #z-3z^3-3Dz^2+Dz^4-C+3Cz^2+3CDz-CDz^3=3Kz-Kz^3+KD-3KDz^2+3KCz^2-CKz^4+CDKz-3CDKz^3
                    #(-CK-D)z^4+(CD+3-3CDK-K)z^3+(3KC-3KD-3C+3D)z^2+(3K+CDK-3CD-1)z+KD+C
                    roots = np.roots([(-C*K-D).apply_to(None), (C*D+SymConst(3.0)-SymConst(3.0)*C*D*K-K).apply_to(None), (SymConst(3.0)*(K*C-K*D-C+D)).apply_to(None), (SymConst(3.0)*K+C*D*K-SymConst(3.0)*C*D-SymConst(1.0)).apply_to(None), (K*D+C).apply_to(None)])
                    return [[2*np.arctan(r.real)] for r in roots if np.isclose(r.imag, 0.0)]
                elif len(adbl) != 0:                   
                    value = -SymFunc('atan', [bdbl[0], adbl[0]]).apply_to(None)
                    return [[value/2], [np.pi+value/2]]
                value = -SymFunc('atan', [b[0], a[0]]).apply_to(None)
                return [[value], [np.pi+value]]
            else:
                newfuncs = []
                for i in range(1, len(functions)): #b0/a0=bi/ai
                    newfuncs.append(b[0]*a[i]-a[0]*b[i])
                print(b, a)
                print(newfuncs)
                newsols = partial_deriv_solver_inner(newfuncs, symbols[1:])
                combined = [[-SymFunc('atan', [b[0], a[0]]).apply_to({sym: solution[i] for i, sym in enumerate(symbols[1:])})] + solution for solution in newsols]
                combined += [[x[0]+np.pi] + x[1:] for x in combined]
                return combined
        solutions = partial_deriv_solver_inner(functions, symbols)
        for func in functions:
            for sol in solutions:
                assert np.isclose(0.0, func.apply_to({symbols[i]: sol[i] for i in range(len(symbols))}))
        return min((solution for solution in solutions), key=lambda x: costfunc(symbols, x))
    def __init__(self, newsums):
        assert len(newsums) >= 2
        self.sums = newsums
    def __add__(self, other):
        return SymExpr.makeSymExpr([self, other])
    def __sub__(self, other):
        return SymExpr.makeSymExpr([self, -other])
    def __mul__(self, other):
        if not sym_expand: return SymTerm.makeSymTerm([self, other])
        newsums = []
        for sum in self.sums:
            newsums.append(sum * other)
        return SymExpr.makeSymExpr(newsums)
    def __neg__(self):
        return SymExpr.makeSymExpr([-x for x in self.sums])
    def __hash__(self): return hash(tuple(self.sums))
    def __eq__(self, other): return self.sums == other.sums
    def apply_to(self, symdict):
        return np.sum(x.apply_to(symdict) for x in self.sums)
    def num_ops(self):
        ops = {'+': len(self.sums)-1, '*': 0, 'tosinwave': 0}
        for innerops in (x.num_ops() for x in self.sums if isinstance(x, (SymTerm, SymFunc))):
            ops['+'] += innerops['+']; ops['*'] += innerops['*']; ops['tosinwave'] += innerops['tosinwave']
        return ops
    def partial_deriv(self, symbol):
        return SymExpr.makeSymExpr([x.partial_deriv(symbol) for x in self.sums])
    def __repr__(self):
        return str(self)
    def __str__(self):
        #if sym_expand: return '+'.join(str(x) for x in self.sums)
        return '(' + '+'.join(str(x) for x in self.sums) + ')'
class SymTerm:
    def makeSymTerm(newvar):
        newvar = list(itertools.chain.from_iterable([x.var if isinstance(x, SymTerm) else [x] for x in filter(lambda x: not isinstance(x, SymConst) or x.c != 1.0, newvar)]))
        if any(isinstance(x, SymConst) and x.c == 0.0 for x in newvar): return SymConst(0.0)
        #if sym_expand: assert all(not isinstance(x, SymExpr) for x in newvar)
        consts = {k: list(g) for k, g in itertools.groupby(sorted(newvar, key=lambda x: isinstance(x, SymConst)),
            key=lambda x: isinstance(x, SymConst))}
        if True in consts and len(consts[True])>1:
            newvar = [SymConst(np.prod([x.c for x in consts[True]]))] + (consts[False] if False in consts else [])
        if len(newvar) == 0: return SymConst(1.0)
        return SymTerm(newvar) if len(newvar) > 1 else newvar[0]
    def __init__(self, newvar):
        assert len(newvar) >= 2
        self.var = newvar        
    def __add__(self, other):
        return SymExpr.makeSymExpr([self, other])
    def __sub__(self, other):
        return SymExpr.makeSymExpr([self, -other])
    def __mul__(self, other):
        if sym_expand and isinstance(other, SymExpr): return other * self
        else: return SymTerm.makeSymTerm([self, other])
    def apply_to(self, symdict):
        return np.prod([x.apply_to(symdict) for x in self.var])
    def __neg__(self): return SymTerm.makeSymTerm([SymConst(-1.0)] + self.var)
    def num_ops(self):
        ops = {'+': 0, '*': (1 if any(isinstance(x, SymConst) and x.c != -1.0 or isinstance(x, SymSymbol) for x in self.var) else 0) + sum(1 for x in self.var if not isinstance(x, (SymConst, SymSymbol)))-1, 'tosinwave': 0}
        for innerops in (x.num_ops() for x in self.var if isinstance(x, (SymExpr, SymFunc))):
            ops['+'] += innerops['+']; ops['*'] += innerops['*']; ops['tosinwave'] += innerops['tosinwave']
        return ops
    def partial_deriv(self, symbol): #product rule
        return SymExpr.makeSymExpr([SymTerm.makeSymTerm(self.var[:i] + [x.partial_deriv(symbol)] + self.var[i+1:]) for i, x in enumerate(self.var)])
    def __repr__(self): return str(self)
    def __str__(self): return '*'.join(str(x) for x in self.var)
class SymConst:
    def __init__(self, c):
        self.c = c
    def __add__(self, other):
        if self.c == 0: return other
        return SymExpr.makeSymExpr([self, other])
    def __sub__(self, other):
        if self.c == 0: return -other
        return SymExpr.makeSymExpr([self, -other])
    def __mul__(self, other):
        if self.c == 0.0: return self
        elif self.c == 1.0: return other
        elif isinstance(other, SymConst): return SymConst(self.c * other.c)
        return other * self
    def __truediv__(self, other):
        if isinstance(other, SymConst): return SymConst(self.c / other.c)
    def __neg__(self): return SymConst(-self.c)
    def apply_to(self, symdict): return self.c
    def partial_deriv(self, symbol): return SymConst(0.0)
    def __repr__(self): return str(self)
    def __str__(self): return str(self.c)
class SymSymbol:
    SymOrder = {'ϴ': 0, 'φ': 1, 'λ': 2}
    def __init__(self, sym): self.sym = sym
    def __add__(self, other): return SymExpr.makeSymExpr([self, other])
    def __neg__(self): return SymTerm.makeSymTerm([SymConst(-1.0), self])
    def apply_to(self, symdict): return symdict[self.sym]
    def partial_deriv(self, symbol): return SymConst(1.0) if symbol == self else SymConst(0.0)
    def __hash__(self): return hash(self.sym)
    def __mul__(self, other): return SymTerm.makeSymTerm([self, other])
    def __lt__(self, other): return SymSymbol.SymOrder[self.sym[0]] < SymSymbol.SymOrder[other.sym[0]] if int(self.sym[1:]) == int(other.sym[1:]) else int(self.sym[1:]) < int(other.sym[1:])
    def __eq__(self, other): return self.sym == other.sym
    def __repr__(self): return str(self)
    def __str__(self): return self.sym
class SymFunc:
    FuncOrder = {'sin': 0, 'cos': 1, 'sqrt': 2, 'atan': 3, 'tosinwave': 4, 'pi': 5, 'tan': 6}
    FuncNames = {'sin': 'sin', 'cos': 'cos', 'sqrt': '√', 'atan': 'atan', 'pi': 'π', 'tan': 'tan'}
    def makeSymFunc(func, sym):
        if func == 'sin' and isinstance(sym, SymConst): return SymConst(np.sin(sym.c))
        if func == 'cos' and isinstance(sym, SymConst): return SymConst(np.cos(sym.c))
        if func == 'tan' and isinstance(sym, SymConst): return SymConst(np.tan(sym.c))
        if func == 'sqrt' and isinstance(sym, SymConst): return SymConst(np.sqrt(sym.c))
        if func == 'atan' and isinstance(sym[0], SymConst) and isinstance(sym[1], SymConst): return SymConst(np.arctan2(sym[0].c, sym[1].c))
        if func == 'tosinwave':
            if isinstance(sym[0], SymConst) and isinstance(sym[1], SymConst) and isinstance(sym[2], SymConst):
                return SymConst(SymFunc.tosinwave(sym[0].c, sym[1].c, sym[2].c))
            if isinstance(sym[1], SymConst) and isinstance(sym[2], SymConst):
                return SymTerm([SymFunc('sqrt', sym[1].c*sym[1].c+sym[2].c*sym[2].c), SymFunc('sin', SymTerm([sym[0], np.arctan2(sym[2].c, sym[1].c)]))])
        return SymFunc(func, sym)
    def __init__(self, func, sym):
        self.func, self.sym = func, sym
    def __add__(self, other):
        return SymExpr.makeSymExpr([self, other])
    def __sub__(self, other):
        return SymExpr.makeSymExpr([self, -other])
    def __mul__(self, other):
        if sym_expand and isinstance(other, SymExpr): return other * self
        return SymTerm.makeSymTerm([self, other])
    def __neg__(self): return SymTerm.makeSymTerm([SymConst(-1.0), self])
    def tosinwave(x, a, b):
        return np.sqrt(a*a+b*b)*np.sin(x+np.arctan2(b, a))
    def apply_to(self, symdict):
        funcs = {'sin': np.sin, 'cos': np.cos, 'tan': np.tan, 'sqrt': np.sqrt, 'atan': np.arctan2, 'tosinwave': SymFunc.tosinwave}
        if isinstance(self.sym, list): return funcs[self.func](*(x.apply_to(symdict) for x in self.sym)) 
        return funcs[self.func](self.sym.apply_to(symdict))
    def num_ops(self):
        ops = {'+': 0, '*': 0, 'tosinwave': 1 if self.func == 'tosinwave' else 0}
        if isinstance(self.sym, list):
            for innerops in (x.num_ops() for x in self.sym if isinstance(x, (SymExpr, SymTerm, SymFunc))):
                ops['+'] += innerops['+']; ops['*'] += innerops['*']; ops['tosinwave'] += innerops['tosinwave']
        return ops
    def partial_deriv(self, symbol):
        if self.func == 'sin' and symbol == self.sym: return SymFunc('cos', self.sym)
        elif self.func == 'cos' and symbol == self.sym: return -SymFunc('sin', self.sym)
        #tan, sqrt and atan and tosinwave require division implemented
        return SymConst(0.0)
    def __hash__(self): return hash((self.func, *self.sym)) if isinstance(self.sym, list) else hash((self.func, self.sym))
    def __lt__(self, other):
        if ((self.sym.sums[0] if isinstance(self.sym, SymExpr) else self.sym) ==
            (other.sym.sums[0] if isinstance(other.sym, SymExpr) else other.sym)): return SymFunc.FuncOrder[self.func] < SymFunc.FuncOrder[other.func]
        return ((self.sym.sums[0] if isinstance(self.sym, SymExpr) else self.sym) <
                 (other.sym.sums[0] if isinstance(other.sym, SymExpr) else other.sym))
    def __eq__(self, other): return self.func == other.func and type(self.sym) == type(other.sym) and self.sym == other.sym
    def __repr__(self): return str(self)
    def __str__(self): return self.func + '(' + (",".join([str(x) for x in self.sym]) if isinstance(self.sym, list) else str(self.sym)) + ')'
def make_u3_sym(parameters):
    cosTheta, sinTheta = SymFunc.makeSymFunc('cos', parameters[0]),  SymFunc.makeSymFunc('sin', parameters[0])
    cosPhi, sinPhi = SymFunc.makeSymFunc('cos', parameters[1]), SymFunc.makeSymFunc('sin', parameters[1])
    cosLambda, sinLambda = SymFunc.makeSymFunc('cos', parameters[2]), SymFunc.makeSymFunc('sin', parameters[2])
    #cosPhiLambda, sinPhiLambda = SymFunc.makeSymFunc('cos', parameters[1]+parameters[2]), SymFunc.makeSymFunc('sin', parameters[1]+parameters[2])
    #cos(a+b)=cos(a)cos(b)-sin(a)sin(b)
    #sin(a+b)=sin(a)cos(b)+cos(a)sin(b)
    cosPhiLambda, sinPhiLambda = cosPhi*cosLambda-sinPhi*sinLambda, sinPhi*cosLambda+cosPhi*sinLambda 
    return [[(cosTheta, SymConst(0)),
             (-cosLambda * sinTheta, -sinLambda * sinTheta)],
            [(cosPhi * sinTheta, sinPhi * sinTheta),
             (cosPhiLambda * cosTheta, sinPhiLambda * cosTheta)]] 
def make_ry_sym(parameters):
    cosTheta, sinTheta = SymFunc.makeSymFunc('cos', parameters[0]), SymFunc.makeSymFunc('sin', parameters[0])
    return [[(cosTheta, SymConst(0)),
             (-sinTheta, SymConst(0))],
            [(sinTheta, SymConst(0)),
             (cosTheta, SymConst(0))]]
def make_cry_sym(parameters): return make_ry_sym(parameters)
def twoByTwoFloat_sym(A, B):
    res = [[[None, None] for _ in range(len(B[0]))] for x in range(2)]
    for j in range(2):
        for i in range(len(B[0])):
            res[j][i][0] = (A[j][0][0]*B[0][i][0]-A[j][0][1]*B[0][i][1]) + (A[j][1][0]*B[1][i][0]-A[j][1][1]*B[1][i][1])
            res[j][i][1] = (A[j][0][0]*B[0][i][1]+A[j][0][1]*B[0][i][0]) + (A[j][1][0]*B[1][i][1]+A[j][1][1]*B[1][i][0])
    return res
def apply_to_qbit_loop_sym(unitary, num_qbits, target_qbit, control_qbit, gate):
    pow2qb = 1 << num_qbits
    t = np.roll(np.arange(num_qbits), target_qbit)
    idxs = np.arange(pow2qb).reshape(*([2]*num_qbits)).transpose(t).reshape(-1, 2)
    for pair in (idxs if control_qbit is None else idxs[(idxs[:,0] & (1<<control_qbit)) != 0,:]):
        unitary[pair[0]], unitary[pair[1]] = twoByTwoFloat_sym(gate, [unitary[pair[0]], unitary[pair[1]]])
    return unitary
def process_gates_sym(unitary, num_qbits, parameters, target_qbits, control_qbits):
    unitary = [x[:] for x in unitary]
    return process_gates_loop_sym(unitary, num_qbits, parameters, target_qbits, control_qbits, apply_to_qbit_loop_sym)
def process_gates_loop_sym(unitary, num_qbits, parameters, target_qbits, control_qbits, apply_to_qbit_func):
    for param, target_qbit, control_qbit in zip(parameters, target_qbits, control_qbits):
        unitary = apply_to_qbit_func(unitary, num_qbits, target_qbit, None if control_qbit == target_qbit else control_qbit, (make_u3_sym(param) if control_qbit is None or control_qbit==target_qbit else make_cry_sym(param)))
    return unitary
def normal_cost_func_ops(num_qbits, levels):
    num_cry = num_qbits * levels
    num_u3 = num_qbits*(num_qbits-1)*levels+num_qbits
    return {'+': (1<<num_qbits) + (4**num_qbits)*(6*num_u3+3*num_cry), '*': (4**num_qbits)*(8*num_u3+4*num_cry)}
def sym_min_param(num_qbits, uni_sym, target_qbits, control_qbits, param_consts, param_sym, paramidxs):
    idx = 0
    param_consts = [x[:] for x in param_consts]
    paramidx = []
    for i, p in enumerate(param_consts):
        if idx in paramidxs: param_consts[i][0] = param_sym[i][0]; paramidx.append((i, 0))
        idx+=1
        if not param_consts[i][1] is None:
            if idx in paramidxs: param_consts[i][1] = param_sym[i][1]; paramidx.append((i, 1))
            idx+=1
        if not param_consts[i][2] is None:
            if idx in paramidxs: param_consts[i][2] = param_sym[i][2]; paramidx.append((i, 2))
            idx+=1
    target_sym = process_gates_sym(uni_sym, num_qbits, param_consts, target_qbits, control_qbits)
    final_sym = SymConst(0.0)
    for i in range(1<<num_qbits):
        final_sym = final_sym + target_sym[i][i][0]
    return final_sym, paramidx
def sym_execute():
    from qiskit import QuantumCircuit, transpile
    from qgd_python.utils import get_unitary_from_qiskit_circuit
    import sys
    filename = "/home/morse/ibm_qx_mapping/examples/" + 'ham3_102' + ".qasm"
    qc_trial = QuantumCircuit.from_qasm_file( filename )
    qc_trial = transpile(qc_trial, optimization_level=3, basis_gates=['cz', 'cx', 'u3'], layout_method='sabre')
    Umtx_orig = get_unitary_from_qiskit_circuit( qc_trial )
    #Umtx_orig = np.eye(1<<3, dtype=np.complex128)
    levels = 1
    from qgd_python.decomposition.qgd_N_Qubit_Decomposition_adaptive import qgd_N_Qubit_Decomposition_adaptive
    cDecompose = qgd_N_Qubit_Decomposition_adaptive( Umtx_orig.conj().T, level_limit_max=5, level_limit_min=0 )
    for idx in range(levels):
        cDecompose.add_Adaptive_Layers()
    cDecompose.add_Finalyzing_Layer_To_Gate_Structure()
    cDecompose.set_Cost_Function_Variant(0)
    num_of_parameters = cDecompose.get_Parameter_Num()
    xn = np.random.random(num_of_parameters)*2*np.pi #np.ones(num_of_parameters, dtype=np.float64)*np.pi/2
    cDecompose.set_Optimized_Parameters(xn)
    cDecompose.Prepare_Gates_To_Export()
    gates = cDecompose.get_Gates()
    #target_qbits, control_qbits = get_gate_structure(levels, qbit_num)
    target_qbits = np.array([x['target_qbit'] for x in reversed(gates)], dtype=np.uint8)
    control_qbits = np.array([x['control_qbit'] if 'control_qbit' in x else x['target_qbit'] for x in reversed(gates)], dtype=np.uint8)
    params = np.array([[x['Theta']/2, 0 if x['type'] == 'CRY' else x['Phi'], 0 if x['type'] == 'CRY' else x['Lambda']] for x in reversed(gates)], dtype=np.float64)
    #print(1-np.trace(np.real(process_gates(Umtx_orig.conj().T, Umtx_orig.shape[0].bit_length()-1, params, target_qbits, control_qbits)))/Umtx_orig.shape[0], cDecompose.Optimization_Problem(xn), cDecompose.Optimization_Problem_Combined(xn)[0])
    num_qbits = Umtx_orig.shape[0].bit_length()-1
    num_cry = num_qbits * levels
    num_u3 = num_qbits*(num_qbits-1)*levels+num_qbits
    #Umtx_orig = np.random.rand(*Umtx_orig.shape) + np.random.rand(*Umtx_orig.shape)*1j
    print(normal_cost_func_ops(num_qbits, levels), num_cry, num_u3, sum(1 for x in gates if x['type'] == 'CRY'), len(gates), num_of_parameters)
    uni_stamped = [[(SymConst(np.real(Umtx_orig[j][i])), SymConst(-np.imag(Umtx_orig[j][i]))) for j in range(Umtx_orig.shape[1])] for i in range(Umtx_orig.shape[0])]
    param_stamped = [[SymConst(x['Theta']/2), None if x['type'] == 'CRY' else SymConst(x['Phi']), None if x['type'] == 'CRY' else SymConst(x['Lambda'])] for i, x in reversed(list(enumerate(gates)))]
    uni_sym = [[(SymSymbol('r' + str(i) + '_' + str(j)), SymSymbol('i' + str(i) + '_' + str(j))) for j in range(Umtx_orig.shape[1])] for i in range(Umtx_orig.shape[0])]
    param_sym = [[SymSymbol('ϴ' + str(i)), None if x['type'] == 'CRY' else SymSymbol('φ' + str(i)), None if x['type'] == 'CRY' else SymSymbol('λ' + str(i))] for i, x in reversed(list(enumerate(gates)))]
    param_symdict = dict(collections.ChainMap(*
               [{'ϴ' + str(i): x['Theta']/2, 'φ' + str(i): None if x['type'] == 'CRY' else x['Phi'], 'λ' + str(i): None if x['type'] == 'CRY' else x['Lambda']} for i, x in reversed(list(enumerate(gates)))]))
    for i in itertools.combinations(range(num_of_parameters), 2):
        minsym, paramidx = sym_min_param(Umtx_orig.shape[0].bit_length()-1, uni_stamped, target_qbits, control_qbits, param_stamped, param_sym, {*i})
        print(minsym, paramidx)
        def costfunc(syms, value):
            oldvals = {sym: param_symdict[sym] for sym in syms}
            for i, sym in enumerate(syms):
                param_symdict[sym] = value[i]
            cost = 1.0-minsym.apply_to(param_symdict)/(1<<num_qbits)
            for sym in syms:
                param_symdict[sym] = oldvals[sym]
            return cost
        partials = [minsym.partial_deriv(param_sym[j][k]) for j, k in paramidx]
        symbols = [param_sym[j][k].sym for j, k in paramidx]
        print(costfunc(symbols, SymExpr.partial_deriv_solver(partials, symbols, costfunc)))
        print(cDecompose.Optimization_Problem(xn), 1.0-minsym.apply_to(param_symdict)/(1<<num_qbits))
    #print(1.0-np.trace(np.real(process_gates(Umtx_orig.conj().T, Umtx_orig.shape[0].bit_length()-1, params, target_qbits, control_qbits)))/(1<<num_qbits))
    assert False
    num_gates = len(gates)
    target_sym = process_gates_sym(uni_sym, Umtx_orig.shape[0].bit_length()-1, param_sym[-num_gates:], target_qbits[-num_gates:], control_qbits[-num_gates:])
    symdict = dict(collections.ChainMap(*[{'r' + str(i) + '_' + str(j): np.real(Umtx_orig[j][i]) for j in range(Umtx_orig.shape[1]) for i in range(Umtx_orig.shape[0])},
               {'i' + str(i) + '_' + str(j): -np.imag(Umtx_orig[j][i]) for j in range(Umtx_orig.shape[1]) for i in range(Umtx_orig.shape[0])}] +
               [{'ϴ' + str(i): x['Theta']/2, 'φ' + str(i): None if x['type'] == 'CRY' else x['Phi'], 'λ' + str(i): None if x['type'] == 'CRY' else x['Lambda']} for i, x in reversed(list(enumerate(gates[:num_gates])))]))
    #sin(arctan(x))=x/sqrt(1+x^2), cos(arctan(x))=1/sqrt(1+x^2)
    #arctan(-x)=-arctan(x)
    #arctan(1/x)=-PI/2-arctan(x) if x<0 else PI/2-arctan(x)
    #x+y+z+xy+xz+yz+xyz=x(1+y(1+z)+z)+y(1+z)+z
    for i in range(Umtx_orig.shape[0]):               
        print(target_sym[i][i][0])
        print(target_sym[i][i][0].apply_to(symdict))
        print(target_sym[i][i][1].apply_to(symdict))
        print(process_gates(Umtx_orig.conj().T, Umtx_orig.shape[0].bit_length()-1, params[-num_gates:], target_qbits[-num_gates:], control_qbits[-num_gates:])[i,i])
        print(target_sym[i][i][0].num_ops()) #len(target_sym[i][i][0].sums)
#train_model(num_of_parameters, 100000)
#transformer_model()
#newton_method()
#SymExpr.check_poly_funcs()
sym_execute()
#poly_cost_func()
