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

def make_u3(parameters):
    return np.array(
        [[np.cos(parameters[0]*2/2), -np.exp(parameters[2]*1j)*np.sin(parameters[0]*2/2)],
         [np.exp(parameters[1]*1j)*np.sin(parameters[0]*2/2), np.exp((parameters[1]+parameters[2])*1j)*np.cos(parameters[0]*2/2)]])
def make_ry(parameters):
    return make_u3([parameters[0], 0, 0])
    #return np.array(
    #    [[np.cos(parameters[0]*2/2), -np.sin(parameters[0]*2/2)],
    #     [np.sin(parameters[0]*2/2), np.cos(parameters[0]*2/2)]])
def make_controlled(gate):
    return np.block([[np.eye(2), np.zeros((2, 2))], [np.zeros((2, 2)), gate]]) #[np.ix_(*([[0,2,1,3]]*2))]
def make_cry(parameters):
    return make_ry(parameters) #make_controlled(make_ry(parameters))
def apply_to_qbit(unitary, num_qbits, target_qbit, control_qbit, gate):
    pow2qb = 1 << num_qbits
    t = np.arange(num_qbits)
    if not control_qbit is None:
        t[:-1] = np.roll(t[:-1], (target_qbit - control_qbit) % num_qbits)
        gate = make_controlled(gate)
    t = np.roll(t, -target_qbit)
    idxs = np.arange(pow2qb).reshape(*((2,)*num_qbits)).transpose(t).flatten().tolist()
    return np.kron(np.eye(pow2qb>>(1 if control_qbit is None else 2), dtype=np.bool_), gate)[np.ix_(idxs, idxs)].astype(unitary.dtype) @ unitary
def make_apply_to_qbit_loop(num_qbits):
    twos = tuple([2]*num_qbits)
    def apply_to_qbit_loop(unitary, _, target_qbit, control_qbit, gate):
        pow2qb = 1 << num_qbits
        t = np.roll(np.arange(num_qbits), target_qbit)
        idxs = np.arange(pow2qb).reshape(twos).transpose(to_fixed_tuple(t, num_qbits)).copy().reshape(-1, 2) #.reshape(*([2]*num_qbits)).transpose(t).reshape(-1, 2)
        for pair in (idxs if control_qbit is None else idxs[(idxs[:,0] & (1<<control_qbit)) != 0,:]):
            unitary[pair,:] = twoByTwoFloat(gate, unitary[pair,:])
            #unitary[pair,:] = gate @ unitary[pair,:]
        return unitary
    return apply_to_qbit_loop
def process_gates32(unitary, num_qbits, parameters, target_qbits, control_qbits):
    return process_gates(unitary.astype(np.complex64), num_qbits, parameters, target_qbits, control_qbits).astype(np.complex128)
def process_gates(unitary, num_qbits, parameters, target_qbits, control_qbits):
    if unitary.dtype == np.dtype(np.complex128): unitary = np.copy(unitary)
    return process_gates_loop(unitary, num_qbits, parameters, target_qbits, control_qbits, make_apply_to_qbit_loop(num_qbits)) #apply_to_qbit
def process_gates_loop(unitary, num_qbits, parameters, target_qbits, control_qbits, apply_to_qbit_func):
    for param, target_qbit, control_qbit in zip(parameters, target_qbits, control_qbits):
        unitary = apply_to_qbit_func(unitary, num_qbits, target_qbit, None if control_qbit == target_qbit else control_qbit, (make_u3(param) if control_qbit is None or control_qbit==target_qbit else make_cry(param)).astype(unitary.dtype))
    return unitary

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
def wolfe_conditions_line(f, xn, delta, oldmn, m): #compute learning rate by backtracking line search
    c, tau, alpha, feval = 0.5, 0.5, [1.0], []
    #assert m < 0
    t = -c * m #t = max(0, -c * m)
    while True:
        ftry = f(xn + delta * alpha[-1])
        #if len(feval) and feval[-1] == ftry: break
        feval.append(ftry)
        #if oldmn - feval[-1] > 0: break
        if oldmn - feval[-1] >= alpha[-1] * t: break        
        if alpha[-1] < 1e-10: break
        alpha.append(tau * alpha[-1])
    #print(alpha, feval, oldmn, m, t, min(enumerate(alpha), key=lambda x: feval[x[0]])[1])
    return min(enumerate(zip(alpha, feval)), key=lambda x: feval[x[0]])[1]
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
    cDecompose.set_Cost_Function_Variant(6)
    # adding decomposing layers to the gat structure
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
    def f(xn):
        return cDecompose.Optimization_Problem(xn)
        #mat, mat_deriv = cDecompose.Optimization_Problem_Combined_Unitary(xn)
        #print(cDecompose.Optimization_Problem(xn),costfunc(mat)) 
        #return costfunc(mat)
    def grad(xn):
        return cDecompose.Optimization_Problem_Combined(xn)[1]
    def costfunc(mat):
        #return np.sum(np.abs(mat-np.eye(mat.shape[1], dtype=mat.dtype)))
        return np.sum(np.square((mat-np.eye(mat.shape[1], dtype=mat.dtype)).view(np.float64)))
        #return 1.0-np.real(np.trace(mat))/mat.shape[1]
    from scipy.optimize import least_squares
    def fun(xn):
        mat, mat_deriv = cDecompose.Optimization_Problem_Combined_Unitary(xn)
        #return np.real(mat - np.eye(mat.shape[1], dtype=mat.dtype)).flatten()
        return (mat - np.eye(mat.shape[1], dtype=mat.dtype)).view(np.float64).flatten()
    def jac(xn):
        mat, mat_deriv = cDecompose.Optimization_Problem_Combined_Unitary(xn)
        #return np.real(np.stack(mat_deriv, -1)).reshape(mat.flatten().shape[0], num_of_parameters)
        mat = mat.view(np.float64).flatten()
        return np.stack(mat_deriv, -1).view(np.float64).reshape(mat.shape[0]//2, num_of_parameters, 2).transpose(0,2,1).reshape(mat.shape[0], num_of_parameters)
    #print(fourier_coeffs_mc(f, [[0]*num_of_parameters, [2*np.pi]*num_of_parameters], 100))
    def oneparammin(xn):
        cDecompose.set_Cost_Function_Variant(0)
        cost = cDecompose.Optimization_Problem(xn)
        while True:
            shift, costs = np.zeros((len(xn),)), np.zeros((len(xn),))
            for d in range(len(xn)):
                val = xn[d]
                xn[d] += np.pi/2
                c1 = cDecompose.Optimization_Problem(xn)
                #f(x)=C+A*sin(x+phi) and f(x+PI/2)=C+A*cos(x+phi)
                #(f(x)-C)/(f(x+PI/2)-C)=tan(x+phi)                
                xn[d] += np.pi/2
                c2 = cDecompose.Optimization_Problem(xn)
                C = (cost+c2)/2
                truephi = np.arctan2(cost-C, c1-C)               
                truephi = (truephi-val) % (2 * np.pi) #sin(x+phi)=-1 3PI/2-phi
                shift[d] = 3*np.pi/2-truephi
                origsin = np.sin(val+truephi)
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
                xn[d] = val
            minidx = min(enumerate(costs), key=lambda x: x[1])[0]
            cost = costs[minidx]
            xn[minidx] = shift[minidx]
            print(cost) 
        assert False 
    #ret = least_squares(fun, xn, jac, (0, 2*np.pi), verbose=2)
    #ret = least_squares(fun, xn, jac, method='lm', verbose=2)
    #ret = least_squares(f, xn, grad, (0, 2*np.pi), verbose=2)
    #ret = least_squares(fun, xn, jac, (0, 2*np.pi), verbose=2)
    #print(f(ret.x)); assert False
    oneparammin(xn)
    while True:
        mat, mat_deriv = cDecompose.Optimization_Problem_Combined_Unitary(xn)
        cost = costfunc(mat)
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
        #damping = lbda*np.eye(num_of_parameters, dtype=J.dtype)
        damping = lbda*np.diag(np.diag(genJ))
        try:
            delta = np.linalg.lstsq(genJ - damping, -grad, rcond=None)[0]
            #delta = np.linalg.solve(genJ, -grad)
        except np.linalg.LinAlgError:
            print("Singular matrix", np.linalg.matrix_rank(J.T @ J), num_of_parameters)
            delta = np.linalg.lstsq(genJ, -grad, rcond=None)[0]
            #xn = np.random.random(num_of_parameters)*2*np.pi
            #continue
        drctn = np.dot(grad, delta)
        lr, newc = wolfe_conditions_line(f, xn, delta, cost, drctn)
        if drctn < 0 and newc < cost - 0.0001 and cost - newc >= -0.5 * drctn * lr:
            xn = (xn + delta * lr)# % (2 * np.pi)
            if lbda > 1e-7: lbda = lbda / 10
        else:
            oneparammin(xn)
            #xn = np.random.random(num_of_parameters)*2*np.pi
            #continue
            """
            while True:
                amount = np.random.randint(num_of_parameters // 10, num_of_parameters // 5)
                idxes = np.random.choice(num_of_parameters, amount)
                mask = np.zeros(num_of_parameters); mask[idxes] = True
                rand = np.zeros(num_of_parameters); rand[idxes] = np.random.random(amount)*2*np.pi
                if True: #f(np.where(mask, rand, xn)) < cost:
                    xn = np.where(mask, rand, xn)
                    break
            """
            if lbda < 0.1: lbda = lbda * 10
        #xn = np.real(xn + np.linalg.solve(J.conjugate().T @ J, -J.conjugate().T @ mat)) % (2 * np.pi)      
        #Jplus = np.linalg.inv(J.conjugate().T @ J) @ J.conjugate().T
        #xn = xn - np.real(Jplus @ mat) % (2 * np.pi)
        
    
#train_model(num_of_parameters, 100000)
#transformer_model()
newton_method()
