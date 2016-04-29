//
//  tinydl.cpp
//  tiny deep learning - very small neural network simulator
//
//  Created by Matt Oberdorfer.
//  Copyright (c) 2016 Matt Oberdorfer. All rights reserved.
//
//  Many deep learning and neural network simulators have been written.
//  Many of them in C++. This one is was created to conduct educational
//  experiments, which is why it is very small, configurable and extendable.
//  You can create easily feedforward networks, recurrent networks
//  convolutional neural networks for classification and regression
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.


///////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <math.h>
#include <stdint.h>

///////////////////////////////////////////////////////////////////////////////////

#define RANDOM NAN

typedef unsigned long ulong;
typedef double ((*func_ptr)(double));
class neuron;

///////////////////////////////////////////////////////////////////////////////////

// connection between two neurons

class connection {
    friend class net;
    friend class layer;
    friend class neuron;
    
    public:
        connection(double init_weight);
    
    private:
        double  weight;
        double  actual;
        double *actual_ptr;
        double  delta_weight, prev_delta_weight;
        double  error, prev_error, prev_avg_error;
        neuron *input_neuron;
        ulong   batch_size;
};


// randS - ultra-simple pseudo-random generator to ensure repeatability on other platforms
// https://en.wikipedia.org/wiki/Linear_feedback_shift_register

uint16_t lfsr = 0xACE1u, bit;
uint16_t randS() {
    bit  = ((lfsr >> 0) ^ (lfsr >> 2) ^ (lfsr >> 3) ^ (lfsr >> 5) ) & 1;
    return lfsr =  (lfsr >> 1) | (bit << 15);
}

// randomWeight - Returns a pseudo random double in the range [0.0..max_distribution ] - offset

double randomWeight(double max_distribution, double offset) {
    return (randS() / double(0xFFFF)) * max_distribution - offset;
}

// connection constructor
// @ init_weight : initial weight or RANDOM -> random value between -1.0 and 1.0

connection::connection(double init_weight) {
    delta_weight = prev_delta_weight = 0;
    error = prev_error = prev_avg_error = 0;
    weight =  isnan(init_weight) ? randomWeight(1, 0) : init_weight;
    actual_ptr = &actual;
    input_neuron = NULL;
    actual = 1.0; // default for bias
    batch_size = 0;
}

///////////////////////////////////////////////////////////////////////////////////

class neuron {
    friend class net;
    friend class layer;
    
    public:
        neuron(ulong layer_no, ulong neuron_no, func_ptr *trans_func, double eta, double alpha);
        void feedForward();
        void backpropagateError();
        void updateWeights();
    
    private:
        ulong layer_id;
        ulong neuron_id;
    
        connection  **in_cons_ptr_table;
        ulong in_con_num;
        connection  **out_cons_ptr_table;
        ulong out_con_num;
    
        func_ptr *transFunc;
    
        double  net;
        double  out;
        double  eta;
        double  alpha;
        double  prev_error_in, last_prev_error_in;
};

// neuron constructor
// @layer_no : index of layer
// @neuron_no: index of neuron in this layer
// @transFunc: pointer to list of transfer function pointers
//             1st is transfer function, 2nd is derivate of transfer function
// @eta      : initial learning rate for this neuron
// @alpha    : initial momentum rate for this neuron

double IdentityFunc(double x) {return x;}
func_ptr IdentityFunc_ptr_table[] = {IdentityFunc, IdentityFunc};

neuron::neuron(ulong layer_no, ulong neuron_no, func_ptr *trans_func, double eta, double alpha) {
    // default is the identity function
    if( trans_func == NULL)
        trans_func = &IdentityFunc_ptr_table[0];
    transFunc  = trans_func;
    net = out  = 0;
    this->eta  = eta;
    this->alpha = alpha;
    in_con_num = out_con_num = 0;
    in_cons_ptr_table = out_cons_ptr_table = NULL;
    layer_id   = layer_no;
    neuron_id  = neuron_no;
    prev_error_in = last_prev_error_in = 1;
}

// propagate - retrieve input, calculate output and send actual output

void neuron::feedForward() {
    
    // fan in net input
    net = 0;
    for(int j=0; j < in_con_num; j++) {
        net += *(*(in_cons_ptr_table+j))->actual_ptr;
    }
    
    // activate
    out =  (*(*transFunc))(net);

    // fan out actual values
    for(int j=0; j<out_con_num; j++) {
        
        // calculate and store actual output in outgoing connection
        *(*(out_cons_ptr_table+j))->actual_ptr  = out * (*(out_cons_ptr_table+j))->weight;
        
    }
}


// backpropagateError - retrieve error (target-input), calculate and send error and weight updates

void neuron::backpropagateError() {
 
    double error_out, error_in = 0;
    double prev_out, delta_weight;
    connection *con;
    
    // fan in error for this neuron
    if(out_con_num > 0) {
        for(int j=0; j<out_con_num; j++) {
            error_in     += (*(out_cons_ptr_table+j))->error * (*(out_cons_ptr_table+j))->weight;
        }
    }
    
    error_out =  (error_in) * (*(*(transFunc+1)))(out);
    
    // fan out
    for(int j=0; j < in_con_num; j++) {
        
        con = (*(in_cons_ptr_table+j));
        
        // propagate (error * derivate of transfer function)
        con->error      = error_out;
        
        // if this incoming connection does NOT have an input neuron
        if( con->input_neuron != NULL)
            prev_out = con->input_neuron->out;
        else // otherwise just get the input value
            prev_out = *con->actual_ptr;
        
        // store updates to weights : eta * error * derivate of transfer(current out) * prev out + (if set) momentum
        delta_weight = - eta * con->error * prev_out + alpha * con->prev_delta_weight;
        
        // avoid local minima: keep the momentum by remembering previous delta weight
        con->prev_delta_weight   = delta_weight;
        con->delta_weight       += delta_weight;
        con->batch_size++;        
    }
}

// updateWeights - updates all weights for this neuron

void neuron::updateWeights() {
    // fan out actual values
    for(int j=0; j<out_con_num; j++) {
        connection *con =(*(out_cons_ptr_table+j));
        
        // updates the weights with the deltas from the last back proppagation
        if(con->batch_size) {
            con->weight      += con->delta_weight/con->batch_size;
            con->batch_size   = 0;
        }
        con->delta_weight =  con->prev_delta_weight = 0;
    }
}

///////////////////////////////////////////////////////////////////////////////////

#define INPUT_LAYER     1
#define HIDDEN_LAYER    2
#define OUTPUT_LAYER    3

class layer {
    friend class net;
    
    public:
        layer(ulong layer_no, ulong neuron_num, func_ptr *func_ptr, double eta, double alpha);
        ~layer();
    
    private:
        ulong neuron_num;
        neuron **neuron_ptr_table;
        int   layer_type;  // 0 input, 1, hidden, 2, output
};

// layer - constructor
// @layer_no  : index of this layer
// @neuron_num: number of neurons in this layer
// @transFunc : pointer to list of transfer function pointers
//              1st is transfer function, 2nd is derivate of transfer function
// @eta       : initial learning rate for this layer
// @alpha     : initial momentum rate for this layer

layer::layer(ulong layer_no, ulong neuron_num, func_ptr *trans_func, double eta,  double alpha) {
    this->neuron_num = neuron_num;
    this->layer_type = HIDDEN_LAYER;
    this->neuron_ptr_table = (neuron **) malloc(sizeof(neuron *) * neuron_num);
    for(int i =0; i < neuron_num; i++)
        *(this->neuron_ptr_table+i) = new neuron(layer_no, i, trans_func, eta, alpha);
}

///////////////////////////////////////////////////////////////////////////////////

class net {
    public:
        net(ulong layer_num, ulong *neuron_num, func_ptr *trans_func, double *eta, double *alpha);
        ~net();
        void connectLayers(ulong layer_1, ulong layer_2, double *weight_table, double bias_weight);
        void connectInputLayer(ulong layer_no, double **input_ptr_table);
        void connectOutputLayer(ulong layer_no);
        void connectNeurons(neuron *neuron_1, neuron *neuron_2, double weight);
        void setInputLayer(ulong layer_no, double *input_table);
        double setTargetOutputLayer(ulong layer_no, double *target_table);
        ulong feedForward(double *results);
        void backpropagateError();
        void updateWeights();
        void showValues(neuron *n);
        void displayValues();
    
    private:
        ulong   layer_num;
        layer **layer_ptr_table;
        neuron *bias;
};

// net - constructor
// @layer_num : number of layers in this net
// @neuron_num: pointer to a list that specifies how many neurons are in each layer
// @transFunc : pointer to list of (number of layers x 2) transfer function pointers
//              2 pointers for each layer
//              each 1st is transfer function, 2nd is derivate of transfer function
// @eta       : list of initial learning rate for each layer of this net
// @alpha     : list of initial momentum rate for each layer of this net

net::net(ulong layer_num, ulong *neuron_num, func_ptr *trans_func, double *eta, double *alpha) {
    this->layer_num = layer_num;
    bias = new neuron(-1,-1, NULL, ((eta == NULL)? 0.5 : *eta),  ((alpha == NULL)?0.0:*alpha));
    // create a single input connection for the bias neuron with 1.0
    bias->in_cons_ptr_table = (connection  **)malloc(sizeof(connection *));
    bias->in_con_num = 1;
    // incoming connection weight for input neurons is always 1.0
    *(bias->in_cons_ptr_table) = new connection(1.0);
    *(*(bias->in_cons_ptr_table))->actual_ptr = 1.0;

    layer_ptr_table = (layer **) malloc(sizeof(layer *) * layer_num);
    for(int i =0; i < layer_num; i++)
        *(this->layer_ptr_table+i) = new layer(i, *(neuron_num+i), (trans_func+i*2), ((eta == NULL)? 0.5 : *(eta+i)), ((alpha == NULL)?0.0:*(alpha+i)));
}

// connectInputLayer - create input layer and input sources for all neurons in this layer
//                  opitonally point the input neurons to external sources
// layer_no       : index of layer
// input_ptr_table: pointer to a list of addresses of double input values
//                  the net will read the actual value store at these addresses
//                  if NULL - actual will contain the input value

void net::connectInputLayer(ulong layer_no, double **input_ptr_table) {
    layer *l = *(this->layer_ptr_table+layer_no);
    neuron *n;
    l->layer_type = INPUT_LAYER;

    for(int i=0; i < l->neuron_num; i++) {
        n = *(l->neuron_ptr_table+i);
        n->in_cons_ptr_table = (connection  **)malloc(sizeof(connection *));
        n->in_con_num = 1;
        // incoming connection weight for input neurons is always 1.0
        *(n->in_cons_ptr_table) = new connection(1.0);
        if( input_ptr_table != NULL)
          (*(n->in_cons_ptr_table))->actual_ptr = input_ptr_table[i];
    }
}

// setInputLayer - set input value for all neurons in this layer
// layer_no       : index of layer
// input_table    : pointer to a list of double input values

void net::setInputLayer(ulong layer_no, double *input_table) {
    layer *l = *(this->layer_ptr_table+layer_no);
    neuron *n;
    
    for(int i=0; i < l->neuron_num; i++) {
        n = *(l->neuron_ptr_table+i);
        *(*(n->in_cons_ptr_table))->actual_ptr = *input_table++;
    }
}

// connectOutputLayer - point the input neurons to their sources
// @layer_no       : index of layer

void net::connectOutputLayer(ulong layer_no) {
    layer *l = *(this->layer_ptr_table+layer_no);
    neuron *n;
    l->layer_type = OUTPUT_LAYER;

    for(int i=0; i < l->neuron_num; i++) {
        n = *(l->neuron_ptr_table+i);
        n->out_cons_ptr_table = (connection  **)malloc(sizeof(connection *));
        n->out_con_num = 1;
        *(n->out_cons_ptr_table) = new connection(1);
    }
}

// setTargetOutputLayer
// @layer_no       : index of layer
// @target_table - list of double value (target)
// returns the total error per neuron (of the previous feed forward step)

double net::setTargetOutputLayer(ulong layer_no, double *target_table) {
    
    layer *l = *(this->layer_ptr_table+layer_no);
    double err =0;
    neuron *n;
    
    double net_error = 0;
    
    for(int i=0; i < l->neuron_num; i++) {
        n = *(l->neuron_ptr_table+i);
        (*(n->out_cons_ptr_table))->error = err = -(*(target_table+i) - n->out);
        net_error += err * err;
    }
    
    net_error /= ( 2 * l->neuron_num );
    
    return net_error;
}

// connectLayers - fully connect two layers of the net
// @layer_1       : index of layer 1
// @layer_2       : index of layer 2
// @weight_table  : pointer to list of z ( = x: num of neurons from layer 1 x y: num of neurons from layer 2)
//                  weight values used to initialize weight between the layers
//                  if ptr  = NULL -> random values between 0.0 and 1.0
//                  otherwise: the first y values correspond with the 1st of x neurons of layer 1
//                             the second y values correspond with the 2nd of x neurons of layer 1
// @bias_weight   : initial weight value for bias neuron weights (can be RANDOM)

void net::connectLayers(ulong layer_1, ulong layer_2, double *weight_table, double bias_weight) {
     layer *l1 = *(layer_ptr_table+layer_1),
           *l2 = *(layer_ptr_table+layer_2);
     neuron *n, *n1, *n2;
     connection *con;
     
     for(int i=0; i < l1->neuron_num; i++) {
         n = *(l1->neuron_ptr_table+i);
         n->out_cons_ptr_table = (connection  **)realloc(n->out_cons_ptr_table, sizeof(connection *) *
         (n->out_con_num + l2->neuron_num));
     }
     
     for(int i=0; i < l2->neuron_num; i++) {
         n = *(l2->neuron_ptr_table+i);
         
         // connect first input to the bias neuron
         if( n->in_con_num == 0) {
             connection *bias_con = new connection(bias_weight);
             
             bias->out_cons_ptr_table = (connection  **)realloc(bias->out_cons_ptr_table, sizeof(connection *) * (bias->out_con_num+1) );
             
             *(bias->out_cons_ptr_table  + bias->out_con_num++) = bias_con;
             
             n->in_cons_ptr_table = (connection  **)realloc(n->in_cons_ptr_table, sizeof(connection *) * (n->in_con_num+1) );
             
             *(n->in_cons_ptr_table  + n->in_con_num++) = bias_con;
             bias_con->input_neuron = bias;
         }
         
         n->in_cons_ptr_table = (connection  **)realloc(n->in_cons_ptr_table, sizeof(connection *) *
                                                            (n->in_con_num + l1->neuron_num));
     }
    
     for(int i=0; i < l1->neuron_num; i++)
         for(int j=0; j < l2->neuron_num; j++) {
         
             if( weight_table != NULL)
                 con = new connection(*(weight_table+i*l2->neuron_num+j));
             else
                 con = new connection(RANDOM);
             
             n1 = *(l1->neuron_ptr_table+i);
             n2 = *(l2->neuron_ptr_table+j);
             
             con->input_neuron = n1;
             *(n1->out_cons_ptr_table+n1->out_con_num+j) = con;
             *(n2->in_cons_ptr_table+n2->in_con_num+i)   = con;
         
         }
    
     for(int i=0; i<l1->neuron_num; i++) {
         n = *(l1->neuron_ptr_table+i);
         n->out_con_num += l2->neuron_num;
     }
     
     for(int j=0; j<l2->neuron_num; j++) {
         n = *(l2->neuron_ptr_table+j);
         n->in_con_num += l1->neuron_num;
     }

}


// connectNeurons - connect two specific neurons
// @neuron_1 - @neuron_2: neurons to connect
// @weight: initializing weight -> either RANDOM (NAN) -> random values between 0.0 and 1.0 OR with value of weight

void net::connectNeurons(neuron *neuron_1, neuron *neuron_2, double weight) {
    
    connection *con = new connection(weight);

    if( neuron_1->out_con_num == 0)
        neuron_1->out_cons_ptr_table = (connection  **)malloc(sizeof(connection *));
    else
        neuron_1->out_cons_ptr_table = (connection  **)realloc(neuron_1->out_cons_ptr_table, sizeof(connection *) * (neuron_1->out_con_num+1) );
    
    *(neuron_1->out_cons_ptr_table  + neuron_1->out_con_num) = con;
    neuron_1->out_con_num++;
    
    if( neuron_2->in_con_num == 0) {
        neuron_2->in_cons_ptr_table = (connection  **) malloc(sizeof(connection *) * 2);
        // create bias input connection
        connection *bias = new connection(weight);
        *(neuron_2->in_cons_ptr_table  + neuron_2->in_con_num ++) = bias;
    } else
        neuron_2->in_cons_ptr_table = (connection  **)realloc(neuron_2->in_cons_ptr_table, sizeof(connection *) * (neuron_2->in_con_num+1) );
    
    *(neuron_2->in_cons_ptr_table  + neuron_2->in_con_num) = con;
    neuron_2->in_con_num++;
    
}

// propagate - get input and propagate values through the entire network
// @results : pointer to buffer where resulting output of net is stored

ulong net::feedForward(double *results) {
    layer *l;
    ulong res_count = 0;
    
    bias->feedForward();
    
    for(long k = 0; k < layer_num; k++) {
        l= *(layer_ptr_table+k);
        for(long i=0; i < l->neuron_num; i++) {
            (*(l->neuron_ptr_table+i))->feedForward();
            if(l->layer_type == OUTPUT_LAYER)
                *(results + res_count++) = (*(l->neuron_ptr_table+i))->out;
        }
    }
    return res_count;
}

// backpropagateError - determine the error and propagate it back through the entire network

void net::backpropagateError() {
    layer *l1;
    
    for(long k = layer_num-1; k >= 0; k--) {
        l1 = *(layer_ptr_table+k);
        for(int i=0; i < l1->neuron_num; i++) {
            (*(l1->neuron_ptr_table+i))->backpropagateError();
        }
    }
    bias->backpropagateError();
}

// updateWeights - update weights throughout the entire network

void net::updateWeights() {
    layer *l1;
    
    for(long k = layer_num-1; k >= 0; k--) {
        l1 = *(layer_ptr_table+k);
        for(int i=0; i < l1->neuron_num; i++) {
            (*(l1->neuron_ptr_table+i))->updateWeights();
        }
    }
    bias->updateWeights();
}

// showValues - print internal values of neuron and connections
// @n - pointer to neuron to be displayed

void net::showValues(neuron *n) {
    
    printf("--------------------------------------------------\n");
    printf("**** L: %ld N: %ld ****\n", n->layer_id, n->neuron_id);
    for(long i=0; i < n->in_con_num; i++) {
        printf("IN %ld > w: %lf a: %lf\n",i ,(*(n->in_cons_ptr_table+i))->weight, *(*(n->in_cons_ptr_table+i))->actual_ptr);
    }
    printf("--> net: %lf | out: %lf\n", n->net, n->out);
    for(long i=0; i < n->out_con_num; i++) {
        printf("OT %ld > w: %lf a: %lf\n",i ,(*(n->out_cons_ptr_table+i))->weight, *(*(n->out_cons_ptr_table+i))->actual_ptr);
    }
}

// displayValues - print internal values of all neurons in net

void net::displayValues() {
    layer *l;
    
    showValues(bias);
    
    for(long k = 0; k < layer_num; k++) {
        l= *(layer_ptr_table+k);
        for(long i=0; i < l->neuron_num; i++) {
            showValues((*(l->neuron_ptr_table+i)));
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////

// tanh is a sigmoid curve ranges from -1 to +1
double TanH(double net) {
    return tanh(net);
}
double TanHDer(double net) {
    return 1-tanh(net)*tanh(net);
}
double TanHInv(double x) {
    return atanh(x);
}

// identity function delivers
double Identity(double net) {
    return net;
}
double IdentityDer(double net) {
    return 1;
}
double IdentityInv(double net) {
    return net;
}

// sigmoid curve that ranges 0.0 to 1.0:
double Sig(double x) {
    return (1/(1+(exp(-x))));
}
double SigDer(double x) {
    return x * (1-x);
}

// gaussian curve
double Gaussian(double x) {
    return exp(-((x * x) / 2.0));
}
double GaussianDer(double x) {
    return -x * exp(-(x * x) / 2.0);
}

