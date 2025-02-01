/**
  ******************************************************************************
  * @file    model.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifdef __cplusplus
extern "C" {
#endif

#ifndef SINGLE_FILE
#include "number.h"
#include "model.h"
// #include <chrono>

 // InputLayer is excluded
#include "max_pooling1d_132.c" // InputLayer is excluded
#include "conv1d_108.c"
#include "weights/conv1d_108.c" // InputLayer is excluded
#include "max_pooling1d_133.c" // InputLayer is excluded
#include "conv1d_109.c"
#include "weights/conv1d_109.c" // InputLayer is excluded
#include "max_pooling1d_134.c" // InputLayer is excluded
#include "conv1d_110.c"
#include "weights/conv1d_110.c" // InputLayer is excluded
#include "max_pooling1d_135.c" // InputLayer is excluded
#include "conv1d_111.c"
#include "weights/conv1d_111.c" // InputLayer is excluded
#include "max_pooling1d_136.c" // InputLayer is excluded
#include "flatten_17.c" // InputLayer is excluded
#include "dense_17.c"
#include "weights/dense_17.c"
#endif


void cnn(
  const input_t input,
  dense_17_output_type dense_17_output) {
  
  // Output array allocation
  static union {
    max_pooling1d_132_output_type max_pooling1d_132_output;
    max_pooling1d_133_output_type max_pooling1d_133_output;
    max_pooling1d_134_output_type max_pooling1d_134_output;
    max_pooling1d_135_output_type max_pooling1d_135_output;
    max_pooling1d_136_output_type max_pooling1d_136_output;
    flatten_17_output_type flatten_17_output;
  } activations1;

  static union {
    conv1d_108_output_type conv1d_108_output;
    conv1d_109_output_type conv1d_109_output;
    conv1d_110_output_type conv1d_110_output;
    conv1d_111_output_type conv1d_111_output;
  } activations2;


// Model layers call chain 
  
  
  max_pooling1d_132( // First layer uses input passed as model parameter
    input,
    activations1.max_pooling1d_132_output
    );
  
  
  conv1d_108(
    activations1.max_pooling1d_132_output,
    conv1d_108_kernel,
    conv1d_108_bias,
    activations2.conv1d_108_output
    );
  
  
  max_pooling1d_133(
    activations2.conv1d_108_output,
    activations1.max_pooling1d_133_output
    );
  
  
  conv1d_109(
    activations1.max_pooling1d_133_output,
    conv1d_109_kernel,
    conv1d_109_bias,
    activations2.conv1d_109_output
    );
  
  
  max_pooling1d_134(
    activations2.conv1d_109_output,
    activations1.max_pooling1d_134_output
    );
  
  
  conv1d_110(
    activations1.max_pooling1d_134_output,
    conv1d_110_kernel,
    conv1d_110_bias,
    activations2.conv1d_110_output
    );
  
  
  max_pooling1d_135(
    activations2.conv1d_110_output,
    activations1.max_pooling1d_135_output
    );
  
  
  conv1d_111(
    activations1.max_pooling1d_135_output,
    conv1d_111_kernel,
    conv1d_111_bias,
    activations2.conv1d_111_output
    );
  
  
  max_pooling1d_136(
    activations2.conv1d_111_output,
    activations1.max_pooling1d_136_output
    );
  
  
  flatten_17(
    activations1.max_pooling1d_136_output,
    activations1.flatten_17_output
    );
  
  
  dense_17(
    activations1.flatten_17_output,
    dense_17_kernel,
    dense_17_bias,// Last layer uses output passed as model parameter
    dense_17_output
    );
}

#ifdef __cplusplus
} // extern "C"
#endif