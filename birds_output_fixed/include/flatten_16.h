/**
  ******************************************************************************
  * @file    flatten.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _FLATTEN_16_H_
#define _FLATTEN_16_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define OUTPUT_DIM 224

typedef int16_t flatten_16_output_type[OUTPUT_DIM];

#if 0
void flatten_16(
  const number_t input[7][32], 			      // IN
	number_t output[OUTPUT_DIM]); 			                // OUT
#endif

#undef OUTPUT_DIM

#endif//_FLATTEN_16_H_