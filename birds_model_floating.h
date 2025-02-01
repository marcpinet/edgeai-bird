#define SINGLE_FILE
/**
  ******************************************************************************
  * @file    defines.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, Université Côte d'Azur, LEAT, France
  * @version 2.1.0
  * @date    10 january 2024
  * @brief   Global C pre-processor definitions to use to build all source files (incl. CMSIS-NN)
  */

/* CMSIS-NN round mode definition */
#if defined(WITH_CMSIS_NN) || defined(WITH_NMSIS_NN)



#error "Unrecognized round mode, only floor and nearest are supported by CMSIS-NN"

#endif // defined(WITH_CMSIS_NN) || defined(WITH_NMSIS_NN)
/**
  ******************************************************************************
  * @file    number.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    2 february 2021
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifdef __cplusplus
extern "C" {
#endif

#ifndef __NUMBER_H__
#define __NUMBER_H__

#include <stdint.h>
#include <stddef.h>
#include <math.h>

#ifdef TRAPV_SHIFT
#include <limits.h>
#include <stdio.h>
#include <assert.h>
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#endif

#define _clamp_to(type, number) clamp_to_number_t_ ## type (number)
#define clamp_to(type, number) _clamp_to(type, number)
#define _scale(type, number, scale_factor, round_mode) scale_number_t_ ## type (number, scale_factor, round_mode)
#define scale(type, number, scale_factor, round_mode) _scale(type, number, scale_factor, round_mode)
#define _scale_and_clamp_to(type, number, scale_factor, round_mode) scale_and_clamp_to_number_t_ ## type (number, scale_factor, round_mode)
#define scale_and_clamp_to(type, number, scale_factor, round_mode) _scale_and_clamp_to(type, number, scale_factor, round_mode)

typedef enum {
  ROUND_MODE_NONE,
  ROUND_MODE_FLOOR,
  ROUND_MODE_NEAREST,
} round_mode_t;

// Idea 1: Write the smallest min max interval of the net, could be an issue for hybrid int type network
// Idea 2: listing any interval and add type in name in a switch case like <- better but painfull
// #define NUMBER_MIN		// Max value for this numeric type
// #define NUMBER_MAX		// Min value for this numeric type

// // Idea 1: List of all types and write any corresponding function 
// typedef  number_t;		// Standard size numeric type used for weights and activations
// typedef  long_number_t;	// Long numeric type used for intermediate results

#define NUMBER_MIN_FLOAT -2147483648
#define NUMBER_MAX_FLOAT 2147483647

static inline float min_float(
    float a,
    float b) {
	if (a <= b)
		return a;
	return b;
}

static inline float max_float(
    float a,
    float b) {
	if (a >= b)
		return a;
	return b;
}

static inline float scale_number_t_float(
  float number, int scale_factor, round_mode_t round_mode) {
	return number;
}
static inline float clamp_to_number_t_float(
  float number) {
	return (float) number;
}
static inline float scale_and_clamp_to_number_t_float(
  float number, int scale_factor, round_mode_t round_mode) {
	return (float) number;
}


#define NUMBER_MIN_INT32_T -2147483648
#define NUMBER_MAX_INT32_T 2147483647

static inline int64_t min_int32_t(
    int64_t a,
    int64_t b) {
	if (a <= b)
		return a;
	return b;
}

static inline int64_t max_int32_t(
    int64_t a,
    int64_t b) {
	if (a >= b)
		return a;
	return b;
}

static inline int64_t scale_number_t_int32_t(
  int64_t number, int scale_factor, round_mode_t round_mode) {


  if (scale_factor <= 0) {
#ifdef TRAPV_SHIFT
    // Check for possible overflow of left shift
    if (number > INT64_MAX >> -scale_factor) {
      fprintf(stderr,
              "Error: scale() overflow, number=%ld, scale_factor=%d, limit=%d\n",
              number,
              scale_factor,
              INT16_MAX >> -scale_factor);
      assert(number <= INT64_MAX >> -scale_factor);
    }
#endif
    // No rounding to apply when shifting left
    return number << - scale_factor;
  } else {
    if (round_mode == ROUND_MODE_NEAREST) {
      number += (1 << (scale_factor - 1)); // +0.5 in fixed-point
    }
    return number >> scale_factor;
  }
}
static inline int32_t clamp_to_number_t_int32_t(
  int64_t number) {
	return (int32_t) max_int32_t(
      NUMBER_MIN_INT32_T,
      min_int32_t(
        NUMBER_MAX_INT32_T, number));
}
static inline int32_t scale_and_clamp_to_number_t_int32_t(
  int64_t number, int scale_factor, round_mode_t round_mode) {
#ifdef WITH_CMSIS_NN
  // Not really CMSIS-NN but use SSAT anyway
  if (scale_factor <= 0) {
    // No rounding to apply when shifting left
    return __SSAT(number << - scale_factor, sizeof(int32_t) * 8);
  } else {
    if (round_mode == ROUND_MODE_NEAREST) {
      number += (1 << (scale_factor - 1)); // +0.5 in fixed-point
    }
    return __SSAT(number >> scale_factor, sizeof(int32_t) * 8);
  }
#else
  number = scale_number_t_int32_t(number, scale_factor, round_mode);
  return clamp_to_number_t_int32_t(number);
#endif
}




static inline void int64_t_to_float(int64_t * tabint, float * tabfloat, long tabdim, int scale_factor){
  for (int i=0; i<tabdim; i++){
    tabfloat[i] = (float)tabint[i] / (1<<scale_factor);
  }
}

static inline void int32_t_to_float(int32_t * tabint, float * tabfloat, long tabdim, int scale_factor){
  for (int i=0; i<tabdim; i++){
    tabfloat[i] = (float)tabint[i] / (1<<scale_factor);
  }
}

static inline void int16_t_to_float(int16_t * tabint, float * tabfloat, long tabdim, int scale_factor){
  for (int i=0; i<tabdim; i++){
    tabfloat[i] = ((float)tabint[i]) / (1<<scale_factor);
  }
}

static inline void int8_t_to_float(int8_t * tabint, float * tabfloat, long tabdim, int scale_factor){
  for (int i=0; i<tabdim; i++){
    tabfloat[i] = ((float)tabint[i]) / (1<<scale_factor);
  }
}
#endif //__NUMBER_H__

#ifdef __cplusplus
} // extern "C"
#endif
/**
  ******************************************************************************
  * @file    maxpool1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _MAX_POOLING1D_132_H_
#define _MAX_POOLING1D_132_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  1
#define INPUT_SAMPLES   16000
#define POOL_SIZE       2
#define POOL_STRIDE     2
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

typedef float max_pooling1d_132_output_type[POOL_LENGTH][INPUT_CHANNELS];

#if 0
void max_pooling1d_132(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  number_t output[POOL_LENGTH][INPUT_CHANNELS]); 	// OUT
#endif

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH

#endif//_MAX_POOLING1D_132_H_
/**
  ******************************************************************************
  * @file    maxpool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "max_pooling1d_132.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#endif

#define INPUT_CHANNELS  1
#define INPUT_SAMPLES   16000
#define POOL_SIZE       2
#define POOL_STRIDE     2
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

// For fixed point quantization
#define INPUT_SCALE_FACTOR 0
#define OUTPUT_SCALE_FACTOR 0
#define OUTPUT_ROUND_MODE ROUND_MODE_NONE
#define NUMBER_T float
#define LONG_NUMBER_T float


static inline void max_pooling1d_132(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  NUMBER_T output[POOL_LENGTH][INPUT_CHANNELS]) {	// OUT

  unsigned short pos_x, k; 	// loop indexes for output volume
  unsigned int x;
  static LONG_NUMBER_T max[INPUT_CHANNELS];

  for (pos_x = 0; pos_x < POOL_LENGTH; pos_x++) {
    for (k = 0; k < INPUT_CHANNELS; k++) {
#ifdef ACTIVATION_LINEAR
      max[k] = input[pos_x*POOL_STRIDE][k];
      x = 1;
#elif defined(ACTIVATION_RELU)
      max[k] = 0;
      x = 0;
#else
#error "Unsupported activation function"
#endif
    }

    for (; x < POOL_SIZE; x++) {
      for (k = 0; k < INPUT_CHANNELS; k++) {
        if (max[k] < input[(pos_x * POOL_STRIDE) + x][k])
          max[k] = input[(pos_x * POOL_STRIDE) + x][k];
      }
    }

    for (k = 0; k < INPUT_CHANNELS; k++) {
      output[pos_x][k] = scale_and_clamp_to(NUMBER_T, max[k], INPUT_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
    }
  }
}

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH
#undef ACTIVATION_LINEAR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    conv1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _CONV1D_108_H_
#define _CONV1D_108_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      1
#define INPUT_SAMPLES       8000
#define CONV_FILTERS        8
#define CONV_KERNEL_SIZE    40
#define CONV_STRIDE         4

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

typedef float conv1d_108_output_type[CONV_OUTSAMPLES][CONV_FILTERS];

#if 0
void conv1d_108(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const number_t kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS],  // IN

  const number_t bias[CONV_FILTERS],						                          // IN

  number_t output[CONV_OUTSAMPLES][CONV_FILTERS]);                       // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES

#endif//_CONV1D_108_H_
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Sébastien Bilavarn, LEAT, CNRS, Université Côte d'Azur, France
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "conv1d_108.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_CHANNELS      1
#define INPUT_SAMPLES       8000
#define CONV_FILTERS        8
#define CONV_KERNEL_SIZE    40
#define CONV_STRIDE         4
#define CONV_GROUPS         1
#define CHANNELS_PER_GROUP  (INPUT_CHANNELS / CONV_GROUPS)
#define FILTERS_PER_GROUP   (CONV_FILTERS / CONV_GROUPS)

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 0
#define BIASES_SCALE_FACTOR 0
#define TMP_SCALE_FACTOR 0
#define INPUT_SCALE_FACTOR 0
#define OUTPUT_SCALE_FACTOR 0
#define OUTPUT_ROUND_MODE ROUND_MODE_NONE
#define NUMBER_T float
#define LONG_NUMBER_T float


static inline void conv1d_108(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const NUMBER_T kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS],  // IN

  const NUMBER_T bias[CONV_FILTERS],						                          // IN

  NUMBER_T output[CONV_OUTSAMPLES][CONV_FILTERS]) {                       // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  int input_x;
  LONG_NUMBER_T output_acc;

  for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
    for (k = 0; k < CONV_FILTERS; k++) { 
      output_acc = 0;

      for (x = 0; x < CONV_KERNEL_SIZE; x++) {
        input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;

        if (input_x >= 0 && input_x < INPUT_SAMPLES) { // ZeroPadding1D
          for (z = 0; z < INPUT_CHANNELS / CONV_GROUPS; z++) {
            output_acc += (LONG_NUMBER_T)input[input_x][z + (k / FILTERS_PER_GROUP) * CHANNELS_PER_GROUP] * (LONG_NUMBER_T)kernel[k][x][z];
          }
        }
      }

    // Scale for possible additional precision of bias
    output_acc = scale(NUMBER_T, output_acc, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);

    // Scale bias to match accumulator
    output_acc += scale(NUMBER_T, (LONG_NUMBER_T)bias[k], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);

      
#ifdef ACTIVATION_LINEAR
      output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
      // Activation function: ReLU
      if (output_acc < 0) {
        output[pos_x][k] = 0;
      } else {
#if defined(ACTIVATION_RELU6)
        if (output_acc > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
          output_acc = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
        }
#endif
        output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      }
#else
#error "Unsupported activation function"
#endif
    }
  }

#else

#if BIASES_SCALE_FACTOR > WEIGHTS_SCALE_FACTOR
#error "CMSIS-NN does not support BIASES_SCALE_FACTOR larger than WEIGHTS_SCALE_FACTOR"
#endif

#error "Data type unsupported by CMSIS-NN"

#endif
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef CONV_GROUPS
#undef CHANNELS_PER_GROUP
#undef FILTERS_PER_GROUP
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_RELU
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef TMP_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    weights/conv1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_CHANNELS    1
#define CONV_FILTERS      8
#define CONV_KERNEL_SIZE  40
#define CONV_GROUPS       1


const float  conv1d_108_bias[CONV_FILTERS] = {-0x1.cd04600000000p-5, -0x1.c3abda0000000p-5, 0x1.541de80000000p-7, 0x1.b440c20000000p-7, -0x1.143aae0000000p-3, 0x1.0f68c80000000p-6, 0x1.13d8b20000000p-6, 0x1.16e5660000000p-6}
;

const float  conv1d_108_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{{-0x1.35de580000000p-4}
, {-0x1.6d3fbc0000000p-4}
, {-0x1.431ff40000000p-5}
, {0x1.7087000000000p-5}
, {-0x1.51b07c0000000p-6}
, {0x1.972ef80000000p-6}
, {-0x1.487ec20000000p-8}
, {-0x1.f5a7b60000000p-4}
, {-0x1.1b2a8a0000000p-5}
, {0x1.3b04380000000p-4}
, {-0x1.4ce9160000000p-5}
, {-0x1.00ce1c0000000p-8}
, {0x1.fe50dc0000000p-6}
, {-0x1.d443d80000000p-8}
, {0x1.fb74e20000000p-8}
, {-0x1.7c4fb20000000p-7}
, {0x1.ec3e500000000p-5}
, {0x1.dca7dc0000000p-4}
, {0x1.79d74c0000000p-4}
, {0x1.c6496e0000000p-6}
, {0x1.6fd8b60000000p-4}
, {0x1.4a00520000000p-5}
, {0x1.1eb0420000000p-3}
, {0x1.6bf3fa0000000p-9}
, {0x1.8080060000000p-6}
, {0x1.f795fc0000000p-5}
, {0x1.ba99b00000000p-5}
, {0x1.46c5460000000p-5}
, {0x1.0faae00000000p-4}
, {0x1.ea3d940000000p-4}
, {0x1.7796080000000p-5}
, {0x1.abf0900000000p-4}
, {-0x1.1113320000000p-5}
, {0x1.6b24c80000000p-5}
, {-0x1.dacfc00000000p-5}
, {0x1.c8b0ca0000000p-5}
, {-0x1.f0c7420000000p-5}
, {0x1.130e7c0000000p-5}
, {-0x1.2ab2780000000p-3}
, {-0x1.a06c460000000p-5}
}
, {{0x1.4e6d3a0000000p-3}
, {0x1.042a580000000p-3}
, {-0x1.4103ee0000000p-3}
, {-0x1.8e5ca80000000p-6}
, {0x1.61d69e0000000p-3}
, {-0x1.31a2f60000000p-3}
, {0x1.d71a120000000p-5}
, {-0x1.a364b00000000p-9}
, {0x1.3d1da80000000p-4}
, {0x1.dc5ed80000000p-5}
, {0x1.00d8e60000000p-4}
, {0x1.45d1160000000p-4}
, {0x1.b004260000000p-4}
, {-0x1.3d734e0000000p-6}
, {-0x1.fd94b80000000p-5}
, {0x1.e2f6120000000p-4}
, {0x1.1a5a440000000p-3}
, {0x1.bb4d8e0000000p-4}
, {-0x1.150cce0000000p-7}
, {-0x1.e3faf20000000p-5}
, {0x1.1c4a200000000p-3}
, {0x1.8317080000000p-8}
, {0x1.b89e8c0000000p-4}
, {0x1.5452140000000p-3}
, {-0x1.07ce560000000p-3}
, {-0x1.40948e0000000p-4}
, {0x1.f1bea40000000p-6}
, {0x1.aafc720000000p-4}
, {0x1.e47fb40000000p-5}
, {-0x1.46e9a40000000p-4}
, {-0x1.3dd59a0000000p-3}
, {0x1.09605a0000000p-6}
, {-0x1.4414b60000000p-4}
, {0x1.29a0040000000p-4}
, {0x1.eb2a7a0000000p-6}
, {-0x1.0c18ee0000000p-4}
, {-0x1.68007c0000000p-4}
, {-0x1.77dd1e0000000p-5}
, {0x1.429e5c0000000p-5}
, {-0x1.5dd6040000000p-4}
}
, {{-0x1.286eda0000000p+1}
, {-0x1.7b07920000000p+1}
, {-0x1.436e640000000p+1}
, {-0x1.7b8bca0000000p+1}
, {-0x1.5e73c20000000p+1}
, {-0x1.8c649c0000000p+1}
, {-0x1.57a5220000000p+1}
, {-0x1.277b8e0000000p+1}
, {-0x1.c6974a0000000p+0}
, {-0x1.561d720000000p+0}
, {-0x1.a1e2be0000000p-1}
, {-0x1.72745a0000000p+0}
, {-0x1.6702080000000p-1}
, {-0x1.49d5b40000000p+0}
, {-0x1.1db6fc0000000p+0}
, {-0x1.9d03060000000p+0}
, {-0x1.ab44000000000p-1}
, {-0x1.c944ac0000000p+0}
, {-0x1.058dce0000000p+0}
, {-0x1.91c0c60000000p+0}
, {-0x1.b40d7a0000000p+0}
, {-0x1.92ce9e0000000p+0}
, {-0x1.5f81f00000000p+0}
, {-0x1.c161bc0000000p+0}
, {-0x1.be71880000000p+0}
, {-0x1.5107fe0000000p+1}
, {-0x1.148c1c0000000p+1}
, {-0x1.0f580e0000000p+1}
, {-0x1.7cf05e0000000p+1}
, {-0x1.65b7420000000p+1}
, {-0x1.95d3e20000000p+1}
, {-0x1.b72e2e0000000p+1}
, {-0x1.a874c20000000p+1}
, {-0x1.f0eff20000000p+1}
, {-0x1.e419600000000p+1}
, {-0x1.a12f340000000p+1}
, {-0x1.bedfee0000000p+1}
, {-0x1.99974c0000000p+1}
, {-0x1.434a200000000p+1}
, {-0x1.7d78300000000p+1}
}
, {{0x1.3c0ab20000000p+0}
, {0x1.73930e0000000p+0}
, {0x1.ee34f60000000p+0}
, {0x1.41ef0c0000000p-1}
, {0x1.72c8660000000p+0}
, {0x1.07cc720000000p+0}
, {0x1.24c40c0000000p+1}
, {0x1.18a72c0000000p-1}
, {0x1.d8cdd20000000p+0}
, {0x1.8a1d180000000p+0}
, {0x1.6533160000000p+1}
, {0x1.2f4fea0000000p+1}
, {0x1.7dbcde0000000p+1}
, {0x1.c3c4ba0000000p-1}
, {0x1.7316900000000p+1}
, {0x1.e4ed300000000p+0}
, {0x1.d808400000000p+1}
, {0x1.ceb6720000000p+1}
, {0x1.735ff60000000p+0}
, {0x1.b8faa60000000p+1}
, {0x1.ac01a60000000p+1}
, {0x1.1749360000000p+1}
, {0x1.8e3bee0000000p+1}
, {0x1.0a31700000000p+2}
, {0x1.dbb3d20000000p-4}
, {0x1.d71ebc0000000p+1}
, {0x1.ee30aa0000000p+1}
, {0x1.abeb480000000p+0}
, {0x1.4d75c80000000p+1}
, {0x1.cc2bf60000000p+1}
, {0x1.a3ade00000000p+0}
, {0x1.e641140000000p+0}
, {0x1.ce260e0000000p+1}
, {0x1.4ca6a20000000p+1}
, {-0x1.dab5ec0000000p-1}
, {0x1.2321ee0000000p+1}
, {0x1.6786820000000p+1}
, {-0x1.7eba360000000p-2}
, {0x1.62ddb60000000p+1}
, {0x1.52dd920000000p+1}
}
, {{0x1.1ab5280000000p-3}
, {-0x1.026f400000000p-3}
, {0x1.6eeec40000000p-3}
, {-0x1.141e900000000p-2}
, {0x1.9e35e40000000p-4}
, {-0x1.3285a80000000p-4}
, {0x1.a7bf4c0000000p-3}
, {0x1.0a2c240000000p-4}
, {0x1.e376220000000p-4}
, {-0x1.7b6f860000000p-6}
, {0x1.ac30660000000p-4}
, {0x1.b08bbe0000000p-7}
, {0x1.2e75ae0000000p-5}
, {0x1.f547760000000p-5}
, {0x1.881bfa0000000p-5}
, {0x1.5614840000000p-4}
, {-0x1.5f64160000000p-3}
, {0x1.8c91420000000p-6}
, {-0x1.4d265c0000000p-4}
, {0x1.9306820000000p-3}
, {-0x1.3d8abc0000000p-3}
, {0x1.723cfe0000000p-4}
, {-0x1.5e3ed00000000p-4}
, {0x1.8695200000000p-4}
, {0x1.79c71a0000000p-5}
, {0x1.3028980000000p-3}
, {0x1.403ac20000000p-3}
, {-0x1.5d53020000000p-5}
, {0x1.5ce7b80000000p-3}
, {-0x1.0e12300000000p-3}
, {0x1.2ae7ce0000000p-4}
, {-0x1.567ed40000000p-4}
, {0x1.cc2ec80000000p-4}
, {0x1.a8f40c0000000p-4}
, {0x1.6e84920000000p-4}
, {0x1.0b01260000000p-4}
, {0x1.bf30e20000000p-5}
, {-0x1.8a7cd60000000p-5}
, {-0x1.4961ee0000000p-3}
, {-0x1.6a22fe0000000p-4}
}
, {{0x1.94d49c0000000p+1}
, {0x1.053b0c0000000p+1}
, {0x1.20d7ba0000000p+0}
, {0x1.21d8540000000p-1}
, {0x1.51d3080000000p+1}
, {-0x1.a57ab40000000p-7}
, {0x1.7803d40000000p+0}
, {0x1.15b9a80000000p+1}
, {0x1.d3820c0000000p+0}
, {0x1.e27e000000000p+0}
, {0x1.c9f6320000000p+1}
, {0x1.c222a00000000p-1}
, {0x1.e6a4ae0000000p+1}
, {-0x1.0de2800000000p+1}
, {0x1.cad39e0000000p+1}
, {0x1.566c320000000p+1}
, {0x1.8e45ee0000000p+1}
, {0x1.1b4cc40000000p+2}
, {-0x1.8124420000000p-2}
, {0x1.2625be0000000p+2}
, {0x1.ac014c0000000p-1}
, {0x1.6c0b480000000p+1}
, {0x1.a4c1560000000p+1}
, {0x1.cf89780000000p+1}
, {0x1.0740840000000p+1}
, {0x1.ddaf700000000p+1}
, {0x1.7e422e0000000p+1}
, {0x1.ee0b1a0000000p-1}
, {0x1.92ae240000000p+1}
, {0x1.48728e0000000p+1}
, {0x1.49aa300000000p+1}
, {0x1.f89d060000000p+0}
, {0x1.6d64c40000000p+1}
, {0x1.f5cf8a0000000p+0}
, {0x1.83248c0000000p-4}
, {0x1.08e0aa0000000p+1}
, {0x1.a31d760000000p+0}
, {0x1.e667e40000000p-1}
, {0x1.21387e0000000p+1}
, {0x1.4923940000000p+1}
}
, {{0x1.53ba160000000p+1}
, {0x1.4eff9c0000000p+0}
, {0x1.ba34e80000000p+0}
, {0x1.5125d20000000p+1}
, {0x1.a871880000000p-1}
, {0x1.3b691c0000000p+0}
, {0x1.dd7c420000000p+1}
, {-0x1.e941fe0000000p+0}
, {0x1.f708380000000p+1}
, {-0x1.02ebbe0000000p+2}
, {0x1.660d9c0000000p+2}
, {0x1.2fcf100000000p+0}
, {0x1.0a79ac0000000p+2}
, {0x1.0d440c0000000p+2}
, {0x1.7a350e0000000p+0}
, {0x1.2259c80000000p+2}
, {0x1.af1f0c0000000p-1}
, {0x1.18d78c0000000p+2}
, {-0x1.3327be0000000p-2}
, {0x1.f9c5360000000p+1}
, {0x1.5ace0c0000000p+1}
, {0x1.d3566e0000000p+1}
, {0x1.53afd20000000p+0}
, {0x1.499a580000000p+1}
, {0x1.83642c0000000p+1}
, {0x1.3c88fc0000000p+1}
, {0x1.e389920000000p+0}
, {0x1.fef6b80000000p+1}
, {0x1.1c6b600000000p+1}
, {0x1.370ad80000000p+0}
, {0x1.afebfe0000000p+1}
, {0x1.44f0ce0000000p+1}
, {-0x1.3e707a0000000p+0}
, {0x1.f15ec00000000p+0}
, {0x1.3a04700000000p+1}
, {0x1.6f74e80000000p-2}
, {0x1.610eaa0000000p+0}
, {0x1.541b500000000p+0}
, {0x1.0d05a40000000p-1}
, {0x1.475c400000000p+0}
}
, {{0x1.8c5aaa0000000p+1}
, {0x1.0a64840000000p+1}
, {0x1.6e979c0000000p+1}
, {0x1.dd8d440000000p-1}
, {0x1.a284ba0000000p+1}
, {-0x1.1da3d20000000p+0}
, {0x1.bebc4e0000000p+1}
, {0x1.6e8e360000000p+1}
, {0x1.9f8b100000000p+0}
, {0x1.4d2a000000000p+1}
, {0x1.8948520000000p+1}
, {0x1.e788ac0000000p+0}
, {0x1.01820a0000000p+2}
, {0x1.fff6280000000p+0}
, {0x1.08a7b20000000p+2}
, {-0x1.c71d740000000p-1}
, {0x1.2ac3580000000p+2}
, {0x1.40de0e0000000p+2}
, {-0x1.5cd1c00000000p+0}
, {0x1.79ac6a0000000p+2}
, {-0x1.494c860000000p+0}
, {0x1.e086280000000p+1}
, {0x1.ae8bac0000000p+1}
, {0x1.10c1740000000p+2}
, {0x1.60ab000000000p+1}
, {0x1.0027180000000p+1}
, {0x1.d6ab660000000p+1}
, {0x1.a510fe0000000p+1}
, {0x1.275e5c0000000p+1}
, {0x1.01e59e0000000p+2}
, {0x1.5b61220000000p+1}
, {-0x1.313e6c0000000p-2}
, {0x1.754a4c0000000p+1}
, {0x1.22ee880000000p+1}
, {-0x1.b3fce00000000p-3}
, {0x1.6f5e1e0000000p+0}
, {0x1.26541a0000000p+1}
, {-0x1.2883580000000p-1}
, {0x1.0dbf260000000p+1}
, {0x1.2acf220000000p+1}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_GROUPS
/**
  ******************************************************************************
  * @file    maxpool1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _MAX_POOLING1D_133_H_
#define _MAX_POOLING1D_133_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  8
#define INPUT_SAMPLES   1991
#define POOL_SIZE       4
#define POOL_STRIDE     4
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

typedef float max_pooling1d_133_output_type[POOL_LENGTH][INPUT_CHANNELS];

#if 0
void max_pooling1d_133(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  number_t output[POOL_LENGTH][INPUT_CHANNELS]); 	// OUT
#endif

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH

#endif//_MAX_POOLING1D_133_H_
/**
  ******************************************************************************
  * @file    maxpool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "max_pooling1d_133.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#endif

#define INPUT_CHANNELS  8
#define INPUT_SAMPLES   1991
#define POOL_SIZE       4
#define POOL_STRIDE     4
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

// For fixed point quantization
#define INPUT_SCALE_FACTOR 0
#define OUTPUT_SCALE_FACTOR 0
#define OUTPUT_ROUND_MODE ROUND_MODE_NONE
#define NUMBER_T float
#define LONG_NUMBER_T float


static inline void max_pooling1d_133(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  NUMBER_T output[POOL_LENGTH][INPUT_CHANNELS]) {	// OUT

  unsigned short pos_x, k; 	// loop indexes for output volume
  unsigned int x;
  static LONG_NUMBER_T max[INPUT_CHANNELS];

  for (pos_x = 0; pos_x < POOL_LENGTH; pos_x++) {
    for (k = 0; k < INPUT_CHANNELS; k++) {
#ifdef ACTIVATION_LINEAR
      max[k] = input[pos_x*POOL_STRIDE][k];
      x = 1;
#elif defined(ACTIVATION_RELU)
      max[k] = 0;
      x = 0;
#else
#error "Unsupported activation function"
#endif
    }

    for (; x < POOL_SIZE; x++) {
      for (k = 0; k < INPUT_CHANNELS; k++) {
        if (max[k] < input[(pos_x * POOL_STRIDE) + x][k])
          max[k] = input[(pos_x * POOL_STRIDE) + x][k];
      }
    }

    for (k = 0; k < INPUT_CHANNELS; k++) {
      output[pos_x][k] = scale_and_clamp_to(NUMBER_T, max[k], INPUT_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
    }
  }
}

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH
#undef ACTIVATION_LINEAR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    conv1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _CONV1D_109_H_
#define _CONV1D_109_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      8
#define INPUT_SAMPLES       497
#define CONV_FILTERS        16
#define CONV_KERNEL_SIZE    2
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

typedef float conv1d_109_output_type[CONV_OUTSAMPLES][CONV_FILTERS];

#if 0
void conv1d_109(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const number_t kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS],  // IN

  const number_t bias[CONV_FILTERS],						                          // IN

  number_t output[CONV_OUTSAMPLES][CONV_FILTERS]);                       // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES

#endif//_CONV1D_109_H_
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Sébastien Bilavarn, LEAT, CNRS, Université Côte d'Azur, France
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "conv1d_109.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_CHANNELS      8
#define INPUT_SAMPLES       497
#define CONV_FILTERS        16
#define CONV_KERNEL_SIZE    2
#define CONV_STRIDE         1
#define CONV_GROUPS         1
#define CHANNELS_PER_GROUP  (INPUT_CHANNELS / CONV_GROUPS)
#define FILTERS_PER_GROUP   (CONV_FILTERS / CONV_GROUPS)

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 0
#define BIASES_SCALE_FACTOR 0
#define TMP_SCALE_FACTOR 0
#define INPUT_SCALE_FACTOR 0
#define OUTPUT_SCALE_FACTOR 0
#define OUTPUT_ROUND_MODE ROUND_MODE_NONE
#define NUMBER_T float
#define LONG_NUMBER_T float


static inline void conv1d_109(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const NUMBER_T kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS],  // IN

  const NUMBER_T bias[CONV_FILTERS],						                          // IN

  NUMBER_T output[CONV_OUTSAMPLES][CONV_FILTERS]) {                       // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  int input_x;
  LONG_NUMBER_T output_acc;

  for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
    for (k = 0; k < CONV_FILTERS; k++) { 
      output_acc = 0;

      for (x = 0; x < CONV_KERNEL_SIZE; x++) {
        input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;

        if (input_x >= 0 && input_x < INPUT_SAMPLES) { // ZeroPadding1D
          for (z = 0; z < INPUT_CHANNELS / CONV_GROUPS; z++) {
            output_acc += (LONG_NUMBER_T)input[input_x][z + (k / FILTERS_PER_GROUP) * CHANNELS_PER_GROUP] * (LONG_NUMBER_T)kernel[k][x][z];
          }
        }
      }

    // Scale for possible additional precision of bias
    output_acc = scale(NUMBER_T, output_acc, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);

    // Scale bias to match accumulator
    output_acc += scale(NUMBER_T, (LONG_NUMBER_T)bias[k], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);

      
#ifdef ACTIVATION_LINEAR
      output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
      // Activation function: ReLU
      if (output_acc < 0) {
        output[pos_x][k] = 0;
      } else {
#if defined(ACTIVATION_RELU6)
        if (output_acc > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
          output_acc = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
        }
#endif
        output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      }
#else
#error "Unsupported activation function"
#endif
    }
  }

#else

#if BIASES_SCALE_FACTOR > WEIGHTS_SCALE_FACTOR
#error "CMSIS-NN does not support BIASES_SCALE_FACTOR larger than WEIGHTS_SCALE_FACTOR"
#endif

#error "Data type unsupported by CMSIS-NN"

#endif
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef CONV_GROUPS
#undef CHANNELS_PER_GROUP
#undef FILTERS_PER_GROUP
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_RELU
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef TMP_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    weights/conv1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_CHANNELS    8
#define CONV_FILTERS      16
#define CONV_KERNEL_SIZE  2
#define CONV_GROUPS       1


const float  conv1d_109_bias[CONV_FILTERS] = {-0x1.5194d80000000p-1, -0x1.57f51e0000000p-5, 0x1.a77f340000000p-6, -0x1.5382820000000p-4, -0x1.79d01e0000000p-5, 0x1.fc315a0000000p-6, -0x1.2595320000000p-3, -0x1.c9e0320000000p-6, -0x1.48e5f60000000p-5, -0x1.0bfa720000000p-4, 0x1.add05e0000000p-3, -0x1.0107320000000p-3, -0x1.1da7320000000p-3, -0x1.24390c0000000p-4, -0x1.0fd4be0000000p-5, -0x1.1451fa0000000p-4}
;

const float  conv1d_109_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{{-0x1.82b24c0000000p-3, 0x1.d19dd60000000p-4, 0x1.4544ca0000000p-1, -0x1.8e4f160000000p-6, 0x1.1105d00000000p-3, 0x1.60834c0000000p-1, 0x1.2607c40000000p-1, 0x1.c9c4020000000p-6}
, {-0x1.d736480000000p-3, 0x1.5d89200000000p-3, -0x1.3a519e0000000p+0, 0x1.0940ec0000000p-6, 0x1.347c000000000p-2, 0x1.2f08420000000p-1, 0x1.22b6440000000p-2, 0x1.e52f620000000p-2}
}
, {{-0x1.1109a40000000p-3, 0x1.08135c0000000p-2, -0x1.a14e680000000p-2, 0x1.f700580000000p-3, -0x1.2694860000000p-3, 0x1.0c9e500000000p-1, -0x1.baa8740000000p-6, -0x1.42cf240000000p-3}
, {0x1.5fc7180000000p-3, -0x1.2279da0000000p-3, -0x1.dc9eaa0000000p-2, 0x1.9dfc1c0000000p-2, -0x1.55fbdc0000000p-2, 0x1.6646a20000000p-1, 0x1.4bada80000000p-1, 0x1.0ed4600000000p-1}
}
, {{0x1.63b59e0000000p-7, 0x1.33a9ca0000000p-3, 0x1.a8131e0000000p-1, -0x1.34931a0000000p-1, 0x1.44539a0000000p-2, -0x1.1a45c60000000p+0, -0x1.9626220000000p+0, -0x1.214abc0000000p+0}
, {-0x1.73ea480000000p-6, -0x1.2c9a060000000p-4, 0x1.5177d00000000p-1, -0x1.1b3af60000000p+0, 0x1.f18f5c0000000p-5, -0x1.bebf3a0000000p+0, -0x1.1572820000000p+1, -0x1.21789c0000000p+1}
}
, {{0x1.80ab9e0000000p-5, -0x1.328ebe0000000p-3, -0x1.6adfe80000000p-5, 0x1.4d48280000000p-3, -0x1.93d7b00000000p-2, 0x1.1a9d7c0000000p-2, -0x1.3049c00000000p-5, -0x1.187afc0000000p-2}
, {0x1.3a13c80000000p-3, -0x1.7c74e80000000p-4, -0x1.5fe29e0000000p-3, -0x1.06cc520000000p-1, -0x1.408ca80000000p-3, -0x1.48cc460000000p-3, -0x1.b957a20000000p-2, 0x1.dc88d20000000p-5}
}
, {{0x1.10de920000000p-4, 0x1.5590200000000p-2, -0x1.7584a60000000p-2, 0x1.268f840000000p-2, -0x1.7ee0e60000000p-4, 0x1.855fd40000000p-2, 0x1.1856180000000p-1, 0x1.3bcea40000000p-1}
, {-0x1.051a400000000p-2, -0x1.032b840000000p-2, -0x1.41dc8e0000000p-2, -0x1.26ebc20000000p-4, 0x1.3d22c60000000p-3, 0x1.38423a0000000p-1, 0x1.e5232c0000000p-2, 0x1.447eb20000000p-1}
}
, {{-0x1.270c2c0000000p-2, 0x1.0aa0e40000000p-5, 0x1.ed69100000000p-1, 0x1.60c00e0000000p-2, 0x1.005f1e0000000p-2, 0x1.88785e0000000p-1, 0x1.05a1ba0000000p+0, 0x1.ac48a80000000p-1}
, {-0x1.1affee0000000p-2, 0x1.4e95e80000000p-2, 0x1.8da62e0000000p+0, -0x1.a2db060000000p-1, -0x1.6a85040000000p-3, -0x1.09cca20000000p+0, -0x1.055fea0000000p+0, -0x1.57b9240000000p-1}
}
, {{-0x1.0f7c8e0000000p-7, 0x1.248f180000000p-2, 0x1.013f940000000p-1, -0x1.374b080000000p-1, 0x1.0d15840000000p-3, -0x1.bdd37c0000000p-2, -0x1.9ee2760000000p-3, -0x1.2fd8bc0000000p+0}
, {-0x1.95cf020000000p-4, 0x1.c605aa0000000p-6, 0x1.df353c0000000p-2, -0x1.6f0d660000000p-1, 0x1.6b67220000000p-2, -0x1.df4ab60000000p-1, -0x1.3aa3540000000p-1, -0x1.6b67ae0000000p-2}
}
, {{-0x1.8d06200000000p-2, -0x1.c1b4460000000p-3, -0x1.12b9120000000p-3, -0x1.8041be0000000p-2, -0x1.b08e200000000p-3, 0x1.63ef440000000p-4, -0x1.dc15360000000p-4, -0x1.2012440000000p-1}
, {-0x1.385e920000000p-3, 0x1.48cb440000000p-5, -0x1.2a0e440000000p-2, 0x1.34878a0000000p-3, 0x1.e3448e0000000p-3, -0x1.87b2b80000000p-4, 0x1.aa01da0000000p-4, -0x1.b0dcb00000000p-3}
}
, {{0x1.2b05c40000000p-3, 0x1.03319a0000000p-3, 0x1.c447300000000p-4, -0x1.10e12e0000000p-2, -0x1.075c580000000p-3, 0x1.17731c0000000p-3, -0x1.34c3360000000p-2, -0x1.706a800000000p-2}
, {0x1.70015e0000000p-3, -0x1.e665a80000000p-3, -0x1.471f720000000p-2, -0x1.338eaa0000000p-5, -0x1.306b1a0000000p-2, -0x1.2e2d180000000p-2, -0x1.33e10c0000000p-2, -0x1.2ae5760000000p-4}
}
, {{0x1.60ea4c0000000p-2, -0x1.4533660000000p-3, 0x1.c7a61c0000000p-4, -0x1.bab9aa0000000p-8, -0x1.2e7f200000000p-2, 0x1.2415ee0000000p-1, 0x1.90976a0000000p-1, 0x1.18d6820000000p-2}
, {0x1.9a27ce0000000p-4, -0x1.1c87f20000000p-3, 0x1.63b6560000000p-1, 0x1.2550f80000000p-3, 0x1.e904c00000000p-4, 0x1.2fde3c0000000p-3, 0x1.b9569e0000000p-2, -0x1.95a7d60000000p-4}
}
, {{0x1.a237e60000000p-3, 0x1.2ab49c0000000p-2, 0x1.6db7de0000000p+0, -0x1.82de800000000p-1, -0x1.2e784a0000000p-3, -0x1.053cea0000000p+0, -0x1.f3d3740000000p-1, -0x1.1cad920000000p+0}
, {-0x1.12a73c0000000p-2, -0x1.bf76780000000p-5, 0x1.1d6fa20000000p+0, -0x1.5f514e0000000p+0, -0x1.3e01c00000000p-4, -0x1.0be1000000000p+1, -0x1.d957580000000p+0, -0x1.c8125e0000000p+0}
}
, {{-0x1.4890ac0000000p-4, 0x1.4d29780000000p-3, 0x1.95c1ae0000000p-4, 0x1.3ecba00000000p-3, -0x1.52155e0000000p-3, 0x1.e5e2a80000000p-3, -0x1.6809020000000p-5, -0x1.f6349e0000000p-3}
, {0x1.05ffd00000000p-2, -0x1.69cba60000000p-2, -0x1.bd9a400000000p-3, -0x1.5f4bfc0000000p-3, -0x1.1457100000000p-2, 0x1.c123080000000p-5, -0x1.321eae0000000p-5, -0x1.b105800000000p-4}
}
, {{-0x1.9f05020000000p-4, -0x1.0295340000000p-4, 0x1.0596e20000000p-2, -0x1.ffba920000000p-3, -0x1.4a6d9e0000000p-3, -0x1.0150b80000000p-3, -0x1.91448c0000000p-2, 0x1.e8d5280000000p-6}
, {0x1.e7a6b60000000p-3, -0x1.923dce0000000p-4, -0x1.23dcd00000000p-2, 0x1.bd58300000000p-3, -0x1.f4db3c0000000p-3, -0x1.0525100000000p-2, -0x1.2eb0e20000000p-2, -0x1.0eb0620000000p-3}
}
, {{0x1.7a60920000000p-7, 0x1.1c43a00000000p-2, 0x1.2ad4300000000p-3, 0x1.9e8d6e0000000p-3, -0x1.c92f660000000p-4, 0x1.4ef8b40000000p-1, 0x1.275b9c0000000p-3, 0x1.3c1d760000000p-1}
, {-0x1.fe153c0000000p-3, 0x1.d28e3e0000000p-3, -0x1.12fbf60000000p-1, 0x1.05388e0000000p-4, -0x1.5ea9d40000000p-2, 0x1.16cf1a0000000p+0, 0x1.ebba860000000p-1, 0x1.038f620000000p+0}
}
, {{0x1.9828b00000000p-4, -0x1.5037700000000p-3, -0x1.69d0a00000000p-2, -0x1.28ebce0000000p-2, 0x1.7573b40000000p-7, 0x1.a2f8da0000000p-3, -0x1.0fb5780000000p-4, -0x1.268c320000000p-4}
, {0x1.eabbc20000000p-3, -0x1.0db6a00000000p-2, -0x1.5e4a4c0000000p-3, 0x1.5b21580000000p-3, 0x1.d897220000000p-6, -0x1.74b2d20000000p-2, -0x1.b6568c0000000p-5, -0x1.f611f40000000p-4}
}
, {{-0x1.10bb620000000p-2, -0x1.b035440000000p-6, 0x1.b094280000000p-3, 0x1.f6bddc0000000p-2, 0x1.60f6aa0000000p-3, 0x1.af08c20000000p-1, 0x1.131f240000000p-1, 0x1.2276780000000p-1}
, {-0x1.3b8a100000000p-2, 0x1.6ade220000000p-2, -0x1.26c48a0000000p-1, 0x1.2bd6280000000p-1, -0x1.fdc3fe0000000p-3, 0x1.b2b7620000000p-2, 0x1.0f56940000000p-1, 0x1.5ce04e0000000p-1}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_GROUPS
/**
  ******************************************************************************
  * @file    maxpool1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _MAX_POOLING1D_134_H_
#define _MAX_POOLING1D_134_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  16
#define INPUT_SAMPLES   496
#define POOL_SIZE       4
#define POOL_STRIDE     4
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

typedef float max_pooling1d_134_output_type[POOL_LENGTH][INPUT_CHANNELS];

#if 0
void max_pooling1d_134(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  number_t output[POOL_LENGTH][INPUT_CHANNELS]); 	// OUT
#endif

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH

#endif//_MAX_POOLING1D_134_H_
/**
  ******************************************************************************
  * @file    maxpool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "max_pooling1d_134.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#endif

#define INPUT_CHANNELS  16
#define INPUT_SAMPLES   496
#define POOL_SIZE       4
#define POOL_STRIDE     4
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

// For fixed point quantization
#define INPUT_SCALE_FACTOR 0
#define OUTPUT_SCALE_FACTOR 0
#define OUTPUT_ROUND_MODE ROUND_MODE_NONE
#define NUMBER_T float
#define LONG_NUMBER_T float


static inline void max_pooling1d_134(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  NUMBER_T output[POOL_LENGTH][INPUT_CHANNELS]) {	// OUT

  unsigned short pos_x, k; 	// loop indexes for output volume
  unsigned int x;
  static LONG_NUMBER_T max[INPUT_CHANNELS];

  for (pos_x = 0; pos_x < POOL_LENGTH; pos_x++) {
    for (k = 0; k < INPUT_CHANNELS; k++) {
#ifdef ACTIVATION_LINEAR
      max[k] = input[pos_x*POOL_STRIDE][k];
      x = 1;
#elif defined(ACTIVATION_RELU)
      max[k] = 0;
      x = 0;
#else
#error "Unsupported activation function"
#endif
    }

    for (; x < POOL_SIZE; x++) {
      for (k = 0; k < INPUT_CHANNELS; k++) {
        if (max[k] < input[(pos_x * POOL_STRIDE) + x][k])
          max[k] = input[(pos_x * POOL_STRIDE) + x][k];
      }
    }

    for (k = 0; k < INPUT_CHANNELS; k++) {
      output[pos_x][k] = scale_and_clamp_to(NUMBER_T, max[k], INPUT_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
    }
  }
}

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH
#undef ACTIVATION_LINEAR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    conv1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _CONV1D_110_H_
#define _CONV1D_110_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      16
#define INPUT_SAMPLES       124
#define CONV_FILTERS        32
#define CONV_KERNEL_SIZE    2
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

typedef float conv1d_110_output_type[CONV_OUTSAMPLES][CONV_FILTERS];

#if 0
void conv1d_110(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const number_t kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS],  // IN

  const number_t bias[CONV_FILTERS],						                          // IN

  number_t output[CONV_OUTSAMPLES][CONV_FILTERS]);                       // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES

#endif//_CONV1D_110_H_
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Sébastien Bilavarn, LEAT, CNRS, Université Côte d'Azur, France
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "conv1d_110.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_CHANNELS      16
#define INPUT_SAMPLES       124
#define CONV_FILTERS        32
#define CONV_KERNEL_SIZE    2
#define CONV_STRIDE         1
#define CONV_GROUPS         1
#define CHANNELS_PER_GROUP  (INPUT_CHANNELS / CONV_GROUPS)
#define FILTERS_PER_GROUP   (CONV_FILTERS / CONV_GROUPS)

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 0
#define BIASES_SCALE_FACTOR 0
#define TMP_SCALE_FACTOR 0
#define INPUT_SCALE_FACTOR 0
#define OUTPUT_SCALE_FACTOR 0
#define OUTPUT_ROUND_MODE ROUND_MODE_NONE
#define NUMBER_T float
#define LONG_NUMBER_T float


static inline void conv1d_110(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const NUMBER_T kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS],  // IN

  const NUMBER_T bias[CONV_FILTERS],						                          // IN

  NUMBER_T output[CONV_OUTSAMPLES][CONV_FILTERS]) {                       // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  int input_x;
  LONG_NUMBER_T output_acc;

  for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
    for (k = 0; k < CONV_FILTERS; k++) { 
      output_acc = 0;

      for (x = 0; x < CONV_KERNEL_SIZE; x++) {
        input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;

        if (input_x >= 0 && input_x < INPUT_SAMPLES) { // ZeroPadding1D
          for (z = 0; z < INPUT_CHANNELS / CONV_GROUPS; z++) {
            output_acc += (LONG_NUMBER_T)input[input_x][z + (k / FILTERS_PER_GROUP) * CHANNELS_PER_GROUP] * (LONG_NUMBER_T)kernel[k][x][z];
          }
        }
      }

    // Scale for possible additional precision of bias
    output_acc = scale(NUMBER_T, output_acc, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);

    // Scale bias to match accumulator
    output_acc += scale(NUMBER_T, (LONG_NUMBER_T)bias[k], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);

      
#ifdef ACTIVATION_LINEAR
      output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
      // Activation function: ReLU
      if (output_acc < 0) {
        output[pos_x][k] = 0;
      } else {
#if defined(ACTIVATION_RELU6)
        if (output_acc > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
          output_acc = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
        }
#endif
        output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      }
#else
#error "Unsupported activation function"
#endif
    }
  }

#else

#if BIASES_SCALE_FACTOR > WEIGHTS_SCALE_FACTOR
#error "CMSIS-NN does not support BIASES_SCALE_FACTOR larger than WEIGHTS_SCALE_FACTOR"
#endif

#error "Data type unsupported by CMSIS-NN"

#endif
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef CONV_GROUPS
#undef CHANNELS_PER_GROUP
#undef FILTERS_PER_GROUP
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_RELU
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef TMP_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    weights/conv1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_CHANNELS    16
#define CONV_FILTERS      32
#define CONV_KERNEL_SIZE  2
#define CONV_GROUPS       1


const float  conv1d_110_bias[CONV_FILTERS] = {-0x1.10cdf20000000p-6, 0x1.ef6d540000000p-8, 0x1.345b0a0000000p-4, -0x1.77d8020000000p-7, -0x1.74c39c0000000p-3, -0x1.061d560000000p-4, -0x1.35c32a0000000p-5, -0x1.ebcef00000000p-5, -0x1.a5d0440000000p-3, -0x1.3879ee0000000p-3, -0x1.135c320000000p-2, -0x1.1313760000000p-5, -0x1.d84ae80000000p-3, -0x1.73d4a60000000p-5, 0x1.ae00760000000p-6, -0x1.ed3f740000000p-5, 0x1.f6397e0000000p-4, -0x1.35e5060000000p-6, -0x1.c33fb40000000p-3, -0x1.2689340000000p-3, 0x1.31baa00000000p-5, 0x1.57aaf80000000p-2, -0x1.530e9e0000000p-3, -0x1.160b0c0000000p-3, 0x1.0928080000000p-7, -0x1.0fa9dc0000000p-3, 0x1.0276740000000p-3, -0x1.bbabdc0000000p-5, 0x1.02f21a0000000p-3, 0x1.1343380000000p-3, 0x1.4663480000000p-2, -0x1.c0592e0000000p-4}
;

const float  conv1d_110_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{{-0x1.06b7aa0000000p-3, 0x1.ac498a0000000p-9, -0x1.3927a60000000p-3, 0x1.3771540000000p-4, -0x1.e98c860000000p-5, -0x1.1b50820000000p-3, 0x1.2386aa0000000p-4, -0x1.4ab9040000000p-3, 0x1.7086040000000p-3, -0x1.32bfb40000000p-7, 0x1.1d9a900000000p-3, 0x1.4485fc0000000p-5, -0x1.2b60180000000p-9, -0x1.3157060000000p-3, -0x1.5da4d00000000p-3, -0x1.6e2e5a0000000p-3}
, {-0x1.67e4dc0000000p-4, 0x1.bab04e0000000p-9, 0x1.02a8c60000000p-4, 0x1.243abc0000000p-3, -0x1.255f9c0000000p-2, -0x1.78ed660000000p-5, -0x1.91eca20000000p-3, 0x1.ddd7d80000000p-4, 0x1.2c794e0000000p-4, -0x1.2957340000000p-2, -0x1.fc5b880000000p-3, 0x1.4064b20000000p-4, -0x1.b47bc00000000p-8, -0x1.b290480000000p-3, 0x1.531c540000000p-3, 0x1.c1b50e0000000p-3}
}
, {{-0x1.de09040000000p-2, 0x1.84446c0000000p-3, -0x1.f20c040000000p-3, 0x1.3d22d40000000p-4, 0x1.1d5cda0000000p-2, 0x1.da406c0000000p-1, -0x1.efcccc0000000p-2, 0x1.0e35440000000p-3, -0x1.6b0ad80000000p-3, 0x1.6184be0000000p-2, -0x1.a399fe0000000p-4, -0x1.49cb4a0000000p-3, -0x1.49f5c20000000p-4, 0x1.5715aa0000000p-2, 0x1.aee18e0000000p-4, 0x1.e6c01a0000000p-2}
, {0x1.6f0fba0000000p-6, -0x1.3c4a520000000p-1, -0x1.6b5ff40000000p-6, 0x1.6d73680000000p-3, -0x1.0112ac0000000p-1, -0x1.7a83720000000p-1, 0x1.bcf4800000000p-1, -0x1.7e15cc0000000p-3, 0x1.eda9040000000p-5, -0x1.6254ea0000000p-2, -0x1.9647a80000000p+0, 0x1.23771e0000000p-3, 0x1.7b1f700000000p-5, -0x1.6800880000000p-1, 0x1.0e2b720000000p-7, -0x1.d175520000000p-2}
}
, {{0x1.c86f640000000p-3, -0x1.c511820000000p-4, 0x1.b11d140000000p-1, 0x1.ed46d80000000p-4, 0x1.06800a0000000p-2, -0x1.1575f60000000p-1, -0x1.59d62e0000000p-5, -0x1.28893e0000000p-4, -0x1.dd68dc0000000p-3, 0x1.06c09e0000000p-2, -0x1.1771800000000p-2, 0x1.411b700000000p-4, -0x1.42f3540000000p-3, 0x1.7ce8740000000p-3, -0x1.21ec340000000p-3, 0x1.a928b60000000p-3}
, {-0x1.4f7b600000000p-2, 0x1.769bf40000000p-2, 0x1.c7dfb40000000p-2, -0x1.d3515c0000000p-8, 0x1.74c4520000000p-5, -0x1.36b63c0000000p-1, 0x1.9c548e0000000p-2, -0x1.33c6020000000p-3, -0x1.f021220000000p-4, -0x1.47e3c00000000p-4, -0x1.60aef20000000p-1, 0x1.5c0ca40000000p-4, -0x1.a43ebc0000000p-3, 0x1.dcf6200000000p-4, 0x1.76153c0000000p-4, 0x1.aaab1a0000000p-2}
}
, {{-0x1.b75ff20000000p-1, -0x1.766f980000000p-2, -0x1.b78a2a0000000p-6, 0x1.b21a380000000p-3, -0x1.fcbba40000000p-2, -0x1.e692400000000p-1, -0x1.906a9e0000000p-1, -0x1.f537540000000p-3, -0x1.1b71700000000p-7, -0x1.2532d60000000p-4, -0x1.38686e0000000p+0, 0x1.40d01e0000000p-2, 0x1.d50c5e0000000p-3, -0x1.2050b20000000p-1, 0x1.cc62f60000000p-4, -0x1.3550ca0000000p-3}
, {-0x1.91ff920000000p-1, 0x1.a455780000000p-2, 0x1.0aa4f20000000p-1, -0x1.818fe20000000p-3, 0x1.77d46c0000000p-3, 0x1.33957c0000000p-1, -0x1.1b29700000000p-1, -0x1.13db2c0000000p-3, 0x1.218b5a0000000p-4, 0x1.4a9b660000000p-2, -0x1.1eeb9c0000000p+0, 0x1.d7a7080000000p-3, -0x1.f712580000000p-4, 0x1.6ea7f60000000p-2, -0x1.6cbf460000000p-4, 0x1.8ca3da0000000p-2}
}
, {{0x1.1b7f400000000p-3, -0x1.92686c0000000p-2, 0x1.32c72e0000000p-3, -0x1.583df40000000p-4, -0x1.8c32b60000000p-2, 0x1.555a260000000p-3, 0x1.d632ae0000000p-4, -0x1.3f0af60000000p-4, -0x1.ffb7300000000p-7, 0x1.db86340000000p-6, -0x1.289b560000000p-2, 0x1.5089b40000000p-4, -0x1.ebdf7c0000000p-3, -0x1.afc9720000000p-3, 0x1.ecfa0e0000000p-3, -0x1.1ffff20000000p-3}
, {0x1.0bb4c80000000p-3, -0x1.a6b79c0000000p-3, -0x1.9e967e0000000p-3, -0x1.d2c73e0000000p-3, -0x1.1aac7c0000000p-3, -0x1.d9282c0000000p-3, 0x1.cea5fc0000000p-3, -0x1.59be480000000p-3, 0x1.468f600000000p-3, -0x1.a8d1ea0000000p-2, 0x1.23395c0000000p-4, -0x1.b4a7c20000000p-5, 0x1.9187020000000p-3, -0x1.2f593a0000000p-2, -0x1.9b9e200000000p-5, -0x1.b8ac6c0000000p-3}
}
, {{0x1.0910ce0000000p-3, 0x1.8fad7e0000000p-3, -0x1.8dbd9e0000000p-4, -0x1.b975bc0000000p-5, -0x1.3067660000000p-2, -0x1.e7af0c0000000p-3, 0x1.92167c0000000p-5, -0x1.ef9b220000000p-7, -0x1.dcb00c0000000p-3, 0x1.5d446c0000000p-4, 0x1.2e6f540000000p-3, -0x1.9b73e60000000p-3, -0x1.76c2720000000p-3, -0x1.4fcec60000000p-3, 0x1.3d97c60000000p-4, 0x1.34276e0000000p-5}
, {-0x1.13ab320000000p-3, -0x1.351d3a0000000p-2, 0x1.2a18b60000000p-4, -0x1.9574a20000000p-3, 0x1.cd7cc00000000p-4, -0x1.5704280000000p-4, 0x1.6dbf8a0000000p-3, -0x1.5a10200000000p-3, 0x1.778fc00000000p-3, 0x1.deb6800000000p-10, -0x1.779e1c0000000p-2, -0x1.a6c5d20000000p-3, 0x1.4058340000000p-4, -0x1.15ead00000000p-4, 0x1.9dce7a0000000p-3, -0x1.2651ba0000000p-2}
}
, {{0x1.840ac80000000p-4, -0x1.2962920000000p-2, -0x1.3d442a0000000p-6, -0x1.cbaf480000000p-3, -0x1.b41b1a0000000p-3, -0x1.b8fc0a0000000p-3, -0x1.73c4e60000000p-3, 0x1.4810a80000000p-3, -0x1.a5bdb60000000p-4, 0x1.6a63f60000000p-4, 0x1.3c90b80000000p-6, -0x1.c77cda0000000p-3, 0x1.e4add00000000p-3, -0x1.165aa40000000p-3, -0x1.31f7c20000000p-3, 0x1.f2d92a0000000p-4}
, {-0x1.8f636e0000000p-4, -0x1.23308c0000000p-2, 0x1.a634ac0000000p-4, 0x1.320b6e0000000p-4, 0x1.6e54600000000p-4, -0x1.bc17f40000000p-3, -0x1.b2f2700000000p-3, 0x1.540c140000000p-6, -0x1.1116be0000000p-3, -0x1.23d5580000000p-5, 0x1.28b2000000000p-3, 0x1.e07f660000000p-3, -0x1.c19d8a0000000p-4, 0x1.9636780000000p-6, -0x1.965c920000000p-3, -0x1.f0fcf00000000p-4}
}
, {{0x1.2fa4dc0000000p-4, -0x1.91c3ae0000000p-3, -0x1.987f6a0000000p-3, -0x1.46c3c80000000p-6, -0x1.7a45d00000000p-3, -0x1.363ea20000000p-4, -0x1.a3b7020000000p-3, -0x1.123a0e0000000p-4, -0x1.136ea00000000p-3, 0x1.72be720000000p-4, -0x1.c848a20000000p-5, 0x1.4d320c0000000p-3, 0x1.4a8c900000000p-3, -0x1.65d4220000000p-5, 0x1.b5fa4e0000000p-3, -0x1.18c1dc0000000p-3}
, {0x1.3990940000000p-4, 0x1.5e663a0000000p-11, 0x1.99ad820000000p-3, -0x1.13d6040000000p-5, -0x1.c4946c0000000p-5, -0x1.a414920000000p-3, 0x1.7d11340000000p-5, -0x1.e746fc0000000p-3, -0x1.1bb2240000000p-3, 0x1.285a540000000p-4, 0x1.86843c0000000p-6, -0x1.b209000000000p-4, 0x1.691d700000000p-3, 0x1.629c3a0000000p-3, -0x1.8261c00000000p-3, -0x1.0892260000000p-2}
}
, {{0x1.95cf600000000p-2, -0x1.f101d60000000p-3, -0x1.1ce2c60000000p-8, -0x1.ad45fe0000000p-4, -0x1.0e20b60000000p-4, -0x1.4728420000000p-2, -0x1.3433ca0000000p-5, -0x1.d8da560000000p-4, 0x1.8875e20000000p-6, -0x1.29e47a0000000p-3, -0x1.3eb3c60000000p-2, -0x1.a90b1c0000000p-5, -0x1.cf7cda0000000p-5, -0x1.69fb180000000p-3, 0x1.1628b20000000p-3, -0x1.5493e40000000p-4}
, {0x1.42e5e80000000p-2, -0x1.22a2d20000000p-2, -0x1.0d7d180000000p-4, -0x1.bd23d20000000p-4, 0x1.6320ac0000000p-3, 0x1.861bce0000000p-4, 0x1.8932640000000p-6, -0x1.ade41e0000000p-4, -0x1.5f00760000000p-3, 0x1.f0d9460000000p-6, -0x1.321c140000000p-8, -0x1.8f57f60000000p-3, -0x1.e526a80000000p-3, -0x1.f233020000000p-3, -0x1.d454420000000p-7, 0x1.ecf23e0000000p-4}
}
, {{-0x1.ed46540000000p-5, 0x1.d296b80000000p-4, 0x1.0776160000000p-2, -0x1.8f43e00000000p-3, 0x1.56c7120000000p-7, 0x1.410af60000000p-3, 0x1.3bf5c00000000p-3, -0x1.fb441e0000000p-2, 0x1.ef72060000000p-7, -0x1.b0d21a0000000p-3, -0x1.8421720000000p-3, 0x1.1cf3500000000p-2, 0x1.e7de9c0000000p-5, -0x1.8453b60000000p-4, -0x1.ee3ade0000000p-4, 0x1.4137260000000p-4}
, {0x1.7ce4c20000000p-3, -0x1.0b61bc0000000p-6, -0x1.898c3e0000000p-4, -0x1.092cec0000000p-5, -0x1.f1c6cc0000000p-3, 0x1.01eb480000000p-6, 0x1.11b8f80000000p-3, -0x1.7ee0d40000000p-3, 0x1.d32e280000000p-5, 0x1.9afa760000000p-3, -0x1.a915ca0000000p-2, 0x1.4f94660000000p-3, -0x1.0e35b40000000p-3, -0x1.4b8db80000000p-3, -0x1.aabbb20000000p-4, -0x1.4ff97e0000000p-8}
}
, {{-0x1.ec90e40000000p-2, -0x1.08b65e0000000p-5, -0x1.1ef6dc0000000p-1, -0x1.02689e0000000p-2, -0x1.1aa0ca0000000p-3, -0x1.0bb1320000000p-1, 0x1.5742b00000000p-2, -0x1.e557740000000p-3, 0x1.a3eab40000000p-3, -0x1.3229bc0000000p-2, -0x1.0f79280000000p-1, 0x1.6470920000000p-5, 0x1.18aa280000000p-5, -0x1.8332420000000p-4, -0x1.812cac0000000p-3, 0x1.4197040000000p-4}
, {-0x1.37284e0000000p-3, 0x1.69e0340000000p-4, 0x1.a620640000000p-2, 0x1.97e9060000000p-10, 0x1.7b32140000000p-2, -0x1.33daf00000000p+0, 0x1.6288b00000000p-3, -0x1.2de19a0000000p-2, -0x1.9feb000000000p-8, 0x1.05848e0000000p-2, -0x1.cfbf580000000p-2, 0x1.970fb20000000p-3, 0x1.2c84fe0000000p-2, 0x1.961dfa0000000p-3, 0x1.42741a0000000p-5, 0x1.168bea0000000p-1}
}
, {{0x1.f4f8360000000p-5, -0x1.09eac60000000p-2, 0x1.2cbda20000000p-3, -0x1.15c3ce0000000p-3, -0x1.fccff00000000p-4, 0x1.f8f94c0000000p-7, 0x1.9a84240000000p-3, -0x1.0c3bf20000000p-2, 0x1.6024640000000p-3, -0x1.051ac00000000p-3, -0x1.2112d40000000p-3, 0x1.2afabc0000000p-5, -0x1.f0dfc40000000p-3, -0x1.6ad5d20000000p-3, 0x1.c258700000000p-5, -0x1.38e5680000000p-2}
, {0x1.aac8340000000p-3, 0x1.0220180000000p-4, 0x1.9b47c60000000p-3, 0x1.46d40a0000000p-3, 0x1.9842cc0000000p-5, 0x1.d5005e0000000p-5, 0x1.9ffba60000000p-5, -0x1.0ac87c0000000p-2, -0x1.5120480000000p-3, -0x1.bc44580000000p-2, -0x1.f20bdc0000000p-3, -0x1.d1c8400000000p-4, 0x1.d3fc660000000p-4, 0x1.2de37a0000000p-6, -0x1.09bf8a0000000p-3, -0x1.08e69a0000000p-5}
}
, {{-0x1.13c6800000000p-3, -0x1.5694760000000p-2, -0x1.4193fc0000000p-4, 0x1.ccdfe80000000p-5, -0x1.51ad360000000p-2, 0x1.b6227c0000000p-5, -0x1.c8bee80000000p-4, 0x1.b1365a0000000p-3, 0x1.c97e020000000p-3, -0x1.132fd00000000p-2, -0x1.86d7d40000000p-3, -0x1.281c3c0000000p-5, -0x1.9de00e0000000p-4, 0x1.a1a6fc0000000p-5, -0x1.86ca200000000p-4, -0x1.ca495a0000000p-4}
, {0x1.7015380000000p-5, -0x1.8c58fe0000000p-3, -0x1.4bff2a0000000p-7, 0x1.e23b760000000p-3, -0x1.ce78000000000p-4, -0x1.92db960000000p-3, 0x1.b71d8c0000000p-4, -0x1.fb607c0000000p-4, 0x1.2421460000000p-4, -0x1.12dac40000000p-1, -0x1.47eda40000000p-3, -0x1.752e7c0000000p-3, 0x1.5050960000000p-6, -0x1.149a060000000p-2, 0x1.9dca860000000p-3, -0x1.e472340000000p-4}
}
, {{-0x1.95a2840000000p-6, 0x1.0d60be0000000p-4, -0x1.249a000000000p-3, 0x1.557e420000000p-3, -0x1.99bede0000000p-3, -0x1.cdb1ee0000000p-3, -0x1.d765980000000p-3, -0x1.2886080000000p-2, -0x1.fedd860000000p-3, 0x1.d291240000000p-6, -0x1.f3dcce0000000p-4, -0x1.7849b40000000p-5, 0x1.50df7a0000000p-3, -0x1.a829780000000p-3, -0x1.a9668e0000000p-5, -0x1.7f48a20000000p-4}
, {-0x1.4b5dca0000000p-3, -0x1.5acc900000000p-7, -0x1.b78b180000000p-6, -0x1.fb26180000000p-3, 0x1.3e40440000000p-3, -0x1.ae615c0000000p-9, -0x1.b1d6700000000p-5, -0x1.a8236a0000000p-7, 0x1.04b7b80000000p-7, 0x1.ad89c80000000p-4, -0x1.c5f6320000000p-6, 0x1.b7cbce0000000p-5, 0x1.fcc62e0000000p-6, -0x1.37b6060000000p-2, -0x1.78994c0000000p-5, -0x1.f367a20000000p-5}
}
, {{-0x1.5ee0620000000p-3, -0x1.5423a20000000p-3, -0x1.6ca5120000000p-1, 0x1.1f1ef20000000p-8, -0x1.ba3cb20000000p-4, -0x1.0aef260000000p-2, -0x1.11fac60000000p-1, 0x1.f266c20000000p-4, -0x1.1911240000000p-3, -0x1.3357d00000000p-2, -0x1.0fc1d80000000p+0, -0x1.2624820000000p-3, -0x1.7a81d00000000p-5, 0x1.41370e0000000p-4, -0x1.a1bce60000000p-3, 0x1.4242de0000000p-4}
, {-0x1.4ba2de0000000p+0, -0x1.2346b60000000p-2, -0x1.6a6da60000000p-5, 0x1.15336a0000000p-4, -0x1.762a320000000p-3, 0x1.b98e120000000p-1, 0x1.10bf4a0000000p-1, -0x1.28aaea0000000p-2, -0x1.4536ce0000000p-6, 0x1.b6d2080000000p-3, 0x1.b4aef40000000p-1, -0x1.df35280000000p-3, 0x1.8a08920000000p-5, 0x1.1537660000000p-4, -0x1.db8f4a0000000p-4, -0x1.e11af20000000p-3}
}
, {{0x1.b795c00000000p-6, -0x1.34ca6c0000000p-2, -0x1.a5c3e20000000p-3, 0x1.338c2e0000000p-4, -0x1.5a4b320000000p-2, 0x1.164d760000000p-7, 0x1.a453b60000000p-3, -0x1.e9b1460000000p-5, 0x1.9288920000000p-3, -0x1.2069920000000p-2, 0x1.061bee0000000p-8, 0x1.b528900000000p-6, 0x1.0cc7260000000p-5, -0x1.13d6480000000p-4, -0x1.d03f440000000p-4, 0x1.3e7a140000000p-6}
, {-0x1.1b82200000000p-4, 0x1.6eb50a0000000p-5, 0x1.bbfd120000000p-4, 0x1.d29c240000000p-4, -0x1.02332e0000000p-2, -0x1.87982c0000000p-3, 0x1.ec33400000000p-5, -0x1.bed1120000000p-5, -0x1.5d41c20000000p-3, 0x1.c77fc80000000p-6, -0x1.10912c0000000p-4, 0x1.dae06e0000000p-5, 0x1.0c3a460000000p-3, -0x1.4409aa0000000p-3, 0x1.be3bb80000000p-5, -0x1.5b69ca0000000p-6}
}
, {{-0x1.522ff40000000p+0, 0x1.3380fc0000000p-2, 0x1.fd25860000000p-7, 0x1.95ba520000000p-3, 0x1.a72f9e0000000p-4, 0x1.f2c0b40000000p-3, 0x1.6903a20000000p-2, 0x1.43ee9c0000000p-3, 0x1.d4c7e20000000p-3, 0x1.45d5280000000p-4, -0x1.4ff5180000000p-6, 0x1.2a2cd00000000p-4, 0x1.6cd1a20000000p-2, 0x1.7b29ec0000000p-4, -0x1.4cb77e0000000p-5, 0x1.cf0b800000000p-2}
, {-0x1.bd0eda0000000p-1, -0x1.ae2a980000000p-3, -0x1.ebbba20000000p-3, 0x1.3114100000000p-3, -0x1.223a3e0000000p-1, -0x1.0c96c60000000p+0, -0x1.7b2afa0000000p-3, 0x1.32fb7c0000000p-2, 0x1.7018180000000p-3, -0x1.2cc62a0000000p-1, -0x1.4c3b400000000p-1, 0x1.7e1cc20000000p-5, -0x1.6353560000000p-3, -0x1.ecfb120000000p-2, -0x1.3968c20000000p-3, -0x1.88c3980000000p-1}
}
, {{-0x1.67fa460000000p-1, 0x1.0d2f620000000p-2, -0x1.54ffde0000000p+0, -0x1.f49be40000000p-5, 0x1.7a50380000000p-3, -0x1.cc63800000000p-3, -0x1.52e7fc0000000p-1, -0x1.2184a40000000p-2, 0x1.485dfc0000000p-3, 0x1.0791d20000000p-2, -0x1.773f540000000p-1, 0x1.1a33280000000p-3, 0x1.3da53c0000000p-4, 0x1.da780a0000000p-5, 0x1.82ee4e0000000p-7, 0x1.2922b00000000p-2}
, {-0x1.68c6500000000p-4, -0x1.0a54300000000p+0, -0x1.72c1c80000000p-2, 0x1.45ded80000000p-7, -0x1.e5789a0000000p-1, 0x1.7112da0000000p-1, 0x1.19d76c0000000p-7, 0x1.b3a27e0000000p-4, -0x1.04a6440000000p-6, -0x1.ea83c00000000p-3, 0x1.633b280000000p-1, -0x1.4bde640000000p-4, -0x1.720b360000000p-4, -0x1.073ab20000000p+0, 0x1.7c48cc0000000p-7, -0x1.ad620e0000000p-1}
}
, {{0x1.0a052e0000000p-2, 0x1.570f240000000p-5, -0x1.21f0800000000p-2, -0x1.1927560000000p-3, -0x1.8af33a0000000p-6, 0x1.f849320000000p-5, 0x1.28b8420000000p-7, 0x1.5302020000000p-3, -0x1.4b79960000000p-3, -0x1.e3f9500000000p-3, 0x1.412b6c0000000p-6, 0x1.38ce520000000p-2, -0x1.f3c1a20000000p-4, -0x1.a65cc20000000p-3, 0x1.fc79e20000000p-7, 0x1.8fe9e80000000p-4}
, {0x1.46c9a60000000p-2, -0x1.842e960000000p-12, -0x1.9cb27c0000000p-4, 0x1.39c9f60000000p-4, -0x1.7b9eea0000000p-2, -0x1.53cea20000000p-2, -0x1.d956360000000p-3, 0x1.6f1c320000000p-4, -0x1.d13e3a0000000p-3, 0x1.7deb6c0000000p-3, -0x1.6a41940000000p-3, 0x1.9855540000000p-3, 0x1.c724460000000p-8, 0x1.0be0120000000p-4, 0x1.b587100000000p-3, -0x1.cea2d60000000p-3}
}
, {{-0x1.449fae0000000p-3, 0x1.9c23de0000000p-4, 0x1.4694ea0000000p-5, 0x1.aa54420000000p-3, 0x1.e3eb720000000p-8, 0x1.bce2dc0000000p+0, -0x1.a420140000000p-4, -0x1.86882c0000000p-3, -0x1.1d0c7a0000000p-6, 0x1.dd31ca0000000p-3, 0x1.771fcc0000000p-4, -0x1.babea60000000p-4, 0x1.6f8bac0000000p-2, 0x1.08d3be0000000p-2, 0x1.5b58700000000p-5, 0x1.93343a0000000p-3}
, {0x1.c7ca680000000p-2, 0x1.d53e1c0000000p-4, 0x1.059e400000000p-1, -0x1.74b8be0000000p-7, -0x1.d5f2da0000000p-4, 0x1.7143a60000000p-1, 0x1.31d62c0000000p-5, 0x1.fa32140000000p-4, -0x1.37d0ee0000000p-2, -0x1.0a71d20000000p-5, -0x1.9eea860000000p-2, -0x1.e8d1d40000000p-3, 0x1.f32c140000000p-3, -0x1.303f6c0000000p-4, 0x1.f9cc680000000p-5, 0x1.c8076a0000000p-5}
}
, {{-0x1.59243a0000000p-1, -0x1.2f7e940000000p-1, 0x1.62f5ac0000000p-7, 0x1.9cb2ea0000000p-4, -0x1.4792e40000000p-2, 0x1.89bb120000000p-1, 0x1.582eee0000000p-1, -0x1.6aa6220000000p-3, 0x1.6a16440000000p-3, 0x1.caffbc0000000p-5, 0x1.3ce08a0000000p-1, -0x1.e92af60000000p-5, 0x1.cf422e0000000p-4, -0x1.08f2060000000p-1, -0x1.79e2ec0000000p-4, -0x1.8b60de0000000p-2}
, {-0x1.61179a0000000p-4, 0x1.23f3800000000p-3, -0x1.1865ec0000000p+0, -0x1.3da11a0000000p-3, 0x1.299de40000000p-2, -0x1.fc044a0000000p-2, 0x1.a5e0560000000p-4, 0x1.0d9fba0000000p-3, -0x1.4ea2d20000000p-4, 0x1.6cb2c40000000p-4, -0x1.796f680000000p-1, -0x1.346f4a0000000p-3, -0x1.f0948e0000000p-7, 0x1.a634060000000p-3, -0x1.0bdc860000000p-5, 0x1.26eb940000000p-2}
}
, {{-0x1.40059c0000000p-4, -0x1.33e8a40000000p-2, -0x1.9c07d80000000p-1, 0x1.adb9ca0000000p-3, -0x1.3a93740000000p-1, -0x1.40bfbe0000000p+0, -0x1.129eb40000000p-5, 0x1.3186260000000p-5, -0x1.87580c0000000p-3, -0x1.0119fc0000000p-1, -0x1.96a0c80000000p-2, 0x1.d231000000000p-3, 0x1.5b5f240000000p-2, -0x1.0431c00000000p-1, 0x1.813c7e0000000p-3, -0x1.10016c0000000p+0}
, {-0x1.983dd00000000p-1, -0x1.c6bde40000000p-8, -0x1.d68ca40000000p+0, 0x1.1aede00000000p-4, -0x1.ce56120000000p-5, 0x1.aa6b780000000p-4, 0x1.3b2bce0000000p-3, 0x1.7ded080000000p-4, 0x1.416a340000000p-3, 0x1.1dc0260000000p-3, -0x1.0821960000000p-4, 0x1.21e3540000000p-4, 0x1.0c11b00000000p-2, 0x1.2ee4880000000p-3, 0x1.47a79c0000000p-4, 0x1.1f32620000000p-5}
}
, {{0x1.79eb2a0000000p-3, -0x1.94d4d60000000p-5, 0x1.1cd05a0000000p-3, 0x1.8a9be80000000p-4, -0x1.2715840000000p-2, -0x1.8986420000000p-2, 0x1.0c36e20000000p-4, -0x1.7a79a40000000p-3, 0x1.fe7ed00000000p-6, -0x1.33bb000000000p-10, -0x1.145ce40000000p-3, -0x1.5d94a00000000p-3, -0x1.66d76a0000000p-3, -0x1.10048a0000000p-2, -0x1.a367de0000000p-3, -0x1.f4f36c0000000p-5}
, {-0x1.082cfa0000000p-4, -0x1.a381e00000000p-6, 0x1.c92c860000000p-4, 0x1.467c6c0000000p-2, -0x1.94ea320000000p-2, -0x1.780b480000000p-4, 0x1.a21d780000000p-3, -0x1.4548dc0000000p-2, -0x1.1c7ab40000000p-3, -0x1.b995b20000000p-2, -0x1.20b0b80000000p-2, -0x1.a0250c0000000p-4, -0x1.75e8d60000000p-4, -0x1.cc969a0000000p-2, 0x1.5733ca0000000p-3, 0x1.08cbe40000000p-5}
}
, {{-0x1.191dbc0000000p-4, -0x1.5251240000000p-4, 0x1.29607c0000000p-3, 0x1.c66d880000000p-4, -0x1.20ff6e0000000p-2, -0x1.18cc160000000p-2, -0x1.10c37e0000000p-4, 0x1.b610dc0000000p-5, 0x1.67ea0c0000000p-3, -0x1.c91e560000000p-4, 0x1.58703a0000000p-9, -0x1.790ae60000000p-4, 0x1.faaa360000000p-3, -0x1.28ba080000000p-2, -0x1.6810e40000000p-3, -0x1.1858b60000000p-3}
, {-0x1.09c13c0000000p-2, 0x1.2273ea0000000p-5, -0x1.a4db7a0000000p-3, -0x1.c1e8d60000000p-3, -0x1.b2145e0000000p-4, -0x1.7d12760000000p-3, -0x1.6195d40000000p-4, -0x1.29ff9e0000000p-2, -0x1.990e340000000p-5, -0x1.d66ada0000000p-3, -0x1.47eea80000000p-6, 0x1.e738e20000000p-3, -0x1.e33c940000000p-3, 0x1.57c0980000000p-3, 0x1.5877cc0000000p-3, 0x1.cb31fa0000000p-6}
}
, {{0x1.0113400000000p-1, 0x1.6ce03a0000000p-3, 0x1.4337620000000p-2, -0x1.d2b3dc0000000p-3, -0x1.ce1ede0000000p-3, 0x1.901c8a0000000p-5, 0x1.41f17a0000000p-3, -0x1.01d4600000000p-2, 0x1.c2edc80000000p-3, 0x1.1dda0c0000000p-2, -0x1.0b0bde0000000p+0, -0x1.695a2a0000000p-3, 0x1.9797c80000000p-5, 0x1.6b0a820000000p-7, 0x1.3df5460000000p-5, -0x1.4ed0960000000p-4}
, {-0x1.adc3880000000p-2, 0x1.702d3e0000000p-3, 0x1.f321720000000p-2, 0x1.ae00340000000p-3, 0x1.d77e1e0000000p-3, -0x1.0841f20000000p-1, 0x1.543ba60000000p-3, 0x1.8976a20000000p-4, 0x1.feb4dc0000000p-3, 0x1.da43180000000p-3, 0x1.97d53c0000000p-2, -0x1.2c314e0000000p-7, 0x1.3fa7d40000000p-3, -0x1.3889fe0000000p-4, -0x1.3e03e00000000p-3, 0x1.11f0640000000p-2}
}
, {{-0x1.0031080000000p-4, -0x1.2424fa0000000p-2, 0x1.3dc64e0000000p-3, -0x1.29fa180000000p-4, -0x1.19f7980000000p-3, 0x1.3f9bd80000000p-4, -0x1.51bba40000000p-3, 0x1.96a3a60000000p-3, -0x1.1672c20000000p-2, -0x1.6c23ac0000000p-4, -0x1.cbdfac0000000p-4, 0x1.25c1460000000p-2, 0x1.e2687a0000000p-3, -0x1.fa789e0000000p-4, 0x1.3528240000000p-4, -0x1.12ba260000000p-6}
, {-0x1.e3470a0000000p-3, -0x1.3a72020000000p-2, -0x1.8ea3c80000000p-3, 0x1.aac40a0000000p-5, 0x1.575cee0000000p-5, 0x1.42e9980000000p-3, 0x1.df47240000000p-3, -0x1.115a840000000p-3, -0x1.cd15bc0000000p-5, -0x1.619f5e0000000p-2, -0x1.33a7560000000p-2, 0x1.a9c4c60000000p-3, -0x1.d441f40000000p-4, -0x1.13964e0000000p-5, 0x1.c5548a0000000p-4, -0x1.8e329e0000000p-3}
}
, {{-0x1.4791be0000000p-2, 0x1.635aee0000000p-3, 0x1.29a0140000000p-1, 0x1.a2b4880000000p-7, 0x1.540fec0000000p-3, -0x1.6ad8600000000p-3, -0x1.1339540000000p-4, -0x1.fb23760000000p-4, -0x1.9d94c00000000p-8, 0x1.1549260000000p-4, -0x1.f5b83a0000000p-1, -0x1.14ff1a0000000p-2, -0x1.b5fff20000000p-3, -0x1.3037080000000p-4, 0x1.5cea740000000p-3, 0x1.2026ce0000000p-2}
, {-0x1.be23ec0000000p-1, 0x1.19db9a0000000p-3, 0x1.6671260000000p-4, 0x1.65994a0000000p-3, -0x1.2bfa7e0000000p-3, -0x1.f785060000000p-2, 0x1.0207280000000p-2, -0x1.8674420000000p-3, -0x1.f115a00000000p-4, 0x1.62d63a0000000p-3, -0x1.11952c0000000p+0, -0x1.e6e7d40000000p-3, 0x1.0d64620000000p-5, -0x1.431aa00000000p-3, 0x1.d029060000000p-3, 0x1.909c240000000p-3}
}
, {{-0x1.2ba7160000000p-3, -0x1.2db8f40000000p-3, -0x1.947f8e0000000p-4, -0x1.7495980000000p-3, -0x1.b82e5c0000000p-4, -0x1.488e4a0000000p-8, 0x1.23711e0000000p-4, -0x1.06f26e0000000p-2, 0x1.15b0580000000p-4, -0x1.54a3f20000000p-2, -0x1.f3ef8a0000000p-3, 0x1.0cb57a0000000p-5, 0x1.1875420000000p-3, -0x1.35e7b00000000p-2, -0x1.27dc5a0000000p-3, -0x1.8c65ee0000000p-3}
, {0x1.6c03940000000p-3, 0x1.6eb5ec0000000p-4, 0x1.0d8db20000000p-5, -0x1.6cfeb20000000p-4, 0x1.bc296e0000000p-4, -0x1.2e9a760000000p-5, -0x1.e27c540000000p-5, 0x1.0d86c20000000p-6, 0x1.73ed300000000p-4, 0x1.9d3cb60000000p-6, -0x1.c076d60000000p-3, 0x1.3ff3f00000000p-5, -0x1.f8c1440000000p-4, -0x1.30a3500000000p-4, -0x1.470e1a0000000p-3, -0x1.1878800000000p-2}
}
, {{-0x1.5eee500000000p-9, 0x1.4a83120000000p-4, -0x1.ba516e0000000p-3, 0x1.00219e0000000p-3, 0x1.9d25fc0000000p-4, -0x1.10c2640000000p+0, -0x1.964f040000000p-2, -0x1.f847840000000p-3, -0x1.e475240000000p-5, -0x1.0787000000000p-2, -0x1.6e17f20000000p-1, 0x1.c578f20000000p-3, -0x1.132a080000000p-7, -0x1.84547e0000000p-2, 0x1.9000620000000p-5, -0x1.734c080000000p-8}
, {-0x1.2c7fd20000000p+0, 0x1.74f3d60000000p-3, 0x1.0e3e740000000p-3, -0x1.59262e0000000p-4, 0x1.932cb40000000p-4, -0x1.403da40000000p-1, 0x1.ff62d20000000p-6, 0x1.d1b50a0000000p-4, -0x1.2193f80000000p-2, 0x1.ab86760000000p-6, -0x1.9b4fc80000000p+0, 0x1.dd42fc0000000p-3, -0x1.3b90d00000000p-6, 0x1.99d09a0000000p-3, -0x1.8c0fee0000000p-5, 0x1.836dd20000000p-2}
}
, {{-0x1.2e99080000000p-3, -0x1.d420d00000000p-2, -0x1.1617f60000000p+0, 0x1.266b8a0000000p-4, -0x1.01e1c40000000p-1, -0x1.369e820000000p-2, -0x1.ed2f120000000p-2, -0x1.3ffc0c0000000p-3, -0x1.dc9b880000000p-4, -0x1.71b6ae0000000p-1, 0x1.87afca0000000p-1, 0x1.0912e40000000p-3, -0x1.140c780000000p-3, -0x1.2664280000000p-1, 0x1.0c8d200000000p-3, -0x1.baf8b00000000p-2}
, {0x1.e6d5240000000p-3, -0x1.008c440000000p-1, -0x1.76dc4e0000000p-1, 0x1.86b87c0000000p-3, -0x1.d811160000000p-2, -0x1.037a920000000p-2, -0x1.2df3280000000p+0, -0x1.3aa7040000000p-4, -0x1.b7b98a0000000p-3, -0x1.9dd6ee0000000p-1, 0x1.7a06860000000p-1, -0x1.058bbc0000000p-3, 0x1.58823c0000000p-2, -0x1.82d6580000000p-1, 0x1.34b0f40000000p-6, -0x1.4c69640000000p-1}
}
, {{0x1.2272ee0000000p-1, -0x1.89f36a0000000p-3, -0x1.f1cdbc0000000p-3, -0x1.613b600000000p-7, -0x1.135e8c0000000p-4, 0x1.81840c0000000p-3, -0x1.89bc6a0000000p-3, -0x1.62c0140000000p-2, 0x1.9ae2300000000p-5, -0x1.9ea1800000000p-3, 0x1.59b7a60000000p-1, 0x1.ddca400000000p-7, 0x1.09293e0000000p-2, -0x1.2f183e0000000p-1, -0x1.e688e20000000p-4, -0x1.8ef1300000000p-4}
, {0x1.032cc00000000p+1, -0x1.1d69f60000000p-4, -0x1.7afb540000000p-1, -0x1.4b22200000000p-3, -0x1.1e00920000000p-2, 0x1.070aee0000000p-2, -0x1.1fda4c0000000p-4, -0x1.1bd0220000000p-2, -0x1.79b23c0000000p-3, 0x1.ae52140000000p-4, 0x1.51d6d80000000p-5, -0x1.7a38540000000p-3, -0x1.6589500000000p-6, -0x1.56632a0000000p-1, 0x1.c37fb80000000p-4, -0x1.aa39f00000000p-3}
}
, {{-0x1.dce6d80000000p-1, -0x1.ec46200000000p-3, -0x1.05a0a20000000p-1, -0x1.5a76dc0000000p-4, -0x1.41d9dc0000000p-2, 0x1.7abba00000000p-1, -0x1.0dccae0000000p-2, -0x1.145f2a0000000p-8, 0x1.4c62080000000p-8, -0x1.eb02760000000p-5, -0x1.8b7d340000000p-1, -0x1.08f42c0000000p-2, -0x1.814aba0000000p-2, -0x1.3527440000000p-3, -0x1.1dd9620000000p-5, -0x1.13db0e0000000p-2}
, {-0x1.92f1040000000p-2, 0x1.64af720000000p-3, -0x1.8475c80000000p-5, -0x1.4c4a3c0000000p-3, 0x1.0e08ec0000000p-2, 0x1.1d17e80000000p+1, 0x1.d8cd6e0000000p-7, -0x1.2e58480000000p-7, 0x1.3387720000000p-4, 0x1.a82bca0000000p-2, 0x1.2401dc0000000p-4, 0x1.1a76300000000p-5, -0x1.1f62d60000000p-4, 0x1.b17a7c0000000p-4, -0x1.27d9100000000p-3, 0x1.71c5a60000000p-3}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_GROUPS
/**
  ******************************************************************************
  * @file    maxpool1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _MAX_POOLING1D_135_H_
#define _MAX_POOLING1D_135_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  32
#define INPUT_SAMPLES   123
#define POOL_SIZE       4
#define POOL_STRIDE     4
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

typedef float max_pooling1d_135_output_type[POOL_LENGTH][INPUT_CHANNELS];

#if 0
void max_pooling1d_135(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  number_t output[POOL_LENGTH][INPUT_CHANNELS]); 	// OUT
#endif

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH

#endif//_MAX_POOLING1D_135_H_
/**
  ******************************************************************************
  * @file    maxpool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "max_pooling1d_135.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#endif

#define INPUT_CHANNELS  32
#define INPUT_SAMPLES   123
#define POOL_SIZE       4
#define POOL_STRIDE     4
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

// For fixed point quantization
#define INPUT_SCALE_FACTOR 0
#define OUTPUT_SCALE_FACTOR 0
#define OUTPUT_ROUND_MODE ROUND_MODE_NONE
#define NUMBER_T float
#define LONG_NUMBER_T float


static inline void max_pooling1d_135(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  NUMBER_T output[POOL_LENGTH][INPUT_CHANNELS]) {	// OUT

  unsigned short pos_x, k; 	// loop indexes for output volume
  unsigned int x;
  static LONG_NUMBER_T max[INPUT_CHANNELS];

  for (pos_x = 0; pos_x < POOL_LENGTH; pos_x++) {
    for (k = 0; k < INPUT_CHANNELS; k++) {
#ifdef ACTIVATION_LINEAR
      max[k] = input[pos_x*POOL_STRIDE][k];
      x = 1;
#elif defined(ACTIVATION_RELU)
      max[k] = 0;
      x = 0;
#else
#error "Unsupported activation function"
#endif
    }

    for (; x < POOL_SIZE; x++) {
      for (k = 0; k < INPUT_CHANNELS; k++) {
        if (max[k] < input[(pos_x * POOL_STRIDE) + x][k])
          max[k] = input[(pos_x * POOL_STRIDE) + x][k];
      }
    }

    for (k = 0; k < INPUT_CHANNELS; k++) {
      output[pos_x][k] = scale_and_clamp_to(NUMBER_T, max[k], INPUT_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
    }
  }
}

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH
#undef ACTIVATION_LINEAR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    conv1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _CONV1D_111_H_
#define _CONV1D_111_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      32
#define INPUT_SAMPLES       30
#define CONV_FILTERS        32
#define CONV_KERNEL_SIZE    2
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

typedef float conv1d_111_output_type[CONV_OUTSAMPLES][CONV_FILTERS];

#if 0
void conv1d_111(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const number_t kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS],  // IN

  const number_t bias[CONV_FILTERS],						                          // IN

  number_t output[CONV_OUTSAMPLES][CONV_FILTERS]);                       // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES

#endif//_CONV1D_111_H_
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Sébastien Bilavarn, LEAT, CNRS, Université Côte d'Azur, France
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "conv1d_111.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_CHANNELS      32
#define INPUT_SAMPLES       30
#define CONV_FILTERS        32
#define CONV_KERNEL_SIZE    2
#define CONV_STRIDE         1
#define CONV_GROUPS         1
#define CHANNELS_PER_GROUP  (INPUT_CHANNELS / CONV_GROUPS)
#define FILTERS_PER_GROUP   (CONV_FILTERS / CONV_GROUPS)

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 0
#define BIASES_SCALE_FACTOR 0
#define TMP_SCALE_FACTOR 0
#define INPUT_SCALE_FACTOR 0
#define OUTPUT_SCALE_FACTOR 0
#define OUTPUT_ROUND_MODE ROUND_MODE_NONE
#define NUMBER_T float
#define LONG_NUMBER_T float


static inline void conv1d_111(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const NUMBER_T kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS],  // IN

  const NUMBER_T bias[CONV_FILTERS],						                          // IN

  NUMBER_T output[CONV_OUTSAMPLES][CONV_FILTERS]) {                       // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  int input_x;
  LONG_NUMBER_T output_acc;

  for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
    for (k = 0; k < CONV_FILTERS; k++) { 
      output_acc = 0;

      for (x = 0; x < CONV_KERNEL_SIZE; x++) {
        input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;

        if (input_x >= 0 && input_x < INPUT_SAMPLES) { // ZeroPadding1D
          for (z = 0; z < INPUT_CHANNELS / CONV_GROUPS; z++) {
            output_acc += (LONG_NUMBER_T)input[input_x][z + (k / FILTERS_PER_GROUP) * CHANNELS_PER_GROUP] * (LONG_NUMBER_T)kernel[k][x][z];
          }
        }
      }

    // Scale for possible additional precision of bias
    output_acc = scale(NUMBER_T, output_acc, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);

    // Scale bias to match accumulator
    output_acc += scale(NUMBER_T, (LONG_NUMBER_T)bias[k], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);

      
#ifdef ACTIVATION_LINEAR
      output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
      // Activation function: ReLU
      if (output_acc < 0) {
        output[pos_x][k] = 0;
      } else {
#if defined(ACTIVATION_RELU6)
        if (output_acc > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
          output_acc = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
        }
#endif
        output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      }
#else
#error "Unsupported activation function"
#endif
    }
  }

#else

#if BIASES_SCALE_FACTOR > WEIGHTS_SCALE_FACTOR
#error "CMSIS-NN does not support BIASES_SCALE_FACTOR larger than WEIGHTS_SCALE_FACTOR"
#endif

#error "Data type unsupported by CMSIS-NN"

#endif
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef CONV_GROUPS
#undef CHANNELS_PER_GROUP
#undef FILTERS_PER_GROUP
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_RELU
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef TMP_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    weights/conv1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_CHANNELS    32
#define CONV_FILTERS      32
#define CONV_KERNEL_SIZE  2
#define CONV_GROUPS       1


const float  conv1d_111_bias[CONV_FILTERS] = {0x1.42fee20000000p-6, -0x1.9f47cc0000000p-5, -0x1.88c1e40000000p-3, -0x1.e442260000000p-4, -0x1.4dd6b20000000p-4, -0x1.9d99ec0000000p-5, -0x1.9bda880000000p-3, -0x1.1e671a0000000p-4, 0x1.1a629c0000000p-2, -0x1.c7cfc20000000p-5, -0x1.92f78c0000000p-3, -0x1.0430160000000p-2, -0x1.5b6b3e0000000p-4, 0x1.31b5a00000000p-2, 0x1.e1e00a0000000p-3, -0x1.f983ec0000000p-3, -0x1.07070e0000000p-3, 0x1.d8a8c20000000p-4, -0x1.d4cdea0000000p-5, -0x1.7e7ef80000000p-6, 0x1.3223f40000000p-5, -0x1.e723ce0000000p-6, -0x1.e603560000000p-6, -0x1.556be40000000p-5, 0x1.e834360000000p-7, -0x1.65a7e40000000p-4, -0x1.fd46040000000p-5, -0x1.68665a0000000p-5, -0x1.fab7a40000000p-3, -0x1.0c127c0000000p-4, -0x1.5a1cde0000000p-4, -0x1.aec40e0000000p-3}
;

const float  conv1d_111_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{{0x1.ae18460000000p-6, 0x1.85678c0000000p+0, -0x1.b3b7880000000p-10, -0x1.3d983a0000000p-1, 0x1.c802740000000p-5, 0x1.2cdcc60000000p-3, 0x1.758f500000000p-5, 0x1.49cde80000000p-3, -0x1.3a71560000000p-3, -0x1.6393fa0000000p-3, -0x1.51e4880000000p-4, -0x1.eae37c0000000p-5, 0x1.5537620000000p-4, 0x1.93abd40000000p-3, -0x1.9f9d0c0000000p-2, 0x1.6384f00000000p-3, 0x1.3074ba0000000p+0, -0x1.9326f80000000p-2, -0x1.a7799a0000000p-5, 0x1.6405fe0000000p-2, -0x1.8cdf1e0000000p+0, -0x1.a09c8e0000000p-3, 0x1.21e3420000000p-2, 0x1.50be240000000p-5, -0x1.add0500000000p-5, 0x1.c4c89e0000000p-7, 0x1.4656c80000000p-8, -0x1.2b24000000000p-3, -0x1.d5fef20000000p-6, 0x1.5ebd6e0000000p-2, 0x1.3f69920000000p-1, -0x1.6ecde00000000p-2}
, {-0x1.9f192e0000000p-3, 0x1.9f467e0000000p-1, -0x1.a773160000000p-3, 0x1.3900860000000p-3, 0x1.0217520000000p-5, -0x1.037ee00000000p-3, -0x1.4d7b940000000p-4, 0x1.9ae6a60000000p-3, 0x1.b1a1920000000p-2, 0x1.5b0da20000000p-4, -0x1.3e00e20000000p-2, -0x1.2c880e0000000p-3, 0x1.c8e1680000000p-7, 0x1.6e7e740000000p-7, -0x1.017aac0000000p-2, -0x1.9749440000000p-7, 0x1.3e14e60000000p-1, -0x1.701ace0000000p+0, 0x1.b0235c0000000p-5, 0x1.4f6e780000000p-3, -0x1.0876a40000000p-2, 0x1.a69fc80000000p-2, 0x1.697fce0000000p-2, 0x1.50f3940000000p-4, -0x1.7f6f420000000p-3, -0x1.4c9cd00000000p-7, 0x1.986a0c0000000p-3, 0x1.360a200000000p-2, 0x1.a6c7220000000p-3, 0x1.7a08140000000p-2, 0x1.95e58c0000000p-2, 0x1.3b207c0000000p-1}
}
, {{0x1.318cc20000000p-3, -0x1.a9b68a0000000p-3, 0x1.faa3700000000p-6, -0x1.238c1e0000000p-4, 0x1.135a140000000p-3, 0x1.7741f40000000p-3, -0x1.d81bce0000000p-6, -0x1.76f3940000000p-3, -0x1.ade7140000000p-3, -0x1.5d608e0000000p-4, -0x1.47a13a0000000p-2, -0x1.6fa0060000000p-5, 0x1.23e14a0000000p-4, -0x1.b9eae20000000p-5, -0x1.afaa3e0000000p-3, -0x1.b15f600000000p-4, -0x1.198c4e0000000p-2, -0x1.54705e0000000p-5, -0x1.2192e00000000p-4, -0x1.1998a00000000p-4, 0x1.da10000000000p-4, -0x1.1fd95a0000000p-3, -0x1.5640460000000p-4, -0x1.3edb180000000p-8, -0x1.0237820000000p-3, -0x1.b5b55e0000000p-5, -0x1.31ac580000000p-3, -0x1.47b58a0000000p-6, -0x1.5b59b80000000p-3, 0x1.90fda80000000p-6, -0x1.2001360000000p-4, -0x1.24cc360000000p-4}
, {-0x1.b47b300000000p-3, 0x1.29ce840000000p-3, -0x1.b23a940000000p-3, -0x1.45f7da0000000p-4, 0x1.c919520000000p-4, 0x1.ac24320000000p-3, 0x1.dd68f80000000p-6, -0x1.712b7a0000000p-7, 0x1.cc29240000000p-5, 0x1.2de5b80000000p-3, 0x1.69df740000000p-4, -0x1.05e2a60000000p-2, 0x1.6dccea0000000p-5, -0x1.00a6bc0000000p-2, -0x1.f12d5e0000000p-3, -0x1.b84d280000000p-3, -0x1.63bbea0000000p-4, 0x1.e07ea00000000p-7, 0x1.a119240000000p-4, -0x1.ae06400000000p-3, -0x1.00dd0e0000000p-4, 0x1.8096740000000p-5, -0x1.c74e760000000p-7, 0x1.37ccdc0000000p-4, -0x1.e709e60000000p-6, -0x1.726c1c0000000p-5, -0x1.0f7ef60000000p-4, 0x1.2364d20000000p-3, -0x1.c182460000000p-4, 0x1.863e700000000p-7, -0x1.1d93ce0000000p-2, 0x1.a1e3c00000000p-6}
}
, {{0x1.e4c4840000000p-5, -0x1.74c4380000000p-2, -0x1.324ac20000000p-3, -0x1.99989e0000000p-3, -0x1.29b09e0000000p-4, -0x1.ccab540000000p-3, -0x1.075e440000000p-2, 0x1.9a80e80000000p-4, 0x1.ec5c1c0000000p-4, -0x1.2bd2b20000000p-5, -0x1.5b875e0000000p-2, -0x1.1193540000000p-4, -0x1.0e812c0000000p-3, -0x1.f8dada0000000p-3, -0x1.ec01120000000p-3, 0x1.c334ac0000000p-6, -0x1.3c13020000000p-2, -0x1.29f8080000000p-1, 0x1.7a7b340000000p-4, -0x1.badbcc0000000p-5, 0x1.2bf6da0000000p-5, -0x1.64e60c0000000p-2, -0x1.2837b40000000p-3, -0x1.9f54060000000p-3, -0x1.82ec7e0000000p-2, 0x1.2a5a4a0000000p-4, -0x1.2c493e0000000p-3, -0x1.13eee20000000p-2, -0x1.ecf0600000000p-8, -0x1.3d88140000000p-9, -0x1.0943800000000p-5, -0x1.5180040000000p-3}
, {-0x1.628db60000000p-5, -0x1.6cbe8c0000000p-2, -0x1.8e8ec20000000p-3, -0x1.8088c00000000p-10, 0x1.ca69140000000p-4, 0x1.abbf660000000p-5, -0x1.88e9140000000p-3, -0x1.b57e3e0000000p-4, 0x1.aa23ae0000000p-4, -0x1.320fa40000000p-3, -0x1.e950ae0000000p-3, 0x1.0d90560000000p-3, -0x1.b5dcce0000000p-3, -0x1.fa85fe0000000p-4, -0x1.d0b76c0000000p-2, -0x1.71f3ee0000000p-4, -0x1.186cfe0000000p-4, -0x1.13ceb40000000p-2, 0x1.5b82a40000000p-3, -0x1.11553a0000000p-3, -0x1.6a330e0000000p-2, -0x1.89096c0000000p-5, 0x1.847ef80000000p-4, 0x1.9812960000000p-5, -0x1.465a160000000p-2, -0x1.14f0be0000000p-2, -0x1.a797400000000p-3, -0x1.7dc4be0000000p-3, 0x1.88fb860000000p-7, -0x1.1b8d7e0000000p-3, -0x1.d4c9860000000p-3, -0x1.70f2500000000p-2}
}
, {{-0x1.5956ac0000000p-5, -0x1.b2377c0000000p-5, -0x1.7f04920000000p-5, 0x1.3eb5180000000p-1, 0x1.2da4140000000p-6, -0x1.14b3d80000000p-4, -0x1.29dfc00000000p-6, 0x1.ae26180000000p-9, -0x1.4385be0000000p-2, -0x1.1d0be00000000p-3, -0x1.9482040000000p-1, 0x1.1a79320000000p-6, 0x1.15a0c60000000p-3, -0x1.1e95060000000p-5, 0x1.ff31340000000p-8, -0x1.181d880000000p-3, 0x1.da9b0a0000000p-3, 0x1.166b680000000p-1, 0x1.13a4740000000p-3, -0x1.bd51c40000000p-4, -0x1.1b83020000000p-3, 0x1.286b120000000p+1, -0x1.650cc00000000p-3, -0x1.33e9440000000p-3, -0x1.fc7fe00000000p-2, 0x1.5e2aca0000000p-4, -0x1.534bfa0000000p-2, 0x1.2b665e0000000p-4, 0x1.0467880000000p-3, 0x1.065df20000000p-1, 0x1.58f7d80000000p-2, 0x1.7bafaa0000000p-3}
, {0x1.80d7160000000p-4, -0x1.6a131c0000000p-3, -0x1.4fb7000000000p-5, 0x1.65da320000000p-2, 0x1.29ed8c0000000p-5, 0x1.2431620000000p-3, -0x1.a0410e0000000p-3, -0x1.5ddf220000000p-4, -0x1.2fe8e00000000p-3, -0x1.26bde20000000p-3, -0x1.5a8f840000000p-2, -0x1.23b2a00000000p-4, 0x1.5598d60000000p-6, -0x1.44d5d60000000p-5, -0x1.35bebe0000000p-1, -0x1.a8e01c0000000p-3, -0x1.d70dcc0000000p-2, 0x1.0b89c40000000p-3, -0x1.f54cb40000000p-5, 0x1.db7aa40000000p-3, 0x1.fba70c0000000p-3, -0x1.9cf9ee0000000p+0, 0x1.44ebd80000000p-3, 0x1.6e515c0000000p-3, -0x1.e553e40000000p-3, -0x1.8368520000000p-5, 0x1.1f6b7e0000000p-3, -0x1.6c488e0000000p-5, -0x1.b454d60000000p-4, -0x1.3f76d20000000p-1, -0x1.871ff20000000p-1, 0x1.d85a740000000p-2}
}
, {{-0x1.e1d5940000000p-3, -0x1.3930d20000000p-2, -0x1.9639fc0000000p-2, -0x1.5468c00000000p-2, -0x1.afaeee0000000p-4, 0x1.a15bf60000000p-5, 0x1.82f0a80000000p-4, -0x1.dd60240000000p-4, -0x1.2ebc6e0000000p-3, 0x1.c05c960000000p-3, -0x1.8aad500000000p-2, -0x1.98fe3e0000000p-4, -0x1.63f3f60000000p-4, 0x1.88268c0000000p-5, 0x1.906c9a0000000p-4, 0x1.1bccbc0000000p-4, -0x1.74f74e0000000p-2, -0x1.ffe2e20000000p-3, -0x1.01d0320000000p-3, -0x1.88071a0000000p-2, -0x1.8f58f20000000p-3, -0x1.e7fa3c0000000p-2, 0x1.a4747e0000000p-3, 0x1.f759560000000p-5, -0x1.f994c40000000p-3, 0x1.4de07e0000000p-3, -0x1.56fbd40000000p-3, 0x1.1aae0a0000000p-3, -0x1.3e2f1a0000000p-2, -0x1.0bb0ae0000000p-2, -0x1.c5573a0000000p-2, 0x1.2ffc640000000p-4}
, {0x1.52f9ae0000000p-4, -0x1.3d95d80000000p-3, -0x1.3401960000000p-2, -0x1.13224c0000000p-2, 0x1.b7fdc00000000p-6, -0x1.11249c0000000p-3, -0x1.c8e0400000000p-5, 0x1.68878a0000000p-3, 0x1.906ed00000000p-3, 0x1.67c2c20000000p-6, 0x1.2bdb2e0000000p-6, 0x1.d5d9b00000000p-4, -0x1.213d680000000p-5, -0x1.aa47740000000p-3, 0x1.7f1e9e0000000p-2, -0x1.80d3340000000p-4, -0x1.c6fad20000000p-3, -0x1.aa9e4e0000000p-3, -0x1.99c1900000000p-9, -0x1.2ed9860000000p-3, -0x1.8d77120000000p-3, -0x1.be70660000000p-4, -0x1.4e78ae0000000p-7, 0x1.3bdaf20000000p-4, -0x1.e716e40000000p-3, -0x1.45289c0000000p-3, -0x1.a24d9e0000000p-3, -0x1.5a75940000000p-3, -0x1.357bb40000000p-3, -0x1.3c28200000000p-2, -0x1.adaf6a0000000p-2, -0x1.cfeb2e0000000p-3}
}
, {{-0x1.32c6400000000p-4, -0x1.e7dc1a0000000p-3, -0x1.adc2880000000p-3, 0x1.f8f8080000000p-3, 0x1.24dc300000000p-4, -0x1.833ec20000000p-3, -0x1.eb667a0000000p-3, 0x1.b30dd80000000p-4, 0x1.4fe4e20000000p-3, -0x1.5003700000000p-3, -0x1.38e5060000000p-5, 0x1.1ad72e0000000p-5, 0x1.0ce39a0000000p-3, -0x1.1ba3180000000p-4, 0x1.02a9d00000000p-4, 0x1.5c6b7e0000000p-4, -0x1.2afc8e0000000p-4, -0x1.5b54bc0000000p-3, -0x1.8685800000000p-3, -0x1.840ec60000000p-3, -0x1.163b480000000p-8, -0x1.e349b00000000p-3, 0x1.2d12bc0000000p-5, -0x1.2338560000000p-5, -0x1.4a624a0000000p-3, -0x1.10d1500000000p-3, -0x1.649cc60000000p-4, -0x1.b828480000000p-4, 0x1.6515680000000p-3, -0x1.0b5d6c0000000p-3, -0x1.6dfd940000000p-5, 0x1.7a73720000000p-4}
, {-0x1.d3697a0000000p-6, 0x1.c6c2480000000p-4, 0x1.37f0340000000p-4, -0x1.076e020000000p-3, 0x1.940a4e0000000p-4, -0x1.26925e0000000p-5, 0x1.35a72a0000000p-4, 0x1.55a39a0000000p-5, -0x1.7a68fc0000000p-5, 0x1.a7dd3e0000000p-4, -0x1.30a5da0000000p-3, -0x1.8043260000000p-5, 0x1.19baf60000000p-4, -0x1.7cd1ae0000000p-6, 0x1.45b62c0000000p-4, 0x1.12cf1a0000000p-3, 0x1.1468f60000000p-4, -0x1.2f70440000000p-4, -0x1.6465c00000000p-3, -0x1.2e1cec0000000p-2, -0x1.d07e900000000p-3, 0x1.5b124a0000000p-4, -0x1.56c3060000000p-3, -0x1.0a6b300000000p-10, -0x1.4e2e2e0000000p-4, 0x1.47edd60000000p-3, -0x1.9e507c0000000p-3, -0x1.a2eeec0000000p-7, -0x1.1fc7ae0000000p-10, 0x1.3ba17c0000000p-5, 0x1.92c2620000000p-4, -0x1.f4a6260000000p-11}
}
, {{0x1.53df400000000p-3, -0x1.2d39b00000000p+0, -0x1.5728d40000000p-5, -0x1.c3ac8e0000000p-1, 0x1.3b933e0000000p-5, -0x1.073e440000000p-3, 0x1.74979a0000000p-3, -0x1.4fbc060000000p-5, 0x1.31f6920000000p-5, -0x1.71de160000000p-4, -0x1.7c72340000000p-1, -0x1.0389320000000p-6, 0x1.f1c3520000000p-5, 0x1.657cc20000000p-3, 0x1.a0bd8a0000000p-3, 0x1.400fb40000000p-3, -0x1.1079220000000p+0, 0x1.75ff900000000p-1, -0x1.5d574c0000000p-5, -0x1.c8c5240000000p-3, 0x1.5f78300000000p-1, -0x1.5f50f80000000p+0, 0x1.eccaf80000000p-5, 0x1.61aa0e0000000p-2, -0x1.952bfc0000000p-2, 0x1.8d02b20000000p-3, 0x1.406b6c0000000p-3, 0x1.8793140000000p-3, -0x1.afad120000000p-6, 0x1.4a36300000000p+0, 0x1.01712c0000000p+0, 0x1.5c367c0000000p-3}
, {-0x1.9b937c0000000p-3, -0x1.111e300000000p-3, 0x1.07119e0000000p-2, 0x1.187e720000000p-1, 0x1.e6131e0000000p-5, -0x1.841fbe0000000p-6, -0x1.4c30b00000000p-4, -0x1.de63ea0000000p-4, -0x1.f0ebc80000000p-6, 0x1.47aa580000000p-2, 0x1.5c2a4c0000000p-2, 0x1.65feec0000000p-4, -0x1.bf7fa40000000p-4, -0x1.0e9dea0000000p-3, -0x1.20847e0000000p+0, 0x1.cd1e7e0000000p-4, 0x1.537c200000000p-3, -0x1.2eeae40000000p-1, -0x1.86c7d80000000p-5, -0x1.3052e60000000p-1, 0x1.01c2900000000p-2, -0x1.3136ec0000000p+0, -0x1.6c88e60000000p-5, -0x1.dbb7b00000000p-8, 0x1.3572180000000p-3, -0x1.04c0520000000p-2, 0x1.0c86820000000p-3, -0x1.196fcc0000000p-2, 0x1.f87d740000000p-2, 0x1.befc880000000p-2, 0x1.f4908e0000000p-3, 0x1.6d71ce0000000p-10}
}
, {{-0x1.f0fd660000000p-9, -0x1.5e49ec0000000p-5, 0x1.3060ca0000000p-5, -0x1.81aa080000000p-4, -0x1.50ec0a0000000p-4, -0x1.8945ae0000000p-3, -0x1.efdace0000000p-7, -0x1.37fbfa0000000p-5, 0x1.5cb0460000000p-3, 0x1.b570960000000p-7, 0x1.27a1fa0000000p-4, -0x1.0595960000000p-3, 0x1.6d3c640000000p-7, 0x1.969b840000000p-4, -0x1.5761d80000000p-3, -0x1.6440c00000000p-3, -0x1.82a5ca0000000p-3, -0x1.26cffe0000000p-5, -0x1.57c51e0000000p-3, -0x1.5c4a9e0000000p-3, 0x1.5b09320000000p-8, -0x1.7531d20000000p-5, -0x1.a6bee40000000p-4, -0x1.71c9080000000p-3, -0x1.63e6cc0000000p-3, 0x1.09b3a20000000p-4, -0x1.45bcac0000000p-7, 0x1.183e980000000p-5, 0x1.e28e140000000p-5, 0x1.1ceeba0000000p-3, -0x1.6629860000000p-3, 0x1.6994ca0000000p-4}
, {-0x1.dba07c0000000p-4, -0x1.e63c720000000p-3, -0x1.0e39e80000000p-2, 0x1.c68e060000000p-7, 0x1.17ad360000000p-3, 0x1.34b4960000000p-3, 0x1.ef8ba20000000p-5, -0x1.20a1fc0000000p-5, -0x1.425ac80000000p-4, -0x1.82e3e60000000p-3, -0x1.8a44d00000000p-3, 0x1.c6ae7c0000000p-6, 0x1.e1f5e60000000p-4, -0x1.055da80000000p-5, 0x1.f5b41c0000000p-4, -0x1.74e6d40000000p-3, 0x1.0136080000000p-3, -0x1.3ac7be0000000p-3, 0x1.ce7f260000000p-5, -0x1.950f540000000p-5, -0x1.daeffa0000000p-5, 0x1.c6881e0000000p-4, 0x1.a15bba0000000p-5, -0x1.9aa5040000000p-3, -0x1.545a1a0000000p-4, -0x1.43472a0000000p-4, -0x1.375e1c0000000p-2, 0x1.4b3f220000000p-5, -0x1.20c2700000000p-2, -0x1.d3d5500000000p-4, -0x1.497c860000000p-3, 0x1.9c812c0000000p-5}
}
, {{0x1.5b1d600000000p-3, -0x1.15560c0000000p-1, 0x1.5b398e0000000p-3, -0x1.4d7d660000000p+0, -0x1.fe20d40000000p-7, 0x1.635ff20000000p-4, 0x1.43da740000000p-5, -0x1.8d42440000000p-4, -0x1.7501300000000p-3, 0x1.66207e0000000p-4, 0x1.2c0bf20000000p-6, 0x1.8041320000000p-4, 0x1.900f160000000p-3, 0x1.5be8e20000000p-4, -0x1.9f88120000000p-2, 0x1.a48b4e0000000p-4, 0x1.0ded040000000p-1, 0x1.f384480000000p-2, 0x1.c9bdac0000000p-4, -0x1.66427e0000000p-5, -0x1.3d61540000000p-4, -0x1.c061520000000p-4, -0x1.6423380000000p-4, 0x1.08f8140000000p-3, -0x1.de83680000000p-6, 0x1.ff95160000000p-4, 0x1.93896c0000000p-2, -0x1.7f5d920000000p-6, 0x1.c83efc0000000p-2, -0x1.f3d6440000000p-2, -0x1.3992300000000p-2, -0x1.d8e24a0000000p-1}
, {-0x1.2a6c060000000p-8, 0x1.73cc580000000p-2, 0x1.cf0ea00000000p-4, -0x1.5f62d60000000p-1, -0x1.4da8e40000000p-3, 0x1.c3c7a40000000p-4, -0x1.b394e80000000p-7, 0x1.76e3d40000000p-3, 0x1.08fbf20000000p-3, -0x1.0b79480000000p-3, -0x1.a29f080000000p-1, 0x1.e8d96a0000000p-4, -0x1.83a8ee0000000p-7, -0x1.4583160000000p-3, -0x1.eb4e420000000p-1, -0x1.a47a3e0000000p-7, 0x1.8aeff20000000p-2, 0x1.5dc2f00000000p-5, -0x1.f3bbbe0000000p-3, -0x1.7763120000000p-2, -0x1.bf216a0000000p-2, 0x1.656f2a0000000p-2, -0x1.3722340000000p-3, -0x1.5e82720000000p-3, 0x1.14db160000000p-3, -0x1.e2435c0000000p-4, 0x1.4e2a9c0000000p-2, 0x1.71fcc60000000p-4, 0x1.4a46480000000p-3, -0x1.de430a0000000p-2, -0x1.722c360000000p-2, -0x1.4e783c0000000p+0}
}
, {{-0x1.9fb3e80000000p-5, -0x1.f92ae80000000p-2, 0x1.90fd480000000p-2, -0x1.0930de0000000p-2, 0x1.c7c1ba0000000p-4, -0x1.afb0340000000p-4, 0x1.6eace40000000p-4, -0x1.8654880000000p-3, 0x1.58ccd80000000p-6, 0x1.8f211a0000000p-4, -0x1.1a20c60000000p-2, 0x1.e747b60000000p-7, 0x1.1da9500000000p-4, -0x1.93c4c00000000p-7, 0x1.b763580000000p-7, 0x1.84b8be0000000p-4, -0x1.0e7b220000000p-2, 0x1.1dfee60000000p-3, -0x1.49c1560000000p-3, 0x1.d345c40000000p-3, 0x1.eda2b40000000p-5, -0x1.5a127c0000000p-7, -0x1.4094b00000000p-4, -0x1.4e53020000000p-3, 0x1.0dff3a0000000p-2, -0x1.85b2560000000p-7, 0x1.21a8860000000p-2, -0x1.c2b2ec0000000p-4, 0x1.5425d00000000p-3, 0x1.29618a0000000p-3, -0x1.81c54c0000000p-4, -0x1.d3b9260000000p-3}
, {-0x1.37db800000000p-3, -0x1.d46c7c0000000p-1, 0x1.745c8e0000000p-2, -0x1.98ba020000000p-3, 0x1.7083940000000p-4, 0x1.c355c00000000p-3, -0x1.14e0bc0000000p-3, -0x1.4fa8400000000p-4, 0x1.1707cc0000000p-4, -0x1.a636ee0000000p-5, -0x1.27697e0000000p-4, -0x1.95f3860000000p-4, -0x1.ade4bc0000000p-7, -0x1.b4c79c0000000p-4, -0x1.7742300000000p-2, 0x1.3180f80000000p-4, -0x1.7f9d760000000p-2, -0x1.0ca8280000000p-1, -0x1.d2835a0000000p-3, 0x1.b455e40000000p-4, 0x1.f64c8c0000000p-4, -0x1.a76d360000000p-2, 0x1.491e100000000p-3, 0x1.be67f40000000p-3, 0x1.102a360000000p-2, 0x1.9c64540000000p-5, 0x1.47c2760000000p-2, 0x1.3ebf780000000p-3, 0x1.5cb3f80000000p-4, -0x1.e2dd1e0000000p-2, -0x1.646c580000000p+0, -0x1.7e3b360000000p-6}
}
, {{0x1.6083800000000p-3, 0x1.4e6a0c0000000p-1, 0x1.d219ca0000000p-2, 0x1.1dd6340000000p-1, 0x1.4168420000000p-4, 0x1.563d0c0000000p-5, -0x1.01b59a0000000p-3, 0x1.d2b4e60000000p-4, 0x1.33f69a0000000p-3, 0x1.0c62bc0000000p-3, 0x1.62c3a40000000p-2, 0x1.3e06dc0000000p-3, -0x1.67a2ac0000000p-3, 0x1.43c1c40000000p-5, -0x1.a214320000000p-6, -0x1.182d560000000p-3, -0x1.38d0960000000p-1, -0x1.20565e0000000p-1, 0x1.04458c0000000p-4, 0x1.a110380000000p-3, -0x1.0fb9760000000p-1, -0x1.52bd120000000p-1, 0x1.8993120000000p-2, 0x1.016e540000000p-3, 0x1.8bc84c0000000p-3, 0x1.5b48100000000p-6, 0x1.a6809e0000000p-2, 0x1.a83fe20000000p-4, 0x1.529cfa0000000p-2, 0x1.2a22dc0000000p+0, 0x1.cb4cae0000000p-4, -0x1.d8378a0000000p-1}
, {-0x1.476d680000000p-3, -0x1.9271f60000000p-2, -0x1.e08c700000000p-3, 0x1.77d4960000000p-2, 0x1.1bb8c00000000p-6, -0x1.24197e0000000p-7, -0x1.3a74b20000000p-3, -0x1.08df460000000p-5, -0x1.2d3ce80000000p-2, -0x1.ff16520000000p-5, -0x1.dafdc20000000p-2, -0x1.eb31e60000000p-9, -0x1.b528180000000p-4, 0x1.c9364a0000000p-4, -0x1.5690fe0000000p-2, -0x1.6894a40000000p-4, -0x1.68fdf20000000p+0, -0x1.7485ee0000000p-4, 0x1.eaf3ee0000000p-4, 0x1.7a6a440000000p-6, 0x1.2994680000000p-2, -0x1.85f1de0000000p+1, 0x1.589af80000000p-4, 0x1.a82be40000000p-4, -0x1.0d979c0000000p-3, 0x1.1d0d3c0000000p-3, -0x1.652eba0000000p-3, 0x1.acbde40000000p-3, 0x1.b2e4780000000p-6, -0x1.688a460000000p+0, -0x1.5096c80000000p+1, -0x1.870ffa0000000p-1}
}
, {{-0x1.27e38e0000000p-5, -0x1.3f4d360000000p-3, -0x1.9189ae0000000p-4, -0x1.9ee8d40000000p-5, -0x1.26ef420000000p-3, 0x1.bf08d40000000p-3, -0x1.5779540000000p-4, -0x1.a65f300000000p-3, -0x1.fed59a0000000p-8, 0x1.2521c80000000p-4, 0x1.b8d0840000000p-4, -0x1.a4fb4c0000000p-3, -0x1.049a7e0000000p-7, -0x1.eafd560000000p-5, -0x1.1051d60000000p-3, 0x1.97e6420000000p-3, -0x1.4166b40000000p-2, -0x1.3276a40000000p-3, -0x1.2fa5480000000p-3, -0x1.1be5600000000p-6, -0x1.fe4be60000000p-4, -0x1.8b95620000000p-9, -0x1.79908e0000000p-3, -0x1.5c69360000000p-3, -0x1.72ce9c0000000p-2, -0x1.1538820000000p-5, -0x1.504d120000000p-3, 0x1.2b1ee00000000p-5, -0x1.7ebc260000000p-3, -0x1.e211dc0000000p-3, -0x1.1e86c20000000p-5, -0x1.02b2980000000p-2}
, {0x1.3870a40000000p-4, -0x1.8f8eb80000000p-3, -0x1.c8f58a0000000p-5, -0x1.1d93ac0000000p-2, 0x1.e2777a0000000p-4, -0x1.cf1a6a0000000p-4, -0x1.3231880000000p-4, -0x1.b561280000000p-6, 0x1.39bad80000000p-7, -0x1.692ee60000000p-4, -0x1.11798a0000000p-2, -0x1.22e8160000000p-3, 0x1.1fa0c20000000p-3, -0x1.14d4a00000000p-4, -0x1.5ddd2e0000000p-3, 0x1.ca51c20000000p-3, -0x1.5eb1a40000000p-2, -0x1.0c6af80000000p-4, 0x1.20baea0000000p-3, -0x1.52be3a0000000p-2, -0x1.1cd6b40000000p-3, -0x1.caba5c0000000p-3, 0x1.2da0ea0000000p-9, 0x1.f8cb240000000p-4, -0x1.c3f4c80000000p-3, 0x1.1b9ca40000000p-4, -0x1.eb6c2e0000000p-3, -0x1.0c9da20000000p-5, -0x1.afd3ca0000000p-4, -0x1.7a5a8a0000000p-3, -0x1.45286c0000000p-3, -0x1.4f19220000000p-3}
}
, {{-0x1.4f556c0000000p-7, -0x1.cb72060000000p-4, -0x1.89ba9e0000000p-3, -0x1.81b3900000000p-8, -0x1.4d639e0000000p-3, -0x1.1c86b80000000p-3, 0x1.2957520000000p-7, 0x1.de08940000000p-4, 0x1.3ea1060000000p-4, -0x1.1ff7040000000p-6, 0x1.a737620000000p-5, -0x1.4de4380000000p-3, -0x1.7b98b40000000p-4, -0x1.42e9340000000p-3, 0x1.3a71560000000p-7, 0x1.7ee5dc0000000p-3, -0x1.b56ff40000000p-4, 0x1.4c7cf20000000p-4, 0x1.95b7fa0000000p-5, -0x1.1dc1300000000p-3, 0x1.05c8ac0000000p-3, -0x1.2ea2560000000p-2, -0x1.4b3e720000000p-3, 0x1.f6af980000000p-5, -0x1.8e07880000000p-5, 0x1.bdbed00000000p-3, -0x1.30c6180000000p-3, -0x1.fa78f00000000p-3, 0x1.120ece0000000p-3, -0x1.4ad0c60000000p-3, 0x1.7f83480000000p-4, -0x1.33785a0000000p-3}
, {-0x1.a7c71e0000000p-3, -0x1.3e70c20000000p-3, 0x1.42fc840000000p-8, -0x1.70b4e40000000p-3, -0x1.a88a160000000p-6, 0x1.78055e0000000p-4, 0x1.84bb240000000p-5, 0x1.f703500000000p-4, 0x1.4b7d4a0000000p-3, -0x1.501bb00000000p-4, 0x1.82e39e0000000p-4, 0x1.73e3e80000000p-3, -0x1.00800e0000000p-2, -0x1.76445a0000000p-3, 0x1.4c5e200000000p-5, -0x1.82defc0000000p-3, -0x1.bb007e0000000p-3, 0x1.e986520000000p-6, 0x1.baddda0000000p-2, -0x1.2c24ac0000000p-3, -0x1.3f943e0000000p-3, 0x1.0f058a0000000p-4, 0x1.821e4a0000000p-4, -0x1.2e123e0000000p-3, 0x1.2d81f60000000p-4, -0x1.391cca0000000p-4, -0x1.e091060000000p-5, -0x1.1b01a60000000p-3, -0x1.b6ccbe0000000p-4, 0x1.6d07a00000000p-3, -0x1.9719d60000000p-3, -0x1.724fe00000000p-2}
}
, {{-0x1.d6f1ce0000000p-3, -0x1.3683c60000000p+0, -0x1.30ec5a0000000p-4, -0x1.abf0fe0000000p-4, -0x1.0692660000000p-6, -0x1.cd5e7a0000000p-3, 0x1.19e1d60000000p-5, -0x1.63c72a0000000p-5, -0x1.ede2560000000p-5, -0x1.8598840000000p-5, -0x1.87d3440000000p-1, -0x1.e18d740000000p-3, 0x1.83bd080000000p-8, 0x1.c3560c0000000p-4, 0x1.4a37ba0000000p-3, -0x1.20fbfc0000000p-3, -0x1.6fc5120000000p-2, 0x1.19ca720000000p-2, -0x1.96f39e0000000p-5, -0x1.a005c60000000p-3, 0x1.9b58660000000p-2, -0x1.7643d00000000p-2, -0x1.2d22d40000000p-3, -0x1.eb87220000000p-4, -0x1.65546c0000000p-2, 0x1.0267680000000p-2, -0x1.2e4c880000000p-2, -0x1.33c8960000000p-4, 0x1.2c48b00000000p-2, -0x1.8a121e0000000p-3, -0x1.715e200000000p-1, -0x1.415dcc0000000p-1}
, {0x1.b88ec00000000p-6, 0x1.5607b40000000p-1, -0x1.6461480000000p+0, 0x1.f2a6620000000p-4, 0x1.5df2800000000p-4, 0x1.3fb9f00000000p-8, -0x1.9573180000000p-4, -0x1.39b6800000000p-3, -0x1.abca3c0000000p-7, -0x1.1fb5120000000p-3, -0x1.0d730c0000000p+1, 0x1.4124500000000p-3, -0x1.4120b80000000p-3, -0x1.cf0c220000000p-6, -0x1.f24b900000000p-8, 0x1.1719280000000p-6, -0x1.28afba0000000p+0, 0x1.63c96c0000000p-3, 0x1.45284c0000000p-8, 0x1.08a9880000000p-1, 0x1.a4c8600000000p-4, -0x1.77812e0000000p+0, 0x1.6ce35e0000000p-3, 0x1.9c26c80000000p-5, 0x1.0a064c0000000p-2, 0x1.320df60000000p-5, -0x1.6b74c60000000p+0, -0x1.8d0eb80000000p-7, -0x1.07e9980000000p+1, 0x1.b60c720000000p-1, 0x1.28191a0000000p-3, 0x1.2cb4040000000p-1}
}
, {{0x1.83edae0000000p-3, -0x1.22f3fc0000000p+0, -0x1.5ee21c0000000p-5, -0x1.2d0ad20000000p-2, 0x1.265e2c0000000p-3, 0x1.2400f20000000p-3, 0x1.ebe28a0000000p-8, 0x1.56ab760000000p-3, 0x1.31e50c0000000p-3, -0x1.9a071a0000000p-4, -0x1.ad03ac0000000p-4, -0x1.213bdc0000000p-3, 0x1.afc2bc0000000p-4, -0x1.aecc560000000p-3, 0x1.b6b0d60000000p-3, -0x1.7fb9f80000000p-4, -0x1.41c8540000000p-3, 0x1.c9ef100000000p-1, 0x1.49eeae0000000p-3, -0x1.6367b40000000p-2, 0x1.6e64420000000p-2, 0x1.1045ea0000000p-3, -0x1.44ba760000000p-5, 0x1.8beeb00000000p-5, 0x1.5c2a2c0000000p-4, -0x1.6a2eda0000000p-6, 0x1.b86c7c0000000p-3, 0x1.3519560000000p-2, 0x1.50ce820000000p-2, -0x1.409f660000000p-1, -0x1.ae66b40000000p-4, -0x1.c79acc0000000p-1}
, {-0x1.3c07d40000000p-5, -0x1.b68eb80000000p-1, -0x1.844ab80000000p-6, -0x1.4638aa0000000p-2, 0x1.933bce0000000p-4, -0x1.77f1f40000000p-6, 0x1.d840840000000p-3, -0x1.038ac80000000p-7, -0x1.17ca7e0000000p-3, 0x1.0256580000000p-2, -0x1.577d520000000p-1, 0x1.de899a0000000p-6, 0x1.f3ed2a0000000p-4, 0x1.b702a20000000p-4, -0x1.6f185c0000000p-1, 0x1.0112720000000p-3, 0x1.33bf8e0000000p-4, -0x1.21a6b60000000p-2, -0x1.4c8ed40000000p-6, -0x1.9187080000000p-1, -0x1.8ffa580000000p-2, 0x1.507b860000000p-2, -0x1.54fe0a0000000p-4, 0x1.74d4020000000p-6, -0x1.33b1240000000p-3, 0x1.3fa94e0000000p-2, 0x1.396c4c0000000p-3, -0x1.c600fc0000000p-4, 0x1.fff73c0000000p-3, -0x1.7ed3e40000000p-1, 0x1.4c3eb40000000p-3, -0x1.f65d960000000p-1}
}
, {{-0x1.eb66900000000p-3, -0x1.104c5c0000000p-3, -0x1.72af900000000p-6, 0x1.5cf32e0000000p-4, -0x1.0d1e7c0000000p-3, -0x1.3b978e0000000p-3, 0x1.dd04cc0000000p-6, 0x1.a46c320000000p-4, 0x1.a484160000000p-9, -0x1.7df7d20000000p-4, -0x1.1cd4aa0000000p-4, -0x1.04edb60000000p-2, 0x1.396ade0000000p-6, -0x1.8646800000000p-3, -0x1.4815360000000p-3, 0x1.6cf7820000000p-3, -0x1.fd07d40000000p-4, -0x1.d40e6a0000000p-2, -0x1.e241780000000p-4, -0x1.1fcbe00000000p-1, 0x1.1d23960000000p-2, -0x1.1de7980000000p-2, -0x1.9fcca20000000p-3, -0x1.f484c00000000p-5, -0x1.13fb120000000p-2, -0x1.8763d40000000p-4, -0x1.081b560000000p-3, -0x1.8a33420000000p-4, -0x1.2704a80000000p-4, 0x1.76c2200000000p-3, 0x1.afa26e0000000p-3, -0x1.02c8de0000000p-2}
, {0x1.bf699c0000000p-4, -0x1.042dc80000000p-4, -0x1.355a9c0000000p-3, -0x1.f7add60000000p-3, 0x1.f9a0960000000p-10, 0x1.cca6b80000000p-4, -0x1.0ad8ce0000000p-3, 0x1.e451c40000000p-4, 0x1.d33e9c0000000p-3, 0x1.d071680000000p-4, -0x1.72725a0000000p-6, -0x1.92124c0000000p-10, -0x1.32225a0000000p-3, 0x1.15349c0000000p-6, -0x1.91661c0000000p-2, -0x1.5e56580000000p-3, -0x1.a0b0780000000p-5, -0x1.5bb4b20000000p-3, -0x1.467b940000000p-6, -0x1.5d0dac0000000p-4, -0x1.e0507a0000000p-2, -0x1.95fd460000000p-4, 0x1.09bf900000000p-2, 0x1.deda480000000p-4, -0x1.901ef00000000p-4, 0x1.30788a0000000p-6, -0x1.9c5c560000000p-3, 0x1.45ae140000000p-4, -0x1.0ac74a0000000p-3, 0x1.434c760000000p-7, 0x1.07ae6a0000000p-2, -0x1.ccc7420000000p-2}
}
, {{-0x1.60433a0000000p-3, -0x1.2ba98e0000000p-6, -0x1.4bedf80000000p-3, -0x1.d088320000000p-6, -0x1.65481e0000000p-4, 0x1.61a3780000000p-11, -0x1.dd31740000000p-5, -0x1.0946400000000p-3, -0x1.3074b20000000p-5, -0x1.5a6e640000000p-3, 0x1.2e51340000000p-5, 0x1.16874c0000000p-3, -0x1.d1e6520000000p-4, 0x1.9a5fac0000000p-4, -0x1.8c10500000000p-6, 0x1.5a15640000000p-5, -0x1.5255500000000p-2, -0x1.77e9480000000p-3, -0x1.ca4d540000000p-6, -0x1.ae887a0000000p-4, -0x1.d6b7280000000p-5, -0x1.2e4e680000000p-2, 0x1.a0f3e00000000p-5, 0x1.e69e280000000p-4, -0x1.40ff5e0000000p-2, 0x1.1b62560000000p-3, -0x1.3f67440000000p-3, -0x1.9aa4e60000000p-5, -0x1.28217c0000000p-4, -0x1.3c73460000000p-3, 0x1.2b45520000000p-3, -0x1.059b3e0000000p-2}
, {-0x1.79789e0000000p-4, -0x1.7983960000000p-2, -0x1.9673b60000000p-3, -0x1.c151d80000000p-3, 0x1.7cd32e0000000p-9, 0x1.d9164a0000000p-7, -0x1.38a4940000000p-3, -0x1.586b680000000p-3, -0x1.238d7a0000000p-4, 0x1.0b87040000000p-4, 0x1.5856360000000p-3, 0x1.528b6a0000000p-3, 0x1.0720e40000000p-7, -0x1.8f46600000000p-8, 0x1.5ae4940000000p-4, 0x1.8d1dce0000000p-4, 0x1.b9c74a0000000p-7, -0x1.c750ea0000000p-3, 0x1.5df8240000000p-8, -0x1.7a5f0c0000000p-6, 0x1.d3140a0000000p-5, 0x1.952e4c0000000p-9, 0x1.909b360000000p-4, 0x1.2df39c0000000p-3, 0x1.6cd4de0000000p-4, 0x1.2c70de0000000p-3, -0x1.1063ca0000000p-5, -0x1.1c99c40000000p-4, -0x1.eefd740000000p-5, 0x1.3781e20000000p-9, 0x1.657e7e0000000p-3, -0x1.e0959c0000000p-3}
}
, {{0x1.60be120000000p-6, -0x1.0f71ce0000000p-6, 0x1.ceb7f00000000p-3, -0x1.078b220000000p-1, -0x1.d4d45a0000000p-5, 0x1.6b94420000000p-3, 0x1.7314c80000000p-4, 0x1.6eee1c0000000p-3, -0x1.e855cc0000000p-3, 0x1.f6a77e0000000p-5, -0x1.61c5900000000p-2, 0x1.aef4900000000p-4, -0x1.2d2ed60000000p-6, 0x1.f9751e0000000p-9, -0x1.58ee8c0000000p-2, -0x1.76c90a0000000p-10, 0x1.17d11c0000000p-2, 0x1.bf25e40000000p-3, -0x1.b78fee0000000p-4, 0x1.ffd59c0000000p-3, -0x1.6a2e380000000p-3, -0x1.434a5e0000000p-1, -0x1.44d7560000000p-4, -0x1.644aae0000000p-3, 0x1.6d1f220000000p-4, -0x1.48e89c0000000p-2, 0x1.d208240000000p-6, 0x1.a6120a0000000p-3, 0x1.731f0c0000000p-3, -0x1.828a240000000p-2, -0x1.5e77280000000p+0, 0x1.b14e7e0000000p-2}
, {-0x1.e83b500000000p-4, -0x1.4d77140000000p-5, 0x1.3cd54c0000000p-2, -0x1.9187020000000p-2, -0x1.c9820c0000000p-4, -0x1.75ae940000000p-7, 0x1.181be40000000p-4, 0x1.09e19e0000000p-3, 0x1.6dea1c0000000p-5, -0x1.5898bc0000000p-4, -0x1.18ed440000000p-3, -0x1.3073220000000p-4, 0x1.62de1e0000000p-4, 0x1.043ac20000000p-3, -0x1.9160940000000p-2, 0x1.b2fd100000000p-4, 0x1.5a8c800000000p-2, 0x1.0000160000000p-3, 0x1.a2cbde0000000p-3, 0x1.6bb04a0000000p-4, -0x1.b38c780000000p-5, -0x1.89a0a20000000p-2, -0x1.6216460000000p-3, 0x1.a32fc40000000p-3, 0x1.4c606a0000000p-2, -0x1.b0d0920000000p-6, 0x1.c95bf40000000p-3, 0x1.34b0be0000000p-4, 0x1.1a0f720000000p-2, 0x1.47dcb40000000p-2, -0x1.5bd82c0000000p-1, 0x1.69d28e0000000p-2}
}
, {{-0x1.06da220000000p-2, -0x1.9b51ec0000000p-3, 0x1.baf1cc0000000p-4, 0x1.d5ccfa0000000p-5, -0x1.83f8480000000p-3, 0x1.cad25e0000000p-5, -0x1.a369140000000p-5, -0x1.64ec360000000p-4, 0x1.1923620000000p-2, 0x1.e9b9180000000p-4, -0x1.6920a40000000p-3, -0x1.5a44020000000p-3, -0x1.78177c0000000p-6, -0x1.9869d80000000p-3, -0x1.74ccec0000000p-4, 0x1.2900ca0000000p-5, -0x1.8979d80000000p-4, 0x1.8d98960000000p-4, 0x1.20c87c0000000p-6, -0x1.1c6c3c0000000p-3, -0x1.bf00720000000p-5, -0x1.d194b60000000p-3, -0x1.97b70a0000000p-5, -0x1.59fd080000000p-4, -0x1.13de140000000p-2, 0x1.214e100000000p-3, -0x1.b68e800000000p-4, 0x1.1492d00000000p-3, 0x1.05bec00000000p-3, -0x1.fb8c360000000p-4, 0x1.94b74c0000000p-5, 0x1.35917a0000000p-5}
, {0x1.c90a500000000p-4, -0x1.dab4820000000p-3, -0x1.3135ee0000000p-4, 0x1.0b69d80000000p-4, -0x1.479f1e0000000p-4, -0x1.cb05e80000000p-8, -0x1.571c440000000p-4, -0x1.fd88460000000p-5, -0x1.0ed4380000000p-3, -0x1.4520160000000p-6, -0x1.99511c0000000p-7, -0x1.485df40000000p-4, 0x1.0fed3c0000000p-4, -0x1.57c7920000000p-3, -0x1.9306be0000000p-11, -0x1.862f900000000p-3, -0x1.9c10aa0000000p-4, 0x1.cc0f500000000p-4, 0x1.c1df1a0000000p-4, -0x1.24a44c0000000p-6, 0x1.8d33880000000p-6, -0x1.7a95160000000p-5, -0x1.a84a320000000p-3, 0x1.4bd1e60000000p-4, -0x1.cc650e0000000p-3, -0x1.0590fe0000000p-3, 0x1.6719660000000p-7, 0x1.370d5e0000000p-7, -0x1.5e60040000000p-3, -0x1.646ea40000000p-5, -0x1.712a340000000p-4, -0x1.fcc0ee0000000p-3}
}
, {{-0x1.21b76e0000000p-2, 0x1.fcbc9a0000000p-4, -0x1.0b92cc0000000p-4, 0x1.58a9f80000000p-4, -0x1.f556080000000p-4, -0x1.6f74b40000000p-6, -0x1.645b520000000p-4, 0x1.ad6e020000000p-3, -0x1.57841a0000000p-3, -0x1.92a2940000000p-4, -0x1.c2e6520000000p-3, -0x1.ea416e0000000p-4, -0x1.a2a37a0000000p-4, -0x1.315ae40000000p-3, 0x1.250d9a0000000p-5, 0x1.2ca9600000000p-3, -0x1.b76bfa0000000p-4, -0x1.e82b7a0000000p-4, 0x1.80a8e00000000p-3, -0x1.1a56c80000000p-3, -0x1.883bb80000000p-4, -0x1.ce4b700000000p-3, -0x1.d577e00000000p-5, -0x1.0c064a0000000p-3, -0x1.a881e40000000p-4, -0x1.1a709c0000000p-3, -0x1.f650a40000000p-3, -0x1.f982860000000p-11, -0x1.78ac840000000p-3, 0x1.9abaae0000000p-4, -0x1.e178520000000p-6, -0x1.c8f8a60000000p-3}
, {0x1.1dcfd60000000p-3, -0x1.f6a3060000000p-6, -0x1.98a81c0000000p-4, 0x1.3b2d060000000p-3, 0x1.4b68e60000000p-4, -0x1.1cbc240000000p-4, -0x1.b7c64e0000000p-10, 0x1.9ed3140000000p-3, 0x1.2ac3800000000p-3, 0x1.55952a0000000p-6, -0x1.62b15e0000000p-3, 0x1.7650d80000000p-4, 0x1.623e960000000p-3, -0x1.e750520000000p-3, -0x1.c2d4120000000p-7, 0x1.ba95e40000000p-6, -0x1.9c81800000000p-5, 0x1.03c83a0000000p-3, -0x1.a0a51a0000000p-6, 0x1.9649540000000p-7, 0x1.b490040000000p-6, -0x1.17d6ce0000000p-4, -0x1.ad747e0000000p-6, -0x1.9d0f620000000p-5, -0x1.39649a0000000p-2, 0x1.57498c0000000p-4, -0x1.127c840000000p-5, -0x1.1e51500000000p-2, 0x1.015a400000000p-5, -0x1.0010760000000p-2, -0x1.89b9380000000p-5, -0x1.a89afa0000000p-3}
}
, {{-0x1.c2bea40000000p-4, -0x1.b710c40000000p-2, 0x1.15f6c60000000p-2, -0x1.a2e5f60000000p-2, 0x1.5bccf20000000p-3, 0x1.97a8d40000000p-4, -0x1.0e21c00000000p-4, 0x1.3e444a0000000p-3, 0x1.5da1c40000000p-6, 0x1.1888360000000p-5, -0x1.09d9ae0000000p-2, 0x1.50f0fc0000000p-3, -0x1.d7ef420000000p-8, 0x1.54f73c0000000p-3, -0x1.54f0c80000000p-3, -0x1.04932c0000000p-3, -0x1.3d9dfc0000000p-3, 0x1.7a6bd20000000p-2, 0x1.585d920000000p-3, -0x1.1d31c20000000p-3, 0x1.6e626a0000000p-2, 0x1.f2019a0000000p-2, -0x1.2decfa0000000p-9, -0x1.13aba00000000p-5, -0x1.78e4da0000000p-4, 0x1.36238e0000000p-3, 0x1.19099a0000000p-2, 0x1.1bffca0000000p-8, 0x1.049e100000000p-2, 0x1.cf603e0000000p-1, 0x1.5da2780000000p-3, -0x1.f7af040000000p-4}
, {0x1.497e4e0000000p-4, -0x1.95d7340000000p-2, 0x1.2648ca0000000p-3, -0x1.03a7840000000p-3, 0x1.1203640000000p-6, -0x1.9a63780000000p-5, -0x1.7e3b5c0000000p-5, -0x1.8d98d60000000p-4, -0x1.083a800000000p-3, 0x1.19baf80000000p-7, -0x1.fddcf40000000p-4, 0x1.0664c40000000p-2, 0x1.441f300000000p-3, -0x1.fb82e00000000p-4, 0x1.0e79420000000p-2, -0x1.47da300000000p-7, -0x1.06ea8e0000000p-2, 0x1.063cde0000000p-1, 0x1.0aed260000000p-10, -0x1.303de40000000p-3, -0x1.1e95d60000000p-1, -0x1.a730c20000000p-3, 0x1.22c4120000000p-3, 0x1.7971600000000p-3, 0x1.eb400a0000000p-4, 0x1.9e105a0000000p-3, 0x1.9cb4820000000p-4, 0x1.ae7cca0000000p-5, 0x1.52fc360000000p-2, -0x1.028ba80000000p-1, -0x1.6dcf300000000p-1, 0x1.4b2ab60000000p-3}
}
, {{-0x1.1c81f60000000p-3, -0x1.088b100000000p-3, -0x1.8cb9480000000p-3, 0x1.4f75f60000000p-4, -0x1.9146e40000000p-4, -0x1.170a100000000p-6, -0x1.0f0fd00000000p-6, -0x1.c42fd60000000p-6, 0x1.4ee1a40000000p-3, 0x1.97b9ce0000000p-3, -0x1.706bb40000000p-4, -0x1.dbfcca0000000p-3, 0x1.76e2240000000p-3, 0x1.1465ac0000000p-3, -0x1.f9e77a0000000p-3, -0x1.03ab380000000p-3, -0x1.9e98a20000000p-3, 0x1.da14be0000000p-4, 0x1.1da22e0000000p-3, -0x1.f46e7e0000000p-3, 0x1.5679e40000000p-5, -0x1.568e980000000p-5, -0x1.1afe260000000p-4, 0x1.3014fc0000000p-3, -0x1.17420a0000000p-2, 0x1.ac416e0000000p-13, 0x1.b5da8e0000000p-4, -0x1.4e5c9e0000000p-4, 0x1.5321080000000p-3, 0x1.3104320000000p-4, -0x1.b9715e0000000p-4, -0x1.4c59360000000p-3}
, {-0x1.3f170a0000000p-3, -0x1.985a980000000p-4, 0x1.48740e0000000p-7, -0x1.8c96ec0000000p-4, -0x1.0ff54c0000000p-3, 0x1.72abd00000000p-5, 0x1.1490460000000p-3, -0x1.91dc400000000p-7, 0x1.7b0c360000000p-5, 0x1.e539980000000p-5, -0x1.1e5cf00000000p-2, 0x1.ac64ce0000000p-4, -0x1.1aad960000000p-3, 0x1.b7a3c00000000p-6, 0x1.5497340000000p-6, 0x1.54488a0000000p-8, -0x1.0828e00000000p-5, -0x1.100ca40000000p-3, -0x1.0b03de0000000p-4, -0x1.65aebe0000000p-3, 0x1.7a91560000000p-5, -0x1.c874b20000000p-4, 0x1.13fe0a0000000p-4, 0x1.9772de0000000p-4, -0x1.f65eaa0000000p-3, 0x1.94c0840000000p-5, -0x1.f09b820000000p-4, 0x1.46ca340000000p-3, 0x1.514d320000000p-4, 0x1.5dd00a0000000p-4, -0x1.7925ee0000000p-4, 0x1.f88eee0000000p-4}
}
, {{-0x1.ca9c120000000p-4, 0x1.3ba4f40000000p-6, -0x1.000ea40000000p-4, 0x1.1c04740000000p-4, -0x1.4a312a0000000p-3, 0x1.0cd9ae0000000p-3, 0x1.07c9bc0000000p-3, -0x1.8421e40000000p-7, -0x1.660db80000000p-4, -0x1.85823c0000000p-3, -0x1.23cf340000000p-3, 0x1.d14b380000000p-5, -0x1.c91e980000000p-4, 0x1.b5b9060000000p-4, -0x1.5b27b20000000p-4, 0x1.9872740000000p-5, -0x1.25597e0000000p-2, 0x1.d9ff6e0000000p-5, -0x1.29e3480000000p-4, -0x1.3c47b40000000p-6, -0x1.e0a8560000000p-4, 0x1.ff9af40000000p-5, -0x1.1710880000000p-4, 0x1.fa64a40000000p-6, -0x1.83d2ce0000000p-5, -0x1.dfa7a00000000p-4, 0x1.159e1a0000000p-7, 0x1.9406ec0000000p-4, 0x1.4fbf8a0000000p-6, 0x1.9328d40000000p-4, -0x1.c37afa0000000p-3, -0x1.e042440000000p-8}
, {-0x1.cbd8240000000p-4, 0x1.591a0e0000000p-4, -0x1.263eae0000000p-4, 0x1.2419ee0000000p-5, -0x1.414e8c0000000p-3, -0x1.8ecb0e0000000p-5, -0x1.6b2da40000000p-3, 0x1.93b9960000000p-7, 0x1.57633a0000000p-4, 0x1.5294700000000p-7, -0x1.d6e64c0000000p-4, -0x1.3aaf620000000p-3, 0x1.7862ba0000000p-7, 0x1.9d589e0000000p-4, 0x1.e34d6a0000000p-6, -0x1.ca35e60000000p-3, -0x1.2743560000000p-2, -0x1.d446dc0000000p-5, -0x1.3b28ac0000000p-3, -0x1.b626900000000p-3, -0x1.335c600000000p-6, -0x1.1ee1360000000p-2, -0x1.60638a0000000p-3, 0x1.c04c000000000p-4, -0x1.bbdc940000000p-3, 0x1.ef37c60000000p-3, 0x1.5664180000000p-4, 0x1.02d0f60000000p-3, -0x1.f25a2c0000000p-3, -0x1.fc53dc0000000p-3, 0x1.6d32680000000p-5, -0x1.21f7960000000p-3}
}
, {{-0x1.2d158e0000000p-4, -0x1.24edf40000000p-4, -0x1.5807420000000p-4, -0x1.b3c91a0000000p-7, -0x1.5077200000000p-4, -0x1.2f1ed00000000p-4, -0x1.eccf140000000p-3, -0x1.18799a0000000p-5, 0x1.2089900000000p-3, -0x1.e3edfa0000000p-7, -0x1.8e8a280000000p-6, 0x1.2565880000000p-4, 0x1.c1c66a0000000p-4, -0x1.6f9cda0000000p-3, -0x1.8266900000000p-6, -0x1.ae6f300000000p-5, -0x1.a09e6a0000000p-2, -0x1.7568440000000p-3, 0x1.6195b60000000p-8, -0x1.9ea80e0000000p-2, 0x1.05da5e0000000p-3, -0x1.c2e6360000000p-3, -0x1.a412060000000p-4, 0x1.f25a600000000p-5, -0x1.665ccc0000000p-4, 0x1.acc30c0000000p-4, -0x1.675d920000000p-2, -0x1.df725c0000000p-3, -0x1.3779c80000000p-3, 0x1.eed4e40000000p-5, -0x1.865a3a0000000p-3, -0x1.a2fece0000000p-3}
, {0x1.6d559c0000000p-5, -0x1.4933f60000000p-2, -0x1.08eb060000000p-2, -0x1.b29d6c0000000p-3, 0x1.0e83a00000000p-3, -0x1.7ac22a0000000p-4, -0x1.d43bb20000000p-5, -0x1.77bea60000000p-14, 0x1.4094580000000p-5, 0x1.0ddda40000000p-3, 0x1.42dda20000000p-4, 0x1.bd39820000000p-4, -0x1.8c71500000000p-7, 0x1.4cb13a0000000p-6, -0x1.f955b60000000p-3, 0x1.0976f60000000p-3, -0x1.a797780000000p-2, 0x1.40f7de0000000p-7, 0x1.d82aba0000000p-4, -0x1.be33020000000p-3, 0x1.a18a0c0000000p-6, -0x1.3a55b40000000p-2, 0x1.d412180000000p-3, 0x1.b9d79e0000000p-3, -0x1.5c65580000000p-3, 0x1.004ec20000000p-5, -0x1.78e6d80000000p-5, -0x1.097d280000000p-3, -0x1.a9e4580000000p-5, -0x1.08872a0000000p-5, -0x1.5433280000000p-4, -0x1.bb804a0000000p-6}
}
, {{0x1.982e660000000p-6, -0x1.7045880000000p-3, 0x1.39ab2c0000000p-4, -0x1.35232c0000000p-2, -0x1.43ea5c0000000p-4, -0x1.21a3f40000000p-3, -0x1.0c79980000000p-2, 0x1.3b9a480000000p-3, 0x1.dd7d2c0000000p-4, -0x1.4a40ee0000000p-2, -0x1.3260220000000p-2, 0x1.4b0f420000000p-4, -0x1.31c9080000000p-4, 0x1.068ab20000000p-3, -0x1.6a1ae20000000p-1, -0x1.c7c8800000000p-8, -0x1.a0882c0000000p-3, 0x1.38a2660000000p-4, -0x1.0eb83a0000000p-4, -0x1.2f32ac0000000p-2, -0x1.b76eb40000000p-2, -0x1.ce38bc0000000p-1, -0x1.a32a180000000p-4, -0x1.5f92cc0000000p-3, -0x1.2dfc6e0000000p-2, 0x1.0354a80000000p-2, -0x1.37ed560000000p-6, 0x1.39544c0000000p-4, 0x1.c952f40000000p-4, 0x1.7fa7c60000000p+0, 0x1.b73eb80000000p-3, -0x1.3dd6260000000p-1}
, {0x1.b45ce20000000p-5, 0x1.19a55c0000000p-1, 0x1.1e916e0000000p-4, 0x1.d44c340000000p-2, -0x1.0b289e0000000p-3, -0x1.8e79500000000p-4, -0x1.2287180000000p-4, -0x1.5ada780000000p-3, 0x1.56d5860000000p-6, 0x1.a20bf40000000p-7, 0x1.4849840000000p-2, 0x1.38b3f20000000p-3, 0x1.194fb40000000p-4, -0x1.7050200000000p-3, 0x1.1d316c0000000p-5, 0x1.1c764a0000000p-3, 0x1.d200880000000p-2, -0x1.90598c0000000p-2, 0x1.2a66c80000000p-3, 0x1.30fb4a0000000p-3, -0x1.2d74300000000p-2, -0x1.2e00aa0000000p-1, 0x1.946cba0000000p-7, 0x1.1277600000000p-3, 0x1.dfea180000000p-3, 0x1.bb56aa0000000p-5, 0x1.e46e060000000p-2, -0x1.84cda00000000p-4, 0x1.0d7d2c0000000p-1, 0x1.2e235c0000000p-1, -0x1.bfda900000000p-4, 0x1.f8a3e00000000p-3}
}
, {{-0x1.d270f00000000p-4, 0x1.7feb360000000p+0, -0x1.505cb00000000p-2, -0x1.ad41280000000p-12, -0x1.30da820000000p-3, 0x1.96fa8c0000000p-3, 0x1.9439de0000000p-4, 0x1.58de720000000p-3, -0x1.7f60a80000000p-3, 0x1.293dc80000000p-3, -0x1.8fff580000000p-3, -0x1.9a9c100000000p-4, -0x1.3af2e00000000p-3, 0x1.0ab2b60000000p-5, -0x1.9fd89c0000000p-1, -0x1.49557a0000000p-4, 0x1.3c00b40000000p+0, -0x1.e88b400000000p-2, 0x1.7bd8720000000p-3, 0x1.512dae0000000p-2, -0x1.7077e20000000p-1, 0x1.9ba6b60000000p-5, -0x1.4ffef60000000p-7, -0x1.395ecc0000000p-6, -0x1.eacd5c0000000p-3, -0x1.bf18e60000000p-4, -0x1.fa56d80000000p-4, -0x1.4afd560000000p-2, -0x1.85771a0000000p-2, 0x1.4b24b00000000p-2, 0x1.c0bd3c0000000p-2, -0x1.c056880000000p-2}
, {-0x1.fa2f680000000p-5, 0x1.b60d4e0000000p-1, -0x1.4079440000000p-3, 0x1.4c1a580000000p-1, 0x1.66171c0000000p-3, 0x1.ad21520000000p-4, 0x1.eac2160000000p-4, -0x1.03667a0000000p-3, 0x1.eaf4b00000000p-4, 0x1.6996e20000000p-3, -0x1.558e020000000p-4, -0x1.5384760000000p-3, 0x1.050e940000000p-3, -0x1.667c5c0000000p-7, -0x1.4811b80000000p-1, 0x1.6907120000000p-5, 0x1.0a32960000000p+0, -0x1.186a180000000p+0, -0x1.117a200000000p-3, 0x1.f2df620000000p-4, -0x1.aaab480000000p-3, 0x1.6795ea0000000p-1, 0x1.e99c840000000p-3, -0x1.13dd9c0000000p-4, 0x1.2006460000000p-3, 0x1.b0b7ea0000000p-4, 0x1.690dfa0000000p-3, -0x1.5692400000000p-5, 0x1.1332d00000000p-2, 0x1.58ce6e0000000p-3, 0x1.e932f20000000p-3, 0x1.ef14640000000p-2}
}
, {{0x1.9642cc0000000p-4, -0x1.8f73cc0000000p-2, -0x1.075ece0000000p-5, -0x1.8781760000000p-3, -0x1.0d82a60000000p-4, -0x1.929b4e0000000p-3, -0x1.241bb40000000p-7, 0x1.2f57d60000000p-3, -0x1.71d0880000000p-3, -0x1.c7c9700000000p-4, -0x1.4009920000000p-4, 0x1.65bf060000000p-9, 0x1.8c9f960000000p-3, -0x1.a6881e0000000p-3, -0x1.8849480000000p-3, 0x1.519c4e0000000p-3, -0x1.3ff9440000000p-3, -0x1.0cfa980000000p-3, 0x1.8086a20000000p-3, -0x1.8cae0c0000000p-4, -0x1.37c23a0000000p-3, -0x1.3547300000000p-2, -0x1.72e7a20000000p-4, 0x1.5419360000000p-3, -0x1.01ff420000000p-3, 0x1.29ce320000000p-3, -0x1.3c6e300000000p-4, -0x1.c551bc0000000p-4, -0x1.417b160000000p-1, 0x1.b073f00000000p-5, -0x1.8faa120000000p-3, -0x1.7368f60000000p-2}
, {-0x1.22d43a0000000p-2, 0x1.b09f860000000p-5, -0x1.78818a0000000p-3, -0x1.b6e8260000000p-4, -0x1.5d46b00000000p-6, -0x1.4d7d660000000p-3, 0x1.4d786c0000000p-4, -0x1.1b3b800000000p-4, 0x1.278c780000000p-4, -0x1.9bb8620000000p-5, -0x1.76adde0000000p-2, 0x1.4c5cb20000000p-4, -0x1.65eed80000000p-4, 0x1.32f28c0000000p-4, -0x1.b71a9c0000000p-3, -0x1.b8c2cc0000000p-4, 0x1.17527c0000000p-5, 0x1.94a87a0000000p-5, 0x1.cbfb340000000p-4, -0x1.aecf320000000p-4, -0x1.f174c20000000p-4, -0x1.6483e80000000p-6, -0x1.10c4a00000000p-6, -0x1.bb9c1c0000000p-3, -0x1.262f100000000p-2, 0x1.c6e7220000000p-3, -0x1.f81f860000000p-5, -0x1.d67b400000000p-4, -0x1.31f8500000000p-2, 0x1.9eac280000000p-4, -0x1.d426400000000p-3, -0x1.92fb880000000p-2}
}
, {{0x1.af76c20000000p-4, 0x1.a6e8540000000p-4, -0x1.3f03400000000p-2, -0x1.bfa1f40000000p-4, -0x1.aea2ac0000000p-3, 0x1.5a83ac0000000p-5, 0x1.9e215a0000000p-4, 0x1.6dbe100000000p-3, 0x1.4945140000000p-5, 0x1.1425400000000p-3, -0x1.5a5dbe0000000p-3, 0x1.3171fe0000000p-3, -0x1.f06e180000000p-7, -0x1.5882240000000p-3, 0x1.22b2960000000p-3, -0x1.cdb8920000000p-3, 0x1.f7ad7e0000000p-5, -0x1.f5040a0000000p-7, -0x1.1c3c920000000p-3, -0x1.9a36280000000p-5, -0x1.90b7aa0000000p-3, -0x1.98f4540000000p-3, -0x1.0ff0420000000p-3, -0x1.c3f0540000000p-4, -0x1.2a7b060000000p-3, -0x1.136e700000000p-6, -0x1.1881fc0000000p-2, -0x1.44d30a0000000p-4, -0x1.58f3c00000000p-5, -0x1.cf70320000000p-5, -0x1.83a2060000000p-3, 0x1.3343160000000p-4}
, {-0x1.14497a0000000p-3, 0x1.c343fe0000000p-4, -0x1.565ad40000000p-3, 0x1.4db4560000000p-3, -0x1.44fbde0000000p-3, 0x1.a937fc0000000p-5, 0x1.ad8e180000000p-4, 0x1.01ad220000000p-3, -0x1.ca02220000000p-4, -0x1.f99be80000000p-3, -0x1.1954040000000p-3, -0x1.1e768e0000000p-3, 0x1.10b9140000000p-3, 0x1.d2d5e40000000p-4, -0x1.13c78c0000000p-3, -0x1.a0d6920000000p-4, 0x1.248d220000000p-4, -0x1.0b559a0000000p-4, -0x1.2331980000000p-3, 0x1.d558a40000000p-7, -0x1.42dc2e0000000p-9, -0x1.3588720000000p-2, -0x1.ccae540000000p-5, -0x1.3e2b000000000p-3, 0x1.ef95b80000000p-8, -0x1.cba89e0000000p-4, -0x1.60d7960000000p-3, -0x1.40d5320000000p-4, -0x1.ca3a720000000p-5, 0x1.88b7840000000p-5, 0x1.21655a0000000p-6, -0x1.eddcc80000000p-3}
}
, {{-0x1.74c11a0000000p-3, -0x1.abaf7a0000000p-1, -0x1.4767880000000p-1, 0x1.ee81e00000000p-3, -0x1.6285400000000p-3, 0x1.93b03a0000000p-7, -0x1.ea4a300000000p-3, 0x1.a39c480000000p-3, 0x1.6408080000000p-5, -0x1.771b300000000p-3, -0x1.0298340000000p+0, -0x1.3dd51a0000000p-3, -0x1.531e680000000p-3, -0x1.8be2420000000p-4, -0x1.290ce80000000p-4, 0x1.2ed6660000000p-3, -0x1.e736220000000p-1, -0x1.a2460e0000000p-5, 0x1.8d5af00000000p-3, 0x1.1e77940000000p-1, 0x1.2d12ae0000000p+0, -0x1.fcaa440000000p-1, 0x1.9769480000000p-5, 0x1.4c84180000000p-3, -0x1.d20de60000000p-2, 0x1.1c3ce80000000p-5, -0x1.4ac9100000000p-1, -0x1.8e16800000000p-3, -0x1.1691680000000p-1, 0x1.b334680000000p-5, 0x1.31e33a0000000p-3, 0x1.a97a880000000p-2}
, {-0x1.4937cc0000000p-5, -0x1.9899120000000p-2, -0x1.7e6e2a0000000p-1, -0x1.15efc00000000p-1, 0x1.2b6adc0000000p-4, -0x1.46b4780000000p-4, -0x1.0a3c0e0000000p-3, -0x1.14a6e40000000p-3, 0x1.6120780000000p-2, 0x1.f0ef680000000p-7, -0x1.8806760000000p-1, 0x1.3d160a0000000p-4, -0x1.993f1e0000000p-4, -0x1.02e5f00000000p-2, 0x1.f362ec0000000p-5, -0x1.43b8ce0000000p-3, -0x1.01b08e0000000p-1, 0x1.ea2ed00000000p-1, -0x1.1b705e0000000p-4, -0x1.b162da0000000p-2, -0x1.5fc7f00000000p-1, -0x1.c211040000000p-3, -0x1.540e260000000p-3, -0x1.015cd20000000p-3, -0x1.48357a0000000p-1, -0x1.7a1ffa0000000p-3, 0x1.2cff0c0000000p-3, 0x1.46c2640000000p-4, -0x1.12ec080000000p-3, 0x1.b32f8c0000000p-1, 0x1.33c9ec0000000p-1, -0x1.9573340000000p-2}
}
, {{-0x1.e21b460000000p-3, -0x1.0b4cbc0000000p-2, 0x1.c74c1c0000000p-5, 0x1.84d5260000000p-4, -0x1.dea3780000000p-5, -0x1.9fae200000000p-3, 0x1.510bf20000000p-4, -0x1.8e1da00000000p-4, -0x1.970a9e0000000p-4, -0x1.7961b00000000p-4, 0x1.2df7ca0000000p-4, 0x1.3e133c0000000p-3, 0x1.7080060000000p-3, -0x1.7def260000000p-5, -0x1.ffb1e00000000p-7, 0x1.1b53c00000000p-3, 0x1.f922760000000p-4, 0x1.a7bf7a0000000p-5, 0x1.5d1cca0000000p-6, -0x1.c97a6c0000000p-5, -0x1.9b9c440000000p-4, -0x1.28651e0000000p-2, -0x1.c4d5c80000000p-6, 0x1.0732360000000p-4, -0x1.4fa3fc0000000p-4, 0x1.604c480000000p-4, -0x1.3ab2b60000000p-3, -0x1.9d1c340000000p-4, -0x1.9f99960000000p-3, -0x1.2151940000000p-2, 0x1.8c4c380000000p-4, -0x1.69b3f20000000p-3}
, {-0x1.1610fc0000000p-3, -0x1.de19160000000p-5, -0x1.8f60f00000000p-6, -0x1.bee2900000000p-3, -0x1.dea43c0000000p-5, 0x1.056d060000000p-3, 0x1.6445ca0000000p-3, 0x1.f172240000000p-6, -0x1.b4db960000000p-6, -0x1.3d1fd80000000p-3, -0x1.ab23f00000000p-4, 0x1.2ea1680000000p-6, 0x1.8fe7ba0000000p-3, 0x1.17fddc0000000p-3, 0x1.2a846e0000000p-4, 0x1.06d2840000000p-3, 0x1.c38f440000000p-5, -0x1.d12ed20000000p-6, -0x1.3f1a1e0000000p-7, -0x1.e4db7a0000000p-3, -0x1.94b8540000000p-3, 0x1.a396220000000p-9, 0x1.e82d7e0000000p-4, 0x1.4ad2c60000000p-3, -0x1.48058c0000000p-2, -0x1.54e34c0000000p-3, -0x1.7fe4940000000p-4, -0x1.5723400000000p-3, -0x1.0d65620000000p-2, -0x1.e19ba00000000p-5, -0x1.c9fcb80000000p-6, -0x1.3048300000000p-2}
}
, {{0x1.c3d8c00000000p-6, -0x1.525a4c0000000p-2, -0x1.5a88180000000p-2, 0x1.f01ffe0000000p-3, 0x1.a291fe0000000p-6, -0x1.b4a1040000000p-3, -0x1.abb0300000000p-3, 0x1.fb515e0000000p-6, -0x1.ea23160000000p-5, 0x1.fe50c40000000p-6, -0x1.bf1e0a0000000p-3, -0x1.11c2f40000000p-2, -0x1.83a8080000000p-3, 0x1.3e35320000000p-5, 0x1.2433780000000p-4, 0x1.b38d680000000p-5, -0x1.5b17060000000p-3, 0x1.4a77160000000p-4, -0x1.a642260000000p-3, -0x1.3880000000000p-2, 0x1.94ed6c0000000p-8, -0x1.7a75f80000000p-6, -0x1.6bd2a20000000p-4, 0x1.00a4e80000000p-3, -0x1.d4b13e0000000p-4, 0x1.ff9fac0000000p-3, 0x1.888f500000000p-5, -0x1.6c78120000000p-10, -0x1.2acf460000000p-4, 0x1.b61a380000000p-3, 0x1.b10ebc0000000p-5, -0x1.9108200000000p-3}
, {-0x1.5d10020000000p-3, -0x1.468bd20000000p-5, -0x1.6fa3b20000000p-3, -0x1.21de0e0000000p-3, -0x1.d614580000000p-3, 0x1.0b2a660000000p-4, -0x1.515bf00000000p-6, 0x1.9482200000000p-4, -0x1.7323f80000000p-7, 0x1.2c75260000000p-5, -0x1.91bd4e0000000p-4, -0x1.664fb00000000p-5, 0x1.f158280000000p-5, -0x1.1eb91c0000000p-2, -0x1.296ae40000000p-5, -0x1.6e4be80000000p-3, 0x1.cd55820000000p-4, -0x1.5022dc0000000p-4, 0x1.80e5c20000000p-3, -0x1.9d86320000000p-4, -0x1.53de3c0000000p-5, 0x1.b1e6cc0000000p-4, 0x1.fb465c0000000p-4, -0x1.95e3080000000p-6, -0x1.86197a0000000p-5, -0x1.47aba20000000p-3, -0x1.75ad460000000p-8, 0x1.431df00000000p-5, 0x1.1b30ce0000000p-3, -0x1.3ae82c0000000p-3, -0x1.d895860000000p-2, -0x1.a93df80000000p-3}
}
, {{-0x1.7202860000000p-4, -0x1.74af8c0000000p-3, -0x1.336ff60000000p-2, 0x1.6638500000000p-1, -0x1.948e840000000p-3, -0x1.808bb00000000p-7, 0x1.537d4e0000000p-5, -0x1.4bc5660000000p-3, -0x1.74ae940000000p-2, 0x1.56f7000000000p-4, -0x1.1f55180000000p-4, 0x1.3449aa0000000p-8, -0x1.d8c5920000000p-4, -0x1.48cf740000000p-4, -0x1.f5393a0000000p-1, -0x1.086ede0000000p-4, 0x1.2be6040000000p-2, -0x1.c5c6440000000p-4, -0x1.c1126a0000000p-6, -0x1.7d0b120000000p-3, 0x1.f151c60000000p-3, 0x1.1b15e60000000p+1, 0x1.22ba120000000p-4, 0x1.c064240000000p-4, -0x1.f906520000000p-2, -0x1.30c2de0000000p-4, -0x1.a798420000000p-3, 0x1.6a28340000000p-6, -0x1.2768ae0000000p-3, 0x1.3161c40000000p-1, 0x1.781fc60000000p-1, 0x1.85a9140000000p-4}
, {0x1.e634d80000000p-4, 0x1.729f120000000p-6, 0x1.d0d9240000000p-8, 0x1.421ba20000000p-2, -0x1.a5d92c0000000p-2, 0x1.46e3ce0000000p-4, 0x1.8e1f9a0000000p-4, 0x1.b35fce0000000p-5, 0x1.60a6160000000p-10, -0x1.6bad4a0000000p-4, 0x1.ea8cde0000000p-2, 0x1.64d7b60000000p-4, 0x1.a30a2c0000000p-2, -0x1.4ce2a20000000p-7, 0x1.14ad080000000p-2, 0x1.628ec40000000p-7, -0x1.6fa0100000000p-1, 0x1.b596360000000p-3, 0x1.df61760000000p-5, 0x1.3c5bda0000000p-4, -0x1.348b3e0000000p-5, -0x1.a08dce0000000p-1, 0x1.e95df80000000p-4, 0x1.f27c440000000p-6, 0x1.159e680000000p-4, 0x1.6af09e0000000p-3, 0x1.28ea340000000p-3, -0x1.14d7440000000p-2, -0x1.8938200000000p-3, -0x1.2cfb4a0000000p+0, -0x1.d8c9cc0000000p-1, 0x1.c03ee60000000p-2}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_GROUPS
/**
  ******************************************************************************
  * @file    maxpool1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _MAX_POOLING1D_136_H_
#define _MAX_POOLING1D_136_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  32
#define INPUT_SAMPLES   29
#define POOL_SIZE       4
#define POOL_STRIDE     4
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

typedef float max_pooling1d_136_output_type[POOL_LENGTH][INPUT_CHANNELS];

#if 0
void max_pooling1d_136(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  number_t output[POOL_LENGTH][INPUT_CHANNELS]); 	// OUT
#endif

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH

#endif//_MAX_POOLING1D_136_H_
/**
  ******************************************************************************
  * @file    maxpool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "max_pooling1d_136.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#endif

#define INPUT_CHANNELS  32
#define INPUT_SAMPLES   29
#define POOL_SIZE       4
#define POOL_STRIDE     4
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

// For fixed point quantization
#define INPUT_SCALE_FACTOR 0
#define OUTPUT_SCALE_FACTOR 0
#define OUTPUT_ROUND_MODE ROUND_MODE_NONE
#define NUMBER_T float
#define LONG_NUMBER_T float


static inline void max_pooling1d_136(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  NUMBER_T output[POOL_LENGTH][INPUT_CHANNELS]) {	// OUT

  unsigned short pos_x, k; 	// loop indexes for output volume
  unsigned int x;
  static LONG_NUMBER_T max[INPUT_CHANNELS];

  for (pos_x = 0; pos_x < POOL_LENGTH; pos_x++) {
    for (k = 0; k < INPUT_CHANNELS; k++) {
#ifdef ACTIVATION_LINEAR
      max[k] = input[pos_x*POOL_STRIDE][k];
      x = 1;
#elif defined(ACTIVATION_RELU)
      max[k] = 0;
      x = 0;
#else
#error "Unsupported activation function"
#endif
    }

    for (; x < POOL_SIZE; x++) {
      for (k = 0; k < INPUT_CHANNELS; k++) {
        if (max[k] < input[(pos_x * POOL_STRIDE) + x][k])
          max[k] = input[(pos_x * POOL_STRIDE) + x][k];
      }
    }

    for (k = 0; k < INPUT_CHANNELS; k++) {
      output[pos_x][k] = scale_and_clamp_to(NUMBER_T, max[k], INPUT_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
    }
  }
}

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH
#undef ACTIVATION_LINEAR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    flatten.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _FLATTEN_17_H_
#define _FLATTEN_17_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define OUTPUT_DIM 224

typedef float flatten_17_output_type[OUTPUT_DIM];

#if 0
void flatten_17(
  const number_t input[7][32], 			      // IN
	number_t output[OUTPUT_DIM]); 			                // OUT
#endif

#undef OUTPUT_DIM

#endif//_FLATTEN_17_H_
/**
  ******************************************************************************
  * @file    flatten.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 2.0.0
  * @date    26 november 2021
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "flatten_17.h"
#include "number.h"
#endif

#define OUTPUT_DIM 224

#define NUMBER_T float
#define LONG_NUMBER_T float

static inline void flatten_17(
  const NUMBER_T input[7][32], 			      // IN
	NUMBER_T output[OUTPUT_DIM]) {			                // OUT

  NUMBER_T *input_flat = (NUMBER_T *)input;

  // Copy data from input to output only if input and output don't point to the same memory address already
  if (input_flat != output) {
    for (size_t i = 0; i < OUTPUT_DIM; i++) {
      output[i] = input_flat[i];
    }
  }
}

#undef OUTPUT_DIM
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    fc.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _DENSE_17_H_
#define _DENSE_17_H_

#ifndef SINGLE_FILE
#include "number.h"
#include <stdint.h>
#endif

#define INPUT_SAMPLES 224
#define FC_UNITS 3

typedef float dense_17_output_type[FC_UNITS];

#if 0
void dense_17(
  const number_t input[INPUT_SAMPLES], 			      // IN
	const number_t kernel[FC_UNITS][INPUT_SAMPLES],  // IN

	const number_t bias[FC_UNITS],			              // IN

	number_t output[FC_UNITS]); 			                // OUT
#endif

#undef INPUT_SAMPLES
#undef FC_UNITS

#endif//_DENSE_17_H_
/**
  ******************************************************************************
  * @file    fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "dense_17.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_SAMPLES 224
#define FC_UNITS 3
#define ACTIVATION_LINEAR

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 0
#define BIASES_SCALE_FACTOR 0
#define TMP_SCALE_FACTOR 0
#define INPUT_SCALE_FACTOR 0
#define OUTPUT_SCALE_FACTOR 0
#define OUTPUT_ROUND_MODE ROUND_MODE_NONE
#define NUMBER_T float
#define LONG_NUMBER_T float


static inline void dense_17(
  const NUMBER_T input[INPUT_SAMPLES], 			      // IN
	const NUMBER_T kernel[FC_UNITS][INPUT_SAMPLES],  // IN

	const NUMBER_T bias[FC_UNITS],			              // IN

	NUMBER_T output[FC_UNITS]) {			                // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short k, z; 
  LONG_NUMBER_T output_acc;

  for (k = 0; k < FC_UNITS; k++) { 
    output_acc = 0;
    for (z = 0; z < INPUT_SAMPLES; z++) 
      output_acc = output_acc + ((LONG_NUMBER_T)kernel[k][z] * (LONG_NUMBER_T)input[z]);

    output_acc = scale(NUMBER_T, output_acc, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);

    output_acc += scale(NUMBER_T, (LONG_NUMBER_T)bias[k], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);


    // Activation function
#ifdef ACTIVATION_LINEAR
    // Linear (MEANS NONE)
    output[k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
    // ReLU
    if (output_acc < 0) {
      output[k] = 0;
    } else {
#if defined(ACTIVATION_RELU6)
      if (output_acc > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
        output_acc = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
      }
#endif
      output[k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
    }
#else
#error "Unsupported activation function"
#endif
  }
#else

#if BIASES_SCALE_FACTOR > WEIGHTS_SCALE_FACTOR
#error "CMSIS-NN does not support BIASES_SCALE_FACTOR larger than WEIGHTS_SCALE_FACTOR"
#endif

#error "Data type unsupported by CMSIS-NN"

#endif
}

#undef INPUT_SAMPLES
#undef FC_UNITS
#undef ACTIVATION_LINEAR
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    weights/fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_SAMPLES 224
#define FC_UNITS 3


const float dense_17_bias[FC_UNITS] = {-0x1.33f8540000000p-1, 0x1.df25160000000p-1, -0x1.cfdd840000000p-2}
;

const float dense_17_kernel[FC_UNITS][INPUT_SAMPLES] = {{0x1.8c12c20000000p-6, 0x1.2263e20000000p-5, 0x1.1a997c0000000p-2, 0x1.4ebe220000000p-1, -0x1.dd97160000000p-3, 0x1.940c920000000p-7, -0x1.9f2e720000000p-1, -0x1.00c18e0000000p-4, -0x1.8776e40000000p-5, 0x1.9f62ec0000000p-3, -0x1.6f36820000000p-1, -0x1.5aaa420000000p-3, 0x1.1729da0000000p-7, -0x1.2d8fb80000000p-2, -0x1.adfae00000000p-2, 0x1.d6841e0000000p-7, -0x1.a4cdaa0000000p-7, 0x1.08cf540000000p-1, -0x1.f89bf00000000p-5, -0x1.a60eea0000000p-7, 0x1.131be00000000p-3, 0x1.c1090e0000000p-6, -0x1.3bc1940000000p-3, 0x1.ee4d340000000p-5, -0x1.2913c20000000p-1, 0x1.1bcbe60000000p-3, 0x1.2bfa8e0000000p-3, 0x1.90d2ca0000000p-5, 0x1.7e7d520000000p-1, 0x1.46860e0000000p-4, -0x1.2537560000000p-6, 0x1.9e55da0000000p-2, 0x1.41197e0000000p-2, 0x1.e1cfb80000000p-5, 0x1.2b251c0000000p-7, 0x1.47dc2a0000000p-1, -0x1.7567920000000p-4, 0x1.15b32e0000000p-4, -0x1.2a0a360000000p+0, -0x1.4e7c380000000p-8, 0x1.bc31340000000p-3, -0x1.ae4b2e0000000p-4, -0x1.69333c0000000p-1, -0x1.7dd9220000000p-4, -0x1.186cda0000000p-3, -0x1.92c52e0000000p-1, -0x1.3491e20000000p-3, 0x1.6380d80000000p-6, -0x1.19ee920000000p-3, 0x1.3e0ab40000000p-2, 0x1.9405a80000000p-6, -0x1.9b323a0000000p-4, -0x1.eb9e5e0000000p-6, 0x1.fe2b780000000p-7, 0x1.1e7eac0000000p-3, -0x1.68a8980000000p-4, -0x1.c1bc8a0000000p-3, 0x1.4973d20000000p-2, 0x1.53c6440000000p-4, -0x1.cfd7340000000p-5, 0x1.56e6e20000000p-1, 0x1.2640560000000p-6, 0x1.4506e20000000p-4, 0x1.6be8720000000p-1, 0x1.cd7dc80000000p-3, -0x1.179b720000000p-5, 0x1.0e86c40000000p-5, 0x1.b694420000000p-1, -0x1.1636c40000000p-4, -0x1.deceb00000000p-4, -0x1.219b7a0000000p+0, 0x1.ce30e20000000p-7, 0x1.7a69100000000p-5, 0x1.88796c0000000p-6, -0x1.e96f7a0000000p-2, -0x1.7fb2820000000p-4, 0x1.3dcc2c0000000p-2, -0x1.e773500000000p-1, -0x1.1d17ec0000000p-2, 0x1.04baac0000000p-2, 0x1.595e840000000p-3, 0x1.66cc380000000p-4, -0x1.1d25f40000000p-6, -0x1.9035200000000p-4, 0x1.697d860000000p-3, -0x1.a50f820000000p-5, -0x1.2fd5de0000000p-4, -0x1.aac34c0000000p-4, -0x1.1316980000000p-2, 0x1.21d37e0000000p-3, -0x1.4d3e560000000p-5, -0x1.48a7880000000p-3, 0x1.6f27d60000000p+0, 0x1.2a880a0000000p-4, 0x1.f35cfc0000000p-6, 0x1.15f6bc0000000p-2, 0x1.120bec0000000p-2, -0x1.5430040000000p-6, 0x1.3cfa720000000p-4, 0x1.0f92440000000p-1, -0x1.ab8c4c0000000p-4, 0x1.7d33280000000p-4, -0x1.13ff980000000p+0, 0x1.9a1fb40000000p-7, -0x1.96eaee0000000p-4, 0x1.c71cfe0000000p-3, -0x1.ad7e340000000p-2, -0x1.ce6eaa0000000p-3, -0x1.0c6ac60000000p-3, -0x1.d0d04c0000000p-2, -0x1.8d46c00000000p-2, -0x1.1d59760000000p-3, -0x1.0543280000000p-6, 0x1.6b26f80000000p-5, 0x1.ad930c0000000p-5, -0x1.202c940000000p-5, 0x1.5da09c0000000p-3, 0x1.10db9e0000000p-3, 0x1.3b58580000000p-5, 0x1.41d7900000000p-4, -0x1.8fd65e0000000p-3, 0x1.9df37e0000000p-4, -0x1.a4525e0000000p-3, 0x1.4db2580000000p-7, 0x1.5740520000000p-1, -0x1.295b1a0000000p-7, -0x1.0dc81a0000000p-3, 0x1.6aa6420000000p-1, 0x1.f224700000000p-3, 0x1.d65fd40000000p-4, 0x1.235f7c0000000p-2, 0x1.a090fa0000000p-1, -0x1.8e82760000000p-3, -0x1.7273000000000p-3, -0x1.a2f3b40000000p-1, -0x1.32d9a00000000p-4, 0x1.eabeb80000000p-2, 0x1.b441cc0000000p-6, -0x1.c75b600000000p-1, -0x1.0075760000000p-2, -0x1.68f6700000000p-5, -0x1.01a1f40000000p-2, -0x1.8cc16c0000000p-2, 0x1.9d89340000000p-3, 0x1.2c79fa0000000p-4, 0x1.ff10180000000p-3, -0x1.2ea9ba0000000p-4, -0x1.8717d40000000p-8, 0x1.4fede40000000p-4, -0x1.bd81420000000p-5, 0x1.118bf80000000p-3, 0x1.e4a8d40000000p-5, -0x1.0b8a580000000p-5, 0x1.c71b2c0000000p-3, 0x1.696fd60000000p-3, 0x1.d8492c0000000p-4, 0x1.078c180000000p-6, 0x1.2b4fdc0000000p-6, -0x1.4647460000000p-3, 0x1.0bf8e80000000p-1, -0x1.4b87fa0000000p-4, -0x1.7fd1260000000p-3, 0x1.8f72720000000p-3, 0x1.ffb7ea0000000p-2, -0x1.ddef720000000p-3, 0x1.d2b17a0000000p-4, -0x1.258c0a0000000p+0, -0x1.18f7b40000000p-6, 0x1.b69ab00000000p-5, 0x1.1c0c020000000p-3, -0x1.1d0cb60000000p-1, 0x1.f5d8d60000000p-5, -0x1.7c63aa0000000p-5, -0x1.04f5da0000000p-3, -0x1.e98c1e0000000p-2, -0x1.9537040000000p-4, -0x1.e40eee0000000p-5, 0x1.fff7520000000p-4, 0x1.a8527a0000000p-4, 0x1.5527ee0000000p-4, 0x1.dec6740000000p-3, -0x1.1a4c1a0000000p-3, -0x1.8bbd7a0000000p-5, -0x1.f4881c0000000p-6, -0x1.15cd780000000p-5, 0x1.0e6bf20000000p-2, -0x1.5b91780000000p-3, -0x1.5af1460000000p-4, 0x1.4fa4780000000p+0, 0x1.bd2c340000000p-5, -0x1.8750540000000p-3, 0x1.70b47e0000000p-2, -0x1.1c6c180000000p-5, 0x1.7380040000000p-4, 0x1.a0b3700000000p-3, 0x1.0254440000000p+0, -0x1.e7eb180000000p-3, 0x1.1847b80000000p-8, -0x1.c4433e0000000p-1, -0x1.636b980000000p-3, 0x1.c04db00000000p-2, -0x1.f3343a0000000p-4, -0x1.85a8c00000000p-1, -0x1.cf70b40000000p-3, 0x1.c133220000000p-4, -0x1.291b940000000p-1, -0x1.1e5fd00000000p-1, 0x1.96f45a0000000p-4, 0x1.3806720000000p-4, 0x1.2f94b60000000p-2, 0x1.f923a80000000p-11, 0x1.a23b220000000p-8, 0x1.13acb20000000p-3, 0x1.27a6800000000p-3, -0x1.3a77260000000p-9, -0x1.42803e0000000p-4, -0x1.6aa09c0000000p-2, 0x1.ed8a320000000p-2, -0x1.3d7a800000000p-3, -0x1.e0e4560000000p-4, 0x1.421c000000000p-1, -0x1.efcaaa0000000p-5, 0x1.db59a80000000p-4, 0x1.1e70620000000p-2}
, {0x1.c552d00000000p-3, -0x1.6cfbd80000000p-4, 0x1.ccf6ba0000000p-8, -0x1.4c64700000000p-1, 0x1.7a1b0e0000000p-2, 0x1.b0e6920000000p-5, -0x1.e09d8a0000000p-4, 0x1.c08cac0000000p-8, -0x1.d9fabc0000000p-1, -0x1.0bfa900000000p-1, 0x1.d63a2e0000000p-8, 0x1.5812160000000p-6, 0x1.fea4420000000p-4, 0x1.be69640000000p-2, -0x1.1624640000000p-2, 0x1.62f0f60000000p-3, 0x1.30870e0000000p-4, -0x1.22a01a0000000p-2, -0x1.7db9160000000p-3, -0x1.41900a0000000p-5, 0x1.fd3f040000000p-3, -0x1.8864820000000p-3, 0x1.6fb2d20000000p-5, -0x1.6e43e60000000p-4, 0x1.8104ea0000000p-1, 0x1.adcd7e0000000p-2, -0x1.01da6a0000000p-4, 0x1.99bdea0000000p-6, -0x1.c7f5a60000000p-2, -0x1.0cc5780000000p-3, -0x1.68ef820000000p-6, -0x1.14ba820000000p-2, 0x1.a269260000000p-2, 0x1.47a0d60000000p-4, -0x1.a327e60000000p-7, -0x1.55095a0000000p-2, -0x1.0cc08e0000000p-6, 0x1.4409e00000000p-6, 0x1.49f78e0000000p-2, -0x1.a80e280000000p-9, -0x1.5a40160000000p-1, -0x1.00283e0000000p-2, 0x1.97f0bc0000000p-3, 0x1.761ce80000000p-3, 0x1.d43b960000000p-3, 0x1.caab360000000p-1, -0x1.c61dd60000000p-2, -0x1.a4f81a0000000p-5, -0x1.dc03d60000000p-6, -0x1.61d14c0000000p-5, 0x1.69d9380000000p-5, 0x1.c27c320000000p-7, 0x1.ac04700000000p-3, 0x1.73ad440000000p-4, -0x1.62bacc0000000p-3, -0x1.5f07060000000p-4, -0x1.49984e0000000p-6, 0x1.cb63760000000p-2, -0x1.3d38920000000p-3, -0x1.94a00e0000000p-4, -0x1.cb758a0000000p-3, -0x1.f53cd60000000p-5, -0x1.0a6f3e0000000p-9, -0x1.a813420000000p-1, 0x1.0e9cb00000000p-2, -0x1.a985520000000p-5, -0x1.e463ea0000000p-3, -0x1.75018c0000000p-3, 0x1.f4b8b20000000p-3, 0x1.7c9c520000000p-4, 0x1.1d9d3c0000000p-2, 0x1.806c3a0000000p-3, -0x1.2539540000000p-1, -0x1.28b3c80000000p-4, -0x1.437e2c0000000p-3, 0x1.b152460000000p-6, 0x1.1252460000000p-3, 0x1.f9fcec0000000p-3, -0x1.3a0ca20000000p-3, -0x1.117bbe0000000p-4, 0x1.7c69800000000p-7, -0x1.48549e0000000p-3, -0x1.3e8d240000000p-3, 0x1.5108740000000p-4, 0x1.547d9c0000000p-5, -0x1.46aabe0000000p-3, -0x1.d7bb340000000p-6, 0x1.37b7100000000p-3, 0x1.c00be40000000p-2, 0x1.95e0d80000000p-2, -0x1.b311b20000000p-10, 0x1.c527400000000p-7, -0x1.b63fd00000000p-6, -0x1.5ed26c0000000p-4, -0x1.690e6e0000000p-3, -0x1.467a8c0000000p-1, 0x1.601eb60000000p-2, -0x1.abcbec0000000p-4, -0x1.359cc20000000p-4, 0x1.1333220000000p-4, 0x1.2ce7a40000000p-3, 0x1.97b0a40000000p-4, 0x1.a7ec720000000p-3, 0x1.19e0840000000p-4, -0x1.da93620000000p-2, -0x1.06261e0000000p-1, 0x1.5dbdf40000000p-2, 0x1.2846680000000p-2, -0x1.49273a0000000p-5, 0x1.fe0d2c0000000p-2, -0x1.f34e6c0000000p-3, 0x1.2f8c4c0000000p-5, -0x1.e7e0120000000p-3, -0x1.4c5d480000000p-4, 0x1.6e43b40000000p-5, -0x1.32bf240000000p-9, -0x1.47a1760000000p-2, 0x1.46b3a80000000p-5, -0x1.6d9ae60000000p-9, 0x1.75a5ec0000000p-5, 0x1.fa66f40000000p-4, 0x1.36a2300000000p-2, 0x1.4791fe0000000p-4, -0x1.312f4e0000000p-6, 0x1.7c1b780000000p-2, 0x1.05d5080000000p-6, -0x1.6b79a40000000p-4, -0x1.6cbd880000000p-1, 0x1.7f0e940000000p-1, 0x1.15acd40000000p-5, -0x1.25fe560000000p-2, -0x1.6ab6800000000p-4, 0x1.721c4c0000000p-5, 0x1.a69f400000000p-4, -0x1.a463520000000p-6, -0x1.1aa5f80000000p-7, -0x1.d62f900000000p-2, -0x1.2cbaee0000000p-3, 0x1.93903a0000000p-4, 0x1.e2fa9a0000000p-3, 0x1.056efa0000000p-3, 0x1.172e8c0000000p-1, -0x1.15bd900000000p-5, 0x1.4c780e0000000p-4, -0x1.08db7a0000000p-3, -0x1.f26d5e0000000p-3, 0x1.a6cbbe0000000p-4, 0x1.3b83a00000000p-5, 0x1.445e460000000p-4, -0x1.d287260000000p-4, 0x1.86cd5a0000000p-4, 0x1.ed1b820000000p-5, 0x1.968c4e0000000p-3, 0x1.ffa5920000000p-2, -0x1.87e0fc0000000p-3, 0x1.82be0e0000000p-5, 0x1.43cb540000000p-1, -0x1.4f29da0000000p-3, 0x1.be95d20000000p-4, -0x1.f40fa20000000p-2, 0x1.b1d3340000000p-2, -0x1.61536c0000000p-3, -0x1.e73ec80000000p-6, -0x1.fe1cdc0000000p-3, 0x1.4f7bac0000000p-2, -0x1.676b860000000p-3, 0x1.848a0e0000000p-3, 0x1.5d5f660000000p-6, -0x1.43e2080000000p-1, -0x1.fb58580000000p-3, -0x1.2c25e20000000p-2, 0x1.1648ea0000000p-2, 0x1.a2a82e0000000p-3, 0x1.23fafc0000000p-1, 0x1.c04b880000000p-7, -0x1.bce9800000000p-4, -0x1.1a90ec0000000p-3, -0x1.7b26200000000p-3, 0x1.aa42c80000000p-5, -0x1.398f960000000p-5, -0x1.6e468a0000000p-3, -0x1.e369da0000000p-5, 0x1.9ac9be0000000p-7, 0x1.2367d80000000p-5, 0x1.38cb8c0000000p-2, 0x1.1dc1540000000p-2, -0x1.ab352a0000000p-5, -0x1.7431dc0000000p-4, -0x1.6ec00a0000000p-2, 0x1.3b01a40000000p-3, -0x1.885ede0000000p-6, -0x1.7edfa00000000p-2, 0x1.da778e0000000p-2, -0x1.a52db40000000p-4, -0x1.f117ac0000000p-3, -0x1.de5d160000000p-2, 0x1.2febf00000000p-3, 0x1.cfef440000000p-4, -0x1.6859380000000p-3, 0x1.29dc6c0000000p-6, -0x1.00519c0000000p+0, -0x1.2278f00000000p-2, -0x1.1651300000000p-3, 0x1.8cb19a0000000p-8, -0x1.14d7220000000p-3, 0x1.7d97280000000p-1, -0x1.27ba920000000p-4, -0x1.4bbc160000000p-10, -0x1.adbf1a0000000p-3, -0x1.bfe1620000000p-3, 0x1.15e9980000000p-5, -0x1.7144020000000p-3, -0x1.c3664a0000000p-2, -0x1.38e1cc0000000p-5, 0x1.115ca40000000p-5, -0x1.01c9fa0000000p-3, 0x1.2c7d640000000p-2, 0x1.028ae80000000p-1, -0x1.3db4360000000p-2, 0x1.deb3a80000000p-9, -0x1.8c17fc0000000p-3, -0x1.7d7aaa0000000p-7, -0x1.68b4e80000000p-4, -0x1.0208300000000p-1}
, {-0x1.73afde0000000p-2, 0x1.c5384e0000000p-5, 0x1.6557fe0000000p-3, -0x1.f999600000000p-5, -0x1.66a9f00000000p-6, 0x1.7276260000000p-3, 0x1.83ecb20000000p-1, -0x1.3387700000000p-3, 0x1.d502620000000p-1, 0x1.634a7c0000000p-2, 0x1.2fce960000000p-1, -0x1.118ba60000000p-4, -0x1.172bb00000000p-3, -0x1.d821140000000p-3, 0x1.7193440000000p-1, -0x1.a850740000000p-3, 0x1.d956500000000p-6, 0x1.3c6d160000000p-6, -0x1.0bbdde0000000p-6, 0x1.1832fc0000000p-3, -0x1.0f884c0000000p-4, 0x1.cb12060000000p-7, -0x1.85e38e0000000p-6, 0x1.516b6e0000000p-5, -0x1.34dd700000000p-2, -0x1.176b4a0000000p-1, 0x1.c241ea0000000p-7, -0x1.ee34940000000p-6, -0x1.06b1e20000000p-1, -0x1.9f39580000000p-5, 0x1.1875020000000p-3, -0x1.8e8a7e0000000p-3, -0x1.a15df40000000p-1, 0x1.8d553c0000000p-5, 0x1.4b6f840000000p-2, -0x1.2288960000000p-5, 0x1.7369d40000000p-3, 0x1.10ce200000000p-3, 0x1.17ea940000000p-1, -0x1.3d0b360000000p-3, 0x1.fd05780000000p-2, 0x1.a397da0000000p-5, 0x1.7765880000000p-1, 0x1.df85ce0000000p-3, 0x1.17ce320000000p-3, -0x1.206ed20000000p-1, 0x1.1207600000000p-1, 0x1.9ccf600000000p-5, 0x1.9b71b20000000p-3, -0x1.1247b60000000p-2, 0x1.473a580000000p-3, 0x1.c3a4320000000p-6, -0x1.556a4e0000000p-5, 0x1.990ab60000000p-3, 0x1.5a77360000000p-4, -0x1.6ffe880000000p-6, 0x1.679b220000000p-4, -0x1.1947600000000p-1, 0x1.5c25f20000000p-5, 0x1.9d73100000000p-4, -0x1.cfdae40000000p-2, 0x1.7088e00000000p-5, -0x1.3bfc000000000p-6, 0x1.65c6d20000000p-4, -0x1.38570a0000000p-2, 0x1.74ab680000000p-3, 0x1.2ec51a0000000p-2, -0x1.2d703c0000000p-1, -0x1.5f44ac0000000p-4, 0x1.c0d74e0000000p-12, 0x1.5e42760000000p-1, 0x1.375c640000000p-4, 0x1.f0d19a0000000p-2, 0x1.4525740000000p-2, 0x1.993aec0000000p-1, 0x1.bc65e20000000p-9, -0x1.1e02d40000000p-4, 0x1.8d1e040000000p-2, 0x1.20c7de0000000p-1, -0x1.4e5e840000000p-5, -0x1.db7ba60000000p-7, -0x1.f10bac0000000p-5, 0x1.b78b5c0000000p-4, -0x1.72f65c0000000p-5, 0x1.db722e0000000p-5, 0x1.8d48380000000p-5, 0x1.66fa600000000p-3, 0x1.22e15e0000000p-3, -0x1.5a79f20000000p-2, -0x1.6c53e20000000p-1, 0x1.89c6160000000p-3, -0x1.0f18d20000000p-4, -0x1.1716e60000000p+0, 0x1.4ec3f40000000p-3, -0x1.cea9ae0000000p-5, 0x1.1878460000000p-2, -0x1.1314280000000p-1, -0x1.16285a0000000p-4, 0x1.a710860000000p-3, -0x1.db96f60000000p-2, 0x1.aa48820000000p-4, 0x1.bd65a80000000p-3, 0x1.28cc2c0000000p-1, -0x1.be96240000000p-8, 0x1.c5f3b60000000p-3, 0x1.d534420000000p-3, 0x1.792c320000000p-3, -0x1.33515e0000000p-3, 0x1.5d5f360000000p-4, -0x1.ef18ea0000000p-2, 0x1.2094580000000p-1, 0x1.475ca80000000p-6, 0x1.d162be0000000p-5, 0x1.f238ac0000000p-3, -0x1.51fb540000000p-4, 0x1.0cb6760000000p-4, 0x1.1794560000000p-3, -0x1.45c91a0000000p-4, 0x1.3a76dc0000000p-3, -0x1.1768280000000p-7, -0x1.e053980000000p-4, -0x1.13e7580000000p-1, 0x1.00d6580000000p-4, -0x1.bcf2e60000000p-6, -0x1.d1e8000000000p-1, -0x1.55dc2c0000000p-3, 0x1.5654200000000p-3, -0x1.c8cfec0000000p-6, -0x1.5f6f6a0000000p-1, -0x1.4fe3220000000p-4, 0x1.690ace0000000p-2, -0x1.3ab3b80000000p-1, 0x1.d07e2c0000000p-3, -0x1.986c1e0000000p-5, 0x1.058d180000000p-1, -0x1.491bf00000000p-3, 0x1.65fa6e0000000p-3, -0x1.d974320000000p-8, 0x1.afc1440000000p-1, -0x1.84f2e60000000p-6, -0x1.3145da0000000p-5, -0x1.79d3600000000p-2, 0x1.48fede0000000p-2, -0x1.3d04140000000p-6, -0x1.6a7c6c0000000p-8, 0x1.940c040000000p-3, 0x1.80cd7a0000000p-3, 0x1.6847940000000p-5, 0x1.48f2540000000p-3, 0x1.0dc6200000000p-3, 0x1.a20d1a0000000p-7, 0x1.5adcde0000000p-5, -0x1.b043e20000000p-4, -0x1.479f2e0000000p-1, 0x1.3e59ba0000000p-4, 0x1.90aed80000000p-6, -0x1.8339c20000000p-1, -0x1.9b7c540000000p-3, 0x1.96618a0000000p-4, -0x1.4205b00000000p-3, -0x1.88c3000000000p-2, -0x1.ab68fe0000000p-5, -0x1.363d7a0000000p-6, -0x1.de90460000000p-3, 0x1.8479600000000p-3, 0x1.6332020000000p-6, 0x1.98d0700000000p-1, -0x1.51cc680000000p-4, 0x1.2cba520000000p-3, 0x1.00b76c0000000p-13, 0x1.06fb1a0000000p-1, -0x1.12644a0000000p-3, 0x1.10d2d80000000p-4, -0x1.b0b8380000000p-1, 0x1.1b98260000000p-3, -0x1.d2d43a0000000p-4, 0x1.d2c2120000000p-3, -0x1.503ea40000000p-8, 0x1.c4a9de0000000p-9, 0x1.35a1180000000p-3, -0x1.3050200000000p-3, -0x1.9ca1920000000p-4, 0x1.8355ae0000000p-3, -0x1.0a9f0a0000000p-3, -0x1.76d1ce0000000p-2, -0x1.1ba0960000000p-1, 0x1.12b5060000000p-4, 0x1.3ad23a0000000p-4, -0x1.49ddc80000000p-1, -0x1.a9d0d60000000p-5, -0x1.9bd3a80000000p-5, 0x1.58cdf40000000p-3, -0x1.401b280000000p-1, 0x1.ad82cc0000000p-3, 0x1.c6ed6e0000000p-4, -0x1.8cb24e0000000p-2, 0x1.321c060000000p-3, 0x1.ae275a0000000p-5, 0x1.4710aa0000000p-1, -0x1.081bac0000000p-4, 0x1.5d1c580000000p-2, 0x1.e99bf40000000p-3, 0x1.d536c40000000p-1, -0x1.cd18020000000p-6, -0x1.9ea0b00000000p-5, -0x1.283b320000000p-1, 0x1.96d2aa0000000p-2, 0x1.045d2e0000000p-3, 0x1.148bce0000000p-3, -0x1.f6a9fe0000000p-3, -0x1.90d76a0000000p-5, 0x1.86da520000000p-4, 0x1.3f861c0000000p-2, -0x1.5d63c00000000p-7, -0x1.4ea63a0000000p-4, 0x1.13bdce0000000p-5, 0x1.7e06100000000p-4, -0x1.b971320000000p-1, -0x1.61dcbe0000000p-7, 0x1.782b100000000p-4, -0x1.f319fe0000000p-3, 0x1.bf22700000000p-6, -0x1.636e760000000p-6, -0x1.0d93bc0000000p-2}
}
;

#undef INPUT_SAMPLES
#undef FC_UNITS
/**
  ******************************************************************************
  * @file    model.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    08 july 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */


#ifdef __cplusplus
extern "C" {
#endif

#ifndef __MODEL_H__
#define __MODEL_H__

#ifndef SINGLE_FILE
#include "number.h"

 // InputLayer is excluded
#include "max_pooling1d_132.h" // InputLayer is excluded
#include "conv1d_108.h" // InputLayer is excluded
#include "max_pooling1d_133.h" // InputLayer is excluded
#include "conv1d_109.h" // InputLayer is excluded
#include "max_pooling1d_134.h" // InputLayer is excluded
#include "conv1d_110.h" // InputLayer is excluded
#include "max_pooling1d_135.h" // InputLayer is excluded
#include "conv1d_111.h" // InputLayer is excluded
#include "max_pooling1d_136.h" // InputLayer is excluded
#include "flatten_17.h" // InputLayer is excluded
#include "dense_17.h"
#endif


#define MODEL_INPUT_DIM_0 16000
#define MODEL_INPUT_DIM_1 1
#define MODEL_INPUT_DIMS 16000 * 1

#define MODEL_OUTPUT_SAMPLES 3

#define MODEL_INPUT_SCALE_FACTOR 0 // scale factor of InputLayer
#define MODEL_INPUT_ROUND_MODE ROUND_MODE_NONE
#define MODEL_INPUT_NUMBER_T float
#define MODEL_INPUT_LONG_NUMBER_T float

// node 0 is InputLayer so use its output shape as input shape of the model
// typedef  input_t[16000][1];
typedef float input_t[16000][1];
typedef dense_17_output_type output_t;


void cnn(
  const input_t input,
  output_t output);

void reset(void);

#endif//__MODEL_H__


#ifdef __cplusplus
} // extern "C"
#endif
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
