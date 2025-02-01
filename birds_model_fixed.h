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


#define ARM_NN_TRUNCATE 1
#define RISCV_NN_TRUNCATE 1

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

#define NUMBER_MIN_INT16_T -32768
#define NUMBER_MAX_INT16_T 32767

static inline int32_t min_int16_t(
    int32_t a,
    int32_t b) {
	if (a <= b)
		return a;
	return b;
}

static inline int32_t max_int16_t(
    int32_t a,
    int32_t b) {
	if (a >= b)
		return a;
	return b;
}

static inline int32_t scale_number_t_int16_t(
  int32_t number, int scale_factor, round_mode_t round_mode) {


  if (scale_factor <= 0) {
#ifdef TRAPV_SHIFT
    // Check for possible overflow of left shift
    if (number > INT32_MAX >> -scale_factor) {
      fprintf(stderr,
              "Error: scale() overflow, number=%d, scale_factor=%d, limit=%d\n",
              number,
              scale_factor,
              INT16_MAX >> -scale_factor);
      assert(number <= INT32_MAX >> -scale_factor);
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
static inline int16_t clamp_to_number_t_int16_t(
  int32_t number) {
	return (int16_t) max_int16_t(
      NUMBER_MIN_INT16_T,
      min_int16_t(
        NUMBER_MAX_INT16_T, number));
}
static inline int16_t scale_and_clamp_to_number_t_int16_t(
  int32_t number, int scale_factor, round_mode_t round_mode) {
#ifdef WITH_CMSIS_NN
  // Not really CMSIS-NN but use SSAT anyway
  if (scale_factor <= 0) {
    // No rounding to apply when shifting left
    return __SSAT(number << - scale_factor, sizeof(int16_t) * 8);
  } else {
    if (round_mode == ROUND_MODE_NEAREST) {
      number += (1 << (scale_factor - 1)); // +0.5 in fixed-point
    }
    return __SSAT(number >> scale_factor, sizeof(int16_t) * 8);
  }
#else
  number = scale_number_t_int16_t(number, scale_factor, round_mode);
  return clamp_to_number_t_int16_t(number);
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

typedef int16_t max_pooling1d_132_output_type[POOL_LENGTH][INPUT_CHANNELS];

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
#define INPUT_SCALE_FACTOR 13
#define OUTPUT_SCALE_FACTOR 13
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


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

typedef int16_t conv1d_108_output_type[CONV_OUTSAMPLES][CONV_FILTERS];

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
#define WEIGHTS_SCALE_FACTOR 13
#define BIASES_SCALE_FACTOR 13
#define TMP_SCALE_FACTOR 13
#define INPUT_SCALE_FACTOR 13
#define OUTPUT_SCALE_FACTOR 13
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


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

  static q15_t bufferA[2*INPUT_CHANNELS*CONV_KERNEL_SIZE];
#if INPUT_CHANNELS % 2 == 0 && CONV_FILTERS % 2 == 0
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_fast_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_fast_nonsquare(
#endif
#else
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_basic_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_basic_nonsquare(
#endif
#endif
                                      (q15_t*)input, //Im_in
                                      INPUT_SAMPLES, //dim_im_in_x
                                      1, //dim_im_in_y
                                      INPUT_CHANNELS, //ch_im_in
                                      (q15_t*)kernel, //wt
                                      CONV_FILTERS, //ch_im_out
                                      CONV_KERNEL_SIZE, //dim_kernel_x
                                      1, //dim_kernel_y
                                      ZEROPADDING_LEFT, //padding_x, left and right must be equal
                                      0, //padding_y
                                      CONV_STRIDE, //stride_x
                                      1, //stride_y
                                      (q15_t*)bias, //bias
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - BIASES_SCALE_FACTOR, //bias_shift
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, //out_shift
                                      (q15_t*)output, //Im_out
                                      CONV_OUTSAMPLES, //dim_im_out_x
                                      1, //dim_im_out_y
                                      bufferA, //bufferA
                                      NULL //bufferB, unused
                                      );
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#endif
#elif !defined(ACTIVATION_LINEAR)
#error "Unsupported activation with CMSIS-NN"
#endif


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


const int16_t  conv1d_108_bias[CONV_FILTERS] = {-462, -452, 85, 109, -1105, 135, 137, 139}
;

const int16_t  conv1d_108_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{{-620}
, {-731}
, {-324}
, {368}
, {-169}
, {203}
, {-42}
, {-1004}
, {-284}
, {630}
, {-333}
, {-33}
, {255}
, {-59}
, {63}
, {-96}
, {492}
, {953}
, {755}
, {227}
, {735}
, {330}
, {1146}
, {22}
, {192}
, {503}
, {442}
, {326}
, {543}
, {980}
, {375}
, {855}
, {-274}
, {363}
, {-475}
, {456}
, {-497}
, {275}
, {-1195}
, {-417}
}
, {{1337}
, {1040}
, {-1285}
, {-200}
, {1415}
, {-1223}
, {471}
, {-27}
, {634}
, {476}
, {513}
, {651}
, {864}
, {-159}
, {-510}
, {965}
, {1129}
, {886}
, {-70}
, {-484}
, {1137}
, {48}
, {881}
, {1361}
, {-1056}
, {-642}
, {248}
, {853}
, {484}
, {-654}
, {-1272}
, {132}
, {-649}
, {595}
, {245}
, {-537}
, {-721}
, {-376}
, {322}
, {-700}
}
, {{-18972}
, {-24258}
, {-20700}
, {-24291}
, {-22429}
, {-25370}
, {-21994}
, {-18911}
, {-14547}
, {-10948}
, {-6687}
, {-11855}
, {-5745}
, {-10555}
, {-9143}
, {-13217}
, {-6837}
, {-14633}
, {-8370}
, {-12857}
, {-13954}
, {-12890}
, {-11249}
, {-14381}
, {-14287}
, {-21570}
, {-17700}
, {-17367}
, {-24381}
, {-22894}
, {-25973}
, {-28108}
, {-27166}
, {-31804}
, {-30983}
, {-26700}
, {-28600}
, {-26214}
, {-20691}
, {-24415}
}
, {{10113}
, {11890}
, {15814}
, {5150}
, {11865}
, {8441}
, {18737}
, {4490}
, {15129}
, {12611}
, {22860}
, {19411}
, {24431}
, {7228}
, {23749}
, {15517}
, {30210}
, {29613}
, {11883}
, {28222}
, {27392}
, {17874}
, {25486}
, {32767}
, {951}
, {30151}
, {31628}
, {13693}
, {21341}
, {29450}
, {13429}
, {15560}
, {29577}
, {21289}
, {-7596}
, {18632}
, {23009}
, {-3062}
, {22711}
, {21687}
}
, {{1130}
, {-1034}
, {1467}
, {-2209}
, {828}
, {-614}
, {1694}
, {532}
, {966}
, {-190}
, {856}
, {108}
, {302}
, {501}
, {392}
, {684}
, {-1406}
, {198}
, {-667}
, {1612}
, {-1271}
, {740}
, {-701}
, {781}
, {377}
, {1216}
, {1280}
, {-350}
, {1395}
, {-1081}
, {597}
, {-685}
, {920}
, {849}
, {733}
, {534}
, {447}
, {-395}
, {-1318}
, {-725}
}
, {{25909}
, {16718}
, {9242}
, {4637}
, {21620}
, {-106}
, {12032}
, {17774}
, {14960}
, {15439}
, {29309}
, {7202}
, {31145}
, {-17273}
, {29364}
, {21915}
, {25489}
, {32767}
, {-3082}
, {32767}
, {6848}
, {23298}
, {26928}
, {29666}
, {16848}
, {30571}
, {24464}
, {7904}
, {25771}
, {21020}
, {21098}
, {16147}
, {23385}
, {16057}
, {774}
, {16952}
, {13411}
, {7782}
, {18510}
, {21064}
}
, {{21742}
, {10719}
, {14150}
, {21577}
, {6791}
, {10093}
, {30559}
, {-15657}
, {32194}
, {-32768}
, {32767}
, {9721}
, {32767}
, {32767}
, {12102}
, {32767}
, {6897}
, {32767}
, {-2458}
, {32369}
, {22195}
, {29909}
, {10869}
, {21094}
, {24793}
, {20258}
, {15473}
, {32701}
, {18202}
, {9953}
, {27642}
, {20796}
, {-10191}
, {15915}
, {20097}
, {2939}
, {11297}
, {10883}
, {4304}
, {10475}
}
, {{25366}
, {17049}
, {23461}
, {7640}
, {26785}
, {-9141}
, {28591}
, {23459}
, {13297}
, {21322}
, {25170}
, {15601}
, {32767}
, {16382}
, {32767}
, {-7282}
, {32767}
, {32767}
, {-11163}
, {32767}
, {-10538}
, {30753}
, {27554}
, {32767}
, {22570}
, {16393}
, {30122}
, {26948}
, {18903}
, {32767}
, {22232}
, {-2442}
, {23890}
, {18619}
, {-1744}
, {11755}
, {18837}
, {-4745}
, {17263}
, {19123}
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

typedef int16_t max_pooling1d_133_output_type[POOL_LENGTH][INPUT_CHANNELS];

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
#define INPUT_SCALE_FACTOR 13
#define OUTPUT_SCALE_FACTOR 13
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


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

typedef int16_t conv1d_109_output_type[CONV_OUTSAMPLES][CONV_FILTERS];

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
#define WEIGHTS_SCALE_FACTOR 13
#define BIASES_SCALE_FACTOR 13
#define TMP_SCALE_FACTOR 13
#define INPUT_SCALE_FACTOR 13
#define OUTPUT_SCALE_FACTOR 13
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


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

  static q15_t bufferA[2*INPUT_CHANNELS*CONV_KERNEL_SIZE];
#if INPUT_CHANNELS % 2 == 0 && CONV_FILTERS % 2 == 0
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_fast_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_fast_nonsquare(
#endif
#else
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_basic_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_basic_nonsquare(
#endif
#endif
                                      (q15_t*)input, //Im_in
                                      INPUT_SAMPLES, //dim_im_in_x
                                      1, //dim_im_in_y
                                      INPUT_CHANNELS, //ch_im_in
                                      (q15_t*)kernel, //wt
                                      CONV_FILTERS, //ch_im_out
                                      CONV_KERNEL_SIZE, //dim_kernel_x
                                      1, //dim_kernel_y
                                      ZEROPADDING_LEFT, //padding_x, left and right must be equal
                                      0, //padding_y
                                      CONV_STRIDE, //stride_x
                                      1, //stride_y
                                      (q15_t*)bias, //bias
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - BIASES_SCALE_FACTOR, //bias_shift
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, //out_shift
                                      (q15_t*)output, //Im_out
                                      CONV_OUTSAMPLES, //dim_im_out_x
                                      1, //dim_im_out_y
                                      bufferA, //bufferA
                                      NULL //bufferB, unused
                                      );
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#endif
#elif !defined(ACTIVATION_LINEAR)
#error "Unsupported activation with CMSIS-NN"
#endif


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


const int16_t  conv1d_109_bias[CONV_FILTERS] = {-5402, -344, 211, -680, -378, 254, -1175, -229, -329, -536, 1719, -1029, -1143, -585, -272, -553}
;

const int16_t  conv1d_109_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{{-1547, 931, 5204, -200, 1092, 5640, 4704, 228}
, {-1885, 1398, -10059, 132, 2467, 4848, 2325, 3881}
}
, {{-1093, 2112, -3339, 2012, -1179, 4297, -222, -1292}
, {1407, -1162, -3813, 3311, -2736, 5732, 5306, 4333}
}
, {{88, 1230, 6785, -4938, 2594, -9033, -12997, -9258}
, {-186, -602, 5399, -9064, 497, -14296, -17757, -18527}
}
, {{384, -1227, -363, 1333, -3231, 2260, -305, -2244}
, {1256, -761, -1408, -4205, -1283, -1316, -3531, 476}
}
, {{545, 2732, -2989, 2356, -766, 3114, 4485, 5052}
, {-2089, -2074, -2575, -590, 1268, 4996, 3881, 5191}
}
, {{-2361, 266, 7894, 2822, 2050, 6279, 8372, 6852}
, {-2264, 2676, 12724, -6702, -1451, -8506, -8364, -5500}
}
, {{-68, 2340, 4115, -4981, 1076, -3567, -1660, -9724}
, {-812, 227, 3833, -5873, 2907, -7669, -5035, -2908}
}
, {{-3177, -1799, -1099, -3075, -1731, 711, -953, -4610}
, {-1250, 328, -2385, 1234, 1933, -784, 852, -1732}
}
, {{1196, 1036, 904, -2184, -1054, 1117, -2471, -2948}
, {1472, -1946, -2617, -308, -2436, -2418, -2464, -598}
}
, {{2823, -1301, 911, -56, -2420, 4673, 6409, 2246}
, {820, -1139, 5691, 1173, 978, 1215, 3530, -812}
}
, {{1672, 2389, 11702, -6190, -1210, -8360, -7998, -9110}
, {-2198, -448, 9133, -11243, -637, -17145, -15147, -14595}
}
, {{-658, 1332, 811, 1275, -1353, 1943, -361, -2009}
, {2095, -2895, -1783, -1406, -2211, 449, -307, -867}
}
, {{-831, -518, 2092, -2047, -1322, -1030, -3211, 244}
, {1950, -805, -2335, 1781, -2004, -2090, -2422, -1083}
}
, {{94, 2274, 1195, 1658, -915, 5359, 1181, 5057}
, {-2041, 1866, -4400, 522, -2806, 8921, 7867, 8305}
}
, {{816, -1345, -2895, -2376, 93, 1675, -544, -590}
, {1962, -2158, -1402, 1388, 236, -2982, -439, -1005}
}
, {{-2182, -217, 1730, 4021, 1411, 6896, 4401, 4647}
, {-2525, 2902, -4717, 4797, -2040, 3477, 4341, 5582}
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

typedef int16_t max_pooling1d_134_output_type[POOL_LENGTH][INPUT_CHANNELS];

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
#define INPUT_SCALE_FACTOR 13
#define OUTPUT_SCALE_FACTOR 13
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


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

typedef int16_t conv1d_110_output_type[CONV_OUTSAMPLES][CONV_FILTERS];

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
#define WEIGHTS_SCALE_FACTOR 13
#define BIASES_SCALE_FACTOR 13
#define TMP_SCALE_FACTOR 13
#define INPUT_SCALE_FACTOR 13
#define OUTPUT_SCALE_FACTOR 13
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


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

  static q15_t bufferA[2*INPUT_CHANNELS*CONV_KERNEL_SIZE];
#if INPUT_CHANNELS % 2 == 0 && CONV_FILTERS % 2 == 0
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_fast_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_fast_nonsquare(
#endif
#else
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_basic_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_basic_nonsquare(
#endif
#endif
                                      (q15_t*)input, //Im_in
                                      INPUT_SAMPLES, //dim_im_in_x
                                      1, //dim_im_in_y
                                      INPUT_CHANNELS, //ch_im_in
                                      (q15_t*)kernel, //wt
                                      CONV_FILTERS, //ch_im_out
                                      CONV_KERNEL_SIZE, //dim_kernel_x
                                      1, //dim_kernel_y
                                      ZEROPADDING_LEFT, //padding_x, left and right must be equal
                                      0, //padding_y
                                      CONV_STRIDE, //stride_x
                                      1, //stride_y
                                      (q15_t*)bias, //bias
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - BIASES_SCALE_FACTOR, //bias_shift
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, //out_shift
                                      (q15_t*)output, //Im_out
                                      CONV_OUTSAMPLES, //dim_im_out_x
                                      1, //dim_im_out_y
                                      bufferA, //bufferA
                                      NULL //bufferB, unused
                                      );
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#endif
#elif !defined(ACTIVATION_LINEAR)
#error "Unsupported activation with CMSIS-NN"
#endif


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


const int16_t  conv1d_110_bias[CONV_FILTERS] = {-137, 61, 616, -94, -1492, -525, -310, -492, -1688, -1250, -2203, -276, -1890, -372, 215, -494, 1004, -155, -1805, -1179, 305, 2749, -1357, -1113, 66, -1087, 1033, -444, 1035, 1101, 2611, -897}
;

const int16_t  conv1d_110_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{{-1051, 26, -1253, 622, -490, -1134, 583, -1323, 1474, -77, 1142, 324, -19, -1222, -1399, -1465}
, {-720, 27, 517, 1168, -2347, -377, -1608, 955, 600, -2379, -2034, 640, -55, -1739, 1356, 1798}
}
, {{-3825, 1553, -1993, 634, 2282, 7588, -3967, 1080, -1453, 2828, -840, -1320, -660, 2744, 861, 3894}
, {183, -5061, -182, 1461, -4114, -6057, 7119, -1529, 493, -2835, -13001, 1165, 379, -5761, 67, -3724}
}
, {{1825, -907, 6929, 986, 2100, -4440, -346, -594, -1910, 2102, -2236, 642, -1292, 1523, -1160, 1700}
, {-2684, 2996, 3646, -59, 372, -4972, 3298, -1232, -993, -656, -5643, 696, -1681, 953, 748, 3413}
}
, {{-7030, -2996, -220, 1736, -4070, -7786, -6407, -2005, -71, -587, -9998, 2566, 1876, -4614, 920, -1238}
, {-6432, 3362, 4266, -1543, 1503, 4921, -4531, -1104, 579, 2644, -9182, 1886, -1007, 2933, -730, 3173}
}
, {{1133, -3220, 1227, -689, -3170, 1365, 940, -639, -128, 237, -2373, 673, -1968, -1728, 1971, -1152}
, {1070, -1691, -1659, -1868, -1131, -1893, 1850, -1383, 1306, -3399, 582, -437, 1606, -2427, -412, -1763}
}
, {{1060, 1598, -796, -442, -2436, -1951, 402, -124, -1907, 698, 1209, -1646, -1500, -1344, 635, 308}
, {-1103, -2473, 596, -1622, 922, -687, 1462, -1385, 1502, 14, -3005, -1692, 640, -556, 1655, -2355}
}
, {{776, -2380, -159, -1839, -1745, -1764, -1488, 1312, -844, 724, 158, -1822, 1938, -1114, -1224, 997}
, {-799, -2330, 844, 612, 732, -1777, -1740, 170, -1093, -292, 1186, 1921, -900, 203, -1626, -994}
}
, {{607, -1608, -1634, -164, -1514, -621, -1679, -549, -1102, 741, -457, 1332, 1322, -358, 1751, -1124}
, {627, 5, 1638, -276, -453, -1681, 381, -1950, -1135, 592, 195, -869, 1444, 1418, -1546, -2117}
}
, {{3246, -1989, -36, -859, -541, -2618, -309, -946, 196, -1192, -2550, -426, -464, -1448, 1112, -682}
, {2583, -2326, -539, -891, 1420, 780, 196, -860, -1405, 248, -39, -1598, -1941, -1993, -118, 985}
}
, {{-494, 933, 2107, -1598, 85, 1284, 1263, -4059, 123, -1732, -1553, 2279, 487, -777, -989, 642}
, {1523, -134, -788, -266, -1992, 128, 1094, -1532, 467, 1643, -3401, 1342, -1081, -1327, -854, -42}
}
, {{-3941, -265, -4592, -2068, -1131, -4284, 2746, -1942, 1679, -2450, -4344, 356, 280, -775, -1541, 643}
, {-1245, 723, 3377, 12, 3033, -9852, 1418, -2416, -52, 2092, -3710, 1628, 2404, 1624, 322, 4456}
}
, {{500, -2128, 1202, -1112, -1018, 126, 1642, -2146, 1408, -1045, -1157, 298, -1988, -1452, 450, -2504}
, {1707, 516, 1645, 1307, 408, 469, 415, -2135, -1349, -3555, -1993, -932, 935, 150, -1063, -265}
}
, {{-1104, -2741, -644, 460, -2702, 438, -914, 1732, 1829, -2202, -1564, -297, -828, 417, -782, -917}
, {368, -1586, -83, 1928, -925, -1612, 878, -1015, 584, -4398, -1312, -1493, 168, -2213, 1655, -969}
}
, {{-203, 538, -1171, 1365, -1639, -1847, -1886, -2373, -2044, 233, -1000, -377, 1347, -1697, -426, -767}
, {-1326, -87, -220, -2029, 1273, -27, -434, -107, 65, 859, -227, 439, 254, -2494, -377, -500}
}
, {{-1404, -1361, -5835, 35, -885, -2136, -4384, 996, -1125, -2459, -8697, -1177, -379, 642, -1671, 644}
, {-10613, -2331, -363, 554, -1497, 7064, 4363, -2374, -163, 1755, 6986, -1917, 394, 554, -952, -1925}
}
, {{219, -2471, -1688, 615, -2771, 69, 1681, -490, 1610, -2308, 32, 218, 268, -552, -929, 159}
, {-568, 366, 887, 933, -2066, -1567, 492, -447, -1398, 227, -546, 474, 1072, -1297, 446, -174}
}
, {{-10822, 2460, 127, 1622, 846, 1995, 2888, 1295, 1875, 651, -168, 596, 2918, 758, -333, 3704}
, {-7121, -1721, -1967, 1220, -4644, -8595, -1517, 2455, 1472, -4813, -5316, 382, -1422, -3944, -1254, -6285}
}
, {{-5760, 2153, -10912, -501, 1513, -1842, -5423, -2317, 1313, 2108, -6004, 1128, 635, 474, 96, 2377}
, {-722, -8523, -2967, 81, -7768, 5905, 70, 871, -131, -1963, 5683, -664, -741, -8424, 95, -6871}
}
, {{2128, 343, -2320, -1125, -198, 504, 74, 1356, -1326, -1936, 160, 2502, -1000, -1690, 127, 799}
, {2614, -4, -826, 627, -3037, -2719, -1894, 734, -1861, 1527, -1450, 1633, 56, 535, 1750, -1851}
}
, {{-1299, 824, 326, 1705, 60, 14236, -841, -1563, -143, 1908, 750, -886, 2940, 2118, 347, 1612}
, {3646, 938, 4185, -94, -940, 5908, 305, 1012, -2495, -267, -3320, -1956, 1996, -609, 505, 456}
}
, {{-5523, -4856, 88, 825, -2621, 6299, 5506, -1451, 1448, 458, 5070, -490, 926, -4240, -756, -3164}
, {-707, 1167, -8973, -1271, 2380, -4065, 843, 1078, -670, 729, -6039, -1234, -125, 1688, -268, 2359}
}
, {{-641, -2464, -6593, 1718, -5034, -10264, -275, 305, -1566, -4114, -3254, 1864, 2778, -4164, 1540, -8705}
, {-6532, -57, -15058, 565, -463, 852, 1260, 763, 1285, 1143, -529, 579, 2144, 1211, 655, 287}
}
, {{1511, -405, 1139, 789, -2361, -3149, 536, -1514, 255, -10, -1106, -1399, -1436, -2177, -1678, -501}
, {-529, -210, 914, 2611, -3240, -753, 1672, -2603, -1138, -3533, -2310, -833, -748, -3685, 1372, 264}
}
, {{-563, -677, 1189, 908, -2312, -2247, -546, 438, 1439, -915, 21, -755, 2026, -2374, -1441, -1122}
, {-2127, 290, -1684, -1800, -869, -1525, -708, -2384, -410, -1882, -164, 1948, -1933, 1375, 1377, 229}
}
, {{4113, 1459, 2585, -1867, -1849, 400, 1287, -2063, 1803, 2286, -8546, -1446, 407, 90, 317, -670}
, {-3439, 1472, 3993, 1720, 1885, -4229, 1360, 786, 2042, 1897, 3262, -76, 1278, -626, -1273, 2191}
}
, {{-513, -2338, 1271, -596, -1128, 639, -1351, 1626, -2228, -729, -920, 2350, 1929, -1013, 618, -138}
, {-1934, -2516, -1595, 426, 343, 1291, 1917, -1094, -462, -2829, -2462, 1703, -937, -276, 906, -1593}
}
, {{-2621, 1421, 4762, 104, 1360, -1452, -551, -1015, -52, 554, -8028, -2216, -1752, -609, 1395, 2305}
, {-7139, 1127, 716, 1430, -1200, -4029, 2064, -1562, -995, 1419, -8755, -1948, 269, -1293, 1856, 1602}
}
, {{-1199, -1207, -809, -1491, -881, -42, 582, -2104, 555, -2726, -2000, 268, 1121, -2480, -1184, -1586}
, {1456, 733, 269, -730, 888, -303, -483, 134, 743, 206, -1794, 319, -1010, -610, -1309, -2244}
}
, {{-22, 661, -1770, 1024, 826, -8729, -3251, -2018, -485, -2109, -5858, 1813, -69, -3107, 400, -47}
, {-9616, 1491, 1080, -691, 806, -5124, 255, 931, -2317, 213, -13162, 1909, -158, 1639, -397, 3099}
}
, {{-1211, -3746, -8899, 588, -4127, -2485, -3946, -1280, -954, -5916, 6266, 1060, -1105, -4711, 1074, -3544}
, {1947, -4105, -5998, 1562, -3777, -2076, -9663, -630, -1759, -6622, 6048, -1047, 2756, -6190, 154, -5319}
}
, {{4647, -1576, -1992, -89, -551, 1542, -1575, -2839, 410, -1659, 5531, 119, 2121, -4850, -974, -798}
, {16587, -571, -6064, -1325, -2289, 2104, -576, -2271, -1511, 860, 337, -1513, -179, -5479, 902, -1705}
}
, {{-7631, -1970, -4187, -693, -2575, 6059, -2159, -35, 41, -492, -6328, -2120, -3083, -1237, -286, -2207}
, {-3224, 1426, -389, -1330, 2160, 18245, 118, -76, 615, 3393, 584, 282, -575, 866, -1184, 1479}
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

typedef int16_t max_pooling1d_135_output_type[POOL_LENGTH][INPUT_CHANNELS];

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
#define INPUT_SCALE_FACTOR 13
#define OUTPUT_SCALE_FACTOR 13
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


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

typedef int16_t conv1d_111_output_type[CONV_OUTSAMPLES][CONV_FILTERS];

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
#define WEIGHTS_SCALE_FACTOR 13
#define BIASES_SCALE_FACTOR 13
#define TMP_SCALE_FACTOR 13
#define INPUT_SCALE_FACTOR 13
#define OUTPUT_SCALE_FACTOR 13
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


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

  static q15_t bufferA[2*INPUT_CHANNELS*CONV_KERNEL_SIZE];
#if INPUT_CHANNELS % 2 == 0 && CONV_FILTERS % 2 == 0
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_fast_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_fast_nonsquare(
#endif
#else
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_basic_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_basic_nonsquare(
#endif
#endif
                                      (q15_t*)input, //Im_in
                                      INPUT_SAMPLES, //dim_im_in_x
                                      1, //dim_im_in_y
                                      INPUT_CHANNELS, //ch_im_in
                                      (q15_t*)kernel, //wt
                                      CONV_FILTERS, //ch_im_out
                                      CONV_KERNEL_SIZE, //dim_kernel_x
                                      1, //dim_kernel_y
                                      ZEROPADDING_LEFT, //padding_x, left and right must be equal
                                      0, //padding_y
                                      CONV_STRIDE, //stride_x
                                      1, //stride_y
                                      (q15_t*)bias, //bias
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - BIASES_SCALE_FACTOR, //bias_shift
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, //out_shift
                                      (q15_t*)output, //Im_out
                                      CONV_OUTSAMPLES, //dim_im_out_x
                                      1, //dim_im_out_y
                                      bufferA, //bufferA
                                      NULL //bufferB, unused
                                      );
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#endif
#elif !defined(ACTIVATION_LINEAR)
#error "Unsupported activation with CMSIS-NN"
#endif


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


const int16_t  conv1d_111_bias[CONV_FILTERS] = {161, -416, -1572, -969, -668, -414, -1648, -573, 2259, -456, -1612, -2082, -695, 2445, 1927, -2023, -1053, 945, -469, -192, 306, -244, -244, -342, 122, -716, -510, -361, -2027, -537, -693, -1724}
;

const int16_t  conv1d_111_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{{215, 12460, -14, -5082, 456, 1203, 373, 1319, -1258, -1423, -676, -491, 682, 1614, -3325, 1422, 9742, -3226, -424, 2848, -12700, -1667, 2319, 336, -430, 113, 40, -1197, -235, 2805, 5110, -2935}
, {-1661, 6644, -1694, 1252, 258, -1038, -667, 1643, 3469, 694, -2545, -1203, 114, 91, -2060, -102, 5089, -11780, 432, 1341, -2116, 3380, 2891, 673, -1534, -84, 1633, 2480, 1691, 3024, 3247, 5042}
}
, {{1222, -1703, 253, -584, 1101, 1501, -237, -1500, -1720, -699, -2622, -368, 583, -442, -1727, -867, -2253, -341, -580, -564, 948, -1152, -685, -40, -1033, -438, -1223, -164, -1390, 200, -577, -586}
, {-1746, 1191, -1737, -652, 914, 1712, 238, -93, 460, 1207, 723, -2096, 365, -2054, -1989, -1762, -712, 120, 834, -1721, -514, 384, -114, 623, -244, -371, -543, 1165, -900, 97, -2285, 208}
}
, {{484, -2983, -1226, -1639, -596, -1843, -2107, 821, 984, -300, -2781, -548, -1083, -2020, -1969, 225, -2529, -4768, 756, -443, 299, -2856, -1185, -1662, -3096, 596, -1202, -2208, -62, -20, -266, -1351}
, {-355, -2918, -1595, -13, 916, 427, -1572, -875, 852, -1225, -1958, 1078, -1752, -1014, -3718, -740, -561, -2207, 1390, -1094, -2898, -394, 776, 408, -2611, -2216, -1695, -1528, 98, -1135, -1876, -2952}
}
, {{-346, -435, -384, 5099, 150, -554, -149, 26, -2589, -1141, -6473, 141, 1110, -287, 63, -1121, 1898, 4454, 1102, -891, -1135, 18970, -1429, -1232, -4068, 700, -2715, 598, 1041, 4197, 2759, 1518}
, {769, -1449, -336, 2862, 297, 1168, -1666, -700, -1216, -1179, -2773, -584, 170, -325, -4956, -1700, -3769, 1070, -502, 1901, 2030, -13216, 1299, 1465, -1942, -388, 1149, -365, -873, -5112, -6258, 3778}
}
, {{-1928, -2506, -3250, -2724, -864, 417, 773, -955, -1211, 1793, -3158, -818, -712, 392, 800, 567, -2984, -2048, -1032, -3137, -1598, -3904, 1681, 503, -2023, 1335, -1372, 1130, -2546, -2142, -3627, 607}
, {677, -1271, -2465, -2202, 219, -1093, -457, 1442, 1601, 179, 149, 939, -290, -1706, 3064, -770, -1820, -1707, -26, -1212, -1590, -893, -84, 631, -1949, -1301, -1674, -1386, -1238, -2530, -3438, -1856}
}
, {{-614, -1952, -1720, 2019, 585, -1549, -1966, 870, 1343, -1345, -313, 282, 1075, -568, 517, 696, -598, -1390, -1563, -1553, -35, -1934, 301, -292, -1322, -1092, -714, -881, 1428, -1070, -366, 756}
, {-234, 909, 623, -1054, 808, -295, 619, 341, -379, 847, -1219, -385, 563, -191, 651, 1099, 552, -607, -1426, -2417, -1858, 694, -1372, -9, -669, 1311, -1658, -105, -9, 315, 805, -8}
}
, {{1359, -9640, -344, -7227, 315, -1053, 1490, -336, 305, -740, -6088, -130, 497, 1429, 1666, 1280, -8720, 5983, -350, -1828, 5623, -11243, 492, 2829, -3242, 1588, 1281, 1566, -216, 10566, 8238, 1392}
, {-1647, -1093, 2104, 4487, 486, -195, -665, -957, -249, 2621, 2785, 715, -895, -1083, -9233, 922, 1357, -4847, -391, -4870, 2062, -9767, -365, -60, 1237, -2087, 1074, -2252, 4035, 3575, 2002, 11}
}
, {{-32, -351, 304, -772, -674, -1574, -124, -312, 1394, 109, 591, -1047, 91, 813, -1374, -1426, -1547, -295, -1376, -1394, 43, -374, -846, -1480, -1424, 531, -82, 280, 482, 1139, -1433, 723}
, {-952, -1945, -2162, 113, 1118, 1234, 495, -289, -645, -1548, -1578, 227, 963, -262, 1003, -1492, 1028, -1260, 462, -406, -475, 909, 417, -1643, -681, -647, -2491, 331, -2311, -936, -1318, 412}
}
, {{1388, -4438, 1388, -10672, -128, 710, 323, -795, -1493, 716, 150, 768, 1600, 695, -3325, 841, 4318, 3996, 915, -359, -635, -897, -713, 1059, -240, 1023, 3228, -192, 3649, -3999, -2509, -7567}
, {-38, 2974, 926, -5623, -1335, 903, -109, 1499, 1059, -1070, -6698, 977, -97, -1303, -7861, -106, 3159, 349, -1999, -3004, -3578, 2859, -1245, -1403, 1107, -965, 2673, 739, 1321, -3827, -2962, -10704}
}
, {{-416, -4042, 3207, -2122, 911, -864, 733, -1562, 172, 798, -2258, 121, 571, -101, 109, 777, -2164, 1143, -1320, 1869, 493, -87, -642, -1338, 2159, -98, 2317, -902, 1360, 1189, -772, -1871}
, {-1248, -7495, 2978, -1635, 737, 1805, -1108, -672, 558, -423, -591, -812, -108, -874, -3003, 611, -3069, -4299, -1867, 872, 1004, -3388, 1316, 1785, 2177, 412, 2622, 1274, 697, -3863, -11406, -192}
}
, {{1410, 5350, 3728, 4573, 642, 342, -1031, 933, 1231, 1073, 2838, 1272, -1439, 323, -210, -1121, -5006, -4614, 520, 1668, -4348, -5420, 3148, 1029, 1583, 173, 3380, 848, 2708, 9540, 918, -7556}
, {-1310, -3220, -1923, 3006, 141, -74, -1258, -265, -2410, -512, -3800, -31, -875, 914, -2741, -722, -11552, -746, 981, 189, 2380, -24957, 689, 848, -1079, 1140, -1429, 1714, 217, -11538, -21542, -6257}
}
, {{-296, -1278, -804, -415, -1180, 1788, -687, -1690, -64, 586, 881, -1684, -66, -491, -1090, 1631, -2572, -1226, -1215, -142, -1021, -25, -1511, -1394, -2967, -278, -1346, 299, -1531, -1929, -287, -2070}
, {624, -1599, -457, -2285, 964, -927, -613, -219, 78, -723, -2188, -1164, 1150, -554, -1400, 1833, -2806, -537, 1154, -2710, -1140, -1835, 18, 1009, -1808, 567, -1966, -269, -864, -1514, -1301, -1341}
}
, {{-84, -919, -1575, -49, -1334, -1139, 74, 956, 637, -144, 423, -1336, -760, -1292, 78, 1531, -875, 664, 405, -1144, 1047, -2422, -1325, 502, -399, 1782, -1220, -2026, 1096, -1324, 767, -1230}
, {-1696, -1274, 40, -1475, -213, 752, 388, 1006, 1325, -673, 773, 1487, -2053, -1498, 332, -1548, -1773, 244, 3542, -1201, -1279, 542, 772, -1209, 603, -627, -481, -1133, -878, 1460, -1629, -2963}
}
, {{-1884, -9937, -610, -856, -132, -1846, 281, -356, -494, -390, -6270, -1927, 48, 902, 1320, -1156, -2943, 2254, -407, -1665, 3290, -2995, -1205, -984, -2859, 2067, -2419, -616, 2402, -1577, -5910, -5142}
, {220, 5472, -11405, 997, 699, 39, -811, -1255, -107, -1151, -17245, 1284, -1285, -232, -63, 139, -9494, 1423, 40, 4234, 841, -12017, 1459, 412, 2128, 306, -11631, -100, -16891, 7008, 1184, 4811}
}
, {{1551, -9311, -351, -2409, 1177, 1168, 61, 1370, 1223, -821, -859, -1157, 863, -1724, 1754, -768, -1288, 7326, 1319, -2844, 2931, 1089, -325, 395, 696, -182, 1761, 2472, 2694, -5130, -861, -7290}
, {-317, -7017, -195, -2610, 806, -188, 1889, -65, -1120, 2066, -5496, 239, 999, 878, -5874, 1028, 615, -2318, -167, -6425, -3200, 2691, -682, 186, -1231, 2557, 1253, -909, 2047, -6126, 1328, -8038}
}
, {{-1966, -1090, -186, 697, -1077, -1263, 238, 840, 26, -764, -570, -2088, 156, -1562, -1313, 1459, -1019, -3745, -965, -4605, 2281, -2288, -1664, -501, -2208, -783, -1057, -789, -591, 1499, 1726, -2071}
, {894, -521, -1238, -2015, 15, 921, -1068, 968, 1868, 928, -186, -13, -1225, 138, -3212, -1402, -417, -1391, -164, -699, -3843, -812, 2125, 957, -801, 152, -1650, 651, -1068, 80, 2109, -3687}
}
, {{-1410, -150, -1328, -233, -715, 5, -478, -1062, -305, -1386, 302, 1114, -932, 820, -199, 346, -2707, -1504, -230, -862, -471, -2419, 416, 973, -2568, 1133, -1278, -411, -593, -1266, 1197, -2093}
, {-755, -3021, -1626, -1798, 23, 118, -1251, -1378, -584, 535, 1377, 1354, 65, -50, 693, 794, 110, -1822, 43, -190, 467, 25, 801, 1207, 729, 1201, -273, -570, -495, 19, 1429, -1923}
}
, {{176, -136, 1850, -4217, -469, 1454, 742, 1467, -1954, 502, -2831, 861, -151, 31, -2760, -12, 2238, 1788, -880, 2047, -1449, -5173, -650, -1426, 730, -2632, 233, 1688, 1484, -3093, -11215, 3466}
, {-977, -334, 2534, -3213, -916, -94, 560, 1063, 365, -690, -1124, -609, 709, 1040, -3212, 869, 2772, 1024, 1675, 727, -436, -3150, -1417, 1676, 2659, -217, 1829, 617, 2256, 2622, -5566, 2894}
}
, {{-2103, -1646, 885, 469, -1552, 458, -420, -714, 2249, 979, -1445, -1386, -189, -1634, -746, 297, -787, 795, 144, -1138, -448, -1863, -408, -692, -2207, 1157, -878, 1106, 1046, -1016, 404, 309}
, {914, -1899, -611, 534, -656, -58, -687, -510, -1084, -163, -103, -657, 543, -1376, -7, -1561, -825, 920, 899, -147, 198, -379, -1698, 663, -1842, -1047, 89, 77, -1402, -357, -739, -2036}
}
, {{-2318, 1017, -536, 689, -1003, -184, -713, 1717, -1375, -806, -1804, -981, -838, -1222, 293, 1202, -879, -977, 1538, -1130, -785, -1850, -470, -1073, -850, -1130, -2010, -8, -1507, 821, -241, -1828}
, {1143, -252, -818, 1260, 662, -570, -14, 1659, 1195, 170, -1419, 748, 1416, -1950, -113, 221, -413, 1039, -209, 101, 218, -560, -215, -414, -2508, 686, -275, -2291, 257, -2049, -394, -1699}
}
, {{-902, -3513, 2223, -3352, 1391, 815, -541, 1273, 174, 280, -2127, 1347, -59, 1363, -1364, -1043, -1271, 3027, 1377, -1141, 2931, 3984, -19, -276, -754, 1240, 2248, 35, 2084, 7414, 1398, -1008}
, {658, -3247, 1177, -1039, 137, -411, -383, -796, -1057, 70, -1020, 2099, 1296, -1016, 2163, -82, -2104, 4195, 8, -1217, -4586, -1693, 1163, 1509, 982, 1656, 825, 430, 2711, -4137, -5853, 1324}
}
, {{-1139, -1059, -1587, 670, -803, -140, -136, -227, 1339, 1630, -737, -1904, 1499, 1105, -2024, -1039, -1659, 948, 1142, -2002, 342, -343, -566, 1216, -2235, 1, 875, -669, 1356, 610, -883, -1330}
, {-1277, -817, 82, -794, -1088, 370, 1106, -101, 379, 485, -2291, 856, -1131, 219, 170, 42, -265, -1089, -535, -1431, 378, -913, 551, 814, -2010, 404, -994, 1307, 674, 699, -755, 1009}
}
, {{-918, 157, -513, 568, -1321, 1075, 1055, -98, -717, -1559, -1168, 465, -915, 875, -695, 408, -2347, 473, -596, -159, -962, 511, -559, 253, -388, -960, 69, 808, 167, 806, -1806, -61}
, {-920, 690, -589, 292, -1286, -399, -1453, 100, 686, 84, -942, -1259, 94, 826, 241, -1833, -2363, -469, -1261, -1753, -154, -2296, -1410, 896, -1776, 1980, 684, 1035, -1994, -2034, 365, -1160}
}
, {{-603, -586, -689, -109, -673, -607, -1972, -281, 1154, -121, -200, 586, 899, -1471, -194, -431, -3333, -1494, 44, -3318, 1047, -1804, -841, 498, -717, 857, -2875, -1918, -1246, 494, -1562, -1676}
, {365, -2634, -2120, -1739, 1082, -758, -469, -1, 320, 1079, 645, 890, -100, 166, -2022, 1061, -3389, 80, 944, -1785, 208, -2515, 1872, 1767, -1394, 256, -377, -1062, -426, -265, -681, -222}
}
, {{204, -1474, 627, -2474, -648, -1159, -2148, 1262, 954, -2643, -2452, 662, -612, 1050, -5794, -57, -1667, 625, -542, -2426, -3516, -7396, -839, -1407, -2416, 2074, -156, 626, 914, 12276, 1756, -5086}
, {436, 4506, 573, 3746, -1069, -797, -582, -1388, 171, 104, 2626, 1250, 562, -1474, 285, 1137, 3728, -3203, 1193, 1219, -2412, -4833, 101, 1097, 1919, 443, 3875, -778, 4311, 4834, -896, 2018}
}
, {{-933, 12285, -2691, -4, -1220, 1627, 808, 1379, -1534, 1188, -1600, -822, -1260, 266, -6654, -659, 10112, -3909, 1519, 2697, -5896, 411, -84, -157, -1964, -895, -1013, -2648, -3116, 2649, 3589, -3587}
, {-507, 7008, -1282, 5313, 1432, 858, 981, -1038, 981, 1446, -684, -1359, 1044, -90, -5250, 361, 8518, -8974, -1094, 997, -1707, 5753, 1958, -552, 1152, 865, 1444, -343, 2201, 1379, 1956, 3960}
}
, {{812, -3196, -264, -1567, -540, -1611, -74, 1213, -1480, -912, -641, 22, 1586, -1691, -1570, 1350, -1280, -1076, 1538, -794, -1248, -2475, -742, 1360, -1032, 1191, -633, -907, -5144, 432, -1599, -2972}
, {-2327, 432, -1507, -878, -175, -1334, 666, -567, 591, -412, -2998, 664, -716, 613, -1757, -882, 279, 404, 919, -862, -995, -179, -137, -1775, -2354, 1819, -505, -941, -2448, 829, -1873, -3224}
}
, {{862, 845, -2553, -896, -1723, 346, 828, 1462, 329, 1104, -1386, 1221, -125, -1379, 1162, -1847, 503, -126, -1137, -411, -1603, -1636, -1088, -904, -1194, -138, -2245, -650, -345, -464, -1551, 614}
, {-1106, 902, -1370, 1334, -1300, 425, 859, 1030, -917, -2023, -1126, -1146, 1090, 933, -1104, -834, 585, -535, -1165, 117, -21, -2477, -461, -1273, 61, -920, -1412, -642, -459, 392, 144, -1976}
}
, {{-1492, -6843, -5239, 1978, -1419, 100, -1962, 1678, 356, -1501, -8276, -1272, -1357, -792, -595, 1211, -7796, -419, 1589, 4583, 9634, -8139, 407, 1330, -3729, 284, -5293, -1593, -4458, 435, 1223, 3403}
, {-330, -3269, -6119, -4447, 598, -654, -1065, -1107, 2825, 124, -6273, 634, -819, -2072, 499, -1295, -4124, 7842, -567, -3468, -5629, -1801, -1361, -1030, -5252, -1513, 1203, 653, -1100, 6962, 4924, -3244}
}
, {{-1929, -2139, 455, 777, -479, -1663, 674, -797, -815, -755, 603, 1272, 1474, -382, -128, 1133, 1010, 423, 174, -458, -824, -2372, -227, 526, -672, 704, -1259, -827, -1663, -2315, 792, -1447}
, {-1113, -479, -200, -1788, -479, 1045, 1425, 248, -219, -1269, -855, 151, 1599, 1119, 597, 1051, 451, -233, -80, -1940, -1619, 26, 976, 1323, -2625, -1364, -768, -1373, -2156, -482, -229, -2435}
}
, {{225, -2707, -2773, 1984, 209, -1747, -1711, 253, -491, 255, -1789, -2191, -1551, 318, 584, 435, -1389, 660, -1690, -2500, 50, -190, -728, 1026, -938, 2046, 392, -12, -598, 1752, 433, -1605}
, {-1397, -327, -1471, -1160, -1881, 534, -169, 809, -93, 300, -804, -359, 497, -2294, -298, -1466, 922, -673, 1539, -828, -340, 867, 1014, -203, -391, -1311, -47, 323, 1132, -1260, -3781, -1701}
}
, {{-741, -1491, -2460, 5731, -1619, -97, 339, -1328, -2982, 685, -575, 38, -946, -658, -8020, -529, 2399, -908, -225, -1525, 1989, 18117, 581, 896, -4041, -610, -1695, 181, -1182, 4886, 6017, 779}
, {972, 185, 58, 2576, -3375, 653, 796, 435, 11, -728, 3924, 713, 3352, -84, 2213, 88, -5883, 1750, 479, 632, -309, -6665, 978, 249, 555, 1451, 1187, -2215, -1573, -9632, -7565, 3585}
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

typedef int16_t max_pooling1d_136_output_type[POOL_LENGTH][INPUT_CHANNELS];

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
#define INPUT_SCALE_FACTOR 13
#define OUTPUT_SCALE_FACTOR 13
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


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

typedef int16_t flatten_17_output_type[OUTPUT_DIM];

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

#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t

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

typedef int16_t dense_17_output_type[FC_UNITS];

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
#define WEIGHTS_SCALE_FACTOR 13
#define BIASES_SCALE_FACTOR 13
#define TMP_SCALE_FACTOR 13
#define INPUT_SCALE_FACTOR 13
#define OUTPUT_SCALE_FACTOR 13
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


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

  static q15_t bufferA[INPUT_SAMPLES];
#ifdef WITH_CMSIS_NN
  arm_fully_connected_q15(
#elif defined(WITH_NMSIS_NN)
  riscv_fully_connected_q15(
#endif
                             (q15_t*)input,
                             (q15_t*)kernel,
                             INPUT_SAMPLES,
                             FC_UNITS,
                             INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - BIASES_SCALE_FACTOR,
                             INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR,
                             (q15_t*)bias,
                             (q15_t*)output,
                             (q15_t*)bufferA);
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q15((q15_t*)output, FC_UNITS);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q15((q15_t*)output, FC_UNITS);
#endif
#elif !defined(ACTIVATION_LINEAR)
#error "Unsupported activation with CMSIS-NN"
#endif


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


const int16_t dense_17_bias[FC_UNITS] = {-4928, 7666, -3711}
;

const int16_t dense_17_kernel[FC_UNITS][INPUT_SAMPLES] = {{198, 290, 2260, 5355, -1911, 101, -6643, -514, -392, 1661, -5876, -1387, 69, -2413, -3440, 117, -106, 4236, -505, -106, 1100, 224, -1264, 494, -4754, 1135, 1199, 400, 6119, 653, -147, 3314, 2568, 481, 74, 5245, -747, 555, -9538, -42, 1776, -861, -5780, -764, -1122, -6445, -1235, 177, -1128, 2544, 202, -823, -246, 127, 1145, -722, -1799, 2635, 679, -464, 5486, 147, 650, 5822, 1845, -280, 270, 7017, -557, -958, -9268, 115, 378, 196, -3916, -768, 2542, -7800, -2281, 2085, 1381, 717, -143, -801, 1445, -422, -608, -854, -2201, 1159, -334, -1315, 11748, 597, 249, 2223, 2192, -171, 633, 4345, -856, 762, -8832, 102, -814, 1820, -3436, -1850, -1074, -3719, -3179, -1142, -131, 363, 429, -289, 1398, 1091, 315, 643, -1600, 827, -1682, 83, 5492, -75, -1080, 5802, 1992, 940, 2330, 6665, -1595, -1482, -6704, -614, 3925, 218, -7286, -2052, -361, -2062, -3175, 1654, 600, 2044, -606, -49, 671, -446, 1094, 484, -268, 1820, 1445, 944, 131, 149, -1306, 4287, -664, -1536, 1597, 4093, -1912, 933, -9394, -141, 438, 1136, -4561, 501, -381, -1044, -3917, -811, -485, 1023, 848, 682, 1915, -1130, -396, -251, -278, 2163, -1391, -694, 10740, 445, -1566, 2949, -285, 743, 1666, 8266, -1952, 35, -7237, -1422, 3586, -999, -6235, -1854, 898, -4754, -4582, 813, 624, 2428, 7, 52, 1102, 1182, -20, -646, -2902, 3948, -1270, -962, 5153, -496, 950, 2291}
, {1813, -730, 57, -5319, 3024, 432, -962, 56, -7584, -4288, 58, 172, 1021, 3571, -2226, 1419, 609, -2326, -1527, -322, 2036, -1570, 367, -733, 6160, 3438, -516, 204, -3648, -1076, -181, -2214, 3347, 655, -105, -2729, -135, 162, 2639, -27, -5541, -2050, 1631, 1496, 1872, 7338, -3633, -421, -239, -354, 361, 112, 1712, 743, -1419, -703, -165, 3675, -1269, -810, -1838, -502, -17, -6786, 2164, -426, -1938, -1493, 2002, 761, 2284, 1537, -4692, -594, -1294, 216, 1097, 2023, -1257, -547, 95, -1314, -1275, 674, 340, -1307, -236, 1246, 3584, 3247, -14, 113, -220, -702, -1445, -5224, 2816, -856, -620, 550, 1203, 815, 1695, 563, -3797, -4195, 2797, 2370, -330, 4080, -1998, 303, -1952, -665, 366, -20, -2622, 326, -23, 373, 1012, 2485, 655, -153, 3040, 130, -727, -5836, 6128, 277, -2352, -726, 370, 845, -211, -71, -3762, -1203, 807, 1931, 1045, 4466, -278, 664, -1060, -1994, 845, 315, 648, -934, 781, 493, 1626, 4093, -1568, 386, 5180, -1341, 893, -4001, 3470, -1414, -244, -2041, 2683, -1438, 1554, 174, -5183, -2030, -2402, 2226, 1674, 4671, 112, -890, -1131, -1517, 426, -314, -1466, -484, 102, 291, 2502, 2286, -428, -745, -2935, 1260, -197, -3063, 3795, -843, -1989, -3827, 1215, 927, -1442, 148, -8203, -2324, -1114, 49, -1108, 6105, -592, -11, -1719, -1792, 277, -1478, -3612, -313, 273, -1032, 2403, 4136, -2542, 29, -1585, -96, -722, -4129}
, {-2974, 453, 1429, -506, -180, 1481, 6206, -1231, 7504, 2842, 4860, -548, -1117, -1889, 5913, -1698, 236, 158, -134, 1120, -544, 114, -195, 337, -2471, -4471, 112, -248, -4204, -416, 1121, -1595, -6678, 397, 2651, -291, 1485, 1091, 4478, -1269, 4072, 419, 6006, 1918, 1119, -4615, 4384, 412, 1645, -2195, 1308, 225, -342, 1636, 692, -184, 719, -4501, 348, 826, -3711, 368, -158, 715, -2499, 1490, 2422, -4824, -703, 3, 5604, 622, 3974, 2601, 6547, 27, -573, 3176, 4620, -335, -119, -498, 879, -371, 475, 397, 1435, 1163, -2772, -5830, 1575, -543, -8931, 1339, -463, 2243, -4402, -557, 1692, -3805, 852, 1781, 4748, -56, 1815, 1876, 1508, -1230, 698, -3961, 4617, 163, 465, 1992, -676, 537, 1118, -652, 1257, -70, -961, -4415, 513, -223, -7455, -1368, 1369, -229, -5623, -672, 2888, -5036, 1857, -409, 4184, -1317, 1431, -60, 6908, -195, -306, -3023, 2631, -159, -46, 1616, 1539, 360, 1315, 1079, 104, 346, -865, -5242, 636, 200, -6196, -1646, 812, -1289, -3143, -428, -156, -1915, 1553, 177, 6541, -676, 1202, 1, 4207, -1098, 545, -6924, 1134, -934, 1867, -43, 28, 1238, -1218, -826, 1549, -1067, -2999, -4539, 549, 629, -5278, -426, -412, 1379, -5122, 1718, 909, -3174, 1224, 430, 5233, -529, 2792, 1958, 7507, -231, -415, -4740, 3254, 1041, 1106, -2011, -401, 781, 2556, -88, -670, 275, 764, -7064, -89, 752, -1997, 223, -178, -2157}
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

#define MODEL_INPUT_SCALE_FACTOR 13 // scale factor of InputLayer
#define MODEL_INPUT_ROUND_MODE ROUND_MODE_FLOOR
#define MODEL_INPUT_NUMBER_T int16_t
#define MODEL_INPUT_LONG_NUMBER_T int32_t

// node 0 is InputLayer so use its output shape as input shape of the model
// typedef  input_t[16000][1];
typedef int16_t input_t[16000][1];
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
