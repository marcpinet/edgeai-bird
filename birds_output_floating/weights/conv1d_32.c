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


const float  conv1d_32_bias[CONV_FILTERS] = {-0x1.e62a7a0000000p-5, -0x1.e90b760000000p-5, -0x1.c3258a0000000p-6, 0x1.feadb60000000p-8, 0x1.fc9a580000000p-9, 0x1.a5d5d40000000p-7, 0x1.c1e4120000000p-8, -0x1.e96e9e0000000p-5}
;

const float  conv1d_32_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{{0x1.7871680000000p-5}
, {-0x1.aaf88c0000000p-4}
, {-0x1.1bccea0000000p-4}
, {-0x1.02cba40000000p-9}
, {-0x1.86cf8e0000000p-6}
, {-0x1.f8a7c60000000p-4}
, {-0x1.d16bb40000000p-4}
, {-0x1.65c1ea0000000p-8}
, {-0x1.9e613c0000000p-14}
, {-0x1.165f200000000p-4}
, {-0x1.6f36ec0000000p-5}
, {-0x1.93fa9c0000000p-4}
, {0x1.466b260000000p-6}
, {0x1.7e4fce0000000p-5}
, {-0x1.95c4380000000p-5}
, {-0x1.53c8440000000p-4}
, {0x1.d93dca0000000p-10}
, {-0x1.73c4280000000p-3}
, {-0x1.d0c6500000000p-5}
, {-0x1.0e89ba0000000p-3}
, {-0x1.2dd8f60000000p-4}
, {-0x1.24a07c0000000p-3}
, {-0x1.291e1a0000000p-5}
, {-0x1.9fd33e0000000p-5}
, {-0x1.fd50740000000p-6}
, {0x1.56b6940000000p-7}
, {-0x1.d06fb60000000p-6}
, {0x1.352dee0000000p-10}
, {-0x1.3c7b160000000p-5}
, {0x1.565c3c0000000p-7}
, {-0x1.381aec0000000p-8}
, {0x1.35523a0000000p-6}
, {-0x1.69b67c0000000p-7}
, {0x1.5279980000000p-5}
, {-0x1.743bde0000000p-4}
, {0x1.215b140000000p-6}
, {-0x1.1220c80000000p-6}
, {0x1.699cb40000000p-4}
, {0x1.60c0200000000p-6}
, {0x1.8a7d8a0000000p-4}
}
, {{-0x1.ee11460000000p-5}
, {-0x1.1c777a0000000p-3}
, {-0x1.5f380a0000000p-6}
, {-0x1.7395900000000p-3}
, {0x1.b6cd1e0000000p-5}
, {0x1.958d220000000p-5}
, {0x1.59dd900000000p-5}
, {-0x1.612aee0000000p-3}
, {0x1.ecdea20000000p-7}
, {-0x1.73af920000000p-6}
, {-0x1.1476040000000p-3}
, {-0x1.46d3f80000000p-4}
, {-0x1.eca81e0000000p-4}
, {-0x1.256ec60000000p-5}
, {-0x1.c30be60000000p-4}
, {-0x1.6dcb300000000p-3}
, {0x1.1cced20000000p-4}
, {-0x1.3a13e60000000p-4}
, {0x1.367d560000000p-4}
, {-0x1.6115d20000000p-3}
, {0x1.8773420000000p-5}
, {0x1.5cf1340000000p-6}
, {-0x1.7c4b4a0000000p-9}
, {-0x1.1a98420000000p-3}
, {0x1.47929e0000000p-4}
, {-0x1.c8d6ce0000000p-8}
, {0x1.bed81a0000000p-5}
, {-0x1.a43f1c0000000p-5}
, {-0x1.a2e8c00000000p-4}
, {-0x1.d9ae040000000p-7}
, {-0x1.0514ee0000000p-3}
, {-0x1.50c4560000000p-3}
, {-0x1.249bca0000000p-3}
, {0x1.d6a3d80000000p-5}
, {-0x1.9c3ae40000000p-4}
, {0x1.440bac0000000p-5}
, {0x1.fa99f40000000p-5}
, {0x1.49974e0000000p-4}
, {-0x1.3c57660000000p-4}
, {0x1.4f6bfa0000000p-7}
}
, {{-0x1.295c740000000p-1}
, {-0x1.c13c140000000p-1}
, {-0x1.9825780000000p-1}
, {-0x1.ae51400000000p-1}
, {-0x1.870a5e0000000p-1}
, {-0x1.c800d60000000p-1}
, {-0x1.6665160000000p-1}
, {-0x1.8e5f200000000p-1}
, {-0x1.1eda280000000p-1}
, {-0x1.ec3c4a0000000p-2}
, {0x1.72ef540000000p-4}
, {0x1.25f43e0000000p-3}
, {0x1.434e060000000p-1}
, {0x1.0743b00000000p-1}
, {0x1.6cddf40000000p-1}
, {0x1.2f4a220000000p-1}
, {0x1.90c40a0000000p-1}
, {0x1.36b11e0000000p-1}
, {0x1.af9d620000000p-1}
, {0x1.3841740000000p-1}
, {0x1.520bba0000000p-1}
, {0x1.69ec980000000p-1}
, {0x1.4fee780000000p-2}
, {0x1.3853b60000000p-1}
, {-0x1.5f20940000000p-3}
, {0x1.2031b00000000p-2}
, {-0x1.0540f60000000p-1}
, {-0x1.9f99cc0000000p-3}
, {-0x1.1e08540000000p-1}
, {-0x1.04eec60000000p-1}
, {-0x1.b84a480000000p-2}
, {-0x1.e57bb80000000p-1}
, {-0x1.1788940000000p-2}
, {-0x1.e66d880000000p-1}
, {-0x1.faa5080000000p-3}
, {-0x1.2e4a340000000p-1}
, {-0x1.e711b80000000p-3}
, {-0x1.026f1c0000000p-2}
, {0x1.4986600000000p-12}
, {-0x1.331f100000000p-7}
}
, {{0x1.6213f20000000p-1}
, {-0x1.7fea600000000p-1}
, {0x1.9397f00000000p-1}
, {-0x1.56e5c20000000p+0}
, {0x1.b0bdbe0000000p-1}
, {-0x1.7735d60000000p+0}
, {0x1.f255f80000000p-1}
, {-0x1.4f705c0000000p+0}
, {0x1.f2c7d20000000p-1}
, {-0x1.550d500000000p-1}
, {0x1.5326240000000p-1}
, {-0x1.f63c2c0000000p-3}
, {0x1.6a1cb00000000p-2}
, {0x1.ed16c80000000p-3}
, {-0x1.e2929e0000000p-5}
, {0x1.ad16d80000000p-2}
, {-0x1.24a2ec0000000p-4}
, {0x1.3552ac0000000p-1}
, {0x1.445c040000000p-2}
, {0x1.af311e0000000p-3}
, {0x1.29e42c0000000p-2}
, {-0x1.8ec56e0000000p-2}
, {0x1.26eeb60000000p-3}
, {-0x1.8d61680000000p-2}
, {-0x1.2a33fa0000000p-4}
, {0x1.b905180000000p-2}
, {-0x1.1f6acc0000000p-2}
, {0x1.a35cd40000000p-1}
, {-0x1.0aac160000000p+0}
, {0x1.b04f5a0000000p-1}
, {-0x1.2e46380000000p+0}
, {0x1.88537a0000000p-1}
, {-0x1.ed38ac0000000p-6}
, {0x1.2751ae0000000p-1}
, {0x1.4e62880000000p-1}
, {-0x1.baa4dc0000000p-2}
, {0x1.85dd460000000p-1}
, {-0x1.232d640000000p+0}
, {0x1.694eb20000000p-1}
, {-0x1.2d36340000000p+0}
}
, {{0x1.5562800000000p-2}
, {0x1.3c3ca00000000p-2}
, {0x1.0582ac0000000p-2}
, {0x1.52845c0000000p-2}
, {0x1.6060300000000p-3}
, {0x1.32bca40000000p-2}
, {0x1.4141c00000000p-2}
, {0x1.b544cc0000000p-4}
, {0x1.e7ba9e0000000p-2}
, {-0x1.c0de2a0000000p-3}
, {0x1.ae16980000000p-1}
, {-0x1.2203380000000p+0}
, {0x1.7bd26c0000000p-1}
, {-0x1.8141960000000p+0}
, {0x1.89a04a0000000p-1}
, {-0x1.30a3f60000000p+0}
, {0x1.73b6760000000p-1}
, {-0x1.a5cb0c0000000p-5}
, {0x1.21d1020000000p-2}
, {0x1.e062ca0000000p-2}
, {-0x1.aa98660000000p-1}
, {0x1.3890ba0000000p-1}
, {-0x1.0594360000000p+0}
, {0x1.6d42680000000p-1}
, {-0x1.d942e40000000p-4}
, {0x1.28c75e0000000p-1}
, {0x1.1c47b00000000p-1}
, {-0x1.00b55a0000000p-2}
, {0x1.3439220000000p-1}
, {-0x1.7223ba0000000p-1}
, {0x1.215c300000000p-1}
, {-0x1.3373ce0000000p-2}
, {0x1.65fc9a0000000p-2}
, {0x1.26f4740000000p-2}
, {-0x1.1c51d40000000p-3}
, {0x1.1aee460000000p-1}
, {-0x1.8f75d40000000p-1}
, {0x1.ed8dd00000000p-2}
, {-0x1.4878900000000p-1}
, {0x1.0f8d240000000p-1}
}
, {{0x1.f8fdec0000000p-2}
, {-0x1.5d9de20000000p-2}
, {0x1.8ac5000000000p-1}
, {-0x1.eebf9a0000000p-1}
, {0x1.5281de0000000p-1}
, {-0x1.4ec83c0000000p+0}
, {0x1.6b461e0000000p-1}
, {-0x1.33c34c0000000p+0}
, {0x1.f25ba80000000p-2}
, {-0x1.7b2e560000000p-1}
, {0x1.1ce6e00000000p-5}
, {-0x1.0499740000000p-1}
, {-0x1.19bd380000000p-1}
, {-0x1.bbb4b00000000p-2}
, {-0x1.1d3df40000000p-1}
, {-0x1.bcc37c0000000p-5}
, {0x1.d02c620000000p-3}
, {0x1.90c6fc0000000p-3}
, {0x1.9f3eb80000000p-1}
, {-0x1.4746340000000p-3}
, {0x1.c6ec980000000p-1}
, {-0x1.4da13e0000000p-1}
, {0x1.9e2b540000000p-1}
, {-0x1.265cea0000000p-2}
, {0x1.ae9f880000000p-1}
, {0x1.8369400000000p-2}
, {0x1.6d26c40000000p-2}
, {0x1.8aa9a60000000p-1}
, {-0x1.2360520000000p-1}
, {0x1.2ff86c0000000p-1}
, {-0x1.5c74e60000000p+0}
, {0x1.6b14200000000p-2}
, {-0x1.ee85120000000p-1}
, {0x1.0411060000000p-2}
, {-0x1.ed1b020000000p-3}
, {-0x1.b4d5b40000000p-3}
, {0x1.c8a1940000000p-5}
, {-0x1.c2202a0000000p-1}
, {0x1.b89c040000000p-3}
, {-0x1.95d0840000000p-1}
}
, {{-0x1.b203920000000p-2}
, {0x1.e0bd5c0000000p-2}
, {-0x1.8f31ca0000000p-1}
, {0x1.3839660000000p-1}
, {-0x1.ada3ca0000000p-1}
, {0x1.0535160000000p-1}
, {0x1.6b13da0000000p-4}
, {0x1.4a494a0000000p-2}
, {0x1.7e4e5a0000000p-1}
, {-0x1.4ca15e0000000p-1}
, {0x1.00b7280000000p+0}
, {-0x1.898b460000000p+0}
, {0x1.c432fc0000000p-1}
, {-0x1.63d61a0000000p+0}
, {0x1.fc43e00000000p-1}
, {-0x1.54dcd20000000p-2}
, {0x1.64aa800000000p-1}
, {0x1.51a5e40000000p-3}
, {-0x1.7ea0560000000p-2}
, {-0x1.0ba92a0000000p-5}
, {-0x1.0dd23a0000000p+0}
, {-0x1.63b3760000000p-2}
, {-0x1.9070580000000p-2}
, {-0x1.be67d20000000p-2}
, {0x1.7af9b20000000p-1}
, {-0x1.cce0be0000000p-1}
, {0x1.e5029c0000000p-1}
, {-0x1.7d8e000000000p+0}
, {0x1.0e6f200000000p+0}
, {-0x1.833fdc0000000p+0}
, {0x1.d938b20000000p-1}
, {-0x1.1ad2460000000p-1}
, {0x1.c3b5e20000000p-1}
, {0x1.4225340000000p-2}
, {0x1.bfd95e0000000p-4}
, {0x1.472e2e0000000p-1}
, {-0x1.410cbc0000000p-1}
, {0x1.480ef80000000p-1}
, {-0x1.af068e0000000p-1}
, {0x1.04f1600000000p-1}
}
, {{0x1.150afc0000000p-7}
, {0x1.d7527c0000000p-4}
, {0x1.4078020000000p-4}
, {-0x1.40eda40000000p-3}
, {0x1.c4cbfc0000000p-5}
, {0x1.22dde40000000p-4}
, {-0x1.9745180000000p-4}
, {-0x1.e349c40000000p-6}
, {0x1.425f6e0000000p-3}
, {-0x1.7ee0620000000p-7}
, {0x1.76cfe40000000p-4}
, {0x1.3cc83c0000000p-3}
, {-0x1.dfd1f00000000p-6}
, {-0x1.9b55680000000p-4}
, {0x1.3dfc020000000p-3}
, {0x1.2727100000000p-5}
, {-0x1.0ccc3e0000000p-4}
, {0x1.d983440000000p-5}
, {-0x1.cb20640000000p-6}
, {-0x1.3d00ee0000000p-3}
, {-0x1.e680200000000p-8}
, {0x1.4e06600000000p-9}
, {-0x1.7b0ef60000000p-4}
, {0x1.f4b5480000000p-4}
, {0x1.d115760000000p-11}
, {-0x1.ca98d00000000p-5}
, {0x1.c0aefe0000000p-4}
, {0x1.cc79ac0000000p-7}
, {-0x1.0c03320000000p-4}
, {0x1.39ab140000000p-4}
, {0x1.fd2e9a0000000p-4}
, {0x1.cdf9e60000000p-6}
, {-0x1.0ade6c0000000p-5}
, {-0x1.3330de0000000p-3}
, {0x1.56ee040000000p-6}
, {0x1.9fee560000000p-4}
, {-0x1.012fde0000000p-4}
, {-0x1.51f4fc0000000p-5}
, {0x1.62c3560000000p-4}
, {0x1.658cc80000000p-8}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_GROUPS