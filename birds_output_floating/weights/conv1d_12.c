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


const float  conv1d_12_bias[CONV_FILTERS] = {0x1.59d3080000000p-7, 0x1.0bb2560000000p-7, 0x1.899cfa0000000p-5, 0x1.0831300000000p-6, -0x1.d330520000000p-4, 0x1.a008640000000p-7, 0x1.fafb360000000p-7, -0x1.eb25b40000000p-5}
;

const float  conv1d_12_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{{-0x1.a6dc340000000p-8}
, {0x1.3eb41c0000000p-4}
, {0x1.e8419e0000000p-3}
, {-0x1.0d05400000000p-4}
, {0x1.584c640000000p-2}
, {0x1.96f7e20000000p-5}
, {0x1.355b0c0000000p-1}
, {0x1.2ed3fc0000000p-4}
, {-0x1.a254720000000p-5}
, {0x1.05872c0000000p-1}
, {-0x1.f228480000000p-1}
, {0x1.7a05020000000p-2}
, {-0x1.a8708c0000000p+0}
, {0x1.061eb80000000p-2}
, {-0x1.84563a0000000p-1}
, {-0x1.0763fe0000000p-1}
, {0x1.00a80c0000000p-3}
, {-0x1.72f3dc0000000p+0}
, {0x1.61e6fa0000000p-1}
, {-0x1.a15c3e0000000p+1}
, {0x1.bba22c0000000p-1}
, {-0x1.cc85dc0000000p+1}
, {0x1.1f6fe20000000p+0}
, {-0x1.8489ae0000000p+0}
, {0x1.20531e0000000p+0}
, {0x1.ab01d00000000p-2}
, {0x1.248c580000000p-1}
, {0x1.41f4640000000p+0}
, {-0x1.2b56160000000p+0}
, {0x1.0e9e4e0000000p+0}
, {-0x1.817da20000000p-1}
, {0x1.e45a260000000p-1}
, {0x1.df27a40000000p-3}
, {-0x1.217f7a0000000p-1}
, {0x1.4d69e40000000p-1}
, {-0x1.328b060000000p+1}
, {0x1.5dafc00000000p-1}
, {-0x1.4da52e0000000p+1}
, {0x1.45c4b20000000p-2}
, {-0x1.2f3f620000000p+0}
}
, {{-0x1.c98fd40000000p-1}
, {0x1.65b40e0000000p+0}
, {-0x1.b585c00000000p+0}
, {0x1.7fd07a0000000p+0}
, {-0x1.bf8b860000000p+0}
, {0x1.e28b460000000p+0}
, {-0x1.824d220000000p+1}
, {0x1.02243a0000000p+1}
, {-0x1.44d3d60000000p+1}
, {0x1.d1b62c0000000p+0}
, {-0x1.8d4a7e0000000p+0}
, {0x1.eb9df80000000p-1}
, {-0x1.0f44ec0000000p-2}
, {0x1.d017200000000p-1}
, {0x1.f30ec00000000p-1}
, {-0x1.0eef2a0000000p+1}
, {0x1.cea9e80000000p+0}
, {-0x1.79c0fa0000000p+1}
, {0x1.2f4b800000000p+1}
, {-0x1.8cf06a0000000p+1}
, {0x1.09cb160000000p+1}
, {-0x1.68c5360000000p+0}
, {0x1.c788b00000000p+0}
, {-0x1.17c50e0000000p-2}
, {0x1.e80a940000000p-3}
, {-0x1.c267ea0000000p-1}
, {0x1.5627ee0000000p+0}
, {-0x1.b91edc0000000p-1}
, {0x1.e7393a0000000p+0}
, {-0x1.7b61b00000000p+1}
, {0x1.7ba0a80000000p+1}
, {-0x1.9774760000000p+1}
, {0x1.3dc9ba0000000p+1}
, {-0x1.f5f4740000000p+1}
, {0x1.314b340000000p+1}
, {-0x1.be450a0000000p+0}
, {0x1.4231e00000000p+0}
, {-0x1.1aed420000000p-1}
, {0x1.79c3d40000000p-1}
, {0x1.6254d60000000p-3}
}
, {{-0x1.62027e0000000p+1}
, {0x1.a432f00000000p-2}
, {0x1.39bdc20000000p-3}
, {-0x1.0032d80000000p+0}
, {0x1.6758560000000p+0}
, {-0x1.a271280000000p+1}
, {0x1.27042e0000000p+1}
, {-0x1.3b6b340000000p+2}
, {0x1.29affc0000000p+1}
, {-0x1.0451f20000000p+2}
, {0x1.23006c0000000p+1}
, {-0x1.c68ae60000000p+1}
, {0x1.40a8100000000p+0}
, {-0x1.5f937e0000000p+1}
, {0x1.1962de0000000p+0}
, {-0x1.7442520000000p+1}
, {0x1.29cf080000000p-1}
, {-0x1.afc0780000000p+1}
, {0x1.7fe52c0000000p+0}
, {-0x1.696d080000000p+1}
, {0x1.7ef6440000000p+0}
, {-0x1.9b53000000000p+1}
, {0x1.731f120000000p+0}
, {-0x1.869f020000000p+0}
, {0x1.4a8e0a0000000p-1}
, {-0x1.64c04c0000000p-1}
, {-0x1.3724aa0000000p-1}
, {0x1.e986d40000000p-2}
, {-0x1.c8b4d40000000p-2}
, {0x1.5071ec0000000p-1}
, {-0x1.0db0cc0000000p+1}
, {0x1.dd1ca60000000p-1}
, {-0x1.0d74780000000p+1}
, {0x1.5725ec0000000p-1}
, {-0x1.bb4a3e0000000p+1}
, {0x1.d062a80000000p+0}
, {-0x1.a5d1640000000p+1}
, {0x1.258ff20000000p+0}
, {-0x1.a37c960000000p+1}
, {0x1.cb7daa0000000p+0}
}
, {{0x1.2f7aca0000000p-2}
, {0x1.5d3ac00000000p+0}
, {-0x1.2b4e320000000p-1}
, {0x1.504b200000000p+0}
, {-0x1.68e27a0000000p-1}
, {0x1.3c48140000000p+0}
, {-0x1.9c20ce0000000p+0}
, {0x1.985c2c0000000p+0}
, {-0x1.3915680000000p+0}
, {0x1.a06e240000000p+0}
, {0x1.6397860000000p-2}
, {0x1.f6231c0000000p-1}
, {0x1.d018700000000p-1}
, {0x1.bbff600000000p-1}
, {0x1.58a3420000000p+0}
, {-0x1.cf8bb00000000p-1}
, {0x1.8b1fc20000000p+0}
, {-0x1.20f0c40000000p-1}
, {0x1.8d71ce0000000p+0}
, {-0x1.72ef880000000p-2}
, {0x1.5e5e420000000p+0}
, {-0x1.f401c40000000p-3}
, {0x1.8691a20000000p+0}
, {0x1.ab23940000000p-7}
, {0x1.3821760000000p-1}
, {-0x1.5660480000000p-1}
, {0x1.8d27840000000p+0}
, {-0x1.a40c0a0000000p-1}
, {0x1.cbb6ce0000000p+0}
, {-0x1.c8cf240000000p-1}
, {0x1.2c5b7a0000000p+1}
, {-0x1.88d5ca0000000p+0}
, {0x1.08ed800000000p+1}
, {-0x1.32830e0000000p+0}
, {0x1.12a3460000000p+1}
, {-0x1.f7bb9a0000000p-3}
, {0x1.0658b00000000p+1}
, {0x1.b797b80000000p-1}
, {0x1.0028bc0000000p+1}
, {0x1.1a74ba0000000p+0}
}
, {{0x1.d53ef00000000p-4}
, {0x1.a7873c0000000p-3}
, {0x1.c49a5e0000000p-2}
, {0x1.675dde0000000p-2}
, {0x1.69ce320000000p-1}
, {0x1.844b660000000p-1}
, {0x1.9a63420000000p-1}
, {0x1.4a28200000000p-1}
, {0x1.1614e60000000p-1}
, {0x1.d17de20000000p-3}
, {0x1.19b25c0000000p-3}
, {-0x1.af7f560000000p-2}
, {-0x1.0fdec20000000p-1}
, {-0x1.6efcd20000000p-1}
, {-0x1.a867d20000000p-1}
, {-0x1.8b1cb40000000p-1}
, {-0x1.cf30200000000p-1}
, {-0x1.fbc4be0000000p-1}
, {-0x1.f5a6100000000p-1}
, {-0x1.956bb20000000p-1}
, {-0x1.991bd00000000p-1}
, {-0x1.dcd7240000000p-2}
, {-0x1.d3068e0000000p-3}
, {-0x1.74c78c0000000p-6}
, {0x1.85c18a0000000p-2}
, {0x1.4beada0000000p-1}
, {0x1.cb601a0000000p-1}
, {0x1.ba21c60000000p-1}
, {0x1.d8d4aa0000000p-1}
, {0x1.19d28c0000000p+0}
, {0x1.1e80700000000p+0}
, {0x1.10d6a80000000p+0}
, {0x1.229e0e0000000p-1}
, {0x1.8547ca0000000p-1}
, {0x1.72fe060000000p-2}
, {0x1.b1394e0000000p-4}
, {-0x1.0131ca0000000p-1}
, {-0x1.bfd3080000000p-2}
, {-0x1.1c1c8e0000000p-1}
, {-0x1.32c9aa0000000p-1}
}
, {{-0x1.6e64920000000p+0}
, {0x1.77720c0000000p-1}
, {-0x1.e7a5940000000p+0}
, {0x1.ef4e6a0000000p-2}
, {-0x1.052d780000000p+1}
, {0x1.42562a0000000p+0}
, {-0x1.49fdc00000000p+1}
, {0x1.8da7aa0000000p+0}
, {-0x1.56ed180000000p+1}
, {0x1.917ffa0000000p+0}
, {-0x1.53fa120000000p-1}
, {0x1.8c94080000000p-2}
, {0x1.18687e0000000p-2}
, {0x1.727bf40000000p-1}
, {0x1.3ba9f00000000p+0}
, {-0x1.66bd860000000p+1}
, {0x1.f7f0560000000p+0}
, {-0x1.a820c00000000p+1}
, {0x1.421b4c0000000p+1}
, {-0x1.5eb1a80000000p+1}
, {0x1.edcc420000000p+0}
, {-0x1.5086340000000p+0}
, {0x1.cfc0f40000000p+0}
, {-0x1.1a1d300000000p-2}
, {-0x1.a995900000000p-3}
, {-0x1.2d23680000000p+0}
, {0x1.837b0c0000000p+0}
, {-0x1.77a9b00000000p-1}
, {0x1.8adb560000000p+0}
, {-0x1.3cf7a20000000p+1}
, {0x1.347eee0000000p+1}
, {-0x1.52fd140000000p+1}
, {0x1.06e0fe0000000p+1}
, {-0x1.8ebdee0000000p+1}
, {0x1.1ccc460000000p+1}
, {-0x1.6fb6a60000000p-1}
, {0x1.438b0e0000000p+0}
, {0x1.61ef060000000p-1}
, {0x1.069d460000000p-1}
, {0x1.0ad2e80000000p+0}
}
, {{-0x1.244b6c0000000p+0}
, {-0x1.73ffe00000000p-2}
, {-0x1.9b49b00000000p-1}
, {-0x1.61fe7a0000000p-3}
, {-0x1.fa7bac0000000p-1}
, {-0x1.6daa7e0000000p-1}
, {-0x1.d48a7c0000000p+0}
, {-0x1.7b86360000000p-1}
, {-0x1.6850fa0000000p+0}
, {-0x1.464d900000000p+0}
, {-0x1.2e37de0000000p+1}
, {-0x1.09d4e80000000p+0}
, {-0x1.fdd4220000000p+0}
, {-0x1.71b8e60000000p-1}
, {-0x1.4daba00000000p+1}
, {-0x1.3bfbdc0000000p-2}
, {-0x1.39e68c0000000p+1}
, {0x1.ec1b9c0000000p-9}
, {-0x1.66a61c0000000p+1}
, {0x1.3f4cc80000000p-1}
, {-0x1.f7e5980000000p+0}
, {0x1.1946880000000p+0}
, {-0x1.aa54be0000000p+0}
, {0x1.53c6800000000p+0}
, {-0x1.c3ea820000000p-3}
, {0x1.1a276e0000000p+0}
, {0x1.1a30480000000p-4}
, {0x1.56db5e0000000p-1}
, {0x1.1167260000000p-1}
, {0x1.d114c00000000p-3}
, {0x1.f7c5da0000000p-2}
, {-0x1.d82e280000000p+0}
, {0x1.f255da0000000p-2}
, {-0x1.04e89a0000000p+1}
, {0x1.3168a00000000p-3}
, {-0x1.df510c0000000p+0}
, {-0x1.78a06c0000000p-2}
, {-0x1.5b03d00000000p-1}
, {-0x1.2102560000000p-1}
, {-0x1.0b93720000000p-2}
}
, {{-0x1.5f18360000000p-5}
, {0x1.bcf0360000000p-5}
, {-0x1.c3f77c0000000p-6}
, {-0x1.3f9dc20000000p-3}
, {0x1.1a7ad80000000p-5}
, {0x1.1bf6b20000000p-4}
, {-0x1.18854e0000000p-3}
, {0x1.2964e00000000p-5}
, {-0x1.f59a1e0000000p-4}
, {-0x1.efe6d40000000p-6}
, {-0x1.6f70120000000p-7}
, {-0x1.d1d9c60000000p-6}
, {-0x1.5e6b720000000p-4}
, {0x1.6c67840000000p-7}
, {-0x1.9f71080000000p-4}
, {-0x1.8f920c0000000p-8}
, {0x1.2649560000000p-4}
, {0x1.5f34840000000p-5}
, {-0x1.d2f4be0000000p-7}
, {-0x1.215eca0000000p-8}
, {-0x1.7115ca0000000p-4}
, {-0x1.2892ae0000000p-4}
, {-0x1.c074920000000p-4}
, {-0x1.33caae0000000p-3}
, {-0x1.a95fa60000000p-5}
, {-0x1.6d129e0000000p-6}
, {-0x1.dd29160000000p-6}
, {-0x1.11906c0000000p-5}
, {0x1.ef54fc0000000p-5}
, {-0x1.16b3380000000p-3}
, {-0x1.4d40dc0000000p-5}
, {-0x1.33397a0000000p-3}
, {-0x1.a4695e0000000p-5}
, {-0x1.7711fa0000000p-4}
, {0x1.2662040000000p-4}
, {0x1.cbbbf40000000p-5}
, {0x1.70ed9e0000000p-5}
, {-0x1.8e5c6c0000000p-5}
, {0x1.55d8f40000000p-8}
, {-0x1.0792f40000000p-4}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_GROUPS