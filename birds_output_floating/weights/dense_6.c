/**
  ******************************************************************************
  * @file    weights/fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_SAMPLES 96
#define FC_UNITS 3


const float dense_6_bias[FC_UNITS] = {-0x1.bc585e0000000p-2, 0x1.bb4fa60000000p-2, 0x1.98fdbc0000000p-5}
;

const float dense_6_kernel[FC_UNITS][INPUT_SAMPLES] = {{0x1.83100e0000000p-1, 0x1.4117a20000000p-1, 0x1.051bda0000000p-3, -0x1.a9c6d80000000p-4, 0x1.4d3c480000000p-2, -0x1.ef537c0000000p-1, -0x1.26a2340000000p-3, 0x1.c5c7f40000000p-1, 0x1.3d86540000000p-3, 0x1.14790a0000000p-2, 0x1.abb6a40000000p-3, 0x1.dc141a0000000p-1, 0x1.232caa0000000p-3, -0x1.b8d6fc0000000p-2, 0x1.a220aa0000000p-3, 0x1.34db760000000p-4, 0x1.66e2220000000p-3, -0x1.91cfe00000000p-3, -0x1.015a960000000p-3, -0x1.d687a60000000p-3, 0x1.ccdc760000000p-6, -0x1.6e25ea0000000p-4, -0x1.371e380000000p-3, -0x1.7f7adc0000000p-6, -0x1.22ac240000000p-1, 0x1.1077360000000p-3, 0x1.0b6c940000000p-2, 0x1.2ae76e0000000p-3, 0x1.45e3a60000000p-3, 0x1.044a880000000p-3, 0x1.984c940000000p-3, 0x1.78aefc0000000p-2, 0x1.be6ed40000000p-2, 0x1.5b52720000000p-1, -0x1.57ab3c0000000p-8, 0x1.213e000000000p-6, 0x1.76125a0000000p-2, -0x1.f4ad260000000p-1, 0x1.d895e60000000p-3, 0x1.948a3e0000000p-1, -0x1.cf47b80000000p-14, 0x1.a7bdcc0000000p-4, 0x1.2f54580000000p-2, 0x1.8e321c0000000p-1, -0x1.7dc64e0000000p-4, -0x1.290ce20000000p-3, -0x1.edf8840000000p-5, 0x1.7ee4760000000p-3, 0x1.af56dc0000000p-4, -0x1.5227c00000000p-3, 0x1.9d6fa20000000p-2, 0x1.6c2c660000000p-2, 0x1.7fddba0000000p-3, 0x1.14d9100000000p-4, -0x1.ba2d180000000p-3, -0x1.5935660000000p-3, -0x1.2348600000000p-1, 0x1.64cdba0000000p-3, -0x1.0610ce0000000p-5, 0x1.7d59e40000000p-4, 0x1.7002820000000p-7, 0x1.3574940000000p-4, 0x1.1733520000000p-6, 0x1.1f32c20000000p-2, 0x1.171a040000000p-1, 0x1.b1b6060000000p-1, -0x1.3eaff20000000p-3, 0x1.d8c32c0000000p-4, 0x1.86c3600000000p-1, -0x1.7ac3720000000p-1, 0x1.1b1dec0000000p-3, 0x1.4500300000000p-1, -0x1.7176000000000p-3, 0x1.4bb79c0000000p-2, 0x1.bb06320000000p-3, 0x1.9f68ae0000000p-2, -0x1.821d720000000p-10, -0x1.9c0de80000000p-2, -0x1.7053420000000p-3, 0x1.4ea05a0000000p-2, 0x1.8ce6ac0000000p-4, -0x1.d7ea520000000p-2, 0x1.af4bf80000000p-3, -0x1.57530c0000000p-3, 0x1.6c7e820000000p-2, 0x1.d977a60000000p-3, -0x1.ba3f1c0000000p-6, 0x1.437ada0000000p-3, -0x1.0890ac0000000p-1, 0x1.91fd780000000p-5, -0x1.d4b12c0000000p-5, -0x1.935ace0000000p-4, -0x1.3e27e00000000p-4, 0x1.04c53c0000000p-2, -0x1.02db7e0000000p-3, 0x1.076a420000000p-1}
, {-0x1.ce045e0000000p-3, -0x1.e9fc580000000p-3, -0x1.d673b20000000p-3, 0x1.bf4bfc0000000p-3, -0x1.1d9ea20000000p-2, -0x1.6b373a0000000p-2, -0x1.cc59d00000000p-2, 0x1.2dca080000000p-2, -0x1.926cea0000000p-4, -0x1.9960ea0000000p-1, 0x1.7e2e6c0000000p-1, 0x1.ef03220000000p-1, -0x1.97bdae0000000p-2, 0x1.b4093c0000000p-5, 0x1.5eb3b00000000p-4, 0x1.7c31360000000p-1, 0x1.55c7200000000p-1, -0x1.7bcb500000000p-1, 0x1.13ec9e0000000p-1, 0x1.20e0c40000000p-2, -0x1.48a22e0000000p-2, -0x1.578c840000000p-6, -0x1.ca8f2e0000000p-4, -0x1.3ec6480000000p-2, 0x1.76333c0000000p-4, -0x1.6df60a0000000p-4, -0x1.8ea0980000000p-4, -0x1.10ef840000000p-6, 0x1.bb6bd40000000p-6, -0x1.fbdb900000000p-2, 0x1.02502e0000000p-5, -0x1.a9ec160000000p-2, 0x1.3bfeb40000000p-4, 0x1.4f73440000000p-4, 0x1.1a42ba0000000p-3, 0x1.dd45080000000p-6, -0x1.d4f8940000000p-2, -0x1.7ba82a0000000p-2, -0x1.49082e0000000p-3, 0x1.b33f3a0000000p-3, -0x1.113e600000000p-2, -0x1.4f05500000000p-1, 0x1.e1f2080000000p-2, 0x1.9b28140000000p-1, -0x1.34daa00000000p-1, 0x1.3a3b540000000p-3, 0x1.2b0c1a0000000p-6, 0x1.2865d00000000p+0, 0x1.155a5e0000000p-2, -0x1.0f5cba0000000p-2, 0x1.e19f940000000p-2, -0x1.4a0fee0000000p-9, -0x1.dad94c0000000p-3, 0x1.5fc5800000000p-3, -0x1.d795ba0000000p-4, -0x1.c503ae0000000p-2, 0x1.ce10020000000p-4, 0x1.b8ec180000000p-5, -0x1.62366c0000000p-5, 0x1.11912a0000000p-4, -0x1.3dcfd00000000p-5, -0x1.4ea0b80000000p-2, -0x1.bce7b80000000p-4, -0x1.f3bee40000000p-2, 0x1.ede44a0000000p-3, -0x1.43a2c20000000p-3, -0x1.9f10c00000000p-4, 0x1.05f20a0000000p-6, 0x1.3c2c440000000p-3, -0x1.d10a5e0000000p-2, -0x1.9a9bb00000000p-4, 0x1.5be9280000000p-4, 0x1.2e05d60000000p-6, -0x1.aceab20000000p-2, 0x1.0269380000000p-1, 0x1.44a4ca0000000p+0, -0x1.30b2fa0000000p-1, 0x1.19fb080000000p-1, -0x1.2021120000000p-5, 0x1.09f3980000000p-1, 0x1.fdba060000000p-3, -0x1.8d0e9e0000000p-2, 0x1.acee880000000p-2, 0x1.7ff8120000000p-2, -0x1.0e3fd20000000p-1, 0x1.2b97c00000000p-3, -0x1.2232fe0000000p-3, -0x1.9b5df00000000p-2, 0x1.716d460000000p-2, -0x1.bef00c0000000p-5, 0x1.1fce700000000p-5, 0x1.fcb1600000000p-4, -0x1.4b30ae0000000p-4, -0x1.510a3a0000000p-2, 0x1.3cabce0000000p-7, -0x1.a3b7960000000p-2}
, {-0x1.77bd1e0000000p-1, -0x1.1991940000000p-3, -0x1.3cc9880000000p-3, 0x1.10a42e0000000p-2, 0x1.b021660000000p-3, 0x1.810b780000000p-1, 0x1.13c0d00000000p-2, -0x1.b710380000000p+0, -0x1.0d77020000000p-5, 0x1.6b86860000000p-2, -0x1.6d63260000000p-1, -0x1.0b527a0000000p+1, 0x1.3266e20000000p-1, 0x1.56db520000000p-2, -0x1.dd05540000000p-4, -0x1.5e1eca0000000p-1, -0x1.75bc300000000p-2, 0x1.7ec94e0000000p-1, -0x1.604e020000000p-1, -0x1.3e994e0000000p-3, 0x1.0e4cc60000000p-4, -0x1.b15dea0000000p-4, -0x1.e66c400000000p-4, 0x1.a5c41a0000000p-2, 0x1.fe54e20000000p-4, -0x1.0ac9700000000p-5, -0x1.0a83800000000p-5, -0x1.574cb20000000p-4, 0x1.85efca0000000p-4, 0x1.91253a0000000p-5, -0x1.36df940000000p-3, 0x1.7d73c60000000p-3, -0x1.7898820000000p-1, -0x1.ad990e0000000p-2, -0x1.46860e0000000p-3, 0x1.597f380000000p-4, -0x1.66d8c20000000p-4, 0x1.0f37f00000000p+0, 0x1.8147c20000000p-2, -0x1.2f3c9c0000000p+0, 0x1.1a63280000000p-5, -0x1.befcda0000000p-4, -0x1.bf649a0000000p-1, -0x1.34cacc0000000p+1, 0x1.710dd80000000p-1, 0x1.bc660e0000000p-3, -0x1.f1e44e0000000p-5, -0x1.1a32a60000000p+0, -0x1.125dc80000000p-2, 0x1.3b3a680000000p-1, -0x1.7f32fa0000000p-2, -0x1.60c54e0000000p-3, 0x1.045c440000000p-2, 0x1.206fd80000000p-2, -0x1.e797c60000000p-4, 0x1.8c6f180000000p-2, 0x1.3b67020000000p-4, 0x1.488a7e0000000p-6, -0x1.16006c0000000p-4, 0x1.b7759e0000000p-3, -0x1.0ef32a0000000p-4, 0x1.5acf8e0000000p-3, -0x1.4718480000000p-4, 0x1.f436ea0000000p-3, -0x1.e2b0bc0000000p-1, -0x1.6870020000000p-1, 0x1.1412b00000000p-2, 0x1.880fe20000000p-6, -0x1.897f240000000p-1, 0x1.b46b7a0000000p-1, 0x1.88099e0000000p-5, -0x1.3cafe00000000p+0, 0x1.bbd3ea0000000p-3, 0x1.9e21300000000p-3, -0x1.77effe0000000p-1, -0x1.dd9fec0000000p+0, 0x1.24fd240000000p-1, -0x1.af01b60000000p-6, -0x1.a7cd400000000p-7, -0x1.802a7e0000000p-1, -0x1.5028580000000p-2, 0x1.9df1c20000000p-2, -0x1.48dd4e0000000p-1, 0x1.d36bcc0000000p-4, 0x1.38d6040000000p-2, 0x1.b3ff540000000p-4, 0x1.73b4760000000p-3, 0x1.59c6e40000000p-3, 0x1.3ed6fa0000000p-4, 0x1.0ebdac0000000p-2, -0x1.dbae520000000p-6, -0x1.ab052a0000000p-4, 0x1.0ee5c40000000p-2, 0x1.afd1520000000p-4, 0x1.65dfd00000000p-4, 0x1.bc66780000000p-4}
}
;

#undef INPUT_SAMPLES
#undef FC_UNITS