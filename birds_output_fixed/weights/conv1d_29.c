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


const int16_t  conv1d_29_bias[CONV_FILTERS] = {9, -270, -236, -263, -260, -228, 321, 510, -232, -100, -199, -390, -236, -246, -321, -170}
;

const int16_t  conv1d_29_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{{4592, 194, 2372, 2096, -454, 3148, -471, 1022}
, {3435, 1362, 4117, 2387, 375, 2209, 1026, -1519}
}
, {{2422, 27, -415, 2029, -708, 799, -191, -272}
, {335, 2077, 1205, 2105, -77, 1034, -923, -123}
}
, {{5025, -1142, 264, 1039, 859, -902, -434, 185}
, {2857, 876, 196, -785, -1124, 1186, 959, 1052}
}
, {{2811, 457, 916, 2525, -1074, 1553, 1305, 995}
, {1411, 450, 110, 2399, 906, 1179, -242, -861}
}
, {{1795, 130, -14, 585, 885, -878, -770, 1111}
, {1856, 2207, -687, 1987, -119, 2050, 244, -412}
}
, {{2311, 670, 2880, 2644, -703, 400, 1050, -1577}
, {444, 1815, 1527, 2701, -968, 1119, 393, -942}
}
, {{4924, 150, 2618, 462, -304, 2767, 401, -1638}
, {4701, 1195, 4094, 860, 195, 1743, 994, -299}
}
, {{-61, -1034, -900, -1469, -1378, -1784, 1323, 543}
, {-427, -1472, -1122, -920, -942, -359, -639, 107}
}
, {{-944, -761, 898, -997, -1079, 333, 430, -309}
, {-396, -355, 1178, -749, 463, -1224, -913, -1461}
}
, {{-638, -532, -420, 619, -1082, -63, -227, -10}
, {-259, -919, -311, 670, -1001, -1043, 517, -1325}
}
, {{2166, 35, 2347, 2707, -547, 1631, 414, 808}
, {2294, 2295, 2482, -428, 316, -77, -1207, -586}
}
, {{204, -1132, 1085, 345, -490, 289, 407, -690}
, {-1035, 451, 404, 493, 641, -1372, 257, -981}
}
, {{2992, 928, 1904, 1976, 428, 1686, 347, -183}
, {1410, 1626, 2125, 271, -669, 1070, 1600, -416}
}
, {{-343, 363, -1379, -50, 7, -386, 38, 641}
, {-225, -1400, 405, -972, -224, 232, -691, -145}
}
, {{2212, 1647, 96, -258, 731, 1011, -743, 525}
, {2853, -1224, 1262, 16, 327, -225, 153, 1031}
}
, {{-668, 938, -316, -252, 682, -1363, 189, 871}
, {-140, -311, -177, 1165, 156, -612, -1100, 685}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_GROUPS