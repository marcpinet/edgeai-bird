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


const int16_t  conv1d_27_bias[CONV_FILTERS] = {187, 810, -243, -545, -1563, 334, -162, 566, -252, -456, 1328, 469, -552, 1372, -306, -629, -1036, -131, 1168, -716, -327, -251, -358, -241, 784, -173, -107, -248, -251, -255, -187, -328}
;

const int16_t  conv1d_27_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{{2276, -955, 341, 1925, -134, 252, 1659, 861, 339, -817, -882, -1552, 1038, 36, 1486, 1032, -3911, -4428, 1153, 626, 1618, -3610, 1762, 522, -900, -764, 26, 621, 944, -434, 635, 1512}
, {-1036, 228, -1560, -619, 287, -3141, -1029, 396, -2125, 200, 48, -830, -2765, 621, -1943, -2463, -833, -770, -2089, -147, 1481, -745, -697, 146, 1018, 374, 398, -3914, 490, -523, -125, 151}
}
, {{3155, 40, 1119, 3056, 672, -525, 2702, -216, -730, -812, -400, 1614, -2074, -1249, -1743, 1279, 1198, 362, 267, -405, 1293, 851, 669, 482, -298, -703, -431, -2856, -569, 599, -108, 597}
, {-264, 499, -450, -116, 29, -2400, 1175, 321, -3576, -382, 526, 2452, -3156, -57, -5646, -1798, -515, -289, -3827, 576, 655, 1344, -3490, 47, 76, 61, 552, -2439, -91, 710, -98, 558}
}
, {{-197, -386, -985, -35, -407, -874, 108, 49, -606, -280, -555, 80, -168, -200, 246, 471, -597, -1603, 112, -555, 157, -635, 185, -363, -981, 689, 250, -692, 283, -921, -92, -954}
, {-1235, 287, -965, 438, 331, -808, 144, 402, -916, 475, -232, -294, 373, -331, -1110, 115, -1338, -464, -406, -501, -304, -667, -1232, 481, -496, -621, -821, -356, -437, -986, -1194, -1183}
}
, {{429, 373, -460, -1297, -871, -43, -963, 586, -377, -508, 92, 339, -41, 246, -313, -230, -1304, 101, -910, -398, -330, -827, -1146, 204, -260, 171, 102, -928, -698, 116, 771, 140}
, {-76, 154, -120, -574, -738, -1195, -412, 81, -997, 142, -328, -586, -1028, 154, -554, 147, 96, -212, -485, -173, -794, -853, 65, 172, -936, 766, -881, -511, -643, 485, -786, 548}
}
, {{-5247, 650, 352, -218, -595, -1884, -659, 25, -2810, -142, 380, -2051, -3471, 98, -2292, -748, 469, 1248, -937, -198, 283, 1740, -3519, 19, 765, -613, -599, -1917, 736, 239, -37, 391}
, {-40, 298, -1649, 564, 668, 465, 106, 845, -271, 466, 593, 146, 481, -759, -2017, 284, -101, -2299, -185, 669, -1968, -101, -1650, 249, 8, 198, -1023, 1921, -83, -405, 607, -1870}
}
, {{-866, -176, -5377, -62, -370, 997, 864, 142, -482, 36, -875, 296, -305, -511, 36, 1676, -5098, -5899, 895, -312, -3058, -2567, -48, -91, -323, -47, -302, 102, 459, 415, -217, -5142}
, {-1145, -337, 2269, 1313, -440, 332, 567, -96, -1170, 645, -209, 2136, 93, 177, -600, 932, 711, 2313, -457, 683, 1578, 598, -419, 307, -664, 328, -642, 1243, -21, -539, -274, 2383}
}
, {{331, 587, 838, 493, -777, 569, 452, 526, 314, 522, 711, 23, 9, 886, 274, 1679, -3703, -887, 34, 737, -521, -187, 1143, 160, 1002, 94, 708, 908, 1064, 60, -179, 471}
, {867, -33, 411, 1425, -313, 1586, 81, -170, 11, -34, 1263, 1045, 857, 1008, 1116, 153, -3875, -188, 101, -644, -509, -63, 47, -70, 833, 384, 248, 1096, 409, 97, -94, -256}
}
, {{2673, -48, 1316, 2177, 590, 2231, 1462, 62, 2985, 946, 488, 1650, 861, -211, 1644, 1929, -396, 587, 1971, -568, 1012, 891, 2354, -349, 632, -429, -802, -649, 200, -444, -418, 1078}
, {-1809, -482, -1124, -3500, 610, -4104, -2430, -539, -4183, -783, -516, 2402, -4981, 574, -4764, -3974, -550, -2836, -3060, -235, -907, 2399, -2809, -430, -277, 128, 640, -4140, -650, 1071, 799, -897}
}
, {{-966, -902, -181, -932, -587, -1423, -313, -472, -836, 698, 425, 378, -528, -173, -568, -242, -821, -227, -1464, 386, -204, -42, -204, -102, 528, 452, 380, -1161, -443, -129, -337, -401}
, {-561, 582, 324, 644, 513, -214, -369, 551, 2, 749, 501, 429, -77, 619, -457, 83, 173, -805, -463, -379, -128, -826, -665, -818, -412, 415, -129, -863, 411, -136, -555, 120}
}
, {{2295, 140, -337, 1035, -30, 1155, 1225, 801, 739, -210, 714, -1925, 2215, -149, 1055, 1449, -3361, -348, -59, 133, -428, 1045, 1602, -508, -312, 681, 20, 2188, -629, 28, -625, 374}
, {3233, 1306, 1004, 902, -361, -94, 1419, -631, -710, -466, 738, -856, 2099, 903, 2150, 2064, -2253, 21, 59, 645, 1306, 3710, 1365, -621, 349, -523, -558, 1765, -80, 1885, -708, 492}
}
, {{1394, 912, 397, 104, -771, 1075, 138, 417, 1075, -273, -827, 3566, -740, -442, -837, 602, 3648, 890, 69, 762, 681, 1024, 1910, -186, -134, -687, -440, -462, -68, 228, 212, 446}
, {-1743, -1034, -2296, -1532, 701, -1523, 789, -607, -2483, -563, 417, 4699, -2073, 534, -3369, -3190, -23, -1597, -1752, 68, 1281, 1407, -2434, -330, -1484, 259, -528, -3913, -91, 1032, 768, 31}
}
, {{3649, -433, 1222, 2385, -511, 2045, 1962, 176, 1977, -1037, 341, 78, 1397, -370, 2083, 3006, -793, -167, 1741, 85, 1033, -2865, 3285, -566, -193, 13, 204, 911, 683, -509, 41, 2531}
, {-3930, -598, -6752, -3095, -329, -5701, -2289, 897, -4191, -603, 278, 527, -4964, -796, -4095, -4942, -5471, -4639, -4582, -434, -1103, -3653, -2281, -403, -349, 354, 485, -4651, 26, -19, 669, -1887}
}
, {{1217, 1029, 1856, -4736, -402, -3190, -5705, 624, -1788, 55, -216, -3098, -18, 1411, 1535, -1391, -319, 1861, -3098, 435, 1595, -120, -1029, -182, 886, 753, 75, 111, -110, 305, -273, 1739}
, {526, -68, 1728, -5984, 1041, -3241, -8097, 773, 905, 410, -29, -3679, 16, 131, 3029, -1768, 1060, 1310, -3001, -310, 1424, -650, 1360, 242, 259, 245, -308, -3218, 411, 595, 256, 1105}
}
, {{-886, 274, -1020, 1345, 201, -856, 2732, -498, -848, -550, -960, 483, -947, 928, -388, 1143, -588, -274, 858, -273, 61, -2332, -118, 406, -654, -710, -405, 601, -444, -331, -310, 250}
, {-1704, 701, 2969, 1735, 821, 256, 2364, 424, -80, -267, -701, 692, 1207, 124, -238, 1015, -798, 3575, 490, -131, 1648, -249, 10, 489, -473, 123, -46, 940, -30, -128, 478, 2367}
}
, {{117, -229, 554, -90, -1024, -336, -28, 345, -469, -929, -799, 712, 415, -952, -1106, -832, -568, -865, 403, -999, -752, 324, 269, -811, 626, -568, -1026, -971, 450, -987, -429, -207}
, {-1063, -269, 38, -29, 413, -710, 167, 513, -225, -279, -849, 256, -583, 422, -467, -564, 125, -412, -1010, -70, -280, 167, -464, -448, 540, -775, 196, 211, -879, 627, 776, -253}
}
, {{410, 690, -4963, -909, 233, -1954, -1212, -29, -1625, -444, 145, 1404, -1234, -153, -1852, -893, 593, -5288, -1322, 25, -758, -1694, -389, 646, -670, -817, -2, -827, -412, -350, 217, -1962}
, {2397, 299, -1664, 2053, -837, 3246, 1136, -287, 1713, -559, -1010, 967, 1140, 155, 1365, 1940, -1286, -1520, 812, 168, -821, 664, 1939, 106, 23, 10, 192, 2391, -151, 961, 587, -203}
}
, {{1672, 754, -1745, 92, -596, 698, 585, 2, 1103, 451, 747, 563, 128, -764, 650, 2181, 410, -2513, -71, 957, -677, -1763, 73, 1107, -573, 397, 447, 1436, -308, -406, 780, -1703}
, {1369, -420, -1880, -34, 182, -364, 724, -236, 1857, 14, -459, 950, 1085, -80, 1179, 1156, 686, -292, 973, 904, -1875, 566, 1464, 35, 92, -725, -384, 964, 128, 99, 315, -287}
}
, {{525, 667, 3135, 1529, -1007, -50, -57, 11, -876, -838, -959, -963, 350, -247, 735, 1781, -6065, 1028, 805, 578, 238, -4242, 835, 510, 305, -48, -583, 677, -464, 184, 902, 1283}
, {-592, -110, 3342, 1234, -909, 1601, 668, -854, -494, -1119, -136, -53, 798, -398, -150, 1518, -6734, -219, -227, 88, -573, -4077, 472, -1128, -51, 835, -762, 184, 1004, 380, 142, 1365}
}
, {{-1763, -504, -2336, -136, -555, -2193, 68, 381, 242, -874, -815, 99, -1480, -143, -756, -1511, -990, -462, -590, 300, 1117, -2121, -1356, -1018, 622, -510, -1002, -1242, -622, -788, -342, -981}
, {-328, 189, -253, -654, 519, -466, 49, 308, 762, 249, 74, -38, -1027, -260, -1134, -1350, 1152, 377, -185, -883, 912, -2755, -682, -587, -24, 356, -87, -775, 571, -893, 98, 493}
}
, {{389, -704, -1232, -1625, 174, -1423, 159, -672, -1885, 403, 252, 1846, 466, -135, 1048, -786, -361, -35, -442, -638, -2263, -799, 1098, -495, 230, 840, -122, -640, 773, -962, 812, -402}
, {-147, -907, -353, -969, -58, -2195, -1295, -204, -627, 534, -898, -369, -1572, -1096, -1065, -1262, -809, -1311, -2004, -617, -481, -617, -821, -692, -868, 677, -718, -1090, 270, -59, 556, -802}
}
, {{1767, 101, 443, 992, -319, 1877, 854, 232, 452, -565, -395, -370, 1225, 847, 965, 1288, -3245, -246, -263, 44, -395, -1600, 1083, -513, 466, 209, -478, 1904, 303, 1564, 546, 218}
, {2132, 443, 670, 768, -164, 612, 938, -273, 1198, -658, 658, -1654, 2174, -102, 1942, 1219, -2105, -945, 761, 177, 266, -212, 1073, 672, 426, -222, -314, 2474, -138, 212, 102, -421}
}
, {{-456, 164, -290, 316, -31, -183, -1138, -307, -210, -678, -449, -870, 162, -389, -47, -1129, -819, -179, -179, 605, -1087, -518, -708, 29, -383, -855, -252, 26, -742, 96, -784, -1236}
, {263, -456, -863, -161, -648, -91, 271, -47, -522, 506, -555, 577, -585, -69, -496, 516, -741, -20, 596, -231, -481, -324, -1152, -845, 599, -724, -123, 326, -861, -178, 575, -854}
}
, {{70, -612, 210, -1000, 578, 75, 241, 902, -1598, 591, -452, 596, -556, -373, -456, -403, 253, -1396, 594, -462, -361, 328, -596, 585, 734, 640, 164, -1292, -861, 79, 726, -53}
, {-157, -304, -694, -397, -623, 103, -841, -395, -1819, -550, 260, -871, -963, -1024, -392, -74, 425, -859, -310, -42, 744, 355, -1019, -511, -708, -563, -901, -635, 310, -491, -815, 153}
}
, {{709, 434, 2534, 1216, 648, 15, -105, -39, -110, 849, 871, 718, 1, 64, 84, 679, -4331, -536, 619, 240, 443, -3246, 930, -419, 150, 693, 564, 1457, -307, -14, 86, 449}
, {348, 318, 1500, 1442, -711, 1222, 956, -313, -923, -231, 647, -508, 904, 211, 805, 1962, -5449, -775, 379, 669, 3, -2034, 736, -348, -109, 747, -546, 291, -469, -575, -323, 439}
}
, {{-1451, 356, 2145, 2537, -227, -69, 2293, 658, -34, -413, -77, 1274, 175, 120, -1381, 1507, -656, 3805, 605, -668, 2692, -1416, -1576, 236, 803, 228, -552, 409, 853, -545, 925, 3124}
, {-756, -231, -574, 1498, 750, -776, 1200, -284, -183, -438, 635, 1083, 7, 619, -467, 627, -120, 457, 453, -499, 285, -2883, 759, 164, -570, -487, -460, -567, -817, -216, 733, -1287}
}
, {{324, 135, -278, 767, -321, -750, 827, -347, -366, -279, 403, -649, 620, -44, -424, -69, -537, 105, -189, 811, 24, -696, -174, 325, -283, -558, 557, -298, -448, -299, -422, 73}
, {367, 607, 208, 63, 939, -444, -627, 817, -564, -232, 421, -565, -450, 623, 719, -916, -563, -506, -435, 315, 830, -650, -693, -782, 283, -836, -32, 203, 426, 840, 693, -844}
}
, {{-142, -311, -772, 332, -683, 189, -242, 705, -694, -532, 546, -259, -1170, 761, 237, 821, -862, -715, -682, -880, 302, 676, -1104, -930, -529, 32, -118, 159, -715, -189, 409, -763}
, {266, -371, -444, -594, -139, -1034, -1184, 64, 419, 833, -916, 180, 181, -272, -598, 567, -668, 44, -843, -859, 98, -269, -379, -958, -515, -806, -133, -724, 74, 684, 73, -634}
}
, {{-776, -427, -740, -357, 519, 138, -259, 413, -743, 558, 218, -402, -271, 641, 233, 898, 781, -1083, 161, 446, 309, 670, -589, -621, -37, -901, 401, -973, 814, -862, -862, 260}
, {-624, -314, 71, -891, -826, 429, 189, 596, 10, 376, 85, 108, -411, -284, 641, -418, -1024, -162, 380, -108, 28, 436, -258, -380, -174, 270, 175, -267, 608, -402, -582, 53}
}
, {{174, -584, -144, -143, -802, -917, -201, 76, 271, -185, 670, -557, -847, -1087, 252, -130, 16, -258, -409, 685, -835, 686, -784, 685, -60, -57, 644, 181, -286, 143, 741, 279}
, {436, -120, -631, -68, 242, 87, -303, 870, -847, -719, -171, -672, -268, 591, -691, 24, -846, 318, 559, 397, 306, 56, 71, -716, -319, 512, 246, 232, -245, 543, 61, -801}
}
, {{1338, -806, 1567, 1469, 415, 528, 586, 291, -203, 545, 664, 40, 544, -252, 208, 1738, -3159, -727, 781, -86, -236, 1232, -542, -357, -655, -666, 800, 642, -391, 293, -257, 482}
, {2821, -156, 1640, 32, -525, -337, -637, -779, 168, 543, -292, 181, -453, -328, 845, 1465, -1909, -1652, -703, 413, 704, 3257, 1628, 547, -198, -610, -399, 1272, 522, 1679, 1054, 510}
}
, {{-761, 279, 186, 208, -76, -1013, -508, -412, -1029, -767, -645, -239, -896, -246, -15, -765, -797, -619, -478, -280, -550, -396, 208, 451, -580, 488, 579, 598, -689, -206, -153, 523}
, {427, 205, 264, -936, 497, -579, -819, -486, -935, 157, 415, 436, 144, 91, -492, -873, -1003, 124, -364, 278, 149, -503, 29, -377, -338, 217, -354, -1064, -602, -691, 73, -744}
}
, {{853, -247, -1197, 933, 479, 2036, 837, 382, -492, 563, -292, -505, 2677, 613, 1608, 636, -1654, -1466, 662, -391, 1070, 2327, 1485, 289, -453, -718, 347, 2698, 432, -526, 289, -1643}
, {3418, 1133, 683, 1019, 618, -155, 231, 390, -373, 458, -336, -1788, 1912, 386, 1323, 1565, -42, -186, -895, -579, 1005, 3220, 810, 577, -131, -842, -84, 1982, -285, 317, -554, -287}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_GROUPS