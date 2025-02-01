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
#define CONV_FILTERS      16
#define CONV_KERNEL_SIZE  40
#define CONV_GROUPS       1


const int16_t  conv1d_48_bias[CONV_FILTERS] = {-498, -519, 80, 69, -966, -494, 74, -766, -493, 620, -579, -496, 76, 78, -492, 91}
;

const int16_t  conv1d_48_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{{-491}
, {242}
, {310}
, {-5}
, {-660}
, {-872}
, {95}
, {-365}
, {-324}
, {87}
, {-46}
, {-96}
, {-333}
, {-309}
, {-135}
, {-138}
, {153}
, {240}
, {-880}
, {235}
, {-418}
, {-644}
, {-123}
, {-883}
, {0}
, {-286}
, {-462}
, {-594}
, {273}
, {-509}
, {-707}
, {256}
, {-555}
, {-199}
, {-1116}
, {-953}
, {-142}
, {-300}
, {-210}
, {120}
}
, {{363}
, {-26}
, {736}
, {-1074}
, {1253}
, {1082}
, {437}
, {481}
, {537}
, {457}
, {770}
, {1087}
, {-83}
, {164}
, {527}
, {773}
, {1031}
, {549}
, {-503}
, {37}
, {617}
, {-952}
, {-14}
, {477}
, {-540}
, {-317}
, {1039}
, {-689}
, {-794}
, {170}
, {280}
, {570}
, {-590}
, {338}
, {685}
, {-229}
, {1034}
, {-171}
, {-238}
, {-297}
}
, {{15162}
, {10711}
, {14436}
, {11763}
, {13499}
, {8298}
, {19417}
, {7899}
, {17764}
, {11900}
, {18840}
, {10039}
, {21864}
, {8382}
, {16496}
, {17263}
, {21758}
, {14051}
, {19814}
, {24300}
, {16068}
, {19548}
, {26964}
, {19827}
, {17401}
, {24231}
, {20973}
, {-277}
, {28003}
, {28718}
, {-9056}
, {30030}
, {21360}
, {-7956}
, {23823}
, {17876}
, {-4468}
, {14721}
, {16644}
, {13031}
}
, {{20078}
, {11493}
, {11010}
, {16117}
, {8483}
, {6151}
, {20710}
, {13368}
, {12057}
, {8404}
, {20855}
, {7781}
, {23713}
, {15510}
, {-433}
, {16422}
, {24052}
, {10820}
, {19972}
, {21632}
, {8631}
, {21354}
, {20604}
, {7728}
, {19995}
, {18628}
, {12174}
, {18066}
, {17815}
, {12492}
, {7393}
, {16664}
, {12918}
, {9846}
, {12932}
, {6601}
, {9183}
, {9309}
, {10835}
, {18196}
}
, {{143}
, {923}
, {252}
, {1356}
, {1728}
, {606}
, {376}
, {180}
, {140}
, {1087}
, {1013}
, {130}
, {-593}
, {528}
, {1068}
, {552}
, {301}
, {921}
, {252}
, {481}
, {386}
, {1304}
, {1418}
, {320}
, {799}
, {848}
, {89}
, {-322}
, {1379}
, {739}
, {135}
, {998}
, {961}
, {126}
, {-63}
, {1225}
, {307}
, {795}
, {1251}
, {740}
}
, {{-323}
, {-890}
, {-64}
, {281}
, {-581}
, {-121}
, {553}
, {-112}
, {35}
, {-241}
, {-329}
, {-303}
, {5}
, {1109}
, {-347}
, {-813}
, {321}
, {369}
, {770}
, {261}
, {4}
, {587}
, {193}
, {-865}
, {1008}
, {-107}
, {-331}
, {12}
, {-233}
, {-492}
, {-573}
, {771}
, {337}
, {-431}
, {353}
, {-522}
, {-849}
, {571}
, {-715}
, {-115}
}
, {{17176}
, {10115}
, {7528}
, {12884}
, {13090}
, {11039}
, {18355}
, {6609}
, {19304}
, {8590}
, {17465}
, {13730}
, {18421}
, {7724}
, {18040}
, {15686}
, {17447}
, {18466}
, {26252}
, {4008}
, {19826}
, {23276}
, {14952}
, {18754}
, {22517}
, {15455}
, {7166}
, {26493}
, {23927}
, {4385}
, {16236}
, {22228}
, {3696}
, {19632}
, {21516}
, {-4958}
, {14900}
, {16888}
, {6474}
, {19290}
}
, {{-9207}
, {-10076}
, {-9144}
, {-8882}
, {-9263}
, {-9921}
, {-11013}
, {-11129}
, {-10663}
, {-11292}
, {-10997}
, {-10879}
, {-11745}
, {-11859}
, {-10649}
, {-10447}
, {-12083}
, {-11178}
, {-11770}
, {-11616}
, {-11688}
, {-11408}
, {-12124}
, {-10914}
, {-12018}
, {-10561}
, {-10601}
, {-11304}
, {-12531}
, {-11574}
, {-11994}
, {-12062}
, {-11125}
, {-10956}
, {-11228}
, {-10682}
, {-11288}
, {-10483}
, {-10183}
, {-10426}
}
, {{-335}
, {-433}
, {-245}
, {-1160}
, {280}
, {351}
, {-843}
, {-972}
, {-252}
, {-942}
, {381}
, {-888}
, {38}
, {-414}
, {-508}
, {158}
, {-1061}
, {286}
, {168}
, {-1018}
, {-655}
, {-430}
, {-141}
, {-4}
, {-269}
, {253}
, {-46}
, {-999}
, {889}
, {-771}
, {-125}
, {67}
, {-567}
, {-398}
, {626}
, {-587}
, {335}
, {-1071}
, {2}
, {92}
}
, {{-14596}
, {-13506}
, {-18792}
, {-15685}
, {-17899}
, {-18458}
, {-18721}
, {-18618}
, {-20058}
, {-16892}
, {-20044}
, {-16626}
, {-20499}
, {-18328}
, {-19939}
, {-19409}
, {-21445}
, {-20898}
, {-19037}
, {-20408}
, {-21374}
, {-19374}
, {-21548}
, {-19718}
, {-22276}
, {-19982}
, {-20778}
, {-21219}
, {-16836}
, {-21486}
, {-18954}
, {-22809}
, {-21854}
, {-19651}
, {-22003}
, {-21690}
, {-19185}
, {-22417}
, {-18040}
, {-19214}
}
, {{-10898}
, {-8142}
, {-6225}
, {-2507}
, {-725}
, {-1951}
, {1863}
, {915}
, {5606}
, {1098}
, {3803}
, {-787}
, {1002}
, {-3390}
, {-4585}
, {-13671}
, {-9958}
, {-17310}
, {-13806}
, {-18510}
, {-17390}
, {-12879}
, {-18618}
, {-4394}
, {-7313}
, {407}
, {4955}
, {4192}
, {10931}
, {10067}
, {13155}
, {13815}
, {7690}
, {11416}
, {10039}
, {7785}
, {8351}
, {-515}
, {6483}
, {1229}
}
, {{432}
, {335}
, {-99}
, {-897}
, {970}
, {647}
, {766}
, {645}
, {-124}
, {-713}
, {847}
, {702}
, {-799}
, {288}
, {850}
, {-94}
, {-154}
, {73}
, {-94}
, {-402}
, {-58}
, {-401}
, {475}
, {-73}
, {497}
, {-57}
, {1033}
, {844}
, {430}
, {518}
, {-261}
, {613}
, {952}
, {230}
, {-427}
, {-64}
, {-568}
, {-537}
, {-707}
, {895}
}
, {{11432}
, {13640}
, {6347}
, {11598}
, {16044}
, {-160}
, {17845}
, {11021}
, {18247}
, {11324}
, {14580}
, {11842}
, {16920}
, {8139}
, {18249}
, {11677}
, {18987}
, {17908}
, {16390}
, {16156}
, {16526}
, {17648}
, {16051}
, {17773}
, {19746}
, {11911}
, {20451}
, {19859}
, {8848}
, {19284}
, {12540}
, {10222}
, {11654}
, {16262}
, {5957}
, {9967}
, {12809}
, {4430}
, {2130}
, {12420}
}
, {{17326}
, {14215}
, {11079}
, {11946}
, {12603}
, {6889}
, {18910}
, {16153}
, {13617}
, {7805}
, {21909}
, {8324}
, {21718}
, {13341}
, {8663}
, {15571}
, {26834}
, {9753}
, {20374}
, {22678}
, {13034}
, {22665}
, {18070}
, {22961}
, {20085}
, {12027}
, {21209}
, {15764}
, {14218}
, {24162}
, {11933}
, {4934}
, {19309}
, {20004}
, {1820}
, {15636}
, {14819}
, {-1533}
, {13798}
, {20898}
}
, {{-482}
, {-854}
, {-691}
, {-695}
, {-497}
, {-90}
, {-708}
, {267}
, {321}
, {-762}
, {172}
, {570}
, {302}
, {-651}
, {-123}
, {-35}
, {-1044}
, {-873}
, {-199}
, {670}
, {-432}
, {-595}
, {-491}
, {-534}
, {-878}
, {-845}
, {108}
, {-239}
, {-89}
, {173}
, {-1123}
, {-660}
, {-20}
, {-112}
, {-537}
, {-1133}
, {-476}
, {274}
, {-483}
, {-280}
}
, {{16779}
, {16656}
, {10563}
, {4423}
, {18237}
, {3829}
, {16734}
, {20972}
, {13693}
, {10487}
, {24089}
, {9709}
, {21759}
, {14696}
, {17533}
, {14079}
, {30180}
, {9667}
, {24280}
, {30156}
, {3528}
, {24478}
, {22673}
, {22336}
, {25922}
, {17333}
, {17386}
, {16018}
, {20945}
, {23781}
, {14797}
, {15805}
, {17980}
, {18176}
, {12730}
, {15530}
, {14746}
, {4696}
, {11253}
, {21758}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_GROUPS