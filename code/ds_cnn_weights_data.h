#ifndef CNN_WEIGHTS_DATA_H
#define CNN_WEIGHTS_DATA_H

#include "ds_cnn_weights.h"

namespace weights {

// ================= CONV1 (3x3, 1 -> 16) =================
const data_t CONV1_W[9][C1] =
#include "weights_cstyle_folded/layer_1_conv1_WEIGHTS.txt"
;

const data_t CONV1_B[C1] =
#include "weights_cstyle_folded/layer_1_conv1_BIAS.txt"
;

// ================= DW3x3 (3x3 depthwise, 16 -> 16) =================
const data_t DW_W[C1][9] =
#include "weights_cstyle_folded/layer_2_dw3x3_WEIGHTS.txt"
;

const data_t DW_B[C1] =
#include "weights_cstyle_folded/layer_2_dw3x3_BIAS.txt"
;

// ================= PW1x1 (1x1, 16 -> 8) =================
const data_t PW_W[C1][BOTTLENECK_CH] =
#include "weights_cstyle_folded/layer_3_pw1x1_WEIGHTS.txt"
;

const data_t PW_B[BOTTLENECK_CH] =
#include "weights_cstyle_folded/layer_3_pw1x1_BIAS.txt"
;

// ================= CLS 1x1 (1x1, 8 -> 4) =================
const data_t CLS_W[BOTTLENECK_CH][N_CLASSES] =
#include "weights_cstyle_folded/layer_4_cls_1x1_WEIGHTS.txt"
;

const data_t CLS_B[N_CLASSES] =
#include "weights_cstyle_folded/layer_4_cls_1x1_BIAS.txt"
;

} // namespace weights

#endif // CNN_WEIGHTS_DATA_H
