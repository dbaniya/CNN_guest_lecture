#ifndef CNN_WEIGHTS_H
#define CNN_WEIGHTS_H

#include "ap_fixed.h"

// ---------- types ----------
typedef ap_fixed<16,8> data_t;

// ---------- network sizes ----------
static const int IMG_H = 60;
static const int IMG_W = 80;

static const int C1 = 16;              // conv1 output channels
static const int BOTTLENECK_CH = 8;    // pw output channels
static const int N_CLASSES = 4;

static const int P1_H = 30;            // after maxpool
static const int P1_W = 40;

static const int OH = 5;               // after avgpool(6x8)
static const int OW = 5;

// ---------- weights namespace ----------
namespace weights {

// conv1 (3x3, Cin=1 -> C1)
extern const data_t CONV1_W[9][C1];     // (3*3) x 16
extern const data_t CONV1_B[C1];

// depthwise (3x3 per channel, C1 -> C1)
extern const data_t DW_W[C1][9];        // 16 x (3*3)
extern const data_t DW_B[C1];

// pointwise (1x1, C1 -> BOTTLENECK_CH)
extern const data_t PW_W[C1][BOTTLENECK_CH];  // 16 x 8
extern const data_t PW_B[BOTTLENECK_CH];

// classifier (1x1, BOTTLENECK_CH -> N_CLASSES)
extern const data_t CLS_W[BOTTLENECK_CH][N_CLASSES]; // 8 x 4
extern const data_t CLS_B[N_CLASSES];

} // namespace weights

#endif // CNN_WEIGHTS_H
