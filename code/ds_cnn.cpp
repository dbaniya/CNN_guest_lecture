#include "ap_fixed.h"
#include "ap_int.h"
#include "ds_cnn_weights_data.h"

typedef ap_fixed<16,8>  data_t;
typedef ap_fixed<24,12> acc_t;

// ============================================================
// ReLU
// ============================================================
static inline data_t relu_dt(data_t x){
#pragma HLS INLINE
    return (x>0)?x:(data_t)0;
}

// ============================================================
// CONV3x3 + ReLU + 2x2 MAXPOOL  (TF SAME BIT-TRUE)
// ============================================================
static void conv1_relu_maxpool(
    data_t img[IMG_H][IMG_W],
    data_t out[P1_H][P1_W][C1])
{
#pragma HLS INLINE off
#pragma HLS BIND_OP op=mul impl=dsp

for(int ph=0;ph<P1_H;ph++)
for(int pw=0;pw<P1_W;pw++)
for(int oc=0;oc<C1;oc++)
{
#pragma HLS PIPELINE II=1

    data_t maxv=0;
    int oh0=2*ph;
    int ow0=2*pw;

    for(int p=0;p<4;p++){

        int oh=oh0+(p>>1);
        int ow=ow0+(p&1);

        acc_t acc=weights::CONV1_B[oc];

        for(int kh=0;kh<3;kh++)
        for(int kw=0;kw<3;kw++){

            int ih=oh+kh-1;
            int iw=ow+kw-1;

            // 🔴 ALWAYS MAC (even padding)
            data_t inval =
                ((unsigned)ih < IMG_H &&
                 (unsigned)iw < IMG_W)
                ? img[ih][iw]
                : (data_t)0;

            acc += inval *
                   weights::CONV1_W[kh*3+kw][oc];
        }

        data_t v=relu_dt((data_t)acc);
        if(p==0) maxv=v;
        else if(v>maxv) maxv=v;
    }
    out[ph][pw][oc]=maxv;
}
}

// ============================================================
// DEPTHWISE + POINTWISE (TF SAME BIT-TRUE)
// ============================================================
static void depthwise_pointwise_relu_fused(
    data_t in[P1_H][P1_W][C1],
    data_t out[P1_H][P1_W][BOTTLENECK_CH])
{
#pragma HLS INLINE off
#pragma HLS BIND_OP op=mul impl=dsp

for(int h=0;h<P1_H;h++)
for(int w=0;w<P1_W;w++)
{
    data_t dw_relu[C1];
#pragma HLS ARRAY_PARTITION variable=dw_relu complete dim=1

    // -------- DEPTHWISE --------
    for(int ic=0;ic<C1;ic++)
    {
#pragma HLS PIPELINE II=1

        acc_t acc=weights::DW_B[ic];

        for(int kh=0;kh<3;kh++)
        for(int kw=0;kw<3;kw++){

            int ih=h+kh-1;
            int iw=w+kw-1;

            // 🔴 ALWAYS MAC (even padding)
            data_t inval =
                ((unsigned)ih < P1_H &&
                 (unsigned)iw < P1_W)
                ? in[ih][iw][ic]
                : (data_t)0;

            acc += inval *
                   weights::DW_W[ic][kh*3+kw];
        }
        dw_relu[ic]=relu_dt((data_t)acc);
    }

    // -------- POINTWISE --------
    for(int oc=0;oc<BOTTLENECK_CH;oc++)
    {
#pragma HLS PIPELINE II=1

        acc_t acc=weights::PW_B[oc];

        for(int ic=0;ic<C1;ic++)
            acc+=dw_relu[ic]*
                 weights::PW_W[ic][oc];

        out[h][w][oc]=relu_dt((data_t)acc);
    }
}
}

// ============================================================
// AVGPOOL (6x8) → 5x5
// ============================================================
static void avgpool6x8(
    data_t in[P1_H][P1_W][BOTTLENECK_CH],
    data_t out[OH][OW][BOTTLENECK_CH])
{
#pragma HLS INLINE off
const acc_t inv48=(acc_t)1/48;

for(int oh=0;oh<OH;oh++)
for(int ow=0;ow<OW;ow++)
for(int c=0;c<BOTTLENECK_CH;c++)
{
#pragma HLS PIPELINE II=1

    acc_t acc=0;

    for(int ph=0;ph<6;ph++)
    for(int pw=0;pw<8;pw++)
        acc+=in[oh*6+ph][ow*8+pw][c];

    out[oh][ow][c]=(data_t)(acc*inv48);
}
}

// ============================================================
// GAP → CLS (BIT-TRUE TO PYTHON)
// ============================================================
static int cls_gap_logits_argmax(
    data_t bins[OH][OW][BOTTLENECK_CH],
    data_t logits_out[N_CLASSES])
{
#pragma HLS INLINE off

    acc_t gap[BOTTLENECK_CH];
#pragma HLS ARRAY_PARTITION variable=gap complete dim=1

    // -------- GAP FIRST --------
    for(int ic=0; ic<BOTTLENECK_CH; ic++)
        gap[ic]=0;

    for(int h=0; h<OH; h++)
    for(int w=0; w<OW; w++)
    for(int ic=0; ic<BOTTLENECK_CH; ic++)
    {
#pragma HLS PIPELINE II=1
        gap[ic]+=bins[h][w][ic];
    }

    const acc_t inv25=(acc_t)1/(OH*OW);

    for(int ic=0; ic<BOTTLENECK_CH; ic++)
        gap[ic]*=inv25;

    // -------- CLS AFTER GAP --------
    acc_t logits[N_CLASSES];
#pragma HLS ARRAY_PARTITION variable=logits complete dim=1

    for(int o=0;o<N_CLASSES;o++){
        acc_t acc=weights::CLS_B[o];     // add bias ONCE!
        for(int ic=0;ic<BOTTLENECK_CH;ic++)
            acc+=gap[ic]*
                 weights::CLS_W[ic][o];
        logits[o]=acc;
        logits_out[o]=(data_t)acc;
    }

    // -------- ARGMAX --------
    int argmax=0;
    data_t maxv=(data_t)logits[0];

    for(int o=1;o<N_CLASSES;o++){
        data_t v=(data_t)logits[o];
        if(v>maxv){ maxv=v; argmax=o; }
    }
    return argmax;
}

// ============================================================
// TOP
// ============================================================
void ds_cnn(
    const data_t *input_img,
    int *pred_out,
    data_t logits_out[N_CLASSES]
){

#pragma HLS INTERFACE m_axi port=input_img offset=slave bundle=gmem depth=4800
#pragma HLS INTERFACE s_axilite port=input_img bundle=control
#pragma HLS INTERFACE s_axilite port=pred_out bundle=control
#pragma HLS INTERFACE s_axilite port=logits_out bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control


static data_t img[IMG_H][IMG_W];
static data_t pool1[P1_H][P1_W][C1];
static data_t pw[P1_H][P1_W][BOTTLENECK_CH];
static data_t bins[OH][OW][BOTTLENECK_CH];

#pragma HLS BIND_STORAGE variable=img   type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=pool1 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=pw    type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=bins  type=ram_2p impl=lutram

for(int i=0;i<IMG_H*IMG_W;i++){
#pragma HLS PIPELINE II=1
    img[i/IMG_W][i%IMG_W]=input_img[i];
}

conv1_relu_maxpool(img,pool1);
depthwise_pointwise_relu_fused(pool1,pw);
avgpool6x8(pw,bins);

*pred_out = cls_gap_logits_argmax(bins, logits_out);

}





//
// working code with slight mismatches
//#include "ap_fixed.h"
//#include "ap_int.h"
//#include "cnn_weights_data.h"
//
//typedef ap_fixed<16,8>  data_t;
//typedef ap_fixed<24,12> acc_t;
//
//// ============================================================
//// ReLU
//// ============================================================
//static inline data_t relu_dt(data_t x){
//#pragma HLS INLINE
//    return (x>0)?x:(data_t)0;
//}
//
//// ============================================================
//// CONV3x3 + ReLU + 2x2 MAXPOOL  (TF SAME BIT-TRUE)
//// ============================================================
//static void conv1_relu_maxpool(
//    data_t img[IMG_H][IMG_W],
//    data_t out[P1_H][P1_W][C1])
//{
//#pragma HLS INLINE off
//#pragma HLS BIND_OP op=mul impl=dsp
//
//for(int ph=0;ph<P1_H;ph++)
//for(int pw=0;pw<P1_W;pw++)
//for(int oc=0;oc<C1;oc++)
//{
//#pragma HLS PIPELINE II=1
//
//    data_t maxv=0;
//    int oh0=2*ph;
//    int ow0=2*pw;
//
//    for(int p=0;p<4;p++){
//
//        int oh=oh0+(p>>1);
//        int ow=ow0+(p&1);
//
//        acc_t acc=weights::CONV1_B[oc];
//
//        for(int kh=0;kh<3;kh++)
//        for(int kw=0;kw<3;kw++){
//
//            int ih=oh+kh-1;
//            int iw=ow+kw-1;
//
//            data_t inval =
//                ((unsigned)ih < IMG_H &&
//                 (unsigned)iw < IMG_W)
//                ? img[ih][iw]
//                : (data_t)0;
//
//            acc += inval *
//                   weights::CONV1_W[kh*3+kw][oc];
//        }
//
//        data_t v=relu_dt((data_t)acc);
//        if(p==0) maxv=v;
//        else if(v>maxv) maxv=v;
//    }
//    out[ph][pw][oc]=maxv;
//}
//}
//
//// ============================================================
//// DEPTHWISE + POINTWISE (TF SAME BIT-TRUE)
//// ============================================================
//static void depthwise_pointwise_relu_fused(
//    data_t in[P1_H][P1_W][C1],
//    data_t out[P1_H][P1_W][BOTTLENECK_CH])
//{
//#pragma HLS INLINE off
//#pragma HLS BIND_OP op=mul impl=dsp
//
//for(int h=0;h<P1_H;h++)
//for(int w=0;w<P1_W;w++)
//{
//    data_t dw_relu[C1];
//#pragma HLS ARRAY_PARTITION variable=dw_relu complete dim=1
//
//    // -------- DEPTHWISE --------
//    for(int ic=0;ic<C1;ic++)
//    {
//#pragma HLS PIPELINE II=1
//
//        acc_t acc=weights::DW_B[ic];
//
//        for(int kh=0;kh<3;kh++)
//        for(int kw=0;kw<3;kw++){
//
//            int ih=h+kh-1;
//            int iw=w+kw-1;
//
//            data_t inval =
//                ((unsigned)ih < P1_H &&
//                 (unsigned)iw < P1_W)
//                ? in[ih][iw][ic]
//                : (data_t)0;
//
//            acc += inval *
//                   weights::DW_W[ic][kh*3+kw];
//        }
//        dw_relu[ic]=relu_dt((data_t)acc);
//    }
//
//    // -------- POINTWISE --------
//    for(int oc=0;oc<BOTTLENECK_CH;oc++)
//    {
//#pragma HLS PIPELINE II=1
//
//        acc_t acc=weights::PW_B[oc];
//
//        for(int ic=0;ic<C1;ic++)
//            acc+=dw_relu[ic]*
//                 weights::PW_W[ic][oc];
//
//        out[h][w][oc]=relu_dt((data_t)acc);
//    }
//}
//}
//
//// ============================================================
//// AVGPOOL (6x8) → 5x5
//// ============================================================
//static void avgpool6x8(
//    data_t in[P1_H][P1_W][BOTTLENECK_CH],
//    data_t out[OH][OW][BOTTLENECK_CH])
//{
//#pragma HLS INLINE off
//const acc_t inv48=(acc_t)1/48;
//
//for(int oh=0;oh<OH;oh++)
//for(int ow=0;ow<OW;ow++)
//for(int c=0;c<BOTTLENECK_CH;c++)
//{
//#pragma HLS PIPELINE II=1
//
//    acc_t acc=0;
//
//    for(int ph=0;ph<6;ph++)
//    for(int pw=0;pw<8;pw++)
//        acc+=in[oh*6+ph][ow*8+pw][c];
//
//    out[oh][ow][c]=(data_t)(acc*inv48);
//}
//}
//
//// ============================================================
//// 🔴 CLS → GAP → ARGMAX  (TF BIT-TRUE)
//// ============================================================
//static int cls_gap_argmax(
//    data_t bins[OH][OW][BOTTLENECK_CH])
//{
//#pragma HLS INLINE off
//#pragma HLS BIND_OP op=mul impl=dsp
//
//    acc_t logits[N_CLASSES];
//#pragma HLS ARRAY_PARTITION variable=logits complete dim=1
//
//    // initialize logits
//    for(int o=0;o<N_CLASSES;o++)
//        logits[o]=0;
//
//    // CLS per spatial bin (5x5)
//    for(int h=0;h<OH;h++)
//    for(int w=0;w<OW;w++)
//    {
//        for(int o=0;o<N_CLASSES;o++)
//        {
//#pragma HLS PIPELINE II=1
//            acc_t acc=weights::CLS_B[o];
//
//            for(int ic=0;ic<BOTTLENECK_CH;ic++)
//                acc+=bins[h][w][ic]*
//                     weights::CLS_W[ic][o];
//
//            logits[o]+=acc;
//        }
//    }
//
//    // GAP after CLS
//    const acc_t inv25=(acc_t)1/(OH*OW);
//
//    for(int o=0;o<N_CLASSES;o++)
//        logits[o]*=inv25;
//
//    // ARGMAX
//    int argmax=0;
//    data_t maxv=(data_t)logits[0];
//
//    for(int o=1;o<N_CLASSES;o++){
//        data_t v=(data_t)logits[o];
//        if(v>maxv){ maxv=v; argmax=o; }
//    }
//    return argmax;
//}
//
//// ============================================================
//// TOP
//// ============================================================
//int tiny_cnn(const data_t *input_img){
//
//#pragma HLS INTERFACE m_axi port=input_img offset=slave bundle=gmem depth=4800
//#pragma HLS INTERFACE s_axilite port=input_img bundle=control
//#pragma HLS INTERFACE s_axilite port=return bundle=control
//
//static data_t img[IMG_H][IMG_W];
//static data_t pool1[P1_H][P1_W][C1];
//static data_t pw[P1_H][P1_W][BOTTLENECK_CH];
//static data_t bins[OH][OW][BOTTLENECK_CH];
//
//#pragma HLS BIND_STORAGE variable=img   type=ram_2p impl=bram
//#pragma HLS BIND_STORAGE variable=pool1 type=ram_2p impl=bram
//#pragma HLS BIND_STORAGE variable=pw    type=ram_2p impl=bram
//#pragma HLS BIND_STORAGE variable=bins  type=ram_2p impl=lutram
//
//for(int i=0;i<IMG_H*IMG_W;i++){
//#pragma HLS PIPELINE II=1
//    img[i/IMG_W][i%IMG_W]=input_img[i];
//}
//
//conv1_relu_maxpool(img,pool1);
//depthwise_pointwise_relu_fused(pool1,pw);
//avgpool6x8(pw,bins);
//
//return cls_gap_argmax(bins);
//}





//
//#include "ap_fixed.h"
//#include "ap_int.h"
//#include "cnn_weights_data.h"
//
//typedef ap_fixed<16,8>  data_t;
//typedef ap_fixed<24,12> acc_t;
//
//// ============================================================
//// ReLU
//// ============================================================
//static inline data_t relu_dt(data_t x){
//#pragma HLS INLINE
//    return (x>0)?x:(data_t)0;
//}
//
//// ============================================================
//// CONV3x3 + ReLU + 2x2 MAXPOOL  (TF SAME BIT-TRUE)
//// ============================================================
//static void conv1_relu_maxpool(
//    data_t img[IMG_H][IMG_W],
//    data_t out[P1_H][P1_W][C1])
//{
//#pragma HLS INLINE off
//#pragma HLS BIND_OP op=mul impl=dsp
//
//for(int ph=0;ph<P1_H;ph++)
//for(int pw=0;pw<P1_W;pw++)
//for(int oc=0;oc<C1;oc++)
//{
//#pragma HLS PIPELINE II=1
//
//    data_t maxv=0;
//    int oh0=2*ph;
//    int ow0=2*pw;
//
//    for(int p=0;p<4;p++){
//
//        int oh=oh0+(p>>1);
//        int ow=ow0+(p&1);
//
//        acc_t acc=weights::CONV1_B[oc];
//
//        for(int kh=0;kh<3;kh++)
//        for(int kw=0;kw<3;kw++){
//
//            int ih=oh+kh-1;
//            int iw=ow+kw-1;
//
//            // 🔴 ALWAYS MAC (even padding)
//            data_t inval =
//                ((unsigned)ih < IMG_H &&
//                 (unsigned)iw < IMG_W)
//                ? img[ih][iw]
//                : (data_t)0;
//
//            acc += inval *
//                   weights::CONV1_W[kh*3+kw][oc];
//        }
//
//        data_t v=relu_dt((data_t)acc);
//        if(p==0) maxv=v;
//        else if(v>maxv) maxv=v;
//    }
//    out[ph][pw][oc]=maxv;
//}
//}
//
//// ============================================================
//// DEPTHWISE + POINTWISE (TF SAME BIT-TRUE)
//// ============================================================
//static void depthwise_pointwise_relu_fused(
//    data_t in[P1_H][P1_W][C1],
//    data_t out[P1_H][P1_W][BOTTLENECK_CH])
//{
//#pragma HLS INLINE off
//#pragma HLS BIND_OP op=mul impl=dsp
//
//for(int h=0;h<P1_H;h++)
//for(int w=0;w<P1_W;w++)
//{
//    data_t dw_relu[C1];
//#pragma HLS ARRAY_PARTITION variable=dw_relu complete dim=1
//
//    // -------- DEPTHWISE --------
//    for(int ic=0;ic<C1;ic++)
//    {
//#pragma HLS PIPELINE II=1
//
//        acc_t acc=weights::DW_B[ic];
//
//        for(int kh=0;kh<3;kh++)
//        for(int kw=0;kw<3;kw++){
//
//            int ih=h+kh-1;
//            int iw=w+kw-1;
//
//            // 🔴 ALWAYS MAC (even padding)
//            data_t inval =
//                ((unsigned)ih < P1_H &&
//                 (unsigned)iw < P1_W)
//                ? in[ih][iw][ic]
//                : (data_t)0;
//
//            acc += inval *
//                   weights::DW_W[ic][kh*3+kw];
//        }
//        dw_relu[ic]=relu_dt((data_t)acc);
//    }
//
//    // -------- POINTWISE --------
//    for(int oc=0;oc<BOTTLENECK_CH;oc++)
//    {
//#pragma HLS PIPELINE II=1
//
//        acc_t acc=weights::PW_B[oc];
//
//        for(int ic=0;ic<C1;ic++)
//            acc+=dw_relu[ic]*
//                 weights::PW_W[ic][oc];
//
//        out[h][w][oc]=relu_dt((data_t)acc);
//    }
//}
//}
//
//// ============================================================
//// AVGPOOL (6x8) → 5x5
//// ============================================================
//static void avgpool6x8(
//    data_t in[P1_H][P1_W][BOTTLENECK_CH],
//    data_t out[OH][OW][BOTTLENECK_CH])
//{
//#pragma HLS INLINE off
//const acc_t inv48=(acc_t)1/48;
//
//for(int oh=0;oh<OH;oh++)
//for(int ow=0;ow<OW;ow++)
//for(int c=0;c<BOTTLENECK_CH;c++)
//{
//#pragma HLS PIPELINE II=1
//
//    acc_t acc=0;
//
//    for(int ph=0;ph<6;ph++)
//    for(int pw=0;pw<8;pw++)
//        acc+=in[oh*6+ph][ow*8+pw][c];
//
//    out[oh][ow][c]=(data_t)(acc*inv48);
//}
//}
//
//// ============================================================
//// GAP → CLS (BIT-TRUE TO PYTHON)
//// ============================================================
//static int cls_gap_argmax(
//    data_t bins[OH][OW][BOTTLENECK_CH])
//{
//#pragma HLS INLINE off
//
//    acc_t gap[BOTTLENECK_CH];
//#pragma HLS ARRAY_PARTITION variable=gap complete dim=1
//
//    // -------- GAP FIRST --------
//    for(int ic=0; ic<BOTTLENECK_CH; ic++)
//        gap[ic]=0;
//
//    for(int h=0; h<OH; h++)
//    for(int w=0; w<OW; w++)
//    for(int ic=0; ic<BOTTLENECK_CH; ic++)
//    {
//#pragma HLS PIPELINE II=1
//        gap[ic]+=bins[h][w][ic];
//    }
//
//    const acc_t inv25=(acc_t)1/(OH*OW);
//
//    for(int ic=0; ic<BOTTLENECK_CH; ic++)
//        gap[ic]*=inv25;
//
//    // -------- CLS AFTER GAP --------
//    acc_t logits[N_CLASSES];
//#pragma HLS ARRAY_PARTITION variable=logits complete dim=1
//
//    for(int o=0;o<N_CLASSES;o++){
//        acc_t acc=weights::CLS_B[o];     // add bias ONCE!
//        for(int ic=0;ic<BOTTLENECK_CH;ic++)
//            acc+=gap[ic]*
//                 weights::CLS_W[ic][o];
//        logits[o]=acc;
//    }
//
//    // -------- ARGMAX --------
//    int argmax=0;
//    data_t maxv=(data_t)logits[0];
//
//    for(int o=1;o<N_CLASSES;o++){
//        data_t v=(data_t)logits[o];
//        if(v>maxv){ maxv=v; argmax=o; }
//    }
//    return argmax;
//}
//
//// ============================================================
//// TOP
//// ============================================================
//int tiny_cnn(const data_t *input_img){
//
//#pragma HLS INTERFACE m_axi port=input_img offset=slave bundle=gmem depth=4800
//#pragma HLS INTERFACE s_axilite port=input_img bundle=control
//#pragma HLS INTERFACE s_axilite port=return bundle=control
//
//static data_t img[IMG_H][IMG_W];
//static data_t pool1[P1_H][P1_W][C1];
//static data_t pw[P1_H][P1_W][BOTTLENECK_CH];
//static data_t bins[OH][OW][BOTTLENECK_CH];
//
//#pragma HLS BIND_STORAGE variable=img   type=ram_s2p impl=bram
//#pragma HLS BIND_STORAGE variable=pool1 type=ram_s2p impl=bram
//#pragma HLS BIND_STORAGE variable=pw    type=ram_s2p impl=bram
//#pragma HLS BIND_STORAGE variable=bins  type=ram_s2p impl=lutram
//
//for(int i=0;i<IMG_H*IMG_W;i++){
//#pragma HLS PIPELINE II=1
//    img[i/IMG_W][i%IMG_W]=input_img[i];
//}
//
//conv1_relu_maxpool(img,pool1);
//depthwise_pointwise_relu_fused(pool1,pw);
//avgpool6x8(pw,bins);
//
//return cls_gap_argmax(bins);
//}
//




//// ============================================================
//// TinyCNN HLS Inference (ONE FILE) — MATCHES PYTHON BN-FOLDED MODEL
//// ------------------------------------------------------------
//// Uses constants from cnn_weights.h (included via cnn_weights_data.h)
//// so no redefinitions here.
//// ============================================================
//
//#include "ap_fixed.h"
//#include "ap_int.h"
//#include "cnn_weights_data.h"   // (likely includes cnn_weights.h with IMG_H, IMG_W, C1, ...)
//
//// ---------- TYPES (keep consistent with your project) ----------
//typedef ap_fixed<16, 8>  data_t;
//typedef ap_fixed<24, 12> acc_t;
//
//// ---------- ReLU ----------
//static inline data_t relu_dt(data_t x) {
//#pragma HLS INLINE
//    return (x > 0) ? x : (data_t)0;
//}
//
//// ============================================================
//// CONV3x3 SAME + ReLU + 2x2 MaxPool (fused)
//// Python: Conv2D(16,3x3,same,use_bias=False) + BN(folded) + ReLU + MaxPool2x2
//// HLS: uses weights::CONV1_W and BN-folded bias weights::CONV1_B
//// ============================================================
//static void conv1_relu_maxpool(
//    data_t img[IMG_H][IMG_W],
//    data_t out[P1_H][P1_W][C1])
//{
//#pragma HLS INLINE off
//#pragma HLS BIND_OP op=mul impl=dsp
//
//    for (int ph = 0; ph < P1_H; ph++)
//    for (int pw = 0; pw < P1_W; pw++)
//    for (int oc = 0; oc < C1; oc++)
//    {
//#pragma HLS PIPELINE II=1
//
//        data_t maxv = 0;
//        const int oh0 = 2 * ph;
//        const int ow0 = 2 * pw;
//
//        // pool over 2x2 positions
//        for (int p = 0; p < 4; p++) {
//            const int oh = oh0 + (p >> 1);
//            const int ow = ow0 + (p & 1);
//
//            acc_t acc = (acc_t)weights::CONV1_B[oc];
//
//            // 3x3 conv taps (serialized)
//            for (int kh = 0; kh < 3; kh++)
//            for (int kw = 0; kw < 3; kw++) {
//                const int ih = oh + kh - 1;
//                const int iw = ow + kw - 1;
//
//                data_t inval = 0;
//                if ((unsigned)ih < (unsigned)IMG_H && (unsigned)iw < (unsigned)IMG_W)
//                    inval = img[ih][iw];
//
//                const int k = kh * 3 + kw;
//                acc += (acc_t)inval * (acc_t)weights::CONV1_W[k][oc];
//            }
//
//            const data_t v = relu_dt((data_t)acc);
//            if (p == 0) maxv = v;
//            else if (v > maxv) maxv = v;
//        }
//
//        out[ph][pw][oc] = maxv;
//    }
//}
//
//// ============================================================
//// DEPTHWISE 3x3 SAME + ReLU (BN folded => bias included)
//// Python: DepthwiseConv2D(3x3,same,use_bias=False)+BN(folded)+ReLU
//// weights::DW_W[ic][9], weights::DW_B[ic]
//// ============================================================
//static void depthwise_relu(
//    data_t in[P1_H][P1_W][C1],
//    data_t out[P1_H][P1_W][C1])
//{
//#pragma HLS INLINE off
//#pragma HLS BIND_OP op=mul impl=dsp
//
//    for (int h = 0; h < P1_H; h++)
//    for (int w = 0; w < P1_W; w++)
//    for (int ic = 0; ic < C1; ic++)
//    {
//#pragma HLS PIPELINE II=1
//
//        acc_t acc = (acc_t)weights::DW_B[ic];
//
//        for (int kh = 0; kh < 3; kh++)
//        for (int kw = 0; kw < 3; kw++) {
//            const int ih = h + kh - 1;
//            const int iw = w + kw - 1;
//
//            data_t inval = 0;
//            if ((unsigned)ih < (unsigned)P1_H && (unsigned)iw < (unsigned)P1_W)
//                inval = in[ih][iw][ic];
//
//            const int k = kh * 3 + kw;
//            acc += (acc_t)inval * (acc_t)weights::DW_W[ic][k];
//        }
//
//        out[h][w][ic] = relu_dt((data_t)acc);
//    }
//}
//
//// ============================================================
//// POINTWISE 1x1 (C1 -> BOTTLENECK_CH) + ReLU (BN folded)
//// Python: Conv2D(BOTTLENECK_CH,1x1,same,use_bias=False)+BN(folded)+ReLU
//// weights::PW_W[ic][oc], weights::PW_B[oc]
//// ============================================================
//static void pointwise_relu(
//    data_t in[P1_H][P1_W][C1],
//    data_t out[P1_H][P1_W][BOTTLENECK_CH])
//{
//#pragma HLS INLINE off
//#pragma HLS BIND_OP op=mul impl=dsp
//
//    for (int h = 0; h < P1_H; h++)
//    for (int w = 0; w < P1_W; w++)
//    for (int oc = 0; oc < BOTTLENECK_CH; oc++)
//    {
//#pragma HLS PIPELINE II=1
//
//        acc_t acc = (acc_t)weights::PW_B[oc];
//
//        for (int ic = 0; ic < C1; ic++) {
//            acc += (acc_t)in[h][w][ic] * (acc_t)weights::PW_W[ic][oc];
//        }
//
//        out[h][w][oc] = relu_dt((data_t)acc);
//    }
//}
//
//// ============================================================
//// AVGPOOL (6x8) -> (OHxOW) bins
//// Python: AveragePooling2D(pool_size=(6,8)) on (30x40) => (5x5)
//// ============================================================
//static void avgpool6x8(
//    data_t in[P1_H][P1_W][BOTTLENECK_CH],
//    data_t out[OH][OW][BOTTLENECK_CH])
//{
//#pragma HLS INLINE off
//
//    // Vitis HLS 2020.2: avoid ap_fixed/double ambiguity
//    const acc_t inv48 = (acc_t)1 / 48;
//
//    for (int oh = 0; oh < OH; oh++)
//    for (int ow = 0; ow < OW; ow++)
//    for (int c  = 0; c  < BOTTLENECK_CH; c++)
//    {
//#pragma HLS PIPELINE II=1
//
//        acc_t acc = 0;
//        for (int ph = 0; ph < 6; ph++)
//        for (int pw = 0; pw < 8; pw++) {
//            acc += (acc_t)in[oh * 6 + ph][ow * 8 + pw][c];
//        }
//
//        out[oh][ow][c] = (data_t)(acc * inv48);
//    }
//}
//
//// ============================================================
//// CLS 1x1 (NO BN) + GAP(OHxOW) + ARGMAX
//// Python: Conv2D(num_classes,1x1,use_bias=True) then GlobalAveragePooling2D
//// Implemented as: logits = bias + sum_{h,w,ic} bins[h,w,ic]*W[ic,o]
//// then divide by (OH*OW) and argmax.
//// ============================================================
//static int cls_gap_argmax(
//    data_t bins[OH][OW][BOTTLENECK_CH])
//{
//#pragma HLS INLINE off
//
//    acc_t logits[N_CLASSES];
//#pragma HLS ARRAY_PARTITION variable=logits complete dim=1
//
//    for (int o = 0; o < N_CLASSES; o++)
//        logits[o] = (acc_t)weights::CLS_B[o];
//
//    for (int h = 0; h < OH; h++)
//    for (int w = 0; w < OW; w++)
//    for (int ic = 0; ic < BOTTLENECK_CH; ic++)
//    {
//#pragma HLS PIPELINE II=1
//        const acc_t v = (acc_t)bins[h][w][ic];
//        for (int o = 0; o < N_CLASSES; o++)
//            logits[o] += v * (acc_t)weights::CLS_W[ic][o];
//    }
//
//    const acc_t inv25 = (acc_t)1 / (OH * OW); // should be 1/25 for 5x5
//
//    int argmax = 0;
//    data_t maxv = (data_t)(logits[0] * inv25);
//
//    for (int o = 1; o < N_CLASSES; o++) {
//        data_t v = (data_t)(logits[o] * inv25);
//        if (v > maxv) { maxv = v; argmax = o; }
//    }
//    return argmax;
//}
//
//// ============================================================
//// TOP
//// ============================================================
//int tiny_cnn(const data_t *input_img)
//{
//#pragma HLS INTERFACE m_axi     port=input_img offset=slave bundle=gmem depth=4800
//#pragma HLS INTERFACE s_axilite port=input_img bundle=control
//#pragma HLS INTERFACE s_axilite port=return    bundle=control
//
//    static data_t img[IMG_H][IMG_W];
//    static data_t pool1[P1_H][P1_W][C1];
//    static data_t dw[P1_H][P1_W][C1];
//    static data_t pw[P1_H][P1_W][BOTTLENECK_CH];
//    static data_t bins[OH][OW][BOTTLENECK_CH];
//
//#pragma HLS BIND_STORAGE variable=img   type=ram_2p  impl=bram
//#pragma HLS BIND_STORAGE variable=pool1 type=ram_s2p impl=bram
//#pragma HLS BIND_STORAGE variable=dw    type=ram_s2p impl=bram
//#pragma HLS BIND_STORAGE variable=pw    type=ram_s2p impl=lutram
//#pragma HLS BIND_STORAGE variable=bins  type=ram_s2p impl=lutram
//
//    // preload image from AXI into BRAM
//    for (int i = 0; i < IMG_H * IMG_W; i++) {
//#pragma HLS PIPELINE II=1
//        img[i / IMG_W][i % IMG_W] = input_img[i];
//    }
//
//    conv1_relu_maxpool(img, pool1);
//    depthwise_relu(pool1, dw);
//    pointwise_relu(dw, pw);
//    avgpool6x8(pw, bins);
//
//    return cls_gap_argmax(bins);
//}


//
//
//
//// good code
//#include "ap_fixed.h"
//#include "ap_int.h"
//#include "cnn_weights_data.h"
//
//typedef ap_fixed<24,12> acc_t;
//
//// ------------------------------------------------------------
//static inline data_t relu_dt(data_t x){
//#pragma HLS INLINE
//    return (x > 0) ? x : (data_t)0;
//}
//
//static inline acc_t relu_acc(acc_t x){
//#pragma HLS INLINE
//    return (x > 0) ? x : (acc_t)0;
//}
//
//// ============================================================
//// CONV + MAXPOOL
//// ============================================================
//static void conv3x3_relu_maxpool(
//    data_t img[IMG_H][IMG_W],
//    data_t pool_out[P1_H][P1_W][C1])
//{
//#pragma HLS INLINE off
//#pragma HLS BIND_OP op=mul impl=dsp
//#pragma HLS ALLOCATION operation instances=mul limit=16
//#pragma HLS ALLOCATION operation instances=add limit=16
//
//    for(int ph=0; ph<P1_H; ph++)
//    for(int pw=0; pw<P1_W; pw++)
//    for(int oc=0; oc<C1; oc++)
//    {
//#pragma HLS PIPELINE II=2
//
//        data_t maxv=0;
//        int oh0=2*ph;
//        int ow0=2*pw;
//
//        for(int p=0;p<4;p++){
//            int oh=oh0+(p>>1);
//            int ow=ow0+(p&1);
//
//            acc_t acc=(acc_t)weights::CONV1_B[oc];
//
//            for(int kh=0;kh<3;kh++)
//            for(int kw=0;kw<3;kw++){
//#pragma HLS UNROLL
//                int ih=oh+kh-1;
//                int iw=ow+kw-1;
//                data_t inval=0;
//                if((unsigned)ih<IMG_H && (unsigned)iw<IMG_W)
//                    inval=img[ih][iw];
//
//                int k=kh*3+kw;
//                acc+=(acc_t)inval*
//                     (acc_t)weights::CONV1_W[k][oc];
//            }
//
//            data_t v=relu_dt((data_t)acc);
//            if(p==0) maxv=v;
//            else if(v>maxv) maxv=v;
//        }
//        pool_out[ph][pw][oc]=maxv;
//    }
//}
//
//// ============================================================
//// DW + PW (LATENCY OPTIMIZED)
//// ============================================================
//static void depthwise_pointwise_fused_relu(
//    data_t in[P1_H][P1_W][C1],
//    data_t out[P1_H][P1_W][BOTTLENECK_CH])
//{
//#pragma HLS INLINE off
//#pragma HLS BIND_OP op=mul impl=dsp
//#pragma HLS ALLOCATION operation instances=mul limit=16
//#pragma HLS ALLOCATION operation instances=add limit=12
//
//    for(int h=0;h<P1_H;h++)
//    for(int w=0;w<P1_W;w++)
//    {
//        acc_t dw_relu[C1];
//#pragma HLS ARRAY_PARTITION variable=dw_relu cyclic factor=4 dim=1
//
//        // ---- DEPTHWISE ----
//        for(int ic=0;ic<C1;ic++)
//        {
//#pragma HLS PIPELINE II=2
//            acc_t dw_acc=(acc_t)weights::DW_B[ic];
//
//            for(int kh=0;kh<3;kh++)
//            for(int kw=0;kw<3;kw++){
//#pragma HLS UNROLL
//                int ih=h+kh-1;
//                int iw=w+kw-1;
//                data_t inval=0;
//                if((unsigned)ih<P1_H && (unsigned)iw<P1_W)
//                    inval=in[ih][iw][ic];
//
//                int k=kh*3+kw;
//                dw_acc+=(acc_t)inval*
//                        (acc_t)weights::DW_W[ic][k];
//            }
//
//            dw_relu[ic]=relu_acc(dw_acc);
//        }
//
//        // ---- POINTWISE ----
//        for(int oc=0;oc<BOTTLENECK_CH;oc++)
//        {
//#pragma HLS PIPELINE II=1
//            acc_t pw_acc=(acc_t)weights::PW_B[oc];
//
//            for(int ic=0;ic<C1;ic++)
//                pw_acc+=dw_relu[ic]*
//                        (acc_t)weights::PW_W[ic][oc];
//
//            out[h][w][oc]=relu_dt((data_t)pw_acc);
//        }
//    }
//}
//
//// ============================================================
//// AVGPOOL
//// ============================================================
//static void avgpool6x8_to_5x5(
//    data_t in[P1_H][P1_W][BOTTLENECK_CH],
//    data_t bins[OH][OW][BOTTLENECK_CH])
//{
//#pragma HLS INLINE off
//
//    const acc_t inv48=(acc_t)1.0/(acc_t)48.0;
//
//    for(int oh=0;oh<OH;oh++)
//    for(int ow=0;ow<OW;ow++)
//    for(int c=0;c<BOTTLENECK_CH;c++)
//    {
//        acc_t acc=0;
//#pragma HLS PIPELINE II=1
//        for(int ph=0;ph<6;ph++)
//        for(int pw=0;pw<8;pw++)
//            acc+=(acc_t)in[oh*6+ph][ow*8+pw][c];
//
//        bins[oh][ow][c]=(data_t)(acc*inv48);
//    }
//}
//
//// ============================================================
//static int cls1x1_gap_argmax(
//    data_t bins[OH][OW][BOTTLENECK_CH])
//{
//#pragma HLS INLINE off
//
//    acc_t logits[N_CLASSES];
//
//    for(int o=0;o<N_CLASSES;o++)
//        logits[o]=(acc_t)weights::CLS_B[o];
//
//    for(int h=0;h<OH;h++)
//    for(int w=0;w<OW;w++)
//    for(int ic=0;ic<BOTTLENECK_CH;ic++)
//    {
//#pragma HLS PIPELINE II=1
//        acc_t v=(acc_t)bins[h][w][ic];
//        for(int o=0;o<N_CLASSES;o++)
//            logits[o]+=v*(acc_t)weights::CLS_W[ic][o];
//    }
//
//    acc_t inv25=(acc_t)1.0/(acc_t)25.0;
//    data_t logit_avg[N_CLASSES];
//
//    for(int o=0;o<N_CLASSES;o++)
//        logit_avg[o]=(data_t)(logits[o]*inv25);
//
//    int argmax=0;
//    data_t maxv=logit_avg[0];
//
//    for(int o=1;o<N_CLASSES;o++)
//        if(logit_avg[o]>maxv){
//            maxv=logit_avg[o];
//            argmax=o;
//        }
//    return argmax;
//}
//
//// ============================================================
//int tiny_cnn(const data_t *input_img){
//#pragma HLS INTERFACE m_axi     port=input_img offset=slave bundle=gmem depth=4800
//#pragma HLS INTERFACE s_axilite port=input_img bundle=control
//#pragma HLS INTERFACE s_axilite port=return    bundle=control
//
//    static data_t img[IMG_H][IMG_W];
//    static data_t pool1[P1_H][P1_W][C1];
//    static data_t pw[P1_H][P1_W][BOTTLENECK_CH];
//    static data_t bins[OH][OW][BOTTLENECK_CH];
//
//#pragma HLS BIND_STORAGE variable=img   type=ram_2p  impl=bram
//#pragma HLS BIND_STORAGE variable=pool1 type=ram_s2p impl=bram
//#pragma HLS BIND_STORAGE variable=pw    type=ram_s2p impl=bram
//#pragma HLS BIND_STORAGE variable=bins  type=ram_s2p impl=bram
//
//    for(int i=0;i<IMG_H*IMG_W;i++){
//#pragma HLS PIPELINE II=1
//        img[i/IMG_W][i%IMG_W]=input_img[i];
//    }
//
//    conv3x3_relu_maxpool(img,pool1);
//    depthwise_pointwise_fused_relu(pool1,pw);
//    avgpool6x8_to_5x5(pw,bins);
//
//    return cls1x1_gap_argmax(bins);
//}














//// good code
//#include "ap_fixed.h"
//#include "ap_int.h"
//#include "cnn_weights_data.h"
//
//typedef ap_fixed<24,12> acc_t;
//
//// ------------------------------------------------------------
//static inline data_t relu_dt(data_t x){
//#pragma HLS INLINE
//    return (x > 0) ? x : (data_t)0;
//}
//
//static inline acc_t relu_acc(acc_t x){
//#pragma HLS INLINE
//    return (x > 0) ? x : (acc_t)0;
//}
//
//// ============================================================
//// CONV + MAXPOOL
//// ============================================================
//static void conv3x3_relu_maxpool(
//    data_t img[IMG_H][IMG_W],
//    data_t pool_out[P1_H][P1_W][C1])
//{
//#pragma HLS INLINE off
//#pragma HLS BIND_OP op=mul impl=dsp
//#pragma HLS ALLOCATION operation instances=mul limit=16
//#pragma HLS ALLOCATION operation instances=add limit=16
//
//    for(int ph=0; ph<P1_H; ph++)
//    for(int pw=0; pw<P1_W; pw++)
//    for(int oc=0; oc<C1; oc++)
//    {
//#pragma HLS PIPELINE II=2
//
//        data_t maxv=0;
//        int oh0=2*ph;
//        int ow0=2*pw;
//
//        for(int p=0;p<4;p++){
//            int oh=oh0+(p>>1);
//            int ow=ow0+(p&1);
//
//            acc_t acc=(acc_t)weights::CONV1_B[oc];
//
//            for(int kh=0;kh<3;kh++)
//            for(int kw=0;kw<3;kw++){
//#pragma HLS UNROLL
//                int ih=oh+kh-1;
//                int iw=ow+kw-1;
//                data_t inval=0;
//                if((unsigned)ih<IMG_H && (unsigned)iw<IMG_W)
//                    inval=img[ih][iw];
//
//                int k=kh*3+kw;
//                acc+=(acc_t)inval*
//                     (acc_t)weights::CONV1_W[k][oc];
//            }
//
//            data_t v=relu_dt((data_t)acc);
//            if(p==0) maxv=v;
//            else if(v>maxv) maxv=v;
//        }
//        pool_out[ph][pw][oc]=maxv;
//    }
//}
//
//// ============================================================
//// DW + PW (LATENCY OPTIMIZED)
//// ============================================================
//static void depthwise_pointwise_fused_relu(
//    data_t in[P1_H][P1_W][C1],
//    data_t out[P1_H][P1_W][BOTTLENECK_CH])
//{
//#pragma HLS INLINE off
//#pragma HLS BIND_OP op=mul impl=dsp
//#pragma HLS ALLOCATION operation instances=mul limit=16
//#pragma HLS ALLOCATION operation instances=add limit=12
//
//    for(int h=0;h<P1_H;h++)
//    for(int w=0;w<P1_W;w++)
//    {
//        acc_t dw_relu[C1];
//#pragma HLS ARRAY_PARTITION variable=dw_relu cyclic factor=4 dim=1
//
//        // ---- DEPTHWISE ----
//        for(int ic=0;ic<C1;ic++)
//        {
//#pragma HLS PIPELINE II=2
//            acc_t dw_acc=(acc_t)weights::DW_B[ic];
//
//            for(int kh=0;kh<3;kh++)
//            for(int kw=0;kw<3;kw++){
//#pragma HLS UNROLL
//                int ih=h+kh-1;
//                int iw=w+kw-1;
//                data_t inval=0;
//                if((unsigned)ih<P1_H && (unsigned)iw<P1_W)
//                    inval=in[ih][iw][ic];
//
//                int k=kh*3+kw;
//                dw_acc+=(acc_t)inval*
//                        (acc_t)weights::DW_W[ic][k];
//            }
//
//            dw_relu[ic]=relu_acc(dw_acc);
//        }
//
//        // ---- POINTWISE ----
//        for(int oc=0;oc<BOTTLENECK_CH;oc++)
//        {
//#pragma HLS PIPELINE II=1
//            acc_t pw_acc=(acc_t)weights::PW_B[oc];
//
//            for(int ic=0;ic<C1;ic++)
//                pw_acc+=dw_relu[ic]*
//                        (acc_t)weights::PW_W[ic][oc];
//
//            out[h][w][oc]=relu_dt((data_t)pw_acc);
//        }
//    }
//}
//
//// ============================================================
//// AVGPOOL
//// ============================================================
//static void avgpool6x8_to_5x5(
//    data_t in[P1_H][P1_W][BOTTLENECK_CH],
//    data_t bins[OH][OW][BOTTLENECK_CH])
//{
//#pragma HLS INLINE off
//
//    const acc_t inv48=(acc_t)1.0/(acc_t)48.0;
//
//    for(int oh=0;oh<OH;oh++)
//    for(int ow=0;ow<OW;ow++)
//    for(int c=0;c<BOTTLENECK_CH;c++)
//    {
//        acc_t acc=0;
//#pragma HLS PIPELINE II=1
//        for(int ph=0;ph<6;ph++)
//        for(int pw=0;pw<8;pw++)
//            acc+=(acc_t)in[oh*6+ph][ow*8+pw][c];
//
//        bins[oh][ow][c]=(data_t)(acc*inv48);
//    }
//}
//
//// ============================================================
//static int cls1x1_gap_argmax(
//    data_t bins[OH][OW][BOTTLENECK_CH])
//{
//#pragma HLS INLINE off
//
//    acc_t logits[N_CLASSES];
//
//    for(int o=0;o<N_CLASSES;o++)
//        logits[o]=(acc_t)weights::CLS_B[o];
//
//    for(int h=0;h<OH;h++)
//    for(int w=0;w<OW;w++)
//    for(int ic=0;ic<BOTTLENECK_CH;ic++)
//    {
//#pragma HLS PIPELINE II=1
//        acc_t v=(acc_t)bins[h][w][ic];
//        for(int o=0;o<N_CLASSES;o++)
//            logits[o]+=v*(acc_t)weights::CLS_W[ic][o];
//    }
//
//    acc_t inv25=(acc_t)1.0/(acc_t)25.0;
//    data_t logit_avg[N_CLASSES];
//
//    for(int o=0;o<N_CLASSES;o++)
//        logit_avg[o]=(data_t)(logits[o]*inv25);
//
//    int argmax=0;
//    data_t maxv=logit_avg[0];
//
//    for(int o=1;o<N_CLASSES;o++)
//        if(logit_avg[o]>maxv){
//            maxv=logit_avg[o];
//            argmax=o;
//        }
//    return argmax;
//}
//
//// ============================================================
//int tiny_cnn(const data_t *input_img){
//#pragma HLS INTERFACE m_axi     port=input_img offset=slave bundle=gmem depth=4800
//#pragma HLS INTERFACE s_axilite port=input_img bundle=control
//#pragma HLS INTERFACE s_axilite port=return    bundle=control
//
//    static data_t img[IMG_H][IMG_W];
//    static data_t pool1[P1_H][P1_W][C1];
//    static data_t pw[P1_H][P1_W][BOTTLENECK_CH];
//    static data_t bins[OH][OW][BOTTLENECK_CH];
//
//#pragma HLS BIND_STORAGE variable=img   type=ram_2p  impl=bram
//#pragma HLS BIND_STORAGE variable=pool1 type=ram_s2p impl=bram
//#pragma HLS BIND_STORAGE variable=pw    type=ram_s2p impl=bram
//#pragma HLS BIND_STORAGE variable=bins  type=ram_s2p impl=bram
//
//    for(int i=0;i<IMG_H*IMG_W;i++){
//#pragma HLS PIPELINE II=1
//        img[i/IMG_W][i%IMG_W]=input_img[i];
//    }
//
//    conv3x3_relu_maxpool(img,pool1);
//    depthwise_pointwise_fused_relu(pool1,pw);
//    avgpool6x8_to_5x5(pw,bins);
//
//    return cls1x1_gap_argmax(bins);
//}
//
//
//
//






//// good code
//#include "ap_fixed.h"
//#include "ap_int.h"
//#include "cnn_weights_data.h"
//
//typedef ap_fixed<24,12> acc_t;
//
//// ------------------------------------------------------------
//static inline data_t relu_dt(data_t x){
//#pragma HLS INLINE
//    return (x > 0) ? x : (data_t)0;
//}
//
//static inline acc_t relu_acc(acc_t x){
//#pragma HLS INLINE
//    return (x > 0) ? x : (acc_t)0;
//}
//
//// ============================================================
//// CONV + MAXPOOL
//// ============================================================
//static void conv3x3_relu_maxpool(
//    data_t img[IMG_H][IMG_W],
//    data_t pool_out[P1_H][P1_W][C1])
//{
//#pragma HLS INLINE off
//#pragma HLS BIND_OP op=mul impl=dsp
//#pragma HLS ALLOCATION operation instances=mul limit=16
//#pragma HLS ALLOCATION operation instances=add limit=16
//
//    for(int ph=0; ph<P1_H; ph++)
//    for(int pw=0; pw<P1_W; pw++)
//    for(int oc=0; oc<C1; oc++)
//    {
//#pragma HLS PIPELINE II=2
//
//        data_t maxv=0;
//        int oh0=2*ph;
//        int ow0=2*pw;
//
//        for(int p=0;p<4;p++){
//            int oh=oh0+(p>>1);
//            int ow=ow0+(p&1);
//
//            acc_t acc=(acc_t)weights::CONV1_B[oc];
//
//            for(int kh=0;kh<3;kh++)
//            for(int kw=0;kw<3;kw++){
//#pragma HLS UNROLL
//                int ih=oh+kh-1;
//                int iw=ow+kw-1;
//                data_t inval=0;
//                if((unsigned)ih<IMG_H && (unsigned)iw<IMG_W)
//                    inval=img[ih][iw];
//
//                int k=kh*3+kw;
//                acc+=(acc_t)inval*
//                     (acc_t)weights::CONV1_W[k][oc];
//            }
//
//            data_t v=relu_dt((data_t)acc);
//            if(p==0) maxv=v;
//            else if(v>maxv) maxv=v;
//        }
//        pool_out[ph][pw][oc]=maxv;
//    }
//}
//
//// ============================================================
//// DW + PW (LATENCY OPTIMIZED)
//// ============================================================
//static void depthwise_pointwise_fused_relu(
//    data_t in[P1_H][P1_W][C1],
//    data_t out[P1_H][P1_W][BOTTLENECK_CH])
//{
//#pragma HLS INLINE off
//#pragma HLS BIND_OP op=mul impl=dsp
//#pragma HLS ALLOCATION operation instances=mul limit=16
//#pragma HLS ALLOCATION operation instances=add limit=12
//
//    for(int h=0;h<P1_H;h++)
//    for(int w=0;w<P1_W;w++)
//    {
//        acc_t dw_relu[C1];
//#pragma HLS ARRAY_PARTITION variable=dw_relu cyclic factor=4 dim=1
//
//        // ---- DEPTHWISE ----
//        for(int ic=0;ic<C1;ic++)
//        {
//#pragma HLS PIPELINE II=2
//            acc_t dw_acc=(acc_t)weights::DW_B[ic];
//
//            for(int kh=0;kh<3;kh++)
//            for(int kw=0;kw<3;kw++){
//#pragma HLS UNROLL
//                int ih=h+kh-1;
//                int iw=w+kw-1;
//                data_t inval=0;
//                if((unsigned)ih<P1_H && (unsigned)iw<P1_W)
//                    inval=in[ih][iw][ic];
//
//                int k=kh*3+kw;
//                dw_acc+=(acc_t)inval*
//                        (acc_t)weights::DW_W[ic][k];
//            }
//
//            dw_relu[ic]=relu_acc(dw_acc);
//        }
//
//        // ---- POINTWISE ----
//        for(int oc=0;oc<BOTTLENECK_CH;oc++)
//        {
//#pragma HLS PIPELINE II=1
//            acc_t pw_acc=(acc_t)weights::PW_B[oc];
//
//            for(int ic=0;ic<C1;ic++)
//                pw_acc+=dw_relu[ic]*
//                        (acc_t)weights::PW_W[ic][oc];
//
//            out[h][w][oc]=relu_dt((data_t)pw_acc);
//        }
//    }
//}
//
//// ============================================================
//// AVGPOOL
//// ============================================================
//static void avgpool6x8_to_5x5(
//    data_t in[P1_H][P1_W][BOTTLENECK_CH],
//    data_t bins[OH][OW][BOTTLENECK_CH])
//{
//#pragma HLS INLINE off
//
//    const acc_t inv48=(acc_t)1.0/(acc_t)48.0;
//
//    for(int oh=0;oh<OH;oh++)
//    for(int ow=0;ow<OW;ow++)
//    for(int c=0;c<BOTTLENECK_CH;c++)
//    {
//        acc_t acc=0;
//#pragma HLS PIPELINE II=1
//        for(int ph=0;ph<6;ph++)
//        for(int pw=0;pw<8;pw++)
//            acc+=(acc_t)in[oh*6+ph][ow*8+pw][c];
//
//        bins[oh][ow][c]=(data_t)(acc*inv48);
//    }
//}
//
//// ============================================================
//static int cls1x1_gap_argmax(
//    data_t bins[OH][OW][BOTTLENECK_CH])
//{
//#pragma HLS INLINE off
//
//    acc_t logits[N_CLASSES];
//
//    for(int o=0;o<N_CLASSES;o++)
//        logits[o]=(acc_t)weights::CLS_B[o];
//
//    for(int h=0;h<OH;h++)
//    for(int w=0;w<OW;w++)
//    for(int ic=0;ic<BOTTLENECK_CH;ic++)
//    {
//#pragma HLS PIPELINE II=1
//        acc_t v=(acc_t)bins[h][w][ic];
//        for(int o=0;o<N_CLASSES;o++)
//            logits[o]+=v*(acc_t)weights::CLS_W[ic][o];
//    }
//
//    acc_t inv25=(acc_t)1.0/(acc_t)25.0;
//    data_t logit_avg[N_CLASSES];
//
//    for(int o=0;o<N_CLASSES;o++)
//        logit_avg[o]=(data_t)(logits[o]*inv25);
//
//    int argmax=0;
//    data_t maxv=logit_avg[0];
//
//    for(int o=1;o<N_CLASSES;o++)
//        if(logit_avg[o]>maxv){
//            maxv=logit_avg[o];
//            argmax=o;
//        }
//    return argmax;
//}
//
//// ============================================================
//int tiny_cnn(const data_t *input_img){
//#pragma HLS INTERFACE m_axi     port=input_img offset=slave bundle=gmem depth=4800
//#pragma HLS INTERFACE s_axilite port=input_img bundle=control
//#pragma HLS INTERFACE s_axilite port=return    bundle=control
//#pragma HLS DATAFLOW
//
//    static data_t img[IMG_H][IMG_W];
//    static data_t pool1[P1_H][P1_W][C1];
//    static data_t pw[P1_H][P1_W][BOTTLENECK_CH];
//    static data_t bins[OH][OW][BOTTLENECK_CH];
//
//#pragma HLS BIND_STORAGE variable=img   type=ram_2p  impl=bram
//#pragma HLS BIND_STORAGE variable=pool1 type=ram_s2p impl=bram
//#pragma HLS BIND_STORAGE variable=pw    type=ram_s2p impl=bram
//#pragma HLS BIND_STORAGE variable=bins  type=ram_s2p impl=bram
//
//    for(int i=0;i<IMG_H*IMG_W;i++){
//#pragma HLS PIPELINE II=1
//        img[i/IMG_W][i%IMG_W]=input_img[i];
//    }
//
//    conv3x3_relu_maxpool(img,pool1);
//    depthwise_pointwise_fused_relu(pool1,pw);
//    avgpool6x8_to_5x5(pw,bins);
//
//    return cls1x1_gap_argmax(bins);
//}
//
//
//












//#include "ap_fixed.h"
//#include "hls_stream.h"
//#include "hls_vector.h"
//#include "cnn_weights_data.h"
//
//// ============================================================
//// TYPES
//// ============================================================
//
//typedef ap_fixed<24,12> acc_t;
//typedef data_t dt_t;
//
//typedef hls::vector<dt_t,C1> vec_c1_t;
//typedef hls::vector<dt_t,BOTTLENECK_CH> vec_bn_t;
//
//static inline dt_t relu_dt(dt_t x){
//#pragma HLS INLINE
//    return (x>0)?x:(dt_t)0;
//}
//static inline acc_t relu_acc(acc_t x){
//#pragma HLS INLINE
//    return (x>0)?x:(acc_t)0;
//}
//
//// ============================================================
//// MEM → STREAM
//// ============================================================
//
//static void mem2stream(const dt_t *in,hls::stream<dt_t> &s){
//    for(int i=0;i<IMG_H*IMG_W;i++){
//#pragma HLS PIPELINE II=1
//        s.write(in[i]);
//    }
//}
//
//// ============================================================
//// CONV3×3 + RELU + MAXPOOL
//// BN already folded in exported weights
//// ============================================================
//
//static void conv3x3_relu_maxpool_stream(
//    hls::stream<dt_t> &in,
//    hls::stream<vec_c1_t> &out)
//{
//#pragma HLS BIND_OP op=mul impl=dsp
//
//    static dt_t img[IMG_H][IMG_W];
//#pragma HLS BIND_STORAGE variable=img type=ram_2p impl=bram
//
//    for(int i=0;i<IMG_H*IMG_W;i++){
//#pragma HLS PIPELINE II=1
//        img[i/IMG_W][i%IMG_W]=in.read();
//    }
//
//    for(int ph=0; ph<P1_H; ph++)
//    for(int pw=0; pw<P1_W; pw++)
//    {
//#pragma HLS PIPELINE II=1
//        vec_c1_t maxv;
//
//        for(int oc=0; oc<C1; oc++){
//#pragma HLS PIPELINE II=1
//            dt_t maxv_oc=0;
//
//            int oh0=2*ph;
//            int ow0=2*pw;
//
//            for(int p=0;p<4;p++){
//#pragma HLS UNROLL
//                int oh=oh0+(p>>1);
//                int ow=ow0+(p&1);
//
//                acc_t acc=weights::CONV1_B[oc];
//
//                for(int kh=0;kh<3;kh++)
//                for(int kw=0;kw<3;kw++){
//#pragma HLS UNROLL
//                    int ih=oh+kh-1;
//                    int iw=ow+kw-1;
//                    dt_t inval=0;
//
//                    if((unsigned)ih<IMG_H &&
//                       (unsigned)iw<IMG_W)
//                        inval=img[ih][iw];
//
//                    acc+=inval*
//                         weights::CONV1_W[kh*3+kw][oc];
//                }
//
//                dt_t v=relu_dt((dt_t)acc);
//                if(p==0) maxv_oc=v;
//                else if(v>maxv_oc) maxv_oc=v;
//            }
//            maxv[oc]=maxv_oc;
//        }
//        out.write(maxv);
//    }
//}
//
//// ============================================================
//// DW + PW (BN folded)
//// ============================================================
//
//static void dw_pw_stream(
//    hls::stream<vec_c1_t> &in,
//    hls::stream<vec_bn_t> &out)
//{
//    static vec_c1_t inmap[P1_H][P1_W];
//#pragma HLS BIND_STORAGE variable=inmap type=ram_s2p impl=bram
//
//    for(int i=0;i<P1_H*P1_W;i++){
//#pragma HLS PIPELINE II=1
//        inmap[i/P1_W][i%P1_W]=in.read();
//    }
//
//    for(int h=0;h<P1_H;h++)
//    for(int w=0;w<P1_W;w++){
//#pragma HLS PIPELINE II=1
//
//        acc_t dw[C1];
//#pragma HLS ARRAY_PARTITION variable=dw complete
//
//        for(int ic=0;ic<C1;ic++){
//#pragma HLS UNROLL
//            acc_t acc=weights::DW_B[ic];
//
//            for(int kh=0;kh<3;kh++)
//            for(int kw=0;kw<3;kw++){
//#pragma HLS UNROLL
//                int ih=h+kh-1;
//                int iw=w+kw-1;
//                dt_t v=0;
//                if((unsigned)ih<P1_H &&
//                   (unsigned)iw<P1_W)
//                    v=inmap[ih][iw][ic];
//
//                acc+=v*
//                     weights::DW_W[ic][kh*3+kw];
//            }
//            dw[ic]=relu_acc(acc);
//        }
//
//        vec_bn_t o;
//#pragma HLS ARRAY_PARTITION variable=o complete
//
//        for(int oc=0;oc<BOTTLENECK_CH;oc++){
//#pragma HLS UNROLL
//            acc_t acc=weights::PW_B[oc];
//            for(int ic=0;ic<C1;ic++)
//#pragma HLS UNROLL
//                acc+=dw[ic]*
//                     weights::PW_W[ic][oc];
//            o[oc]=relu_dt((dt_t)acc);
//        }
//        out.write(o);
//    }
//}
//
//// ============================================================
//// AVGPOOL 6×8 → 5×5 bins
//// ============================================================
//
//static void avgpool_stream(
//    hls::stream<vec_bn_t> &in,
//    hls::stream<vec_bn_t> &out)
//{
//    static vec_bn_t fmap[P1_H][P1_W];
//#pragma HLS BIND_STORAGE variable=fmap type=ram_s2p impl=bram
//
//    for(int i=0;i<P1_H*P1_W;i++){
//#pragma HLS PIPELINE II=1
//        fmap[i/P1_W][i%P1_W]=in.read();
//    }
//
//    const acc_t inv48=(acc_t)1.0/48;
//
//    for(int oh=0;oh<OH;oh++)
//    for(int ow=0;ow<OW;ow++){
//#pragma HLS PIPELINE II=1
//        vec_bn_t o;
//
//        for(int c=0;c<BOTTLENECK_CH;c++){
//#pragma HLS UNROLL
//            acc_t acc=0;
//            for(int ph=0;ph<6;ph++)
//            for(int pw=0;pw<8;pw++)
//                acc+=fmap[oh*6+ph][ow*8+pw][c];
//
//            o[c]=(dt_t)(acc*inv48);
//        }
//        out.write(o);
//    }
//}
//
//// ============================================================
//// GAP + CLS 1×1  ⭐ PYTHON-COMPATIBLE
//// ============================================================
//
//static int cls_stream(hls::stream<vec_bn_t> &in){
//
//    acc_t gap[BOTTLENECK_CH];
//#pragma HLS ARRAY_PARTITION variable=gap complete
//
//    for(int ic=0;ic<BOTTLENECK_CH;ic++)
//#pragma HLS UNROLL
//        gap[ic]=0;
//
//    for(int i=0;i<OH*OW;i++){
//#pragma HLS PIPELINE II=1
//        vec_bn_t v=in.read();
//        for(int ic=0;ic<BOTTLENECK_CH;ic++)
//#pragma HLS UNROLL
//            gap[ic]+=v[ic];
//    }
//
//    const acc_t inv25=(acc_t)1.0/25;
//    for(int ic=0;ic<BOTTLENECK_CH;ic++)
//#pragma HLS UNROLL
//        gap[ic]*=inv25;
//
//    acc_t logits[N_CLASSES];
//#pragma HLS ARRAY_PARTITION variable=logits complete
//
//    for(int o=0;o<N_CLASSES;o++){
//#pragma HLS UNROLL
//        acc_t acc=weights::CLS_B[o];
//        for(int ic=0;ic<BOTTLENECK_CH;ic++)
//#pragma HLS UNROLL
//            acc+=gap[ic]*
//                 weights::CLS_W[ic][o];
//        logits[o]=acc;
//    }
//
//    int arg=0;
//    acc_t maxv=logits[0];
//
//    for(int o=1;o<N_CLASSES;o++){
//#pragma HLS UNROLL
//        if(logits[o]>maxv){
//            maxv=logits[o];
//            arg=o;
//        }
//    }
//    return arg;
//}
//
//// ============================================================
//// TOP
//// ============================================================
//
//int tiny_cnn(const dt_t *input_img){
//#pragma HLS INTERFACE m_axi     port=input_img offset=slave bundle=gmem
//#pragma HLS INTERFACE s_axilite port=input_img bundle=control
//#pragma HLS INTERFACE s_axilite port=return    bundle=control
//#pragma HLS DATAFLOW
//
//    hls::stream<dt_t> s0;
//    hls::stream<vec_c1_t> s2;
//    hls::stream<vec_bn_t> s3,s4;
//
//#pragma HLS STREAM variable=s0 depth=128
//#pragma HLS STREAM variable=s2 depth=128
//#pragma HLS STREAM variable=s3 depth=128
//#pragma HLS STREAM variable=s4 depth=128
//
//    mem2stream(input_img,s0);
//    conv3x3_relu_maxpool_stream(s0,s2);
//    dw_pw_stream(s2,s3);
//    avgpool_stream(s3,s4);
//    return cls_stream(s4);
//}



// Works but very high LUTRAM
//#include "ap_fixed.h"
//#include "hls_stream.h"
//#include "hls_vector.h"
//#include "cnn_weights_data.h"
//
//// ============================================================
//// TYPES
//// ============================================================
//typedef ap_fixed<24,12> acc_t;
//typedef data_t dt_t;
//
//typedef hls::vector<dt_t,C1> vec_c1_t;
//typedef hls::vector<dt_t,BOTTLENECK_CH> vec_bn_t;
//
//static inline dt_t relu_dt(dt_t x){
//#pragma HLS INLINE
//    return (x>0)?x:(dt_t)0;
//}
//static inline acc_t relu_acc(acc_t x){
//#pragma HLS INLINE
//    return (x>0)?x:(acc_t)0;
//}
//
//// ============================================================
//// LINE BUFFER (NO PRAGMAS HERE)
//// ============================================================
//template<int W, typename T>
//struct LineBuf3 {
//    T r0[W];
//    T r1[W];
//    T r2[W];
//
//    inline void shift_up(int c){
//#pragma HLS INLINE
//        r0[c]=r1[c];
//        r1[c]=r2[c];
//    }
//    inline void insert_bottom(T v,int c){
//#pragma HLS INLINE
//        r2[c]=v;
//    }
//    inline T get(int r,int c){
//#pragma HLS INLINE
//        return (r==0)?r0[c]:(r==1?r1[c]:r2[c]);
//    }
//};
//
//// ============================================================
//// WINDOW (NO PRAGMAS HERE)
//// ============================================================
//template<typename T>
//struct Win3 {
//    T w[3][3];
//
//    inline void shift_left(){
//#pragma HLS INLINE
//        for(int i=0;i<3;i++){
//#pragma HLS UNROLL
//            w[i][0]=w[i][1];
//            w[i][1]=w[i][2];
//        }
//    }
//    inline void insert_col(T a,T b,T c){
//#pragma HLS INLINE
//        w[0][2]=a;
//        w[1][2]=b;
//        w[2][2]=c;
//    }
//    inline T get(int r,int c){
//#pragma HLS INLINE
//        return w[r][c];
//    }
//};
//
//// ============================================================
//// MEM → STREAM
//// ============================================================
//static void mem2stream(const dt_t *in,hls::stream<dt_t> &s){
//    for(int i=0;i<IMG_H*IMG_W;i++){
//#pragma HLS PIPELINE II=1
//        s.write(in[i]);
//    }
//}
//
//// ============================================================
//// CONV + RELU + MAXPOOL
//// ============================================================
//static void conv3x3_relu_maxpool_stream(
//    hls::stream<dt_t> &in,
//    hls::stream<vec_c1_t> &out)
//{
//#pragma HLS BIND_OP op=mul impl=dsp
//#pragma HLS ALLOCATION operation instances=mul limit=16
//
//    static dt_t img[IMG_H][IMG_W];
//#pragma HLS BIND_STORAGE variable=img type=ram_2p impl=bram
//
//    for(int i=0;i<IMG_H*IMG_W;i++){
//#pragma HLS PIPELINE II=1
//        img[i/IMG_W][i%IMG_W]=in.read();
//    }
//
//    for(int ph=0; ph<P1_H; ph++)
//    for(int pw=0; pw<P1_W; pw++)
//    {
//#pragma HLS PIPELINE II=1
//        vec_c1_t maxv;
//
//        for(int oc=0; oc<C1; oc++){
//#pragma HLS UNROLL factor=4
//            dt_t maxv_oc=0;
//
//            int oh0=2*ph;
//            int ow0=2*pw;
//
//            for(int p=0;p<4;p++){
//#pragma HLS UNROLL
//                int oh=oh0+(p>>1);
//                int ow=ow0+(p&1);
//
//                acc_t acc=weights::CONV1_B[oc];
//
//                for(int kh=0;kh<3;kh++)
//                for(int kw=0;kw<3;kw++){
//#pragma HLS UNROLL
//                    int ih=oh+kh-1;
//                    int iw=ow+kw-1;
//                    dt_t inval=0;
//
//                    if((unsigned)ih<IMG_H &&
//                       (unsigned)iw<IMG_W)
//                        inval=img[ih][iw];
//
//                    acc+=inval*
//                         weights::CONV1_W[kh*3+kw][oc];
//                }
//
//                dt_t v=relu_dt((dt_t)acc);
//                if(p==0) maxv_oc=v;
//                else if(v>maxv_oc) maxv_oc=v;
//            }
//            maxv[oc]=maxv_oc;
//        }
//        out.write(maxv);
//    }
//}
//
//// ============================================================
//// DW + PW STREAMING
//// ============================================================
//static void dw_pw_stream(
//    hls::stream<vec_c1_t> &in,
//    hls::stream<vec_bn_t> &out)
//{
//#pragma HLS BIND_OP op=mul impl=dsp
//#pragma HLS ALLOCATION operation instances=mul limit=16
//
//    LineBuf3<P1_W,vec_c1_t> lb;
//#pragma HLS BIND_STORAGE variable=lb.r0 type=ram_s2p impl=bram
//#pragma HLS BIND_STORAGE variable=lb.r1 type=ram_s2p impl=bram
//#pragma HLS BIND_STORAGE variable=lb.r2 type=ram_s2p impl=bram
//
//    Win3<vec_c1_t> win;
//#pragma HLS ARRAY_PARTITION variable=win.w complete dim=0
//
//    vec_c1_t zero;
//#pragma HLS ARRAY_PARTITION variable=zero complete
//    for(int i=0;i<C1;i++){
//#pragma HLS UNROLL
//        zero[i]=0;
//    }
//
//    for(int i=0;i<P1_W;i++){
//#pragma HLS PIPELINE
//        lb.r0[i]=zero;
//        lb.r1[i]=zero;
//        lb.r2[i]=zero;
//    }
//
//    for(int h=0; h<=P1_H; h++)
//    for(int w=0; w<=P1_W; w++){
//#pragma HLS PIPELINE II=1
//
//        vec_c1_t pix=zero;
//        if(h<P1_H && w<P1_W)
//            pix=in.read();
//
//        if(w<P1_W){
//            lb.shift_up(w);
//            lb.insert_bottom(pix,w);
//
//            win.shift_left();
//            win.insert_col(lb.get(0,w),
//                           lb.get(1,w),
//                           lb.get(2,w));
//        }
//
//        if(h>0 && w>0){
//
//            acc_t dw[C1];
//#pragma HLS ARRAY_PARTITION variable=dw cyclic factor=2
//
//            for(int ic=0;ic<C1;ic++){
//#pragma HLS UNROLL factor=2
//                acc_t acc=weights::DW_B[ic];
//
//                for(int kh=0;kh<3;kh++)
//                for(int kw=0;kw<3;kw++){
//#pragma HLS UNROLL
//                    int ih=(h-1)+kh-1;
//                    int iw=(w-1)+kw-1;
//
//                    dt_t v=0;
//                    if(ih>=0&&ih<P1_H&&
//                       iw>=0&&iw<P1_W)
//                        v=win.get(kh,kw)[ic];
//
//                    acc+=v*
//                         weights::DW_W[ic][kh*3+kw];
//                }
//                dw[ic]=relu_acc(acc);
//            }
//
//            vec_bn_t o;
//#pragma HLS ARRAY_PARTITION variable=o complete
//
//            for(int oc=0;oc<BOTTLENECK_CH;oc++){
//                acc_t acc=weights::PW_B[oc];
//                for(int ic=0;ic<C1;ic++){
//#pragma HLS UNROLL factor=2
//                    acc+=dw[ic]*
//                         weights::PW_W[ic][oc];
//                }
//                o[oc]=relu_dt((dt_t)acc);
//            }
//            out.write(o);
//        }
//    }
//}
//
//// ============================================================
//// AVGPOOL STREAMING
//// ============================================================
//static void avgpool_stream(
//    hls::stream<vec_bn_t> &in,
//    hls::stream<vec_bn_t> &out)
//{
//    acc_t acc[OH][OW][BOTTLENECK_CH];
////#pragma HLS ARRAY_PARTITION variable=acc complete dim=3
////#pragma HLS BIND_STORAGE variable=acc type=ram_s2p impl=lutram
//
////#pragma HLS ARRAY_PARTITION variable=acc cyclic factor=2 dim=3
//#pragma HLS BIND_STORAGE variable=acc type=ram_2p impl=bram
//
//    for(int oh=0;oh<OH;oh++)
//    for(int ow=0;ow<OW;ow++)
//    for(int c=0;c<BOTTLENECK_CH;c++){
//#pragma HLS PIPELINE
//        acc[oh][ow][c]=0;
//    }
//
//    for(int h=0;h<P1_H;h++)
//    for(int w=0;w<P1_W;w++){
//#pragma HLS PIPELINE II=1
//        vec_bn_t v=in.read();
//
//        int oh=h/6;
//        int ow=w/8;
//
//        for(int c=0;c<BOTTLENECK_CH;c++){
//#pragma HLS UNROLL factor=2
//            acc[oh][ow][c]+=v[c];
//        }
//    }
//
//    const acc_t inv48=(acc_t)1.0/48;
//
//    for(int oh=0;oh<OH;oh++)
//    for(int ow=0;ow<OW;ow++){
//#pragma HLS PIPELINE II=1
//        vec_bn_t o;
//#pragma HLS ARRAY_PARTITION variable=o complete
//        for(int c=0;c<BOTTLENECK_CH;c++){
//#pragma HLS UNROLL factor=2
//            o[c]=(dt_t)(acc[oh][ow][c]*inv48);
//        }
//        out.write(o);
//    }
//}
//
//// ============================================================
//// CLS
//// ============================================================
//static int cls_stream(hls::stream<vec_bn_t> &in){
//    acc_t logits[N_CLASSES];
//#pragma HLS ARRAY_PARTITION variable=logits complete
//
//    for(int o=0;o<N_CLASSES;o++)
//#pragma HLS UNROLL
//        logits[o]=weights::CLS_B[o];
//
//    for(int i=0;i<OH*OW;i++){
//#pragma HLS PIPELINE II=1
//        vec_bn_t v=in.read();
//        for(int ic=0;ic<BOTTLENECK_CH;ic++)
//        for(int o=0;o<N_CLASSES;o++)
//#pragma HLS UNROLL
//            logits[o]+=v[ic]*
//                       weights::CLS_W[ic][o];
//    }
//
//    const acc_t inv25=(acc_t)1.0/25;
//
//    int arg=0;
//    acc_t maxv=logits[0]*inv25;
//
//    for(int o=1;o<N_CLASSES;o++){
//#pragma HLS UNROLL
//        acc_t vv=logits[o]*inv25;
//        if(vv>maxv){maxv=vv;arg=o;}
//    }
//    return arg;
//}
//
//// ============================================================
//// TOP
//// ============================================================
//int tiny_cnn(const dt_t *input_img){
//#pragma HLS INTERFACE m_axi     port=input_img offset=slave bundle=gmem
//#pragma HLS INTERFACE s_axilite port=input_img bundle=control
//#pragma HLS INTERFACE s_axilite port=return    bundle=control
//#pragma HLS DATAFLOW
//
//    hls::stream<dt_t> s0;
//    hls::stream<vec_c1_t> s2;
//    hls::stream<vec_bn_t> s3,s4;
//
//#pragma HLS STREAM variable=s0 depth=16 type=fifo impl=bram
//#pragma HLS STREAM variable=s2 depth=16 type=fifo impl=bram
//#pragma HLS STREAM variable=s3 depth=16 type=fifo impl=bram
//#pragma HLS STREAM variable=s4 depth=16 type=fifo impl=bram
//
//    mem2stream(input_img,s0);
//    conv3x3_relu_maxpool_stream(s0,s2);
//    dw_pw_stream(s2,s3);
//    avgpool_stream(s3,s4);
//    return cls_stream(s4);
//}
//

//
//#include "ap_fixed.h"
//#include "hls_stream.h"
//#include "hls_vector.h"
//#include "cnn_weights_data.h"
//
//// ============================================================
//// TYPES
//// ============================================================
//typedef ap_fixed<24,12> acc_t;
//typedef data_t dt_t;
//
//typedef hls::vector<dt_t,C1> vec_c1_t;
//typedef hls::vector<dt_t,BOTTLENECK_CH> vec_bn_t;
//
//static inline dt_t relu_dt(dt_t x){
//#pragma HLS INLINE
//    return (x>0)?x:(dt_t)0;
//}
//static inline acc_t relu_acc(acc_t x){
//#pragma HLS INLINE
//    return (x>0)?x:(acc_t)0;
//}
//
//// ============================================================
//// MEM → STREAM
//// ============================================================
//static void mem2stream(const dt_t *in,hls::stream<dt_t> &s){
//    for(int i=0;i<IMG_H*IMG_W;i++){
//#pragma HLS PIPELINE II=1
//        s.write(in[i]);
//    }
//}
//
//// ============================================================
//// CONV + RELU + MAXPOOL  (DSP REUSE SAFE)
//// ============================================================
//static void conv3x3_relu_maxpool_stream(
//    hls::stream<dt_t> &in,
//    hls::stream<vec_c1_t> &out)
//{
//#pragma HLS BIND_OP op=mul impl=dsp
//#pragma HLS ALLOCATION operation instances=mul limit=16
//#pragma HLS ALLOCATION operation instances=add limit=16
//
//    static dt_t img[IMG_H][IMG_W];
//#pragma HLS BIND_STORAGE variable=img type=ram_2p impl=bram
//
//    for(int i=0;i<IMG_H*IMG_W;i++){
//#pragma HLS PIPELINE II=1
//        img[i/IMG_W][i%IMG_W]=in.read();
//    }
//
//    for(int ph=0; ph<P1_H; ph++)
//    for(int pw=0; pw<P1_W; pw++)
//    {
//#pragma HLS PIPELINE II=1
//
//        vec_c1_t maxv;
//
//        for(int oc=0; oc<C1; oc++){
//#pragma HLS UNROLL factor=4
//
//            dt_t maxv_oc=0;
//            int oh0=2*ph;
//            int ow0=2*pw;
//
//            for(int p=0;p<4;p++){
//#pragma HLS UNROLL
//                int oh=oh0+(p>>1);
//                int ow=ow0+(p&1);
//
//                acc_t acc=weights::CONV1_B[oc];
//
//                for(int kh=0;kh<3;kh++)
//                for(int kw=0;kw<3;kw++){
//#pragma HLS UNROLL
//                    int ih=oh+kh-1;
//                    int iw=ow+kw-1;
//                    dt_t inval=0;
//
//                    if((unsigned)ih<IMG_H &&
//                       (unsigned)iw<IMG_W)
//                        inval=img[ih][iw];
//
//                    acc+=inval*
//                         weights::CONV1_W[kh*3+kw][oc];
//                }
//
//                dt_t v=relu_dt((dt_t)acc);
//                if(p==0) maxv_oc=v;
//                else if(v>maxv_oc) maxv_oc=v;
//            }
//            maxv[oc]=maxv_oc;
//        }
//        out.write(maxv);
//    }
//}
//
//// ============================================================
//// DW + PW (DSP SAFE)
//// ============================================================
//static void dw_pw_stream(
//    hls::stream<vec_c1_t> &in,
//    hls::stream<vec_bn_t> &out)
//{
//#pragma HLS BIND_OP op=mul impl=dsp
//#pragma HLS ALLOCATION operation instances=mul limit=16
//#pragma HLS ALLOCATION operation instances=add limit=16
//
//    static vec_c1_t inmap[P1_H][P1_W];
//#pragma HLS BIND_STORAGE variable=inmap type=ram_s2p impl=bram
//
//    for(int i=0;i<P1_H*P1_W;i++){
//#pragma HLS PIPELINE II=1
//        inmap[i/P1_W][i%P1_W]=in.read();
//    }
//
//    for(int h=0;h<P1_H;h++)
//    for(int w=0;w<P1_W;w++){
//#pragma HLS PIPELINE II=1
//
//        acc_t dw[C1];
//#pragma HLS ARRAY_PARTITION variable=dw cyclic factor=4
//
//        for(int ic=0;ic<C1;ic++){
//#pragma HLS UNROLL factor=4
//            acc_t acc=weights::DW_B[ic];
//
//            for(int kh=0;kh<3;kh++)
//            for(int kw=0;kw<3;kw++){
//#pragma HLS UNROLL
//                int ih=h+kh-1;
//                int iw=w+kw-1;
//                dt_t v=0;
//                if((unsigned)ih<P1_H &&
//                   (unsigned)iw<P1_W)
//                    v=inmap[ih][iw][ic];
//
//                acc+=v*
//                     weights::DW_W[ic][kh*3+kw];
//            }
//            dw[ic]=relu_acc(acc);
//        }
//
//        vec_bn_t o;
//#pragma HLS ARRAY_PARTITION variable=o complete dim=1
//
//        for(int oc=0;oc<BOTTLENECK_CH;oc++){
//#pragma HLS UNROLL factor=2
//
//            acc_t acc=weights::PW_B[oc];
//
//            for(int ic=0;ic<C1;ic++){
//#pragma HLS UNROLL factor=4
//                acc+=dw[ic]*weights::PW_W[ic][oc];
//            }
//            o[oc]=relu_dt((dt_t)acc);
//        }
//        out.write(o);
//    }
//}
//
//// ============================================================
//// AVGPOOL 6x8
//// ============================================================
//static void avgpool_stream(
//    hls::stream<vec_bn_t> &in,
//    hls::stream<vec_bn_t> &out)
//{
//    static vec_bn_t fmap[P1_H][P1_W];
//#pragma HLS BIND_STORAGE variable=fmap type=ram_s2p impl=bram
//
//    for(int i=0;i<P1_H*P1_W;i++){
//#pragma HLS PIPELINE II=1
//        fmap[i/P1_W][i%P1_W]=in.read();
//    }
//
//    const acc_t inv48=(acc_t)1.0/48;
//
//    for(int oh=0;oh<OH;oh++)
//    for(int ow=0;ow<OW;ow++){
//#pragma HLS PIPELINE II=1
//        vec_bn_t o;
//
//        for(int c=0;c<BOTTLENECK_CH;c++){
//#pragma HLS UNROLL factor=2
//            acc_t acc=0;
//            for(int ph=0;ph<6;ph++)
//            for(int pw=0;pw<8;pw++)
//                acc+=fmap[oh*6+ph][ow*8+pw][c];
//
//            o[c]=(dt_t)(acc*inv48);
//        }
//        out.write(o);
//    }
//}
//
//// ============================================================
//// CLS
//// ============================================================
//static int cls_stream(hls::stream<vec_bn_t> &in){
//    acc_t logits[N_CLASSES];
//#pragma HLS ARRAY_PARTITION variable=logits complete
//
//    for(int o=0;o<N_CLASSES;o++)
//#pragma HLS UNROLL
//        logits[o]=weights::CLS_B[o];
//
//    for(int i=0;i<OH*OW;i++){
//#pragma HLS PIPELINE II=1
//        vec_bn_t v=in.read();
//        for(int ic=0;ic<BOTTLENECK_CH;ic++)
//        for(int o=0;o<N_CLASSES;o++)
//#pragma HLS UNROLL
//            logits[o]+=v[ic]*weights::CLS_W[ic][o];
//    }
//
//    const acc_t inv25=(acc_t)1.0/25;
//
//    int arg=0;
//    acc_t maxv=logits[0]*inv25;
//
//    for(int o=1;o<N_CLASSES;o++){
//#pragma HLS UNROLL
//        acc_t vv=logits[o]*inv25;
//        if(vv>maxv){maxv=vv;arg=o;}
//    }
//    return arg;
//}
//
//// ============================================================
//// TOP
//// ============================================================
//int tiny_cnn(const dt_t *input_img){
//#pragma HLS INTERFACE m_axi     port=input_img offset=slave bundle=gmem
//#pragma HLS INTERFACE s_axilite port=input_img bundle=control
//#pragma HLS INTERFACE s_axilite port=return    bundle=control
//#pragma HLS DATAFLOW
//
//    hls::stream<dt_t> s0;
//    hls::stream<vec_c1_t> s2;
//    hls::stream<vec_bn_t> s3,s4;
//
//#pragma HLS STREAM variable=s0 depth=32
//#pragma HLS STREAM variable=s2 depth=32
//#pragma HLS STREAM variable=s3 depth=32
//#pragma HLS STREAM variable=s4 depth=32
//
//    mem2stream(input_img,s0);
//    conv3x3_relu_maxpool_stream(s0,s2);
//    dw_pw_stream(s2,s3);
//    avgpool_stream(s3,s4);
//    return cls_stream(s4);
//}



//streaming work but too many resources
//#include "ap_fixed.h"
//#include "hls_stream.h"
//#include "hls_vector.h"
//#include "cnn_weights_data.h"
//
//// ============================================================
//// TYPES
//// ============================================================
//typedef ap_fixed<24,12> acc_t;
//typedef data_t dt_t;
//
//typedef hls::vector<dt_t,C1> vec_c1_t;
//typedef hls::vector<dt_t,BOTTLENECK_CH> vec_bn_t;
//
//static inline dt_t relu_dt(dt_t x){
//#pragma HLS INLINE
//    return (x>0)?x:(dt_t)0;
//}
//static inline acc_t relu_acc(acc_t x){
//#pragma HLS INLINE
//    return (x>0)?x:(acc_t)0;
//}
//
//// ============================================================
//// MEM → STREAM
//// ============================================================
//static void mem2stream(const dt_t *in,hls::stream<dt_t> &s){
//    for(int i=0;i<IMG_H*IMG_W;i++){
//#pragma HLS PIPELINE II=1
//        s.write(in[i]);
//    }
//}
//
//// ============================================================
//// ✅ CONV + RELU + MAXPOOL (FUSED)
//// EXACT SAME math as BRAM version
//// ============================================================
//static void conv3x3_relu_maxpool_stream(
//    hls::stream<dt_t> &in,
//    hls::stream<vec_c1_t> &out)
//{
//#pragma HLS BIND_OP op=mul impl=dsp
//#pragma HLS ALLOCATION operation instances=mul limit=64
//#pragma HLS ALLOCATION operation instances=add limit=64
//
//    static dt_t img[IMG_H][IMG_W];
//#pragma HLS BIND_STORAGE variable=img type=ram_2p impl=bram
//
//    // load whole image (still streaming in)
//    for(int i=0;i<IMG_H*IMG_W;i++){
//#pragma HLS PIPELINE II=1
//        img[i/IMG_W][i%IMG_W]=in.read();
//    }
//
//    for(int ph=0; ph<P1_H; ph++)
//    for(int pw=0; pw<P1_W; pw++)
//    {
//#pragma HLS PIPELINE II=1
//        vec_c1_t maxv;
//
//        for(int oc=0; oc<C1; oc++){
////#pragma HLS UNROLL
//#pragma HLS PIPELINE II=1
//            dt_t maxv_oc=0;
//
//            int oh0=2*ph;
//            int ow0=2*pw;
//
//            for(int p=0;p<4;p++){
//#pragma HLS UNROLL
//                int oh=oh0+(p>>1);
//                int ow=ow0+(p&1);
//
//                acc_t acc=weights::CONV1_B[oc];
//
//                for(int kh=0;kh<3;kh++)
//                for(int kw=0;kw<3;kw++){
//#pragma HLS UNROLL
//                    int ih=oh+kh-1;
//                    int iw=ow+kw-1;
//                    dt_t inval=0;
//
//                    if((unsigned)ih<IMG_H &&
//                       (unsigned)iw<IMG_W)
//                        inval=img[ih][iw];
//
//                    acc+=inval*
//                         weights::CONV1_W[kh*3+kw][oc];
//                }
//
//                dt_t v=relu_dt((dt_t)acc);
//                if(p==0) maxv_oc=v;
//                else if(v>maxv_oc) maxv_oc=v;
//            }
//            maxv[oc]=maxv_oc;
//        }
//        out.write(maxv);
//    }
//}
//
//// ============================================================
//// DW + PW
//// ============================================================
//static void dw_pw_stream(
//    hls::stream<vec_c1_t> &in,
//    hls::stream<vec_bn_t> &out)
//{
//    static vec_c1_t inmap[P1_H][P1_W];
//#pragma HLS BIND_STORAGE variable=inmap type=ram_s2p impl=bram
//
//    for(int i=0;i<P1_H*P1_W;i++){
//#pragma HLS PIPELINE II=1
//        inmap[i/P1_W][i%P1_W]=in.read();
//    }
//
//    for(int h=0;h<P1_H;h++)
//    for(int w=0;w<P1_W;w++){
//#pragma HLS PIPELINE II=1
//
//        acc_t dw[C1];
//#pragma HLS ARRAY_PARTITION variable=dw cyclic factor=4
//
//        for(int ic=0;ic<C1;ic++){
//#pragma HLS UNROLL
//            acc_t acc=weights::DW_B[ic];
//
//            for(int kh=0;kh<3;kh++)
//            for(int kw=0;kw<3;kw++){
//#pragma HLS UNROLL
//                int ih=h+kh-1;
//                int iw=w+kw-1;
//                dt_t v=0;
//                if((unsigned)ih<P1_H &&
//                   (unsigned)iw<P1_W)
//                    v=inmap[ih][iw][ic];
//
//                acc+=v*
//                     weights::DW_W[ic][kh*3+kw];
//            }
//            dw[ic]=relu_acc(acc);
//        }
//
//        vec_bn_t o;
//#pragma HLS ARRAY_PARTITION variable=o complete dim=1
//
//        for(int oc=0;oc<BOTTLENECK_CH;oc++){
//#pragma HLS UNROLL
//            acc_t acc=weights::PW_B[oc];
//            for(int ic=0;ic<C1;ic++)
//#pragma HLS UNROLL
//                acc+=dw[ic]*
//                     weights::PW_W[ic][oc];
//            o[oc]=relu_dt((dt_t)acc);
//        }
//        out.write(o);
//    }
//}
//
//// ============================================================
//// AVGPOOL 6x8 (EXACT MATCH)
//// ============================================================
//static void avgpool_stream(
//    hls::stream<vec_bn_t> &in,
//    hls::stream<vec_bn_t> &out)
//{
//    static vec_bn_t fmap[P1_H][P1_W];
//#pragma HLS BIND_STORAGE variable=fmap type=ram_s2p impl=bram
//
//    for(int i=0;i<P1_H*P1_W;i++){
//#pragma HLS PIPELINE II=1
//        fmap[i/P1_W][i%P1_W]=in.read();
//    }
//
//    const acc_t inv48=(acc_t)1.0/48;
//
//    for(int oh=0;oh<OH;oh++)
//    for(int ow=0;ow<OW;ow++){
//#pragma HLS PIPELINE II=1
//        vec_bn_t o;
//
//        for(int c=0;c<BOTTLENECK_CH;c++){
//#pragma HLS UNROLL
//            acc_t acc=0;
//            for(int ph=0;ph<6;ph++)
//            for(int pw=0;pw<8;pw++)
//                acc+=fmap[oh*6+ph][ow*8+pw][c];
//
//            o[c]=(dt_t)(acc*inv48);
//        }
//        out.write(o);
//    }
//}
//
//// ============================================================
//// CLS
//// ============================================================
//static int cls_stream(hls::stream<vec_bn_t> &in){
//    acc_t logits[N_CLASSES];
//#pragma HLS ARRAY_PARTITION variable=logits complete
//
//    for(int o=0;o<N_CLASSES;o++)
//#pragma HLS UNROLL
//        logits[o]=weights::CLS_B[o];
//
//    for(int i=0;i<OH*OW;i++){
//#pragma HLS PIPELINE II=1
//        vec_bn_t v=in.read();
//        for(int ic=0;ic<BOTTLENECK_CH;ic++)
//        for(int o=0;o<N_CLASSES;o++)
//#pragma HLS UNROLL
//            logits[o]+=v[ic]*
//                       weights::CLS_W[ic][o];
//    }
//
//    const acc_t inv25=(acc_t)1.0/25;
//
//    int arg=0;
//    acc_t maxv=logits[0]*inv25;
//
//    for(int o=1;o<N_CLASSES;o++){
//#pragma HLS UNROLL
//        acc_t vv=logits[o]*inv25;
//        if(vv>maxv){maxv=vv;arg=o;}
//    }
//    return arg;
//}
//
//// ============================================================
//// TOP
//// ============================================================
//int tiny_cnn(const dt_t *input_img){
//#pragma HLS INTERFACE m_axi     port=input_img offset=slave bundle=gmem
//#pragma HLS INTERFACE s_axilite port=input_img bundle=control
//#pragma HLS INTERFACE s_axilite port=return    bundle=control
//#pragma HLS DATAFLOW
//
//    hls::stream<dt_t> s0;
//    hls::stream<vec_c1_t> s2;
//    hls::stream<vec_bn_t> s3,s4;
//
//#pragma HLS STREAM variable=s0 depth=64
//#pragma HLS STREAM variable=s2 depth=64
//#pragma HLS STREAM variable=s3 depth=64
//#pragma HLS STREAM variable=s4 depth=64
//
//    mem2stream(input_img,s0);
//    conv3x3_relu_maxpool_stream(s0,s2);
//    dw_pw_stream(s2,s3);
//    avgpool_stream(s3,s4);
//    return cls_stream(s4);
//}


//
//
//// good code
//#include "ap_fixed.h"
//#include "ap_int.h"
//#include "cnn_weights_data.h"
//
//typedef ap_fixed<24,12> acc_t;
//
//// ------------------------------------------------------------
//static inline data_t relu_dt(data_t x){
//#pragma HLS INLINE
//    return (x > 0) ? x : (data_t)0;
//}
//
//static inline acc_t relu_acc(acc_t x){
//#pragma HLS INLINE
//    return (x > 0) ? x : (acc_t)0;
//}
//
//// ============================================================
//// CONV + MAXPOOL
//// ============================================================
//static void conv3x3_relu_maxpool(
//    data_t img[IMG_H][IMG_W],
//    data_t pool_out[P1_H][P1_W][C1])
//{
//#pragma HLS INLINE off
//#pragma HLS BIND_OP op=mul impl=dsp
//#pragma HLS ALLOCATION operation instances=mul limit=16
//#pragma HLS ALLOCATION operation instances=add limit=16
//
//    for(int ph=0; ph<P1_H; ph++)
//    for(int pw=0; pw<P1_W; pw++)
//    for(int oc=0; oc<C1; oc++)
//    {
//#pragma HLS PIPELINE II=2
//
//        data_t maxv=0;
//        int oh0=2*ph;
//        int ow0=2*pw;
//
//        for(int p=0;p<4;p++){
//            int oh=oh0+(p>>1);
//            int ow=ow0+(p&1);
//
//            acc_t acc=(acc_t)weights::CONV1_B[oc];
//
//            for(int kh=0;kh<3;kh++)
//            for(int kw=0;kw<3;kw++){
//#pragma HLS UNROLL
//                int ih=oh+kh-1;
//                int iw=ow+kw-1;
//                data_t inval=0;
//                if((unsigned)ih<IMG_H && (unsigned)iw<IMG_W)
//                    inval=img[ih][iw];
//
//                int k=kh*3+kw;
//                acc+=(acc_t)inval*
//                     (acc_t)weights::CONV1_W[k][oc];
//            }
//
//            data_t v=relu_dt((data_t)acc);
//            if(p==0) maxv=v;
//            else if(v>maxv) maxv=v;
//        }
//        pool_out[ph][pw][oc]=maxv;
//    }
//}
//
//// ============================================================
//// DW + PW (LATENCY OPTIMIZED)
//// ============================================================
//static void depthwise_pointwise_fused_relu(
//    data_t in[P1_H][P1_W][C1],
//    data_t out[P1_H][P1_W][BOTTLENECK_CH])
//{
//#pragma HLS INLINE off
//#pragma HLS BIND_OP op=mul impl=dsp
//#pragma HLS ALLOCATION operation instances=mul limit=16
//#pragma HLS ALLOCATION operation instances=add limit=12
//
//    for(int h=0;h<P1_H;h++)
//    for(int w=0;w<P1_W;w++)
//    {
//        acc_t dw_relu[C1];
//#pragma HLS ARRAY_PARTITION variable=dw_relu cyclic factor=4 dim=1
//
//        // ---- DEPTHWISE ----
//        for(int ic=0;ic<C1;ic++)
//        {
//#pragma HLS PIPELINE II=2
//            acc_t dw_acc=(acc_t)weights::DW_B[ic];
//
//            for(int kh=0;kh<3;kh++)
//            for(int kw=0;kw<3;kw++){
//#pragma HLS UNROLL
//                int ih=h+kh-1;
//                int iw=w+kw-1;
//                data_t inval=0;
//                if((unsigned)ih<P1_H && (unsigned)iw<P1_W)
//                    inval=in[ih][iw][ic];
//
//                int k=kh*3+kw;
//                dw_acc+=(acc_t)inval*
//                        (acc_t)weights::DW_W[ic][k];
//            }
//
//            dw_relu[ic]=relu_acc(dw_acc);
//        }
//
//        // ---- POINTWISE ----
//        for(int oc=0;oc<BOTTLENECK_CH;oc++)
//        {
//#pragma HLS PIPELINE II=1
//            acc_t pw_acc=(acc_t)weights::PW_B[oc];
//
//            for(int ic=0;ic<C1;ic++)
//                pw_acc+=dw_relu[ic]*
//                        (acc_t)weights::PW_W[ic][oc];
//
//            out[h][w][oc]=relu_dt((data_t)pw_acc);
//        }
//    }
//}
//
//// ============================================================
//// AVGPOOL
//// ============================================================
//static void avgpool6x8_to_5x5(
//    data_t in[P1_H][P1_W][BOTTLENECK_CH],
//    data_t bins[OH][OW][BOTTLENECK_CH])
//{
//#pragma HLS INLINE off
//
//    const acc_t inv48=(acc_t)1.0/(acc_t)48.0;
//
//    for(int oh=0;oh<OH;oh++)
//    for(int ow=0;ow<OW;ow++)
//    for(int c=0;c<BOTTLENECK_CH;c++)
//    {
//        acc_t acc=0;
//#pragma HLS PIPELINE II=1
//        for(int ph=0;ph<6;ph++)
//        for(int pw=0;pw<8;pw++)
//            acc+=(acc_t)in[oh*6+ph][ow*8+pw][c];
//
//        bins[oh][ow][c]=(data_t)(acc*inv48);
//    }
//}
//
//// ============================================================
//static int cls1x1_gap_argmax(
//    data_t bins[OH][OW][BOTTLENECK_CH])
//{
//#pragma HLS INLINE off
//
//    acc_t logits[N_CLASSES];
//
//    for(int o=0;o<N_CLASSES;o++)
//        logits[o]=(acc_t)weights::CLS_B[o];
//
//    for(int h=0;h<OH;h++)
//    for(int w=0;w<OW;w++)
//    for(int ic=0;ic<BOTTLENECK_CH;ic++)
//    {
//#pragma HLS PIPELINE II=1
//        acc_t v=(acc_t)bins[h][w][ic];
//        for(int o=0;o<N_CLASSES;o++)
//            logits[o]+=v*(acc_t)weights::CLS_W[ic][o];
//    }
//
//    acc_t inv25=(acc_t)1.0/(acc_t)25.0;
//    data_t logit_avg[N_CLASSES];
//
//    for(int o=0;o<N_CLASSES;o++)
//        logit_avg[o]=(data_t)(logits[o]*inv25);
//
//    int argmax=0;
//    data_t maxv=logit_avg[0];
//
//    for(int o=1;o<N_CLASSES;o++)
//        if(logit_avg[o]>maxv){
//            maxv=logit_avg[o];
//            argmax=o;
//        }
//    return argmax;
//}
//
//// ============================================================
//int tiny_cnn(const data_t *input_img){
//#pragma HLS INTERFACE m_axi     port=input_img offset=slave bundle=gmem depth=4800
//#pragma HLS INTERFACE s_axilite port=input_img bundle=control
//#pragma HLS INTERFACE s_axilite port=return    bundle=control
//#pragma HLS DATAFLOW
//
//    static data_t img[IMG_H][IMG_W];
//    static data_t pool1[P1_H][P1_W][C1];
//    static data_t pw[P1_H][P1_W][BOTTLENECK_CH];
//    static data_t bins[OH][OW][BOTTLENECK_CH];
//
//#pragma HLS BIND_STORAGE variable=img   type=ram_2p  impl=bram
//#pragma HLS BIND_STORAGE variable=pool1 type=ram_s2p impl=bram
//#pragma HLS BIND_STORAGE variable=pw    type=ram_s2p impl=bram
//#pragma HLS BIND_STORAGE variable=bins  type=ram_s2p impl=bram
//
//    for(int i=0;i<IMG_H*IMG_W;i++){
//#pragma HLS PIPELINE II=1
//        img[i/IMG_W][i%IMG_W]=input_img[i];
//    }
//
//    conv3x3_relu_maxpool(img,pool1);
//    depthwise_pointwise_fused_relu(pool1,pw);
//    avgpool6x8_to_5x5(pw,bins);
//
//    return cls1x1_gap_argmax(bins);
//}
//
//
//








//
//
//
//// good code
//#include "ap_fixed.h"
//#include "ap_int.h"
//#include "cnn_weights_data.h"
//
//typedef ap_fixed<24,12> acc_t;
//
//// ------------------------------------------------------------
//static inline data_t relu_dt(data_t x){
//#pragma HLS INLINE
//    return (x > 0) ? x : (data_t)0;
//}
//
//static inline acc_t relu_acc(acc_t x){
//#pragma HLS INLINE
//    return (x > 0) ? x : (acc_t)0;
//}
//
//// ============================================================
//// CONV + MAXPOOL
//// ============================================================
//static void conv3x3_relu_maxpool(
//    data_t img[IMG_H][IMG_W],
//    data_t pool_out[P1_H][P1_W][C1])
//{
//#pragma HLS INLINE off
//#pragma HLS BIND_OP op=mul impl=dsp
//#pragma HLS ALLOCATION operation instances=mul limit=16
//#pragma HLS ALLOCATION operation instances=add limit=16
//
//    for(int ph=0; ph<P1_H; ph++)
//    for(int pw=0; pw<P1_W; pw++)
//    for(int oc=0; oc<C1; oc++)
//    {
//#pragma HLS PIPELINE II=2
//
//        data_t maxv=0;
//        int oh0=2*ph;
//        int ow0=2*pw;
//
//        for(int p=0;p<4;p++){
//            int oh=oh0+(p>>1);
//            int ow=ow0+(p&1);
//
//            acc_t acc=(acc_t)weights::CONV1_B[oc];
//
//            for(int kh=0;kh<3;kh++)
//            for(int kw=0;kw<3;kw++){
//#pragma HLS UNROLL
//                int ih=oh+kh-1;
//                int iw=ow+kw-1;
//                data_t inval=0;
//                if((unsigned)ih<IMG_H && (unsigned)iw<IMG_W)
//                    inval=img[ih][iw];
//
//                int k=kh*3+kw;
//                acc+=(acc_t)inval*
//                     (acc_t)weights::CONV1_W[k][oc];
//            }
//
//            data_t v=relu_dt((data_t)acc);
//            if(p==0) maxv=v;
//            else if(v>maxv) maxv=v;
//        }
//        pool_out[ph][pw][oc]=maxv;
//    }
//}
//
//// ============================================================
//// DW + PW (LATENCY OPTIMIZED)
//// ============================================================
//static void depthwise_pointwise_fused_relu(
//    data_t in[P1_H][P1_W][C1],
//    data_t out[P1_H][P1_W][BOTTLENECK_CH])
//{
//#pragma HLS INLINE off
//#pragma HLS BIND_OP op=mul impl=dsp
//#pragma HLS ALLOCATION operation instances=mul limit=16
//#pragma HLS ALLOCATION operation instances=add limit=12
//
//    for(int h=0;h<P1_H;h++)
//    for(int w=0;w<P1_W;w++)
//    {
//        acc_t dw_relu[C1];
//#pragma HLS ARRAY_PARTITION variable=dw_relu cyclic factor=4 dim=1
//
//        // ---- DEPTHWISE ----
//        for(int ic=0;ic<C1;ic++)
//        {
//#pragma HLS PIPELINE II=2
//            acc_t dw_acc=(acc_t)weights::DW_B[ic];
//
//            for(int kh=0;kh<3;kh++)
//            for(int kw=0;kw<3;kw++){
//#pragma HLS UNROLL
//                int ih=h+kh-1;
//                int iw=w+kw-1;
//                data_t inval=0;
//                if((unsigned)ih<P1_H && (unsigned)iw<P1_W)
//                    inval=in[ih][iw][ic];
//
//                int k=kh*3+kw;
//                dw_acc+=(acc_t)inval*
//                        (acc_t)weights::DW_W[ic][k];
//            }
//
//            dw_relu[ic]=relu_acc(dw_acc);
//        }
//
//        // ---- POINTWISE ----
//        for(int oc=0;oc<BOTTLENECK_CH;oc++)
//        {
//#pragma HLS PIPELINE II=1
//            acc_t pw_acc=(acc_t)weights::PW_B[oc];
//
//            for(int ic=0;ic<C1;ic++)
//                pw_acc+=dw_relu[ic]*
//                        (acc_t)weights::PW_W[ic][oc];
//
//            out[h][w][oc]=relu_dt((data_t)pw_acc);
//        }
//    }
//}
//
//// ============================================================
//// AVGPOOL
//// ============================================================
//static void avgpool6x8_to_5x5(
//    data_t in[P1_H][P1_W][BOTTLENECK_CH],
//    data_t bins[OH][OW][BOTTLENECK_CH])
//{
//#pragma HLS INLINE off
//
//    const acc_t inv48=(acc_t)1.0/(acc_t)48.0;
//
//    for(int oh=0;oh<OH;oh++)
//    for(int ow=0;ow<OW;ow++)
//    for(int c=0;c<BOTTLENECK_CH;c++)
//    {
//        acc_t acc=0;
//#pragma HLS PIPELINE II=1
//        for(int ph=0;ph<6;ph++)
//        for(int pw=0;pw<8;pw++)
//            acc+=(acc_t)in[oh*6+ph][ow*8+pw][c];
//
//        bins[oh][ow][c]=(data_t)(acc*inv48);
//    }
//}
//
//// ============================================================
//static int cls1x1_gap_argmax(
//    data_t bins[OH][OW][BOTTLENECK_CH])
//{
//#pragma HLS INLINE off
//
//    acc_t logits[N_CLASSES];
//
//    for(int o=0;o<N_CLASSES;o++)
//        logits[o]=(acc_t)weights::CLS_B[o];
//
//    for(int h=0;h<OH;h++)
//    for(int w=0;w<OW;w++)
//    for(int ic=0;ic<BOTTLENECK_CH;ic++)
//    {
//#pragma HLS PIPELINE II=1
//        acc_t v=(acc_t)bins[h][w][ic];
//        for(int o=0;o<N_CLASSES;o++)
//            logits[o]+=v*(acc_t)weights::CLS_W[ic][o];
//    }
//
//    acc_t inv25=(acc_t)1.0/(acc_t)25.0;
//    data_t logit_avg[N_CLASSES];
//
//    for(int o=0;o<N_CLASSES;o++)
//        logit_avg[o]=(data_t)(logits[o]*inv25);
//
//    int argmax=0;
//    data_t maxv=logit_avg[0];
//
//    for(int o=1;o<N_CLASSES;o++)
//        if(logit_avg[o]>maxv){
//            maxv=logit_avg[o];
//            argmax=o;
//        }
//    return argmax;
//}
//
//// ============================================================
//int tiny_cnn(const data_t *input_img){
//#pragma HLS INTERFACE m_axi     port=input_img offset=slave bundle=gmem depth=4800
//#pragma HLS INTERFACE s_axilite port=input_img bundle=control
//#pragma HLS INTERFACE s_axilite port=return    bundle=control
//
//    static data_t img[IMG_H][IMG_W];
//    static data_t pool1[P1_H][P1_W][C1];
//    static data_t pw[P1_H][P1_W][BOTTLENECK_CH];
//    static data_t bins[OH][OW][BOTTLENECK_CH];
//
//#pragma HLS BIND_STORAGE variable=img   type=ram_2p  impl=bram
//#pragma HLS BIND_STORAGE variable=pool1 type=ram_s2p impl=bram
//#pragma HLS BIND_STORAGE variable=pw    type=ram_s2p impl=bram
//#pragma HLS BIND_STORAGE variable=bins  type=ram_s2p impl=bram
//
//    for(int i=0;i<IMG_H*IMG_W;i++){
//#pragma HLS PIPELINE II=1
//        img[i/IMG_W][i%IMG_W]=input_img[i];
//    }
//
//    conv3x3_relu_maxpool(img,pool1);
//    depthwise_pointwise_fused_relu(pool1,pw);
//    avgpool6x8_to_5x5(pw,bins);
//
//    return cls1x1_gap_argmax(bins);
//}
//
//
//
//




//
//
//
//// good code
//#include "ap_fixed.h"
//#include "ap_int.h"
//#include "cnn_weights_data.h"
//
//typedef ap_fixed<24,12> acc_t;
//
//// ------------------------------------------------------------
//static inline data_t relu_dt(data_t x){
//#pragma HLS INLINE
//    return (x > 0) ? x : (data_t)0;
//}
//
//static inline acc_t relu_acc(acc_t x){
//#pragma HLS INLINE
//    return (x > 0) ? x : (acc_t)0;
//}
//
//// ============================================================
//// CONV + MAXPOOL
//// ============================================================
//static void conv3x3_relu_maxpool(
//    data_t img[IMG_H][IMG_W],
//    data_t pool_out[P1_H][P1_W][C1])
//{
//#pragma HLS INLINE off
//#pragma HLS BIND_OP op=mul impl=dsp
//#pragma HLS ALLOCATION operation instances=mul limit=16
//#pragma HLS ALLOCATION operation instances=add limit=16
//
//    for(int ph=0; ph<P1_H; ph++)
//    for(int pw=0; pw<P1_W; pw++)
//    for(int oc=0; oc<C1; oc++)
//    {
//#pragma HLS PIPELINE II=2
//
//        data_t maxv=0;
//        int oh0=2*ph;
//        int ow0=2*pw;
//
//        for(int p=0;p<4;p++){
//            int oh=oh0+(p>>1);
//            int ow=ow0+(p&1);
//
//            acc_t acc=(acc_t)weights::CONV1_B[oc];
//
//            for(int kh=0;kh<3;kh++)
//            for(int kw=0;kw<3;kw++){
//#pragma HLS UNROLL
//                int ih=oh+kh-1;
//                int iw=ow+kw-1;
//                data_t inval=0;
//                if((unsigned)ih<IMG_H && (unsigned)iw<IMG_W)
//                    inval=img[ih][iw];
//
//                int k=kh*3+kw;
//                acc+=(acc_t)inval*
//                     (acc_t)weights::CONV1_W[k][oc];
//            }
//
//            data_t v=relu_dt((data_t)acc);
//            if(p==0) maxv=v;
//            else if(v>maxv) maxv=v;
//        }
//        pool_out[ph][pw][oc]=maxv;
//    }
//}
//
//// ============================================================
//// DW + PW (LATENCY OPTIMIZED)
//// ============================================================
//static void depthwise_pointwise_fused_relu(
//    data_t in[P1_H][P1_W][C1],
//    data_t out[P1_H][P1_W][BOTTLENECK_CH])
//{
//#pragma HLS INLINE off
//#pragma HLS BIND_OP op=mul impl=dsp
//#pragma HLS ALLOCATION operation instances=mul limit=16
//#pragma HLS ALLOCATION operation instances=add limit=12
//
//    for(int h=0;h<P1_H;h++)
//    for(int w=0;w<P1_W;w++)
//    {
//        acc_t dw_relu[C1];
//#pragma HLS ARRAY_PARTITION variable=dw_relu cyclic factor=4 dim=1
//
//        // ---- DEPTHWISE ----
//        for(int ic=0;ic<C1;ic++)
//        {
//#pragma HLS PIPELINE II=2
//            acc_t dw_acc=(acc_t)weights::DW_B[ic];
//
//            for(int kh=0;kh<3;kh++)
//            for(int kw=0;kw<3;kw++){
//#pragma HLS UNROLL
//                int ih=h+kh-1;
//                int iw=w+kw-1;
//                data_t inval=0;
//                if((unsigned)ih<P1_H && (unsigned)iw<P1_W)
//                    inval=in[ih][iw][ic];
//
//                int k=kh*3+kw;
//                dw_acc+=(acc_t)inval*
//                        (acc_t)weights::DW_W[ic][k];
//            }
//
//            dw_relu[ic]=relu_acc(dw_acc);
//        }
//
//        // ---- POINTWISE ----
//        for(int oc=0;oc<BOTTLENECK_CH;oc++)
//        {
//#pragma HLS PIPELINE II=1
//            acc_t pw_acc=(acc_t)weights::PW_B[oc];
//
//            for(int ic=0;ic<C1;ic++)
//                pw_acc+=dw_relu[ic]*
//                        (acc_t)weights::PW_W[ic][oc];
//
//            out[h][w][oc]=relu_dt((data_t)pw_acc);
//        }
//    }
//}
//
//// ============================================================
//// AVGPOOL
//// ============================================================
//static void avgpool6x8_to_5x5(
//    data_t in[P1_H][P1_W][BOTTLENECK_CH],
//    data_t bins[OH][OW][BOTTLENECK_CH])
//{
//#pragma HLS INLINE off
//
//    const acc_t inv48=(acc_t)1.0/(acc_t)48.0;
//
//    for(int oh=0;oh<OH;oh++)
//    for(int ow=0;ow<OW;ow++)
//    for(int c=0;c<BOTTLENECK_CH;c++)
//    {
//        acc_t acc=0;
//#pragma HLS PIPELINE II=1
//        for(int ph=0;ph<6;ph++)
//        for(int pw=0;pw<8;pw++)
//            acc+=(acc_t)in[oh*6+ph][ow*8+pw][c];
//
//        bins[oh][ow][c]=(data_t)(acc*inv48);
//    }
//}
//
//// ============================================================
//static int cls1x1_gap_argmax(
//    data_t bins[OH][OW][BOTTLENECK_CH])
//{
//#pragma HLS INLINE off
//
//    acc_t logits[N_CLASSES];
//
//    for(int o=0;o<N_CLASSES;o++)
//        logits[o]=(acc_t)weights::CLS_B[o];
//
//    for(int h=0;h<OH;h++)
//    for(int w=0;w<OW;w++)
//    for(int ic=0;ic<BOTTLENECK_CH;ic++)
//    {
//#pragma HLS PIPELINE II=1
//        acc_t v=(acc_t)bins[h][w][ic];
//        for(int o=0;o<N_CLASSES;o++)
//            logits[o]+=v*(acc_t)weights::CLS_W[ic][o];
//    }
//
//    acc_t inv25=(acc_t)1.0/(acc_t)25.0;
//    data_t logit_avg[N_CLASSES];
//
//    for(int o=0;o<N_CLASSES;o++)
//        logit_avg[o]=(data_t)(logits[o]*inv25);
//
//    int argmax=0;
//    data_t maxv=logit_avg[0];
//
//    for(int o=1;o<N_CLASSES;o++)
//        if(logit_avg[o]>maxv){
//            maxv=logit_avg[o];
//            argmax=o;
//        }
//    return argmax;
//}
//
//// ============================================================
//int tiny_cnn(const data_t *input_img){
//#pragma HLS INTERFACE m_axi     port=input_img offset=slave bundle=gmem depth=4800
//#pragma HLS INTERFACE s_axilite port=input_img bundle=control
//#pragma HLS INTERFACE s_axilite port=return    bundle=control
//
//    static data_t img[IMG_H][IMG_W];
//    static data_t pool1[P1_H][P1_W][C1];
//    static data_t pw[P1_H][P1_W][BOTTLENECK_CH];
//    static data_t bins[OH][OW][BOTTLENECK_CH];
//
//#pragma HLS BIND_STORAGE variable=img   type=ram_2p  impl=bram
//#pragma HLS BIND_STORAGE variable=pool1 type=ram_s2p impl=bram
//#pragma HLS BIND_STORAGE variable=pw    type=ram_s2p impl=bram
//#pragma HLS BIND_STORAGE variable=bins  type=ram_s2p impl=bram
//
//    for(int i=0;i<IMG_H*IMG_W;i++){
//#pragma HLS PIPELINE II=1
//        img[i/IMG_W][i%IMG_W]=input_img[i];
//    }
//
//    conv3x3_relu_maxpool(img,pool1);
//    depthwise_pointwise_fused_relu(pool1,pw);
//    avgpool6x8_to_5x5(pw,bins);
//
//    return cls1x1_gap_argmax(bins);
//}
//
//
//
//




//
//#include "ap_fixed.h"
//#include "ap_int.h"
//#include "cnn_weights_data.h"   // includes cnn_weights.h too
//
//// data_t comes from cnn_weights.h
//typedef ap_fixed<24,12> acc_t;
//
//// ------------------------------------------------------------
//static inline data_t relu_dt(data_t x){
//#pragma HLS INLINE
//    return (x > 0) ? x : (data_t)0;
//}
//
//static inline acc_t relu_acc(acc_t x){
//#pragma HLS INLINE
//    return (x > 0) ? x : (acc_t)0;
//}
//
//// ------------------------------------------------------------
//// Conv3x3 SAME + ReLU + 2x2 MaxPool
//// ------------------------------------------------------------
//static void conv3x3_relu_maxpool(
//    data_t img[IMG_H][IMG_W],
//    data_t pool_out[P1_H][P1_W][C1])
//{
//#pragma HLS INLINE off
//
//    for(int ph=0; ph<P1_H; ph++)
//    for(int pw=0; pw<P1_W; pw++)
//    for(int oc=0; oc<C1; oc++)
//    {
//        data_t maxv=0;
//        int oh0=2*ph;
//        int ow0=2*pw;
//
//        for(int p=0;p<4;p++){
//            int oh=oh0+(p>>1);
//            int ow=ow0+(p&1);
//
//            acc_t acc=(acc_t)weights::CONV1_B[oc];
//
//            for(int kh=0;kh<3;kh++)
//            for(int kw=0;kw<3;kw++){
//                int ih=oh+kh-1;
//                int iw=ow+kw-1;
//                data_t inval=0;
//                if((unsigned)ih<IMG_H && (unsigned)iw<IMG_W)
//                    inval=img[ih][iw];
//
//                int k=kh*3+kw;
//                acc+=(acc_t)inval*(acc_t)weights::CONV1_W[k][oc];
//            }
//
//            data_t v=relu_dt((data_t)acc);
//            if(p==0) maxv=v;
//            else if(v>maxv) maxv=v;
//        }
//        pool_out[ph][pw][oc]=maxv;
//    }
//}
//
//// ------------------------------------------------------------
//// LATENCY-OPTIMIZED DW + PW FUSED
//// Depthwise computed once per pixel → reused for PW
//// ------------------------------------------------------------
//static void depthwise_pointwise_fused_relu(
//    data_t in[P1_H][P1_W][C1],
//    data_t out[P1_H][P1_W][BOTTLENECK_CH])
//{
//#pragma HLS INLINE off
//
//// keep MACs in DSP
//#pragma HLS BIND_OP op=mul impl=dsp
//
//// bound parallelism (Z7010 safe)
//#pragma HLS ALLOCATION operation instances=mul limit=4
//#pragma HLS ALLOCATION operation instances=add limit=2
//
//    for(int h=0;h<P1_H;h++)
//    for(int w=0;w<P1_W;w++)
//    {
//        // ---- Depthwise once per pixel ----
//        acc_t dw_relu[C1];
//#pragma HLS ARRAY_PARTITION variable=dw_relu cyclic factor=2 dim=1
//
//        for(int ic=0;ic<C1;ic++)
//        {
//            acc_t dw_acc=(acc_t)weights::DW_B[ic];
//
//            for(int kh=0;kh<3;kh++)
//            for(int kw=0;kw<3;kw++)
//            {
//                int ih=h+kh-1;
//                int iw=w+kw-1;
//
//                data_t inval=0;
//                if((unsigned)ih<P1_H && (unsigned)iw<P1_W)
//                    inval=in[ih][iw][ic];
//
//                int k=kh*3+kw;
//                dw_acc+=(acc_t)inval*
//                        (acc_t)weights::DW_W[ic][k];
//            }
//
//            dw_relu[ic]=relu_acc(dw_acc);
//        }
//
//        // ---- Pointwise reuse ----
//        for(int oc=0;oc<BOTTLENECK_CH;oc++)
//        {
//            acc_t pw_acc=(acc_t)weights::PW_B[oc];
//
//            for(int ic=0;ic<C1;ic++)
//                pw_acc+=dw_relu[ic]*
//                        (acc_t)weights::PW_W[ic][oc];
//
//            out[h][w][oc]=relu_dt((data_t)pw_acc);
//        }
//    }
//}
//
//// ------------------------------------------------------------
//static void avgpool6x8_to_5x5(
//    data_t in[P1_H][P1_W][BOTTLENECK_CH],
//    data_t bins[OH][OW][BOTTLENECK_CH])
//{
//#pragma HLS INLINE off
//
//    const acc_t denom=(acc_t)48;
//
//    for(int oh=0;oh<OH;oh++)
//    for(int ow=0;ow<OW;ow++)
//    for(int c=0;c<BOTTLENECK_CH;c++)
//    {
//        acc_t acc=0;
//#pragma HLS PIPELINE II=2
//        for(int ph=0;ph<6;ph++)
//        for(int pw=0;pw<8;pw++)
//            acc+=(acc_t)in[oh*6+ph][ow*8+pw][c];
//
//        bins[oh][ow][c]=(data_t)(acc/denom);
//    }
//}
//
//// ------------------------------------------------------------
//static int cls1x1_gap_argmax(
//    data_t bins[OH][OW][BOTTLENECK_CH])
//{
//#pragma HLS INLINE off
//
//    acc_t logits[N_CLASSES];
//
//    for(int o=0;o<N_CLASSES;o++)
//        logits[o]=(acc_t)weights::CLS_B[o];
//
//    for(int h=0;h<OH;h++)
//    for(int w=0;w<OW;w++)
//    for(int ic=0;ic<BOTTLENECK_CH;ic++)
//    {
//        acc_t v=(acc_t)bins[h][w][ic];
//        for(int o=0;o<N_CLASSES;o++)
//            logits[o]+=v*(acc_t)weights::CLS_W[ic][o];
//    }
//
//    acc_t inv25=(acc_t)1.0/(acc_t)25.0;
//    data_t logit_avg[N_CLASSES];
//
//    for(int o=0;o<N_CLASSES;o++)
//        logit_avg[o]=(data_t)(logits[o]*inv25);
//
//    int argmax=0;
//    data_t maxv=logit_avg[0];
//
//    for(int o=1;o<N_CLASSES;o++)
//        if(logit_avg[o]>maxv){
//            maxv=logit_avg[o];
//            argmax=o;
//        }
//    return argmax;
//}
//
//// ------------------------------------------------------------
//int tiny_cnn(const data_t *input_img){
//#pragma HLS INTERFACE m_axi     port=input_img offset=slave bundle=gmem depth=4800
//#pragma HLS INTERFACE s_axilite port=input_img bundle=control
//#pragma HLS INTERFACE s_axilite port=return    bundle=control
//
//    static data_t img[IMG_H][IMG_W];
//    static data_t pool1[P1_H][P1_W][C1];
//    static data_t pw[P1_H][P1_W][BOTTLENECK_CH];
//    static data_t bins[OH][OW][BOTTLENECK_CH];
//
//#pragma HLS BIND_STORAGE variable=img   type=ram_2p  impl=bram
//#pragma HLS BIND_STORAGE variable=pool1 type=ram_s2p impl=bram
//#pragma HLS BIND_STORAGE variable=pw    type=ram_s2p impl=bram
//#pragma HLS BIND_STORAGE variable=bins  type=ram_s2p impl=bram
//
//    for(int i=0;i<IMG_H*IMG_W;i++){
//#pragma HLS PIPELINE II=1
//        img[i/IMG_W][i%IMG_W]=input_img[i];
//    }
//
//    conv3x3_relu_maxpool(img,pool1);
//    depthwise_pointwise_fused_relu(pool1,pw);
//    avgpool6x8_to_5x5(pw,bins);
//
//    return cls1x1_gap_argmax(bins);
//}
//




//
//#include "ap_fixed.h"
//#include "ap_int.h"
//#include "cnn_weights_data.h"   // includes cnn_weights.h too
//
//// data_t comes from cnn_weights.h
//typedef ap_fixed<24,12> acc_t;
//
//// ------------------------------------------------------------
//static inline data_t relu_dt(data_t x){
//#pragma HLS INLINE
//    return (x > 0) ? x : (data_t)0;
//}
//
//static inline acc_t relu_acc(acc_t x){
//#pragma HLS INLINE
//    return (x > 0) ? x : (acc_t)0;
//}
//
//// ------------------------------------------------------------
//// Conv3x3 SAME + ReLU + 2x2 MaxPool
//// ------------------------------------------------------------
//static void conv3x3_relu_maxpool(
//    data_t img[IMG_H][IMG_W],
//    data_t pool_out[P1_H][P1_W][C1])
//{
//#pragma HLS INLINE off
//
//    for(int ph=0; ph<P1_H; ph++)
//    for(int pw=0; pw<P1_W; pw++)
//    for(int oc=0; oc<C1; oc++)
//    {
//        data_t maxv=0;
//        int oh0=2*ph;
//        int ow0=2*pw;
//
//        for(int p=0;p<4;p++){
//            int oh=oh0+(p>>1);
//            int ow=ow0+(p&1);
//
//            acc_t acc=(acc_t)weights::CONV1_B[oc];
//
//            for(int kh=0;kh<3;kh++)
//            for(int kw=0;kw<3;kw++){
//                int ih=oh+kh-1;
//                int iw=ow+kw-1;
//                data_t inval=0;
//                if((unsigned)ih<IMG_H && (unsigned)iw<IMG_W)
//                    inval=img[ih][iw];
//
//                int k=kh*3+kw;
//                acc+=(acc_t)inval*(acc_t)weights::CONV1_W[k][oc];
//            }
//
//            data_t v=relu_dt((data_t)acc);
//            if(p==0) maxv=v;
//            else if(v>maxv) maxv=v;
//        }
//        pool_out[ph][pw][oc]=maxv;
//    }
//}
//
//// ------------------------------------------------------------
//// SERIAL DW + PW FUSED  (Z7010 LUT SAFE)
//// ------------------------------------------------------------
//static void depthwise_pointwise_fused_relu(
//    data_t in[P1_H][P1_W][C1],
//    data_t out[P1_H][P1_W][BOTTLENECK_CH])
//{
//#pragma HLS INLINE off
//
//// FORCE DSP MAC (fixed-point mul)
//#pragma HLS BIND_OP op=mul impl=dsp
//
//// limit add/mul instances to avoid parallel LUT adder trees
//#pragma HLS ALLOCATION operation instances=add limit=2
//#pragma HLS ALLOCATION operation instances=mul limit=4
//
//    for(int h=0;h<P1_H;h++)
//    for(int w=0;w<P1_W;w++)
//    {
//        // SERIALIZE PW OUTPUTS
//        for(int oc=0;oc<BOTTLENECK_CH;oc++)
//        {
//            acc_t pw_acc=(acc_t)weights::PW_B[oc];
//
//            for(int ic=0;ic<C1;ic++)
//            {
//                acc_t dw_acc=(acc_t)weights::DW_B[ic];
//
//                for(int kh=0;kh<3;kh++)
//                for(int kw=0;kw<3;kw++)
//                {
//                    int ih=h+kh-1;
//                    int iw=w+kw-1;
//                    data_t inval=0;
//                    if((unsigned)ih<P1_H && (unsigned)iw<P1_W)
//                        inval=in[ih][iw][ic];
//
//                    int k=kh*3+kw;
//                    dw_acc+=(acc_t)inval*
//                            (acc_t)weights::DW_W[ic][k];
//                }
//
//                dw_acc=relu_acc(dw_acc);
//
//                pw_acc+=dw_acc*
//                        (acc_t)weights::PW_W[ic][oc];
//            }
//
//            out[h][w][oc]=relu_dt((data_t)pw_acc);
//        }
//    }
//}
//
//// ------------------------------------------------------------
//static void avgpool6x8_to_5x5(
//    data_t in[P1_H][P1_W][BOTTLENECK_CH],
//    data_t bins[OH][OW][BOTTLENECK_CH])
//{
//#pragma HLS INLINE off
//    const acc_t denom=(acc_t)48;
//
//    for(int oh=0;oh<OH;oh++)
//    for(int ow=0;ow<OW;ow++)
//    for(int c=0;c<BOTTLENECK_CH;c++)
//    {
//        acc_t acc=0;
//        for(int ph=0;ph<6;ph++)
//        for(int pw=0;pw<8;pw++)
//            acc+=(acc_t)in[oh*6+ph][ow*8+pw][c];
//
//        bins[oh][ow][c]=(data_t)(acc/denom);
//    }
//}
//
//// ------------------------------------------------------------
//static int cls1x1_gap_argmax(
//    data_t bins[OH][OW][BOTTLENECK_CH])
//{
//#pragma HLS INLINE off
//
//    acc_t logits[N_CLASSES];
//
//    for(int o=0;o<N_CLASSES;o++)
//        logits[o]=(acc_t)weights::CLS_B[o];
//
//    for(int h=0;h<OH;h++)
//    for(int w=0;w<OW;w++)
//    for(int ic=0;ic<BOTTLENECK_CH;ic++)
//    {
//        acc_t v=(acc_t)bins[h][w][ic];
//        for(int o=0;o<N_CLASSES;o++)
//            logits[o]+=v*(acc_t)weights::CLS_W[ic][o];
//    }
//
//    acc_t inv25=(acc_t)1.0/(acc_t)25.0;
//    data_t logit_avg[N_CLASSES];
//
//    for(int o=0;o<N_CLASSES;o++)
//        logit_avg[o]=(data_t)(logits[o]*inv25);
//
//    int argmax=0;
//    data_t maxv=logit_avg[0];
//
//    for(int o=1;o<N_CLASSES;o++)
//        if(logit_avg[o]>maxv){
//            maxv=logit_avg[o];
//            argmax=o;
//        }
//    return argmax;
//}
//
//// ------------------------------------------------------------
//int tiny_cnn(const data_t *input_img){
//#pragma HLS INTERFACE m_axi     port=input_img offset=slave bundle=gmem depth=4800
//#pragma HLS INTERFACE s_axilite port=input_img bundle=control
//#pragma HLS INTERFACE s_axilite port=return    bundle=control
//
//    static data_t img[IMG_H][IMG_W];
//    static data_t pool1[P1_H][P1_W][C1];
//    static data_t pw[P1_H][P1_W][BOTTLENECK_CH];
//    static data_t bins[OH][OW][BOTTLENECK_CH];
//
//    // Choose ONE memory mapping method: BIND_STORAGE only (no RESOURCE core=RAM_1P_BRAM)
//#pragma HLS BIND_STORAGE variable=img   type=ram_2p  impl=bram
//#pragma HLS BIND_STORAGE variable=pool1 type=ram_s2p impl=bram
//#pragma HLS BIND_STORAGE variable=pw    type=ram_s2p impl=bram
//#pragma HLS BIND_STORAGE variable=bins  type=ram_s2p impl=bram
//
//    // If your HLS accepts these on namespace arrays, keep them; otherwise remove.
//    // They are inside function scope (legal).
//#pragma HLS RESOURCE variable=weights::CONV1_W core=ROM_1P_BRAM
//#pragma HLS RESOURCE variable=weights::DW_W    core=ROM_1P_BRAM
//#pragma HLS RESOURCE variable=weights::PW_W    core=ROM_1P_BRAM
//#pragma HLS RESOURCE variable=weights::CLS_W   core=ROM_1P_BRAM
//
//    for(int i=0;i<IMG_H*IMG_W;i++){
//#pragma HLS PIPELINE II=1
//        img[i/IMG_W][i%IMG_W]=input_img[i];
//    }
//
//    conv3x3_relu_maxpool(img,pool1);
//    depthwise_pointwise_fused_relu(pool1,pw);
//    avgpool6x8_to_5x5(pw,bins);
//
//    return cls1x1_gap_argmax(bins);
//}




//
//#include "ap_fixed.h"
//#include "ap_int.h"
//#include "cnn_weights_data.h"   // includes cnn_weights.h too
//
//// IMPORTANT:
//// - data_t is defined in cnn_weights.h (already included via cnn_weights_data.h)
//// - but acc_t is NOT defined in your project, so we define it here.
//typedef ap_fixed<24,12> acc_t;
//
//// ------------------------------------------------------------
//// ReLU for data_t and acc_t
//// ------------------------------------------------------------
//static inline data_t relu_dt(data_t x) {
//#pragma HLS INLINE
//    return (x > 0) ? x : (data_t)0;
//}
//
//static inline acc_t relu_acc(acc_t x) {
//#pragma HLS INLINE
//    return (x > 0) ? x : (acc_t)0;
//}
//
//// ------------------------------------------------------------
//// Conv3x3 SAME + ReLU + 2x2 MaxPool fused
//// Input:  img[60][80]
//// Output: pool_out[30][40][16]
//// Uses BN-folded weights: CONV1_W[9][16], CONV1_B[16]
//// ------------------------------------------------------------
//static void conv3x3_relu_maxpool(
//    data_t img[IMG_H][IMG_W],
//    data_t pool_out[P1_H][P1_W][C1]
//) {
//#pragma HLS INLINE off
//
//    for (int ph = 0; ph < P1_H; ph++) {
//        for (int pw = 0; pw < P1_W; pw++) {
//
//            const int oh0 = 2 * ph;
//            const int ow0 = 2 * pw;
//
//            for (int oc = 0; oc < C1; oc++) {
//
//                data_t maxv = 0;
//
//                // 2x2 maxpool over 4 conv positions
//                for (int p = 0; p < 4; p++) {
//                    const int oh = oh0 + (p >> 1);
//                    const int ow = ow0 + (p & 1);
//
//                    acc_t acc = (acc_t)weights::CONV1_B[oc];
//
//                    // 3x3 conv SAME
//                    for (int kh = 0; kh < 3; kh++) {
//                        for (int kw = 0; kw < 3; kw++) {
//                            const int ih = oh + kh - 1;
//                            const int iw = ow + kw - 1;
//
//                            data_t inval = 0;
//                            if ((unsigned)ih < (unsigned)IMG_H &&
//                                (unsigned)iw < (unsigned)IMG_W) {
//                                inval = img[ih][iw];
//                            }
//
//                            const int k = kh * 3 + kw; // 0..8
//                            acc += (acc_t)inval * (acc_t)weights::CONV1_W[k][oc];
//                        }
//                    }
//
//                    const data_t v = relu_dt((data_t)acc);
//                    if (p == 0) maxv = v;
//                    else if (v > maxv) maxv = v;
//                }
//
//                pool_out[ph][pw][oc] = maxv;
//            }
//        }
//    }
//}
//
//// ------------------------------------------------------------
//// FUSED: Depthwise 3x3 SAME + ReLU (kept in acc_t) + Pointwise 1x1 + ReLU
//// Input:  in[30][40][16]
//// Output: out[30][40][8]
//// Uses BN-folded weights: DW_W[16][9], DW_B[16], PW_W[16][8], PW_B[8]
//// ------------------------------------------------------------
//static void depthwise_pointwise_fused_relu(
//    data_t in[P1_H][P1_W][C1],
//    data_t out[P1_H][P1_W][BOTTLENECK_CH]
//) {
//#pragma HLS INLINE off
//
//    // Tune this if DSP is high: try limit=4 or limit=2
//#pragma HLS ALLOCATION operation instances=mul limit=8
//#pragma HLS ALLOCATION operation instances=fmul limit=8
//
//    for (int h = 0; h < P1_H; h++) {
//        for (int w = 0; w < P1_W; w++) {
//
//#pragma HLS PIPELINE II=1
//
//            acc_t pw_acc[BOTTLENECK_CH];
//#pragma HLS ARRAY_PARTITION variable=pw_acc complete dim=1
//
//            // init with PW bias
//            for (int oc = 0; oc < BOTTLENECK_CH; oc++) {
//#pragma HLS UNROLL
//                pw_acc[oc] = (acc_t)weights::PW_B[oc];
//            }
//
//            // depthwise per-channel -> relu -> pointwise accumulate
//            for (int ic = 0; ic < C1; ic++) {
//
//                acc_t dw_acc = (acc_t)weights::DW_B[ic];
//
//                // 3x3 SAME depthwise
//                for (int kh = 0; kh < 3; kh++) {
//                    for (int kw = 0; kw < 3; kw++) {
//                        const int ih = h + kh - 1;
//                        const int iw = w + kw - 1;
//
//                        data_t inval = 0;
//                        if ((unsigned)ih < (unsigned)P1_H &&
//                            (unsigned)iw < (unsigned)P1_W) {
//                            inval = in[ih][iw][ic];
//                        }
//
//                        const int k = kh * 3 + kw; // 0..8
//                        dw_acc += (acc_t)inval * (acc_t)weights::DW_W[ic][k];
//                    }
//                }
//
//                // IMPORTANT: ReLU BEFORE PW, keep as acc_t
//                dw_acc = relu_acc(dw_acc);
//
//                // pointwise 16 -> 8
//                for (int oc = 0; oc < BOTTLENECK_CH; oc++) {
//#pragma HLS UNROLL factor=1
//                    pw_acc[oc] += dw_acc * (acc_t)weights::PW_W[ic][oc];
//                }
//            }
//
//            // output cast + ReLU
//            for (int oc = 0; oc < BOTTLENECK_CH; oc++) {
//#pragma HLS UNROLL
//                out[h][w][oc] = relu_dt((data_t)pw_acc[oc]);
//            }
//        }
//    }
//}
//
//// ------------------------------------------------------------
//// AvgPool 6x8 -> 5x5
//// Input:  in[30][40][8]
//// Output: bins[5][5][8]
//// ------------------------------------------------------------
//static void avgpool6x8_to_5x5(
//    data_t in[P1_H][P1_W][BOTTLENECK_CH],
//    data_t bins[OH][OW][BOTTLENECK_CH]
//) {
//#pragma HLS INLINE off
//    const acc_t denom = (acc_t)(6 * 8); // 48
//
//    for (int oh = 0; oh < OH; oh++) {
//        for (int ow = 0; ow < OW; ow++) {
//            for (int c = 0; c < BOTTLENECK_CH; c++) {
//
//                acc_t acc = 0;
//
//                for (int ph = 0; ph < 6; ph++) {
//                    for (int pw = 0; pw < 8; pw++) {
//                        acc += (acc_t)in[oh * 6 + ph][ow * 8 + pw][c];
//                    }
//                }
//
//                bins[oh][ow][c] = (data_t)(acc / denom);
//            }
//        }
//    }
//}
//
//// ------------------------------------------------------------
//// Classifier 1x1 (8 -> 4) applied on each 5x5 bin + GAP + Argmax
//// Input: bins[5][5][8]
//// Uses weights: CLS_W[8][4], CLS_B[4]  (NO BN)
//// ------------------------------------------------------------
//static int cls1x1_gap_argmax(
//    data_t bins[OH][OW][BOTTLENECK_CH]
//) {
//#pragma HLS INLINE off
//
//    acc_t logits[N_CLASSES];
//    for (int o = 0; o < N_CLASSES; o++) {
//        logits[o] = (acc_t)weights::CLS_B[o];
//    }
//
//    for (int h = 0; h < OH; h++) {
//        for (int w = 0; w < OW; w++) {
//            for (int ic = 0; ic < BOTTLENECK_CH; ic++) {
//                const acc_t v = (acc_t)bins[h][w][ic];
//                for (int o = 0; o < N_CLASSES; o++) {
//                    logits[o] += v * (acc_t)weights::CLS_W[ic][o];
//                }
//            }
//        }
//    }
//
//    const acc_t inv25 = (acc_t)1.0 / (acc_t)25.0;
//
//    data_t logit_avg[N_CLASSES];
//    for (int o = 0; o < N_CLASSES; o++) {
//        logit_avg[o] = (data_t)(logits[o] * inv25);
//    }
//
//    int argmax = 0;
//    data_t maxv = logit_avg[0];
//    for (int o = 1; o < N_CLASSES; o++) {
//        if (logit_avg[o] > maxv) {
//            maxv = logit_avg[o];
//            argmax = o;
//        }
//    }
//    return argmax;
//}
//
//// ------------------------------------------------------------
//// TOP
//// input_img: AXI flat array length 60*80 with normalized pixels
//// returns: class index 0..3
//// ------------------------------------------------------------
//int tiny_cnn(const data_t *input_img) {
//#pragma HLS INTERFACE m_axi     port=input_img offset=slave bundle=gmem depth=4800
//#pragma HLS INTERFACE s_axilite port=input_img bundle=control
//#pragma HLS INTERFACE s_axilite port=return    bundle=control
//
//    // On-chip buffers
//    static data_t img[IMG_H][IMG_W];
//    static data_t pool1[P1_H][P1_W][C1];
//    static data_t pw[P1_H][P1_W][BOTTLENECK_CH];
//    static data_t bins[OH][OW][BOTTLENECK_CH];
//
//#pragma HLS BIND_STORAGE variable=img   type=ram_2p  impl=bram
//#pragma HLS BIND_STORAGE variable=pool1 type=ram_s2p impl=bram
//#pragma HLS BIND_STORAGE variable=pw    type=ram_s2p impl=bram
//#pragma HLS BIND_STORAGE variable=bins  type=ram_s2p impl=bram
//
//    // Preload AXI -> BRAM
//    for (int i = 0; i < IMG_H * IMG_W; i++) {
//#pragma HLS PIPELINE II=1
//        img[i / IMG_W][i % IMG_W] = input_img[i];
//    }
//
//    conv3x3_relu_maxpool(img, pool1);
//
//    // Replaces depthwise3x3_relu + pointwise1x1_relu (dw buffer removed)
//    depthwise_pointwise_fused_relu(pool1, pw);
//
//    avgpool6x8_to_5x5(pw, bins);
//
//    return cls1x1_gap_argmax(bins);
//}


//#include "ap_fixed.h"
//#include "ap_int.h"
//#include "cnn_weights_data.h"   // includes cnn_weights.h too
//
//typedef ap_fixed<16,8>  data_t;
//typedef ap_fixed<24,12> acc_t;
//
//// ------------------------------------------------------------
//// ReLU
//// ------------------------------------------------------------
//static inline data_t relu(data_t x) {
//#pragma HLS INLINE
//    return (x > 0) ? x : (data_t)0;
//}
//
//// ------------------------------------------------------------
//// Conv3x3 SAME + ReLU + 2x2 MaxPool fused
//// Input:  img[60][80]
//// Output: pool_out[30][40][16]
//// Uses BN-folded weights: CONV1_W[9][16], CONV1_B[16]
//// ------------------------------------------------------------
//static void conv3x3_relu_maxpool(
//    data_t img[IMG_H][IMG_W],
//    data_t pool_out[P1_H][P1_W][C1]
//) {
//#pragma HLS INLINE off
//
//    for (int ph = 0; ph < P1_H; ph++) {
//        for (int pw = 0; pw < P1_W; pw++) {
//
//            const int oh0 = 2 * ph;
//            const int ow0 = 2 * pw;
//
//            for (int oc = 0; oc < C1; oc++) {
//
//                data_t maxv = 0;
//
//                // 2x2 maxpool over 4 conv positions
//                for (int p = 0; p < 4; p++) {
//                    const int oh = oh0 + (p >> 1);
//                    const int ow = ow0 + (p & 1);
//
//                    acc_t acc = (acc_t)weights::CONV1_B[oc];
//
//                    // 3x3 conv SAME
//                    for (int kh = 0; kh < 3; kh++) {
//                        for (int kw = 0; kw < 3; kw++) {
//                            const int ih = oh + kh - 1;
//                            const int iw = ow + kw - 1;
//
//                            data_t inval = 0;
//                            if ((unsigned)ih < (unsigned)IMG_H &&
//                                (unsigned)iw < (unsigned)IMG_W) {
//                                inval = img[ih][iw];
//                            }
//
//                            const int k = kh * 3 + kw; // 0..8
//                            acc += (acc_t)inval * (acc_t)weights::CONV1_W[k][oc];
//                        }
//                    }
//
//                    const data_t v = relu((data_t)acc);
//                    if (p == 0) maxv = v;
//                    else if (v > maxv) maxv = v;
//                }
//
//                pool_out[ph][pw][oc] = maxv;
//            }
//        }
//    }
//}
//
//// ------------------------------------------------------------
//// Depthwise 3x3 SAME + ReLU
//// Input:  in[30][40][16]
//// Output: out[30][40][16]
//// Uses BN-folded weights: DW_W[16][9], DW_B[16]
//// ------------------------------------------------------------
//static void depthwise3x3_relu(
//    data_t in[P1_H][P1_W][C1],
//    data_t out[P1_H][P1_W][C1]
//) {
//#pragma HLS INLINE off
//
//    for (int h = 0; h < P1_H; h++) {
//        for (int w = 0; w < P1_W; w++) {
//            for (int c = 0; c < C1; c++) {
//
//                acc_t acc = (acc_t)weights::DW_B[c];
//
//                for (int kh = 0; kh < 3; kh++) {
//                    for (int kw = 0; kw < 3; kw++) {
//                        const int ih = h + kh - 1;
//                        const int iw = w + kw - 1;
//
//                        data_t inval = 0;
//                        if ((unsigned)ih < (unsigned)P1_H &&
//                            (unsigned)iw < (unsigned)P1_W) {
//                            inval = in[ih][iw][c];
//                        }
//
//                        const int k = kh * 3 + kw; // 0..8
//                        acc += (acc_t)inval * (acc_t)weights::DW_W[c][k];
//                    }
//                }
//
//                out[h][w][c] = relu((data_t)acc);
//            }
//        }
//    }
//}
//
//// ------------------------------------------------------------
//// Pointwise 1x1 (16 -> 8) + ReLU
//// Input:  in[30][40][16]
//// Output: out[30][40][8]
//// Uses BN-folded weights: PW_W[16][8], PW_B[8]
//// ------------------------------------------------------------
//static void pointwise1x1_relu(
//    data_t in[P1_H][P1_W][C1],
//    data_t out[P1_H][P1_W][BOTTLENECK_CH]
//) {
//#pragma HLS INLINE off
//
//    for (int h = 0; h < P1_H; h++) {
//        for (int w = 0; w < P1_W; w++) {
//            for (int oc = 0; oc < BOTTLENECK_CH; oc++) {
//
//                acc_t acc = (acc_t)weights::PW_B[oc];
//
//                for (int ic = 0; ic < C1; ic++) {
//                    acc += (acc_t)in[h][w][ic] * (acc_t)weights::PW_W[ic][oc];
//                }
//
//                out[h][w][oc] = relu((data_t)acc);
//            }
//        }
//    }
//}
//
//// ------------------------------------------------------------
//// AvgPool 6x8 -> 5x5
//// Input:  in[30][40][8]
//// Output: bins[5][5][8]
//// ------------------------------------------------------------
//static void avgpool6x8_to_5x5(
//    data_t in[P1_H][P1_W][BOTTLENECK_CH],
//    data_t bins[OH][OW][BOTTLENECK_CH]
//) {
//#pragma HLS INLINE off
//    const acc_t denom = (acc_t)(6 * 8); // 48
//
//    for (int oh = 0; oh < OH; oh++) {
//        for (int ow = 0; ow < OW; ow++) {
//            for (int c = 0; c < BOTTLENECK_CH; c++) {
//
//                acc_t acc = 0;
//
//                for (int ph = 0; ph < 6; ph++) {
//                    for (int pw = 0; pw < 8; pw++) {
//                        acc += (acc_t)in[oh * 6 + ph][ow * 8 + pw][c];
//                    }
//                }
//
//                bins[oh][ow][c] = (data_t)(acc / denom);
//            }
//        }
//    }
//}
//
//// ------------------------------------------------------------
//// Classifier 1x1 (8 -> 4) applied on each 5x5 bin + GAP + Argmax
//// Input: bins[5][5][8]
//// Uses weights: CLS_W[8][4], CLS_B[4]  (NO BN)
//// ------------------------------------------------------------
//static int cls1x1_gap_argmax(
//    data_t bins[OH][OW][BOTTLENECK_CH]
//) {
//#pragma HLS INLINE off
//
//    // logits = bias
//    acc_t logits[N_CLASSES];
//    for (int o = 0; o < N_CLASSES; o++) {
//        logits[o] = (acc_t)weights::CLS_B[o];
//    }
//
//    // accumulate over all bins and input channels:
//    // logits[o] += sum_{h,w,ic} bins[h][w][ic] * CLS_W[ic][o]
//    for (int h = 0; h < OH; h++) {
//        for (int w = 0; w < OW; w++) {
//            for (int ic = 0; ic < BOTTLENECK_CH; ic++) {
//                const acc_t v = (acc_t)bins[h][w][ic];
//                for (int o = 0; o < N_CLASSES; o++) {
//                    logits[o] += v * (acc_t)weights::CLS_W[ic][o];
//                }
//            }
//        }
//    }
//
//    // GAP over 5x5: divide by 25 (matches GlobalAveragePooling2D)
//    const acc_t inv25 = (acc_t)1.0 / (acc_t)25.0;
//    data_t logit_avg[N_CLASSES];
//    for (int o = 0; o < N_CLASSES; o++) {
//        logit_avg[o] = (data_t)(logits[o] * inv25);
//    }
//
//    // Argmax
//    int argmax = 0;
//    data_t maxv = logit_avg[0];
//    for (int o = 1; o < N_CLASSES; o++) {
//        if (logit_avg[o] > maxv) {
//            maxv = logit_avg[o];
//            argmax = o;
//        }
//    }
//    return argmax;
//}
//
//// ------------------------------------------------------------
//// TOP
//// input_img: AXI flat array length 60*80 with normalized pixels
//// returns: class index 0..3
//// ------------------------------------------------------------
//int tiny_cnn(const data_t *input_img) {
//#pragma HLS INTERFACE m_axi     port=input_img offset=slave bundle=gmem depth=4800
//#pragma HLS INTERFACE s_axilite port=input_img bundle=control
//#pragma HLS INTERFACE s_axilite port=return    bundle=control
//
//    // On-chip buffers
//    static data_t img[IMG_H][IMG_W];
//    static data_t pool1[P1_H][P1_W][C1];
//    static data_t dw[P1_H][P1_W][C1];
//    static data_t pw[P1_H][P1_W][BOTTLENECK_CH];
//    static data_t bins[OH][OW][BOTTLENECK_CH];
//
//#pragma HLS BIND_STORAGE variable=img   type=ram_2p impl=bram
//#pragma HLS BIND_STORAGE variable=pool1 type=ram_s2p impl=bram
//#pragma HLS BIND_STORAGE variable=dw    type=ram_s2p impl=bram
//#pragma HLS BIND_STORAGE variable=pw    type=ram_s2p impl=bram
//#pragma HLS BIND_STORAGE variable=bins  type=ram_s2p impl=bram
//
//    // Preload AXI -> BRAM
//    for (int i = 0; i < IMG_H * IMG_W; i++) {
//#pragma HLS PIPELINE II=1
//        img[i / IMG_W][i % IMG_W] = input_img[i];
//    }
//
//    conv3x3_relu_maxpool(img, pool1);
//    depthwise3x3_relu(pool1, dw);
//    pointwise1x1_relu(dw, pw);
//    avgpool6x8_to_5x5(pw, bins);
//
//    const int pred = cls1x1_gap_argmax(bins);
//    return pred;
//}
