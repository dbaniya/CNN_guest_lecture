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



