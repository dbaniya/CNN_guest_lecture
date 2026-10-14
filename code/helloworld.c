

#include <stdio.h>
#include <stdint.h>

#include "platform.h"
#include "xil_printf.h"
#include "xil_io.h"
#include "xil_cache.h"
#include "xparameters.h"
#include "xtime_l.h"
#include "xuartps.h"

#include "xgpio.h"   // AXI GPIO

extern char inbyte(void);
extern void outbyte(char c);

// ============================================================
// UART BAUD CONFIG
// ============================================================
#ifndef UART_DEVICE_ID
#define UART_DEVICE_ID   XPAR_XUARTPS_0_DEVICE_ID
#endif

#ifndef UART_BAUD
#define UART_BAUD        4000000u
#endif

// ============================================================
// DWPW CNN input (60x80)
// ============================================================
#define IMG_H     60
#define IMG_W     80
#define N_INPUTS  (IMG_H * IMG_W)
#define RX_BYTES  (N_INPUTS * 2)

// Frame protocol
#define FRAME_START_0  0xA6
#define FRAME_START_1  0x6A

// ============================================================
// DWPW CNN AXI-Lite base + regs
// ============================================================
#define IP_BASE_ADDR  XPAR_DS_CNN_0_S_AXI_CONTROL_BASEADDR

#define REG_AP_CTRL         0x00
#define REG_INPUT_IMG_L     0x10
#define REG_INPUT_IMG_H     0x14
#define REG_PRED_OUT        0x1C
#define REG_PRED_OUT_CTRL   0x20
#define REG_LOGITS_BASE     0x30
#define AP_DONE_BIT         1



static inline void ip_write(uint32_t off, uint32_t val)
{
    Xil_Out32(IP_BASE_ADDR + off, val);
}

static inline uint32_t ip_read(uint32_t off)
{
    return Xil_In32(IP_BASE_ADDR + off);
}

// ============================================================
// AXI GPIO (TIMING PIN)
// ============================================================
#ifndef GPIO_DEV_ID
#define GPIO_DEV_ID   XPAR_AXI_GPIO_DEVICE_ID
#endif
#ifndef GPIO_CH
#define GPIO_CH       1
#endif
#ifndef GPIO_BIT
#define GPIO_BIT      0
#endif

static XGpio Gpio;

static int gpio_init(void)
{
    int st = XGpio_Initialize(&Gpio, GPIO_DEV_ID);
    if (st != XST_SUCCESS) {
        xil_printf("XGpio_Initialize failed: %d\r\n", st);
        return st;
    }

    // 0 = output, 1 = input
    XGpio_SetDataDirection(&Gpio, GPIO_CH, 0x00000000);
    XGpio_DiscreteWrite(&Gpio, GPIO_CH, 0x00000000);

    xil_printf("AXI GPIO init OK (dev=%d ch=%d bit=%d)\r\n",
               GPIO_DEV_ID, GPIO_CH, GPIO_BIT);
    return XST_SUCCESS;
}

static inline void timing_pin_high(void)
{
    XGpio_DiscreteWrite(&Gpio, GPIO_CH, (1u << GPIO_BIT));
}

static inline void timing_pin_low(void)
{
    XGpio_DiscreteWrite(&Gpio, GPIO_CH, 0x00000000);
}

// ============================================================
// UART helpers
// ============================================================
static void uart_set_baudrate(u32 baud)
{
    XUartPs Uart;
    XUartPs_Config *Cfg = XUartPs_LookupConfig(UART_DEVICE_ID);
    if (!Cfg) return;

    if (XUartPs_CfgInitialize(&Uart, Cfg, Cfg->BaseAddress) != XST_SUCCESS)
        return;

    XUartPs_SetOperMode(&Uart, XUARTPS_OPER_MODE_NORMAL);
    (void)XUartPs_SetBaudRate(&Uart, baud);
}

static uint8_t read_byte_blocking(void)
{
    return (uint8_t)inbyte();
}

static void wait_for_frame_header(void)
{
    while (1) {
        uint8_t b0 = read_byte_blocking();
        if (b0 == FRAME_START_0) {
            uint8_t b1 = read_byte_blocking();
            if (b1 == FRAME_START_1)
                return;
        }
    }
}

// ============================================================
// Buffers
// ============================================================
static uint8_t  rx_buf[RX_BYTES];
static int16_t  ddr_input[N_INPUTS] __attribute__((aligned(64)));

// ============================================================
// Reply frame (Option A: TRUE TIME IN MICROSECONDS):
//   0xA5 <pred> <time_us u32 LE> 0x5A
// ============================================================
static void uart_send_pred_time_logits_frame(
    uint8_t pred,
    uint32_t time_us,
    int16_t logit0,
    int16_t logit1,
    int16_t logit2,
    int16_t logit3
)
{
    outbyte((char)0xA5);

    outbyte((char)pred);

    outbyte((char)((time_us >> 0)  & 0xFF));
    outbyte((char)((time_us >> 8)  & 0xFF));
    outbyte((char)((time_us >> 16) & 0xFF));
    outbyte((char)((time_us >> 24) & 0xFF));

    outbyte((char)((logit0 >> 0) & 0xFF));
    outbyte((char)((logit0 >> 8) & 0xFF));

    outbyte((char)((logit1 >> 0) & 0xFF));
    outbyte((char)((logit1 >> 8) & 0xFF));

    outbyte((char)((logit2 >> 0) & 0xFF));
    outbyte((char)((logit2 >> 8) & 0xFF));

    outbyte((char)((logit3 >> 0) & 0xFF));
    outbyte((char)((logit3 >> 8) & 0xFF));

    outbyte((char)0x5A);
}

int main(void)
{
    init_platform();
    uart_set_baudrate(UART_BAUD);

    if (gpio_init() != XST_SUCCESS) {
        xil_printf("GPIO init failed. Check AXI GPIO in BD + addresses.\r\n");
        cleanup_platform();
        return 1;
    }

    xil_printf("COUNTS_PER_SECOND=%lu\r\n", (unsigned long)COUNTS_PER_SECOND);
    xil_printf("\r\nFramed UART(int16 Q8.8) -> DWPW cnn\r\n");
    xil_printf("UART_DEVICE_ID=%d  UART_BAUD=%lu\r\n",
               UART_DEVICE_ID, (unsigned long)UART_BAUD);
    xil_printf("RX_BYTES=%d\r\n", RX_BYTES);
    xil_printf("Reply: 0xA5 <pred> <time_us u32 LE> 0x5A\r\n");

    timing_pin_low();

    while (1)
    {
        wait_for_frame_header();

        uint16_t len =
            (uint16_t)read_byte_blocking() |
            ((uint16_t)read_byte_blocking() << 8);

        if (len != RX_BYTES) {
            xil_printf("Bad length: %d\r\n", len);
            continue;
        }

        for (int i = 0; i < RX_BYTES; i++)
            rx_buf[i] = read_byte_blocking();

        for (int i = 0; i < N_INPUTS; i++) {
            uint16_t u =
                ((uint16_t)rx_buf[i*2 + 1] << 8) |
                ((uint16_t)rx_buf[i*2 + 0]);
            ddr_input[i] = (int16_t)u;
        }

        Xil_DCacheFlushRange((UINTPTR)ddr_input, N_INPUTS * sizeof(int16_t));

        UINTPTR addr = (UINTPTR)ddr_input;
//        ip_write(REG_INPUT_IMG_L, (uint32_t)(addr & 0xFFFFFFFFu));
//        ip_write(REG_INPUT_IMG_H, (uint32_t)((addr >> 32) & 0xFFFFFFFFu));

        ip_write(REG_INPUT_IMG_L, (uint32_t)addr);
        ip_write(REG_INPUT_IMG_H, 0u);

        // measure
        XTime t0, t1;
        XTime_GetTime(&t0);

        timing_pin_high();
        ip_write(REG_AP_CTRL, 0x01); // ap_start

        while (((ip_read(REG_AP_CTRL) >> AP_DONE_BIT) & 0x1u) == 0u) { }

        timing_pin_low();

        XTime_GetTime(&t1);
        (void)ip_read(REG_AP_CTRL);

        uint64_t dt_ticks = (uint64_t)(t1 - t0);

        // TRUE elapsed time in microseconds (PS timer)
        uint32_t time_us =
            (uint32_t)((dt_ticks * 1000000ULL) / (uint64_t)COUNTS_PER_SECOND);

        // Print AFTER pulse (in ms with 3 decimals, no float)
        xil_printf("accel(sw): %lu.%03lu ms\r\n",
                   (unsigned long)(time_us / 1000u),
                   (unsigned long)(time_us % 1000u));


        uint32_t pred = ip_read(REG_PRED_OUT);

        uint32_t w0 = ip_read(REG_LOGITS_BASE + 0x0);
        uint32_t w1 = ip_read(REG_LOGITS_BASE + 0x4);

        int16_t logit0 = (int16_t)(w0 & 0xFFFF);
        int16_t logit1 = (int16_t)((w0 >> 16) & 0xFFFF);

        int16_t logit2 = (int16_t)(w1 & 0xFFFF);
        int16_t logit3 = (int16_t)((w1 >> 16) & 0xFFFF);

        xil_printf(
            "logits=[%d %d %d %d]\r\n",
            logit0,
            logit1,
            logit2,
            logit3
        );


        // Reply with pred + TRUE time in microseconds
        uart_send_pred_time_logits_frame(
            (uint8_t)(pred & 0xFFu),
            time_us,
            logit0,
            logit1,
            logit2,
            logit3
        );
    }

    cleanup_platform();
    return 0;
}
