#include "am_mcu_apollo.h"
#include "am_bsp.h"
#include "am_util.h"
#include "model.h"
#include "model_specs.h"
#include <stdbool.h>
#include <stddef.h>

_Static_assert(BENCH_REPEATS > 0 && MODEL_SAMPLE_COUNT <= UINT32_MAX / BENCH_REPEATS,
               "Benchmark run count must fit uint32");

volatile uint32_t board_done;
volatile uint32_t board_errors;

extern uint32_t __dwn_stack_start, __dwn_stack_end;
extern const char __dwn_static_ram_bytes, __dwn_mram_image_bytes;

typedef struct {
    uint32_t runs, cpu_hz, minimum, maximum, mismatches;
    uint64_t total_cycles;
} Timing;
static Timing timing;

static bool report_error(const char *message)
{
    am_util_stdio_printf("ERROR: %s\n", message);
    return false;
}

static uint32_t cpu_clock_hz(void)
{
    switch (PWRCTRL->MCUPERFREQ_b.MCUPERFSTATUS) {
    case PWRCTRL_MCUPERFREQ_MCUPERFSTATUS_LP: return 96000000u;
    case PWRCTRL_MCUPERFREQ_MCUPERFSTATUS_HP: return 192000000u;
    default: return 0;
    }
}

static bool benchmark(void)
{
    timing.cpu_hz = cpu_clock_hz();
    if (!timing.cpu_hz) return report_error("Unsupported CPU clock");
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    if (DWT->CTRL & DWT_CTRL_NOCYCCNT_Msk) return report_error("DWT counter unavailable");
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
    __DSB(); __ISB();
    const uint32_t before = DWT->CYCCNT;
    for (uint32_t i = 0; i < 32u; ++i) { __NOP(); }
    if (before == DWT->CYCCNT) return report_error("DWT counter is not advancing");

    /* STIMER guards against an ambiguous wrap of the 32-bit cycle counter. */
    am_hal_stimer_config(AM_HAL_STIMER_HFRC_6MHZ);
    am_hal_stimer_counter_clear();
    if (!am_hal_stimer_is_running()) return report_error("STIMER is not running");
    timing.minimum = UINT32_MAX;
    for (uint32_t repeat = 0; repeat < BENCH_REPEATS; ++repeat) {
        for (uint32_t i = 0; i < MODEL_SAMPLE_COUNT; ++i) {
            const uint8_t *input = input_data + (size_t)i * MODEL_INPUT_BYTES;
            const uint32_t wall_start = am_hal_stimer_counter_get();
            __DSB(); __ISB();
            const uint32_t start = DWT->CYCCNT;
            const uint32_t prediction = model_predict(input);
            __DSB();
            const uint32_t cycles = DWT->CYCCNT - start;
            const uint32_t ticks = am_hal_stimer_counter_get() - wall_start;
            if (!cycles || ((uint64_t)ticks + 2u) * timing.cpu_hz >=
                           (uint64_t)UINT32_MAX * 6000000u)
                return report_error("Invalid cycle count or counter wrap");
            if (cpu_clock_hz() != timing.cpu_hz) return report_error("CPU clock changed");
            timing.total_cycles += cycles;
            if (cycles < timing.minimum) timing.minimum = cycles;
            if (cycles > timing.maximum) timing.maximum = cycles;
            if (prediction != expected_data[i]) ++timing.mismatches;
            ++timing.runs;
        }
    }
    const double mean = (double)timing.total_cycles / timing.runs;
    am_util_stdio_printf("CPU: %u Hz; runs: %u; benchmark mismatches: %u\n",
                         (unsigned)timing.cpu_hz, (unsigned)timing.runs, (unsigned)timing.mismatches);
    if (timing.mismatches) return report_error("Benchmark predictions differ from references");
    am_util_stdio_printf("Cycles: min=%u mean=%.1f max=%u\n", (unsigned)timing.minimum,
                         mean, (unsigned)timing.maximum);
    am_util_stdio_printf("Latency (encoding + inference): min=%.3f mean=%.3f max=%.3f ms\n",
                         timing.minimum * 1000.0 / timing.cpu_hz,
                         mean * 1000.0 / timing.cpu_hz,
                         timing.maximum * 1000.0 / timing.cpu_hz);
    return true;
}

static bool report_memory(void)
{
    const uintptr_t bottom = (uintptr_t)&__dwn_stack_start, top = (uintptr_t)&__dwn_stack_end;
    const uint32_t irq_state = __get_PRIMASK();
    __disable_irq();
    const uintptr_t sp = __get_MSP();
    uintptr_t cursor = bottom;
    if (sp < bottom || sp > top) {
        __set_PRIMASK(irq_state);
        return report_error("Stack pointer outside reserved SRAM");
    }
    /* Startup paints the stack. Include reserved space in the current frame. */
    while (cursor < top && *(volatile uint32_t *)cursor == 0xA5A5A5A5u) cursor += 4u;
    if (sp < cursor) cursor = sp;
    __set_PRIMASK(irq_state);
    const uint32_t reserved = (uint32_t)(top - bottom), observed = (uint32_t)(top - cursor);
    const uint32_t margin = (uint32_t)(cursor - bottom);
    const uint32_t ram = (uint32_t)(uintptr_t)&__dwn_static_ram_bytes;
    const uint32_t image = (uint32_t)(uintptr_t)&__dwn_mram_image_bytes;
    const uint32_t test_data = MODEL_SAMPLE_COUNT * (MODEL_INPUT_BYTES + 4u);
    am_util_stdio_printf("Flash/MRAM image: %u bytes (complete application)\n", (unsigned)image);
    am_util_stdio_printf("Model binary: %u bytes; test inputs/references: %u bytes\n",
                         (unsigned)MODEL_BLOB_BYTES, (unsigned)test_data);
    am_util_stdio_printf("Code + model + metadata: %u bytes (excluding test data)\n",
                         (unsigned)(image - test_data));
    am_util_stdio_printf("Static SRAM: %u bytes; activation/scratch buffers: %u bytes\n",
                         (unsigned)ram, (unsigned)MODEL_WORKSPACE_BYTES);
    am_util_stdio_printf("Stack reserved: %u; observed: %u; margin: %u bytes\n",
                         (unsigned)reserved, (unsigned)observed, (unsigned)margin);
    am_util_stdio_printf("SRAM reserved: %u bytes; heap reserved: 0 bytes\n", (unsigned)(ram + reserved));
    if (!margin) return report_error("Stack watermark exhausted; SRAM peak estimate invalid");
    am_util_stdio_printf("SRAM observed peak estimate: %u bytes\n", (unsigned)(ram + observed));
    return true;
}

int main(void)
{
    am_hal_cachectrl_config(&am_hal_cachectrl_defaults);
    am_hal_cachectrl_enable();
    am_bsp_low_power_init();
    if (am_bsp_uart_printf_enable()) {
        board_errors = 1;
        board_done = 1;
        while (1) { __NOP(); }
    }

    am_util_stdio_printf("\ngeneral_version_v2 - Apollo4 Blue Lite\n");
#ifdef USE_BITSLICE
    am_util_stdio_printf("Backend: bitslice (scalar fallback when required)\n");
#else
    am_util_stdio_printf("Backend: scalar\n");
#endif
    am_util_stdio_printf("Model: %s; samples: %u\n", MODEL_NAME, (unsigned)MODEL_SAMPLE_COUNT);
    am_util_stdio_printf("Raw input: encoding and inference run on the board\n");
    for (uint32_t i = 0; i < MODEL_SAMPLE_COUNT; ++i) {
        const uint32_t prediction = model_predict(input_data + (size_t)i * MODEL_INPUT_BYTES);
        const uint32_t expected = expected_data[i];
        if (prediction != expected) ++board_errors;
        am_util_stdio_printf("Input %u: predicted=%u expected=%u %s\n", (unsigned)i,
                             (unsigned)prediction, (unsigned)expected,
                             prediction == expected ? "PASS" : "FAIL");
    }
    am_util_stdio_printf("Verification mismatches: %u / %u\n", (unsigned)board_errors,
                         (unsigned)MODEL_SAMPLE_COUNT);
    if (!board_errors) {
        am_util_stdio_printf("Benchmark: %u repetitions per input; full encoding + prediction\n",
                             (unsigned)BENCH_REPEATS);
        if (!benchmark()) ++board_errors;
    }

    if (!report_memory()) ++board_errors;
    am_util_stdio_printf("Errors: %u\n", (unsigned)board_errors);
    am_util_stdio_printf("Run completed%s\n", board_errors ? " with errors" : "");
    am_bsp_uart_printf_disable();
    board_done = 1;
    while (1) { __NOP(); }
}
