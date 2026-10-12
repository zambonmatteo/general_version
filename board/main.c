#include "am_mcu_apollo.h"
#include "am_bsp.h"
#include "am_util.h"
#include "model.h"
#include "model_specs.h"
#include "deploy_data.h"

/* Runtime state/workspace use SRAM; model/descriptors and encoded inputs use
 * MRAM. Input binarization runs on the PC. */
static Model model;
static uint8_t workspace[MODEL_WORKSPACE_BYTES];
volatile uint32_t validation_mismatches = 0;
#if MEASURE_COUNT > 0
volatile uint32_t measurement_predictions[MEASURE_COUNT];
#endif

/* Benchmark statistics; valid is set only after all timing checks pass. */
typedef struct {
    uint32_t valid, runs, cpu_hz, cpu_hz_end;
    uint32_t min_cycles, max_cycles, mismatches;
    double mean_cycles, mean_ms;
} Timing;
volatile Timing timing;

/* Linker symbols: stack bounds and absolute byte counts read as addresses. */
extern uint32_t __dwn_stack_start, __dwn_stack_end;
extern const char __dwn_static_ram_bytes, __dwn_mram_image_bytes;

#if MEASURE_COUNT > 0
/* Read the active CPU clock; zero means an unsupported mode. */
static uint32_t cpu_clock_hz(void)
{
    switch (PWRCTRL->MCUPERFREQ_b.MCUPERFSTATUS) {
    case PWRCTRL_MCUPERFREQ_MCUPERFSTATUS_LP: return 96000000u;
    case PWRCTRL_MCUPERFREQ_MCUPERFSTATUS_HP: return 192000000u;
    default: return 0;
    }
}
#endif

static void fail(ModelStatus status)
{
    am_util_stdio_printf("ERROR: model operation failed (%u)\n", (unsigned)status);
    am_bsp_uart_printf_disable();
    while (1) { __NOP(); }
}

static void measure_latency(void)
{
#if MEASURE_COUNT > 0
    timing.valid = 0;
    /* Run untimed inferences to record baseline predictions for consistency checks. */
    for (uint32_t i = 0; i < MEASURE_COUNT; ++i) {
        uint32_t prediction;
        ModelStatus status = model_forward(&model, measure_inputs + (size_t)i * MODEL_INPUT_BYTES,
            MODEL_INPUT_BYTES, workspace, sizeof(workspace), &prediction);
        if (status != MODEL_OK) fail(status);
        measurement_predictions[i] = prediction;
    }
    timing.cpu_hz = cpu_clock_hz();
    if (!timing.cpu_hz) return;

    /* Enable the CPU cycle counter and confirm that it advances. */
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    if (DWT->CTRL & DWT_CTRL_NOCYCCNT_Msk) return;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
    __DSB(); __ISB();
    uint32_t before = DWT->CYCCNT;
    for (uint32_t i = 0; i < 32u; ++i) { __NOP(); }
    if (before == DWT->CYCCNT) return;

    /* This firmware owns STIMER (6 MHz) to guard against ambiguous DWT wrap.
     * Each inference must fit within one DWT period and one STIMER period. */
    am_hal_stimer_config(AM_HAL_STIMER_HFRC_6MHZ);
    am_hal_stimer_counter_clear();
    if (!am_hal_stimer_is_running()) return;
    uint32_t minimum = UINT32_MAX, maximum = 0;
    uint64_t total = 0;
    for (uint32_t repeat = 0; repeat < BENCH_REPEATS; ++repeat) {
        for (uint32_t i = 0; i < MEASURE_COUNT; ++i) {
            uint32_t prediction;
            const uint8_t *input = measure_inputs + (size_t)i * MODEL_INPUT_BYTES;
            uint32_t wall_start = am_hal_stimer_counter_get();
            __DSB(); __ISB();
            uint32_t start = DWT->CYCCNT;
            ModelStatus status = model_forward(&model, input, MODEL_INPUT_BYTES,
                                               workspace, sizeof(workspace), &prediction);
            __DSB();
            uint32_t elapsed = DWT->CYCCNT - start;
            uint32_t wall_ticks = am_hal_stimer_counter_get() - wall_start;
            if (status != MODEL_OK) fail(status);
            /* STIMER reads are outside DWT timing; two extra ticks guard wrap.
             * Reject ambiguous wraps, zero cycles or observed CPU clock changes. */
            if (((uint64_t)wall_ticks + 2u) * timing.cpu_hz >=
                (uint64_t)UINT32_MAX * 6000000u || !elapsed || cpu_clock_hz() != timing.cpu_hz)
                return;
            if (prediction != measurement_predictions[i]) ++timing.mismatches;
            total += elapsed;
            if (elapsed < minimum) minimum = elapsed;
            if (elapsed > maximum) maximum = elapsed;
            ++timing.runs;
        }
    }
    
    /* Final timing requires a stable clock and matching baseline predictions. */
    timing.cpu_hz_end = cpu_clock_hz();
    timing.min_cycles = minimum; timing.max_cycles = maximum;
    if (timing.cpu_hz_end != timing.cpu_hz || timing.mismatches) return;
    timing.mean_cycles = (double)total / timing.runs;
    timing.mean_ms = timing.mean_cycles * 1000.0 / timing.cpu_hz;
    timing.valid = 1;
#endif
}

/* Report linked memory sizes and stack usage observed before this report. */
static int print_memory(void)
{
    uintptr_t bottom = (uintptr_t)&__dwn_stack_start, top = (uintptr_t)&__dwn_stack_end;
    /* Mask interrupts during the stack scan, then restore their prior state. */
    uint32_t irq_state = __get_PRIMASK();
    __disable_irq();
    uintptr_t sp = __get_MSP(), cursor = bottom;
    if (sp < bottom || sp > top) {
        __set_PRIMASK(irq_state);
        am_util_stdio_printf("MEMORY ERROR: stack pointer outside reserved region\n");
        return 0;
    }
    /* Scan the startup marker; include unwritten space in the current frame. */
    while (cursor < top && *(volatile uint32_t *)cursor == 0xA5A5A5A5u) cursor += 4u;
    if (sp < cursor) cursor = sp;
    __set_PRIMASK(irq_state);
    uint32_t reserved = (uint32_t)(top - bottom), observed = (uint32_t)(top - cursor);
    uint32_t margin = (uint32_t)(cursor - bottom);
    uint32_t ram = (uint32_t)(uintptr_t)&__dwn_static_ram_bytes;
    uint32_t flash = (uint32_t)(uintptr_t)&__dwn_mram_image_bytes;
    uint32_t data = (MEASURE_COUNT + TEST_COUNT) * MODEL_INPUT_BYTES + TEST_COUNT * 4u;
    /* MRAM includes samples; SRAM peak is static RAM + observed stack, an estimate. */
    am_util_stdio_printf("Flash/MRAM: %u bytes (complete firmware)\n", (unsigned)flash);
    am_util_stdio_printf("Model blob: %u bytes; encoded samples/predictions: %u bytes\n",
                         (unsigned)MODEL_BLOB_BYTES, (unsigned)data);
    am_util_stdio_printf("Code + model + descriptors: %u bytes (excluding samples)\n", (unsigned)(flash - data));
    am_util_stdio_printf("Static RAM: %u bytes; workspace: %u bytes\n", (unsigned)ram, (unsigned)sizeof(workspace));
    am_util_stdio_printf("Stack reserved: %u; observed: %u; margin: %u bytes\n",
                         (unsigned)reserved, (unsigned)observed, (unsigned)margin);
    am_util_stdio_printf("SRAM reserved: %u bytes\n", (unsigned)(ram + reserved));
    if (margin) am_util_stdio_printf("SRAM observed peak estimate: %u bytes\n", (unsigned)(ram + observed));
    else am_util_stdio_printf("SRAM observed peak estimate: INVALID (watermark exhausted)\n");
    return margin != 0;
}

/* Initialize the board/model, verify predictions and report benchmark results. */
int main(void)
{
    /* Enable caches, board defaults and blocking UART output (115200, 8-N-1). */
    am_hal_cachectrl_config(&am_hal_cachectrl_defaults);
    am_hal_cachectrl_enable();
    am_bsp_low_power_init();
    if (am_bsp_uart_printf_enable()) {
        while (1) { __NOP(); }
    }
    am_util_stdio_printf("\nApollo4 BL EVB implementations\n");
    am_util_stdio_printf("Backend: %s\n", INFERENCE_BACKEND_NAME);
    /* Validate embedded model data and check the compiled workspace size. */
    ModelStatus status = model_init(&model, &model_spec, model_blob, MODEL_BLOB_BYTES);
    if (status != MODEL_OK || model_workspace_bytes(&model) != sizeof(workspace)) fail(status);
    if (TEST_COUNT) am_util_stdio_printf("Verifying predictions...\n");
    else am_util_stdio_printf("Verification skipped (no test inputs)\n");
    /* Compare selected test inputs with their reference predictions, if supplied. */
#if TEST_COUNT > 0
    for (uint32_t i = 0; i < TEST_COUNT; ++i) {
        uint32_t prediction;
        status = model_forward(&model, test_inputs + (size_t)i * MODEL_INPUT_BYTES, MODEL_INPUT_BYTES,
                               workspace, sizeof(workspace), &prediction);
        if (status != MODEL_OK) fail(status);
        if (prediction != test_expected[i]) ++validation_mismatches;
    }
#endif
    am_util_stdio_printf("Verification mismatches: %u / %u\n", (unsigned)validation_mismatches, (unsigned)TEST_COUNT);
    /* Skip timing on mismatches; absent tests leave the mismatch count at zero. */
    if (!validation_mismatches && MEASURE_COUNT) {
        am_util_stdio_printf("Running inference benchmark...\n");
        measure_latency();
    } else if (validation_mismatches) am_util_stdio_printf("Measurements skipped (verification failed)\n");
    else am_util_stdio_printf("Benchmark skipped (no measurement inputs)\n");
    if (timing.valid) {
        am_util_stdio_printf("CPU: %u Hz; runs: %u; cycles min=%u mean=%.3f max=%u\n",
            (unsigned)timing.cpu_hz, (unsigned)timing.runs, (unsigned)timing.min_cycles,
            timing.mean_cycles, (unsigned)timing.max_cycles);
        am_util_stdio_printf("Latency: %.6f ms/inference\n\n", timing.mean_ms);
    } else am_util_stdio_printf("Latency: N/A (no measurement inputs or invalid benchmark)\n\n");
    int success = print_memory() && !validation_mismatches;
#if MEASURE_COUNT > 0
    success = success && timing.valid;
#endif
    am_util_stdio_printf("Run completed%s\n", success ? "" : " with errors");
    /* Flush the final UART bytes before powering down the interface. */
    am_bsp_uart_printf_disable();
    /* Wait for the next reset after reporting. */
    while (1) { __NOP(); }
}
