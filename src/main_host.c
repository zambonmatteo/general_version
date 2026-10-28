#include <stdio.h>
#include <time.h>
#include "model.h"
#include "model_specs.h"

static Model model;
static uint8_t workspace[MODEL_WORKSPACE_BYTES];
#if MEASURE_COUNT > 0
static uint32_t measurement_predictions[MEASURE_COUNT];
#endif

int main(void)
{
    ModelStatus status = model_init(&model, &model_spec, model_blob, MODEL_BLOB_BYTES);
    if (status != MODEL_OK || model_workspace_bytes(&model) != sizeof(workspace)) {
        fprintf(stderr, "Model initialization failed: %u\n", (unsigned)status);
        return 2;
    }
    printf("General %s model: %u operations, %u classes\n", INFERENCE_BACKEND_NAME,
           (unsigned)MODEL_OPERATION_COUNT, (unsigned)MODEL_CLASS_COUNT);
    printf("Model blob: %u bytes; workspace: %u bytes\n",
           (unsigned)MODEL_BLOB_BYTES, (unsigned)sizeof(workspace));
    unsigned mismatches = 0;
#if TEST_COUNT > 0
    for (uint32_t i = 0; i < TEST_COUNT; ++i) {
        uint32_t prediction = 0;
        status = model_forward(&model, test_inputs + (size_t)i * MODEL_INPUT_BYTES, MODEL_INPUT_BYTES,
                               workspace, sizeof(workspace), &prediction);
        if (status != MODEL_OK) { fprintf(stderr, "Inference failed: %u\n", (unsigned)status); return 2; }
        if (prediction != test_expected[i]) ++mismatches;
        printf("Check %u (source %u): predicted=%u expected=%u %s\n", (unsigned)i,
               (unsigned)test_source_indices[i], (unsigned)prediction, (unsigned)test_expected[i],
               prediction == test_expected[i] ? "PASS" : "FAIL");
    }
#endif
    printf("Verification mismatches: %u / %u\n", mismatches, (unsigned)TEST_COUNT);
#if MEASURE_COUNT > 0
    printf("Running inference benchmark...\n");
    /* Warm-up is excluded. PC clock() reports CPU time, not MCU latency. */
    uint32_t warmup;
    status = model_forward(&model, measure_inputs, MODEL_INPUT_BYTES, workspace, sizeof(workspace), &warmup);
    if (status != MODEL_OK) return 2;
    double total = 0;
    for (uint32_t repeat = 0; repeat < BENCH_REPEATS; ++repeat) {
        for (uint32_t i = 0; i < MEASURE_COUNT; ++i) {
            uint32_t prediction;
            const clock_t start = clock();
            status = model_forward(&model, measure_inputs + (size_t)i * MODEL_INPUT_BYTES, MODEL_INPUT_BYTES,
                                   workspace, sizeof(workspace), &prediction);
            const clock_t end = clock();
            if (status != MODEL_OK) return 2;
            if (start == (clock_t)-1 || end == (clock_t)-1 || end < start) {
                fprintf(stderr, "PC CPU clock unavailable or wrapped\n"); return 2;
            }
            total += (double)(end - start);
            measurement_predictions[i] = prediction;
        }
    }
    for (uint32_t i = 0; i < MEASURE_COUNT; ++i)
        printf("Input %u: predicted=%u\n", (unsigned)i, (unsigned)measurement_predictions[i]);
    printf("Host CPU time: %.6f ms/inference (%u runs; PC clock resolution applies)\n",
           total * 1000.0 / (double)CLOCKS_PER_SEC / (double)(MEASURE_COUNT * BENCH_REPEATS),
           (unsigned)(MEASURE_COUNT * BENCH_REPEATS));
#else
    printf("No measurement inputs supplied\n");
#endif
    return mismatches ? 1 : 0;
}
