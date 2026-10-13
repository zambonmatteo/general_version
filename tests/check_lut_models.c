#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "lut.h"
#include "tensor.h"
#include "original_specs.h"
#include "model_specs.h"

#define REQUIRE(condition) do { if (!(condition)) { \
    fprintf(stderr, "Check failed at line %u: %s\n", (unsigned)__LINE__, #condition); \
    exit(1); } } while (0)

static void expect_bind_error(LutLayerSpec spec, uint8_t *blob, ModelStatus expected)
{
    LutLayer layer = {.spec = &spec, .tables = blob, .indices = blob, .table_ids = blob};
    REQUIRE(lut_layer_bind(&layer, &spec, blob, 64) == expected);
    REQUIRE(!layer.spec && !layer.tables && !layer.indices && !layer.table_ids);
}

static void check_invalid_layouts(void)
{
    uint8_t blob[64] = {0};
    LutLayerSpec good = {
        .type = LUT_FULLY_CONNECTED, .n_luts = 4, .n_inputs = 4, .index_bytes = 2,
        .tables = {0, 4}, .indices = {8, 32},
        .n_tables = 2, .table_id_bytes = 1, .table_ids = {4, 4},
        .input_shape = {1, 1, 8}, .output_shape = {1, 1, 4}
    };
    blob[5] = blob[7] = 1;
    LutLayer layer;
    REQUIRE(lut_layer_bind(&layer, &good, blob, sizeof(blob)) == MODEL_OK);
    Lut a, b, c;
    REQUIRE(lut_layer_get(&layer, 0, &a) == MODEL_OK);
    REQUIRE(lut_layer_get(&layer, 1, &b) == MODEL_OK);
    REQUIRE(lut_layer_get(&layer, 2, &c) == MODEL_OK);
    REQUIRE(a.table == c.table && a.table != b.table && a.indices != c.indices);
    REQUIRE(lut_layer_get(&layer, 4, &a) == MODEL_INVALID_ARGUMENT && !a.table);
    for (uint32_t width = 2; width <= 4; width *= 2) {
        LutLayerSpec wide = good;
        wide.table_id_bytes = width;
        wide.table_ids = (BlobSection){40, 4 * width};
        blob[40 + width] = 1;
        REQUIRE(lut_layer_bind(&layer, &wide, blob, sizeof(blob)) == MODEL_OK);
        REQUIRE(lut_layer_get(&layer, 1, &b) == MODEL_OK && b.table == blob + 2);
        blob[40 + width] = 0;
        blob[40 + width - 1] = 1;
        expect_bind_error(wide, blob, MODEL_INVALID_INDEX);
        blob[40 + width - 1] = 0;
    }

    LutLayerSpec bad = good;
    bad.n_tables = 0; expect_bind_error(bad, blob, MODEL_INVALID_SPEC);
    bad = good; bad.n_tables = 5; expect_bind_error(bad, blob, MODEL_INVALID_SPEC);
    bad = good; bad.table_id_bytes = 3; expect_bind_error(bad, blob, MODEL_INVALID_SPEC);
    bad = good; bad.table_ids.bytes = 3; expect_bind_error(bad, blob, MODEL_INVALID_SPEC);
    bad = good; bad.tables.bytes = 3; expect_bind_error(bad, blob, MODEL_INVALID_SPEC);
    bad = good; bad.table_ids.offset = 63; expect_bind_error(bad, blob, MODEL_INVALID_BLOB);
    bad = good; bad.tables.offset = 63; expect_bind_error(bad, blob, MODEL_INVALID_BLOB);
    bad = good; bad.indices.offset = 63; expect_bind_error(bad, blob, MODEL_INVALID_BLOB);
    bad = good; bad.table_ids.offset = 2; expect_bind_error(bad, blob, MODEL_INVALID_SPEC);
    bad = good; bad.table_ids.offset = 8; expect_bind_error(bad, blob, MODEL_INVALID_SPEC);
    bad = good; bad.indices.offset = 2; expect_bind_error(bad, blob, MODEL_INVALID_SPEC);
    bad = good; bad.table_id_bytes = 0; expect_bind_error(bad, blob, MODEL_INVALID_SPEC);
    bad = good; bad.table_id_bytes = 0; bad.table_ids = (BlobSection){0, 0};
    expect_bind_error(bad, blob, MODEL_INVALID_SPEC);
    bad = good; bad.n_tables = 1; bad.tables.bytes = 2;
    expect_bind_error(bad, blob, MODEL_INVALID_SPEC);
    blob[7] = 2; expect_bind_error(good, blob, MODEL_INVALID_INDEX); blob[7] = 1;
    blob[8] = 8; expect_bind_error(good, blob, MODEL_INVALID_INDEX); blob[8] = 0;

    bad = good; bad.n_tables = 257; bad.n_luts = 258;
    bad.tables.bytes = 514; bad.table_ids.bytes = 258;
    expect_bind_error(bad, blob, MODEL_INVALID_SPEC);
    LutLayerSpec shared = good;
    shared.n_tables = 1; shared.tables.bytes = 2;
    shared.table_id_bytes = 0; shared.table_ids = (BlobSection){0, 0};
    REQUIRE(lut_layer_bind(&layer, &shared, blob, sizeof(blob)) == MODEL_OK);
    REQUIRE(lut_layer_get(&layer, 0, &a) == MODEL_OK);
    REQUIRE(lut_layer_get(&layer, 3, &b) == MODEL_OK);
    REQUIRE(a.table == b.table && a.indices != b.indices);
}

static uint8_t *read_file(const char *path, size_t *bytes)
{
    FILE *file = fopen(path, "rb");
    REQUIRE(file && fseek(file, 0, SEEK_END) == 0);
    long length = ftell(file);
    REQUIRE(length > 0 && fseek(file, 0, SEEK_SET) == 0);
    *bytes = (size_t)length;
    uint8_t *data = malloc(*bytes);
    REQUIRE(data && fread(data, 1, *bytes, file) == *bytes && fclose(file) == 0);
    return data;
}

static void check_layer_views(const uint8_t *source, const uint8_t *runtime)
{
    for (uint32_t i = 0; i < MODEL_LUT_LAYER_COUNT; ++i) {
        LutLayer a, b;
        REQUIRE(lut_layer_bind(&a, &original_lut_specs[i], source, ORIGINAL_MODEL_BLOB_BYTES) == MODEL_OK);
        REQUIRE(lut_layer_bind(&b, &model_lut_specs[i], runtime, MODEL_BLOB_BYTES) == MODEL_OK);
        for (uint32_t k = 0; k < a.spec->n_luts; ++k) {
            Lut av, bv;
            REQUIRE(lut_layer_get(&a, k, &av) == MODEL_OK && lut_layer_get(&b, k, &bv) == MODEL_OK);
            const size_t row_bytes = (size_t)(((UINT64_C(1) << av.n_inputs) + 7) / 8);
            REQUIRE(memcmp(av.table, bv.table, row_bytes) == 0);
            for (uint32_t j = 0; j < av.n_inputs; ++j)
                REQUIRE(lut_index(&av, j) == lut_index(&bv, j));
        }
    }
}

static void check_intermediates(const uint8_t *source, const uint8_t *runtime,
                                const uint8_t *sample)
{
    uint8_t *current = malloc(MODEL_WORKSPACE_BYTES);
    uint8_t *a = malloc(MODEL_WORKSPACE_BYTES), *b = malloc(MODEL_WORKSPACE_BYTES);
    REQUIRE(current && a && b);
    memcpy(current, sample, MODEL_INPUT_BYTES);
    for (uint32_t i = 0; i < MODEL_OPERATION_COUNT; ++i) {
        const Operation *op = &model_operations[i];
        size_t in_bytes = tensor_bytes(op->input_shape), out_bytes = tensor_bytes(op->output_shape);
        if (op->type == OP_GROUP_SUM) break;
        if (op->type == OP_LUT) {
            LutLayer original, compact;
            REQUIRE(lut_layer_bind(&original, original_operations[i].parameters.lut,
                                   source, ORIGINAL_MODEL_BLOB_BYTES) == MODEL_OK);
            REQUIRE(lut_layer_bind(&compact, op->parameters.lut, runtime, MODEL_BLOB_BYTES) == MODEL_OK);
            REQUIRE(lut_layer_forward(&original, current, in_bytes, a, MODEL_WORKSPACE_BYTES) == MODEL_OK);
            REQUIRE(lut_layer_forward(&compact, current, in_bytes, b, MODEL_WORKSPACE_BYTES) == MODEL_OK);
        } else if (op->type == OP_OR_POOL) {
            REQUIRE(tensor_or_pool(current, in_bytes, op->input_shape, op->parameters.pool,
                                   a, MODEL_WORKSPACE_BYTES) == MODEL_OK);
            REQUIRE(tensor_or_pool(current, in_bytes, op->input_shape, op->parameters.pool,
                                   b, MODEL_WORKSPACE_BYTES) == MODEL_OK);
        } else {
            REQUIRE(op->type == OP_FLATTEN);
            REQUIRE(tensor_flatten(current, in_bytes, op->input_shape, a, MODEL_WORKSPACE_BYTES) == MODEL_OK);
            REQUIRE(tensor_flatten(current, in_bytes, op->input_shape, b, MODEL_WORKSPACE_BYTES) == MODEL_OK);
        }
        REQUIRE(memcmp(a, b, out_bytes) == 0);
        memcpy(current, a, out_bytes);
    }
    free(b); free(a); free(current);
}

int main(int argc, char **argv)
{
    REQUIRE(argc == 4 || argc == 5);
    check_invalid_layouts();
    size_t original_bytes, runtime_bytes, sample_bytes, expected_bytes = 0;
    uint8_t *original = read_file(argv[1], &original_bytes);
    uint8_t *runtime = read_file(argv[2], &runtime_bytes);
    uint8_t *samples = read_file(argv[3], &sample_bytes);
    uint8_t *expected = argc == 5 ? read_file(argv[4], &expected_bytes) : NULL;
    REQUIRE(sample_bytes % MODEL_INPUT_BYTES == 0);
    REQUIRE(!expected || expected_bytes == sample_bytes / MODEL_INPUT_BYTES);
    Model a, b;
    REQUIRE(model_init(&a, &original_spec, original, original_bytes) == MODEL_OK);
    REQUIRE(model_init(&b, &model_spec, runtime, runtime_bytes) == MODEL_OK);
    REQUIRE(model_workspace_bytes(&a) == model_workspace_bytes(&b));
    uint8_t *aw = malloc(MODEL_WORKSPACE_BYTES), *bw = malloc(MODEL_WORKSPACE_BYTES);
    REQUIRE(aw && bw);
    check_layer_views(original, runtime);
    for (size_t i = 0; i < sample_bytes / MODEL_INPUT_BYTES; ++i) {
        const uint8_t *sample = samples + i * MODEL_INPUT_BYTES;
        check_intermediates(original, runtime, sample);
        uint32_t pa, pb;
        REQUIRE(model_forward(&a, sample, MODEL_INPUT_BYTES, aw, MODEL_WORKSPACE_BYTES, &pa) == MODEL_OK);
        REQUIRE(model_forward(&b, sample, MODEL_INPUT_BYTES, bw, MODEL_WORKSPACE_BYTES, &pb) == MODEL_OK);
        REQUIRE(pa == pb && (!expected || pb == expected[i]));
        printf("%u\n", (unsigned)pb);
    }
    free(bw); free(aw); free(expected); free(samples); free(runtime); free(original);
    return 0;
}
