#include "model.h"
#include "lut.h"
#include "tensor.h"
#include <float.h>

/* Table rows are independently byte-padded, including LUTs with n < 3.
 * Specs restrict n to 1..31 so the shift is defined on uint32_t. */
static uint32_t table_bytes(uint32_t n)
{
    return (uint32_t)(((UINT64_C(1) << n) + 7u) / 8u);
}

static int section_fits(BlobSection section, size_t blob_bytes)
{
    return section.offset <= blob_bytes &&
           section.bytes <= blob_bytes - section.offset;
}

static int shape_valid(TensorShape shape)
{
    /* Restrict flattened bit counts to uint32_t, without overflowing products. */
    uint64_t positions = (uint64_t)shape.height * shape.width;
    return shape.channels && positions && positions <= UINT32_MAX &&
           shape.channels <= UINT32_MAX / positions;
}

uint32_t lut_index(const Lut *lut, uint32_t j)
{
    const uint8_t *entry = lut->indices + (size_t)j * lut->index_bytes;
    uint32_t value = 0;
    for (uint32_t b = 0; b < lut->index_bytes; ++b)
        value |= (uint32_t)entry[b] << (8u * b);
    return value;
}

ModelStatus lut_layer_get(const LutLayer *layer, uint32_t k, Lut *lut)
{
    if (!lut) return MODEL_INVALID_ARGUMENT;
    *lut = (Lut){0};
    if (!layer || !layer->spec || !layer->tables || !layer->indices)
        return MODEL_INVALID_ARGUMENT;
    const LutLayerSpec *spec = layer->spec;
    if (k >= spec->n_luts) return MODEL_INVALID_ARGUMENT;

    *lut = (Lut){
        .n_inputs = spec->n_inputs,
        .index_bytes = spec->index_bytes,
        .table = layer->tables + (size_t)k * table_bytes(spec->n_inputs),
        .indices = layer->indices + (size_t)k * spec->n_inputs * spec->index_bytes
    };
    return MODEL_OK;
}

ModelStatus model_init(Model *model, const ModelSpec *spec,
                       const uint8_t *blob, size_t blob_bytes)
{
    if (!model) return MODEL_INVALID_ARGUMENT;
    *model = (Model){0};
    if (!spec || !blob || !spec->operations || !spec->operation_count)
        return MODEL_INVALID_ARGUMENT;
    if (blob_bytes != spec->blob_bytes) return MODEL_INVALID_BLOB;
    size_t maximum = tensor_bytes(spec->input_shape);
    if (!maximum) return MODEL_INVALID_SPEC;
    TensorShape current = spec->input_shape;
    for (uint32_t i = 0; i < spec->operation_count; ++i) {
        const Operation *op = &spec->operations[i];
        if (!tensor_shape_equal(current, op->input_shape)) return MODEL_INVALID_SPEC;
        size_t bytes = tensor_bytes(op->output_shape);
        if (!bytes) return MODEL_INVALID_SPEC;
        switch (op->type) {
        case OP_LUT: {
            LutLayer layer;
            const LutLayerSpec *s = op->parameters.lut;
            if (!s || !tensor_shape_equal(s->input_shape, op->input_shape) ||
                !tensor_shape_equal(s->output_shape, op->output_shape)) return MODEL_INVALID_SPEC;
            ModelStatus status = lut_layer_bind(&layer, s, blob, blob_bytes);
            if (status != MODEL_OK) return status;
            break;
        }
        case OP_OR_POOL: {
            TensorShape expected;
            if (tensor_pool_shape(current, op->parameters.pool, &expected) != MODEL_OK ||
                !tensor_shape_equal(expected, op->output_shape)) return MODEL_INVALID_SPEC;
            break;
        }
        case OP_FLATTEN: {
            const uint32_t bits = current.channels * current.height * current.width;
            if (!tensor_shape_equal(op->output_shape, (TensorShape){1, 1, bits}))
                return MODEL_INVALID_SPEC;
            break;
        }
        case OP_GROUP_SUM: {
            const GroupSumHead h = op->parameters.head;
            if (i != spec->operation_count - 1u || current.channels != 1u || current.height != 1u ||
                !h.classes || current.width % h.classes || !(h.tau > 0.0f && h.tau <= FLT_MAX) ||
                !tensor_shape_equal(op->output_shape, (TensorShape){1, 1, h.classes}))
                return MODEL_INVALID_SPEC;
            /* Scores are compared one at a time, so no score buffer is needed. */
            bytes = 0;
            break;
        }
        default: return MODEL_INVALID_SPEC;
        }
        if (bytes > maximum) maximum = bytes;
        current = op->output_shape;
    }
    if (spec->operations[spec->operation_count - 1u].type != OP_GROUP_SUM || maximum > SIZE_MAX / 2u)
        return MODEL_INVALID_SPEC;
    *model = (Model){spec, blob, blob_bytes, maximum};
    return MODEL_OK;
}

size_t model_workspace_bytes(const Model *model)
{
    return model && model->spec ? 2u * model->activation_bytes : 0;
}

ModelStatus model_forward(const Model *model, const uint8_t *input, size_t input_bytes,
                          uint8_t *workspace, size_t workspace_bytes, uint32_t *prediction)
{
    if (!model || !model->spec || !input || !workspace || !prediction)
        return MODEL_INVALID_ARGUMENT;
    const size_t need_input = tensor_bytes(model->spec->input_shape);
    const size_t need_workspace = model_workspace_bytes(model);
    if (input_bytes < need_input || workspace_bytes < need_workspace ||
        tensor_ranges_overlap(input, need_input, workspace, need_workspace) ||
        tensor_ranges_overlap(model->blob, model->blob_bytes, workspace, need_workspace) ||
        tensor_ranges_overlap(prediction, sizeof(*prediction), workspace, need_workspace) ||
        tensor_ranges_overlap(prediction, sizeof(*prediction), input, need_input) ||
        tensor_ranges_overlap(prediction, sizeof(*prediction), model->blob, model->blob_bytes))
        return MODEL_INVALID_ARGUMENT;

    const uint8_t *src = input;
    uint8_t *dst = workspace;
    const size_t capacity = model->activation_bytes;
    /* The input is already encoded. Alternate two buffers for intermediate
     * tensors; LUT views are constructed without rescanning validated mappers. */
    for (uint32_t i = 0; i < model->spec->operation_count; ++i) {
        const Operation *op = &model->spec->operations[i];
        ModelStatus status;
        const size_t src_bytes = tensor_bytes(op->input_shape);
        switch (op->type) {
        case OP_LUT: {
            const LutLayerSpec *s = op->parameters.lut;
            const LutLayer layer = {s, model->blob + s->tables.offset, model->blob + s->indices.offset};
            status = lut_layer_forward(&layer, src, src_bytes, dst, capacity);
            break;
        }
        case OP_OR_POOL:
            status = tensor_or_pool(src, src_bytes, op->input_shape, op->parameters.pool, dst, capacity);
            break;
        case OP_FLATTEN:
            status = tensor_flatten(src, src_bytes, op->input_shape, dst, capacity);
            break;
        case OP_GROUP_SUM:
            return tensor_group_argmax(src, src_bytes, op->input_shape.width,
                                       op->parameters.head.classes, op->parameters.head.tau, prediction);
        default: return MODEL_INVALID_SPEC;
        }
        if (status != MODEL_OK) return status;
        src = dst;
        dst = dst == workspace ? workspace + capacity : workspace;
    }
    return MODEL_INVALID_SPEC;
}

ModelStatus lut_layer_bind(LutLayer *layer, const LutLayerSpec *spec,
                           const uint8_t *blob, size_t blob_bytes)
{
    if (!layer) return MODEL_INVALID_ARGUMENT;
    *layer = (LutLayer){0};
    if (!spec || !blob) return MODEL_INVALID_ARGUMENT;
    if (!spec->n_luts || !spec->n_inputs || spec->n_inputs > 31u ||
        (spec->index_bytes != 2u && spec->index_bytes != 4u) ||
        !shape_valid(spec->input_shape) || !shape_valid(spec->output_shape))
        return MODEL_INVALID_SPEC;

    const uint64_t wanted_tables = (uint64_t)spec->n_luts * table_bytes(spec->n_inputs);
    const uint64_t wanted_indices = (uint64_t)spec->n_luts * spec->n_inputs * spec->index_bytes;
    if (spec->tables.bytes != wanted_tables || spec->indices.bytes != wanted_indices)
        return MODEL_INVALID_SPEC;
    if (!section_fits(spec->tables, blob_bytes) || !section_fits(spec->indices, blob_bytes))
        return MODEL_INVALID_BLOB;
    /* Tables and mapper entries must describe separate byte ranges. */
    if ((uint64_t)spec->tables.offset < (uint64_t)spec->indices.offset + spec->indices.bytes &&
        (uint64_t)spec->indices.offset < (uint64_t)spec->tables.offset + spec->tables.bytes)
        return MODEL_INVALID_SPEC;

    uint64_t index_limit;
    if (spec->type == LUT_FULLY_CONNECTED) {
        if (spec->input_shape.channels != 1u || spec->input_shape.height != 1u ||
            spec->output_shape.channels != 1u || spec->output_shape.height != 1u ||
            spec->output_shape.width != spec->n_luts)
            return MODEL_INVALID_SPEC;
        index_limit = spec->input_shape.width;
    } else if (spec->type == LUT_CONVOLUTION) {
        const ConvGeometry *g = &spec->convolution;
        if (!g->kernel_h || !g->kernel_w || !g->stride_h || !g->stride_w ||
            spec->output_shape.channels != spec->n_luts)
            return MODEL_INVALID_SPEC;
        const uint64_t padded_h = (uint64_t)spec->input_shape.height + 2ull * g->padding_h;
        const uint64_t padded_w = (uint64_t)spec->input_shape.width + 2ull * g->padding_w;
        if (padded_h < g->kernel_h || padded_w < g->kernel_w ||
            (padded_h - g->kernel_h) / g->stride_h + 1u != spec->output_shape.height ||
            (padded_w - g->kernel_w) / g->stride_w + 1u != spec->output_shape.width)
            return MODEL_INVALID_SPEC;
        index_limit = (uint64_t)g->kernel_h * g->kernel_w;
        if (index_limit > UINT32_MAX || spec->input_shape.channels > UINT32_MAX / index_limit)
            return MODEL_INVALID_SPEC;
        index_limit *= spec->input_shape.channels;
    } else {
        return MODEL_INVALID_SPEC;
    }

    LutLayer bound = {spec, blob + spec->tables.offset, blob + spec->indices.offset};
    /* Validate all LUT routing indices before assigning the bound layer. */
    for (uint32_t k = 0; k < spec->n_luts; ++k) {
        Lut lut;
        (void)lut_layer_get(&bound, k, &lut);
        for (uint32_t j = 0; j < lut.n_inputs; ++j)
            if (lut_index(&lut, j) >= index_limit) return MODEL_INVALID_INDEX;
    }
    *layer = bound;
    return MODEL_OK;
}
