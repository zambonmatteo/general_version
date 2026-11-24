#include "model.h"
#include "model_specs.h"

static uint8_t buffers[2][MODEL_BUFFER_BYTES];
#if MODEL_EDGE_CHANNELS
static float gray[MODEL_SCRATCH_FLOATS];
#endif

uint32_t model_predict(const void *input)
{
#if MODEL_EDGE_CHANNELS
    float *scratch = gray;
#else
    float *scratch = NULL;
#endif
    encoding_forward(&model_encoding, input, buffers[0], scratch);
    uint32_t source = 0;
    for (uint32_t i = 0; i < MODEL_LAYER_COUNT; ++i) {
        const Layer *layer = &model_layers[i];
        if (layer->type == LAYER_GROUP_SUM)
            return layer_group_sum(buffers[source], layer->parameters.groupsum);
        layer_forward(layer, buffers[source], buffers[source ^ 1u]);
        source ^= 1u;
    }
    return 0;
}
