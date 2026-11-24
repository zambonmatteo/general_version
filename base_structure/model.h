#ifndef MODEL_H
#define MODEL_H

#include <stdint.h>

/* Run the compiled model on one raw input. Uses static working memory. */
uint32_t model_predict(const void *input);

#endif
