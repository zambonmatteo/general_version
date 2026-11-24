#ifndef LUT_H
#define LUT_H

#include <stdint.h>


typedef struct {
    const uint8_t *table;
    uint32_t bit_count;
} Lut;


uint8_t lut_read(const Lut *lut, uint32_t bit_address);


#endif
