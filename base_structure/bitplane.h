#ifndef BITPLANE_H
#define BITPLANE_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint32_t channels, height, width;
} Shape;

/* CHW planes use LSB-first bits and independent byte padding. */
static inline size_t bitplane_bytes(uint32_t bits)
{
    return bits / 8u + (bits % 8u != 0u);
}

static inline uint32_t bitplane_get(const uint8_t *data, uint32_t bit)
{
    return (data[bit / 8u] >> (bit % 8u)) & 1u;
}

static inline void bitplane_set(uint8_t *data, uint32_t bit)
{
    data[bit / 8u] |= (uint8_t)(1u << (bit % 8u));
}

#endif
