#include "lut.h"

uint8_t lut_read(const Lut *lut, uint32_t bit_address)
{
    return (uint8_t)((lut->table[bit_address / 8u] >> (bit_address % 8u)) & 1u);
}
