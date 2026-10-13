/* PC-only tool. Output: four little-endian uint32 values (table count,
 * ID width, table bytes, ID bytes), followed by table rows and optional IDs. */
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint64_t hash;
    uint32_t first_row;
    uint32_t table_plus_one;
} Bucket;

static int parse_uint32(const char *text, uint32_t *value)
{
    char *end;
    errno = 0;
    unsigned long long parsed = strtoull(text, &end, 10);
    if (errno || !*text || *end || *text == '-' || !parsed || parsed > UINT32_MAX)
        return 0;
    *value = (uint32_t)parsed;
    return 1;
}

static uint64_t row_hash(const uint8_t *row, size_t bytes)
{
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < bytes; ++i)
        hash = (hash ^ row[i]) * UINT64_C(1099511628211);
    return hash;
}

static int write_unsigned(FILE *file, uint32_t value, uint32_t bytes)
{
    for (uint32_t b = 0; b < bytes; ++b)
        if (fputc((int)((value >> (8u * b)) & 255u), file) == EOF) return 0;
    return 1;
}

static int compact_rows(const uint8_t *data, uint32_t rows, uint32_t row_bytes,
                        FILE *output)
{
    size_t capacity = 1;
    while ((uint64_t)capacity < (uint64_t)rows * 2u) {
        if (capacity > SIZE_MAX / 2u) return 0;
        capacity *= 2u;
    }
    if (capacity > SIZE_MAX / sizeof(Bucket) ||
        (uint64_t)rows * sizeof(uint32_t) > SIZE_MAX) return 0;
    Bucket *buckets = calloc(capacity, sizeof(*buckets));
    uint32_t *ids = malloc((size_t)rows * sizeof(*ids));
    uint32_t *first_rows = malloc((size_t)rows * sizeof(*first_rows));
    if (!buckets || !ids || !first_rows) {
        free(buckets);
        free(ids);
        free(first_rows);
        return 0;
    }

    uint32_t unique = 0;
    for (uint32_t k = 0; k < rows; ++k) {
        const uint8_t *row = data + (size_t)k * row_bytes;
        const uint64_t hash = row_hash(row, row_bytes);
        size_t slot = (size_t)hash & (capacity - 1u);
        while (buckets[slot].table_plus_one) {
            const Bucket *bucket = &buckets[slot];
            /* A matching hash only selects a candidate; equality is byte exact. */
            if (bucket->hash == hash &&
                memcmp(row, data + (size_t)bucket->first_row * row_bytes, row_bytes) == 0)
                break;
            slot = (slot + 1u) & (capacity - 1u);
        }
        if (!buckets[slot].table_plus_one) {
            first_rows[unique] = k;
            buckets[slot] = (Bucket){hash, k, ++unique};
        }
        ids[k] = buckets[slot].table_plus_one - 1u;
    }

    uint32_t id_width = unique == 1u ? 0u : unique <= 256u ? 1u : unique <= 65536u ? 2u : 4u;
    uint64_t pool_bytes = (uint64_t)unique * row_bytes;
    uint64_t id_bytes = (uint64_t)rows * id_width;
    const uint32_t original_bytes = rows * row_bytes;
    int direct = pool_bytes + id_bytes >= original_bytes;
    if (direct) {
        unique = rows;
        id_width = 0;
        pool_bytes = original_bytes;
        id_bytes = 0;
    }
    int ok = write_unsigned(output, unique, 4u) && write_unsigned(output, id_width, 4u) &&
             write_unsigned(output, (uint32_t)pool_bytes, 4u) &&
             write_unsigned(output, (uint32_t)id_bytes, 4u);
    if (direct) {
        if (ok) ok = fwrite(data, 1, original_bytes, output) == original_bytes;
    } else {
        for (uint32_t t = 0; ok && t < unique; ++t)
            ok = fwrite(data + (size_t)first_rows[t] * row_bytes, 1, row_bytes, output) == row_bytes;
        for (uint32_t k = 0; ok && id_width && k < rows; ++k)
            ok = write_unsigned(output, ids[k], id_width);
    }
    free(first_rows);
    free(ids);
    free(buckets);
    return ok;
}

int main(int argc, char **argv)
{
    uint32_t n_inputs, rows;
    if (argc != 5 || !parse_uint32(argv[3], &n_inputs) || n_inputs > 31u ||
        !parse_uint32(argv[4], &rows) || strcmp(argv[1], argv[2]) == 0) {
        fprintf(stderr, "Usage: compact_luts INPUT OUTPUT N_INPUTS N_LUTS\n");
        return 1;
    }
    const uint32_t row_bytes = (uint32_t)(((UINT64_C(1) << n_inputs) + 7u) / 8u);
    const uint64_t total = (uint64_t)rows * row_bytes;
    if (total > UINT32_MAX || total > SIZE_MAX) {
        fprintf(stderr, "LUT section exceeds supported size\n");
        return 1;
    }
    FILE *input = fopen(argv[1], "rb");
    if (!input) { perror("Opening LUT input"); return 1; }
    uint8_t *data = malloc((size_t)total);
    int ok = data && fread(data, 1, (size_t)total, input) == total &&
             fgetc(input) == EOF && !ferror(input);
    if (fclose(input) != 0) ok = 0;
    if (!ok) {
        fprintf(stderr, "Cannot read LUT input or its size disagrees with dimensions\n");
        free(data);
        return 1;
    }
    FILE *output = fopen(argv[2], "wb");
    if (!output) { perror("Opening LUT output"); free(data); return 1; }
    ok = compact_rows(data, rows, row_bytes, output);
    if (fclose(output) != 0) ok = 0;
    free(data);
    if (!ok) fprintf(stderr, "LUT compaction failed (allocation or output error)\n");
    return ok ? 0 : 1;
}
