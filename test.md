The implemented LUT table sharing is correct, but it does not reduce model
storage for any of the 18 available model exports. Its usefulness depends on
how many complete table rows are identical and on the cost of identifying the
shared table used by each neuron. This document describes the implementation,
the checks performed, and why the current representation is not worthwhile
for these models.

A neuron has two separate pieces of model data: a truth table and a list of
input indices. The indices select the input bits and their order; those bits
form the address used to read one result from the table. Sharing the table
does not change the indices, input order, or evaluation of the neuron.
Neurons using the same table can still produce different outputs because
they read different inputs.

For example, a layer containing tables `[F, G, F]` can store a pool `[F, G]`
and table IDs `[0, 1, 0]`. Each neuron retains its original input indices.
Only byte-identical table rows within the same layer are eligible for sharing.
Having the same number of inputs or the same table size is not sufficient.
There is no sharing across layers, input permutation, routing deduplication,
or reuse of computed outputs.

The preparation sequence is:

1. [tools/run.py](tools/run.py) validates the source model and encoding settings
   and prepares any supplied raw inputs using the existing workflow.
2. [tools/compact_model.py](tools/compact_model.py) compiles
   [tools/compact_luts.c](tools/compact_luts.c) with the PC compiler and passes
   each source table section to the resulting executable. Python orchestrates
   this work; the duplicate search and production of the table pool and IDs
   are implemented in C.
3. The C program hashes each complete row, then confirms equality with
   `memcmp`. Hash collisions cannot cause different tables to be merged.
   Distinct rows are stored in their order of first appearance.
4. The C program chooses the smaller representation, including the IDs.
   Python constructs the runtime blob, physically removes redundant rows
   where sharing is selected, and relocates the affected sections.
5. Preparation generates matching C descriptors and runtime layout metadata.
   The selected backend is then compiled against that prepared model.

The compactor runs on the PC, not during initialization or inference on the
board. It is used by `make prepare`, `make host`, `make board`, and the complete
build/run workflow. `make encode` only prepares inputs. The standalone
`tools/prepare_model.py` command produces descriptors for the original blob;
it does not run compaction.

Let `N` be the number of neurons, `U` the number of distinct tables, and `B`
the bytes per table. A LUT with `n` inputs contains `2^n` one-bit results, so
the current independently byte-padded row size is `ceil(2^n / 8)`.
The storage decision has three cases:

- If all tables are identical, store one table without per-neuron IDs.
- If only some tables are identical, store the distinct tables and an ID for
  every neuron. IDs are unsigned little-endian values: 1 byte for up to 256
  distinct tables, 2 bytes for up to 65,536, and 4 bytes above that.
- If the pool and IDs occupy as much space as the original section or more,
  preserve the original table rows and do not store IDs.

For partial sharing, the comparison is:

```text
Original bytes = N * B
Shared bytes   = U * B + N * ID_bytes

Use sharing only when Shared bytes < Original bytes.
```

For example, 16 tables with 6 inputs occupy `16 * 8 = 128` bytes. If only three
distinct tables exist, the pool occupies 24 bytes and the one-byte IDs occupy
16 bytes. The result is 40 bytes, saving 88 bytes. If all 16 tables are
identical, one 8-byte table is sufficient, saving 120 bytes without a map.

The temporary output of the C tool contains a 16-byte header describing the
pool and IDs. This header is read by Python and is not embedded in the runtime
blob; it is not part of the model storage calculation.

The source `model.json` and `model.bin` are not modified. Preparation writes:

- `generated/model.bin`: the runtime blob actually embedded in the executable.
- `generated/model_specs.h`: the matching C descriptors and runtime blob size.
- `generated/model_layout.json`: relocated section offsets, layer descriptors,
  source size, runtime size, and saved bytes.
- `generated/source_model.json`: a snapshot of the source description, whose
  offsets refer to the original source blob, not to the runtime blob.

All routing bytes, other source sections, and gaps are preserved. Routing
offsets can change when an earlier table section becomes smaller, but routing
values do not change. The generated runtime files are not a source export to
copy back into `model/`.

In [src/model.h](src/model.h), the layer descriptor now records `n_tables`,
`table_id_bytes`, and the `table_ids` section. In [src/model.c](src/model.c),
`lut_layer_get` selects row zero for a single shared table, row `k` for the
original layout, or the row selected by neuron `k`'s ID for partial sharing.
It then returns the selected table pointer and that neuron's original routing
pointer. Scalar and bitslice use the same LUT views; their evaluation kernels
do not need different truth-table logic.

Initialization validates the table count, ID width, section sizes and ranges,
section overlap, table IDs, and routing indices. IDs are checked before they
can be used to construct table pointers. Tables and IDs remain in immutable
model storage and are not reconstructed as RAM copies. The implementation
does introduce additional descriptors, pointers, and lookup/validation code.

On 13 October 2026, all available exports in the sibling MNIST, CIFAR, and KWS
model directories were inspected: 18 models and 104 LUT layers. The current
C compactor was executed on every model. Every reconstructed LUT row and
every routing section was compared with its source. For all 18 models the
resulting runtime blob was byte-identical to the source blob because no layer
passed the storage-benefit check.

The table below also records a separate analysis of bit-packed IDs. This
alternative uses `ceil(log2(U))` bits per ID, rounded to whole bytes for each
layer's map. It is not implemented in the firmware. For the favorable layers,
the packed maps were constructed in memory and every table was reconstructed
exactly. Duplicate counts are summed within layers, not across the model.

| Model | Source blob bytes | Duplicate rows | Current saving | Bit-packed ID saving |
|---|---:|---:|---:|---:|
| `allLUT2_k32_s0_best` | 43,556 | 8,592 | 0 | 4,244 |
| `allLUT4_k32_s0_best` | 87,036 | 1,831 | 0 | 0 |
| `allLUT8_k32_s0_best` | 417,484 | 0 | 0 | 0 |
| `cifar_k32_s0_n2_best` | 148,556 | 1,638 | 0 | 790 |
| `cifar_k32_s0_n4_best` | 157,036 | 246 | 0 | 0 |
| `cifar_k32_s0_n8_best` | 221,484 | 0 | 0 | 0 |
| `cifar_k32_s0_lr3e2_crop2_best` | 173,996 | 1 | 0 | 0 |
| `cifar_k128_s0_lr3e2_crop2_best` | 275,756 | 0 | 0 | 0 |
| `cifar_k256_s0_lr3e2_crop2_best` | 411,436 | 1 | 0 | 0 |
| `kws_final_k32_s0_bestval` | 35,616 | 0 | 0 | 0 |
| `kws_final_k64_s0_bestval` | 70,816 | 0 | 0 | 0 |
| `kws_final_k128_s0_bestval` | 141,216 | 0 | 0 | 0 |
| `kws_final_k256_s0_bestval` | 154,016 | 0 | 0 | 0 |
| `mnist_k16_s0_uni_aug_best` | 64,172 | 0 | 0 | 0 |
| `mnist_k32_s0_uni_aug_best` | 68,332 | 0 | 0 | 0 |
| `mnist_k64_s0_uni_aug_best` | 76,652 | 0 | 0 | 0 |
| `mnist_k128_s0_uni_aug_best` | 93,292 | 0 | 0 | 0 |
| `mnist_k256_s0_uni_aug_fc1_4000_best` | 166,572 | 0 | 0 | 0 |

For every KWS and MNIST model, all tables are distinct within their layers.
Equal dimensions do not provide an opportunity to share the table contents.
KWS k32, for example, has 32, 128, 1,000, and 600 tables in its four layers;
all are distinct. Its 14,080 table bytes therefore remain unchanged.

For CIFAR n4 and allLUT4, duplicates exist but their tables are only two bytes
each, and too many distinct tables remain. Even a bit-packed ID map costs
more than the removed copies. In CIFAR n4's fourth convolution layer, 1,024
rows contain 814 distinct tables. Removing 210 copies saves 420 bytes, but
the current two-byte IDs cost 2,048 bytes. Sharing would increase the section
from 2,048 to 3,676 bytes, so the implementation preserves its original layout.

For the two CIFAR variants with 2-input LUTs, the current format has a specific
limitation: a table already occupies one byte, and each table ID occupies at
least one byte. Partial sharing therefore cannot save space, even when many
tables repeat. Four-bit IDs would reduce the model blob by 4,244 bytes (9.74%)
for allLUT2 and by 790 bytes (0.53%) for CIFAR n2.

However, each 2-input LUT contains only four useful bits. Storing two complete
tables directly in one byte avoids both a dictionary and an ID map. This
alternative was also checked in memory: the affected source padding bits are
zero, and unpacking reconstructs every original row exactly. It is not
implemented in the firmware.

| Model | Saving with 4-bit IDs | Saving with direct 4-bit tables | Direct-table blob reduction |
|---|---:|---:|---:|
| `allLUT2_k32_s0_best` | 4,244 bytes | 4,348 bytes | 9.98% |
| `cifar_k32_s0_n2_best` | 790 bytes | 848 bytes | 0.57% |

The current sharing feature is therefore not justified as a memory
optimization for these exports: it reduces none of their blobs and adds
support code and metadata. It would become useful for other models containing
enough identical rows to pay for the map, especially larger tables or layers
where every table is identical. For the available 2-input variants, direct
four-bit table storage is a better candidate than extending table sharing.

All savings above concern model data and include the relevant maps. They do
not include new firmware code or descriptor costs, and they do not establish
a latency improvement. Any new storage format would still require firmware
implementation, compilation, equivalence checks, and performance measurement.

The retained [tests/](tests/) directory contains regression checks for the
implemented format. It covers identical, partially identical, and distinct
tables; equal-cost fallback; small LUT padding; ID-width boundaries; separate
layer pools; relocated data and gaps; and invalid descriptors and IDs. It also
compares original and prepared layer outputs and predictions with scalar and
bitslice, and checks source preservation and the preparation/host workflow.
Real-model checks use sibling reference fixtures when they are available.

Run the checks from the project root with GCC on `PATH`:

```text
python -B -m unittest discover -s tests -v
```

Previous verification also checked ten real inputs each for MNIST k32, CIFAR
k32 n4, and KWS k32, with zero prediction mismatches in both backends. ARM
firmware for both backends was compiled and linked using a synthetic model
with shared tables and a measurement input. No firmware was programmed onto
the board. The separate one-off `reports/` audit files have been removed;
their model-level findings are preserved in this document.
