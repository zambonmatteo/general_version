# general_version_v2

LUT inference on the Apollo4 Blue Lite, with results over UART.
Requires Python 3.10+, GNU Make, Arm GNU Toolchain, J-Link and AmbiqSuite 4.5.0
(default SDK path: `../../AmbiqSuite_R4.5.0`).

## Project files

The project separates model preparation on the PC, reusable inference code and
board-specific startup/benchmark code. In the C code, `.c` files implement the
operations and their `.h` headers declare the associated types and functions.

| File | Purpose |
| --- | --- |
| `Makefile` | Runs preparation, compiles and links the selected backend, and invokes J-Link to load the firmware. Also defines SDK paths and benchmark repetitions. |
| `tools/generate.py` | Runs on the PC before compilation. Reads the four files in `data/`, validates model geometry, binary sections, routing indices and sample/reference sizes, then generates `model_specs.h` and `data.S`. |
| `base_structure/bitplane.h` | Defines tensor shapes and helpers to read/write activations packed into bits. |
| `base_structure/lut.c`, `base_structure/lut.h` | Define a LUT view and read the selected output bit from its truth table. |
| `base_structure/encoding.c`, `base_structure/encoding.h` | Convert raw samples into packed binary activations on the MCU, using thermometer thresholds and optional edge filters. |
| `base_structure/layer.c`, `base_structure/layer.h` | Describe and execute LUT convolution, fully connected layers, OR pooling, flattening and GroupSum class selection. |
| `base_structure/model.c`, `base_structure/model.h` | Provide `model_predict()`: encode one sample, run the generated layer sequence with two alternating activation buffers, and return its predicted class. |
| `base_structure/bitslice.c`, `base_structure/bitslice.h` | Accelerate supported convolution and pooling configurations using 32-bit words. Other configurations use the scalar layer routines. |
| `board/main.c` | Initialize the board and UART, verify predictions, measure complete inference latency, and report memory usage and errors. |
| `board/startup_gcc.c` | Provide the interrupt vector table and reset/startup code, initialize program memory and prepare the stack watermark used for memory reporting. |
| `board/linker_script.ld` | Place code, constant data, writable data and stack in the MCU memory regions; define the symbols used to report memory sizes. |
| `board/flash.jlink` | Tell J-Link to load `firmware.elf`, reset the board and start execution. |
| `.gitignore` | Keep user-supplied data, generated builds and Python cache files out of Git while retaining the folder placeholders. |
| `data/.gitkeep`, `build/.gitkeep` | Preserve the otherwise empty input/output directories in the repository. |
| `README.md` | Explain setup, input formats, commands, file responsibilities and measurement scope. |

Python prepares the model description and embeds the supplied binary files; it
does not train the model or encode the samples. Encoding and prediction execute
in C on the MCU. `make` invokes Python automatically, using only Python's standard
library, so no extra Python packages or manual script invocation are needed.

Generated files go under `build/<backend>/`:

| File | Purpose |
| --- | --- |
| `model_specs.h` | Generated C descriptors for encoding and layers, binary offsets, tensor/buffer dimensions and sample counts. |
| `data.S` | Generated assembly directives that embed the model, raw inputs and reference predictions in the firmware. |
| `firmware.elf` | Linked application loaded by J-Link; includes the symbols needed to inspect the firmware. |
| `firmware.bin` | Raw binary image of the same application, produced after linking. |

The workflow is: supply the four data files, let Python validate and describe
them, compile the C/assembly into firmware, and load it with J-Link. After loading,
the board runs the embedded samples autonomously and sends results over UART.

## SDK location

The compiler needs the SDK's MCU headers, HAL and board-support libraries.
Install AmbiqSuite 4.5.0 separately; it is not included in this repository.
`SDK_ROOT` in the Makefile must point to its root directory. Relative paths are
resolved from the project root, where you run `make`.

The defaults match these local directory layouts:

| Project location | Default `SDK_ROOT` | SDK location |
| --- | --- | --- |
| `Desktop/general_version_v2` | `../AmbiqSuite_R4.5.0` | `Desktop/AmbiqSuite_R4.5.0` |
| `Desktop/mcu_implementation/general_version` | `../../AmbiqSuite_R4.5.0` | `Desktop/AmbiqSuite_R4.5.0` |

With the matching layout, use `make flash` directly after supplying your data.
If the SDK is elsewhere, override the path for a command:

```powershell
make flash BACKEND=scalar SDK_ROOT="C:/SDK/AmbiqSuite_R4.5.0"
```

Alternatively, edit the `SDK_ROOT ?= ...` line in your Makefile once. Quote paths
containing spaces. Moving the project or SDK may require updating this path.

## Input files

Place these four matching files in `data/`:

The repository includes `data/` and `build/` with only `.gitkeep` placeholders.
Add your own model and samples locally; their contents and generated build
outputs are ignored by Git.

| Name | Contents |
| --- | --- |
| `model.json` | Model description |
| `model.bin` | LUT tables, routing indices and thresholds |
| `input.bin` | Consecutive raw CHW samples: `uint8` images or little-endian `float32` KWS MFCCs |
| `expected.bin` | One little-endian `uint32` reference class per sample |

To switch models, replace all four files.

### Raw samples and reference predictions

`input.bin` contains headerless samples concatenated in **CHW order**
(channel, row, column). These are the model's raw input values, before
thermometer/edge encoding, rather than packed binary activations:

- For `uint8` images, supply one byte per pixel/channel, in the range 0..255.
  The board divides these values by 255 before applying the encoding thresholds.
- For `float32` inputs such as KWS MFCCs, supply little-endian floats with the
  preprocessing and scale expected by the model. Supply MFCC features, not audio
  recordings; image inputs must likewise contain pixel values, not PNG/JPEG files.

The raw shape comes from `encoding.in_ch` and `encoding.in_hw` in `model.json`.
The type comes from `encoding.input_dtype`, defaulting to `uint8` when omitted.
For a shape of `C x H x W`, each sample occupies `C * H * W` bytes for `uint8`,
or four times that size for `float32`.

`expected.bin` contains exactly one **little-endian uint32 class index** for
every input, in the same order: `N` samples require `4 * N` reference bytes.
These are reference-model predictions for checking agreement, not necessarily
the dataset's ground-truth labels. References stored as `uint8` must be converted
to uint32; simply renaming the old file does not convert it.

The sample count is inferred from file sizes. At least one complete sample and
its reference are required; there is no fixed 32-sample requirement. All supplied
samples are verified and used for timing, without random selection or separate
verification/measurement files. Model data, raw samples and references are
embedded in firmware, so the supplied set must fit in board memory.

Python validates the model and file sizes and generates descriptors and assembly
for embedding the data. **Encoding runs on the board**, followed by LUT inference,
for both backends. A PC C compiler or encoder DLL is not required.

## Run on the board

From the project root, open a UART terminal at **115200 baud, 8N1**
(COM4 in the tests), then run either command:

```powershell
make flash BACKEND=scalar
make flash BACKEND=bitslice
```

The firmware starts automatically and prints verification results, latency (including encoding) and memory usage.
Each sample is timed 10 times by default; use `REPEATS=20` to change this.

### What the latency measures

Each timed call to `model_predict()` starts from one raw sample and includes
thermometer encoding, edge encoding when enabled, and the complete LUT network
through class prediction. Preparation on the PC, firmware loading, reference
comparison and UART reporting are outside the timed interval.

The earlier `mcu_implementation/general_version` encoded samples on the PC and
timed inference on already encoded inputs. Its latency excludes encoding, so
those results and the v2 results measure different work. Rerun benchmarks and
state the timing scope when comparing versions; old latency and memory reports
do not describe this firmware.

### Board validation — 24 November 2026

Tested on the Apollo4 Blue Lite at 96 MHz with Arm GCC 12.2.1 and this Makefile's
`-O2` settings. Each model/backend used 32 raw samples and 10 repetitions per
sample (320 timed inferences). All six runs completed with **zero verification
and benchmark mismatches**.

| Model | Scalar mean | Bitslice mean | Scalar / bitslice |
| --- | --- | --- | --- |
| `kws_final_k32_s0_bestval` | 116.117 ms | 28.677 ms | 4.05x |
| `allLUT2_k32_s0_best` (CIFAR-10) | 178.154 ms | 48.258 ms | 3.69x |
| `mnist_k32_s0_uni_aug_best` | 108.215 ms | 36.327 ms | 2.98x |

These times include MCU encoding and complete inference. Inputs and references
came from the matching `mcu_implementation/implementations/<dataset>/<model>/shared/`
32-sample fixtures; model JSON/binaries came from `<dataset>/models/<model>/`.
The uint8 reference predictions were converted to little-endian uint32.

Firmware image sizes and static SRAM counts were checked against the ELF/BIN
files. The smallest observed stack margin was 2,092 bytes out of 4,096 reserved.
This validates these samples and models; it is not a full-dataset accuracy test,
a worst-case stack bound or validation of every exported model.

## Make commands

| Command | Action |
| --- | --- |
| `make` | Generate and build; equivalent to `make all` and `make board` |
| `make prepare` | Generate only `model_specs.h` and `data.S` |
| `make flash` | Generate, build, load `firmware.elf` and start the test |
| `make clean` | Remove `build/scalar/` or `build/bitslice/`, preserving input files and source code |

All commands accept `BACKEND=scalar` (default) or `BACKEND=bitslice`.
Generated files are placed in `build/scalar/` or `build/bitslice/`.

## Replacing the earlier general_version

This project can occupy the existing `general_version/` repository directory;
its folder name does not affect the build. Preserve the repository's `.git/`
directory and save any uncommitted changes and historical reports before
replacing the implementation. Keep old reports clearly labeled as results of
the earlier version. Exclude generated `build/` files from Git.

The layout and commands have changed: use the four matching files in `data/`,
`make` to build, and `make flash BACKEND=...` to build and load. `make clean`
preserves those input files and removes only the selected backend's build.
The earlier `host`, `encode` and `run` targets are not provided. Models with an
explicit `operations` graph are not accepted by this generator; it supports the
existing conv/fc export format with one learnable LUT layer per convolution block.

When copying this project into `Desktop/mcu_implementation/general_version`, set
`SDK_ROOT ?= ../../AmbiqSuite_R4.5.0` in the destination Makefile. The maintained
copy at that location already uses this default; see **SDK location** above.

Before publishing the replacement, check verification, latency and memory output
on the board for both backends with the models you intend to support.
