# General DWNN implementation

A reusable **C inference engine with scalar and bitslice backends** for MNIST, CIFAR-10 and KWS
models, with a PC preparation pipeline and firmware for the **Ambiq Apollo4 Blue
Lite EVB**. One codebase executes LUT convolution, LUT fully connected, binary OR
pooling, CHW flatten and GroupSum/argmax from generated model descriptors.

The PC reads JSON, checks the blob and binarizes inputs. The board receives the
original model blob, constant descriptors and **already encoded inputs**.

## Files to provide

```text
general_version/
  model/
    model.bin             Original LUT tables, routing indices and thresholds
    model.json            Sections, encoding and network operations
  input/
    inputs.bin            Samples for inference measurements
  input_test/
    inputs.bin            At least ten verification samples, when supplied
    ref_preds.bin         Reference model prediction for every source sample
```

Replace both model files together to change models, and supply matching raw
inputs and reference predictions. During preparation, `generated/model_specs.h`
is automatically generated from `model.json`. This C header contains the layer
descriptors, tensor shapes and sequence of operations used by the inference
engine. A separate `model_meta.h` is not required. Alternatively select an
existing model directory with `MODEL_DIR`; input files are still read from
`general_version/input/` and `general_version/input_test/`.

**`model.json` is the only configuration file and is mandatory**
for preparation, encoding and building. Sample counts are always inferred from
each input binary's size; partial samples are rejected.

Raw inputs contain consecutive, headerless CHW samples. Shape and type (`uint8`
or little-endian `float32`) are determined from `model.json`. The encoder divides
`uint8` values by 255; float32 values must already match the reference model's
input scale and preprocessing.

`input_test/ref_preds.bin` defaults to one `uint8` class index per source sample,
in the same order as `inputs.bin`. For little-endian `uint32` predictions, add
this top-level field to **`model.json`**:

```json
"verification": {"prediction_dtype": "uint32"}
```

Inputs are binarized on the PC using `model.json` settings and `model.bin`
thresholds.

All samples in `input/inputs.bin` are saved as binary inputs in
`bin_input/input/encoded.bin`, then used to measure average inference time over
`REPEATS` runs per sample (10 by default). Binarization is excluded from timing.

For verification, **10 distinct samples** are selected randomly from
`input_test/inputs.bin` and binarized into
`bin_input/input_test/encoded.bin`. Their predicted classes are compared with the
corresponding predictions from `input_test/ref_preds.bin` to check agreement
with the reference model. Verification is skipped if its input file is missing,
empty or contains fewer than ten samples. Each preparation reselects the test
samples; measurement inputs retain their file order.

`tools/run.py` automatically creates `encoded_meta.json` beside each `encoded.bin`.
It records the sample count, shape and binary layout, plus the selected source
indices for verification, and is used to prepare the embedded input data.

## Source layout and backend selection

The layout retains separate `scalar/` and `bitslice/` entry points while sharing model data and the deployment pipeline:

```text
general_version/
  scalar/Makefile       Scalar build entry point
  bitslice/Makefile     Bitslice build entry point
  bitslice/bitslice.h   Kernel limits, support checks and internal interfaces
  bitslice/bitslice.c   Word-parallel convolutions and OR pooling

  src/                 Shared model/LUT structs, scalar kernels and host runner
  board/               Shared Apollo4 benchmark, startup and linker script
  tools/               Shared JSON preparation, encoder and build orchestration
  model/, input/, input_test/   Shared external data
```

`BACKEND=scalar` is the default. Set `BACKEND=bitslice` to enable word-parallel
LUT convolution and OR pooling, processing up to **32 spatial positions of one
sample** at a time. FC layers, flatten and GroupSum remain scalar.

Both backends share the same model files, graph executor and two activation
buffers. Layers unsupported by bitslice **automatically fall back to scalar**,
so both backends support the same model format.

## Run on the PC

Python **3.12+**, GCC and GNU Make must be on `PATH`. No third-party Python
packages are required. Run from `general_version/`:

```powershell
make                            # Scalar backend
make host BACKEND=bitslice       # Bitslice backend
```

Each command prepares the model and inputs, compiles and runs the host
executable, and reports verification results and average **PC inference time**,
subject to `clock()` resolution.
A verification mismatch fails the command. Successful runs save the executable
and `host_report.txt` in `bin_host/`.

Add `REPEATS=<count>` to change repetitions or `MODEL_DIR=<path>` to select a
model directory. `make encode` only binarizes inputs; `make prepare` also
generates deployment data. Neither builds an executable.

## Build for Apollo4

In addition to the PC requirements, install Arm GNU Toolchain with its `bin/`
directory on `PATH`, and **AmbiqSuite SDK 4.5.0** with its HAL and BSP libraries
built. From
`general_version/`, choose one backend:

```powershell
make board BACKEND=scalar
make board BACKEND=bitslice
```

`SDK_ROOT` is optional: the Makefile uses `AMBIQ_SDK_ROOT` when set, otherwise
`../../AmbiqSuite_R4.5.0`. Only add `SDK_ROOT=<your SDK installation path>` if
the SDK is installed elsewhere.

Load/debug **`bin_board/general.axf`**, which includes the model and encoded
inputs. The build also produces `general.bin` (application image without the
bootloader), `general.map` and `general.size` (memory placement and sizes) in
`bin_board/`.

Every preparation clears previous host and board binaries: rebuild the board
firmware after a host run. Switching backend needs no cleanup; **`make clean`
also empties `model/`, `input/` and `input_test/`**.

## Load and run the board

**Provisional procedure: these instructions must be updated and validated for
the final board setup.**

1. Connect the J-Link USB-C connector **J6** with a data cable and power the
   board with **SW4**.
2. Install SEGGER J-Link and VS Code Cortex-Debug.
3. Open **`general_version/`** as the VS Code workspace.
4. In `.vscode/launch.json`, adjust `serverpath`, `armToolchainPath` and `svdFile`
   to match your installations: the supplied paths reflect one Windows setup
   and may differ on your PC. Check `device` against your board
   ([Apollo4 Blue Lite: `AMA4B2KL-KXR`](https://ambiq.com/product/apollo4-blue-lite/)).
5. In a terminal at `general_version/`, compile with `make board BACKEND=scalar`
   or `make board BACKEND=bitslice`. Select **General model - Apollo4** and press
   **F5** to load and run `bin_board/general.axf`. Continue if paused and read
   the SWO console.
6. To repeat the measurement, stop with **Shift+F5**, clear the SWO console and
   press **F5** again; rebuilding is unnecessary if the firmware is unchanged.

For an optional PC check followed by a bitslice firmware build, run in order:

```powershell
make host BACKEND=bitslice       # Compile and run on the PC
make board BACKEND=bitslice      # Compile firmware for the board
```

Check the host's verification results before building the firmware, then load
it with **F5**. To compare backends, repeat with `BACKEND=scalar`, keeping
`input/inputs.bin` and `REPEATS` identical, and save each SWO report. PC timings
do not estimate Apollo4 performance.

**F5 loads existing firmware; it does not compile it.** Rebuild after changing
the model, inputs, code, backend or `REPEATS`. SWO assumes 96 MHz; adjust
`cpuFrequency` for a different clock. Do not pause or step during measurements.

## Results and measurements

SWO reports verification mismatches, inference cycles/latency and memory usage.
Timing covers inference after an untimed warm-up, excluding input binarization,
model initialization, reference comparisons and reporting.

Debugger values:

- `validation_status`: `0` starting, `1` running, `2` completed, `3` error.
- `validation_mismatches`: failed reference checks; completion alone does not
  imply that predictions match.
- `validation_predictions` and `measurement_predictions`: predicted classes;
  `measurement_predictions_valid` confirms that measurement predictions exist.
- `timing.valid`: `1` when timing checks passed. Results are in
  `timing.mean_cycles`, `mean_ms`, `min_cycles`, `max_cycles` and `runs`.

`0 / 0` means verification was skipped. Latency is unavailable when measurement
inputs are absent or timing checks fail. On the board, failed verification skips
the benchmark.

The MRAM total includes the model, embedded samples and reporting code.
The SRAM peak estimate combines static RAM with stack usage observed before the
memory report; it is **an estimate for this run**, not a guaranteed worst case.
Host results are saved in `bin_host/host_report.txt`; board results are read
from SWO/debugger.

## Generated files and model support

`make clean` **deletes the supplied model and raw inputs** by emptying `model/`,
`input/` and `input_test/`. It also removes `bin_input/`, `generated/`,
`bin_host/` and `bin_board/`. Source code, documentation and external model
directories selected through `MODEL_DIR` are preserved.

Normal preparation keeps the supplied files but clears both previous host and
board binaries. Use `make host` or `make board` to build the target you want to
run; `make prepare` generates deployment data only.

## Model description and C interface

`model.json` describes the layers, their execution order and input encoding.
`model.bin` stores LUT tables, routing indices and encoding thresholds.
Preparation checks both files and creates `generated/model_specs.h`: a C header
describing tensor shapes, operations and where layer data is stored in the
binary file. It is generated automatically; update the model files and rebuild
to change the network.

For models that explicitly list their operations, `model.json` must contain
these four fields:

- `total_bytes`: the expected size of `model.bin`, in bytes.
- `layers`: descriptions of the LUT layers and the locations of their binary data.
- `encoding`: settings for binarizing the raw inputs.
- `operations`: the sequence of operations to execute during inference.

Supported operations are convolution (`lut_conv`),
OR pooling (`or_pool`), flattening (`flatten`), fully connected layers (`lut_fc`)
and class selection (`groupsum`). Each operation's output shape must match the
next input shape, with `groupsum` last. JSON without `operations` is resolved
from its `conv` and `fc` records into the same execution descriptors. This form
inserts OR pooling after each convolution, flatten before the FC layers and a
final GroupSum head; pooling defaults to 2x2 and convolution stride to 1 when
omitted. Use `operations` to declare other supported execution sequences.

To call the engine from your own C code, use `src/model.h`:

1. Call `model_init()` once with the generated model description and binary data.
2. Call `model_workspace_bytes()` to find how much working memory to provide.
3. Call `model_forward()` for each binarized sample, passing that workspace and
   a variable that receives the predicted class. Check for `MODEL_OK` after
   initialization and each inference.

The C engine does not binarize raw inputs or allocate memory. Inputs must already
use the encoder's CHW format (channel, height, width), with bits packed least
significant first and each plane padded to whole bytes. Keep the model data and
descriptors valid and unchanged; input, workspace and prediction buffers must
not overlap each other or overwrite them.