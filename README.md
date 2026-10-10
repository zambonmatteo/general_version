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

Python **3.12+**, GCC and GNU Make must be on `PATH`. The PC targets require no
third-party Python packages. Run from `general_version/`:

```powershell
make host                       # Scalar backend
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

**`bin_board/general.axf`** includes the model and encoded
inputs. The build also produces `general.bin` (application image without the
bootloader), `general.map` and `general.size` (memory placement and sizes) in
`bin_board/`.

Every preparation clears previous host and board binaries: rebuild the board
firmware after a host run. Switching backend needs no cleanup; **`make clean`
also empties `model/`, `input/` and `input_test/`**.

## Load and run the board from the terminal

Install **SEGGER J-Link**, including [J-Link Commander](https://kb.segger.com/J-Link_Commander), and add its installation
directory to `PATH`. The default device is `AMAP42KL-KBR`, matching the supplied
board configuration; override it with `DEVICE=<SEGGER device name>` for another
supported configuration. `JLINK=<executable path>` selects a Commander executable,
and `JLINK_SERIAL=<probe serial number>` selects a particular J-Link probe.

Connect the board's J-Link USB connector **J6** with a data cable and power it
with **SW4**. Supply the model/input files described above. Open a serial
monitor, such as the **Serial Monitor extension in VS Code**, select the board's
COM port and **115200 baud**, then start monitoring before loading the firmware:

```powershell
make                            # Prepare, compile, flash and start the firmware
make BACKEND=bitslice            # Same workflow with the bitslice backend
```

The firmware uses the BSP's **UART2 at 115200 baud, 8 data bits, no parity,
1 stop bit and no flow control**. `make` loads the ELF image with **J-Link
Commander**, then resets and starts the firmware. J-Link exits after starting
the board; the firmware performs its verification, timing and memory report
autonomously. Read its output in the serial monitor. A debugger session is
not required, and the serial monitor can stay open during programming.

Python uses only its standard library for preparation, compilation and loading;
no additional Python packages or environment setup are needed. `PYTHON=<path>`
selects another interpreter if required. The board runs only the compiled C
firmware.

The command finishes when programming and startup complete. Its exit status
reports build/programming failures; firmware verification and benchmark results
appear separately in the serial monitor. J-Link diagnostics are saved in
`bin_board/jlink_flash.log`. To repeat the board run without recompiling or
reloading, keep the monitor open and press the board's **RESET** button.

| Command | Action |
| --- | --- |
| `make` / `make run` | Prepare, build, flash and start the firmware |
| `make board` | Prepare and build firmware only |
| `make flash` | Load and start the existing firmware, without rebuilding or capturing UART |
| `make host` | Prepare, build and run on the PC |

`flash` does not prepare model data or clear build outputs.
Rebuild after changing the model, inputs, code, backend or `REPEATS`.
The `scalar/` and `bitslice/` Makefiles expose the same board commands with their
respective backends. To compare them, keep `input/inputs.bin` and `REPEATS`
identical and save each report from the serial monitor.
PC timings do not estimate Apollo4 performance.

## Results and measurements

UART reports verification mismatches, inference cycles/latency and memory usage.
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
the benchmark. Invalid timing with measurement inputs or an invalid stack
report sets `validation_status` to `3` and prints `Run completed with errors`.
Absent measurement inputs are allowed and produce `N/A` latency, without
running inference. `Run completed` indicates successful completion of the
checks that had input data available.

The MRAM total includes the model, embedded samples and reporting code.
The SRAM peak estimate combines static RAM with stack usage observed before the
memory report; it is **an estimate for this run**, not a guaranteed worst case.
Host results are saved in `bin_host/host_report.txt`; board results appear in
the serial monitor. Use its logging or copy function to save the UART report.

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
