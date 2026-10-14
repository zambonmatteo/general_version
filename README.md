# General DWNN implementation

A reusable C inference engine with **scalar and bitslice backends** for exported
MNIST, CIFAR and KWS LUT models, with a PC preparation pipeline and firmware
for the **Ambiq Apollo4 Blue Lite EVB**.

The normal workflow is:

1. Supply a model, raw inputs and matching reference predictions.
2. Optionally check the model on the PC with `make host`.
3. Open a serial monitor to receive the board's output.
4. Run `make` to prepare, compile, load and start the firmware.
5. Read verification, latency and memory results in the serial monitor.

The PC validates JSON and binary data and binarizes inputs. The board executes
inference using the embedded model, generated descriptors and already encoded
inputs. Loading uses **J-Link Commander**; output uses **UART**. The firmware
runs autonomously after loading, without a VS Code debugger session.

## 1. Install the required tools

| Tool | Purpose |
| --- | --- |
| Python 3.12 or later | Model preparation, input encoding and build/loading orchestration |
| GCC | Compile the PC input encoder and the optional host executable |
| GNU Make | Run the commands in this guide |
| Arm GNU Toolchain (`arm-none-eabi-gcc`, `arm-none-eabi-objcopy`, `arm-none-eabi-size`) | Compile and inspect Apollo4 firmware |
| AmbiqSuite SDK 4.5.0 | Apollo4 HAL, BSP, headers and utility sources |
| SEGGER J-Link, including [J-Link Commander](https://kb.segger.com/J-Link_Commander) | Load and start the firmware |
| A serial monitor | Display UART output from the board |

Add Python, Make, compiler and J-Link executable directories to `PATH`.
**GCC is also required for board builds with raw input samples**, because
input encoding runs on the PC.
On Windows, use a GCC build targeting the same architecture as Python
(for example, 64-bit MinGW-w64 with 64-bit Python), so Python can load the
encoder DLL.
Python uses only its standard library; no virtual environment or third-party
Python packages are required.

The SDK's HAL and BSP libraries must already be built at:

```text
AmbiqSuite_R4.5.0/
  mcu/apollo4l/hal/mcu/gcc/bin/libam_hal.a
  boards/apollo4l_blue_evb/bsp/gcc/bin/libam_bsp.a
```

The default SDK path is `../../AmbiqSuite_R4.5.0`, relative to this project.
An `AMBIQ_SDK_ROOT` environment variable overrides that default. Alternatively,
pass the installation path explicitly:

```powershell
make BACKEND=scalar SDK_ROOT="C:/SDK/AmbiqSuite_R4.5.0"
```

VS Code is optional. The commands can run in a normal terminal, and any suitable
serial terminal can receive the output.

## 2. Open the project and start from a clean state

The examples use **Windows PowerShell**. Open a terminal in `general_version/`.
For example, replace the path below with your project location:

```powershell
cd C:\path\to\general_version
```

To start from an empty local setup, run this **before copying the new data**:

```powershell
make clean
```

This command removes the contents of `model/`, `input/` and `input_test/`,
recreating their `.gitkeep` files. It also removes `bin_input/`, `generated/`,
`bin_host/` and `bin_board/`. Source files, the SDK and model/data directories
outside this project are preserved.

Skip this cleanup if you want to reuse data already supplied in the project.
Switching between scalar and bitslice does not require `make clean`.

## 3. Supply the model and input files

Prepare this directory structure:

```text
general_version/
  model/
    model.json
    model.bin
  input/
    inputs.bin
  input_test/
    inputs.bin
    ref_preds.bin
```

| File | What to provide |
| --- | --- |
| `model/model.json` | Exported model description, encoding settings and network geometry/operations |
| `model/model.bin` | Matching LUT tables, routing indices and encoding thresholds |
| `input/inputs.bin` | Raw timing samples; at least one sample enables the benchmark |
| `input_test/inputs.bin` | Raw verification samples; at least ten samples enable verification |
| `input_test/ref_preds.bin` | This model's reference class prediction for every raw verification sample, in the same order |

**Replace `model.json` and `model.bin` together.** When changing models, supply
reference predictions from the new model as well. Raw inputs can be reused only
if their shape, type and preprocessing match the new model's input requirements.

`model.json` is the only configuration file. Sample counts are inferred from
binary file sizes. Preparation generates `generated/model_specs.h` automatically;
you do not need to copy `model_meta.h` or write a C descriptor by hand.
Do not place `config.json` in `input/` or `input_test/`; preparation rejects it.

### Raw input format

Each `inputs.bin` contains consecutive, headerless samples in **CHW order**
(channel, height, width). Raw shape comes from `encoding.in_ch` and
`encoding.in_hw` in `model.json`; raw type comes from `encoding.input_dtype`:

- `uint8`: one byte per value; the PC encoder divides values by 255.
- Little-endian `float32`: four bytes per value; values must already have the
  preprocessing and scale expected by the reference model.

Models with an explicit `operations` list must declare `encoding.input_dtype`.
For exports without `operations`, an omitted input type defaults to `uint8`.

Both backends use the same raw input format. The PC encodes these inputs before
embedding them in the firmware.

Reference predictions default to **one `uint8` class index per raw test sample**.
To use little-endian `uint32` class indices, add this top-level field to
`model.json`:

```json
"verification": {"prediction_dtype": "uint32"}
```

### Sample selection and benchmark size

You can supply files containing more than 32 samples, provided each sample
matches the model's raw input format. **32 is only an example dataset size;
the benchmark has no fixed 32-sample limit.** Counts are inferred from file sizes.

| Input file | Samples used |
| --- | --- |
| `input_test/inputs.bin` | Exactly **10 distinct samples selected randomly** when at least 10 are supplied; fewer than 10 skips verification |
| `input/inputs.bin` | **All supplied samples**, in file order, each timed `REPEATS` times |

Every preparation selects the verification indices again and keeps inputs
and predictions aligned. With exactly ten supplied test samples, all
ten are checked, possibly in a different order. `input_test/ref_preds.bin`
must contain one reference prediction for **every supplied test sample**, in
the same order, even though only ten are selected for verification.

For example, 1,000 samples in `input_test/` require 1,000 reference predictions
and produce 10 checks. Separately, 100 samples in `input/` with `REPEATS=10`
produce **1,000 timed inferences** (`measurement sample count * REPEATS`).

All encoded measurement samples, plus the ten selected verification samples
and their reference predictions when verification is enabled, are embedded in
firmware. Select a measurement subset that fits in board memory together with
the model; a large input file can exceed the available memory.

## 4. Optionally verify on the PC

Run:

```powershell
make host BACKEND=scalar
```

With ten test inputs and matching reference predictions, expect:

```text
Verification mismatches: 0 / 10
```

Each check also prints the source index, predicted class and expected class.
A mismatch makes the host command fail. You can check bitslice similarly:

```powershell
make host BACKEND=bitslice
```

Host results are saved in `bin_host/host_report.txt` only when the command
succeeds; on failure, read the terminal output. PC timing uses `clock()`
and describes the PC's performance; use the board's UART report to measure
Apollo4 latency.

Every preparation, including `make encode`, clears both previous host and
board build directories.
Save a host report elsewhere if you need it after the next build, and rebuild
board firmware after running `make host`.

## 5. Connect the board and open a serial monitor

1. Connect the board's **J-Link USB connector J6** with a data cable.
2. Power the board using **SW4**.
3. Find the serial port assigned to the board. The validated Windows setup used
   **COM4**; the port number can differ on another PC.
4. Open your chosen serial monitor with the settings below.
5. Start receiving **before loading the firmware**, so its initial messages
   are captured.

The firmware uses the BSP's UART2 with these settings:

| Setting | Value |
| --- | --- |
| Baud rate | **115200** |
| Data bits | **8** |
| Parity | **None** |
| Stop bits | **1** |
| Flow control | **None** |

### Tested reader: Microsoft Serial Monitor for VS Code

The VS Code extension used to validate this project is
**[Serial Monitor by Microsoft](https://marketplace.visualstudio.com/items?itemName=ms-vscode.vscode-serial-monitor)**,
extension ID **`ms-vscode.vscode-serial-monitor`**.

1. Install the extension from the VS Code Extensions view.
2. Open the bottom panel, for example with **Terminal > New Terminal**, and
   select the **Serial Monitor** tab.
3. Open a monitor if none is present.
4. Set **Monitor Mode** to **Serial** and **View Mode** to **Text**.
5. Select the board's **Port**, for example **COM4**.
6. Set **Baud rate** to **115200** and **Line ending** to **None**. Use the
   serial parameters in the table above for any additional settings.
7. Click **Start Monitoring**, then run the board command from the next step.

The firmware processes embedded inputs automatically, so there is no command
or sample to type into the monitor. Line-ending selection concerns transmitted
text and does not control how the firmware's output is received.

The VS Code Serial Monitor extension was used for this project, but you can use any serial terminal you prefer with the settings above.

## 6. Compile, load and start the firmware

With the serial monitor receiving, run:

```powershell
make BACKEND=scalar
```

`make` without arguments performs the same scalar workflow. For bitslice, use:

```powershell
make BACKEND=bitslice
```

The command performs these actions in order:

1. Validate the model and available input/reference files.
2. Encode inputs and generate the model/deployment descriptors.
3. Compile the selected backend and link the firmware.
4. Load `bin_board/general.axf` using J-Link Commander.
5. Reset and start the processor, then exit J-Link.

The board then runs verification and timing and prints its report over UART.
**The terminal running `make` shows build/loading messages; the serial monitor
shows firmware results.** `make` finishes after programming and startup, while
the board can still be running its benchmark. Wait for **`Run completed`** in
the serial monitor before recording the result or starting another run.

The command's exit status reports preparation, compilation and loading failures.
Firmware verification and timing errors are reported separately over UART.
J-Link diagnostics are saved in `bin_board/jlink_flash.log`.

To compile first and load later, use:

```powershell
make board BACKEND=bitslice
# Open/start the serial monitor before the next command.
make flash
```

`make flash` loads the existing firmware without rebuilding. Its backend is
the one selected when that firmware was built. Rebuild after changing the
model, inputs, source code, backend or repeat count.

## 7. Read and save the report

A successful run with ten verification samples and ten measurement samples,
using the default ten repetitions, has this structure. The ellipses stand for
values that depend on the model, backend and build:

```text
Apollo4 BL EVB implementations
Backend: scalar
Verifying predictions...
Verification mismatches: 0 / 10
Running inference benchmark...
CPU: 96000000 Hz; runs: 100; cycles min=... mean=... max=...
Latency: ... ms/inference

Flash/MRAM: ... bytes (complete firmware)
Model blob: ... bytes; encoded samples/predictions: ... bytes
Code + model + descriptors: ... bytes (excluding samples)
Static RAM: ... bytes; workspace: ... bytes
Stack reserved: ...; observed: ...; margin: ... bytes
SRAM reserved: ... bytes
SRAM observed peak estimate: ... bytes
Run completed
```

| Report field | Meaning |
| --- | --- |
| `Verification mismatches: 0 / 10` | All ten selected predictions match the reference model; this checks agreement on these inputs, rather than dataset accuracy |
| `CPU` | CPU frequency used to convert cycles to time |
| `runs` | Measurement sample count multiplied by `REPEATS` |
| `cycles min / mean / max` | Cycle statistics across timed inferences |
| `Latency` | Mean inference time on the board, in milliseconds |
| `Flash/MRAM` | Complete linked application firmware, including model, descriptors and embedded samples; excludes the bootloader |
| `Model blob` | Size of the supplied `model.bin` |
| `encoded samples/predictions` | Embedded measurement/test inputs and reference class indices |
| `Code + model + descriptors` | Firmware size after subtracting embedded samples and reference predictions |
| `Static RAM` | Static RAM allocations retained by the linker, including inference buffers when used |
| `workspace` | Required inference working-buffer size; its allocation is included in static RAM when used |
| `Stack reserved / observed / margin` | Allocated stack, observed use and remaining margin |
| `SRAM reserved` | Static RAM plus the entire reserved stack |
| `SRAM observed peak estimate` | Static RAM plus observed stack use; an estimate for this run |

Timing follows an untimed warm-up and covers inference on encoded inputs.
Model initialization, PC input encoding, reference checks and UART reporting
are outside the timed interval. Stack use is observed before the memory report;
the SRAM estimate is not a guaranteed worst case for every model/input.

Missing or empty `input_test/inputs.bin`, or fewer than ten complete test
samples, produces **`0 / 0`**, meaning verification was skipped. With at least
ten test samples, matching `ref_preds.bin` is required; missing or invalid
references fail preparation. Incomplete raw samples also fail preparation.
Missing or empty measurement inputs produce **`Latency: N/A`** and skip the
benchmark. A firmware with no supplied inputs can initialize and print a
report, but performs no inference.

With no inference inputs, the linker can remove unused working buffers. The
`workspace` line still reports the size required to execute this model.

Verification mismatches skip the board benchmark. Invalid timing with supplied
measurement inputs or an invalid stack report produces
**`Run completed with errors`**. Rejected benchmark measurements first print
**`BENCHMARK ERROR:`** with the specific reason and the number of completed
timed inferences. Reasons include an unavailable or stopped cycle counter,
STIMER not running, an ambiguous counter wrap, a changed CPU clock or
predictions differing from the untimed warm-up. Such runs report `Latency: N/A`;
do not use them for latency comparisons. A model-operation failure prints
**`ERROR:`** and stops the run. `Run completed` indicates successful completion
of the checks for which data was available.

Save the report using your serial terminal's logging function or by copying its
text to a file. Board UART reports are not automatically saved by `make`;
`jlink_flash.log` contains programming diagnostics.

## 8. Repeat a run or compare backends

To repeat the same firmware, keep the monitor open and press the board's
**RESET** button. This repeats the embedded tests and benchmark without a
rebuild or upload. The firmware runs once per reset and then waits for the
next reset.

To compare scalar and bitslice:

1. Keep the same model, measurement inputs and `REPEATS` value.
2. Run `make BACKEND=scalar`, wait for completion and save the UART report.
3. Run `make BACKEND=bitslice`, wait for completion and save that report.
4. Compare board latency, reference mismatches and memory usage.

Use the same ten test samples if you also want identical verification sets:
preparation reselects ten indices when more than ten raw test samples are
available. Measurement inputs always retain their supplied order.

For example, with ten measurement inputs:

```powershell
make BACKEND=bitslice REPEATS=20
```

This performs 200 timed inferences. Backend switching needs no cleanup.

## Command and configuration reference

Run these commands from the project root:

| Command | Action |
| --- | --- |
| `make` / `make run` | Prepare, compile, load and start firmware; scalar by default |
| `make BACKEND=bitslice` | The complete workflow using bitslice |
| `make host` | Prepare, compile and run on the PC |
| `make board` | Prepare and compile firmware without loading it |
| `make flash` | Load and start the existing `bin_board/general.axf` |
| `make encode` | Validate the model and supplied inputs/references, then encode inputs without regenerating deployment files or building an application executable |
| `make prepare` | Encode inputs and generate deployment/model files without building an application executable |
| `make clean` | Remove supplied local model/input data and generated outputs; recreate `.gitkeep` files |

Preparation, including `encode`, keeps supplied model/input files but clears
both `bin_host/` and `bin_board/`.
`flash` performs no preparation and preserves the existing build outputs.

| Variable | Default | Purpose |
| --- | --- | --- |
| `BACKEND` | `scalar` | Select `scalar` or `bitslice` when preparing/building |
| `REPEATS` | `10` | Positive integer: timed repetitions per measurement sample; total timed runs must fit `uint32` |
| `MODEL_DIR` | `model` | Directory containing `model.json` and `model.bin` |
| `SDK_ROOT` | `AMBIQ_SDK_ROOT`, otherwise `../../AmbiqSuite_R4.5.0` | SDK installation path |
| `PYTHON` | `python` | Python interpreter |
| `CC` | `gcc` | PC compiler |
| `CFLAGS` | `-O2 -std=c11 -Wall -Wextra -Werror -ffp-contract=off` | PC encoder and host compiler flags; board compiler flags are fixed in `tools/run.py` |
| `ARM_CC` | `arm-none-eabi-gcc` | ARM compiler |
| `JLINK` | `JLink` | J-Link Commander executable |
| `DEVICE` | `AMAP42KL-KBR` | SEGGER device name for the supplied board setup |
| `JLINK_SERIAL` | Empty | Select a J-Link probe; this is not the UART COM port |

Quote paths containing spaces. For example, select an existing model directory:

```powershell
make BACKEND=bitslice MODEL_DIR="../mnist/models/mnist_k64_s0_uni_aug_best"
```

`MODEL_DIR` selects model files only. Inputs and references still come from
this project's `input/` and `input_test/` directories. `make clean` does not
remove a model directory outside this project.

The variant entry points also support host, board, run and flash commands:

```powershell
make -C scalar
make -C bitslice
```

They forward to the root Makefile with their backend selected. Model/input paths
and configuration overrides are interpreted relative to the project root.

### Files generated by preparation and compilation

| Directory/file | Purpose |
| --- | --- |
| `bin_input/` | Encoded inputs, selected verification references and encoding metadata |
| `generated/` | Model/deployment headers, data snapshots and assembler embedding source |
| `bin_host/` | Successful host executable, report and backend metadata |
| `bin_board/general.axf` | Linked ELF firmware used by J-Link loading |
| `bin_board/general.bin` | Raw application image including model/data, without the bootloader; retained as an alternative output |
| `bin_board/general.map` | Linked memory placement |
| `bin_board/general.size` | Section-size report |
| `bin_board/build_info.json` | Backend recorded when compiling firmware |
| `bin_board/jlink_flash.log` | Diagnostics from the latest load attempt |
