"""Prepare a complete model, encode inputs and build/run host or board firmware."""

import argparse
import ctypes
from dataclasses import dataclass
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile

from prepare_model import model_encoding, prepare_operations, render_header, packed_bytes


PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_GROUPS = ("input", "input_test")
TEST_SAMPLE_COUNT = 10
THRESHOLD_LAYOUTS = {"shared": 0, "channel": 1, "row": 2}
DEFAULT_COMPILER_FLAGS = "-O2 -std=c11 -Wall -Wextra -Werror -ffp-contract=off"


@dataclass
class InputJob:
    """One input directory and the information needed to encode its samples."""

    group: str
    config: dict
    input_path: Path
    thresholds_per_group: int
    input_bytes_per_sample: int
    sample_indices: range | list[int]
    expected_predictions: list[int] | None = None


# Read a UTF-8 JSON file and require a top-level object.
def read_json_object(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


# Validate and return the reference prediction storage type.
def reference_prediction_dtype(manifest):
    """Reference-file options also belong to the single model configuration."""
    verification = manifest.get("verification", {})
    if not isinstance(verification, dict) or verification.keys() - {"prediction_dtype"}:
        raise ValueError("model.json: verification only accepts prediction_dtype")
    dtype = verification.get("prediction_dtype", "uint8")
    if dtype not in ("uint8", "uint32"):
        raise ValueError("model.json: verification.prediction_dtype must be uint8 or uint32")
    return dtype


# Check input file size and reject nonfinite float32 samples.
def validate_input_file(input_path, config, sample_size):
    expected_size = sample_size * config["count"]
    if input_path.stat().st_size != expected_size:
        raise ValueError(f"{input_path}: expected exactly {expected_size} bytes")
    if config["input_dtype"] == "float32":
        with input_path.open("rb") as input_file:
            for _ in range(config["count"]):
                sample = input_file.read(sample_size)
                for (value,) in struct.iter_unpack("<f", sample):
                    if not math.isfinite(value):
                        raise ValueError(f"{input_path}: NaN and infinity are not supported")


# Validate available raw input groups and build their encoding jobs.
def collect_input_jobs(project_dir, model_config, classes, prediction_dtype):
    """Use validated model settings and infer sample counts from raw binaries."""
    jobs = []
    channels, height, _ = model_config["shape"]
    groups = {"shared": 1, "channel": channels, "row": height}[model_config["threshold_layout"]]
    threshold_count = len(model_config["thresholds"]) // groups
    sample_size = math.prod(model_config["shape"]) * (4 if model_config["input_dtype"] == "float32" else 1)
    for group in INPUT_GROUPS:
        input_dir = project_dir / group
        if (input_dir / "config.json").exists():
            raise ValueError(f"{input_dir}: remove config.json; model.json is the only configuration")
        input_path = input_dir / "inputs.bin"
        if not input_path.is_file() or input_path.stat().st_size == 0:
            continue
        size = input_path.stat().st_size
        if size % sample_size:
            raise ValueError(f"{input_path}: incomplete sample")
        config = dict(model_config)
        config["count"] = size // sample_size
        if not 1 <= config["count"] <= (1 << 32) - 1:
            raise ValueError(f"{input_path}: sample count must fit a positive uint32")
        if group == "input_test" and config["count"] < TEST_SAMPLE_COUNT:
            continue
        validate_input_file(input_path, config, sample_size)
        predictions = None
        if group == "input_test":
            item_bytes = 1 if prediction_dtype == "uint8" else 4
            path = input_dir / "ref_preds.bin"
            raw = path.read_bytes()
            if len(raw) != config["count"] * item_bytes:
                raise ValueError(f"{path}: one reference prediction is required for every source sample")
            predictions = [v for (v,) in struct.iter_unpack("<B" if item_bytes == 1 else "<I", raw)]
            if any(value >= classes for value in predictions):
                raise ValueError(f"{path}: reference class outside the model's class range")
        jobs.append(InputJob(
            group, config, input_path, threshold_count, sample_size,
            range(config["count"]), predictions,
        ))

    return jobs


# Load the shared C encoder and configure its ctypes signature.
def load_c_encoder(library_path):
    """Describe the C function's argument types so Python can call it."""
    library = ctypes.CDLL(str(library_path))
    encoder = library.encode_input
    float_pointer = ctypes.POINTER(ctypes.c_float)
    encoder.argtypes = [
        ctypes.c_void_p,                         # Input sample
        ctypes.c_int,                            # 1 for float32, 0 for uint8
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,  # C, H, W
        ctypes.c_size_t,                         # Thresholds per group
        ctypes.c_int,                            # Threshold layout
        float_pointer,                          # Thermometer thresholds
        ctypes.c_int,                            # Enable edges
        float_pointer,                          # Edge thresholds
        float_pointer,                          # Edge working buffer
        ctypes.POINTER(ctypes.c_uint8),          # Packed output
    ]
    encoder.restype = None
    return encoder


# Binarize selected samples and save encoded data, metadata and references.
def encode_input_group(encoder, job, output_dir):
    """Read, encode and write one sample at a time."""
    config = job.config
    channels, height, width = config["shape"]
    float_input = config["input_dtype"] == "float32"
    edges_enabled = config["encoding"] == "thermometer_edge"
    plane_count = channels * job.thresholds_per_group + (10 if edges_enabled else 0)
    bytes_per_plane = (height * width + 7) // 8
    output_bytes_per_sample = plane_count * bytes_per_plane

    threshold_values = config["thresholds"]
    edge_values = config.get("edge_thresholds", [])
    thresholds = (ctypes.c_float * len(threshold_values))(*threshold_values)
    edge_thresholds = (ctypes.c_float * len(edge_values))(*edge_values)
    scratch = (ctypes.c_float * (height * width))() if edges_enabled else None
    output_buffer = (ctypes.c_uint8 * output_bytes_per_sample)()
    input_type = ctypes.c_float if float_input else ctypes.c_uint8
    sample_type = input_type * (channels * height * width)

    output_dir.mkdir()
    output_path = output_dir / "encoded.bin"
    with job.input_path.open("rb") as input_file, output_path.open("wb") as output_file:
        for sample_index in job.sample_indices:
            input_file.seek(sample_index * job.input_bytes_per_sample)
            input_sample = sample_type.from_buffer_copy(input_file.read(job.input_bytes_per_sample))
            encoder(
                input_sample, float_input, channels, height, width,
                job.thresholds_per_group, THRESHOLD_LAYOUTS[config["threshold_layout"]],
                thresholds, edges_enabled, edge_thresholds, scratch, output_buffer,
            )
            output_file.write(bytes(output_buffer))

    metadata = {
        "count": len(job.sample_indices),
        "shape": [plane_count, height, width],
        "encoding": config["encoding"],
        "bytes_per_plane": bytes_per_plane,
        "bytes_per_sample": output_bytes_per_sample,
        "bit_order": "lsb_first",
        "plane_order": "channel_then_threshold_then_edges",
        "position_order": "row_major",
        "padding": "zero_to_next_byte_per_plane",
    }
    if job.group == "input_test":
        # Output sample i corresponds to source sample selected_indices[i].
        metadata["source_count"] = config["count"]
        metadata["selected_indices"] = job.sample_indices
        if job.expected_predictions is not None:
            # Keep expected predictions in exactly the order of selected inputs.
            selected = [job.expected_predictions[i] for i in job.sample_indices]
            (output_dir / "ref_preds.bin").write_bytes(struct.pack("<" + "I" * len(selected), *selected))
    metadata_path = output_dir / "encoded_meta.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    sample_count = len(job.sample_indices)
    total_bytes = sample_count * output_bytes_per_sample
    print(f"{job.group}: encoded {sample_count} samples ({total_bytes} bytes)")


# Replace bin_input with prepared outputs, or remove stale encoded inputs.
def replace_previous_output(prepared_output, project_dir=PROJECT_DIR):
    """Clear stale files; copy new output if any inputs were encoded."""
    output_dir = project_dir / "bin_input"
    if output_dir.resolve() != project_dir / "bin_input":
        raise ValueError("bin_input must stay inside the project directory")
    if output_dir.is_symlink() or output_dir.is_junction():
        raise ValueError("bin_input must be an ordinary directory")
    if output_dir.exists():
        for parent, directories, files in os.walk(output_dir, followlinks=False):
            for name in directories + files:
                child = Path(parent) / name
                if child.is_symlink() or child.is_junction():
                    raise ValueError("bin_input must not contain links")
        shutil.rmtree(output_dir)
    if prepared_output is not None:
        shutil.copytree(prepared_output, output_dir)


# Encode all input groups, selecting ten distinct verification samples.
def encode_all_groups(library_path, jobs, project_dir=PROJECT_DIR):
    encoder = load_c_encoder(library_path)
    with tempfile.TemporaryDirectory(prefix="encoded-input-") as temporary_dir:
        prepared_output = Path(temporary_dir)
        for job in jobs:
            if job.group == "input_test":
                # Sample positions, not values: no source position is repeated.
                job.sample_indices = random.SystemRandom().sample(
                    job.sample_indices, TEST_SAMPLE_COUNT
                )
            encode_input_group(encoder, job, prepared_output / job.group)
        replace_previous_output(prepared_output, project_dir)


# Compile encode.c into a shared library callable from Python.
def compile_encoder(compiler, compiler_flags, library_path):
    command = [
        compiler, *shlex.split(compiler_flags), "-shared", "-fPIC",
        str(PROJECT_DIR / "src/encode.c"), "-o", str(library_path),
    ]
    subprocess.run(command, check=True)


# Build a temporary encoder and run encoding in a child process.
def run_with_temporary_library(compiler, compiler_flags, jobs, project_dir=PROJECT_DIR):
    with tempfile.TemporaryDirectory(prefix="input-encoder-") as temporary_dir:
        library_name = "encoder.dll" if os.name == "nt" else "encoder.so"
        library_path = Path(temporary_dir) / library_name
        compile_encoder(compiler, compiler_flags, library_path)

        # Pass validated jobs to a child process. Its exit releases the DLL
        # on Windows, allowing cleanup without re-reading the configuration.
        worker = multiprocessing.get_context("spawn").Process(
            target=execute_safely, args=(encode_all_groups, library_path, jobs, project_dir)
        )
        worker.start()
        worker.join()
        if worker.exitcode != 0:
            raise RuntimeError("Input encoding failed")


# Validate a local project directory and reject links in its path or contents.
def checked_directory(project_dir, name):
    """Check the boundary and reject links before removing directory contents."""
    if name not in ("generated", "bin_host", "bin_board", "bin_input", "model", *INPUT_GROUPS):
        raise ValueError("not a project data or build directory")
    target = project_dir / name
    if target.resolve() != target or target.is_symlink() or target.is_junction():
        raise ValueError(f"{name} must be an ordinary directory inside the project")
    if target.exists():
        if not target.is_dir():
            raise ValueError(f"{name} must be a directory")
        for parent, directories, files in os.walk(target, followlinks=False):
            for child in directories + files:
                path = Path(parent) / child
                if path.is_symlink() or path.is_junction():
                    raise ValueError(f"{name} must not contain links")
    return target


# Remove a validated generated directory, if it exists.
def clear_directory(project_dir, name):
    """Delete only a checked, generated directory inside the selected project."""
    if name not in ("generated", "bin_host", "bin_board"):
        raise ValueError("not a generated build directory")
    target = checked_directory(project_dir, name)
    if target.exists():
        shutil.rmtree(target)


# Empty supplied model/input folders and remove all generated outputs.
def clean_project(project_dir):
    """Reset local supplied data and generated outputs, preserving source files."""
    supplied = ("model", *INPUT_GROUPS)
    names = (*supplied, "bin_input", "generated", "bin_host", "bin_board")
    # Check every directory before deleting anything. External MODEL_DIR paths
    # are never cleaned, and links cannot redirect deletion outside the project.
    targets = [(name, checked_directory(project_dir, name)) for name in names]
    for name, target in targets:
        if target.exists():
            shutil.rmtree(target)
        if name in supplied:
            target.mkdir()
    print("Emptied model/, input/ and input_test/; removed generated build outputs.")


# Publish staged files by replacing a validated generated directory.
def replace_directory(source, project_dir, name):
    """Replace a generated directory only after checking its resolved boundary."""
    clear_directory(project_dir, name)
    target = project_dir / name
    shutil.copytree(source, target)


# Format a path for assembler inclusion and reject quotes or newlines.
def assembler_path(path):
    value = path.as_posix()
    if any(character in value for character in ('"', '\n', '\r')):
        raise ValueError("assembler input paths must not contain quotes or newlines")
    return value


# Prepare model/input snapshots, C headers and assembler embedding directives.
def prepare_deployment(project_dir, manifest, blob, header, operations, repeats, backend):
    """Snapshot the model and embed only encoded samples, never the encoder."""
    output = project_dir / "generated"
    measure_dir = project_dir / "bin_input/input"
    test_dir = project_dir / "bin_input/input_test"
    # Load an encoded group's metadata and data, or return an empty group.
    def group(directory):
        metadata_path = directory / "encoded_meta.json"
        if not metadata_path.is_file():
            return {"count": 0, "selected_indices": []}, b""
        metadata = read_json_object(metadata_path)
        data = (directory / "encoded.bin").read_bytes()
        return metadata, data
    measure_meta, measure_data = group(measure_dir)
    test_meta, test_data = group(test_dir)
    measure_count, test_count = measure_meta["count"], test_meta["count"]
    if measure_count * repeats > (1 << 32) - 1:
        raise ValueError("benchmark run count exceeds uint32")
    if test_count not in (0, TEST_SAMPLE_COUNT):
        raise ValueError("verification must contain exactly ten inputs")
    expected = (test_dir / "ref_preds.bin").read_bytes() if test_count else b""
    if len(expected) != test_count * 4:
        raise ValueError("selected reference predictions are missing or incomplete")
    sample_bytes = packed_bytes(operations[0]["input_shape"])
    for meta, data in ((measure_meta, measure_data), (test_meta, test_data)):
        if len(data) != meta["count"] * sample_bytes:
            raise ValueError("encoded samples disagree with model input geometry")
    declarations = ["/* Generated deployment data; all samples are already encoded. */",
                    "#ifndef GENERAL_DEPLOY_DATA_H", "#define GENERAL_DEPLOY_DATA_H", "#include <stdint.h>",
                    f"#define MEASURE_COUNT {measure_count}u", f"#define TEST_COUNT {test_count}u",
                    f"#define BENCH_REPEATS {repeats}u", f'#define INFERENCE_BACKEND_NAME "{backend}"',
                    "extern const uint8_t model_blob[];",
                    "extern const uint8_t measure_inputs[];", "extern const uint8_t test_inputs[];",
                    "extern const uint32_t test_expected[];"]
    if test_count:
        indices = ", ".join(f"{index}u" for index in test_meta["selected_indices"])
        declarations.append(f"static const uint32_t test_source_indices[TEST_COUNT] = {{ {indices} }};")
    declarations.extend(["#endif", ""])
    assembly = ["/* Generated data embedding for GNU assembler, host and ARM. */",
                "#ifdef _WIN32", '.section .rdata,"dr"', "#else", '.section .rodata.deploy,"a"', "#endif"]
    files = {"model_blob": ("model.bin", blob), "measure_inputs": ("measure.bin", measure_data),
             "test_inputs": ("test.bin", test_data), "test_expected": ("expected.bin", expected)}
    with tempfile.TemporaryDirectory(prefix="model-deployment-") as temporary:
        staging = Path(temporary)
        (staging / "model_specs.h").write_text(header, encoding="utf-8")
        (staging / "deploy_data.h").write_text("\n".join(declarations), encoding="utf-8")
        for symbol, (name, data) in files.items():
            (staging / name).write_bytes(data)
            assembly.extend([".balign 4", f".global {symbol}", f"{symbol}:",
                             f'.incbin "{assembler_path(output / name)}"',
                             f".if (. - {symbol}) != {len(data)}", '.error "Deployment file size mismatch"', ".endif"])
        assembly.extend(["#ifndef _WIN32", '.section .note.GNU-stack,""', "#endif"])
        (staging / "deploy_data.S").write_text("\n".join(assembly) + "\n", encoding="utf-8")
        (staging / "model.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        replace_directory(staging, project_dir, "generated")
    print(f"Prepared {backend} deployment: {len(operations)} operations, {measure_count} measurement inputs, {test_count} checks")


# Return compiler options and source files for the selected backend.
def backend_options(backend):
    """Select kernels at compile time; scalar firmware contains no bitslice code."""
    if backend == "scalar":
        return [], []
    return ["-DMODEL_USE_BITSLICE=1", "-I", str(PROJECT_DIR / "bitslice")], [PROJECT_DIR / "bitslice/bitslice.c"]


# Build and run the host executable, saving outputs only on success.
def build_host(project_dir, compiler, flags, backend):
    generated = project_dir / "generated"
    with tempfile.TemporaryDirectory(prefix="general-host-") as temporary:
        staging = Path(temporary)
        executable = staging / ("general_host.exe" if os.name == "nt" else "general_host")
        sources = [PROJECT_DIR / "src" / name for name in ("main_host.c", "model.c", "lut.c", "tensor.c")]
        options, extra_sources = backend_options(backend)
        sources += extra_sources
        command = [compiler, *shlex.split(flags), *options, "-I", str(PROJECT_DIR / "src"), "-I", str(generated),
                   *(str(source) for source in sources), str(generated / "deploy_data.S"), "-o", str(executable)]
        subprocess.run(command, check=True)
        # Run the exact binary that will be delivered. A failing prediction
        # prevents publication; old runnable artifacts were invalidated first.
        result = subprocess.run([str(executable)], text=True, capture_output=True)
        print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="")
        result.check_returncode()
        (staging / "host_report.txt").write_text(result.stdout, encoding="utf-8")
        (staging / "build_info.json").write_text(json.dumps({"backend": backend}, indent=2) + "\n", encoding="utf-8")
        replace_directory(staging, project_dir, "bin_host")


# Compile Apollo4 firmware and generate its binary and memory reports.
def build_board(project_dir, sdk_root, compiler, backend):
    sdk_root = sdk_root.resolve()
    bsp = sdk_root / "boards/apollo4l_blue_evb/bsp"
    libraries = [sdk_root / "mcu/apollo4l/hal/mcu/gcc/bin/libam_hal.a", bsp / "gcc/bin/libam_bsp.a"]
    if not all(path.is_file() for path in libraries):
        raise ValueError("Ambiq SDK HAL/BSP libraries are missing; build them or select --sdk-root")
    arch = ["-mthumb", "-mcpu=cortex-m4", "-mfpu=fpv4-sp-d16", "-mfloat-abi=hard"]
    includes = [PROJECT_DIR / "src", project_dir / "generated", sdk_root,
                sdk_root / "CMSIS/ARM/Include", sdk_root / "CMSIS/AmbiqMicro/Include",
                sdk_root / "devices", sdk_root / "mcu/apollo4l", sdk_root / "mcu/apollo4l/hal",
                sdk_root / "utils", bsp]
    flags = arch + ["-std=c11", "-O2", "-g3", "-Wall", "-Wextra", "-Wno-unused-parameter",
                    "-ffp-contract=off", "-ffunction-sections", "-fdata-sections", "-fstack-usage",
                    "-DPART_apollo4l", "-DAM_PACKAGE_BGA", "-DAM_PART_APOLLO4L", "-Dgcc"]
    flags += [argument for path in includes for argument in ("-I", str(path))]
    sources = [PROJECT_DIR / "board" / name for name in ("main.c", "startup_gcc.c")]
    sources += [PROJECT_DIR / "src" / name for name in ("model.c", "lut.c", "tensor.c")]
    options, extra_sources = backend_options(backend)
    flags += options
    sources += extra_sources
    sources += [sdk_root / "utils" / (name + ".c") for name in ("am_util_delay", "am_util_stdio", "am_util_string")]
    sources += [project_dir / "generated/deploy_data.S"]
    prefix = compiler[:-3] if compiler.endswith("gcc") else compiler[:-7] if compiler.endswith("gcc.exe") else None
    if prefix is None:
        raise ValueError("--arm-cc must name an arm-none-eabi-gcc executable")
    with tempfile.TemporaryDirectory(prefix="general-board-") as temporary:
        staging = Path(temporary)
        objects = []
        for i, source in enumerate(sources):
            obj = staging / f"{i}_{source.stem}.o"
            subprocess.run([compiler, *flags, "-c", str(source), "-o", str(obj)], check=True)
            objects.append(obj)
        target = staging / "general.axf"
        subprocess.run([compiler, *arch, "-nostartfiles", "-static", "-Wl,--gc-sections,--entry,Reset_Handler",
                        "-Wl,-T," + str(PROJECT_DIR / "board/linker_script.ld"),
                        "-Wl,-Map," + str(staging / "general.map"), *(str(obj) for obj in objects),
                        "-Wl,--start-group", *(str(lib) for lib in libraries), "-lm", "-lc", "-lgcc", "-lnosys",
                        "-Wl,--end-group", "-o", str(target)], check=True)
        subprocess.run([prefix + "objcopy", "-O", "binary", str(target), str(staging / "general.bin")], check=True)
        sizes = subprocess.run([prefix + "size", "-A", str(target)], check=True, capture_output=True, text=True)
        (staging / "general.size").write_text(sizes.stdout, encoding="utf-8")
        subprocess.run([prefix + "size", str(target)], check=True)
        # Only useful firmware artifacts are retained; temporary objects and
        # compiler stack-usage files are removed automatically.
        for obj in objects:
            obj.unlink()
        for path in staging.glob("*.su"):
            path.unlink()
        (staging / "build_info.json").write_text(json.dumps({"backend": backend}, indent=2) + "\n", encoding="utf-8")
        replace_directory(staging, project_dir, "bin_board")
    print(f"Board firmware: {project_dir / 'bin_board/general.axf'}")


# Parse command-line options and run the requested preparation/build target.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("host", "board", "prepare", "encode", "clean"), default="host")
    parser.add_argument("--backend", choices=("scalar", "bitslice"), default="scalar", help="Inference kernels; model files are identical")
    parser.add_argument("--project-dir", type=Path, default=PROJECT_DIR, help="Directory containing model/ and input/")
    parser.add_argument("--model-dir", type=Path, help="Defaults to model/ in the project directory")
    parser.add_argument("--cc", default="gcc", help="PC compiler executable")
    parser.add_argument("--cflags", default=DEFAULT_COMPILER_FLAGS, help="PC compiler flags")
    parser.add_argument("--arm-cc", default="arm-none-eabi-gcc")
    parser.add_argument("--sdk-root", type=Path, default=Path(os.environ.get("AMBIQ_SDK_ROOT", PROJECT_DIR.parent.parent / "AmbiqSuite_R4.5.0")))
    parser.add_argument("--repeats", type=int, default=10, help="Inference repetitions per measurement sample")
    arguments = parser.parse_args()
    if sys.byteorder != "little":
        raise ValueError("host must be little endian")
    project_dir = arguments.project_dir.resolve()
    if not project_dir.is_dir() or not 1 <= arguments.repeats <= (1 << 32) - 1:
        raise ValueError("project directory must exist and repeats must be a positive uint32")
    if arguments.target == "clean":
        clean_project(project_dir)
        return
    # All preparation targets use the same model and randomly selected test set.
    # Remove old runnable artifacts before any new preparation can fail: the
    # debugger must never silently load yesterday's firmware after a failed make.
    clear_directory(project_dir, "bin_host")
    clear_directory(project_dir, "bin_board")
    model_dir = arguments.model_dir.resolve() if arguments.model_dir else project_dir / "model"
    manifest = read_json_object(model_dir / "model.json")
    blob = (model_dir / "model.bin").read_bytes()
    input_shape, specs, operations = prepare_operations(manifest, blob)
    config = model_encoding(manifest, blob)
    header = render_header(input_shape, specs, len(blob), operations)
    prediction_dtype = reference_prediction_dtype(manifest)
    jobs = collect_input_jobs(project_dir, config, operations[-1]["classes"], prediction_dtype)
    if jobs:
        run_with_temporary_library(arguments.cc, arguments.cflags, jobs, project_dir)
    else:
        replace_previous_output(None, project_dir)
    if arguments.target == "encode":
        return
    prepare_deployment(project_dir, manifest, blob, header, operations, arguments.repeats, arguments.backend)
    if arguments.target == "host":
        build_host(project_dir, arguments.cc, arguments.cflags, arguments.backend)
    elif arguments.target == "board":
        build_board(project_dir, arguments.sdk_root, arguments.arm_cc, arguments.backend)


# Run an action and report supported failures with a nonzero exit status.
def execute_safely(action, *arguments):
    """Use the same concise error handling in the main and child processes."""
    try:
        action(*arguments)
    except (ValueError, OSError, KeyError, TypeError, OverflowError,
            RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    execute_safely(main)
