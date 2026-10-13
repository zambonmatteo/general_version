"""Validate model JSON and binary sections, then generate C descriptors.

The PC resolves either an explicit 'operations' sequence or 'conv'/'fc' records
into LUT, pooling, flatten and GroupSum operations. The firmware uses constant
descriptors and reads model data directly from the immutable binary blob.
"""

import argparse
import json
import math
from pathlib import Path
import struct


UINT32_MAX = (1 << 32) - 1
DTYPE_BYTES = {"uint8": 1, "uint16": 2, "uint32": 4, "float32": 4}


# Require an integer within the specified inclusive range.
def integer(value, name, minimum=1):
    if type(value) is not int or not minimum <= value <= UINT32_MAX:
        raise ValueError(f"{name} must be an integer in [{minimum}, {UINT32_MAX}]")
    return value


# Normalize a square size or [height, width] pair into validated dimensions.
def dimensions(value, name):
    """Expand an integer to [size, size] or validate a [height, width] pair."""
    if type(value) is int:
        value = [value, value]
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be a size or [height, width]")
    return tuple(integer(item, name) for item in value)


# Build a CHW shape and check its total bit count fits uint32.
def shape(channels, hw):
    result = (integer(channels, "channels"), *hw)
    if result[0] * result[1] * result[2] > UINT32_MAX:
        raise ValueError("tensor has too many bits for uint32 indices")
    return result


# Validate and index binary sections, checking sizes, ranges and overlaps.
def read_sections(manifest, blob):
    if integer(manifest["total_bytes"], "total_bytes") != len(blob):
        raise ValueError("model.bin size does not match total_bytes")
    sections = {}
    for entry in manifest["layers"]:
        name = entry["name"]
        if not isinstance(name, str) or name in sections:
            raise ValueError("section names must be unique strings")
        offset = integer(entry["offset"], f"{name}.offset", 0)
        length = integer(entry["bytes"], f"{name}.bytes")
        dtype = entry["dtype"]
        if dtype not in DTYPE_BYTES:
            raise ValueError(f"unsupported section dtype: {dtype}")
        if not isinstance(entry["shape"], list) or not entry["shape"]:
            raise ValueError(f"{name}: missing section shape")
        expected = DTYPE_BYTES[dtype]
        for dimension in entry["shape"]:
            expected *= integer(dimension, f"{name}.shape")
        if expected != length or offset + length > len(blob):
            raise ValueError(f"{name}: invalid section size or range")
        sections[name] = entry
    ordered = sorted(sections.values(), key=lambda entry: entry["offset"])
    for previous, current in zip(ordered, ordered[1:]):
        if previous["offset"] + previous["bytes"] > current["offset"]:
            raise ValueError("blob sections overlap")
    return sections


# Find the unique LUT table/index pair matching supported section prefixes.
def lut_sections(sections, prefixes):
    """Resolve one prefix with '_tab' and '_idx' binary sections.

    A '_pw' suffix is part of the section label. Kernel dimensions, padding
    and stride are read from the JSON layer fields.
    """
    matches = [prefix for prefix in prefixes if prefix + "_tab" in sections]
    if len(matches) != 1:
        raise ValueError(f"expected one LUT table section for {prefixes}")
    prefix = matches[0]
    return prefix, sections[prefix + "_tab"], sections[prefix + "_idx"]


# Validate LUT tables, routing and geometry, then build a layer descriptor.
def layer_spec(sections, prefixes, layer, input_shape, output_shape,
               convolution, blob, section_names=None):
    if section_names is None:
        prefix, tables, indices = lut_sections(sections, prefixes)
    else:
        prefix = layer.get("name", section_names[0])
        tables, indices = (sections[name] for name in section_names)
    n_inputs = integer(layer["n"], f"{prefix}.n")
    if n_inputs > 31:
        raise ValueError(f"{prefix}: at most 31 LUT inputs are supported")
    n_luts = output_shape[0] if convolution else output_shape[2]
    if tables["dtype"] != "uint8" or indices["dtype"] not in ("uint16", "uint32"):
        raise ValueError(f"{prefix}: expected packed uint8 tables and uint16/uint32 indices")
    index_bytes = DTYPE_BYTES[indices["dtype"]]
    row_bytes = ((1 << n_inputs) + 7) // 8
    if tables["shape"] != [n_luts, row_bytes]:
        raise ValueError(f"{prefix}: table shape disagrees with LUT dimensions")
    if indices["bytes"] != n_luts * n_inputs * index_bytes:
        raise ValueError(f"{prefix}: mapper size disagrees with LUT dimensions")

    if convolution:
        kh, kw, ph, pw, sh, sw = convolution
        if input_shape[1] + 2 * ph < kh or input_shape[2] + 2 * pw < kw:
            raise ValueError(f"{prefix}: kernel exceeds padded input")
        expected_hw = ((input_shape[1] + 2 * ph - kh) // sh + 1,
                       (input_shape[2] + 2 * pw - kw) // sw + 1)
        if output_shape[1:] != expected_hw:
            raise ValueError(f"{prefix}: inconsistent convolution output shape")
        index_limit = input_shape[0] * kh * kw
        if integer(layer.get("p_full", index_limit), f"{prefix}.p_full") != index_limit:
            raise ValueError(f"{prefix}: p_full disagrees with kernel geometry")
    else:
        index_limit = input_shape[2]
    if index_limit > UINT32_MAX:
        raise ValueError(f"{prefix}: mapper range exceeds uint32")

    # Decode little-endian values explicitly, even when the host is big-endian.
    start = indices["offset"]
    data = memoryview(blob)[start:start + indices["bytes"]]
    index_format = "<H" if index_bytes == 2 else "<I"
    if any(value >= index_limit for (value,) in struct.iter_unpack(index_format, data)):
        raise ValueError(f"{prefix}: mapper index outside its input range")
    return {
        "name": prefix,
        "table_name": tables["name"],
        "indices_name": indices["name"],
        "type": "LUT_CONVOLUTION" if convolution else "LUT_FULLY_CONNECTED",
        "n_luts": n_luts,
        "n_inputs": n_inputs,
        "index_bytes": index_bytes,
        "n_tables": n_luts,
        "table_id_bytes": 0,
        "tables": (tables["offset"], tables["bytes"]),
        "indices": (indices["offset"], indices["bytes"]),
        "table_ids": (0, 0),
        "input_shape": input_shape,
        "output_shape": output_shape,
        "convolution": convolution or (0, 0, 0, 0, 0, 0),
    }


# Build LUT descriptors from JSON 'conv' and 'fc' records.
def prepare_specs(manifest, blob):
    sections = read_sections(manifest, blob)
    encoding = manifest["encoding"]
    input_shape = shape(encoding["c_in"], dimensions(encoding["in_hw"], "encoding.in_hw"))
    geometry = manifest.get("geometry", {})
    specs = []
    for b, block in enumerate(manifest["conv"]):
        # Each 'conv' block must contain exactly one learnable LUT layer.
        if len(block) != 1 or block[0]["kind"] != "learnable":
            raise ValueError("expected one learnable LUT convolution per block")
        layer = block[0]
        in_hw = layer["in_hw"] if "in_hw" in layer else geometry["in_hw"][b]
        out_hw = layer["out_hw"] if "out_hw" in layer else geometry["out_hw"][b]
        inputs = shape(layer["c_in"], dimensions(in_hw, "conv.in_hw"))
        outputs = shape(layer["k_out"], dimensions(out_hw, "conv.out_hw"))
        kh, kw = dimensions(layer["ks"], "conv.ks")
        pad = layer["pad"]
        ph, pw = (pad, pad) if type(pad) is int else pad
        ph = integer(ph, "conv.padding_h", 0)
        pw = integer(pw, "conv.padding_w", 0)
        # Resolve an omitted stride to [1, 1] in the generated descriptor.
        sh, sw = dimensions(layer.get("stride", 1), "conv.stride")
        specs.append(layer_spec(sections, [f"block{b + 1}", f"block{b + 1}_pw"],
                                layer, inputs, outputs, (kh, kw, ph, pw, sh, sw), blob))
    for f, layer in enumerate(manifest["fc"]):
        inputs = shape(1, (1, integer(layer["in"], "fc.in")))
        outputs = shape(1, (1, integer(layer["out"], "fc.out")))
        specs.append(layer_spec(sections, [f"fc{f + 1}"], layer, inputs, outputs, None, blob))
    if not specs or specs[0]["input_shape"] != input_shape:
        raise ValueError("first LUT layer does not match the encoded input")
    consumed = {spec[field] for spec in specs for field in ("table_name", "indices_name")}
    exported = {name for name in sections if name.endswith(("_tab", "_idx"))}
    if consumed != exported:
        raise ValueError("unmapped LUT sections in the model export")
    # prepare_operations inserts pooling and flatten descriptors, then checks
    # that each operation's input shape matches the preceding output shape.
    return input_shape, specs


# Validate an explicit [channels, height, width] tensor shape.
def tensor_shape(value, name):
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must contain [channels, height, width]")
    return shape(value[0], tuple(integer(v, name) for v in value[1:]))


# Compute packed CHW storage with byte padding for each channel.
def packed_bytes(value):
    channels, height, width = value
    return channels * ((height * width + 7) // 8)


# Validate OR-pooling geometry and create its operation descriptor.
def pool_operation(inputs, kernel, stride):
    kh, kw = dimensions(kernel, "pool.kernel")
    sh, sw = dimensions(stride, "pool.stride")
    c, h, w = inputs
    if h < kh or w < kw:
        raise ValueError("pool kernel exceeds input")
    outputs = shape(c, ((h - kh) // sh + 1, (w - kw) // sw + 1))
    return {"type": "OP_OR_POOL", "input_shape": inputs, "output_shape": outputs,
            "pool": (kh, kw, sh, sw)}


# Create an operation that flattens a CHW tensor into a bit vector.
def flatten_operation(inputs):
    return {"type": "OP_FLATTEN", "input_shape": inputs,
            "output_shape": (1, 1, math.prod(inputs))}


# Validate class grouping and float32 tau, then create the GroupSum head.
def head_operation(inputs, classes, tau):
    classes = integer(classes, "groupsum.classes")
    if inputs[:2] != (1, 1) or inputs[2] % classes:
        raise ValueError("GroupSum requires equal class groups in a flat vector")
    if type(tau) not in (float, int) or not math.isfinite(tau) or not 0 < tau <= 3.402823466e38:
        raise ValueError("GroupSum tau must be finite, positive and fit float32")
    # Reject a divisor that rounds to zero in its float32 representation.
    if struct.unpack("<f", struct.pack("<f", tau))[0] == 0:
        raise ValueError("GroupSum tau underflows float32")
    return {"type": "OP_GROUP_SUM", "input_shape": inputs,
            "output_shape": (1, 1, classes), "classes": classes, "tau": float(tau)}


# Create an operation that references a LUT layer descriptor by index.
def lut_operation(spec, index):
    return {"type": "OP_LUT", "input_shape": spec["input_shape"],
            "output_shape": spec["output_shape"], "lut": index}


# Build and validate operations from an explicit list or 'conv'/'fc' records.
def prepare_operations(manifest, blob):
    """Normalize a linear network to explicit LUT/pool/flatten/head operations."""
    if "operations" not in manifest:
        inputs, specs = prepare_specs(manifest, blob)
        operations = []
        geometry = manifest.get("geometry", {})
        pool_kernels = geometry.get("pool_k", manifest.get("arch", {}).get("pool"))
        conv_count = len(manifest["conv"])
        if not conv_count or not manifest["fc"]:
            raise ValueError("JSON without operations requires convolution and FC layers; declare operations for other networks")
        if pool_kernels is not None and len(pool_kernels) != conv_count:
            raise ValueError("JSON without operations requires one pool kernel per convolution block")
        for i, spec in enumerate(specs[:conv_count]):
            operations.append(lut_operation(spec, i))
            # Insert OR pooling after each convolution. Use the declared kernel
            # or default to [2, 2]; pooling stride equals the kernel dimensions.
            kernel = pool_kernels[i] if pool_kernels is not None else [2, 2]
            pool = pool_operation(spec["output_shape"], kernel, kernel)
            if "pool_hw" in geometry and tuple(geometry["pool_hw"][i]) != pool["output_shape"][1:]:
                raise ValueError("declared pool geometry disagrees with the pool operation")
            operations.append(pool)
        operations.append(flatten_operation(operations[-1]["output_shape"]))
        for i in range(conv_count, len(specs)):
            operations.append(lut_operation(specs[i], i))
        if manifest["head_type"] != "groupsum":
            raise ValueError("only the GroupSum head is supported")
        operations.append(head_operation(specs[-1]["output_shape"], manifest["head_k"], manifest["head_tau"]))
    else:
        sections = read_sections(manifest, blob)
        encoding = manifest["encoding"]
        inputs = shape(encoding["c_in"], dimensions(encoding["in_hw"], "encoding.in_hw"))
        current = inputs
        specs, operations = [], []
        if not isinstance(manifest["operations"], list) or not manifest["operations"]:
            raise ValueError("operations must be a nonempty list")
        for entry in manifest["operations"]:
            kind = entry["type"]
            if kind in ("lut_conv", "lut_fc"):
                declared_input = tensor_shape(entry["input_shape"], "operation.input_shape")
                outputs = tensor_shape(entry["output_shape"], "operation.output_shape")
                convolution = None
                if kind == "lut_conv":
                    kh, kw = dimensions(entry["kernel"], "operation.kernel")
                    pad = entry["padding"]
                    if not isinstance(pad, list) or len(pad) != 2:
                        raise ValueError("convolution padding must contain [height, width]")
                    ph, pw = (integer(v, "operation.padding", 0) for v in pad)
                    sh, sw = dimensions(entry["stride"], "operation.stride")
                    convolution = (kh, kw, ph, pw, sh, sw)
                elif declared_input[:2] != (1, 1) or outputs[:2] != (1, 1):
                    raise ValueError("FC operations require flat vector shapes")
                spec = layer_spec(sections, [], entry, declared_input, outputs, convolution, blob,
                                  (entry["table"], entry["indices"]))
                operations.append(lut_operation(spec, len(specs)))
                specs.append(spec)
            elif kind == "or_pool":
                operations.append(pool_operation(current, entry["kernel"], entry["stride"]))
            elif kind == "flatten":
                operations.append(flatten_operation(current))
            elif kind == "groupsum":
                operations.append(head_operation(current, entry["classes"], entry["tau"]))
            else:
                raise ValueError(f"unsupported operation: {kind}")
            current = operations[-1]["output_shape"]
    current = inputs
    for i, operation in enumerate(operations):
        if operation["input_shape"] != current:
            raise ValueError(f"operation {i}: input shape does not match previous output")
        if operation["type"] == "OP_GROUP_SUM" and i != len(operations) - 1:
            raise ValueError("GroupSum must be the final operation")
        current = operation["output_shape"]
    if operations[-1]["type"] != "OP_GROUP_SUM":
        raise ValueError("network must end with GroupSum")
    return inputs, specs, operations


# Validate encoding settings and load thermometer/edge thresholds from the blob.
def model_encoding(manifest, blob):
    """Derive raw input settings from the model and thresholds from its blob."""
    sections = read_sections(manifest, blob)
    encoding = manifest["encoding"]
    if "operations" in manifest and "input_dtype" not in encoding:
        raise ValueError("explicit models must declare encoding.input_dtype")
    h, w = dimensions(encoding["in_hw"], "encoding.in_hw")
    c = integer(encoding["in_ch"], "encoding.in_ch")
    bits = integer(encoding["bits"], "encoding.bits")
    layouts = {"uniform": "shared", "distributive": "channel", "freqwise": "row"}
    layout = layouts[encoding["thermo"]]
    if layout == "row" and c != 1:
        raise ValueError("row thresholds require one raw input channel")
    expected_shape = [bits] if layout == "shared" else [c if layout == "channel" else h, bits]
    if encoding["thermo_shape"] != expected_shape:
        raise ValueError("thermo_shape disagrees with threshold layout")
    if type(encoding["edges"]) is not bool or encoding.get("edge_color", False) is not False:
        raise ValueError("only boolean monochrome edge encoding is supported")
    edges = encoding["edges"]
    if encoding["n_edge_ch"] != (10 if edges else 0) or encoding["c_in"] != c * bits + (10 if edges else 0):
        raise ValueError("encoded channels disagree with thermometer/edge settings")
    # JSON without 'operations' defaults an omitted input_dtype to uint8.
    dtype = encoding.get("input_dtype", "uint8")
    if dtype not in ("uint8", "float32"):
        raise ValueError("input_dtype must be uint8 or float32")
    # Read a float32 section with the expected shape and require finite values.
    def floats(name, expected):
        section = sections[name]
        if section["dtype"] != "float32" or section["shape"] != expected:
            raise ValueError(f"{name}: wrong float32 section shape")
        values = list(struct.unpack_from("<" + "f" * math.prod(expected), blob, section["offset"]))
        if any(not math.isfinite(v) for v in values):
            raise ValueError(f"{name}: nonfinite threshold")
        return values
    thresholds = floats("thermo_thr", expected_shape)
    for start in range(0, len(thresholds), bits):
        if thresholds[start:start + bits] != sorted(thresholds[start:start + bits]):
            raise ValueError("thermometer thresholds must be ordered within each group")
    config = {"input_dtype": dtype, "shape": [c, h, w], "encoding": "thermometer_edge" if edges else "thermometer",
              "threshold_layout": layout, "thresholds": thresholds}
    if edges:
        config["edge_thresholds"] = floats("edge_thr", [5, 2])
    return config


# Format unsigned integers as a C aggregate initializer.
def initializer(values):
    return "{ " + ", ".join(f"{value}u" for value in values) + " }"


# Generate model_specs.h with model descriptors, operations and sizes.
def render_header(input_shape, specs, blob_bytes, operations):
    maximum = max([packed_bytes(input_shape)] + [packed_bytes(op["output_shape"])
                  for op in operations if op["type"] != "OP_GROUP_SUM"])
    if 2 * maximum > UINT32_MAX:
        raise ValueError("workspace exceeds a 32-bit target address space")
    lines = [
        "/* Generated by tools/prepare_model.py from model.json; do not edit.",
        " * Shared execution sequence for both backends; model data stays in the blob. */",
        "#ifndef GENERAL_GENERATED_MODEL_H",
        "#define GENERAL_GENERATED_MODEL_H",
        '#include "model.h"',
        "",
        f"#define MODEL_BLOB_BYTES {blob_bytes}u",
        f"#define MODEL_LUT_LAYER_COUNT {len(specs)}u",
        f"#define MODEL_OPERATION_COUNT {len(operations)}u",
        f"#define MODEL_INPUT_BYTES {packed_bytes(input_shape)}u",
        f"#define MODEL_WORKSPACE_BYTES {2 * maximum}u",
        f'#define MODEL_CLASS_COUNT {operations[-1]["classes"]}u',
    ]
    if specs:
        lines.append("static const LutLayerSpec model_lut_specs[MODEL_LUT_LAYER_COUNT] = {")
    for spec in specs:
        # Export names are not emitted as C identifiers or comments.
        lines.extend([
            "    {",
            f'        .type = {spec["type"]},',
            f'        .n_luts = {spec["n_luts"]}u,',
            f'        .n_inputs = {spec["n_inputs"]}u,',
            f'        .index_bytes = {spec["index_bytes"]}u,',
            f'        .n_tables = {spec["n_tables"]}u,',
            f'        .table_id_bytes = {spec["table_id_bytes"]}u,',
            *(f"        .{field} = {initializer(spec[field])}," for field in
              ("tables", "indices", "table_ids", "input_shape", "output_shape", "convolution")),
            "    },",
        ])
    if specs:
        lines.append("};")
    lines.append("static const Operation model_operations[MODEL_OPERATION_COUNT] = {")
    for op in operations:
        lines.extend(["    {", f'        .type = {op["type"]},',
                      f'        .input_shape = {initializer(op["input_shape"])},',
                      f'        .output_shape = {initializer(op["output_shape"])},'])
        if op["type"] == "OP_LUT":
            lines.append(f'        .parameters.lut = &model_lut_specs[{op["lut"]}],')
        elif op["type"] == "OP_OR_POOL":
            lines.append(f'        .parameters.pool = {initializer(op["pool"])},')
        elif op["type"] == "OP_GROUP_SUM":
            lines.append(f'        .parameters.head = {{ {op["classes"]}u, {op["tau"]!r}f }},')
        lines.append("    },")
    lines.extend(["};", "static const ModelSpec model_spec = {",
                  "    .blob_bytes = MODEL_BLOB_BYTES,",
                  f"    .input_shape = {initializer(input_shape)},",
                  "    .operations = model_operations,",
                  "    .operation_count = MODEL_OPERATION_COUNT,",
                  "};", "#endif", ""])
    return "\n".join(lines)


# Parse model files, validate them and write the generated C header.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Path to the exported model.json")
    parser.add_argument("--blob", type=Path, help="Defaults to model.bin beside the JSON")
    parser.add_argument("--output", required=True, type=Path, help="Generated C header")
    args = parser.parse_args()
    blob_path = args.blob if args.blob is not None else args.manifest.with_name("model.bin")
    # Never allow a generated header to replace either source file.
    if args.output.resolve() in (args.manifest.resolve(), blob_path.resolve()):
        parser.error("output must be different from the JSON and blob paths")
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        blob = blob_path.read_bytes()
        input_shape, specs, operations = prepare_operations(manifest, blob)
        model_encoding(manifest, blob)
        header = render_header(input_shape, specs, len(blob), operations)
    except (ValueError, KeyError, TypeError, IndexError, OverflowError, OSError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(header, encoding="utf-8")
    print(f"Prepared {len(specs)} LUT layers and {len(operations)} operations: {args.output}")


if __name__ == "__main__":
    main()
