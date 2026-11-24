"""Validate model and test data, then generate firmware descriptors."""

import argparse
import json
import math
import re
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parent.parent
UINT32_MAX = (1 << 32) - 1
DTYPE_BYTES = {"uint8": 1, "uint16": 2, "uint32": 4, "float32": 4}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, minimum=1):
    check(type(value) is int and minimum <= value <= UINT32_MAX, "invalid uint32 value")
    return value


def pair(value, minimum=1):
    value = [value, value] if type(value) is int else value
    check(isinstance(value, list) and len(value) == 2, "expected a size or [height, width]")
    return [integer(v, minimum) for v in value]


def shape(channels, hw):
    result = [integer(channels), *pair(hw)]
    check(math.prod(result) <= UINT32_MAX, "tensor bit count exceeds uint32")
    return result


def read_sections(manifest, blob):
    check(integer(manifest["total_bytes"]) == len(blob), "model.bin size differs from total_bytes")
    check(isinstance(manifest["layers"], list) and manifest["layers"], "missing binary sections")
    sections = {}
    for entry in manifest["layers"]:
        name, dtype, dims = entry["name"], entry["dtype"], entry["shape"]
        check(isinstance(name, str) and name and "\0" not in name and name not in sections,
              "section names must be nonempty, unique and without NUL")
        check(dtype in DTYPE_BYTES and isinstance(dims, list) and dims, f"{name}: invalid dtype/shape")
        dims = [integer(v) for v in dims]
        offset, size = integer(entry["offset"], 0), integer(entry["bytes"])
        check(size == math.prod(dims) * DTYPE_BYTES[dtype], f"{name}: wrong section byte size")
        check(offset + size <= len(blob), f"{name}: section outside model.bin")
        sections[name] = dict(name=name, offset_bytes=offset, size_bytes=size, dtype=dtype, shape=dims)
    end = 0
    for section in sorted(sections.values(), key=lambda s: s["offset_bytes"]):
        check(section["offset_bytes"] >= end, "binary sections overlap")
        end = section["offset_bytes"] + section["size_bytes"]
    return sections


def prepare_encoding(entry, sections, blob):
    raw = shape(entry["in_ch"], entry["in_hw"])
    encoded = shape(entry["c_in"], entry["in_hw"])
    bits, layout, edges = integer(entry["bits"]), entry["thermo"], entry["edges"]
    check(layout in ("uniform", "distributive", "freqwise"), "unsupported thermometer layout")
    check(layout != "freqwise" or raw[0] == 1, "freqwise requires one raw channel")
    expected = {"uniform": [bits], "distributive": [raw[0], bits], "freqwise": [raw[1], bits]}[layout]
    check(entry["thermo_shape"] == expected, "wrong thermometer shape")
    check(type(edges) is bool and entry.get("edge_color", False) is False, "unsupported edge encoding")
    edge_channels = 10 if edges else 0
    check(type(entry["n_edge_ch"]) is int and entry["n_edge_ch"] == edge_channels
          and encoded[0] == raw[0] * bits + edge_channels, "wrong encoded channel count")
    dtype = entry.get("input_dtype", "uint8")
    check(dtype in ("uint8", "float32"), "raw input must be uint8 or float32")
    for name, dims in [("thermo_thr", expected)] + ([("edge_thr", [5, 2])] if edges else []):
        section = sections[name]
        check(section["dtype"] == "float32" and section["shape"] == dims, f"{name}: wrong dtype/shape")
        start = section["offset_bytes"]
        values = [v for (v,) in struct.iter_unpack("<f", blob[start:start + section["size_bytes"]])]
        check(all(math.isfinite(v) for v in values), f"{name}: nonfinite threshold")
        if name == "thermo_thr":
            check(all(values[i:i + bits] == sorted(values[i:i + bits]) for i in range(0, len(values), bits)),
                  "unordered thermometer thresholds")
    return dict(raw_shape=raw, encoded_shape=encoded, input_dtype=dtype, bits=bits,
                threshold_layout=layout, edge_channels=edge_channels, thermo_thresholds="thermo_thr",
                edge_thresholds="edge_thr" if edges else None)


def lut_parameters(name, entry, inputs, outputs, conv, sections, blob):
    n = integer(entry["n"])
    check(n <= 31, f"{name}: LUT input bits must be in [1, 31]")
    tables, indices = sections[name + "_tab"], sections[name + "_idx"]
    check(tables["dtype"] == "uint8" and indices["dtype"] in ("uint16", "uint32"),
          f"{name}: expected packed uint8 tables and uint16/uint32 routing")
    count, limit = (outputs[0], inputs[0] * conv[0] * conv[1]) if conv else (outputs[2], inputs[2])
    integer(limit)
    check(integer(entry.get("p_full", limit)) == limit, f"{name}: wrong p_full")
    table_bits, index_size = 1 << n, DTYPE_BYTES[indices["dtype"]]
    table_stride = (table_bits + 7) // 8
    check(tables["shape"] == [count, table_stride], f"{name}: wrong table shape")
    check(indices["shape"] in ([count * n], [count, n])
          and indices["size_bytes"] == count * n * index_size, f"{name}: wrong mapper shape/size")
    start = indices["offset_bytes"]
    data = blob[start:start + indices["size_bytes"]]
    check(all(v < limit for (v,) in struct.iter_unpack("<H" if index_size == 2 else "<I", data)),
          f"{name}: routing index outside [0, {limit})")
    return dict(tables=name + "_tab", indices=name + "_idx", lut_count=count, input_bits=n,
                table_bits=table_bits, table_stride_bytes=table_stride, index_size_bytes=index_size,
                index_stride_bytes=n * index_size, index_limit=limit, convolution=conv)


def prepare_model(manifest, blob):
    check("operations" not in manifest, "expected the existing conv/fc JSON format")
    sections = read_sections(manifest, blob)
    encoding = prepare_encoding(manifest["encoding"], sections, blob)
    current, layers = encoding["encoded_shape"], []
    geometry = manifest.get("geometry", {})
    pools = geometry.get("pool_k", manifest.get("arch", {}).get("pool", [2] * len(manifest["conv"])))
    check(len(pools) == len(manifest["conv"]), "expected one pool kernel per convolution")

    def append(name, kind, inputs, outputs, **parameters):
        nonlocal current
        check(inputs == current, f"{name}: input shape differs from previous output")
        layers.append(dict(name=name, type=kind, input_shape=inputs, output_shape=outputs, **parameters))
        current = outputs

    for i, block in enumerate(manifest["conv"]):
        check(len(block) == 1 and block[0]["kind"] == "learnable", "expected one learnable LUT per block")
        entry = block[0]
        names = [p for p in (f"block{i + 1}", f"block{i + 1}_pw") if p + "_tab" in sections]
        check(len(names) == 1, f"block{i + 1}: expected one table section")
        name = names[0]
        in_hw = entry["in_hw"] if "in_hw" in entry else geometry["in_hw"][i]
        out_hw = entry["out_hw"] if "out_hw" in entry else geometry["out_hw"][i]
        inputs, outputs = shape(entry["c_in"], in_hw), shape(entry["k_out"], out_hw)
        kh, kw = pair(entry["ks"])
        ph, pw = pair(entry["pad"], 0)
        sh, sw = pair(entry.get("stride", 1))
        check(inputs[1] + 2 * ph >= kh and inputs[2] + 2 * pw >= kw, f"{name}: kernel exceeds input")
        check(outputs[1:] == [(inputs[1] + 2 * ph - kh) // sh + 1, (inputs[2] + 2 * pw - kw) // sw + 1],
              f"{name}: wrong convolution output shape")
        lut = lut_parameters(name, entry, inputs, outputs, [kh, kw, ph, pw, sh, sw], sections, blob)
        append(name, "lut_conv", inputs, outputs, lut=lut)
        kh, kw = pair(pools[i])
        check(current[1] >= kh and current[2] >= kw, "pool exceeds input")
        outputs = shape(current[0], [(current[1] - kh) // kh + 1, (current[2] - kw) // kw + 1])
        check("pool_hw" not in geometry or outputs[1:] == pair(geometry["pool_hw"][i]), "wrong pool geometry")
        append(f"pool{i + 1}", "or_pool", current, outputs, pool=[kh, kw, kh, kw])
    append("flatten", "flatten", current, [1, 1, math.prod(current)])
    for i, entry in enumerate(manifest["fc"]):
        name = f"fc{i + 1}"
        inputs, outputs = shape(1, [1, entry["in"]]), shape(1, [1, entry["out"]])
        append(name, "lut_fc", inputs, outputs,
               lut=lut_parameters(name, entry, inputs, outputs, None, sections, blob))
    classes, tau = integer(manifest["head_k"]), manifest["head_tau"]
    check(manifest["head_type"] == "groupsum" and current[2] % classes == 0, "invalid GroupSum groups")
    check(type(tau) in (int, float) and math.isfinite(tau) and 0 < tau <= 3.4028234663852886e38,
          "invalid float32 tau")
    tau = struct.unpack("<f", struct.pack("<f", tau))[0]
    check(tau > 0, "tau underflows float32")
    append("groupsum", "groupsum", current, [1, 1, classes],
           groupsum=dict(classes=classes, bits_per_class=current[2] // classes, tau=tau))
    used = {layer["lut"][key] for layer in layers if "lut" in layer for key in ("tables", "indices")}
    check({n for n in sections if n.endswith(("_tab", "_idx"))} <= used, "unused LUT sections")
    return dict(blob_size_bytes=len(blob), sections=list(sections.values()), encoding=encoding, layers=layers)


def c_string(value):
    return '"' + "".join(chr(b) if 32 <= b <= 126 and b not in (34, 92, 63)
                         else f"\\{b:03o}" for b in value.encode("utf-8")) + '"'


def initializer(value, depth=0):
    # Strings here are already escaped literals or internal C expressions.
    if isinstance(value, dict):
        lines = ["    " * (depth + 1) + f".{k} = {initializer(v, depth + 1)}," for k, v in value.items()]
    elif isinstance(value, list) and not all(type(v) is int for v in value):
        lines = ["    " * (depth + 1) + initializer(v, depth + 1) + "," for v in value]
    elif isinstance(value, list):
        return "{ " + ", ".join(f"{v}u" for v in value) + " }"
    else:
        return f"{value}u" if type(value) is int else (f"{value!r}f" if type(value) is float else value)
    return "{\n" + "\n".join(lines) + "\n" + "    " * depth + "}"


def render_header(model, sample_count, name):
    encoding, sections = model["encoding"], model["sections"]
    macros = {s["name"]: "SECTION_" + re.sub(r"[^A-Z0-9_]", "_", s["name"].upper()) for s in sections}
    check(len(set(macros.values())) == len(macros), "section names produce duplicate C identifiers")
    refs = {name: f"model_data + {macro}_OFFSET_BYTES" for name, macro in macros.items()}
    shapes = [encoding["encoded_shape"]] + [layer["output_shape"] for layer in model["layers"]
                                               if layer["type"] != "groupsum"]
    capacity = max(c * ((h * w + 7) // 8) for c, h, w in shapes)
    input_bytes = math.prod(encoding["raw_shape"]) * (4 if encoding["input_dtype"] == "float32" else 1)
    scratch = math.prod(encoding["raw_shape"][1:]) if encoding["edge_channels"] else 0
    integer(input_bytes)
    integer(2 * capacity + 4 * scratch)
    constants = dict(BLOB_BYTES=model["blob_size_bytes"], INPUT_BYTES=input_bytes, SAMPLE_COUNT=sample_count,
                     BUFFER_BYTES=capacity, EDGE_CHANNELS=encoding["edge_channels"], SCRATCH_FLOATS=scratch,
                     WORKSPACE_BYTES=2 * capacity + 4 * scratch, LAYER_COUNT=len(model["layers"]))
    lines = ['/* Generated by tools/generate.py; do not edit. */', '#ifndef MODEL_SPECS_H',
             '#define MODEL_SPECS_H', '', '#include "layer.h"', '#include "encoding.h"', '',
             'extern const uint8_t model_data[], input_data[];', 'extern const uint32_t expected_data[];',
             '', f'#define MODEL_NAME {c_string(name)}']
    lines += [f"#define MODEL_{key} {value}u" for key, value in constants.items()]
    lines += ['', '/* Binary offsets and sizes use bytes; LUT addresses select local table bits. */']
    for section in sections:
        macro = macros[section["name"]]
        lines += [f"#define {macro}_OFFSET_BYTES {section['offset_bytes']}u",
                  f"#define {macro}_SIZE_BYTES {section['size_bytes']}u"]
    config = dict(raw_shape=encoding["raw_shape"], input_type="INPUT_" + encoding["input_dtype"].upper(),
                  bits=encoding["bits"], threshold_layout="THRESHOLDS_" +
                  dict(uniform="SHARED", distributive="CHANNEL", freqwise="ROW")[encoding["threshold_layout"]],
                  edge_channels=encoding["edge_channels"], thermo_thresholds=refs["thermo_thr"],
                  edge_thresholds=refs["edge_thr"] if encoding["edge_thresholds"] else "NULL")
    lines += ['', f'static const Encoding model_encoding = {initializer(config)};']
    layers = []
    kinds = dict(lut_conv="LUT_CONV", lut_fc="LUT_FC", or_pool="OR_POOL", flatten="FLATTEN", groupsum="GROUP_SUM")
    for layer in model["layers"]:
        item = dict(type="LAYER_" + kinds[layer["type"]],
                    input_shape=layer["input_shape"], output_shape=layer["output_shape"])
        params = {key: dict(value) if isinstance(value, dict) else value for key, value in layer.items()
                  if key in ("lut", "pool", "groupsum")}
        if "lut" in params:
            source = params["lut"]
            params["lut"] = dict(tables=refs[source["tables"]], indices=refs[source["indices"]],
                                 lut_count=source["lut_count"], input_bits=source["input_bits"],
                                 table_stride_bytes=source["table_stride_bytes"],
                                 index_size_bytes=source["index_size_bytes"])
            if source["convolution"]:
                params["lut"]["convolution"] = source["convolution"]
        if params:
            item["parameters"] = params
        layers.append(item)
    lines += ['', f'static const Layer model_layers[MODEL_LAYER_COUNT] = {initializer(layers)};']
    return "\n".join(lines + ["", "#endif", ""])


def render_data(files):
    lines = ['/* Generated binary embedding. */', '.section .rodata.model_data,"a"']
    for symbol, path in zip(("model_data", "input_data", "expected_data"), files):
        lines += ['.balign 4', f'.global {symbol}', f'{symbol}:', f'.incbin {c_string(path.resolve().as_posix())}',
                  f'.if (. - {symbol}) != {path.stat().st_size}', '.error "Embedded file size changed"', '.endif']
    return "\n".join(lines + [""])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build/scalar")
    args = parser.parse_args()
    try:
        manifest, binary, inputs, expected = [args.data_dir / name for name in
                                               ("model.json", "model.bin", "input.bin", "expected.bin")]
        outputs = [args.output_dir / name for name in ("model_specs.h", "data.S")]
        check(not {p.resolve() for p in outputs} & {p.resolve() for p in (manifest, binary, inputs, expected)},
              "generated output would overwrite an input file")
        model = prepare_model(json.loads(manifest.read_text(encoding="utf-8")), binary.read_bytes())
        encoding = model["encoding"]
        sample_bytes = math.prod(encoding["raw_shape"]) * (4 if encoding["input_dtype"] == "float32" else 1)
        raw, predictions = inputs.read_bytes(), expected.read_bytes()
        integer(len(raw))
        integer(len(predictions))
        integer(len(raw) + len(predictions))
        count = len(raw) // sample_bytes
        check(1 <= count <= UINT32_MAX and len(raw) % sample_bytes == 0 and len(predictions) == count * 4,
              "expected whole raw samples and one uint32 prediction per sample")
        check(all(value < model["layers"][-1]["groupsum"]["classes"] for (value,) in struct.iter_unpack("<I", predictions)),
              "reference prediction outside the model's class range")
        header = render_header(model, count, args.data_dir.name)
        assembly = render_data((binary, inputs, expected))
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for path, text in zip(outputs, (header, assembly)):
            path.write_text(text, encoding="utf-8", newline="\n")
    except (OSError, ValueError, KeyError, TypeError, IndexError, OverflowError) as error:
        parser.error(str(error))
    described = sum(s["size_bytes"] for s in model["sections"])
    print(f"Validated {len(model['sections'])} sections: {described}/{model['blob_size_bytes']} bytes described; {count} samples")
    print(f"Generated {args.output_dir}/model_specs.h and data.S")


if __name__ == "__main__":
    main()


