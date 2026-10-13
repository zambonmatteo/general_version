"""Call the C LUT compactor and relocate sections in the runtime model blob."""

import os
from pathlib import Path
import shlex
import struct
import subprocess
import tempfile

from prepare_model import read_sections


def compact_model(manifest, blob, specs, compiler, compiler_flags):
    """Return a runtime blob, relocated layer specs and its layout metadata."""
    sections = read_sections(manifest, blob)
    replacements = {}
    with tempfile.TemporaryDirectory(prefix="lut-compaction-") as temporary:
        directory = Path(temporary)
        executable = directory / ("compact_luts.exe" if os.name == "nt" else "compact_luts")
        if specs:
            subprocess.run([compiler, *shlex.split(compiler_flags),
                            str(Path(__file__).with_name("compact_luts.c")), "-o", str(executable)], check=True)
        for spec in specs:
            name = spec["table_name"]
            if name in replacements:
                continue
            offset, length = spec["tables"]
            source, destination = directory / "tables.bin", directory / "compact.bin"
            source.write_bytes(blob[offset:offset + length])
            subprocess.run([str(executable), str(source), str(destination),
                            str(spec["n_inputs"]), str(spec["n_luts"])], check=True)
            result = destination.read_bytes()
            if len(result) < 16:
                raise ValueError("C LUT compactor returned an incomplete header")
            count, width, table_bytes, id_bytes = struct.unpack_from("<4I", result)
            row_bytes = ((1 << spec["n_inputs"]) + 7) // 8
            if (not 1 <= count <= spec["n_luts"] or width not in (0, 1, 2, 4) or
                    table_bytes != count * row_bytes or id_bytes != spec["n_luts"] * width or
                    len(result) != 16 + table_bytes + id_bytes or table_bytes + id_bytes > length or
                    (width == 0 and count not in (1, spec["n_luts"])) or
                    (width != 0 and not 1 < count < spec["n_luts"])):
                raise ValueError("C LUT compactor returned an invalid layout")
            replacements[name] = (result[16:], count, width, table_bytes, id_bytes)

    runtime = bytearray()
    relocated, layouts = {}, {}
    runtime_sections = []
    cursor = 0
    used_names = set(sections)
    for section in sorted(sections.values(), key=lambda item: item["offset"]):
        start, length = section["offset"], section["bytes"]
        # Preserve all source data and gaps; only LUT table sections are replaced.
        runtime.extend(blob[cursor:start])
        offset = len(runtime)
        entry = dict(section, offset=offset)
        name = section["name"]
        if name in replacements:
            payload, count, width, table_bytes, id_bytes = replacements[name]
            runtime.extend(payload)
            entry.update(bytes=table_bytes, shape=[count, section["shape"][1]])
            ids = (offset + table_bytes, id_bytes) if width else (0, 0)
            layouts[name] = (count, width, ids)
            if width:
                id_name = name + "_ids"
                while id_name in used_names:
                    id_name += "_"
                used_names.add(id_name)
                runtime_sections.append({"name": id_name, "offset": ids[0], "bytes": id_bytes,
                                         "dtype": f"uint{8 * width}", "shape": [id_bytes // width]})
        else:
            runtime.extend(blob[start:start + length])
        relocated[name] = (offset, entry["bytes"])
        runtime_sections.append(entry)
        cursor = start + length
    runtime.extend(blob[cursor:])
    compact_specs = []
    for spec in specs:
        count, width, ids = layouts[spec["table_name"]]
        compact_specs.append(dict(spec, n_tables=count, table_id_bytes=width, table_ids=ids,
                                  tables=relocated[spec["table_name"]],
                                  indices=relocated[spec["indices_name"]]))
    layout = {"format": "general_runtime_v1", "source_blob_bytes": len(blob),
              "total_bytes": len(runtime), "saved_bytes": len(blob) - len(runtime),
              "sections": sorted(runtime_sections, key=lambda item: item["offset"]),
              "lut_layers": compact_specs}
    return bytes(runtime), compact_specs, layout
