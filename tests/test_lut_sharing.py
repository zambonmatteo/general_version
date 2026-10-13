"""Run with: python -B -m unittest discover -s tests -v."""

import hashlib
import json
import os
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from compact_model import compact_model
from prepare_model import model_encoding, packed_bytes, prepare_operations, read_sections, render_header

FLAGS = ["-O2", "-std=c11", "-Wall", "-Wextra", "-Werror", "-ffp-contract=off"]


def synthetic_model(mode="partial", geometry=None, n=6, index_bytes=2):
    blob = bytearray(b"prefix")
    sections, operations = [], []

    def section(name, dtype, shape, data):
        blob.extend(b"\xa5\x5a\x99")
        sections.append({"name": name, "offset": len(blob), "bytes": len(data),
                         "dtype": dtype, "shape": shape})
        blob.extend(data)

    def layer(name, inputs, outputs, n_inputs, conv=None):
        rows = outputs[0] if conv else outputs[2]
        row_bytes = ((1 << n_inputs) + 7) // 8
        generator = random.Random(200 + n_inputs)
        unique = 1 if mode == "shared" else 2 if mode == "tie" else rows if mode == "unique" else 3
        tables = [generator.randbytes(row_bytes) for _ in range(unique)]
        section(name + "_tab", "uint8", [rows, row_bytes],
                b"".join(tables[k % unique] for k in range(rows)))
        limit = inputs[0] * conv[0] * conv[1] if conv else inputs[2]
        indices = [(k * 3 + j * 5) % limit for k in range(rows) for j in range(n_inputs)]
        section(name + "_idx", f"uint{8 * index_bytes}", [rows * n_inputs],
                b"".join(value.to_bytes(index_bytes, "little") for value in indices))
        op = {"type": "lut_conv" if conv else "lut_fc", "n": n_inputs,
              "input_shape": list(inputs), "output_shape": list(outputs),
              "table": name + "_tab", "indices": name + "_idx"}
        if conv:
            op.update(kernel=list(conv[:2]), padding=list(conv[2:4]), stride=list(conv[4:]))
        operations.append(op)

    if geometry:
        inputs = (2, 5, 7)
        kh, kw, ph, pw, sh, sw = geometry
        outputs = (8, (5 + 2 * ph - kh) // sh + 1, (7 + 2 * pw - kw) // sw + 1)
        layer("conv", inputs, outputs, n, geometry)
        operations.append({"type": "or_pool", "kernel": [2, 2], "stride": [2, 2]})
        flat = outputs[0] * (outputs[1] // 2) * (outputs[2] // 2)
        operations.append({"type": "flatten"})
        layer("fc", (1, 1, flat), (1, 1, 16), 6)
    else:
        inputs = (1, 1, 8)
        layer("fc", inputs, (1, 1, 4 if mode == "tie" else 16), n)
    operations.append({"type": "groupsum", "classes": 2, "tau": 1.0})
    section("thermo_thr", "float32", [1], struct.pack("<f", 0.5))
    section("unused_payload", "uint8", [6], b"opaque")
    blob.extend(b"trailing-gap")
    encoding = {"in_hw": list(inputs[1:]), "in_ch": inputs[0], "bits": 1,
                "thermo": "uniform", "thermo_shape": [1], "edges": False,
                "edge_color": False, "n_edge_ch": 0, "c_in": inputs[0], "input_dtype": "uint8"}
    return {"layers": sections, "encoding": encoding, "operations": operations,
            "total_bytes": len(blob)}, bytes(blob)


class LutSharingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="lut-sharing-tests-")
        cls.directory = Path(cls.temporary.name)
        cls.tool = cls.directory / ("compact_luts.exe" if os.name == "nt" else "compact_luts")
        subprocess.run(["gcc", *FLAGS, str(ROOT / "tools/compact_luts.c"), "-o", str(cls.tool)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def compact_rows(self, rows, n):
        source, output = self.directory / "rows.bin", self.directory / "rows_compact.bin"
        source.write_bytes(b"".join(rows))
        subprocess.run([str(self.tool), str(source), str(output), str(n), str(len(rows))], check=True)
        data = output.read_bytes()
        return struct.unpack_from("<4I", data), data[16:]

    def assert_layout(self, manifest, source, original_specs, runtime, specs, layout):
        self.assertEqual(layout["total_bytes"], len(runtime))
        self.assertEqual(layout["saved_bytes"], len(source) - len(runtime))
        runtime_sections = read_sections({"total_bytes": len(runtime), "layers": layout["sections"]}, runtime)
        table_names = {s["table_name"] for s in original_specs}
        previous_end = 0
        for section in sorted(manifest["layers"], key=lambda item: item["offset"]):
            gap = source[previous_end:section["offset"]]
            offset = runtime_sections[section["name"]]["offset"]
            self.assertEqual(runtime[offset - len(gap):offset], gap)
            previous_end = section["offset"] + section["bytes"]
        trailing = source[previous_end:]
        if trailing:
            self.assertEqual(runtime[-len(trailing):], trailing)
        for section in manifest["layers"]:
            if section["name"] not in table_names:
                moved = runtime_sections[section["name"]]
                self.assertEqual(source[section["offset"]:section["offset"] + section["bytes"]],
                                 runtime[moved["offset"]:moved["offset"] + moved["bytes"]])
        for original, compact in zip(original_specs, specs):
            row_bytes = ((1 << original["n_inputs"]) + 7) // 8
            for k in range(original["n_luts"]):
                width = compact["table_id_bytes"]
                id_offset = compact["table_ids"][0] + k * width
                table = (int.from_bytes(runtime[id_offset:id_offset + width], "little") if width
                         else 0 if compact["n_tables"] == 1 else k)
                self.assertLess(table, compact["n_tables"])
                a = original["tables"][0] + k * row_bytes
                b = compact["tables"][0] + table * row_bytes
                self.assertEqual(source[a:a + row_bytes], runtime[b:b + row_bytes])
            a, size = original["indices"]
            b, moved_size = compact["indices"]
            self.assertEqual(size, moved_size)
            self.assertEqual(source[a:a + size], runtime[b:b + size])

    def check_model(self, manifest, source, samples=None, expected=None):
        inputs, originals, operations = prepare_operations(manifest, source)
        model_encoding(manifest, source)
        runtime, specs, layout = compact_model(manifest, source, originals, "gcc", " ".join(FLAGS))
        self.assert_layout(manifest, source, originals, runtime, specs, layout)
        if samples is None:
            samples = random.Random(731).randbytes(16 * packed_bytes(inputs))
        with tempfile.TemporaryDirectory(prefix="lut-model-check-") as temporary:
            directory = Path(temporary)
            header = render_header(inputs, originals, len(source), operations)
            header = (header.replace("MODEL_", "ORIGINAL_MODEL_")
                      .replace("model_lut_specs", "original_lut_specs")
                      .replace("model_operations", "original_operations")
                      .replace("model_spec", "original_spec"))
            (directory / "original_specs.h").write_text(header)
            (directory / "model_specs.h").write_text(render_header(inputs, specs, len(runtime), operations))
            (directory / "source.bin").write_bytes(source)
            (directory / "runtime.bin").write_bytes(runtime)
            (directory / "samples.bin").write_bytes(samples)
            arguments = [str(directory / name) for name in ("source.bin", "runtime.bin", "samples.bin")]
            if expected is not None:
                (directory / "expected.bin").write_bytes(expected)
                arguments.append(str(directory / "expected.bin"))
            predictions = []
            for backend in ("scalar", "bitslice"):
                executable = directory / (backend + (".exe" if os.name == "nt" else ""))
                command = ["gcc", *FLAGS, "-I", str(ROOT / "src"), "-I", str(directory),
                           str(ROOT / "tests/check_lut_models.c"),
                           *(str(ROOT / "src" / name) for name in ("model.c", "lut.c", "tensor.c"))]
                if backend == "bitslice":
                    command += ["-DMODEL_USE_BITSLICE=1", "-I", str(ROOT / "bitslice"),
                                str(ROOT / "bitslice/bitslice.c")]
                subprocess.run([*command, "-o", str(executable)], check=True)
                result = subprocess.run([str(executable), *arguments], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                predictions.append(result.stdout)
            self.assertEqual(predictions[0], predictions[1])
        return layout

    def test_shared_partial_unique_and_equal_cost(self):
        for mode, saved in (("shared", 120), ("partial", 88), ("unique", 0), ("tie", 0)):
            with self.subTest(mode=mode):
                manifest, blob = synthetic_model(mode, n=4 if mode == "tie" else 6)
                layout = self.check_model(manifest, blob)
                self.assertEqual(layout["saved_bytes"], saved)
                if mode == "shared":
                    self.assertEqual(layout["lut_layers"][0]["table_ids"], (0, 0))

    def test_conv_kernels_and_scalar_fallback(self):
        for geometry, n, index_bytes in (((3, 3, 1, 1, 1, 1), 6, 2),
                                         ((2, 3, 1, 0, 1, 1), 6, 4),
                                         ((3, 3, 1, 1, 2, 2), 9, 2)):
            with self.subTest(geometry=geometry):
                manifest, blob = synthetic_model("partial", geometry, n, index_bytes)
                layout = self.check_model(manifest, blob)
                self.assertGreater(layout["saved_bytes"], 0)

    def test_identical_rows_stay_in_separate_layer_pools(self):
        manifest, blob = synthetic_model("partial", (3, 3, 1, 1, 1, 1))
        manifest["layers"][-1]["name"] = "fc_tab_ids"
        manifest["layers"].reverse()
        inputs, originals, _ = prepare_operations(manifest, blob)
        runtime, specs, layout = compact_model(manifest, blob, originals, "gcc", " ".join(FLAGS))
        self.assert_layout(manifest, blob, originals, runtime, specs, layout)
        self.assertNotEqual(specs[0]["tables"][0], specs[1]["tables"][0])
        self.assertEqual(inputs, (2, 5, 7))
        names = {section["name"] for section in layout["sections"]}
        self.assertIn("fc_tab_ids_", names)

    def test_one_and_two_input_rows_preserve_padding(self):
        for n in (1, 2):
            with self.subTest(n=n):
                header, data = self.compact_rows([b"\x81"] * 10, n)
                self.assertEqual(header, (1, 0, 1, 0))
                self.assertEqual(data, b"\x81")
                header, data = self.compact_rows([b"\x01", b"\x81"] * 5, n)
                self.assertEqual(header, (10, 0, 10, 0))
                self.assertEqual(data, b"\x01\x81" * 5)
                manifest, blob = synthetic_model("shared", n=n)
                self.assertEqual(self.check_model(manifest, blob)["saved_bytes"], 15)

    def test_id_width_boundaries(self):
        for count, width in ((256, 1), (257, 2), (65536, 2), (65537, 4)):
            with self.subTest(count=count):
                rows = [k.to_bytes(4, "little") + b"\x55" * 12 for k in range(count)]
                header, data = self.compact_rows(rows * 2, 7)
                self.assertEqual(header, (count, width, count * 16, count * 2 * width))
                self.assertEqual(data[:count * 16], b"".join(rows))
                ids = data[count * 16:]
                for k in (0, count - 1, count, count * 2 - 1):
                    self.assertEqual(int.from_bytes(ids[k * width:(k + 1) * width], "little"), k % count)

    def test_tool_rejects_bad_dimensions_and_sizes(self):
        source, output = self.directory / "bad.bin", self.directory / "bad_compact.bin"
        for data, n, rows in ((b"x", 0, 1), (b"x", 32, 1), (b"x", 6, 0),
                              (b"x", 6, 2), (b"x" * 9, 6, 1), (b"x", 31, 32)):
            with self.subTest(n=n, rows=rows, length=len(data)):
                source.write_bytes(data)
                result = subprocess.run([str(self.tool), str(source), str(output), str(n), str(rows)],
                                        capture_output=True)
                self.assertNotEqual(result.returncode, 0)

    def test_deployment_pipeline_preserves_sources(self):
        manifest, blob = synthetic_model("shared")
        with tempfile.TemporaryDirectory(prefix="lut-deployment-test-") as temporary:
            directory = Path(temporary)
            (directory / "model").mkdir()
            (directory / "model/model.json").write_text(json.dumps(manifest))
            (directory / "model/model.bin").write_bytes(blob)
            originals = {p.name: p.read_bytes() for p in (directory / "model").iterdir()}
            for backend in ("scalar", "bitslice"):
                result = subprocess.run([sys.executable, "-B", str(ROOT / "tools/run.py"), "--target", "host",
                                         "--project-dir", str(directory), "--backend", backend],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                generated = directory / "generated"
                layout = json.loads((generated / "model_layout.json").read_text())
                self.assertEqual((generated / "model.bin").stat().st_size, layout["total_bytes"])
                self.assertEqual(layout["saved_bytes"], 120)
                self.assertFalse((generated / "model.json").exists())
                self.assertEqual(json.loads((generated / "source_model.json").read_text()), manifest)
                self.assertIn(f"Model blob: {len(blob) - 120} bytes", result.stdout)
                for name, data in originals.items():
                    self.assertEqual((directory / "model" / name).read_bytes(), data)
            result = subprocess.run([sys.executable, "-B", str(ROOT / "tools/prepare_model.py"),
                                     str(directory / "model/model.json"), "--output", str(directory / "source.h")],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"#define MODEL_BLOB_BYTES {len(blob)}u", (directory / "source.h").read_text())

    def test_raw_input_pipeline_on_real_models(self):
        fixtures = (("mnist", "mnist_k32_s0_uni_aug_best", "test_images.bin"),
                    ("cifar10", "cifar_k32_s0_n4_best", "test_images.bin"),
                    ("kws", "kws_final_k32_s0_bestval", "test_inputs.bin"))
        found = False
        for family, name, raw_name in fixtures:
            model_dir = ROOT.parent / family / "models" / name
            raw_path = ROOT.parent / family / "data" / raw_name
            if not raw_path.is_file() or not (model_dir / "ref_preds.bin").is_file():
                continue
            found = True
            with self.subTest(model=name), tempfile.TemporaryDirectory(prefix="lut-raw-input-test-") as temporary:
                directory = Path(temporary)
                manifest = json.loads((model_dir / "model.json").read_text())
                config = model_encoding(manifest, (model_dir / "model.bin").read_bytes())
                c, h, w = config["shape"]
                sample_bytes = c * h * w * (4 if config["input_dtype"] == "float32" else 1)
                with raw_path.open("rb") as file:
                    raw = file.read(10 * sample_bytes)
                with (model_dir / "ref_preds.bin").open("rb") as file:
                    expected = file.read(10)
                (directory / "input_test").mkdir()
                (directory / "input_test/inputs.bin").write_bytes(raw)
                (directory / "input_test/ref_preds.bin").write_bytes(expected)
                (directory / "input").mkdir()
                (directory / "input/inputs.bin").write_bytes(raw[:2 * sample_bytes])
                for backend in ("scalar", "bitslice"):
                    result = subprocess.run([sys.executable, "-B", str(ROOT / "tools/run.py"),
                                             "--target", "host", "--project-dir", str(directory),
                                             "--model-dir", str(model_dir), "--backend", backend, "--repeats", "1"],
                                            capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn("Verification mismatches: 0 / 10", result.stdout)
                    for i in range(2):
                        self.assertIn(f"Input {i}: predicted={expected[i]}", result.stdout)
        if not found:
            self.skipTest("No raw-input reference fixtures are available beside this project")

    def test_available_real_models(self):
        model_dirs = (ROOT.parent / "mnist/models/mnist_k32_s0_uni_aug_best",
                      ROOT.parent / "cifar10/models/cifar_k32_s0_n4_best",
                      ROOT.parent / "kws/models/kws_final_k32_s0_bestval")
        found = False
        for directory in model_dirs:
            if not (directory / "ref_encoded.bin").is_file():
                continue
            found = True
            with self.subTest(model=directory.name):
                original_files = {name: hashlib.sha256((directory / name).read_bytes()).digest()
                                  for name in ("model.json", "model.bin")}
                manifest = json.loads((directory / "model.json").read_text())
                source = (directory / "model.bin").read_bytes()
                inputs, _, _ = prepare_operations(manifest, source)
                with (directory / "ref_encoded.bin").open("rb") as file:
                    samples = file.read(10 * packed_bytes(inputs))
                with (directory / "ref_preds.bin").open("rb") as file:
                    expected = file.read(10)
                layout = self.check_model(manifest, source, samples, expected)
                print(f"\n{directory.name}: {len(source)} -> {layout['total_bytes']} bytes; "
                      f"saved {layout['saved_bytes']} bytes", flush=True)
                for name, digest in original_files.items():
                    self.assertEqual(hashlib.sha256((directory / name).read_bytes()).digest(), digest)
        if not found:
            self.skipTest("No reference model fixtures are available beside this project")


if __name__ == "__main__":
    unittest.main()
