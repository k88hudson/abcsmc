// Runs the wasm_smoke example under Node with no imports at all, so it fails
// if the engine needs anything from a JavaScript host or traps at runtime.
// Usage: node tests/wasm_smoke.mjs <path to wasm_smoke.wasm>
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const path = process.argv[2];
assert.ok(path, "usage: wasm_smoke.mjs <wasm file>");

const module = await WebAssembly.compile(await readFile(path));
const imports = WebAssembly.Module.imports(module);
assert.deepEqual(
  imports,
  [],
  `the module must not import from the host, found: ${JSON.stringify(imports)}`,
);

const { exports } = await WebAssembly.instantiate(module, {});

const mean = exports.posterior_mean();
assert.ok(
  Math.abs(mean - 5) < 0.5,
  `posterior mean should be near 5, got ${mean}`,
);
assert.equal(exports.default_run_generations(), 2);

console.log(`wasm smoke ok: posterior mean ${mean.toFixed(3)}`);
