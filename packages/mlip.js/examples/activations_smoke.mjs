// Activation-capture smoke test (Node.js).
//
// Loads pet-mad-xs.gguf, runs predictWithActivations on a water molecule,
// and prints a summary of the returned activation map. Intended as the
// minimum end-to-end check that the new C++ -> WASM -> JS path works.
//
// Run from the repo root:
//   node packages/mlip.js/examples/activations_smoke.mjs <path/to/model.gguf>

import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import createMlipDefault from '../dist/index-node.js';

const MODEL_PATH = process.argv[2]
    ?? resolve(import.meta.dirname, '../../../gguf/pet-mad-xs.gguf');

// dist/index-node.js does `require('./cpu/mlipcpp_wasm.js')` on an ESM file,
// which yields the namespace object — unwrap to the factory function.
const createMlip = typeof createMlipDefault === 'function'
    ? createMlipDefault
    : createMlipDefault.default;
const Module = await createMlip();
console.log(`mlip.js version: ${Module.getVersion()}`);

const buf = await readFile(MODEL_PATH);
// Module.Model.loadFromBuffer expects an ArrayBuffer
const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
const model = Module.Model.loadFromBuffer(ab);
console.log(`Model: ${model.modelType()}, cutoff ${model.cutoff().toFixed(2)} A`);

// H2O
const positions = new Float64Array([
    0.0, 0.0, 0.117,
    0.0, 0.757, -0.469,
    0.0, -0.757, -0.469,
]);
const atomicNumbers = new Int32Array([8, 1, 1]);
const system = Module.AtomicSystem.create(positions, atomicNumbers, null, false);

const t0 = performance.now();
const result = model.predictWithActivations(system);
const elapsed = performance.now() - t0;

console.log(`\nEnergy: ${result.energy.toFixed(6)} eV`);
console.log(`predictWithActivations took ${elapsed.toFixed(1)} ms`);

// activations is a JS Map<number, {name, dtype, shape, data}>.
const acts = result.activations;
console.log(`\nCaptured ${acts.size} tensors`);

// Aggregate by dtype
const byDtype = new Map();
let totalBytes = 0;
for (const a of acts.values()) {
    byDtype.set(a.dtype, (byDtype.get(a.dtype) ?? 0) + 1);
    if (a.data) totalBytes += a.data.byteLength;
}
console.log('By dtype:');
for (const [dt, n] of byDtype) console.log(`  ${dt}: ${n}`);
console.log(`Total payload: ${(totalBytes / 1024).toFixed(1)} KiB`);

// Sanity-check the final output (id 9999) — should match the per-atom
// energies that sum to the reported total (ignoring composition shifts,
// which are added on the C++ side after capture).
const finalOut = acts.get(9999);
if (!finalOut) {
    throw new Error('No final_output (id 9999) in activation map');
}
console.log(`\nfinal_output: name=${finalOut.name} dtype=${finalOut.dtype} shape=[${finalOut.shape.join(',')}]`);
const sum = finalOut.data.reduce((s, v) => s + v, 0);
console.log(`  per-atom energies: [${Array.from(finalOut.data).map(v => v.toFixed(4)).join(', ')}]`);
console.log(`  sum: ${sum.toFixed(6)} eV (energy minus composition = ${(result.energy - sum).toFixed(6)})`);

// Show a small slice of an interesting middle tensor
console.log('\nFirst 5 captured tensors with shape and a value preview:');
let shown = 0;
for (const [id, a] of acts) {
    if (shown >= 5) break;
    const preview = a.data
        ? Array.from(a.data.slice(0, 4)).map(v => typeof v === 'number' ? v.toFixed(3) : String(v))
        : ['(empty)'];
    console.log(`  id=${id.toString().padStart(5)}  ${a.dtype.padEnd(4)}  [${a.shape.join('x').padEnd(14)}]  ${a.name.padEnd(35)}  first=[${preview.join(', ')}${a.data && a.data.length > 4 ? ', ...' : ''}]`);
    shown++;
}
