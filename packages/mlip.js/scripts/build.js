#!/usr/bin/env node
// Build script for mlip.js — collects both CPU and GPU WASM variants and
// emits the package's dist/ layout + a smart loader that picks the variant
// at runtime based on the requested backend.

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, '..');
const repoRoot = path.resolve(rootDir, '../..');
const distDir = path.resolve(rootDir, 'dist');
const srcDir = path.resolve(rootDir, 'src');

const variants = [
    { name: 'cpu', buildDir: path.join(repoRoot, 'wasm-cpu', 'bin') },
    { name: 'gpu', buildDir: path.join(repoRoot, 'wasm-gpu', 'bin') },
];

fs.mkdirSync(distDir, { recursive: true });
fs.mkdirSync(srcDir, { recursive: true });

console.log('Building mlip.js distribution...');

let copied = 0;
for (const v of variants) {
    const jsPath = path.join(v.buildDir, 'mlipcpp_wasm.js');
    const wasmPath = path.join(v.buildDir, 'mlipcpp_wasm.wasm');
    if (!fs.existsSync(jsPath)) {
        console.warn(`  [warn] ${v.name} build missing (${jsPath}) — skipping this variant`);
        continue;
    }

    const distVariantDir = path.join(distDir, v.name);
    const srcVariantDir = path.join(srcDir, v.name);
    fs.mkdirSync(distVariantDir, { recursive: true });
    fs.mkdirSync(srcVariantDir, { recursive: true });

    fs.copyFileSync(jsPath, path.join(distVariantDir, 'mlipcpp_wasm.js'));
    fs.copyFileSync(jsPath, path.join(srcVariantDir, 'mlipcpp_wasm.js'));
    if (fs.existsSync(wasmPath)) {
        fs.copyFileSync(wasmPath, path.join(distVariantDir, 'mlipcpp_wasm.wasm'));
        fs.copyFileSync(wasmPath, path.join(srcVariantDir, 'mlipcpp_wasm.wasm'));
    }
    console.log(`  Copied ${v.name} variant to dist/${v.name}/ and src/${v.name}/`);
    copied += 1;
}

if (copied === 0) {
    console.error('Error: no WASM variants found. Run "npm run build:wasm" first.');
    process.exit(1);
}

// Smart loader: caller passes { backend, cpuWasmUrl?, gpuWasmUrl? }. We
// dynamic-import the right Emscripten module and feed it locateFile so the
// bundler's hashed .wasm URL is used. Dynamic imports let the bundler split
// the two variants into separate chunks — only the chosen one is fetched.
const browserWrapper = `// mlip.js — browser entry with variant-aware loader.
//
// Usage in a bundler (Vite / webpack / Rollup):
//   import createMlipcpp from '@peterspackman/mlip.js'
//   import cpuWasmUrl from '@peterspackman/mlip.js/cpu-wasm?url'
//   import gpuWasmUrl from '@peterspackman/mlip.js/gpu-wasm?url'  // optional
//   const mod = await createMlipcpp({ backend: 'auto', cpuWasmUrl, gpuWasmUrl })
//
// \`backend\`: 'cpu' | 'webgpu' | 'auto' (default 'auto').
//   - 'cpu'   → always load the CPU-only build (no ASYNCIFY, no WebGPU).
//   - 'webgpu'→ always load the GPU build (WebGPU + ASYNCIFY).
//   - 'auto'  → load GPU build if navigator.gpu is present, else CPU.
//
// The two WASM URL options are plumbed to Emscripten's \`locateFile\`. If
// omitted, Emscripten's default (import.meta.url-relative resolution) is used,
// which works outside of bundlers.
export default async function createMlipcpp(options = {}) {
    const { backend = 'auto', cpuWasmUrl, gpuWasmUrl, ...rest } = options;

    const hasWebGPU = typeof navigator !== 'undefined' && 'gpu' in navigator;
    const wantGpu = backend === 'webgpu' || (backend === 'auto' && hasWebGPU);

    if (wantGpu) {
        const mod = await import('./gpu/mlipcpp_wasm.js');
        return mod.default({
            ...rest,
            ...(gpuWasmUrl ? { locateFile: (p) => p.endsWith('.wasm') ? gpuWasmUrl : p } : {}),
        });
    }

    const mod = await import('./cpu/mlipcpp_wasm.js');
    return mod.default({
        ...rest,
        ...(cpuWasmUrl ? { locateFile: (p) => p.endsWith('.wasm') ? cpuWasmUrl : p } : {}),
    });
}
`;
fs.writeFileSync(path.join(distDir, 'index.browser.js'), browserWrapper);
console.log('  Created index.browser.js');

// Node entry: use the CPU variant (no WebGPU in Node anyway).
const nodeWrapper = `// mlip.js — Node.js entry point (CPU variant).
import { createRequire } from 'module';
import path from 'path';
import { fileURLToPath } from 'url';

const require = createRequire(import.meta.url);
const __dirname = path.dirname(fileURLToPath(import.meta.url));

const createModule = require('./cpu/mlipcpp_wasm.js');

export default createModule;
export { createModule };
`;
fs.writeFileSync(path.join(distDir, 'index-node.js'), nodeWrapper);
console.log('  Created index-node.js');

const mainWrapper = `// mlip.js — main entry (auto-detects environment).
let createModule;
if (typeof window !== 'undefined' || typeof self !== 'undefined') {
    createModule = (await import('./index.browser.js')).default;
} else {
    createModule = (await import('./index-node.js')).default;
}
export default createModule;
export { createModule };
`;
fs.writeFileSync(path.join(distDir, 'index.js'), mainWrapper);
console.log('  Created index.js');

const typeDefs = `// Type definitions for mlip.js

export type Backend = 'cpu' | 'webgpu' | 'auto';

export interface CreateOptions {
    /** Which variant to load. Default: 'auto'. */
    backend?: Backend;
    /** URL of the CPU variant's .wasm file (use Vite \`?url\` import). */
    cpuWasmUrl?: string;
    /** URL of the GPU variant's .wasm file. */
    gpuWasmUrl?: string;
    /** Any additional Emscripten Module options. */
    [key: string]: unknown;
}

export interface Vec3 { x: number; y: number; z: number; }

export interface PredictionResult {
    energy: number;
    forces: Float64Array;
    stress?: Float64Array;
}

export interface AtomicSystem {
    numAtoms(): number;
    isPeriodic(): boolean;
    getPositions(): Float64Array;
    getAtomicNumbers(): Int32Array;
    getCell(): Float64Array | null;
}

export interface AtomicSystemStatic {
    create(
        positions: Float64Array,
        atomicNumbers: Int32Array,
        cell: Float64Array | null,
        periodic: boolean
    ): AtomicSystem;
    fromXyzString(xyzContent: string): AtomicSystem;
}

export interface Model {
    modelType(): string;
    cutoff(): number;
    isLoaded(): boolean;
    predictEnergy(system: AtomicSystem): number;
    predict(system: AtomicSystem): PredictionResult;
    predictWithOptions(system: AtomicSystem, useNcForces: boolean): PredictionResult;
}

export interface ModelStatic {
    load(path: string): Model;
    loadFromBuffer(buffer: ArrayBuffer): Model;
    loadFromBufferWithBackend(buffer: ArrayBuffer, backend: string): Model;
}

export interface MlipcppModule {
    AtomicSystem: AtomicSystemStatic;
    Model: ModelStatic;
    getVersion(): string;
    getBackendName(): string;
    setBackend(name: string): void;
}

declare function createMlipcpp(options?: CreateOptions): Promise<MlipcppModule>;

export default createMlipcpp;
export { createMlipcpp };
`;
fs.writeFileSync(path.join(distDir, 'index.d.ts'), typeDefs);
console.log('  Created index.d.ts');

console.log('Build complete!');
