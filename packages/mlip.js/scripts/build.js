#!/usr/bin/env node
// Build script for mlip.js - copies WASM artifacts and creates wrapper modules

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, '..');
const wasmBuildDir = path.resolve(rootDir, '../../wasm/bin');
const distDir = path.resolve(rootDir, 'dist');
const srcDir = path.resolve(rootDir, 'src');

// Ensure directories exist
fs.mkdirSync(distDir, { recursive: true });
fs.mkdirSync(srcDir, { recursive: true });

// Check if WASM build exists
const wasmJsPath = path.join(wasmBuildDir, 'mlipcpp_wasm.js');
if (!fs.existsSync(wasmJsPath)) {
    console.error('Error: WASM build not found at', wasmJsPath);
    console.error('Run "npm run build:wasm" first');
    process.exit(1);
}

console.log('Building mlip.js distribution...');

// Copy WASM JS module to src for development
fs.copyFileSync(wasmJsPath, path.join(srcDir, 'mlipcpp_wasm.js'));
console.log('  Copied mlipcpp_wasm.js to src/');

// Copy to dist (already ES6 module from Emscripten build)
fs.copyFileSync(wasmJsPath, path.join(distDir, 'mlipcpp_wasm.js'));
console.log('  Copied mlipcpp_wasm.js to dist/');

// Create browser wrapper
const browserWrapper = `// mlip.js - Browser entry point
import createMlipcpp from './mlipcpp_wasm.js';

export default createMlipcpp;
export { createMlipcpp };
`;
fs.writeFileSync(path.join(distDir, 'index.browser.js'), browserWrapper);
console.log('  Created index.browser.js');

// Create Node.js wrapper
const nodeWrapper = `// mlip.js - Node.js entry point
import { createRequire } from 'module';
import { fileURLToPath } from 'url';
import path from 'path';

const require = createRequire(import.meta.url);
const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Load the Emscripten module
const createModule = require('./mlipcpp_wasm.js');

export default createModule;
export { createModule };
`;
fs.writeFileSync(path.join(distDir, 'index-node.js'), nodeWrapper);
console.log('  Created index-node.js');

// Create main entry point
const mainWrapper = `// mlip.js - Main entry point (auto-detects environment)
let createModule;

if (typeof window !== 'undefined') {
    // Browser environment
    createModule = (await import('./index.browser.js')).default;
} else {
    // Node.js environment
    createModule = (await import('./index-node.js')).default;
}

export default createModule;
export { createModule };
`;
fs.writeFileSync(path.join(distDir, 'index.js'), mainWrapper);
console.log('  Created index.js');

// Create TypeScript definitions
const typeDefs = `// Type definitions for mlip.js

export interface Vec3 {
    x: number;
    y: number;
    z: number;
}

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
}

export interface ModelStatic {
    load(path: string): Model;
    loadFromBuffer(buffer: ArrayBuffer): Model;
}

export interface MlipcppModule {
    AtomicSystem: AtomicSystemStatic;
    Model: ModelStatic;
    getVersion(): string;
}

declare function createMlipcpp(): Promise<MlipcppModule>;

export default createMlipcpp;
export { createMlipcpp };
`;
fs.writeFileSync(path.join(distDir, 'index.d.ts'), typeDefs);
console.log('  Created index.d.ts');

console.log('Build complete!');
