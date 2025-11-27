# mlip.js

JavaScript/WebAssembly bindings for [mlipcpp](https://github.com/peterspackman/mlipcpp) - Machine Learning Interatomic Potentials.

Run ML potentials (PET, MACE, etc.) directly in the browser or Node.js with no native dependencies.

## Installation

```bash
npm install @mlipcpp/mlip.js
```

## Usage

### Browser (ES6 Modules)

```html
<!DOCTYPE html>
<html>
<head>
    <script type="module">
        import createMlip from '@mlipcpp/mlip.js';

        async function main() {
            const Module = await createMlip();

            // Load a model (must be fetched and provided as ArrayBuffer)
            const response = await fetch('pet-mad.gguf');
            const modelBuffer = await response.arrayBuffer();
            const model = Module.Model.loadFromBuffer(modelBuffer);

            console.log('Model type:', model.modelType());
            console.log('Cutoff:', model.cutoff(), 'Å');

            // Create a water molecule
            const positions = new Float64Array([
                0.0, 0.0, 0.117,   // O
                0.0, 0.757, -0.469, // H
                0.0, -0.757, -0.469 // H
            ]);
            const atomicNumbers = new Int32Array([8, 1, 1]);

            const water = Module.AtomicSystem.create(
                positions,
                atomicNumbers,
                null,  // no cell
                false  // not periodic
            );

            // Predict energy and forces
            const result = model.predict(water);
            console.log('Energy:', result.energy, 'eV');
            console.log('Forces:', result.forces);
        }

        main();
    </script>
</head>
<body>
    <h1>mlip.js Demo</h1>
</body>
</html>
```

### Node.js

```javascript
import createMlip from '@mlipcpp/mlip.js';
import fs from 'fs';

async function main() {
    const Module = await createMlip();

    // Load model from file
    const modelBuffer = fs.readFileSync('pet-mad.gguf');
    const model = Module.Model.loadFromBuffer(modelBuffer.buffer);

    console.log('Model loaded:', model.modelType());

    // Create system from XYZ string
    const xyzString = `3
Water molecule
O  0.0  0.0  0.117
H  0.0  0.757  -0.469
H  0.0  -0.757  -0.469
`;

    const system = Module.AtomicSystem.fromXyzString(xyzString);
    console.log('Atoms:', system.numAtoms());

    // Get energy only (faster)
    const energy = model.predictEnergy(system);
    console.log('Energy:', energy, 'eV');

    // Get energy and forces
    const result = model.predict(system);
    console.log('Forces (eV/Å):');
    for (let i = 0; i < system.numAtoms(); i++) {
        console.log(`  Atom ${i}: [${result.forces[i*3]}, ${result.forces[i*3+1]}, ${result.forces[i*3+2]}]`);
    }
}

main();
```

### Periodic Systems

```javascript
// Silicon crystal
const positions = new Float64Array([
    0.0, 0.0, 0.0,
    1.3575, 1.3575, 1.3575
]);
const atomicNumbers = new Int32Array([14, 14]);

// Cell vectors (row-major, Å)
const cell = new Float64Array([
    5.43, 0.0, 0.0,
    0.0, 5.43, 0.0,
    0.0, 0.0, 5.43
]);

const silicon = Module.AtomicSystem.create(
    positions,
    atomicNumbers,
    cell,
    true  // periodic
);

const result = model.predict(silicon);
console.log('Energy:', result.energy, 'eV');
console.log('Stress (Voigt):', result.stress); // [xx, yy, zz, yz, xz, xy]
```

## API Reference

### Module Functions

- `getVersion()` - Returns mlipcpp version string

### AtomicSystem

Represents an atomic configuration.

#### Static Methods

- `AtomicSystem.create(positions, atomicNumbers, cell, periodic)` - Create from arrays
  - `positions`: `Float64Array` - Flattened [x0,y0,z0, x1,y1,z1, ...] in Ångstroms
  - `atomicNumbers`: `Int32Array` - Atomic numbers [Z0, Z1, ...]
  - `cell`: `Float64Array` or `null` - 3x3 cell matrix (row-major) or null for non-periodic
  - `periodic`: `boolean` - Whether the system is periodic

- `AtomicSystem.fromXyzString(xyzContent)` - Parse XYZ format string

#### Instance Methods

- `numAtoms()` - Number of atoms
- `isPeriodic()` - Whether system is periodic
- `getPositions()` - Get positions as `Float64Array`
- `getAtomicNumbers()` - Get atomic numbers as `Int32Array`
- `getCell()` - Get cell as `Float64Array` or `null`

### Model

Machine learning potential model.

#### Static Methods

- `Model.load(path)` - Load from file path (Emscripten VFS)
- `Model.loadFromBuffer(arrayBuffer)` - Load from ArrayBuffer

#### Instance Methods

- `modelType()` - Model architecture name (e.g., "PET")
- `cutoff()` - Interaction cutoff radius in Ångstroms
- `isLoaded()` - Whether model is loaded
- `predictEnergy(system)` - Predict energy only (faster)
- `predict(system)` - Predict energy, forces, and stress (if periodic)

Returns object with:
- `energy`: Total energy in eV
- `forces`: `Float64Array` of forces in eV/Å
- `stress`: `Float64Array` of stress in Voigt notation (periodic only)

## Supported Models

- **PET** (Pretrained Equivariant Transformer)
- More coming soon (MACE, etc.)

Models must be in GGUF format. See mlipcpp documentation for model conversion.

## Performance

WebAssembly runs ~2-3x slower than native code. For large systems or many evaluations, consider using the native mlipcpp library.

## Building from Source

```bash
# Install Emscripten SDK
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk && ./emsdk install latest && ./emsdk activate latest
source emsdk_env.sh

# Build WASM
cd mlipcpp
./scripts/build_wasm.sh

# Build npm package
cd packages/mlip.js
npm install
npm run build
```

## License

BSD-3-Clause - see [LICENSE](LICENSE)
