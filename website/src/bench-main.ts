// Standalone main-thread benchmark: loads mlip.js directly (no worker) and
// runs a sequence of predicts on the bundled model. Purpose is to isolate
// whether Firefox slowness is specific to the Web Worker context.
//
// Reach it via /mlip.cpp/bench.html in dev (vite serves the root html).

import createMlipcpp from '@peterspackman/mlip.js'
import cpuWasmUrl from '@peterspackman/mlip.js/cpu-wasm?url'
import gpuWasmUrl from '@peterspackman/mlip.js/gpu-wasm?url'

const logEl = document.getElementById('log') as HTMLPreElement
const log = (msg: string) => {
  console.log(msg)
  logEl.textContent += msg + '\n'
}

async function main() {
  log('fetching model...')
  const modelBuf = await fetch(import.meta.env.BASE_URL + 'pet-mad-xs.gguf').then(r => r.arrayBuffer())
  log(`model: ${(modelBuf.byteLength / 1024 / 1024).toFixed(1)} MB`)

  log('instantiating wasm (main thread)...')
  const t0 = performance.now()
  // Force CPU variant for an apples-to-apples cross-browser baseline.
  const mod: any = await (createMlipcpp as any)({ backend: 'cpu', cpuWasmUrl, gpuWasmUrl })
  log(`wasm ready in ${(performance.now() - t0).toFixed(1)} ms`)

  log('loading model (cpu backend)...')
  const t1 = performance.now()
  const model = await mod.Model.loadFromBufferWithBackend(modelBuf, 'cpu')
  log(`model loaded in ${(performance.now() - t1).toFixed(1)} ms; backend = ${await mod.getBackendName()}`)

  // Build a tiny water-like system so we match what the MD panel does with
  // the bundled example.
  const positions = new Float64Array([
    0.0, 0.0, 0.0,
    0.96, 0.0, 0.0,
    -0.24, 0.93, 0.0,
  ])
  const atomicNumbers = new Int32Array([8, 1, 1])

  log('running 30 predicts...')
  for (let i = 0; i < 30; i++) {
    const system = await mod.AtomicSystem.create(positions, atomicNumbers, null, false)
    const ts = performance.now()
    const result = await model.predictWithOptions(system, false)
    const te = performance.now()
    log(`  step ${String(i).padStart(2)}: predict=${(te - ts).toFixed(1)} ms  E=${result.energy.toFixed(4)}`)
  }
  log('done.')
}

main().catch(e => log(`ERROR: ${e.message}\n${e.stack}`))
