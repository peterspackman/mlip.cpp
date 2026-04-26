/// <reference lib="webworker" />
// Web Worker for molecular dynamics simulation
// Runs mlip.js inference off the main thread
//
// Units used throughout:
// - Positions: Angstrom (A)
// - Velocities: A/fs
// - Forces: eV/A (from mlip.js)
// - Energy: eV
// - Mass: amu (atomic mass units)
// - Time: fs (femtoseconds)
// - Temperature: K

import createMlipcpp from '@peterspackman/mlip.js'
import type { MlipcppModule, Model, AtomicSystem } from '@peterspackman/mlip.js'
// Grab both variant WASM URLs so Vite bundles them as static assets. The
// loader picks one at runtime based on the user's backend choice.
import cpuWasmUrl from '@peterspackman/mlip.js/cpu-wasm?url'
import gpuWasmUrl from '@peterspackman/mlip.js/gpu-wasm?url'

interface WorkerState {
  module: MlipcppModule | null
  model: Model | null
  system: AtomicSystem | null
  positions: Float64Array | null
  velocities: Float64Array | null
  forces: Float64Array | null  // Cached forces from previous step
  atomicNumbers: Int32Array | null
  masses: Float64Array | null
  cell: Float64Array | null
  isPeriodic: boolean
  numAtoms: number
  dt: number  // timestep in fs
  temperature: number  // target temperature in K
  isRunning: boolean
  // FIRE optimizer state
  fireAlpha: number
  fireNpos: number
  fireDt: number
  fireMaxDt: number
  cellVelocities: Float64Array | null  // Cell velocities for FIRE (9 components)
  mode: 'md' | 'optimize'
  maxOptSteps: number
  forceThreshold: number
  stressThreshold: number  // Convergence threshold for stress (eV/A^3)
  optStep: number
  optimizeCell: boolean  // Whether to optimize cell in FIRE
  thermostat: 'csvr' | 'none'
  thermostatTau: number  // fs
  useConservativeForces: boolean  // false = NC forces (faster, non-conservative)
  initialTotalEnergy: number | null  // baseline for NVE drift diagnostic
  optimizer: 'lbfgs' | 'fire' | 'cg'
  lbfgs: {
    history: { s: Float64Array; y: Float64Array; rho: number }[]
    currentE: number
    currentG: Float64Array | null
    step: number
  } | null
  cg: {
    currentE: number
    currentG: Float64Array | null  // gradient = -forces
    dir: Float64Array | null       // current search direction
    step: number
    /** Counts steps since the last steepest-descent restart. */
    sinceRestart: number
  } | null
  /** Atoms whose forces are zeroed during optimization (or velocities
   *  zero-initialized in FIRE). null = no constraints. Set per `start`. */
  frozenAtoms: Set<number> | null
}

const state: WorkerState = {
  module: null,
  model: null,
  system: null,
  positions: null,
  velocities: null,
  forces: null,
  atomicNumbers: null,
  masses: null,
  cell: null,
  isPeriodic: false,
  numAtoms: 0,
  dt: 1.0,
  temperature: 300,
  isRunning: false,
  // FIRE optimizer defaults
  fireAlpha: 0.1,
  fireNpos: 0,
  fireDt: 0.1,  // fs - initial timestep for FIRE
  fireMaxDt: 1.0,  // fs - max timestep
  cellVelocities: null,
  mode: 'md',
  maxOptSteps: 100,
  forceThreshold: 0.05,  // eV/A
  stressThreshold: 0.01,  // eV/A^3 (~1.6 GPa)
  optStep: 0,
  optimizeCell: true,  // Default to optimizing cell for periodic systems
  thermostat: 'none',           // NVE by default — honest physics over pretty thermostat
  thermostatTau: 100,
  useConservativeForces: true,  // Conservative forces by default so NVE actually conserves
  initialTotalEnergy: null,
  optimizer: 'lbfgs',
  lbfgs: null,
  cg: null,
  frozenAtoms: null,
}

/** Zero out gradient (or force) entries for atoms the user has frozen.
 *  Called after every gradient compute in CG/L-BFGS, and after each force
 *  evaluation in FIRE. Effect: frozen atoms see zero force → never move. */
function maskFrozen(g: Float64Array): void {
  if (!state.frozenAtoms) return
  for (const i of state.frozenAtoms) {
    const k = i * 3
    if (k + 2 < g.length) {
      g[k] = 0
      g[k + 1] = 0
      g[k + 2] = 0
    }
  }
}

// Standard atomic weights in amu (IUPAC 2021). Covers rows 1-5 plus the common
// heavier elements seen in MLIP training sets. Unknown Z falls back to carbon
// with a console warning so missing entries are visible.
const ATOMIC_MASSES: Record<number, number> = {
  1: 1.008, 2: 4.0026,
  3: 6.94, 4: 9.0122, 5: 10.81, 6: 12.011, 7: 14.007, 8: 15.999, 9: 18.998, 10: 20.180,
  11: 22.990, 12: 24.305, 13: 26.982, 14: 28.085, 15: 30.974, 16: 32.06, 17: 35.45, 18: 39.95,
  19: 39.098, 20: 40.078, 21: 44.956, 22: 47.867, 23: 50.942, 24: 51.996, 25: 54.938, 26: 55.845,
  27: 58.933, 28: 58.693, 29: 63.546, 30: 65.38, 31: 69.723, 32: 72.630, 33: 74.922, 34: 78.971,
  35: 79.904, 36: 83.798,
  37: 85.468, 38: 87.62, 39: 88.906, 40: 91.224, 41: 92.906, 42: 95.95, 44: 101.07, 45: 102.91,
  46: 106.42, 47: 107.87, 48: 112.41, 49: 114.82, 50: 118.71, 51: 121.76, 52: 127.60, 53: 126.90, 54: 131.29,
  55: 132.91, 56: 137.33, 72: 178.49, 73: 180.95, 74: 183.84, 75: 186.21, 76: 190.23, 77: 192.22,
  78: 195.08, 79: 196.97, 80: 200.59, 81: 204.38, 82: 207.2, 83: 208.98,
}

const warnedMassZ = new Set<number>()
function massFor(z: number): number {
  const m = ATOMIC_MASSES[z]
  if (m !== undefined) return m
  if (!warnedMassZ.has(z)) {
    warnedMassZ.add(z)
    console.warn(`[mdWorker] No atomic mass for Z=${z}; using 12.011 (carbon). Dynamics will be wrong for this element.`)
  }
  return 12.011
}

// Physical constants
// kB in eV/K
const KB_EV = 8.617333262e-5

// Conversion factor: eV/(A * amu) -> A/fs^2
// F = ma => a = F/m
// [F] = eV/A, [m] = amu, [a] = A/fs^2
// 1 eV = 1.602176634e-19 J
// 1 amu = 1.66053906660e-27 kg
// 1 A = 1e-10 m, 1 fs = 1e-15 s
// a [m/s^2] = F[N] / m[kg] = F[eV/A] * 1.602176634e-19 / (1e-10) / (m[amu] * 1.66053906660e-27)
// a [A/fs^2] = a [m/s^2] * 1e-10 / (1e-15)^2 = a [m/s^2] * 1e-10 / 1e-30 = a [m/s^2] * 1e20
// Combining: eV_A_to_A_fs2 = 1.602176634e-19 / 1e-10 / 1.66053906660e-27 * 1e-20
//                         = 1.602176634e-19 * 1e10 / 1.66053906660e-27 * 1e-20
//                         = 1.602176634 / 1.66053906660 * 1e-19+10+27-20
//                         = 0.9648533 * 1e-2 = 9.648533e-3
const EV_A_AMU_TO_A_FS2 = 9.648533e-3

// For velocity initialization from temperature:
// <1/2 m v^2> = 3/2 kB T (per atom, 3 DOF)
// <v^2> = 3 kB T / m
// sigma_v = sqrt(kB T / m) for each component
// With kB in eV/K, m in amu, we need v in A/fs
// v^2 [A^2/fs^2] = kB[eV/K] * T[K] / m[amu] * conversion
// 1 eV = 1.602176634e-19 J = 1.602176634e-19 kg m^2/s^2
// 1 amu = 1.66053906660e-27 kg
// v^2 [m^2/s^2] = kB[J/K] * T / m[kg]
// v^2 [A^2/fs^2] = v^2 [m^2/s^2] * (1e10)^2 / (1e15)^2 = v^2 * 1e-10
// kB_for_velocity = kB[eV/K] * 1.602176634e-19 / 1.66053906660e-27 * 1e-10
//                 = 8.617333262e-5 * 1.602176634e-19 / 1.66053906660e-27 * 1e-10
//                 = 8.617333262e-5 * 9.648533e7 * 1e-10
//                 = 8.617333262e-5 * 9.648533e-3
//                 = 8.314e-7... wait let me recalculate
// Actually simpler: kB * T / m with kB in proper units
// kB = 1.380649e-23 J/K, convert to amu*A^2/fs^2/K:
// 1 J = 1 kg*m^2/s^2 = (1/1.66053906660e-27 amu) * (1e10 A)^2 / (1e15 fs)^2
//     = 6.0221e26 amu * 1e20 A^2 / 1e30 fs^2 = 6.0221e16 amu*A^2/fs^2
// kB = 1.380649e-23 * 6.0221e16 = 8.314e-7 amu*A^2/fs^2/K
const KB_AMU_A2_FS2 = 8.314462618e-7

// Initialize velocities from Maxwell-Boltzmann distribution
function initializeVelocities(numAtoms: number, masses: Float64Array, temperature: number): Float64Array {
  const velocities = new Float64Array(numAtoms * 3)

  // Generate random velocities from Gaussian distribution
  for (let i = 0; i < numAtoms; i++) {
    const mass = masses[i]
    // sigma = sqrt(kB * T / m) in A/fs
    const sigma = Math.sqrt(KB_AMU_A2_FS2 * temperature / mass)

    for (let j = 0; j < 3; j++) {
      // Box-Muller transform for Gaussian
      const u1 = Math.random()
      const u2 = Math.random()
      velocities[i * 3 + j] = sigma * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
    }
  }

  // Remove center of mass velocity
  let vx = 0, vy = 0, vz = 0
  let totalMass = 0
  for (let i = 0; i < numAtoms; i++) {
    const mass = masses[i]
    vx += mass * velocities[i * 3]
    vy += mass * velocities[i * 3 + 1]
    vz += mass * velocities[i * 3 + 2]
    totalMass += mass
  }
  vx /= totalMass
  vy /= totalMass
  vz /= totalMass

  for (let i = 0; i < numAtoms; i++) {
    velocities[i * 3] -= vx
    velocities[i * 3 + 1] -= vy
    velocities[i * 3 + 2] -= vz
  }

  // Maxwell-Boltzmann sampling has ~sqrt(2/dof) relative variance in T, which
  // is ~60% for a 3-atom system. Rescale to hit the target T exactly so we
  // don't start hundreds of K off target and blame the thermostat.
  const { temp: tInit } = calculateKineticEnergy(velocities, masses, numAtoms)
  if (tInit > 1e-10) {
    const scale = Math.sqrt(temperature / tInit)
    for (let i = 0; i < velocities.length; i++) velocities[i] *= scale
  }

  return velocities
}

// Calculate kinetic energy and temperature
function calculateKineticEnergy(velocities: Float64Array, masses: Float64Array, numAtoms: number): { ke: number, temp: number } {
  let ke = 0
  for (let i = 0; i < numAtoms; i++) {
    const mass = masses[i]
    const vx = velocities[i * 3]
    const vy = velocities[i * 3 + 1]
    const vz = velocities[i * 3 + 2]
    ke += 0.5 * mass * (vx * vx + vy * vy + vz * vz)
  }
  // ke is in amu * A^2/fs^2, convert to eV
  // 1 amu * A^2/fs^2 = 1.66053906660e-27 kg * (1e-10 m)^2 / (1e-15 s)^2
  //                  = 1.66053906660e-27 * 1e-20 / 1e-30 kg*m^2/s^2
  //                  = 1.66053906660e-27 * 1e10 J
  //                  = 1.66053906660e-17 J
  //                  = 1.66053906660e-17 / 1.602176634e-19 eV
  //                  = 103.6427 eV... that seems too big
  // Let me recalculate:
  // 1 amu*A^2/fs^2 = 1.66053906660e-27 kg * (1e-10)^2 m^2 / (1e-15)^2 s^2
  //                = 1.66053906660e-27 * 1e-20 * 1e30 J
  //                = 1.66053906660e-17 J = 0.1036 eV
  // So conversion factor is ~0.1036 or more precisely:
  const AMU_A2_FS2_TO_EV = 1.66053906660e-17 / 1.602176634e-19  // = 0.10364

  ke *= AMU_A2_FS2_TO_EV

  // Temperature from equipartition: KE = 3/2 * N * kB * T (for N atoms, 3N DOF minus 3 for COM)
  // For small systems, use (3N - 3) DOF
  const dof = Math.max(3 * numAtoms - 3, 1)
  const temp = (2 * ke) / (dof * KB_EV)

  return { ke, temp }
}


// Standard normal sample via Box-Muller. One value per call (the paired
// sample is recomputed next call — cheap enough for thermostat use).
function gaussian(): number {
  const u1 = Math.max(Math.random(), 1e-300)
  const u2 = Math.random()
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
}

// Sum of n independent chi-squared(1) = sum of n squared N(0,1). Exact via
// Box-Muller for n up to a few hundred (our DOF counts are tiny).
function sumSquaredGaussians(n: number): number {
  if (n <= 0) return 0
  let s = 0
  // Consume Gaussians in pairs so Box-Muller isn't wasted.
  for (let i = 0; i < n - 1; i += 2) {
    const u1 = Math.max(Math.random(), 1e-300)
    const u2 = Math.random()
    const r = Math.sqrt(-2 * Math.log(u1))
    const g1 = r * Math.cos(2 * Math.PI * u2)
    const g2 = r * Math.sin(2 * Math.PI * u2)
    s += g1 * g1 + g2 * g2
  }
  if (n % 2 === 1) {
    const g = gaussian()
    s += g * g
  }
  return s
}

// Canonical sampling through velocity rescaling (Bussi, Donadio, Parrinello,
// JCP 126, 014101 (2007)). Samples the canonical distribution exactly while
// being as robust and simple as Berendsen. Assumes COM already removed so
// Nf = 3N - 3.
function csvrThermostat(
  velocities: Float64Array,
  masses: Float64Array,
  numAtoms: number,
  targetTemp: number,
  tau: number,  // fs
  dt: number    // fs
): void {
  const ndeg = Math.max(3 * numAtoms - 3, 1)

  // Current KE in amu·A^2/fs^2 (the native units we work with)
  let kk = 0
  for (let i = 0; i < numAtoms; i++) {
    const m = masses[i]
    const vx = velocities[i * 3], vy = velocities[i * 3 + 1], vz = velocities[i * 3 + 2]
    kk += 0.5 * m * (vx * vx + vy * vy + vz * vz)
  }
  if (kk <= 0) return

  // Target KE (sigma in Bussi's notation) = 0.5 * Nf * kB * T
  const sigma = 0.5 * ndeg * KB_AMU_A2_FS2 * targetTemp

  // Exponential decay factor per step. tau <= 0 disables coupling.
  const factor = tau > 0 ? Math.exp(-dt / tau) : 0

  const rr = gaussian()
  const s2 = sumSquaredGaussians(ndeg - 1)

  const newKk =
    kk
    + (1 - factor) * (sigma * (s2 + rr * rr) / ndeg - kk)
    + 2 * rr * Math.sqrt(kk * sigma / ndeg * (1 - factor) * factor)

  if (newKk <= 0) return  // extremely rare numerical edge; skip this step
  const alpha = Math.sqrt(newKk / kk)
  for (let i = 0; i < velocities.length; i++) velocities[i] *= alpha
}

// Remove center of mass velocity
function removeCOMVelocity(
  velocities: Float64Array,
  masses: Float64Array,
  numAtoms: number
): void {
  let vx = 0, vy = 0, vz = 0
  let totalMass = 0

  for (let i = 0; i < numAtoms; i++) {
    const mass = masses[i]
    vx += mass * velocities[i * 3]
    vy += mass * velocities[i * 3 + 1]
    vz += mass * velocities[i * 3 + 2]
    totalMass += mass
  }

  vx /= totalMass
  vy /= totalMass
  vz /= totalMass

  for (let i = 0; i < numAtoms; i++) {
    velocities[i * 3] -= vx
    velocities[i * 3 + 1] -= vy
    velocities[i * 3 + 2] -= vz
  }
}

// Instantiate the WASM module for the chosen backend. The CPU and GPU
// variants are completely separate binaries (different compile flags), so
// swapping backends means throwing away the old module entirely.
async function instantiateModule(backend: 'cpu' | 'webgpu' | 'auto'): Promise<MlipcppModule> {
  return (createMlipcpp as any)({
    backend,
    cpuWasmUrl,
    gpuWasmUrl,
  }) as Promise<MlipcppModule>
}

// Message handlers
async function handleInit(_data: { modelBuffer?: ArrayBuffer }): Promise<void> {
  try {
    // Defer actual WASM instantiation to loadModel, where the backend is
    // known. Replying 'initialized' here just signals "worker is alive".
    self.postMessage({ type: 'initialized', version: '' })
  } catch (err: any) {
    self.postMessage({ type: 'error', message: `Initialization failed: ${err.message}` })
  }
}

// Track which variant ('cpu' | 'gpu') is currently instantiated so we can
// detect backend changes and rebuild the module when needed.
let loadedVariant: 'cpu' | 'gpu' | null = null
// Cached copy of the last-loaded GGUF bytes so we can re-instantiate the
// model when the user flips the backend selector, without needing the main
// thread to re-post the buffer (which was already transferred).
let cachedModelBuffer: ArrayBuffer | null = null

async function ensureModuleForBackend(requested: 'cpu' | 'webgpu' | 'auto'): Promise<'cpu' | 'webgpu'> {
  const hasWebGPU = typeof navigator !== 'undefined' && 'gpu' in navigator
  // Resolve 'auto' to a concrete variant so we can compare against what's loaded.
  const desired: 'cpu' | 'webgpu' = (requested === 'webgpu' || (requested === 'auto' && hasWebGPU)) ? 'webgpu' : 'cpu'
  const desiredVariant: 'cpu' | 'gpu' = desired === 'webgpu' ? 'gpu' : 'cpu'

  if (state.module && loadedVariant === desiredVariant) return desired

  // Tear down any existing module (and its buffers) before loading a new variant.
  if (state.model && typeof (state.model as any).delete === 'function') {
    try { (state.model as any).delete() } catch { /* ignore */ }
  }
  state.model = null
  if (state.system && typeof (state.system as any).delete === 'function') {
    try { (state.system as any).delete() } catch { /* ignore */ }
  }
  state.system = null
  state.forces = null
  state.initialTotalEnergy = null
  state.module = null
  loadedVariant = null

  state.module = await instantiateModule(desired)
  loadedVariant = desiredVariant
  return desired
}

async function handleLoadModel(data: { buffer: ArrayBuffer, backend?: string }): Promise<void> {
  try {
    const requested = (data.backend || 'auto') as 'cpu' | 'webgpu' | 'auto'

    let backend: 'cpu' | 'webgpu' = 'cpu'
    try {
      backend = await ensureModuleForBackend(requested)
    } catch (gpuErr: any) {
      if (requested === 'cpu') throw gpuErr
      console.warn(`[mlip.cpp] GPU variant failed to load (${gpuErr?.message ?? 'unknown'}); falling back to CPU.`)
      backend = await ensureModuleForBackend('cpu')
    }

    // Release any stale model/system left over from a prior load with the
    // same variant (ensureModuleForBackend only clears them on variant swap).
    if (state.model && typeof (state.model as any).delete === 'function') {
      try { (state.model as any).delete() } catch { /* ignore */ }
    }
    state.model = null
    if (state.system && typeof (state.system as any).delete === 'function') {
      try { (state.system as any).delete() } catch { /* ignore */ }
    }
    state.system = null
    state.forces = null
    state.initialTotalEnergy = null

    const mod = state.module!
    try {
      state.model = await mod.Model.loadFromBufferWithBackend(data.buffer, backend)
    } catch (gpuErr: any) {
      if (backend === 'cpu') throw gpuErr
      console.warn(`[mlip.cpp] WebGPU init failed at model load (${gpuErr?.message ?? 'unknown'}); falling back to CPU.`)
      backend = await ensureModuleForBackend('cpu')
      state.model = await state.module!.Model.loadFromBufferWithBackend(data.buffer, 'cpu')
    }
    // Keep the raw bytes so a backend flip can reload without another postMessage.
    // loadFromBufferWithBackend copies into the model's own memory, so the
    // original ArrayBuffer is safe to retain.
    cachedModelBuffer = data.buffer

    self.postMessage({
      type: 'modelLoaded',
      modelType: await state.model.modelType(),
      cutoff: await state.model.cutoff(),
      backend: await state.module!.getBackendName(),
    })
  } catch (err: any) {
    self.postMessage({ type: 'error', message: `Failed to load model: ${err.message}` })
  }
}

// Re-instantiate the currently loaded model on the requested backend.
// The main thread calls this when the backend selector changes but the user
// hasn't dropped a new .gguf. Reuses the cached model bytes from the last
// successful loadModel call.
async function handleSetBackend(data: { backend?: string }): Promise<void> {
  if (!cachedModelBuffer) {
    // Nothing loaded yet — the next loadModel will use the new backend.
    self.postMessage({ type: 'backendSet', backend: '' })
    return
  }
  if (state.isRunning) handleStop()
  await handleLoadModel({ buffer: cachedModelBuffer, backend: data.backend })
  // handleLoadModel posts 'modelLoaded'; relay a dedicated reply too so the
  // main-thread RPC completes cleanly.
  self.postMessage({
    type: 'backendSet',
    backend: state.module ? await state.module.getBackendName() : '',
  })
}

async function handleSetSystem(data: { xyz: string }): Promise<void> {
  // Module instantiation is normally deferred to loadModel (where the backend
  // is known). Structure-only flows (build / edit a molecule before picking a
  // model) hit this path with no module yet — instantiate the CPU variant on
  // demand so setSystem doesn't fail. A later loadModel with a different
  // backend will swap modules via ensureModuleForBackend.
  if (!state.module) {
    try {
      await ensureModuleForBackend('cpu')
    } catch (err: any) {
      self.postMessage({ type: 'error', message: `Failed to initialize WASM: ${err.message}` })
      return
    }
  }

  try {
    state.system = await state.module!.AtomicSystem.fromXyzString(data.xyz)
    state.numAtoms = await state.system.numAtoms()
    state.isPeriodic = await state.system.isPeriodic()
    state.positions = new Float64Array(await state.system.getPositions())
    state.atomicNumbers = new Int32Array(await state.system.getAtomicNumbers())
    const cellArr = await state.system.getCell()
    state.cell = cellArr ? new Float64Array(cellArr) : null

    // Set up masses
    state.masses = new Float64Array(state.numAtoms)
    for (let i = 0; i < state.numAtoms; i++) {
      const z = state.atomicNumbers[i]
      state.masses[i] = massFor(z)
    }

    // Initialize velocities and clear all cached forces/state
    state.velocities = initializeVelocities(state.numAtoms, state.masses, state.temperature)
    state.forces = null
    state.initialTotalEnergy = null

    // Clear FIRE optimizer cache
    fireForces = null
    fireStress = null
    fireCellForce = null

    // Reset FIRE state
    state.fireAlpha = FIRE_ALPHA_START
    state.fireNpos = 0
    state.fireDt = 0.1
    state.optStep = 0
    state.cellVelocities = null

    // Constraints reset: a fresh structure has no idea what was frozen.
    state.frozenAtoms = null

    self.postMessage({
      type: 'systemSet',
      numAtoms: state.numAtoms,
      isPeriodic: state.isPeriodic,
    })
  } catch (err: any) {
    self.postMessage({ type: 'error', message: `Failed to set system: ${err.message}` })
  }
}

async function handlePredict(): Promise<void> {
  if (!state.module || !state.model || !state.system) {
    self.postMessage({ type: 'error', message: 'System or model not ready' })
    return
  }

  try {
    // Use NC forces for faster prediction (non-conservative forces from forward pass)
    const result = await state.model.predictWithOptions(state.system, true)
    // result.forces is a Float32Array owned by the embind call — copy into a
    // Float64Array we can transfer, to keep the main thread in double precision.
    const forcesOut = new Float64Array(result.forces)
    self.postMessage({
      type: 'prediction',
      energy: result.energy,
      forces: forcesOut,
    }, [forcesOut.buffer])
  } catch (err: any) {
    self.postMessage({ type: 'error', message: `Prediction failed: ${err.message}` })
  }
}

// Predict at arbitrary positions without touching the cached MD state. Reuses
// the loaded species/cell/PBC. Forced to conservative forces because this is
// used for physically meaningful things (Hessian, scans) where NC wouldn't
// give a symmetric/reciprocal-compatible result.
async function handlePredictAt(data: {
  positions: Float64Array,
  id?: number,
}): Promise<void> {
  try {
    const result = await predictAtPositions(data.positions)
    const forcesOut = new Float64Array(result.forces)
    self.postMessage({
      type: 'predictAtResult',
      id: data.id,
      energy: result.energy,
      forces: forcesOut,
    }, [forcesOut.buffer])
  } catch (err: any) {
    self.postMessage({
      type: 'predictAtResult',
      id: data.id,
      error: err?.message ?? String(err),
    })
  }
}

function handleSetParameters(data: {
  dt?: number,
  temperature?: number,
  mode?: 'md' | 'optimize',
  maxOptSteps?: number,
  forceThreshold?: number,
  thermostat?: 'csvr' | 'none',
  thermostatTau?: number,
  useConservativeForces?: boolean,
  optimizer?: 'lbfgs' | 'fire' | 'cg',
}): void {
  if (data.dt !== undefined) state.dt = data.dt
  if (data.temperature !== undefined) state.temperature = data.temperature
  if (data.mode !== undefined) state.mode = data.mode
  if (data.maxOptSteps !== undefined) state.maxOptSteps = data.maxOptSteps
  if (data.forceThreshold !== undefined) state.forceThreshold = data.forceThreshold
  if (data.thermostat !== undefined) state.thermostat = data.thermostat
  if (data.thermostatTau !== undefined) state.thermostatTau = data.thermostatTau
  if (data.useConservativeForces !== undefined) {
    // Changing force type invalidates any cached forces.
    if (state.useConservativeForces !== data.useConservativeForces) {
      state.forces = null
      state.initialTotalEnergy = null
    }
    state.useConservativeForces = data.useConservativeForces
  }
  if (data.optimizer !== undefined) state.optimizer = data.optimizer
  self.postMessage({ type: 'parametersSet', dt: state.dt, temperature: state.temperature })
}

let mdTimeout: ReturnType<typeof setTimeout> | null = null

// Shared predict helper — build an AtomicSystem at arbitrary positions and get
// energy + forces via the currently loaded model. Always conservative (forces
// must be gradients of the energy for optimization and FD Hessian to make
// physical sense).
async function predictAtPositions(
  positions: Float64Array,
): Promise<{ energy: number; forces: ArrayLike<number> }> {
  if (!state.module || !state.model || !state.atomicNumbers) {
    throw new Error('Module/model/system not ready')
  }
  const system = await state.module.AtomicSystem.create(
    positions,
    state.atomicNumbers,
    state.cell,
    state.isPeriodic,
  )
  const result = await state.model.predictWithOptions(system, false)
  return result
}

// ========== L-BFGS optimizer ==========
//
// Limited-memory BFGS for atom positions (no cell DOFs — cell optimization
// stays on FIRE). Implements Nocedal's two-loop recursion with a scaled
// identity initial Hessian.
//
// Knobs:
//   LBFGS_M        history depth (number of (s, y) pairs kept)
//   LBFGS_MAX_STEP cap on the infinity-norm of the displacement per step (Å)
//   LBFGS_LS_MAX   max backtracking line-search trials per step
//   LBFGS_ARMIJO   Armijo sufficient-decrease constant
//
// Line search: start α=1, backtrack α ← α/2 until the energy decreases by
// Armijo · α · g·d. If the budget runs out, take the last trial anyway —
// better than stalling.
const LBFGS_M = 10
const LBFGS_MAX_STEP = 0.2       // Å
const LBFGS_LS_MAX = 5
const LBFGS_ARMIJO = 1e-4

async function resetLBFGS(): Promise<void> {
  state.lbfgs = {
    history: [],
    currentE: 0,
    currentG: null,
    step: 0,
  }
}

// L-BFGS two-loop recursion: returns the search direction d = -H_k g.
function lbfgsDirection(
  g: Float64Array,
  history: { s: Float64Array; y: Float64Array; rho: number }[],
): Float64Array {
  const n = g.length
  const q = new Float64Array(g)
  const alphas = new Array<number>(history.length)

  for (let i = history.length - 1; i >= 0; i--) {
    const h = history[i]
    let sq = 0
    for (let j = 0; j < n; j++) sq += h.s[j] * q[j]
    alphas[i] = h.rho * sq
    for (let j = 0; j < n; j++) q[j] -= alphas[i] * h.y[j]
  }

  // Scaled identity H_0 = (s·y) / (y·y) · I
  let h0 = 1
  if (history.length > 0) {
    const last = history[history.length - 1]
    let yy = 0, sy = 0
    for (let j = 0; j < n; j++) {
      yy += last.y[j] * last.y[j]
      sy += last.s[j] * last.y[j]
    }
    if (yy > 0) h0 = sy / yy
  }

  const r = new Float64Array(n)
  for (let i = 0; i < n; i++) r[i] = h0 * q[i]
  for (let i = 0; i < history.length; i++) {
    const h = history[i]
    let yr = 0
    for (let j = 0; j < n; j++) yr += h.y[j] * r[j]
    const beta = h.rho * yr
    for (let j = 0; j < n; j++) r[j] += (alphas[i] - beta) * h.s[j]
  }

  // d = -H g
  for (let i = 0; i < n; i++) r[i] = -r[i]
  return r
}

function maxInfNorm(v: Float64Array): number {
  let m = 0
  for (let i = 0; i < v.length; i++) {
    const a = Math.abs(v[i])
    if (a > m) m = a
  }
  return m
}

async function runLBFGSStep(): Promise<boolean> {
  if (!state.model || !state.positions || !state.atomicNumbers || !state.lbfgs) return true

  const lb = state.lbfgs
  const n3 = state.positions.length
  const nAtoms = state.atomicNumbers.length

  // First step: get E, g at the current position.
  if (!lb.currentG) {
    const result = await predictAtPositions(state.positions)
    lb.currentE = result.energy
    lb.currentG = new Float64Array(n3)
    for (let i = 0; i < n3; i++) lb.currentG[i] = -result.forces[i]
    maskFrozen(lb.currentG)
  }

  // Convergence check on atomic forces.
  const forcesForCheck = new Float64Array(n3)
  for (let i = 0; i < n3; i++) forcesForCheck[i] = -lb.currentG[i]
  const maxF = calculateMaxForce(forcesForCheck, nAtoms)
  if (maxF < state.forceThreshold) {
    postOptStep(lb.currentE, lb.currentG!, lb.step, nAtoms, true)
    return true
  }

  // Give up after maxOptSteps iterations even if not converged.
  if (lb.step >= state.maxOptSteps) {
    postOptStep(lb.currentE, lb.currentG!, lb.step, nAtoms, false)
    return true
  }

  // Build search direction.
  let d = lbfgsDirection(lb.currentG, lb.history)

  // Safety: if not a descent direction, fall back to steepest descent and
  // discard the curvature history (it's lying to us).
  let dg = 0
  for (let i = 0; i < n3; i++) dg += d[i] * lb.currentG[i]
  if (dg >= 0) {
    d = new Float64Array(n3)
    for (let i = 0; i < n3; i++) d[i] = -lb.currentG[i]
    dg = 0
    for (let i = 0; i < n3; i++) dg += d[i] * lb.currentG[i]
    lb.history.length = 0
  }

  // Cap infinity-norm step size — prevents giant jumps early on when the
  // approximate Hessian is still a scaled identity.
  const dMax = maxInfNorm(d)
  if (dMax > LBFGS_MAX_STEP) {
    const scale = LBFGS_MAX_STEP / dMax
    for (let i = 0; i < n3; i++) d[i] *= scale
    dg *= scale
  }

  // Backtracking line search.
  let alpha = 1
  const trial = new Float64Array(n3)
  let newE = Infinity
  let newForces: ArrayLike<number> | null = null
  let accepted = false
  for (let ls = 0; ls < LBFGS_LS_MAX; ls++) {
    for (let i = 0; i < n3; i++) trial[i] = state.positions[i] + alpha * d[i]
    const r = await predictAtPositions(trial)
    newE = r.energy
    newForces = r.forces
    if (newE <= lb.currentE + LBFGS_ARMIJO * alpha * dg) {
      accepted = true
      break
    }
    alpha *= 0.5
  }
  // If the line search gave up, still accept the last trial — any move beats
  // stalling, and L-BFGS recovers well from imperfect steps as long as we
  // trash the history when it happens.
  if (!accepted) lb.history.length = 0

  if (!newForces) return true  // shouldn't happen; guards the TS narrowing

  const newG = new Float64Array(n3)
  for (let i = 0; i < n3; i++) newG[i] = -newForces[i]
  maskFrozen(newG)

  // Update curvature history (skip if s·y is tiny or negative — indicates
  // non-convexity in this neighbourhood).
  const s = new Float64Array(n3)
  const y = new Float64Array(n3)
  let sy = 0
  for (let i = 0; i < n3; i++) {
    s[i] = trial[i] - state.positions[i]
    y[i] = newG[i] - lb.currentG[i]
    sy += s[i] * y[i]
  }
  if (sy > 1e-12 && accepted) {
    lb.history.push({ s, y, rho: 1 / sy })
    if (lb.history.length > LBFGS_M) lb.history.shift()
  }

  // Commit the move.
  state.positions.set(trial)
  lb.currentE = newE
  lb.currentG = newG
  lb.step++
  state.optStep = lb.step

  postOptStep(lb.currentE, lb.currentG!, lb.step, nAtoms, false)
  return false
}

function postOptStep(
  energy: number,
  currentG: Float64Array,
  step: number,
  nAtoms: number,
  converged: boolean,
): void {
  if (!state.positions) return
  const forcesForReport = new Float64Array(currentG.length)
  for (let i = 0; i < currentG.length; i++) forcesForReport[i] = -currentG[i]
  const maxF = calculateMaxForce(forcesForReport, nAtoms)
  const posOut = new Float64Array(state.positions)
  const cellOut = state.cell ? new Float64Array(state.cell) : null
  const transfers: ArrayBuffer[] = [posOut.buffer]
  if (cellOut) transfers.push(cellOut.buffer)
  self.postMessage({
    type: 'optStep',
    positions: posOut,
    cell: cellOut,
    energy,
    maxForce: maxF,
    maxStress: 0,
    step,
    converged,
  }, transfers)
}

// ========== Polak–Ribière+ Conjugate Gradient ==========
//
// Builds a search direction d_{k+1} = -g_{k+1} + β · d_k with
//   β = max(0, (g_{k+1} - g_k) · g_{k+1} / (g_k · g_k))   (PR+ — automatic restart)
// Backtracking Armijo line search along d. Restart to steepest descent on
// non-descent direction or every 3·N steps. Cheaper per step than L-BFGS,
// often a better fit when you're close-ish to a minimum after manual edits.
const CG_MAX_STEP = 0.2  // Å; cap inf-norm of the trial step
const CG_LS_MAX = 5
const CG_ARMIJO = 1e-4

async function resetCG(): Promise<void> {
  state.cg = {
    currentE: 0,
    currentG: null,
    dir: null,
    step: 0,
    sinceRestart: 0,
  }
}

async function runCGStep(): Promise<boolean> {
  if (!state.model || !state.positions || !state.atomicNumbers || !state.cg) return true

  const cg = state.cg
  const n3 = state.positions.length
  const nAtoms = state.atomicNumbers.length

  // First step: get E, g at the current position; initial direction = -g.
  if (!cg.currentG) {
    const result = await predictAtPositions(state.positions)
    cg.currentE = result.energy
    cg.currentG = new Float64Array(n3)
    for (let i = 0; i < n3; i++) cg.currentG[i] = -result.forces[i]
    maskFrozen(cg.currentG)
    cg.dir = new Float64Array(n3)
    for (let i = 0; i < n3; i++) cg.dir[i] = -cg.currentG[i]
  }

  // Convergence check.
  const forcesForCheck = new Float64Array(n3)
  for (let i = 0; i < n3; i++) forcesForCheck[i] = -cg.currentG[i]
  const maxF = calculateMaxForce(forcesForCheck, nAtoms)
  if (maxF < state.forceThreshold) {
    postOptStep(cg.currentE, cg.currentG, cg.step, nAtoms, true)
    return true
  }

  if (cg.step >= state.maxOptSteps) {
    postOptStep(cg.currentE, cg.currentG, cg.step, nAtoms, false)
    return true
  }

  let d = cg.dir!

  // Periodic restart so accumulated direction noise doesn't pile up.
  if (cg.sinceRestart >= 3 * nAtoms) {
    d = new Float64Array(n3)
    for (let i = 0; i < n3; i++) d[i] = -cg.currentG[i]
    cg.dir = d
    cg.sinceRestart = 0
  }

  // Force a descent direction; if not, restart to steepest descent.
  let dg = 0
  for (let i = 0; i < n3; i++) dg += d[i] * cg.currentG[i]
  if (dg >= 0) {
    d = new Float64Array(n3)
    for (let i = 0; i < n3; i++) d[i] = -cg.currentG[i]
    cg.dir = d
    cg.sinceRestart = 0
    dg = 0
    for (let i = 0; i < n3; i++) dg += d[i] * cg.currentG[i]
  }

  // Cap the trial step's inf-norm.
  const dMax = maxInfNorm(d)
  let scale = 1
  if (dMax > CG_MAX_STEP) {
    scale = CG_MAX_STEP / dMax
  }

  // Armijo backtracking line search along d. Track the lowest-energy trial
  // separately so that if no alpha satisfies Armijo we can still salvage
  // a strict decrease — accepting an increase here is the classic source
  // of CG oscillation, since the next step then runs steepest descent from
  // the inflated energy and bounces back.
  let alpha = scale
  const trial = new Float64Array(n3)
  let newE = Infinity
  let newForces: ArrayLike<number> | null = null
  let bestE = Infinity
  let bestAlpha = 0
  let bestForces: ArrayLike<number> | null = null
  let accepted = false
  for (let ls = 0; ls < CG_LS_MAX; ls++) {
    for (let i = 0; i < n3; i++) trial[i] = state.positions[i] + alpha * d[i]
    const r = await predictAtPositions(trial)
    newE = r.energy
    newForces = r.forces
    if (newE < bestE) {
      bestE = newE
      bestAlpha = alpha
      bestForces = newForces
    }
    if (newE <= cg.currentE + CG_ARMIJO * alpha * dg) {
      accepted = true
      break
    }
    alpha *= 0.5
  }

  // Armijo never satisfied. Fall back to the lowest-energy alpha if it
  // actually beat the current point; otherwise refuse the step and restart
  // with steepest descent next iteration. Refusing prevents an uphill move
  // from poisoning the next direction.
  if (!accepted) {
    if (bestForces !== null && bestE < cg.currentE) {
      alpha = bestAlpha
      newE = bestE
      newForces = bestForces
      for (let i = 0; i < n3; i++) trial[i] = state.positions[i] + alpha * d[i]
    } else {
      const sd = new Float64Array(n3)
      for (let i = 0; i < n3; i++) sd[i] = -cg.currentG[i]
      cg.dir = sd
      cg.sinceRestart = 0
      cg.step++
      state.optStep = cg.step
      postOptStep(cg.currentE, cg.currentG, cg.step, nAtoms, false)
      return false
    }
  }

  if (!newForces) return true

  const newG = new Float64Array(n3)
  for (let i = 0; i < n3; i++) newG[i] = -newForces[i]
  maskFrozen(newG)

  // Polak–Ribière+ β.
  let gOldDot = 0
  let prDot = 0
  for (let i = 0; i < n3; i++) {
    gOldDot += cg.currentG[i] * cg.currentG[i]
    prDot += (newG[i] - cg.currentG[i]) * newG[i]
  }
  let beta = 0
  if (gOldDot > 1e-20) {
    beta = prDot / gOldDot
    if (beta < 0) beta = 0  // PR+ — automatic restart on non-monotone steps
  }
  if (!accepted) beta = 0  // line-search fallback — steepest descent next

  const nextDir = new Float64Array(n3)
  for (let i = 0; i < n3; i++) nextDir[i] = -newG[i] + beta * d[i]

  state.positions.set(trial)
  cg.currentE = newE
  cg.currentG = newG
  cg.dir = nextDir
  cg.step++
  cg.sinceRestart++
  state.optStep = cg.step

  postOptStep(cg.currentE, cg.currentG, cg.step, nAtoms, false)
  return false
}

// FIRE optimizer constants
const FIRE_ALPHA_START = 0.1
const FIRE_F_ALPHA = 0.99
const FIRE_F_INC = 1.1
const FIRE_F_DEC = 0.5
const FIRE_N_MIN = 5
const FIRE_DT_MAX = 1.0  // fs

// Cached forces/stress for FIRE optimization (like MD)
let fireForces: Float64Array | null = null
let fireStress: Float64Array | null = null
let fireCellForce: Float64Array | null = null

// Reset FIRE optimizer state and initialize velocities along force direction
async function resetFIRE(): Promise<void> {
  state.fireAlpha = FIRE_ALPHA_START
  state.fireNpos = 0
  state.fireDt = 0.1  // Start with small timestep
  state.optStep = 0

  // Clear cached forces/stress from previous optimization
  fireForces = null
  fireStress = null
  fireCellForce = null

  // Initialize cell velocities for periodic systems
  if (state.isPeriodic && state.optimizeCell && state.cell) {
    state.cellVelocities = new Float64Array(9)
  } else {
    state.cellVelocities = null
  }

  // Initialize velocities along force direction for faster startup
  if (state.module && state.model && state.positions && state.velocities && state.masses) {
    // Get initial forces
    const system = await state.module.AtomicSystem.create(
      state.positions,
      state.atomicNumbers!,
      state.cell,
      state.isPeriodic
    )
    const result = await state.model.predictWithOptions(system, true)
    const forces = new Float64Array(result.forces)

    // Calculate force magnitude
    let fNorm = 0
    for (let i = 0; i < forces.length; i++) {
      fNorm += forces[i] * forces[i]
    }
    fNorm = Math.sqrt(fNorm)

    // Set initial velocity along force direction with small magnitude
    // v = dt * F / |F| gives unit velocity in force direction scaled by timestep
    if (fNorm > 1e-10) {
      const vScale = state.fireDt * 0.1  // Small initial velocity
      for (let i = 0; i < state.velocities.length; i++) {
        state.velocities[i] = vScale * forces[i] / fNorm
      }
    } else {
      state.velocities.fill(0)
    }
  }
}

// Calculate max force magnitude
function calculateMaxForce(forces: Float64Array, numAtoms: number): number {
  let maxF = 0
  for (let i = 0; i < numAtoms; i++) {
    const fx = forces[i * 3]
    const fy = forces[i * 3 + 1]
    const fz = forces[i * 3 + 2]
    const f = Math.sqrt(fx * fx + fy * fy + fz * fz)
    if (f > maxF) maxF = f
  }
  return maxF
}

// Calculate max stress component (absolute value)
// Stress is in Voigt notation: [xx, yy, zz, yz, xz, xy]
function calculateMaxStress(stress: Float64Array): number {
  let maxS = 0
  for (let i = 0; i < 6; i++) {
    const s = Math.abs(stress[i])
    if (s > maxS) maxS = s
  }
  return maxS
}

// Convert Voigt stress to 3x3 tensor and compute cell gradient
// The cell gradient is: dE/dh = -V * stress * (h^-T) where h is the cell matrix
// For simplicity, we use: cell_force = -V * stress (works for orthogonal cells)
// Stress in eV/A^3, cell in A, so cell_force is in eV/A^2
function stressToCellForce(stress: Float64Array, _cell: Float64Array, volume: number): Float64Array {
  // Convert Voigt [xx, yy, zz, yz, xz, xy] to 3x3 symmetric tensor
  // Then multiply by -volume to get the "force" on the cell
  // For a general cell, the gradient is more complex, but this approximation works
  const cellForce = new Float64Array(9)

  // Diagonal components
  cellForce[0] = -volume * stress[0]  // xx -> h[0,0]
  cellForce[4] = -volume * stress[1]  // yy -> h[1,1]
  cellForce[8] = -volume * stress[2]  // zz -> h[2,2]

  // Off-diagonal (for non-orthogonal cells)
  cellForce[1] = -volume * stress[5]  // xy -> h[0,1]
  cellForce[3] = -volume * stress[5]  // xy -> h[1,0]
  cellForce[2] = -volume * stress[4]  // xz -> h[0,2]
  cellForce[6] = -volume * stress[4]  // xz -> h[2,0]
  cellForce[5] = -volume * stress[3]  // yz -> h[1,2]
  cellForce[7] = -volume * stress[3]  // yz -> h[2,1]

  return cellForce
}

// Calculate cell volume from 3x3 cell matrix (row-major: a, b, c as rows)
function calculateVolume(cell: Float64Array): number {
  // Volume = a · (b × c)
  const ax = cell[0], ay = cell[1], az = cell[2]
  const bx = cell[3], by = cell[4], bz = cell[5]
  const cx = cell[6], cy = cell[7], cz = cell[8]

  // b × c
  const bcx = by * cz - bz * cy
  const bcy = bz * cx - bx * cz
  const bcz = bx * cy - by * cx

  return Math.abs(ax * bcx + ay * bcy + az * bcz)
}

// Run one step of FIRE optimization
// FIRE: Fast Inertial Relaxation Engine
// Reference: Bitzek et al., PRL 97, 170201 (2006)
// Extended to optimize cell using stress tensor for periodic systems
// Uses cached forces for single prediction per step (like MD)
async function runFIREStep(): Promise<boolean> {
  if (!state.module || !state.model || !state.positions || !state.velocities || !state.masses) {
    return true  // converged = done
  }

  const optimizingCell = state.isPeriodic && state.optimizeCell && state.cell && state.cellVelocities

  try {
    state.optStep++

    // If no cached forces, compute initial forces
    if (!fireForces) {
      state.system = await state.module.AtomicSystem.create(
        state.positions,
        state.atomicNumbers!,
        state.cell,
        state.isPeriodic
      )
      const result = await state.model.predictWithOptions(state.system, true)
      fireForces = new Float64Array(result.forces)
      maskFrozen(fireForces)
      fireStress = result.stress ? new Float64Array(result.stress) : null
      if (optimizingCell && fireStress && state.cell) {
        const volume = calculateVolume(state.cell)
        fireCellForce = stressToCellForce(fireStress, state.cell, volume)
      }
    }

    const forces = fireForces
    const stress = fireStress
    const cellForce = fireCellForce

    // Calculate max force and stress
    const maxForce = calculateMaxForce(forces, state.numAtoms)
    const maxStress = (optimizingCell && stress) ? calculateMaxStress(stress) : 0

    // Check convergence (both force and stress must be below threshold)
    const forceConverged = maxForce < state.forceThreshold
    const stressConverged = !optimizingCell || maxStress < state.stressThreshold
    const converged = forceConverged && stressConverged

    if (converged || state.optStep >= state.maxOptSteps) {
      // Get final energy
      state.system = await state.module.AtomicSystem.create(
        state.positions,
        state.atomicNumbers!,
        state.cell,
        state.isPeriodic
      )
      const result = await state.model.predictWithOptions(state.system, true)

      const posOut = new Float64Array(state.positions)
      const cellOut = state.cell ? new Float64Array(state.cell) : null
      const transfers: ArrayBuffer[] = [posOut.buffer]
      if (cellOut) transfers.push(cellOut.buffer)
      self.postMessage({
        type: 'optStep',
        positions: posOut,
        cell: cellOut,
        energy: result.energy,
        maxForce,
        maxStress,
        step: state.optStep,
        converged,
      }, transfers)
      return true  // Done
    }

    // FIRE algorithm
    // 1. Calculate P = F·v (power) - include cell DOFs
    let power = 0
    let vNorm = 0
    let fNorm = 0

    // Atomic DOFs
    for (let i = 0; i < state.numAtoms * 3; i++) {
      power += forces[i] * state.velocities[i]
      vNorm += state.velocities[i] * state.velocities[i]
      fNorm += forces[i] * forces[i]
    }

    // Cell DOFs (scaled by cell mass factor for balanced optimization)
    const cellMassFactor = 1.0
    if (optimizingCell && cellForce && state.cellVelocities) {
      for (let i = 0; i < 9; i++) {
        power += cellForce[i] * state.cellVelocities[i] * cellMassFactor
        vNorm += state.cellVelocities[i] * state.cellVelocities[i] * cellMassFactor
        fNorm += cellForce[i] * cellForce[i] * cellMassFactor
      }
    }

    vNorm = Math.sqrt(vNorm)
    fNorm = Math.sqrt(fNorm)

    // 2. Adjust velocity: v = (1-α)v + α|v|F̂
    if (fNorm > 1e-10) {
      // Atomic velocities
      for (let i = 0; i < state.numAtoms * 3; i++) {
        state.velocities[i] = (1 - state.fireAlpha) * state.velocities[i] +
          state.fireAlpha * vNorm * forces[i] / fNorm
      }
      // Cell velocities
      if (optimizingCell && cellForce && state.cellVelocities) {
        for (let i = 0; i < 9; i++) {
          state.cellVelocities[i] = (1 - state.fireAlpha) * state.cellVelocities[i] +
            state.fireAlpha * vNorm * cellForce[i] / fNorm
        }
      }
    }

    // 3. Check if going downhill (P > 0)
    if (power > 0) {
      state.fireNpos++
      if (state.fireNpos > FIRE_N_MIN) {
        // Increase timestep and decrease alpha
        state.fireDt = Math.min(state.fireDt * FIRE_F_INC, FIRE_DT_MAX)
        state.fireAlpha *= FIRE_F_ALPHA
      }
    } else {
      // Going uphill - reset
      state.fireNpos = 0
      state.fireDt *= FIRE_F_DEC
      state.fireAlpha = FIRE_ALPHA_START
      // Zero velocities
      state.velocities.fill(0)
      if (state.cellVelocities) {
        state.cellVelocities.fill(0)
      }
    }

    // 4. Velocity Verlet with cached forces
    // Use old forces for position update: r += v*dt + 0.5*a*dt^2
    for (let i = 0; i < state.numAtoms; i++) {
      const mass = state.masses[i]
      const accelFactor = EV_A_AMU_TO_A_FS2 / mass
      for (let j = 0; j < 3; j++) {
        const idx = i * 3 + j
        const accel = forces[idx] * accelFactor
        state.positions[idx] += state.velocities[idx] * state.fireDt + 0.5 * accel * state.fireDt * state.fireDt
      }
    }

    // Update cell if optimizing
    const cellAccelFactor = 1e-4
    if (optimizingCell && cellForce && state.cellVelocities && state.cell) {
      for (let i = 0; i < 9; i++) {
        const accel = cellForce[i] * cellAccelFactor
        state.cell[i] += state.cellVelocities[i] * state.fireDt + 0.5 * accel * state.fireDt * state.fireDt
      }
    }

    // Get new forces (single prediction per step)
    state.system = await state.module.AtomicSystem.create(
      state.positions,
      state.atomicNumbers!,
      state.cell,
      state.isPeriodic
    )
    const resultNew = await state.model.predictWithOptions(state.system, true)
    const forcesNew = new Float64Array(resultNew.forces)
    const stressNew = resultNew.stress ? new Float64Array(resultNew.stress) : null

    // Update velocities using average of old and new forces
    for (let i = 0; i < state.numAtoms; i++) {
      const mass = state.masses[i]
      const accelFactor = EV_A_AMU_TO_A_FS2 / mass
      for (let j = 0; j < 3; j++) {
        const idx = i * 3 + j
        const accelOld = forces[idx] * accelFactor
        const accelNew = forcesNew[idx] * accelFactor
        state.velocities[idx] += 0.5 * (accelOld + accelNew) * state.fireDt
      }
    }

    // Update cell velocities
    if (optimizingCell && stressNew && state.cellVelocities && state.cell) {
      const volume = calculateVolume(state.cell)
      const cellForceNew = stressToCellForce(stressNew, state.cell, volume)
      for (let i = 0; i < 9; i++) {
        const accelOld = (cellForce ? cellForce[i] : 0) * cellAccelFactor
        const accelNew = cellForceNew[i] * cellAccelFactor
        state.cellVelocities[i] += 0.5 * (accelOld + accelNew) * state.fireDt
      }
      // Cache new cell force
      fireCellForce = cellForceNew
    }

    // Cache new forces for next step
    maskFrozen(forcesNew)
    fireForces = forcesNew
    fireStress = stressNew

    const maxForceNew = calculateMaxForce(forcesNew, state.numAtoms)
    const maxStressNew = (optimizingCell && stressNew) ? calculateMaxStress(stressNew) : 0

    const posOut = new Float64Array(state.positions)
    const cellOut = state.cell ? new Float64Array(state.cell) : null
    const transfers: ArrayBuffer[] = [posOut.buffer]
    if (cellOut) transfers.push(cellOut.buffer)
    self.postMessage({
      type: 'optStep',
      positions: posOut,
      cell: cellOut,
      energy: resultNew.energy,
      maxForce: maxForceNew,
      maxStress: maxStressNew,
      step: state.optStep,
      converged: false,
    }, transfers)

    return false  // Not done yet
  } catch (err: any) {
    handleStop()
    self.postMessage({ type: 'error', message: `Optimization step failed: ${err.message}` })
    return true
  }
}

async function runMDStep(): Promise<void> {
  if (!state.module || !state.model || !state.positions || !state.velocities || !state.masses) {
    return
  }

  try {
    const t0 = performance.now()

    const useNCForces = !state.useConservativeForces

    // If we don't have cached forces, compute them first
    if (!state.forces) {
      state.system = await state.module.AtomicSystem.create(
        state.positions,
        state.atomicNumbers!,
        state.cell,
        state.isPeriodic
      )
      const result = await state.model.predictWithOptions(state.system, useNCForces)
      state.forces = new Float64Array(result.forces)
    }

    const forcesOld = state.forces

    // Velocity Verlet step 1: update positions using current forces
    // r(t+dt) = r(t) + v(t)*dt + 0.5*a(t)*dt^2
    for (let i = 0; i < state.numAtoms; i++) {
      const mass = state.masses[i]
      const accelFactor = EV_A_AMU_TO_A_FS2 / mass
      for (let j = 0; j < 3; j++) {
        const idx = i * 3 + j
        const accel = forcesOld[idx] * accelFactor
        state.positions[idx] += state.velocities[idx] * state.dt + 0.5 * accel * state.dt * state.dt
      }
    }

    const t1 = performance.now()

    // Get forces at new positions (single prediction per step)
    state.system = await state.module.AtomicSystem.create(
      state.positions,
      state.atomicNumbers!,
      state.cell,
      state.isPeriodic
    )
    const t2 = performance.now()

    const result = await state.model.predictWithOptions(state.system, useNCForces)
    const t3 = performance.now()

    const forcesNew = new Float64Array(result.forces)

    // Velocity Verlet step 2: update velocities using average of old and new forces
    // v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
    for (let i = 0; i < state.numAtoms; i++) {
      const mass = state.masses[i]
      const accelFactor = EV_A_AMU_TO_A_FS2 / mass
      for (let j = 0; j < 3; j++) {
        const idx = i * 3 + j
        const accelOld = forcesOld[idx] * accelFactor
        const accelNew = forcesNew[idx] * accelFactor
        state.velocities[idx] += 0.5 * (accelOld + accelNew) * state.dt
      }
    }

    // Cache forces for next step
    state.forces = forcesNew

    // Remove center of mass motion to prevent drift
    removeCOMVelocity(state.velocities, state.masses, state.numAtoms)

    // Apply thermostat (skipped in NVE mode)
    if (state.thermostat === 'csvr') {
      csvrThermostat(
        state.velocities, state.masses, state.numAtoms,
        state.temperature, state.thermostatTau, state.dt
      )
    }

    // Calculate final KE/T after (optional) thermostat
    const { ke: keNew, temp: tempNew } = calculateKineticEnergy(state.velocities, state.masses, state.numAtoms)

    // NVE drift diagnostic: track total energy relative to first step.
    // Only meaningful with conservative forces + no thermostat. We still report
    // it in other modes so users can see what it's doing.
    const totalE = result.energy + keNew
    if (state.initialTotalEnergy === null) {
      state.initialTotalEnergy = totalE
    }
    const energyDrift = totalE - state.initialTotalEnergy

    const t4 = performance.now()

    // Send update with timing info — transfer typed-array buffers zero-copy.
    // state.forces keeps a copy so the next step can reuse cached forces.
    const posOut = new Float64Array(state.positions)
    const forcesOut = new Float64Array(forcesNew)
    self.postMessage({
      type: 'mdStep',
      positions: posOut,
      energy: result.energy,
      kineticEnergy: keNew,
      temperature: tempNew,
      energyDrift,
      forces: forcesOut,
      timing: {
        verlet1: t1 - t0,
        systemCreate: t2 - t1,
        predict: t3 - t2,
        verlet2: t4 - t3,
        total: t4 - t0,
      },
    }, [posOut.buffer, forcesOut.buffer])
  } catch (err: any) {
    handleStop()
    self.postMessage({ type: 'error', message: `MD step failed: ${err.message}` })
  }
}

// Apply random perturbations to positions
function rattlePositions(amount: number): void {
  if (!state.positions || amount <= 0) return

  for (let i = 0; i < state.positions.length; i++) {
    // Uniform random in [-amount, +amount]
    state.positions[i] += (Math.random() * 2 - 1) * amount
  }
}

async function handleStart(data: { stepsPerFrame?: number, mode?: 'md' | 'optimize', rattleAmount?: number, frozen?: number[] }): Promise<void> {
  if (state.isRunning) return

  // Update mode if provided
  if (data.mode !== undefined) {
    state.mode = data.mode
  }

  // Apply optimization constraints (atoms whose forces will be zeroed during
  // this run). Cleared automatically on the next setSystem.
  state.frozenAtoms = (data.frozen && data.frozen.length > 0) ? new Set(data.frozen) : null

  state.isRunning = true

  if (state.mode === 'optimize') {
    // Cell optimization (periodic + optimizeCell) always uses FIRE — cell
    // dynamics are coupled to atoms and easier to reason about with a
    // velocity-based scheme. Otherwise respect the user's pick.
    const forceFIRE = state.isPeriodic && state.optimizeCell
    const chosen: 'lbfgs' | 'fire' | 'cg' = forceFIRE ? 'fire' : state.optimizer
    self.postMessage({
      type: 'optimizerStarted',
      optimizer: chosen,
      forced: forceFIRE,
    })

    if (chosen === 'lbfgs')      await resetLBFGS()
    else if (chosen === 'cg')    await resetCG()
    else                         await resetFIRE()

    // Apply rattle if requested
    if (data.rattleAmount && data.rattleAmount > 0) {
      rattlePositions(data.rattleAmount)
    }

    // Run optimization steps as fast as possible
    const stepFn =
      chosen === 'lbfgs' ? runLBFGSStep :
      chosen === 'cg'    ? runCGStep    :
                           runFIREStep
    const runOptLoop = async () => {
      if (!state.isRunning) return
      const done = await stepFn()
      if (done) {
        handleStop()
      } else {
        mdTimeout = setTimeout(runOptLoop, 0)
      }
    }
    mdTimeout = setTimeout(runOptLoop, 0)
  } else {
    // MD mode
    const stepsPerFrame = data.stepsPerFrame || 1

    // Run MD steps as fast as possible
    const runMDLoop = async () => {
      if (!state.isRunning) return
      for (let i = 0; i < stepsPerFrame; i++) {
        await runMDStep()
      }
      mdTimeout = setTimeout(runMDLoop, 0)
    }
    mdTimeout = setTimeout(runMDLoop, 0)
  }

  self.postMessage({ type: 'started' })
}

function handleStop(): void {
  if (mdTimeout) {
    clearTimeout(mdTimeout)
    mdTimeout = null
  }
  state.isRunning = false
  self.postMessage({ type: 'stopped' })
}

async function handleStep(): Promise<void> {
  await runMDStep()
}

function handleRattle(data: { amount: number }): void {
  if (!state.positions) {
    self.postMessage({ type: 'error', message: 'No system loaded' })
    return
  }

  rattlePositions(data.amount)

  // Send back the new positions so visualization can update
  const posOut = new Float64Array(state.positions)
  self.postMessage({
    type: 'rattled',
    positions: posOut,
  }, [posOut.buffer])
}

// Message router
self.onmessage = async (e: MessageEvent) => {
  const { type, ...data } = e.data

  switch (type) {
    case 'init':
      await handleInit(data)
      break
    case 'loadModel':
      await handleLoadModel(data)
      break
    case 'setBackend':
      await handleSetBackend(data)
      break
    case 'setSystem':
      await handleSetSystem(data)
      break
    case 'predict':
      await handlePredict()
      break
    case 'setParameters':
      handleSetParameters(data)
      break
    case 'start':
      await handleStart(data)
      break
    case 'stop':
      handleStop()
      break
    case 'step':
      await handleStep()
      break
    case 'rattle':
      handleRattle(data)
      break
    case 'predictAt':
      await handlePredictAt(data)
      break
    default:
      self.postMessage({ type: 'error', message: `Unknown message type: ${type}` })
  }
}

// Signal that worker is ready
self.postMessage({ type: 'ready' })
