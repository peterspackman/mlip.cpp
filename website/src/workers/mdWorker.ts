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

import createMlipcpp, { MlipcppModule, Model, AtomicSystem } from '@peterspackman/mlip.js'

interface WorkerState {
  module: MlipcppModule | null
  model: Model | null
  system: AtomicSystem | null
  positions: Float64Array | null
  velocities: Float64Array | null
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
  mode: 'md' | 'optimize'
  maxOptSteps: number
  forceThreshold: number
  optStep: number
}

const state: WorkerState = {
  module: null,
  model: null,
  system: null,
  positions: null,
  velocities: null,
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
  mode: 'md',
  maxOptSteps: 100,
  forceThreshold: 0.05,  // eV/A
  optStep: 0,
}

// Atomic masses in amu
const ATOMIC_MASSES: Record<number, number> = {
  1: 1.008,   // H
  6: 12.011,  // C
  7: 14.007,  // N
  8: 15.999,  // O
  9: 18.998,  // F
  12: 24.305, // Mg
  14: 28.085, // Si
  15: 30.974, // P
  16: 32.065, // S
  17: 35.453, // Cl
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

// Velocity Verlet position update
function velocityVerletPositions(
  positions: Float64Array,
  velocities: Float64Array,
  forces: Float64Array,
  masses: Float64Array,
  numAtoms: number,
  dt: number
): void {
  // r(t+dt) = r(t) + v(t)*dt + 0.5*a(t)*dt^2
  // a = F/m with conversion factor
  for (let i = 0; i < numAtoms; i++) {
    const mass = masses[i]
    const accelFactor = EV_A_AMU_TO_A_FS2 / mass
    for (let j = 0; j < 3; j++) {
      const idx = i * 3 + j
      const accel = forces[idx] * accelFactor  // A/fs^2
      positions[idx] += velocities[idx] * dt + 0.5 * accel * dt * dt
    }
  }
}

// Velocity Verlet velocity update
function velocityVerletVelocities(
  velocities: Float64Array,
  forcesOld: Float64Array,
  forcesNew: Float64Array,
  masses: Float64Array,
  numAtoms: number,
  dt: number
): void {
  // v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
  for (let i = 0; i < numAtoms; i++) {
    const mass = masses[i]
    const accelFactor = EV_A_AMU_TO_A_FS2 / mass
    for (let j = 0; j < 3; j++) {
      const idx = i * 3 + j
      const accelOld = forcesOld[idx] * accelFactor
      const accelNew = forcesNew[idx] * accelFactor
      velocities[idx] += 0.5 * (accelOld + accelNew) * dt
    }
  }
}

// Berendsen thermostat velocity scaling
function berendsenThermostat(
  velocities: Float64Array,
  currentTemp: number,
  targetTemp: number,
  tau: number,
  dt: number
): void {
  if (currentTemp < 1e-10) return
  const lambda = Math.sqrt(1 + (dt / tau) * (targetTemp / currentTemp - 1))
  for (let i = 0; i < velocities.length; i++) {
    velocities[i] *= lambda
  }
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

// Message handlers
async function handleInit(data: { modelBuffer?: ArrayBuffer }): Promise<void> {
  try {
    state.module = await createMlipcpp()

    if (data.modelBuffer) {
      state.model = state.module.Model.loadFromBuffer(data.modelBuffer)
    }

    self.postMessage({ type: 'initialized', version: state.module.getVersion() })
  } catch (err: any) {
    self.postMessage({ type: 'error', message: `Initialization failed: ${err.message}` })
  }
}

async function handleLoadModel(data: { buffer: ArrayBuffer }): Promise<void> {
  if (!state.module) {
    self.postMessage({ type: 'error', message: 'Module not initialized' })
    return
  }

  try {
    state.model = state.module.Model.loadFromBuffer(data.buffer)
    self.postMessage({
      type: 'modelLoaded',
      modelType: state.model.modelType(),
      cutoff: state.model.cutoff(),
    })
  } catch (err: any) {
    self.postMessage({ type: 'error', message: `Failed to load model: ${err.message}` })
  }
}

function handleSetSystem(data: { xyz: string }): void {
  if (!state.module) {
    self.postMessage({ type: 'error', message: 'Module not initialized' })
    return
  }

  try {
    state.system = state.module.AtomicSystem.fromXyzString(data.xyz)
    state.numAtoms = state.system.numAtoms()
    state.isPeriodic = state.system.isPeriodic()
    state.positions = new Float64Array(state.system.getPositions())
    state.atomicNumbers = new Int32Array(state.system.getAtomicNumbers())
    state.cell = state.system.getCell() ? new Float64Array(state.system.getCell()!) : null

    // Set up masses
    state.masses = new Float64Array(state.numAtoms)
    for (let i = 0; i < state.numAtoms; i++) {
      const z = state.atomicNumbers[i]
      state.masses[i] = ATOMIC_MASSES[z] || 12.0  // Default to carbon mass
    }

    // Initialize velocities
    state.velocities = initializeVelocities(state.numAtoms, state.masses, state.temperature)

    self.postMessage({
      type: 'systemSet',
      numAtoms: state.numAtoms,
      isPeriodic: state.isPeriodic,
    })
  } catch (err: any) {
    self.postMessage({ type: 'error', message: `Failed to set system: ${err.message}` })
  }
}

function handlePredict(): void {
  if (!state.module || !state.model || !state.system) {
    self.postMessage({ type: 'error', message: 'System or model not ready' })
    return
  }

  try {
    // Use NC forces for faster prediction (non-conservative forces from forward pass)
    const result = state.model.predictWithOptions(state.system, true)
    self.postMessage({
      type: 'prediction',
      energy: result.energy,
      forces: Array.from(result.forces),
    })
  } catch (err: any) {
    self.postMessage({ type: 'error', message: `Prediction failed: ${err.message}` })
  }
}

function handleSetParameters(data: {
  dt?: number,
  temperature?: number,
  mode?: 'md' | 'optimize',
  maxOptSteps?: number,
  forceThreshold?: number
}): void {
  if (data.dt !== undefined) state.dt = data.dt
  if (data.temperature !== undefined) state.temperature = data.temperature
  if (data.mode !== undefined) state.mode = data.mode
  if (data.maxOptSteps !== undefined) state.maxOptSteps = data.maxOptSteps
  if (data.forceThreshold !== undefined) state.forceThreshold = data.forceThreshold
  self.postMessage({ type: 'parametersSet', dt: state.dt, temperature: state.temperature })
}

let mdInterval: ReturnType<typeof setInterval> | null = null

// FIRE optimizer constants
const FIRE_ALPHA_START = 0.1
const FIRE_F_ALPHA = 0.99
const FIRE_F_INC = 1.1
const FIRE_F_DEC = 0.5
const FIRE_N_MIN = 5
const FIRE_DT_MAX = 1.0  // fs

// Reset FIRE optimizer state
function resetFIRE(): void {
  state.fireAlpha = FIRE_ALPHA_START
  state.fireNpos = 0
  state.fireDt = 0.1  // Start with small timestep
  state.optStep = 0
  // Zero velocities
  if (state.velocities) {
    state.velocities.fill(0)
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

// Run one step of FIRE optimization
// FIRE: Fast Inertial Relaxation Engine
// Reference: Bitzek et al., PRL 97, 170201 (2006)
function runFIREStep(): boolean {
  if (!state.module || !state.model || !state.positions || !state.velocities || !state.masses) {
    return true  // converged = done
  }

  try {
    state.optStep++

    // Create system with current positions
    state.system = state.module.AtomicSystem.create(
      state.positions,
      state.atomicNumbers!,
      state.cell,
      state.isPeriodic
    )

    // Get forces (use gradient forces for optimization, not NC forces)
    const result = state.model.predictWithOptions(state.system, false)
    const forces = new Float64Array(result.forces)

    // Calculate max force
    const maxForce = calculateMaxForce(forces, state.numAtoms)

    // Check convergence
    if (maxForce < state.forceThreshold || state.optStep >= state.maxOptSteps) {
      self.postMessage({
        type: 'optStep',
        positions: Array.from(state.positions),
        energy: result.energy,
        maxForce,
        step: state.optStep,
        converged: maxForce < state.forceThreshold,
      })
      return true  // Done
    }

    // FIRE algorithm
    // 1. Calculate P = F·v (power)
    let power = 0
    let vNorm = 0
    let fNorm = 0
    for (let i = 0; i < state.numAtoms * 3; i++) {
      power += forces[i] * state.velocities[i]
      vNorm += state.velocities[i] * state.velocities[i]
      fNorm += forces[i] * forces[i]
    }
    vNorm = Math.sqrt(vNorm)
    fNorm = Math.sqrt(fNorm)

    // 2. Adjust velocity: v = (1-α)v + α|v|F̂
    if (fNorm > 1e-10) {
      for (let i = 0; i < state.numAtoms * 3; i++) {
        state.velocities[i] = (1 - state.fireAlpha) * state.velocities[i] +
          state.fireAlpha * vNorm * forces[i] / fNorm
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
    }

    // 4. Euler integration with velocity Verlet-like update
    // Update velocities: v += F/m * dt
    // Update positions: r += v * dt
    for (let i = 0; i < state.numAtoms; i++) {
      const mass = state.masses[i]
      const accelFactor = EV_A_AMU_TO_A_FS2 / mass
      for (let j = 0; j < 3; j++) {
        const idx = i * 3 + j
        // Half-step velocity update
        state.velocities[idx] += 0.5 * forces[idx] * accelFactor * state.fireDt
        // Position update
        state.positions[idx] += state.velocities[idx] * state.fireDt
      }
    }

    // Get new forces for second velocity update
    state.system = state.module.AtomicSystem.create(
      state.positions,
      state.atomicNumbers!,
      state.cell,
      state.isPeriodic
    )
    const resultNew = state.model.predictWithOptions(state.system, false)
    const forcesNew = new Float64Array(resultNew.forces)

    // Second half of velocity update
    for (let i = 0; i < state.numAtoms; i++) {
      const mass = state.masses[i]
      const accelFactor = EV_A_AMU_TO_A_FS2 / mass
      for (let j = 0; j < 3; j++) {
        const idx = i * 3 + j
        state.velocities[idx] += 0.5 * forcesNew[idx] * accelFactor * state.fireDt
      }
    }

    const maxForceNew = calculateMaxForce(forcesNew, state.numAtoms)

    // Send update
    self.postMessage({
      type: 'optStep',
      positions: Array.from(state.positions),
      energy: resultNew.energy,
      maxForce: maxForceNew,
      step: state.optStep,
      converged: false,
    })

    return false  // Not done yet
  } catch (err: any) {
    handleStop()
    self.postMessage({ type: 'error', message: `Optimization step failed: ${err.message}` })
    return true
  }
}

function runMDStep(): void {
  if (!state.module || !state.model || !state.positions || !state.velocities || !state.masses) {
    return
  }

  try {
    // Create system with current positions
    state.system = state.module.AtomicSystem.create(
      state.positions,
      state.atomicNumbers!,
      state.cell,
      state.isPeriodic
    )

    // Get forces at current positions (use NC forces for faster MD)
    const result = state.model.predictWithOptions(state.system, true)
    const forcesOld = new Float64Array(result.forces)

    // Update positions (first half of Verlet)
    velocityVerletPositions(state.positions, state.velocities, forcesOld, state.masses, state.numAtoms, state.dt)

    // Create system with new positions for new forces
    state.system = state.module.AtomicSystem.create(
      state.positions,
      state.atomicNumbers!,
      state.cell,
      state.isPeriodic
    )

    // Get forces at new positions (use NC forces for faster MD)
    const resultNew = state.model.predictWithOptions(state.system, true)
    const forcesNew = new Float64Array(resultNew.forces)

    // Update velocities (second half of Verlet)
    velocityVerletVelocities(state.velocities, forcesOld, forcesNew, state.masses, state.numAtoms, state.dt)

    // Remove center of mass motion to prevent drift
    removeCOMVelocity(state.velocities, state.masses, state.numAtoms)

    // Apply thermostat (tau = 100 fs is a reasonable coupling time)
    const { temp } = calculateKineticEnergy(state.velocities, state.masses, state.numAtoms)
    berendsenThermostat(state.velocities, temp, state.temperature, 100, state.dt)

    // Calculate updated temperature after thermostat
    const { ke: keNew, temp: tempNew } = calculateKineticEnergy(state.velocities, state.masses, state.numAtoms)

    // Send update
    self.postMessage({
      type: 'mdStep',
      positions: Array.from(state.positions),
      energy: resultNew.energy,
      kineticEnergy: keNew,
      temperature: tempNew,
      forces: Array.from(forcesNew),
    })
  } catch (err: any) {
    handleStop()
    self.postMessage({ type: 'error', message: `MD step failed: ${err.message}` })
  }
}

function handleStart(data: { stepsPerFrame?: number, mode?: 'md' | 'optimize' }): void {
  if (state.isRunning) return

  // Update mode if provided
  if (data.mode !== undefined) {
    state.mode = data.mode
  }

  state.isRunning = true

  if (state.mode === 'optimize') {
    // Reset FIRE state for new optimization
    resetFIRE()

    // Run optimization steps at ~30 fps
    mdInterval = setInterval(() => {
      const done = runFIREStep()
      if (done) {
        handleStop()
      }
    }, 33)
  } else {
    // MD mode
    const stepsPerFrame = data.stepsPerFrame || 1

    // Run MD steps at ~30 fps
    mdInterval = setInterval(() => {
      for (let i = 0; i < stepsPerFrame; i++) {
        runMDStep()
      }
    }, 33)
  }

  self.postMessage({ type: 'started' })
}

function handleStop(): void {
  if (mdInterval) {
    clearInterval(mdInterval)
    mdInterval = null
  }
  state.isRunning = false
  self.postMessage({ type: 'stopped' })
}

function handleStep(): void {
  runMDStep()
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
    case 'setSystem':
      handleSetSystem(data)
      break
    case 'predict':
      handlePredict()
      break
    case 'setParameters':
      handleSetParameters(data)
      break
    case 'start':
      handleStart(data)
      break
    case 'stop':
      handleStop()
      break
    case 'step':
      handleStep()
      break
    default:
      self.postMessage({ type: 'error', message: `Unknown message type: ${type}` })
  }
}

// Signal that worker is ready
self.postMessage({ type: 'ready' })
