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

    // Initialize velocities and clear all cached forces/state
    state.velocities = initializeVelocities(state.numAtoms, state.masses, state.temperature)
    state.forces = null

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

let mdTimeout: ReturnType<typeof setTimeout> | null = null

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
function resetFIRE(): void {
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
    const system = state.module.AtomicSystem.create(
      state.positions,
      state.atomicNumbers!,
      state.cell,
      state.isPeriodic
    )
    const result = state.model.predictWithOptions(system, true)
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
function stressToCellForce(stress: Float64Array, cell: Float64Array, volume: number): Float64Array {
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
function runFIREStep(): boolean {
  if (!state.module || !state.model || !state.positions || !state.velocities || !state.masses) {
    return true  // converged = done
  }

  const optimizingCell = state.isPeriodic && state.optimizeCell && state.cell && state.cellVelocities

  try {
    state.optStep++

    // If no cached forces, compute initial forces
    if (!fireForces) {
      state.system = state.module.AtomicSystem.create(
        state.positions,
        state.atomicNumbers!,
        state.cell,
        state.isPeriodic
      )
      const result = state.model.predictWithOptions(state.system, true)
      fireForces = new Float64Array(result.forces)
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
      state.system = state.module.AtomicSystem.create(
        state.positions,
        state.atomicNumbers!,
        state.cell,
        state.isPeriodic
      )
      const result = state.model.predictWithOptions(state.system, true)

      self.postMessage({
        type: 'optStep',
        positions: Array.from(state.positions),
        cell: state.cell ? Array.from(state.cell) : null,
        energy: result.energy,
        maxForce,
        maxStress,
        step: state.optStep,
        converged,
      })
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
    state.system = state.module.AtomicSystem.create(
      state.positions,
      state.atomicNumbers!,
      state.cell,
      state.isPeriodic
    )
    const resultNew = state.model.predictWithOptions(state.system, true)
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
    fireForces = forcesNew
    fireStress = stressNew

    const maxForceNew = calculateMaxForce(forcesNew, state.numAtoms)
    const maxStressNew = (optimizingCell && stressNew) ? calculateMaxStress(stressNew) : 0

    // Send update
    self.postMessage({
      type: 'optStep',
      positions: Array.from(state.positions),
      cell: state.cell ? Array.from(state.cell) : null,
      energy: resultNew.energy,
      maxForce: maxForceNew,
      maxStress: maxStressNew,
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
    const t0 = performance.now()

    // If we don't have cached forces, compute them first
    if (!state.forces) {
      state.system = state.module.AtomicSystem.create(
        state.positions,
        state.atomicNumbers!,
        state.cell,
        state.isPeriodic
      )
      const result = state.model.predictWithOptions(state.system, true)
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
    state.system = state.module.AtomicSystem.create(
      state.positions,
      state.atomicNumbers!,
      state.cell,
      state.isPeriodic
    )
    const t2 = performance.now()

    const result = state.model.predictWithOptions(state.system, true)
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

    // Apply thermostat (tau = 100 fs is a reasonable coupling time)
    const { temp } = calculateKineticEnergy(state.velocities, state.masses, state.numAtoms)
    berendsenThermostat(state.velocities, temp, state.temperature, 100, state.dt)

    // Calculate updated temperature after thermostat
    const { ke: keNew, temp: tempNew } = calculateKineticEnergy(state.velocities, state.masses, state.numAtoms)
    const t4 = performance.now()

    // Send update with timing info
    self.postMessage({
      type: 'mdStep',
      positions: Array.from(state.positions),
      energy: result.energy,
      kineticEnergy: keNew,
      temperature: tempNew,
      forces: Array.from(forcesNew),
      timing: {
        verlet1: t1 - t0,
        systemCreate: t2 - t1,
        predict: t3 - t2,
        verlet2: t4 - t3,
        total: t4 - t0,
      },
    })
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

function handleStart(data: { stepsPerFrame?: number, mode?: 'md' | 'optimize', rattleAmount?: number }): void {
  if (state.isRunning) return

  // Update mode if provided
  if (data.mode !== undefined) {
    state.mode = data.mode
  }

  state.isRunning = true

  if (state.mode === 'optimize') {
    // Reset FIRE state for new optimization
    resetFIRE()

    // Apply rattle if requested
    if (data.rattleAmount && data.rattleAmount > 0) {
      rattlePositions(data.rattleAmount)
    }

    // Run optimization steps as fast as possible
    const runOptLoop = () => {
      if (!state.isRunning) return
      const done = runFIREStep()
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
    const runMDLoop = () => {
      if (!state.isRunning) return
      for (let i = 0; i < stepsPerFrame; i++) {
        runMDStep()
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

function handleStep(): void {
  runMDStep()
}

function handleRattle(data: { amount: number }): void {
  if (!state.positions) {
    self.postMessage({ type: 'error', message: 'No system loaded' })
    return
  }

  rattlePositions(data.amount)

  // Send back the new positions so visualization can update
  self.postMessage({
    type: 'rattled',
    positions: Array.from(state.positions),
  })
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
    case 'rattle':
      handleRattle(data)
      break
    default:
      self.postMessage({ type: 'error', message: `Unknown message type: ${type}` })
  }
}

// Signal that worker is ready
self.postMessage({ type: 'ready' })
