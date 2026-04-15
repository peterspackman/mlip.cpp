// Finite-difference Hessian + diagonalization → normal modes.
//
// Units and conventions:
//   - Positions in Å, forces in eV/Å, masses in amu.
//   - Hessian H[i,j] = ∂²V/∂x_i ∂x_j, computed as -(∂F_j/∂x_i) via central FD.
//   - Mass-weighted: D[i,j] = H[i,j] / sqrt(m_i m_j).
//   - Eigenvalues ω² are in (eV/Å²)/amu; convert to cm⁻¹ via CM_FROM_SQRT_EV_AMU_A2.
//   - Imaginary modes (ω² < 0) are reported with negative frequency by convention.

import { jacobiEigen } from './jacobi'
import { buildTrRotBasis, projectOutTrRot } from './projector'
import type { Simulation } from '../worker/simulation'
import { getSymbol } from '../chem/elements'

// 1/(2π·c) × sqrt(1 eV / (1 amu · Å²))  expressed in cm⁻¹. Derivation:
//   ω [1/s] = sqrt(eig · 9.648533e27)
//   ν [cm⁻¹] = ω / (2π c) with c = 2.99792458e10 cm/s
// So  ν = sqrt(eig) · 521.47
const CM_FROM_SQRT_EV_AMU_A2 = 521.4709

export interface VibMode {
  index: number
  frequencyCm: number       // cm⁻¹, negative for imaginary modes
  imaginary: boolean
  eigenvalue: number        // (eV/Å²)/amu — raw ω²
  displacement: Float64Array // length 3N, un-mass-weighted Cartesian displacement
}

export interface VibResult {
  modes: VibMode[]
  equilibriumPositions: Float64Array
  atomicNumbers: number[]
  nProjected: number  // number of TR directions removed, 0 if projection disabled
}

export interface VibProgress {
  done: number
  total: number
  phase: 'optimize' | 'hessian' | 'diagonalize' | 'done'
}

export interface ComputeVibOptions {
  delta?: number              // FD step, Å
  projectTrRot?: boolean      // project translations (+ rotations if molecule)
  isPeriodic?: boolean        // skips rotation projection if true
}

export async function computeVibrations(
  sim: Simulation,
  positions: Float64Array,
  atomicNumbers: number[],
  masses: Float64Array,
  options: ComputeVibOptions = {},
  onProgress?: (p: VibProgress) => void,
): Promise<VibResult> {
  const delta = options.delta ?? 0.01
  const doProject = options.projectTrRot ?? true
  const isPeriodic = options.isPeriodic ?? false
  const n3 = positions.length
  const n = n3 / 3
  if (n !== atomicNumbers.length) throw new Error('positions/atomicNumbers mismatch')
  if (masses.length !== n) throw new Error('masses length mismatch')

  onProgress?.({ done: 0, total: n3, phase: 'hessian' })

  // Central differences: for each DOF i, evaluate forces at x ± δ e_i.
  // Store F(x+δe_i) in plusF[i * n3 + j], similarly minusF.
  const plusF = new Float64Array(n3 * n3)
  const minusF = new Float64Array(n3 * n3)

  const scratch = new Float64Array(positions)  // reusable work buffer

  for (let i = 0; i < n3; i++) {
    scratch.set(positions)
    scratch[i] += delta
    const fp = await sim.predictAt(scratch)
    plusF.set(fp.forces, i * n3)

    scratch[i] -= 2 * delta
    const fm = await sim.predictAt(scratch)
    minusF.set(fm.forces, i * n3)

    onProgress?.({ done: i + 1, total: n3, phase: 'hessian' })
  }

  // H[i,j] = -(F_j(x + δe_i) - F_j(x - δe_i)) / (2δ)
  //   with sign convention F = -∇V so H = -∂F_j/∂x_i.
  // Symmetrize (H + Hᵀ)/2 to kill FD asymmetry noise.
  const H = new Float64Array(n3 * n3)
  for (let i = 0; i < n3; i++) {
    for (let j = 0; j < n3; j++) {
      H[i * n3 + j] = -(plusF[i * n3 + j] - minusF[i * n3 + j]) / (2 * delta)
    }
  }
  for (let i = 0; i < n3; i++) {
    for (let j = i + 1; j < n3; j++) {
      const avg = 0.5 * (H[i * n3 + j] + H[j * n3 + i])
      H[i * n3 + j] = avg
      H[j * n3 + i] = avg
    }
  }

  // Mass-weight: D[i,j] = H[i,j] / sqrt(m_i m_j)
  const invSqrtM = new Float64Array(n3)
  for (let a = 0; a < n; a++) {
    const s = 1 / Math.sqrt(masses[a])
    invSqrtM[a * 3] = s
    invSqrtM[a * 3 + 1] = s
    invSqrtM[a * 3 + 2] = s
  }
  const D = new Float64Array(n3 * n3)
  for (let i = 0; i < n3; i++) {
    for (let j = 0; j < n3; j++) {
      D[i * n3 + j] = H[i * n3 + j] * invSqrtM[i] * invSqrtM[j]
    }
  }

  // Project out translations (+ rotations if non-periodic) to clean up the
  // 6 (or 3 for crystals / 5 for linear molecules) zero-frequency modes.
  let nProjected = 0
  if (doProject) {
    const basis = buildTrRotBasis(positions, masses, !isPeriodic)
    projectOutTrRot(D, n3, basis.vectors)
    nProjected = basis.nRemoved
  }

  onProgress?.({ done: n3, total: n3, phase: 'diagonalize' })
  // Defer one microtask so the UI can paint the "diagonalizing" state.
  await new Promise((r) => setTimeout(r, 0))

  const { values, vectors } = jacobiEigen(D, n3)

  // Build modes: convert eigenvalues → cm⁻¹, un-mass-weight eigenvectors.
  // Projected-out directions have eigenvalues ~0 and sit at the bottom of the
  // sorted list — skip them by count so the user only sees real vibrations.
  const modes: VibMode[] = []
  for (let k = nProjected; k < n3; k++) {
    const ev = values[k]
    const imaginary = ev < 0
    const freq = (imaginary ? -1 : 1) * Math.sqrt(Math.abs(ev)) * CM_FROM_SQRT_EV_AMU_A2

    const displacement = new Float64Array(n3)
    for (let i = 0; i < n3; i++) {
      // u_i = v_i / sqrt(m_i) — convert mass-weighted back to Cartesian
      displacement[i] = vectors[k * n3 + i] * invSqrtM[i]
    }
    // Normalize displacement so the largest atomic displacement is 1 Å at
    // unit amplitude. Nicer for animation than raw mass-weighted vector norm.
    let maxLen = 0
    for (let a = 0; a < n; a++) {
      const dx = displacement[a * 3]
      const dy = displacement[a * 3 + 1]
      const dz = displacement[a * 3 + 2]
      const len = Math.sqrt(dx * dx + dy * dy + dz * dz)
      if (len > maxLen) maxLen = len
    }
    if (maxLen > 0) {
      for (let i = 0; i < n3; i++) displacement[i] /= maxLen
    }

    modes.push({ index: k, frequencyCm: freq, imaginary, eigenvalue: ev, displacement })
  }

  onProgress?.({ done: n3, total: n3, phase: 'done' })

  return {
    modes,
    equilibriumPositions: new Float64Array(positions),
    atomicNumbers: [...atomicNumbers],
    nProjected,
  }
}

export function formatFrequency(mode: VibMode): string {
  const v = Math.abs(mode.frequencyCm)
  const s = v < 10 ? v.toFixed(2) : v.toFixed(1)
  return mode.imaginary ? `${s}i cm⁻¹` : `${s} cm⁻¹`
}

// Human-readable hint about which atom(s) dominate a mode.
export function modeSummary(mode: VibMode, atomicNumbers: number[]): string {
  const n = atomicNumbers.length
  const weights: { idx: number; w: number }[] = []
  for (let a = 0; a < n; a++) {
    const dx = mode.displacement[a * 3]
    const dy = mode.displacement[a * 3 + 1]
    const dz = mode.displacement[a * 3 + 2]
    weights.push({ idx: a, w: Math.sqrt(dx * dx + dy * dy + dz * dz) })
  }
  weights.sort((a, b) => b.w - a.w)
  const top = weights.slice(0, 2).filter((w) => w.w > 0.15)
  if (!top.length) return ''
  return top.map((w) => `${getSymbol(atomicNumbers[w.idx])}${w.idx + 1}`).join('+')
}
