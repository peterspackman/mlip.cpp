// Valence-aware hydrogen placement.
//
// Given current heavy-atom positions + connectivity, fill in the missing
// hydrogens so that each heavy atom reaches its standard neutral valence.
// Geometry is extrapolated from existing neighbor directions: linear extension
// for 1-existing-1-missing, tetrahedral / sp2 fans for the methyl / methylene
// cases, etc. This is "geometrically reasonable" rather than chemically
// rigorous — no charge / hybridization perception, no SMILES-style bond-order
// inference. Good enough for sketching organics.

import * as THREE from 'three'
import { detectBonds } from '../chem/bonds'

const STANDARD_VALENCE: Record<number, number> = {
  5: 3,   // B
  6: 4,   // C
  7: 3,   // N
  8: 2,   // O
  9: 1,   // F
  14: 4,  // Si
  15: 3,  // P
  16: 2,  // S
  17: 1,  // Cl
  35: 1,  // Br
  53: 1,  // I
}

const XH_BOND_LEN: Record<number, number> = {
  5: 1.19,   // B-H
  6: 1.09,   // C-H
  7: 1.01,   // N-H
  8: 0.96,   // O-H
  14: 1.48,  // Si-H
  15: 1.42,  // P-H
  16: 1.34,  // S-H
}
const DEFAULT_XH = 1.10

const TETRAHEDRAL_COS = -1 / 3                         // cos(109.4712°)
const TETRAHEDRAL_SIN = Math.sqrt(8) / 3               // sin(109.4712°)

export interface FillResult {
  /** New atomic numbers to append (all 1's, length = added.length / 3). */
  added: Float64Array
  /** Number of hydrogens added (also added.length / 3). */
  count: number
}

/** Append-only: compute coordinates of new H atoms to add. Heavy atoms /
 *  their existing positions are not modified — this just returns the new tail.
 *
 *  @param scope  Optional set of heavy-atom indices to fill on. If null/empty,
 *                all heavy atoms are considered.
 */
export function fillHydrogens(
  positions: Float64Array,
  atomicNumbers: number[],
  scope: ReadonlySet<number> | null,
): FillResult {
  const n = atomicNumbers.length
  const adj = buildAdjacency(positions, atomicNumbers)
  const newH: number[] = []

  for (let i = 0; i < n; i++) {
    if (atomicNumbers[i] === 1) continue
    if (scope && scope.size > 0 && !scope.has(i)) continue
    const valence = STANDARD_VALENCE[atomicNumbers[i]]
    if (!valence) continue
    const nbrs = adj.get(i) ?? []
    const need = valence - nbrs.length
    if (need <= 0) continue

    const center = new THREE.Vector3(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2])
    const existing: THREE.Vector3[] = nbrs.map((j) => {
      const v = new THREE.Vector3(
        positions[j * 3]     - positions[i * 3],
        positions[j * 3 + 1] - positions[i * 3 + 1],
        positions[j * 3 + 2] - positions[i * 3 + 2],
      )
      return v.lengthSq() > 1e-12 ? v.normalize() : new THREE.Vector3(0, 0, 1)
    })
    const dirs = missingDirections(existing, need)
    const len = XH_BOND_LEN[atomicNumbers[i]] ?? DEFAULT_XH
    for (const d of dirs) {
      newH.push(center.x + d.x * len, center.y + d.y * len, center.z + d.z * len)
    }
  }

  return { added: new Float64Array(newH), count: newH.length / 3 }
}

function buildAdjacency(positions: Float64Array, atomicNumbers: number[]): Map<number, number[]> {
  const adj = new Map<number, number[]>()
  // detectBonds returns 1-indexed pairs.
  for (const [a, b] of detectBonds(positions, atomicNumbers)) {
    const i = a - 1, j = b - 1
    if (!adj.has(i)) adj.set(i, [])
    if (!adj.has(j)) adj.set(j, [])
    adj.get(i)!.push(j)
    adj.get(j)!.push(i)
  }
  return adj
}

/** Decide where to place `need` new bonds given the unit vectors of any
 *  currently bonded neighbors. Returns unit vectors. */
function missingDirections(existing: THREE.Vector3[], need: number): THREE.Vector3[] {
  if (need <= 0) return []
  const ne = existing.length

  if (ne === 0) return tetrahedronVertices(need)

  if (need === 1) {
    if (ne === 1) {
      // Linear extension; close enough for cap H's.
      return [existing[0].clone().negate()]
    }
    // Bisector / tetrahedral 4th vertex for 2-3 existing.
    const sum = new THREE.Vector3()
    for (const d of existing) sum.add(d)
    if (sum.lengthSq() < 1e-9) {
      // Existing bonds cancel (e.g. linear AB-A-AB) — pick any perpendicular.
      return [perpendicularTo(existing[0])]
    }
    return [sum.negate().normalize()]
  }

  if (ne === 1) {
    // Cone of `need` directions at tetrahedral angle from existing.
    return coneAroundAxis(existing[0], need, 360 / need)
  }

  if (ne === 2 && need === 2) {
    // Methylene-style: 2 H's perpendicular to the (a,b) plane.
    const bisector = existing[0].clone().add(existing[1])
    if (bisector.lengthSq() < 1e-9) {
      // Existing bonds antiparallel — split the perpendicular.
      const perp = perpendicularTo(existing[0])
      const along = existing[0].clone().normalize()
      const out = new THREE.Vector3().crossVectors(perp, along).normalize()
      return [out, out.clone().negate()]
    }
    bisector.negate().normalize()
    const planeNormal = new THREE.Vector3().crossVectors(existing[0], existing[1])
    if (planeNormal.lengthSq() < 1e-9) {
      // Parallel existing — fall back to arbitrary perpendicular.
      planeNormal.copy(perpendicularTo(bisector))
    }
    planeNormal.normalize()
    // Tetrahedral half-angle: each new dir = cos(α)·bisector + sin(α)·n,
    // where α picks an out-of-plane angle. For 2 missing tetrahedral vertices
    // around the plane bisector with existing at the tetrahedral angle, the
    // out-of-plane component is sin(54.7356°) ≈ 0.8165.
    const inPlane = bisector
    const c = 1 / Math.sqrt(3)        // cos(54.7356°) ≈ 0.5774
    const s = Math.sqrt(2 / 3)        // sin(54.7356°) ≈ 0.8165
    const a = inPlane.clone().multiplyScalar(c).addScaledVector(planeNormal, s).normalize()
    const b = inPlane.clone().multiplyScalar(c).addScaledVector(planeNormal, -s).normalize()
    return [a, b]
  }

  // Generic fallback: spread `need` vectors opposing the centroid of existing.
  const sum = new THREE.Vector3()
  for (const d of existing) sum.add(d)
  const axis = sum.lengthSq() > 1e-9 ? sum.negate().normalize() : new THREE.Vector3(0, 0, 1)
  return coneAroundAxis(axis, need, 360 / need, 0)
}

/** Vertices of a regular tetrahedron, returning the first `count` of them. */
function tetrahedronVertices(count: number): THREE.Vector3[] {
  const t = 1 / Math.sqrt(3)
  const all = [
    new THREE.Vector3( t,  t,  t),
    new THREE.Vector3( t, -t, -t),
    new THREE.Vector3(-t,  t, -t),
    new THREE.Vector3(-t, -t,  t),
  ]
  return all.slice(0, Math.max(0, Math.min(4, count)))
}

/** `count` unit vectors arrayed around `axis` at the tetrahedral exterior
 *  angle (109.47°), evenly spaced in azimuth (`step` deg per slot). */
function coneAroundAxis(
  axis: THREE.Vector3,
  count: number,
  step: number,
  startDeg = 0,
): THREE.Vector3[] {
  const u = axis.clone().normalize()
  const v = perpendicularTo(u)
  const w = new THREE.Vector3().crossVectors(u, v).normalize()
  const out: THREE.Vector3[] = []
  for (let k = 0; k < count; k++) {
    const phi = (startDeg + k * step) * Math.PI / 180
    // Tip points along +u with the tetrahedral half-angle off-axis.
    const dir = u.clone().multiplyScalar(TETRAHEDRAL_COS)
      .addScaledVector(v, TETRAHEDRAL_SIN * Math.cos(phi))
      .addScaledVector(w, TETRAHEDRAL_SIN * Math.sin(phi))
      .normalize()
    out.push(dir)
  }
  return out
}

/** Some unit vector perpendicular to `v`. Avoids the degenerate axis. */
function perpendicularTo(v: THREE.Vector3): THREE.Vector3 {
  const a = Math.abs(v.x) < 0.9 ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 1, 0)
  return new THREE.Vector3().crossVectors(v, a).normalize()
}
