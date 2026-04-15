import { getCovalentRadius } from './elements'

export type Bond = [number, number]

const BOND_TOLERANCE = 0.4  // Angstroms

export function detectBonds(
  positions: ArrayLike<number>,
  atomicNumbers: ArrayLike<number>,
): Bond[] {
  const n = atomicNumbers.length
  const bonds: Bond[] = []
  for (let i = 0; i < n; i++) {
    const ri = getCovalentRadius(atomicNumbers[i])
    const xi = positions[i * 3], yi = positions[i * 3 + 1], zi = positions[i * 3 + 2]
    for (let j = i + 1; j < n; j++) {
      const rj = getCovalentRadius(atomicNumbers[j])
      const dx = xi - positions[j * 3]
      const dy = yi - positions[j * 3 + 1]
      const dz = zi - positions[j * 3 + 2]
      const d = Math.sqrt(dx * dx + dy * dy + dz * dz)
      if (d < ri + rj + BOND_TOLERANCE) {
        bonds.push([i + 1, j + 1])  // 1-indexed for SDF
      }
    }
  }
  return bonds
}

export function bondsKey(bonds: Bond[]): string {
  return bonds.map(([a, b]) => `${a}-${b}`).join(',')
}
