import { getAtomicNumber, getSymbol } from './elements'
import type { Lattice } from './cell'

export function parseAtomicNumbers(xyz: string): number[] {
  const lines = xyz.trim().split('\n')
  const n = parseInt(lines[0])
  const out: number[] = []
  for (let i = 0; i < n; i++) {
    const parts = lines[i + 2].trim().split(/\s+/)
    out.push(getAtomicNumber(parts[0]))
  }
  return out
}

export function parsePositions(xyz: string): number[] {
  const lines = xyz.trim().split('\n')
  const n = parseInt(lines[0])
  const out: number[] = []
  for (let i = 0; i < n; i++) {
    const parts = lines[i + 2].trim().split(/\s+/)
    out.push(parseFloat(parts[1]), parseFloat(parts[2]), parseFloat(parts[3]))
  }
  return out
}

/**
 * Serialize to extended XYZ. Includes a Lattice="..." field when periodic so
 * the worker rebuilds the same cell on setSystem().
 */
export function serializeXyz(
  positions: ArrayLike<number>,
  atomicNumbers: ArrayLike<number>,
  lattice: Lattice | null,
  comment = '',
): string {
  const n = atomicNumbers.length
  const lines: string[] = [String(n)]
  if (lattice) {
    const lat = [...lattice.a, ...lattice.b, ...lattice.c].map((x) => x.toFixed(8)).join(' ')
    const props = 'Properties=species:S:1:pos:R:3 pbc="T T T"'
    lines.push(`Lattice="${lat}" ${props} ${comment}`.trim())
  } else {
    lines.push(comment)
  }
  for (let i = 0; i < n; i++) {
    const sym = getSymbol(atomicNumbers[i] as number)
    const x = (positions[i * 3] as number).toFixed(8)
    const y = (positions[i * 3 + 1] as number).toFixed(8)
    const z = (positions[i * 3 + 2] as number).toFixed(8)
    lines.push(`${sym} ${x} ${y} ${z}`)
  }
  return lines.join('\n') + '\n'
}
