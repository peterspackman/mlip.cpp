import { getAtomicNumber } from './elements'

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
