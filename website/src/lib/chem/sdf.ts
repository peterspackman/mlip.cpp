import { getSymbol } from './elements'
import { detectBonds } from './bonds'

// Build a V2000 SDF/MOL record for NGL. SDF handles the full periodic table
// cleanly, which PDB does not.
export function positionsToSdf(
  positions: ArrayLike<number>,
  atomicNumbers: ArrayLike<number>,
): string {
  const n = atomicNumbers.length
  const bonds = detectBonds(positions, atomicNumbers)

  let out = '\n     RDKit          3D\n\n'
  out += `${String(n).padStart(3)}${String(bonds.length).padStart(3)}  0  0  0  0  0  0  0  0999 V2000\n`

  for (let i = 0; i < n; i++) {
    const x = positions[i * 3].toFixed(4).padStart(10)
    const y = positions[i * 3 + 1].toFixed(4).padStart(10)
    const z = positions[i * 3 + 2].toFixed(4).padStart(10)
    const sym = getSymbol(atomicNumbers[i]).padEnd(3)
    out += `${x}${y}${z} ${sym} 0  0  0  0  0  0  0  0  0  0  0  0\n`
  }
  for (const [a, b] of bonds) {
    out += `${String(a).padStart(3)}${String(b).padStart(3)}  1  0\n`
  }
  out += 'M  END\n'
  return out
}
