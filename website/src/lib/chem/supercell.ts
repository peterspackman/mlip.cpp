import type { Lattice } from './cell'

export interface Supercell {
  positions: number[]
  atomicNumbers: number[]
}

export function generateSupercell(
  positions: ArrayLike<number>,
  atomicNumbers: ArrayLike<number>,
  lat: Lattice,
  size: [number, number, number] = [1, 1, 1],
): Supercell {
  const numAtoms = atomicNumbers.length
  const positionsOut: number[] = []
  const atomicNumbersOut: number[] = []
  const [na, nb, nc] = size

  for (let ia = 0; ia < na; ia++) {
    for (let ib = 0; ib < nb; ib++) {
      for (let ic = 0; ic < nc; ic++) {
        const tx = ia * lat.a[0] + ib * lat.b[0] + ic * lat.c[0]
        const ty = ia * lat.a[1] + ib * lat.b[1] + ic * lat.c[1]
        const tz = ia * lat.a[2] + ib * lat.b[2] + ic * lat.c[2]
        for (let i = 0; i < numAtoms; i++) {
          positionsOut.push(
            positions[i * 3] + tx,
            positions[i * 3 + 1] + ty,
            positions[i * 3 + 2] + tz,
          )
          atomicNumbersOut.push(atomicNumbers[i])
        }
      }
    }
  }
  return { positions: positionsOut, atomicNumbers: atomicNumbersOut }
}
