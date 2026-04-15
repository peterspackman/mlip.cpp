export interface Lattice {
  a: [number, number, number]
  b: [number, number, number]
  c: [number, number, number]
}

export function parseLattice(xyz: string): Lattice | null {
  const lines = xyz.trim().split('\n')
  if (lines.length < 2) return null
  const m = lines[1].match(/Lattice="([^"]+)"/)
  if (!m) return null
  const v = m[1].split(/\s+/).map(Number)
  if (v.length !== 9 || v.some(Number.isNaN)) return null
  return {
    a: [v[0], v[1], v[2]],
    b: [v[3], v[4], v[5]],
    c: [v[6], v[7], v[8]],
  }
}

export function volume(lat: Lattice): number {
  const [ax, ay, az] = lat.a
  const [bx, by, bz] = lat.b
  const [cx, cy, cz] = lat.c
  return Math.abs(ax * (by * cz - bz * cy) - bx * (ay * cz - az * cy) + cx * (ay * bz - az * by))
}

// Wrap positions into the unit cell via fractional coordinates.
export function wrapPositions(positions: ArrayLike<number>, lat: Lattice): number[] {
  const [ax, ay, az] = lat.a
  const [bx, by, bz] = lat.b
  const [cx, cy, cz] = lat.c
  const det = ax * (by * cz - bz * cy) - bx * (ay * cz - az * cy) + cx * (ay * bz - az * by)
  const inv = [
    [(by * cz - bz * cy) / det, (cx * bz - bx * cz) / det, (bx * cy - cx * by) / det],
    [(az * cy - ay * cz) / det, (ax * cz - cx * az) / det, (cx * ay - ax * cy) / det],
    [(ay * bz - az * by) / det, (bx * az - ax * bz) / det, (ax * by - bx * ay) / det],
  ]
  const n = positions.length / 3
  const out = new Array(positions.length)
  for (let i = 0; i < n; i++) {
    const x = positions[i * 3], y = positions[i * 3 + 1], z = positions[i * 3 + 2]
    let fa = inv[0][0] * x + inv[0][1] * y + inv[0][2] * z
    let fb = inv[1][0] * x + inv[1][1] * y + inv[1][2] * z
    let fc = inv[2][0] * x + inv[2][1] * y + inv[2][2] * z
    fa -= Math.floor(fa)
    fb -= Math.floor(fb)
    fc -= Math.floor(fc)
    out[i * 3] = fa * ax + fb * bx + fc * cx
    out[i * 3 + 1] = fa * ay + fb * by + fc * cy
    out[i * 3 + 2] = fa * az + fb * bz + fc * cz
  }
  return out
}
