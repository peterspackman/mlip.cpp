// Classical Jacobi eigensolver for symmetric real matrices.
//
// Fine for the sizes we deal with here (a 21-atom aspirin gives a 63x63
// Hessian — well under a millisecond per sweep in JS).
//
// Returns eigenvalues sorted ascending and the corresponding eigenvectors
// as a column-major flat array: eigvec[j * n + i] = i-th component of
// eigenvector j.

export interface Eigen {
  values: Float64Array
  vectors: Float64Array  // column-major: vectors[j * n + i]
}

export function jacobiEigen(A: Float64Array, n: number, tol = 1e-10, maxSweeps = 50): Eigen {
  // Work on a copy; algorithm destroys its matrix.
  const a = new Float64Array(A)
  const v = new Float64Array(n * n)
  for (let i = 0; i < n; i++) v[i * n + i] = 1  // identity

  const idx = (i: number, j: number) => i * n + j

  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    // Off-diagonal L2 norm as convergence proxy.
    let off = 0
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        off += a[idx(p, q)] * a[idx(p, q)]
      }
    }
    if (off < tol) break

    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = a[idx(p, q)]
        if (Math.abs(apq) < tol) continue
        const app = a[idx(p, p)]
        const aqq = a[idx(q, q)]

        // Rotation angle
        const theta = (aqq - app) / (2 * apq)
        const t =
          theta >= 0
            ? 1 / (theta + Math.sqrt(1 + theta * theta))
            : 1 / (theta - Math.sqrt(1 + theta * theta))
        const c = 1 / Math.sqrt(1 + t * t)
        const s = t * c
        const tau = s / (1 + c)

        a[idx(p, p)] = app - t * apq
        a[idx(q, q)] = aqq + t * apq
        a[idx(p, q)] = 0
        a[idx(q, p)] = 0

        for (let r = 0; r < n; r++) {
          if (r !== p && r !== q) {
            const arp = a[idx(r, p)]
            const arq = a[idx(r, q)]
            a[idx(r, p)] = arp - s * (arq + tau * arp)
            a[idx(p, r)] = a[idx(r, p)]
            a[idx(r, q)] = arq + s * (arp - tau * arq)
            a[idx(q, r)] = a[idx(r, q)]
          }
          const vrp = v[idx(r, p)]
          const vrq = v[idx(r, q)]
          v[idx(r, p)] = vrp - s * (vrq + tau * vrp)
          v[idx(r, q)] = vrq + s * (vrp - tau * vrq)
        }
      }
    }
  }

  // Extract diag → eigenvalues, then sort ascending.
  const values = new Float64Array(n)
  for (let i = 0; i < n; i++) values[i] = a[idx(i, i)]

  const order = Array.from({ length: n }, (_, i) => i)
  order.sort((i, j) => values[i] - values[j])

  const sortedVals = new Float64Array(n)
  const sortedVecs = new Float64Array(n * n)
  for (let j = 0; j < n; j++) {
    sortedVals[j] = values[order[j]]
    for (let i = 0; i < n; i++) {
      sortedVecs[j * n + i] = v[i * n + order[j]]
    }
  }

  return { values: sortedVals, vectors: sortedVecs }
}
