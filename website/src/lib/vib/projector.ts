// Translation/rotation projector for molecular Hessians.
//
// In mass-weighted coordinates q_i = x_i * sqrt(m_i), the translation and
// rotation directions span a 6D (or 5D for linear molecules) subspace with
// eigenvalue 0. Finite differences and numerical noise smear those to small
// non-zero eigenvalues that mix with real vibrations. We build the TR basis
// analytically, orthonormalize it, and apply the projector P = I - V V^T to
// the mass-weighted Hessian before diagonalizing.
//
// Reference: the standard approach used in ORCA/psi4/CFOUR/Gaussian and the
// projector in OCC's vibrational analysis.

export interface TrRotBasis {
  vectors: Float64Array[]  // each length 3N, mass-weighted, orthonormalized
  nRemoved: number          // 3, 5, or 6 depending on geometry
}

// Build and orthonormalize mass-weighted translation + rotation directions.
// `positions` is 3N in Å, `masses` is N in amu. Only used for molecules — for
// periodic cells there are 3 acoustic translations but no rotations, so pass
// includeRotations=false.
export function buildTrRotBasis(
  positions: Float64Array,
  masses: Float64Array,
  includeRotations: boolean = true,
  tol: number = 1e-6,
): TrRotBasis {
  const n = masses.length
  const n3 = 3 * n

  // Shift to center of mass — rotations are defined about the COM.
  let cx = 0, cy = 0, cz = 0, mTot = 0
  for (let a = 0; a < n; a++) {
    cx += masses[a] * positions[a * 3]
    cy += masses[a] * positions[a * 3 + 1]
    cz += masses[a] * positions[a * 3 + 2]
    mTot += masses[a]
  }
  cx /= mTot; cy /= mTot; cz /= mTot
  const r = new Float64Array(n3)
  for (let a = 0; a < n; a++) {
    r[a * 3] = positions[a * 3] - cx
    r[a * 3 + 1] = positions[a * 3 + 1] - cy
    r[a * 3 + 2] = positions[a * 3 + 2] - cz
  }
  const sqrtM = new Float64Array(n)
  for (let a = 0; a < n; a++) sqrtM[a] = Math.sqrt(masses[a])

  // Raw translation vectors in mass-weighted coords:
  //   T_α[3a+β] = δ(α,β) * sqrt(m_a)
  const raw: Float64Array[] = []
  for (let alpha = 0; alpha < 3; alpha++) {
    const v = new Float64Array(n3)
    for (let a = 0; a < n; a++) v[a * 3 + alpha] = sqrtM[a]
    raw.push(v)
  }

  // Raw rotation vectors (about COM), mass-weighted:
  //   R_α[3a+β] = ε_αβγ r_a[γ] * sqrt(m_a)
  // i.e. rotation about axis α acts on atom a as (e_α × r_a) * sqrt(m_a).
  if (includeRotations) {
    for (let alpha = 0; alpha < 3; alpha++) {
      const v = new Float64Array(n3)
      for (let a = 0; a < n; a++) {
        const rx = r[a * 3], ry = r[a * 3 + 1], rz = r[a * 3 + 2]
        if (alpha === 0) {        // x axis:  (0, -rz, ry)
          v[a * 3 + 1] = -rz * sqrtM[a]
          v[a * 3 + 2] =  ry * sqrtM[a]
        } else if (alpha === 1) { // y axis:  ( rz, 0, -rx)
          v[a * 3 + 0] =  rz * sqrtM[a]
          v[a * 3 + 2] = -rx * sqrtM[a]
        } else {                  // z axis:  (-ry, rx, 0)
          v[a * 3 + 0] = -ry * sqrtM[a]
          v[a * 3 + 1] =  rx * sqrtM[a]
        }
      }
      raw.push(v)
    }
  }

  // Gram-Schmidt with drop-on-near-zero-norm. For linear molecules one
  // rotation vector becomes (nearly) zero after orthogonalization against
  // the other two — we drop it.
  const ortho: Float64Array[] = []
  for (const u of raw) {
    const v = new Float64Array(u)
    for (const w of ortho) {
      let dot = 0
      for (let i = 0; i < n3; i++) dot += v[i] * w[i]
      for (let i = 0; i < n3; i++) v[i] -= dot * w[i]
    }
    let norm = 0
    for (let i = 0; i < n3; i++) norm += v[i] * v[i]
    norm = Math.sqrt(norm)
    if (norm > tol) {
      for (let i = 0; i < n3; i++) v[i] /= norm
      ortho.push(v)
    }
  }

  return { vectors: ortho, nRemoved: ortho.length }
}

// Apply P D P in-place where P = I - Σ_k v_k v_k^T and v_k are orthonormal.
// For orthonormal V we can decompose:  P D P = D - V V^T D - D V V^T + V V^T D V V^T.
// Implemented as: left-project, then right-project (equivalent since V is ON).
export function projectOutTrRot(D: Float64Array, n: number, V: Float64Array[]): void {
  // Left projection: D ← D - v (v^T D) for each v.
  for (const v of V) {
    // row = v^T D  (length n)
    const row = new Float64Array(n)
    for (let j = 0; j < n; j++) {
      let s = 0
      for (let i = 0; i < n; i++) s += v[i] * D[i * n + j]
      row[j] = s
    }
    for (let i = 0; i < n; i++) {
      const vi = v[i]
      if (vi === 0) continue
      for (let j = 0; j < n; j++) D[i * n + j] -= vi * row[j]
    }
  }

  // Right projection: D ← D - (D v) v^T for each v.
  for (const v of V) {
    // col = D v  (length n)
    const col = new Float64Array(n)
    for (let i = 0; i < n; i++) {
      let s = 0
      for (let j = 0; j < n; j++) s += D[i * n + j] * v[j]
      col[i] = s
    }
    for (let i = 0; i < n; i++) {
      const ci = col[i]
      if (ci === 0) continue
      for (let j = 0; j < n; j++) D[i * n + j] -= ci * v[j]
    }
  }
}
