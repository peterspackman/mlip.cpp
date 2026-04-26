// Trimmed copy of cebuns StructureData types — keeps only what the editor
// renderer needs (atoms, bonds, periodicity, unit cell). Crystallographic
// extras (symmetry ops, ADPs, fragment identifiers, morphology, slabs) live
// upstream in cebuns/data/types.ts and can be re-introduced if needed.

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export interface AtomData {
  element: string;
  atomicNumber: number;
  position: Vec3;
  label?: string;
  /** Fractional coordinates (set when the structure is periodic). */
  fractional?: Vec3;
}

export interface BondData {
  atomA: number;
  atomB: number;
  order: number;
}

export interface UnitCellData {
  a: number;
  b: number;
  c: number;
  alpha: number;
  beta: number;
  gamma: number;
  /**
   * 3x3 matrix in row-major storage. Columns are the cell vectors a, b, c
   * — i.e. cart = matrix · frac. Read components as
   *   a = (m[0], m[3], m[6]),  b = (m[1], m[4], m[7]),  c = (m[2], m[5], m[8]).
   */
  matrix: [number, number, number, number, number, number, number, number, number];
}

/** 0 = molecule, 1 = rod, 2 = slab, 3 = bulk crystal. */
export type PeriodicityRank = 0 | 1 | 2 | 3;

export interface Periodicity {
  rank: PeriodicityRank;
  /** Cartesian lattice vectors along the periodic dimensions. Length = rank. */
  axes: Array<[number, number, number]>;
}

export function derivePeriodicity(unitCell?: UnitCellData): Periodicity {
  if (!unitCell) return { rank: 0, axes: [] };
  const m = unitCell.matrix;
  return {
    rank: 3,
    axes: [
      [m[0], m[3], m[6]],
      [m[1], m[4], m[7]],
      [m[2], m[5], m[8]],
    ],
  };
}

export interface StructureData {
  name: string;
  atoms: AtomData[];
  bonds: BondData[];
  periodicity: Periodicity;
  unitCell?: UnitCellData;
  /** Index of first ghost atom; atoms[ghostStart..] are rendered transparent.
   *  Unused by the editor today but referenced by vendored representations. */
  ghostStart?: number;
  metadata: Record<string, unknown>;
}
