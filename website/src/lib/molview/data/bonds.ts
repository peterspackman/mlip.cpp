import type { AtomData, BondData, UnitCellData } from "./types";
import { getElementByNumber } from "./elements";

const BOND_TOLERANCE = 0.4;

export interface PeriodicBond extends BondData {
  /** Cell offset of atomB relative to atomA. [0,0,0] = same cell. */
  imageOffset: [number, number, number];
}

/** Detect bonds between atoms using sum of covalent radii + tolerance */
export function detectBonds(atoms: AtomData[]): BondData[] {
  const bonds: BondData[] = [];
  const n = atoms.length;

  for (let i = 0; i < n; i++) {
    const ai = atoms[i];
    const ri = getElementByNumber(ai.atomicNumber).covalentRadius;

    for (let j = i + 1; j < n; j++) {
      const aj = atoms[j];
      const rj = getElementByNumber(aj.atomicNumber).covalentRadius;

      const dx = ai.position.x - aj.position.x;
      const dy = ai.position.y - aj.position.y;
      const dz = ai.position.z - aj.position.z;
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

      const maxDist = ri + rj + BOND_TOLERANCE;
      // Minimum distance to avoid bonding overlapping atoms
      const minDist = 0.4;

      if (dist > minDist && dist <= maxDist) {
        bonds.push({ atomA: i, atomB: j, order: 1 });
      }
    }
  }

  return bonds;
}

/**
 * Detect bonds involving new atoms (index >= startIndex).
 * Checks pairs (i, j) where i < j and j >= startIndex.
 * This is O(new × total) instead of O(n²).
 */
export function detectBondsIncremental(atoms: AtomData[], startIndex: number): BondData[] {
  const bonds: BondData[] = [];
  const n = atoms.length;

  for (let j = startIndex; j < n; j++) {
    const aj = atoms[j];
    const rj = getElementByNumber(aj.atomicNumber).covalentRadius;

    for (let i = 0; i < j; i++) {
      const ai = atoms[i];
      const ri = getElementByNumber(ai.atomicNumber).covalentRadius;

      const dx = ai.position.x - aj.position.x;
      const dy = ai.position.y - aj.position.y;
      const dz = ai.position.z - aj.position.z;
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

      const maxDist = ri + rj + BOND_TOLERANCE;
      const minDist = 0.4;

      if (dist > minDist && dist <= maxDist) {
        bonds.push({ atomA: i, atomB: j, order: 1 });
      }
    }
  }

  return bonds;
}

/**
 * Detect bonds including those crossing periodic cell boundaries.
 * Checks all 27 cell translations (±1 in each dimension) for each atom pair.
 * All atoms must have fractional coordinates set.
 *
 * O(n² × 27) where n = unit cell atom count (typically < 200).
 */
export function detectBondsPeriodic(
  atoms: AtomData[],
  unitCell: UnitCellData,
): PeriodicBond[] {
  const bonds: PeriodicBond[] = [];
  const n = atoms.length;
  const m = unitCell.matrix;
  const minDist = 0.4;

  for (let i = 0; i < n; i++) {
    const ai = atoms[i];
    const ri = getElementByNumber(ai.atomicNumber).covalentRadius;

    for (let j = i; j < n; j++) {
      const aj = atoms[j];
      const rj = getElementByNumber(aj.atomicNumber).covalentRadius;
      const maxDist = ri + rj + BOND_TOLERANCE;

      for (let dh = -1; dh <= 1; dh++) {
        for (let dk = -1; dk <= 1; dk++) {
          for (let dl = -1; dl <= 1; dl++) {
            // Skip self-bond in same cell
            if (i === j && dh === 0 && dk === 0 && dl === 0) continue;
            // For same-cell bonds (dh=dk=dl=0), only check j > i to avoid duplicates
            if (dh === 0 && dk === 0 && dl === 0 && j === i) continue;

            // Compute Cartesian position of atom j in the image cell
            const fracJ = aj.fractional!;
            const fx = fracJ.x + dh;
            const fy = fracJ.y + dk;
            const fz = fracJ.z + dl;

            const cartJx = fx * m[0] + fy * m[1] + fz * m[2];
            const cartJy = fx * m[3] + fy * m[4] + fz * m[5];
            const cartJz = fx * m[6] + fy * m[7] + fz * m[8];

            const dx = ai.position.x - cartJx;
            const dy = ai.position.y - cartJy;
            const dz = ai.position.z - cartJz;
            const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

            if (dist > minDist && dist <= maxDist) {
              bonds.push({
                atomA: i,
                atomB: j,
                order: 1,
                imageOffset: [dh, dk, dl],
              });
            }
          }
        }
      }
    }
  }

  return bonds;
}
