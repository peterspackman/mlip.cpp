// Bridge between mlipcpp's structure shape (flat positions array + Z list +
// {a,b,c} lattice) and the cebuns StructureData consumed by ViewerStage.
import { getElementByNumber } from "./data/elements";
import { detectBonds } from "./data/bonds";
import type { Lattice } from "../chem/cell";
import type { AtomData, BondData, StructureData, UnitCellData, Vec3 } from "./data/types";
import { derivePeriodicity } from "./data/types";

function angleDeg(u: [number, number, number], v: [number, number, number]): number {
  const dot = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
  const lu = Math.hypot(u[0], u[1], u[2]);
  const lv = Math.hypot(v[0], v[1], v[2]);
  if (lu === 0 || lv === 0) return 0;
  const c = Math.min(1, Math.max(-1, dot / (lu * lv)));
  return (Math.acos(c) * 180) / Math.PI;
}

export function latticeToUnitCell(lat: Lattice): UnitCellData {
  const a = lat.a, b = lat.b, c = lat.c;
  // Row-major storage; column vectors hold a, b, c (see types.ts comment).
  const matrix: UnitCellData["matrix"] = [
    a[0], b[0], c[0],
    a[1], b[1], c[1],
    a[2], b[2], c[2],
  ];
  return {
    a: Math.hypot(a[0], a[1], a[2]),
    b: Math.hypot(b[0], b[1], b[2]),
    c: Math.hypot(c[0], c[1], c[2]),
    alpha: angleDeg(b, c),
    beta:  angleDeg(a, c),
    gamma: angleDeg(a, b),
    matrix,
  };
}

function fractionalCoords(pos: Vec3, lat: Lattice): Vec3 {
  const [ax, ay, az] = lat.a;
  const [bx, by, bz] = lat.b;
  const [cx, cy, cz] = lat.c;
  const det =
    ax * (by * cz - bz * cy) -
    bx * (ay * cz - az * cy) +
    cx * (ay * bz - az * by);
  // Inverse of [a b c] (column vectors). Each row of the inverse projects
  // onto one fractional coordinate.
  const fa = ((by * cz - bz * cy) * pos.x + (cx * bz - bx * cz) * pos.y + (bx * cy - cx * by) * pos.z) / det;
  const fb = ((az * cy - ay * cz) * pos.x + (ax * cz - cx * az) * pos.y + (cx * ay - ax * cy) * pos.z) / det;
  const fc = ((ay * bz - az * by) * pos.x + (bx * az - ax * bz) * pos.y + (ax * by - bx * ay) * pos.z) / det;
  return { x: fa, y: fb, z: fc };
}

export interface BuildOptions {
  name?: string;
}

/**
 * Build a StructureData from an already-prepared (cartesian) atom list.
 * Caller is responsible for any wrap / supercell expansion done upstream;
 * this is a pure shape transform with bond detection bolted on.
 */
export function buildStructureData(
  positions: ArrayLike<number>,
  atomicNumbers: ArrayLike<number>,
  lattice: Lattice | null,
  opts: BuildOptions = {},
): StructureData {
  const n = atomicNumbers.length;
  const atoms: AtomData[] = new Array(n);
  const unitCell = lattice ? latticeToUnitCell(lattice) : undefined;

  for (let i = 0; i < n; i++) {
    const z = atomicNumbers[i];
    const pos: Vec3 = {
      x: positions[i * 3],
      y: positions[i * 3 + 1],
      z: positions[i * 3 + 2],
    };
    const atom: AtomData = {
      element: getElementByNumber(z).symbol,
      atomicNumber: z,
      position: pos,
    };
    if (lattice) atom.fractional = fractionalCoords(pos, lattice);
    atoms[i] = atom;
  }

  const bonds: BondData[] = detectBonds(atoms);
  return {
    name: opts.name ?? "structure",
    atoms,
    bonds,
    periodicity: derivePeriodicity(unitCell),
    unitCell,
    metadata: {},
  };
}
