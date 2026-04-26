import * as THREE from "three";
import { LineSegments2 } from "three/examples/jsm/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/examples/jsm/lines/LineSegmentsGeometry.js";
import { LineMaterial } from "three/examples/jsm/lines/LineMaterial.js";
import type { UnitCellData } from "./data/types";
import { DEFAULT_LINE_WIDTH } from "./lighting";

const AXIS_COLORS = {
  a: 0xff4444,
  b: 0x44ff44,
  c: 0x4444ff,
};

interface CellEdges {
  a: number[]; // 4 edges parallel to a-axis
  b: number[]; // 4 edges parallel to b-axis
  c: number[]; // 4 edges parallel to c-axis
}

/** 12 edges of a unit cell box grouped by axis direction */
function cellEdgesByAxis(
  a: THREE.Vector3, b: THREE.Vector3, c: THREE.Vector3,
  oh: number, ok: number, ol: number,
): CellEdges {
  const o = new THREE.Vector3()
    .addScaledVector(a, oh)
    .addScaledVector(b, ok)
    .addScaledVector(c, ol);

  const pa = new THREE.Vector3().addVectors(o, a);
  const pb = new THREE.Vector3().addVectors(o, b);
  const pc = new THREE.Vector3().addVectors(o, c);
  const pab = new THREE.Vector3().addVectors(pa, b);
  const pac = new THREE.Vector3().addVectors(pa, c);
  const pbc = new THREE.Vector3().addVectors(pb, c);
  const pabc = new THREE.Vector3().addVectors(pab, c);

  return {
    // 4 edges along a: o→pa, pb→pab, pc→pac, pbc→pabc
    a: [
      ...o.toArray(), ...pa.toArray(),
      ...pb.toArray(), ...pab.toArray(),
      ...pc.toArray(), ...pac.toArray(),
      ...pbc.toArray(), ...pabc.toArray(),
    ],
    // 4 edges along b: o→pb, pa→pab, pc→pbc, pac→pabc
    b: [
      ...o.toArray(), ...pb.toArray(),
      ...pa.toArray(), ...pab.toArray(),
      ...pc.toArray(), ...pbc.toArray(),
      ...pac.toArray(), ...pabc.toArray(),
    ],
    // 4 edges along c: o→pc, pa→pac, pb→pbc, pab→pabc
    c: [
      ...o.toArray(), ...pc.toArray(),
      ...pa.toArray(), ...pac.toArray(),
      ...pb.toArray(), ...pbc.toArray(),
      ...pab.toArray(), ...pabc.toArray(),
    ],
  };
}

function createLineMaterial(color: number, lineWidth: number = DEFAULT_LINE_WIDTH): LineMaterial {
  return new LineMaterial({
    color,
    linewidth: lineWidth,
    transparent: true,
    opacity: 0.7,
    resolution: new THREE.Vector2(window.innerWidth, window.innerHeight),
  });
}

function addColoredLines(group: THREE.Group, positions: number[], color: number, lineWidth: number = DEFAULT_LINE_WIDTH): void {
  if (positions.length === 0) return;
  const geometry = new LineSegmentsGeometry();
  geometry.setPositions(positions);
  const lines = new LineSegments2(geometry, createLineMaterial(color, lineWidth));
  lines.computeLineDistances();
  group.add(lines);
}

function addAxisArrows(
  group: THREE.Group,
  a: THREE.Vector3, b: THREE.Vector3, c: THREE.Vector3,
): void {
  const o = new THREE.Vector3(0, 0, 0);
  const arrowLength = 0.4;
  group.add(new THREE.ArrowHelper(a.clone().normalize(), o, a.length() * arrowLength, AXIS_COLORS.a, 0.15, 0.08));
  group.add(new THREE.ArrowHelper(b.clone().normalize(), o, b.length() * arrowLength, AXIS_COLORS.b, 0.15, 0.08));
  group.add(new THREE.ArrowHelper(c.clone().normalize(), o, c.length() * arrowLength, AXIS_COLORS.c, 0.15, 0.08));
}

/** Update an existing unit cell group's geometry in-place (for trajectory playback) */
export function updateUnitCellGroupGeometry(group: THREE.Group, cell: UnitCellData, lineWidth: number = DEFAULT_LINE_WIDTH): void {
  // Dispose and remove all children
  for (let i = group.children.length - 1; i >= 0; i--) {
    const child = group.children[i];
    if ((child as any).geometry) (child as any).geometry.dispose();
    if ((child as any).material) {
      const mat = (child as any).material;
      if (Array.isArray(mat)) mat.forEach((m: any) => m.dispose());
      else mat.dispose();
    }
    group.remove(child);
  }

  const m = cell.matrix;
  const a = new THREE.Vector3(m[0], m[3], m[6]);
  const b = new THREE.Vector3(m[1], m[4], m[7]);
  const c = new THREE.Vector3(m[2], m[5], m[8]);

  const edges = cellEdgesByAxis(a, b, c, 0, 0, 0);
  addColoredLines(group, edges.a, AXIS_COLORS.a, lineWidth);
  addColoredLines(group, edges.b, AXIS_COLORS.b, lineWidth);
  addColoredLines(group, edges.c, AXIS_COLORS.c, lineWidth);
  addAxisArrows(group, a, b, c);
}

/** Create a wireframe box for a single unit cell at the origin + axis arrows */
export function createUnitCellGroup(cell: UnitCellData, lineWidth: number = DEFAULT_LINE_WIDTH): THREE.Group {
  const group = new THREE.Group();
  const m = cell.matrix;

  const a = new THREE.Vector3(m[0], m[3], m[6]);
  const b = new THREE.Vector3(m[1], m[4], m[7]);
  const c = new THREE.Vector3(m[2], m[5], m[8]);

  const edges = cellEdgesByAxis(a, b, c, 0, 0, 0);
  addColoredLines(group, edges.a, AXIS_COLORS.a, lineWidth);
  addColoredLines(group, edges.b, AXIS_COLORS.b, lineWidth);
  addColoredLines(group, edges.c, AXIS_COLORS.c, lineWidth);

  addAxisArrows(group, a, b, c);

  return group;
}

/**
 * Create wireframe boxes for multiple cell offsets.
 * The origin cell (0,0,0) gets axis arrows; all cells get colored edges.
 */
export function createMultiCellGroup(
  cell: UnitCellData,
  offsets: Array<[number, number, number]>,
  lineWidth: number = DEFAULT_LINE_WIDTH,
): THREE.Group {
  const group = new THREE.Group();
  const m = cell.matrix;

  const a = new THREE.Vector3(m[0], m[3], m[6]);
  const b = new THREE.Vector3(m[1], m[4], m[7]);
  const c = new THREE.Vector3(m[2], m[5], m[8]);

  const allA: number[] = [];
  const allB: number[] = [];
  const allC: number[] = [];

  for (const [oh, ok, ol] of offsets) {
    const edges = cellEdgesByAxis(a, b, c, oh, ok, ol);
    allA.push(...edges.a);
    allB.push(...edges.b);
    allC.push(...edges.c);
  }

  addColoredLines(group, allA, AXIS_COLORS.a, lineWidth);
  addColoredLines(group, allB, AXIS_COLORS.b, lineWidth);
  addColoredLines(group, allC, AXIS_COLORS.c, lineWidth);

  addAxisArrows(group, a, b, c);

  return group;
}
