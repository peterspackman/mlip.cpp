import * as THREE from "three";
import type { Representation, BuildOptions } from "./types";
import type { StructureData } from "../data/types";
import type { MaterialSettings } from "../lighting";
import { getElementColor } from "../data/colors";

/** Convert sRGB 0-1 to linear for use in raw vertex color buffers */
function srgbToLinear(c: number): number {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

export class WireframeRepresentation implements Representation {
  readonly type = "wireframe" as const;
  readonly group = new THREE.Group();

  private lines: THREE.LineSegments | null = null;
  private points: THREE.Points | null = null;

  build(data: StructureData, options?: BuildOptions): void {
    this.dispose();

    if (data.atoms.length === 0) return;

    const co = options?.colorOverride;
    const colorFn = options?.atomColorFn;
    const hidden = options?.hiddenElements;
    const hiddenIdx = options?.hiddenIndices;

    // Build visible atoms
    const visibleAtoms: { atom: typeof data.atoms[0]; index: number }[] = [];
    for (let i = 0; i < data.atoms.length; i++) {
      if (hidden && hidden.has(data.atoms[i].atomicNumber)) continue;
      if (hiddenIdx && hiddenIdx.has(i)) continue;
      visibleAtoms.push({ atom: data.atoms[i], index: i });
    }

    if (visibleAtoms.length === 0) return;

    // Atom points
    const pointsGeo = new THREE.BufferGeometry();
    const positions = new Float32Array(visibleAtoms.length * 3);
    const colors = new Float32Array(visibleAtoms.length * 3);

    for (let vi = 0; vi < visibleAtoms.length; vi++) {
      const atom = visibleAtoms[vi].atom;
      positions[vi * 3] = atom.position.x;
      positions[vi * 3 + 1] = atom.position.y;
      positions[vi * 3 + 2] = atom.position.z;

      if (co) {
        colors[vi * 3] = srgbToLinear(co[0]);
        colors[vi * 3 + 1] = srgbToLinear(co[1]);
        colors[vi * 3 + 2] = srgbToLinear(co[2]);
      } else if (colorFn) {
        const [r, g, b] = colorFn(visibleAtoms[vi].index);
        colors[vi * 3] = srgbToLinear(r);
        colors[vi * 3 + 1] = srgbToLinear(g);
        colors[vi * 3 + 2] = srgbToLinear(b);
      } else {
        const [r, g, b] = getElementColor(atom.atomicNumber);
        colors[vi * 3] = srgbToLinear(r);
        colors[vi * 3 + 1] = srgbToLinear(g);
        colors[vi * 3 + 2] = srgbToLinear(b);
      }
    }

    pointsGeo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    pointsGeo.setAttribute("color", new THREE.BufferAttribute(colors, 3));

    this.points = new THREE.Points(
      pointsGeo,
      new THREE.PointsMaterial({ size: 4, vertexColors: true, sizeAttenuation: false }),
    );
    this.group.add(this.points);

    // Build set of hidden atom indices for bond filtering
    const hiddenAtomSet = hidden
      ? new Set(data.atoms.map((a, i) => hidden.has(a.atomicNumber) ? i : -1).filter(i => i >= 0))
      : null;

    // Bond lines (skip bonds with hidden atoms)
    const visibleBonds = data.bonds.filter(bond => {
      if (!hiddenAtomSet) return true;
      return !hiddenAtomSet.has(bond.atomA) && !hiddenAtomSet.has(bond.atomB);
    });

    if (visibleBonds.length > 0) {
      const lineGeo = new THREE.BufferGeometry();
      const linePositions = new Float32Array(visibleBonds.length * 6);
      const lineColors = new Float32Array(visibleBonds.length * 6);

      for (let i = 0; i < visibleBonds.length; i++) {
        const bond = visibleBonds[i];
        const a = data.atoms[bond.atomA];
        const b = data.atoms[bond.atomB];

        linePositions[i * 6] = a.position.x;
        linePositions[i * 6 + 1] = a.position.y;
        linePositions[i * 6 + 2] = a.position.z;
        linePositions[i * 6 + 3] = b.position.x;
        linePositions[i * 6 + 4] = b.position.y;
        linePositions[i * 6 + 5] = b.position.z;

        if (co) {
          const lr = srgbToLinear(co[0]), lg = srgbToLinear(co[1]), lb = srgbToLinear(co[2]);
          lineColors[i * 6] = lr;
          lineColors[i * 6 + 1] = lg;
          lineColors[i * 6 + 2] = lb;
          lineColors[i * 6 + 3] = lr;
          lineColors[i * 6 + 4] = lg;
          lineColors[i * 6 + 5] = lb;
        } else if (colorFn) {
          const [rA, gA, bA] = colorFn(bond.atomA);
          const [rB, gB, bB] = colorFn(bond.atomB);
          lineColors[i * 6] = srgbToLinear(rA);
          lineColors[i * 6 + 1] = srgbToLinear(gA);
          lineColors[i * 6 + 2] = srgbToLinear(bA);
          lineColors[i * 6 + 3] = srgbToLinear(rB);
          lineColors[i * 6 + 4] = srgbToLinear(gB);
          lineColors[i * 6 + 5] = srgbToLinear(bB);
        } else {
          const [rA, gA, bA] = getElementColor(a.atomicNumber);
          const [rB, gB, bB] = getElementColor(b.atomicNumber);
          lineColors[i * 6] = srgbToLinear(rA);
          lineColors[i * 6 + 1] = srgbToLinear(gA);
          lineColors[i * 6 + 2] = srgbToLinear(bA);
          lineColors[i * 6 + 3] = srgbToLinear(rB);
          lineColors[i * 6 + 4] = srgbToLinear(gB);
          lineColors[i * 6 + 5] = srgbToLinear(bB);
        }
      }

      lineGeo.setAttribute("position", new THREE.BufferAttribute(linePositions, 3));
      lineGeo.setAttribute("color", new THREE.BufferAttribute(lineColors, 3));

      this.lines = new THREE.LineSegments(
        lineGeo,
        new THREE.LineBasicMaterial({ vertexColors: true }),
      );
      this.group.add(this.lines);
    }
  }

  getAtomIndex(_object: THREE.Object3D, _instanceId: number): number {
    return -1;
  }

  getBondAtoms(_object: THREE.Object3D, _instanceId: number): [number, number] | null {
    return null;
  }

  syncSelection(_added: number[], _removed: number[]): void {}
  buildSelectionProxies(_selectedIndices: ReadonlySet<number>): THREE.Object3D[] { return []; }
  updateMaterial(_settings: MaterialSettings): void {}

  dispose(): void {
    if (this.lines) {
      this.lines.geometry.dispose();
      this.group.remove(this.lines);
      this.lines = null;
    }
    if (this.points) {
      this.points.geometry.dispose();
      this.group.remove(this.points);
      this.points = null;
    }
  }
}
