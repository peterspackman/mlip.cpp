import * as THREE from "three";
import type { Representation, BuildOptions } from "./types";
import type { StructureData } from "../data/types";
import type { MaterialSettings } from "../lighting";
import { getElementColor } from "../data/colors";
import { getElementByNumber } from "../data/elements";
import { createStyledMaterials, updateStyledMaterial } from "../materialStyles";
import type { StyledMaterials } from "../materialStyles";
import type { MaterialStyleType } from "../materialStyles";

const PROXY_MATERIAL = new THREE.MeshBasicMaterial({
  transparent: true,
  opacity: 0,
  depthWrite: false,
});

export class SpacefillRepresentation implements Representation {
  readonly type = "spacefill" as const;
  readonly group = new THREE.Group();

  private atomMesh: THREE.InstancedMesh | null = null;
  private _materials: StyledMaterials;

  // Ghost state
  private ghostAttr: THREE.InstancedBufferAttribute | null = null;

  // Visibility mapping
  private _visibleToOriginal: number[] = [];

  constructor(materialStyle: MaterialStyleType = "standard") {
    this._materials = createStyledMaterials(materialStyle);
  }

  build(data: StructureData, options?: BuildOptions): void {
    this.dispose();

    if (data.atoms.length === 0) return;

    const co = options?.colorOverride;
    const colorFn = options?.atomColorFn;
    const hidden = options?.hiddenElements;
    const hiddenIdx = options?.hiddenIndices;
    // Build visible atom list
    const visibleAtomIndices: number[] = [];
    for (let i = 0; i < data.atoms.length; i++) {
      if (hidden && hidden.has(data.atoms[i].atomicNumber)) continue;
      if (hiddenIdx && hiddenIdx.has(i)) continue;
      visibleAtomIndices.push(i);
    }
    this._visibleToOriginal = visibleAtomIndices;

    if (visibleAtomIndices.length === 0) return;

    const sphereGeo = new THREE.SphereGeometry(1, 32, 24);
    this.atomMesh = new THREE.InstancedMesh(sphereGeo, this._materials.atom, visibleAtomIndices.length);

    const matrix = new THREE.Matrix4();
    const color = new THREE.Color();

    for (let vi = 0; vi < visibleAtomIndices.length; vi++) {
      const i = visibleAtomIndices[vi];
      const atom = data.atoms[i];
      const radius = getElementByNumber(atom.atomicNumber).vdwRadius;

      matrix.makeScale(radius, radius, radius);
      matrix.setPosition(atom.position.x, atom.position.y, atom.position.z);
      this.atomMesh.setMatrixAt(vi, matrix);

      if (co) {
        color.setRGB(co[0], co[1], co[2], THREE.SRGBColorSpace);
      } else if (colorFn) {
        const [r, g, b] = colorFn(i);
        color.setRGB(r, g, b, THREE.SRGBColorSpace);
      } else {
        const [r, g, b] = getElementColor(atom.atomicNumber);
        color.setRGB(r, g, b, THREE.SRGBColorSpace);
      }
      this.atomMesh.setColorAt(vi, color);
    }

    // Per-instance ghost attribute
    const ghostArray = new Float32Array(visibleAtomIndices.length);
    if (data.ghostStart !== undefined && data.ghostStart >= 0) {
      for (let vi = 0; vi < visibleAtomIndices.length; vi++) {
        if (visibleAtomIndices[vi] >= data.ghostStart) {
          ghostArray[vi] = 1.0;
        }
      }
    }
    this.ghostAttr = new THREE.InstancedBufferAttribute(ghostArray, 1);
    this.atomMesh.geometry.setAttribute("aGhost", this.ghostAttr);

    this.atomMesh.instanceMatrix.needsUpdate = true;
    if (this.atomMesh.instanceColor) this.atomMesh.instanceColor.needsUpdate = true;
    this.group.add(this.atomMesh);
  }

  getAtomIndex(object: THREE.Object3D, instanceId: number): number {
    if (object === this.atomMesh) {
      return this._visibleToOriginal[instanceId] ?? -1;
    }
    return -1;
  }

  getBondAtoms(_object: THREE.Object3D, _instanceId: number): [number, number] | null {
    return null;
  }

  syncSelection(_added: number[], _removed: number[]): void {
    // Selection highlighting is handled by OutlinePass via proxy meshes
  }

  buildSelectionProxies(selectedIndices: ReadonlySet<number>): THREE.Object3D[] {
    if (selectedIndices.size === 0 || !this.atomMesh) return [];

    const origToVis = new Map<number, number>();
    for (let vi = 0; vi < this._visibleToOriginal.length; vi++) {
      origToVis.set(this._visibleToOriginal[vi], vi);
    }

    const selectedVis: number[] = [];
    for (const idx of selectedIndices) {
      const vi = origToVis.get(idx);
      if (vi !== undefined) selectedVis.push(vi);
    }
    if (selectedVis.length === 0) return [];

    const proxy = new THREE.InstancedMesh(
      this.atomMesh.geometry,
      PROXY_MATERIAL,
      selectedVis.length,
    );
    const mat = new THREE.Matrix4();
    for (let i = 0; i < selectedVis.length; i++) {
      this.atomMesh.getMatrixAt(selectedVis[i], mat);
      proxy.setMatrixAt(i, mat);
    }
    proxy.instanceMatrix.needsUpdate = true;
    return [proxy];
  }

  updatePositions(positions: Float32Array): void {
    if (!this.atomMesh) return;
    const arr = this.atomMesh.instanceMatrix.array as Float32Array;
    for (let vi = 0; vi < this._visibleToOriginal.length; vi++) {
      const ai = this._visibleToOriginal[vi];
      const px = positions[ai * 3];
      const py = positions[ai * 3 + 1];
      const pz = positions[ai * 3 + 2];
      // Instance matrix is 4x4 column-major; position is columns 12,13,14
      const offset = vi * 16;
      arr[offset + 12] = px;
      arr[offset + 13] = py;
      arr[offset + 14] = pz;
    }
    this.atomMesh.instanceMatrix.needsUpdate = true;
  }

  updateMaterial(settings: MaterialSettings): void {
    updateStyledMaterial(this._materials, settings.roughness, settings.metalness);
  }

  dispose(): void {
    if (this.atomMesh) {
      this.atomMesh.geometry.dispose();
      this.group.remove(this.atomMesh);
      this.atomMesh = null;
    }
    this.ghostAttr = null;
    this._visibleToOriginal = [];
  }
}
