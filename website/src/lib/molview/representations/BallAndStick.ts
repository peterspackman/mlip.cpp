import * as THREE from "three";
import type { Representation, BuildOptions } from "./types";
import type { StructureData } from "../data/types";
import type { MaterialSettings } from "../lighting";
import { getElementColor } from "../data/colors";
import { getElementByNumber } from "../data/elements";
import { createStyledMaterials, updateStyledMaterial } from "../materialStyles";
import type { StyledMaterials } from "../materialStyles";
import type { MaterialStyleType } from "../materialStyles";

const ATOM_SCALE = 0.3;
const BOND_RADIUS = 0.08;
const MIN_ATOM_RADIUS = BOND_RADIUS * 1.5;

/** Shared invisible material for selection proxy meshes.
 *  Invisible in beauty pass (opacity 0), but OutlinePass overrides materials
 *  so the geometry still renders into depth/mask buffers. */
const PROXY_MATERIAL = new THREE.MeshBasicMaterial({
  transparent: true,
  opacity: 0,
  depthWrite: false,
});

interface BondConnection {
  bondIndex: number;
  /** true if this atom is atomA of the bond */
  isA: boolean;
}

export class BallAndStickRepresentation implements Representation {
  readonly type = "ball+stick" as const;
  readonly group = new THREE.Group();

  private atomMesh: THREE.InstancedMesh | null = null;
  private bondMesh: THREE.InstancedMesh | null = null;
  private _materials: StyledMaterials;

  /** Maps atomIndex → bond connections for efficient proxy building */
  private atomBondMap = new Map<number, BondConnection[]>();
  /** Maps visible instance index → original atom index */
  private _visibleToOriginal: number[] = [];
  /** Visible bonds (filtered by hidden elements) */
  private _visibleBonds: StructureData["bonds"] = [];

  // Ghost state
  private atomGhostAttr: THREE.InstancedBufferAttribute | null = null;
  private bondGhostAttr: THREE.InstancedBufferAttribute | null = null;

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
    // Build set of visible atom indices and mapping from original to visible index
    const visibleAtomIndices: number[] = [];
    const originalToVisible = new Map<number, number>();
    for (let i = 0; i < data.atoms.length; i++) {
      if (hidden && hidden.has(data.atoms[i].atomicNumber)) continue;
      if (hiddenIdx && hiddenIdx.has(i)) continue;
      originalToVisible.set(i, visibleAtomIndices.length);
      visibleAtomIndices.push(i);
    }

    // Store mapping for getAtomIndex reverse lookup
    this._visibleToOriginal = visibleAtomIndices;

    // --- Atoms ---
    if (visibleAtomIndices.length > 0) {
      const sphereGeo = new THREE.SphereGeometry(1, 32, 24);
      this.atomMesh = new THREE.InstancedMesh(sphereGeo, this._materials.atom, visibleAtomIndices.length);

      const matrix = new THREE.Matrix4();
      const color = new THREE.Color();

      for (let vi = 0; vi < visibleAtomIndices.length; vi++) {
        const i = visibleAtomIndices[vi];
        const atom = data.atoms[i];
        const radius = Math.max(
          getElementByNumber(atom.atomicNumber).covalentRadius * ATOM_SCALE,
          MIN_ATOM_RADIUS,
        );

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

      // Per-instance ghost attribute (0.0 = real, 1.0 = ghost)
      const atomGhostArray = new Float32Array(visibleAtomIndices.length);
      if (data.ghostStart !== undefined && data.ghostStart >= 0) {
        for (let vi = 0; vi < visibleAtomIndices.length; vi++) {
          if (visibleAtomIndices[vi] >= data.ghostStart) {
            atomGhostArray[vi] = 1.0;
          }
        }
      }
      this.atomGhostAttr = new THREE.InstancedBufferAttribute(atomGhostArray, 1);
      this.atomMesh.geometry.setAttribute("aGhost", this.atomGhostAttr);

      this.atomMesh.instanceMatrix.needsUpdate = true;
      if (this.atomMesh.instanceColor) this.atomMesh.instanceColor.needsUpdate = true;
      this.group.add(this.atomMesh);
    }

    // --- Bonds (two half-cylinders per bond) ---
    this.atomBondMap.clear();

    // Filter bonds: skip bonds where either atom is hidden
    const visibleBonds: { bond: typeof data.bonds[0]; originalIndex: number }[] = [];
    for (let i = 0; i < data.bonds.length; i++) {
      const bond = data.bonds[i];
      if (hidden) {
        const aHidden = hidden.has(data.atoms[bond.atomA].atomicNumber);
        const bHidden = hidden.has(data.atoms[bond.atomB].atomicNumber);
        if (aHidden && bHidden) continue;
        // If only one end is hidden, also skip the bond
        if (aHidden || bHidden) continue;
      }
      visibleBonds.push({ bond, originalIndex: i });
    }

    if (visibleBonds.length > 0) {
      const cylinderGeo = new THREE.CylinderGeometry(1, 1, 1, 12);
      const bondCount = visibleBonds.length * 2;
      this.bondMesh = new THREE.InstancedMesh(cylinderGeo, this._materials.bond, bondCount);

      const matrix = new THREE.Matrix4();
      const color = new THREE.Color();
      const start = new THREE.Vector3();
      const end = new THREE.Vector3();
      const midpoint = new THREE.Vector3();
      const direction = new THREE.Vector3();
      const up = new THREE.Vector3(0, 1, 0);
      const quaternion = new THREE.Quaternion();

      for (let vi = 0; vi < visibleBonds.length; vi++) {
        const { bond } = visibleBonds[vi];
        const atomA = data.atoms[bond.atomA];
        const atomB = data.atoms[bond.atomB];

        // Build atom → bond lookup (using visible bond index)
        if (!this.atomBondMap.has(bond.atomA)) this.atomBondMap.set(bond.atomA, []);
        this.atomBondMap.get(bond.atomA)!.push({ bondIndex: vi, isA: true });
        if (!this.atomBondMap.has(bond.atomB)) this.atomBondMap.set(bond.atomB, []);
        this.atomBondMap.get(bond.atomB)!.push({ bondIndex: vi, isA: false });

        start.set(atomA.position.x, atomA.position.y, atomA.position.z);
        end.set(atomB.position.x, atomB.position.y, atomB.position.z);
        midpoint.addVectors(start, end).multiplyScalar(0.5);

        const fullLength = start.distanceTo(end);
        const halfLength = fullLength / 2;
        direction.subVectors(end, start).normalize();
        quaternion.setFromUnitVectors(up, direction);

        // Half A: atomA to midpoint
        const posA = new THREE.Vector3().addVectors(start, midpoint).multiplyScalar(0.5);
        matrix.makeRotationFromQuaternion(quaternion);
        matrix.scale(new THREE.Vector3(BOND_RADIUS, halfLength, BOND_RADIUS));
        matrix.setPosition(posA);
        this.bondMesh.setMatrixAt(vi * 2, matrix);

        if (co) {
          color.setRGB(co[0], co[1], co[2], THREE.SRGBColorSpace);
        } else if (colorFn) {
          const [rA, gA, bA] = colorFn(bond.atomA);
          color.setRGB(rA, gA, bA, THREE.SRGBColorSpace);
        } else {
          const [rA, gA, bA] = getElementColor(atomA.atomicNumber);
          color.setRGB(rA, gA, bA, THREE.SRGBColorSpace);
        }

        this.bondMesh.setColorAt(vi * 2, color);

        // Half B: midpoint to atomB
        const posB = new THREE.Vector3().addVectors(midpoint, end).multiplyScalar(0.5);
        matrix.makeRotationFromQuaternion(quaternion);
        matrix.scale(new THREE.Vector3(BOND_RADIUS, halfLength, BOND_RADIUS));
        matrix.setPosition(posB);
        this.bondMesh.setMatrixAt(vi * 2 + 1, matrix);

        if (co) {
          color.setRGB(co[0], co[1], co[2], THREE.SRGBColorSpace);
        } else if (colorFn) {
          const [rB, gB, bB] = colorFn(bond.atomB);
          color.setRGB(rB, gB, bB, THREE.SRGBColorSpace);
        } else {
          const [rB, gB, bB] = getElementColor(atomB.atomicNumber);
          color.setRGB(rB, gB, bB, THREE.SRGBColorSpace);
        }

        this.bondMesh.setColorAt(vi * 2 + 1, color);
      }

      // Store visible bonds for getBondAtoms lookup
      this._visibleBonds = visibleBonds.map(vb => vb.bond);

      // Per-instance ghost attribute for bonds
      const bondGhostArray = new Float32Array(bondCount);
      if (data.ghostStart !== undefined && data.ghostStart >= 0) {
        for (let vi = 0; vi < visibleBonds.length; vi++) {
          const { bond } = visibleBonds[vi];
          if (bond.atomA >= data.ghostStart) bondGhostArray[vi * 2] = 1.0;
          if (bond.atomB >= data.ghostStart) bondGhostArray[vi * 2 + 1] = 1.0;
        }
      }
      this.bondGhostAttr = new THREE.InstancedBufferAttribute(bondGhostArray, 1);
      this.bondMesh.geometry.setAttribute("aGhost", this.bondGhostAttr);

      this.bondMesh.instanceMatrix.needsUpdate = true;
      if (this.bondMesh.instanceColor) this.bondMesh.instanceColor.needsUpdate = true;
      this.group.add(this.bondMesh);
    }
  }

  getAtomIndex(object: THREE.Object3D, instanceId: number): number {
    if (object === this.atomMesh) {
      return this._visibleToOriginal[instanceId] ?? -1;
    }
    return -1;
  }

  getBondAtoms(object: THREE.Object3D, instanceId: number): [number, number] | null {
    if (object === this.bondMesh) {
      const bondIndex = Math.floor(instanceId / 2);
      if (bondIndex < this._visibleBonds.length) {
        const bond = this._visibleBonds[bondIndex];
        return [bond.atomA, bond.atomB];
      }
    }
    return null;
  }

  syncSelection(_added: number[], _removed: number[]): void {
    // Selection highlighting is handled by OutlinePass via proxy meshes
  }

  buildSelectionProxies(selectedIndices: ReadonlySet<number>): THREE.Object3D[] {
    const proxies: THREE.Object3D[] = [];
    if (selectedIndices.size === 0) return proxies;

    // Build reverse lookup
    const origToVis = new Map<number, number>();
    for (let vi = 0; vi < this._visibleToOriginal.length; vi++) {
      origToVis.set(this._visibleToOriginal[vi], vi);
    }

    // Atom proxy: small InstancedMesh with only selected atoms' transforms
    if (this.atomMesh) {
      const selectedVis: number[] = [];
      for (const idx of selectedIndices) {
        const vi = origToVis.get(idx);
        if (vi !== undefined) selectedVis.push(vi);
      }
      if (selectedVis.length > 0) {
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
        proxy.visible = true;
        proxies.push(proxy);
      }
    }

    // Bond half-cylinder proxies for selected atoms
    if (this.bondMesh) {
      const selectedBondInstances: number[] = [];
      for (const atomIdx of selectedIndices) {
        const connections = this.atomBondMap.get(atomIdx);
        if (!connections) continue;
        for (const { bondIndex, isA } of connections) {
          selectedBondInstances.push(isA ? bondIndex * 2 : bondIndex * 2 + 1);
        }
      }
      if (selectedBondInstances.length > 0) {
        const unique = [...new Set(selectedBondInstances)];
        const proxy = new THREE.InstancedMesh(
          this.bondMesh.geometry,
          PROXY_MATERIAL,
          unique.length,
        );
        const mat = new THREE.Matrix4();
        for (let i = 0; i < unique.length; i++) {
          this.bondMesh.getMatrixAt(unique[i], mat);
          proxy.setMatrixAt(i, mat);
        }
        proxy.instanceMatrix.needsUpdate = true;
        proxy.visible = true;
        proxies.push(proxy);
      }
    }

    return proxies;
  }

  updatePositions(positions: Float32Array): void {
    // Update atom positions in instance matrices
    if (this.atomMesh) {
      const arr = this.atomMesh.instanceMatrix.array as Float32Array;
      for (let vi = 0; vi < this._visibleToOriginal.length; vi++) {
        const ai = this._visibleToOriginal[vi];
        const offset = vi * 16;
        arr[offset + 12] = positions[ai * 3];
        arr[offset + 13] = positions[ai * 3 + 1];
        arr[offset + 14] = positions[ai * 3 + 2];
      }
      this.atomMesh.instanceMatrix.needsUpdate = true;
    }

    // Recompute bond half-cylinder transforms
    if (this.bondMesh && this._visibleBonds.length > 0) {
      const matrix = new THREE.Matrix4();
      const start = new THREE.Vector3();
      const end = new THREE.Vector3();
      const midpoint = new THREE.Vector3();
      const direction = new THREE.Vector3();
      const up = new THREE.Vector3(0, 1, 0);
      const quaternion = new THREE.Quaternion();

      for (let vi = 0; vi < this._visibleBonds.length; vi++) {
        const bond = this._visibleBonds[vi];
        start.set(positions[bond.atomA * 3], positions[bond.atomA * 3 + 1], positions[bond.atomA * 3 + 2]);
        end.set(positions[bond.atomB * 3], positions[bond.atomB * 3 + 1], positions[bond.atomB * 3 + 2]);
        midpoint.addVectors(start, end).multiplyScalar(0.5);

        const fullLength = start.distanceTo(end);
        const halfLength = fullLength / 2;
        direction.subVectors(end, start).normalize();
        quaternion.setFromUnitVectors(up, direction);

        // Half A: atomA to midpoint
        const posA = new THREE.Vector3().addVectors(start, midpoint).multiplyScalar(0.5);
        matrix.makeRotationFromQuaternion(quaternion);
        matrix.scale(new THREE.Vector3(BOND_RADIUS, halfLength, BOND_RADIUS));
        matrix.setPosition(posA);
        this.bondMesh.setMatrixAt(vi * 2, matrix);

        // Half B: midpoint to atomB
        const posB = new THREE.Vector3().addVectors(midpoint, end).multiplyScalar(0.5);
        matrix.makeRotationFromQuaternion(quaternion);
        matrix.scale(new THREE.Vector3(BOND_RADIUS, halfLength, BOND_RADIUS));
        matrix.setPosition(posB);
        this.bondMesh.setMatrixAt(vi * 2 + 1, matrix);
      }
      this.bondMesh.instanceMatrix.needsUpdate = true;
    }
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
    if (this.bondMesh) {
      this.bondMesh.geometry.dispose();
      this.group.remove(this.bondMesh);
      this.bondMesh = null;
    }
    this.atomGhostAttr = null;
    this.bondGhostAttr = null;
    this.atomBondMap.clear();
    this._visibleToOriginal = [];
    this._visibleBonds = [];
  }
}
