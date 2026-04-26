import * as THREE from "three";
import type { StructureData } from "./data/types";
import type { MaterialSettings } from "./lighting";
import type { Representation, RepresentationType, AtomColorFn } from "./representations/types";
import { BallAndStickRepresentation } from "./representations/BallAndStick";
import { SpacefillRepresentation } from "./representations/Spacefill";
import { WireframeRepresentation } from "./representations/Wireframe";
import { TubeRepresentation } from "./representations/Tube";
import { ImpostorSpacefillRepresentation } from "./representations/ImpostorSpacefill";
import { ImpostorBallAndStickRepresentation } from "./representations/ImpostorBallAndStick";
import type { MaterialStyleType } from "./materialStyles";
import { type ImpostorMode, type ImpostorSettings, DEFAULT_IMPOSTOR_SETTINGS } from "./impostorSettings";
export type { ImpostorMode, ImpostorSettings };
export { DEFAULT_IMPOSTOR_SETTINGS };

/** Whether to use impostor (ray-cast) rendering for this rep type and atom count */
function shouldUseImpostor(type: RepresentationType, atomCount: number, settings: ImpostorSettings): boolean {
  if (type !== "spacefill" && type !== "ball+stick") return false;
  switch (settings.mode) {
    case "always": return true;
    case "never": return false;
    case "auto": return atomCount >= settings.threshold;
  }
}

function createRepresentation(type: RepresentationType, materialStyle: MaterialStyleType, useImpostor: boolean): Representation {
  if (useImpostor) {
    switch (type) {
      case "ball+stick": return new ImpostorBallAndStickRepresentation();
      case "spacefill": return new ImpostorSpacefillRepresentation();
    }
  }
  switch (type) {
    case "tube": return new TubeRepresentation(materialStyle);
    case "ball+stick": return new BallAndStickRepresentation(materialStyle);
    case "spacefill": return new SpacefillRepresentation(materialStyle);
    case "wireframe": return new WireframeRepresentation();
    case "ellipsoid": throw new Error("ellipsoid representation not vendored (requires ADP data)");
  }
}

export class StructureObject {
  data: StructureData;
  representation: Representation;
  readonly group = new THREE.Group();
  /** Proxy meshes for OutlinePass (visible=false, only used by post-processing) */
  readonly selectionProxyGroup = new THREE.Group();
  private _colorOverride: [number, number, number] | null = null;
  private _atomColorFn: AtomColorFn | null = null;
  private _hiddenElements: Set<number> | undefined = undefined;
  private _hiddenIndices: ReadonlySet<number> | undefined = undefined;
  private _probability: number | undefined = undefined;
  private _materialStyle: MaterialStyleType = "standard";
  private _repType: RepresentationType = "ball+stick";
  private _useImpostor = false;
  private _impostorSettings: ImpostorSettings = { ...DEFAULT_IMPOSTOR_SETTINGS };
  private _selectionIndices: Set<number> = new Set();
  private _selectionProxies: THREE.Object3D[] = [];

  constructor(data: StructureData, repType: RepresentationType = "ball+stick", hiddenElements?: Set<number>, materialStyle: MaterialStyleType = "standard") {
    this.data = data;
    this._hiddenElements = hiddenElements;
    this._materialStyle = materialStyle;
    this._repType = repType;
    this._useImpostor = shouldUseImpostor(repType, data.atoms.length, this._impostorSettings);
    this.representation = createRepresentation(repType, materialStyle, this._useImpostor);
    this.representation.build(data, this._buildOptions());
    this.group.add(this.representation.group);
  }

  /** True if impostor rendering is currently active (auto for large systems) */
  get isImpostorActive(): boolean {
    return this._useImpostor;
  }

  setRepresentation(type: RepresentationType): void {
    if (this._repType === type) return;
    this._repType = type;
    this._rebuildRepresentation();
  }

  /**
   * Force a full representation rebuild. Use after upstream data the
   * representation reads (e.g. element colours, radii) has changed.
   * Note: bond list is not re-detected — covalent-radius edits affect
   * bond appearance but not bond presence until the structure is reloaded.
   */
  rebuild(): void {
    this._rebuildRepresentation();
  }

  setMaterialStyle(style: MaterialStyleType): void {
    if (this._materialStyle === style) return;
    this._materialStyle = style;
    this._rebuildRepresentation();
  }

  get materialStyle(): MaterialStyleType {
    return this._materialStyle;
  }

  setImpostorSettings(settings: ImpostorSettings): void {
    if (this._impostorSettings.mode === settings.mode &&
        this._impostorSettings.threshold === settings.threshold) return;
    this._impostorSettings = { ...settings };
    this._rebuildRepresentation();
  }

  get impostorSettings(): ImpostorSettings {
    return this._impostorSettings;
  }

  private _rebuildRepresentation(): void {
    this.group.remove(this.representation.group);
    this.representation.dispose();

    this._useImpostor = shouldUseImpostor(this._repType, this.data.atoms.length, this._impostorSettings);
    this.representation = createRepresentation(this._repType, this._materialStyle, this._useImpostor);
    this.representation.build(this.data, this._buildOptions());
    this.group.add(this.representation.group);
  }

  /** Set a flat color override for all atoms/bonds, or null to restore element colors */
  setColorOverride(color: [number, number, number] | null): void {
    if (
      this._colorOverride === color ||
      (this._colorOverride && color &&
        this._colorOverride[0] === color[0] &&
        this._colorOverride[1] === color[1] &&
        this._colorOverride[2] === color[2])
    ) return;

    this._colorOverride = color;
    this._rebuild();
  }

  /** Set a per-atom color function, or null to restore element colors */
  setAtomColorFn(fn: AtomColorFn | null): void {
    if (this._atomColorFn === fn) return;
    this._atomColorFn = fn;
    this._rebuild();
  }

  /** Set suppressed atom indices (hidden but not removed) */
  setSuppressedAtoms(indices: ReadonlySet<number> | undefined): void {
    this._hiddenIndices = indices?.size ? indices : undefined;
    this._rebuild();
  }

  /** Set ellipsoid probability (0–1). Triggers rebuild if in ellipsoid mode. */
  setEllipsoidProbability(p: number): void {
    if (this._probability === p) return;
    this._probability = p;
    if (this.representation.type === "ellipsoid") {
      this._rebuild();
    }
  }

  private _buildOptions() {
    return {
      colorOverride: this._colorOverride,
      atomColorFn: this._atomColorFn,
      hiddenElements: this._hiddenElements,
      hiddenIndices: this._hiddenIndices,
      probability: this._probability,
    };
  }

  /** Rebuild representation with current options, preserving selection state */
  private _rebuild(): void {
    this.group.remove(this.representation.group);
    this.representation.dispose();

    this.representation = createRepresentation(this._repType, this._materialStyle, this._useImpostor);
    this.representation.build(this.data, this._buildOptions());
    this.group.add(this.representation.group);

    // Re-apply selection if there was one
    if (this._selectionIndices.size > 0) {
      this.representation.syncSelection([...this._selectionIndices], []);
      this._rebuildSelectionProxies();
    }
  }

  /** Apply a full selection (all indices treated as added from empty) */
  applyFullSelection(indices: ReadonlySet<number>): void {
    this._selectionIndices = new Set(indices);
    this.representation.syncSelection([...indices], []);
    this._rebuildSelectionProxies();
  }

  /** Incremental selection sync */
  syncSelection(added: number[], removed: number[]): void {
    for (const i of added) this._selectionIndices.add(i);
    for (const i of removed) this._selectionIndices.delete(i);
    this.representation.syncSelection(added, removed);
    this._rebuildSelectionProxies();
  }

  /** Rebuild proxy meshes used by OutlinePass for selection outline */
  private _rebuildSelectionProxies(): void {
    // Remove old proxies
    for (const p of this._selectionProxies) {
      this.selectionProxyGroup.remove(p);
    }
    this._selectionProxies = [];

    if (this._selectionIndices.size === 0) return;

    const proxies = this.representation.buildSelectionProxies(this._selectionIndices);
    for (const p of proxies) {
      this.selectionProxyGroup.add(p);
    }
    this._selectionProxies = proxies;
  }

  /** Toggle hydrogen atom visibility */
  setHydrogenVisible(visible: boolean): void {
    if (visible) {
      if (!this._hiddenElements || !this._hiddenElements.has(1)) return;
      this._hiddenElements.delete(1);
      if (this._hiddenElements.size === 0) this._hiddenElements = undefined;
    } else {
      if (!this._hiddenElements) this._hiddenElements = new Set();
      if (this._hiddenElements.has(1)) return;
      this._hiddenElements.add(1);
    }
    this._rebuild();
  }

  /** Update atom positions in-place without full rebuild (for trajectory playback).
   *  positions is a flat Float32Array: [x0,y0,z0, x1,y1,z1, ...] */
  updatePositions(positions: Float32Array): void {
    // Sync data model
    for (let i = 0; i < this.data.atoms.length; i++) {
      this.data.atoms[i].position.x = positions[i * 3];
      this.data.atoms[i].position.y = positions[i * 3 + 1];
      this.data.atoms[i].position.z = positions[i * 3 + 2];
    }
    // Update GPU buffers in-place
    if (this.representation.updatePositions) {
      this.representation.updatePositions(positions);
    }
  }

  updateMaterial(settings: MaterialSettings): void {
    this.representation.updateMaterial(settings);
  }

  centerOfMass(): THREE.Vector3 {
    const center = new THREE.Vector3();
    if (this.data.atoms.length === 0) return center;

    for (const atom of this.data.atoms) {
      center.x += atom.position.x;
      center.y += atom.position.y;
      center.z += atom.position.z;
    }
    center.divideScalar(this.data.atoms.length);
    return center;
  }

  dispose(): void {
    this.group.remove(this.representation.group);
    this.representation.dispose();
    for (const p of this._selectionProxies) {
      this.selectionProxyGroup.remove(p);
    }
    this._selectionProxies = [];
  }
}
