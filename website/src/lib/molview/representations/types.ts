import type * as THREE from "three";
import type { StructureData } from "../data/types";
import type { MaterialSettings } from "../lighting";
import type { MaterialStyleType } from "../materialStyles";

export type RepresentationType = "tube" | "ball+stick" | "spacefill" | "wireframe" | "ellipsoid";

/** Maps a display atom index to an sRGB [0-1] color triple. */
export type AtomColorFn = (atomIndex: number) => [number, number, number];

export interface BuildOptions {
  colorOverride?: [number, number, number] | null;
  /** Per-atom color function. Overridden by colorOverride. */
  atomColorFn?: AtomColorFn | null;
  hiddenElements?: Set<number>;
  /** Specific atom indices to hide (suppressed atoms) */
  hiddenIndices?: ReadonlySet<number>;
  /** Ellipsoid probability (0–1), default 0.5. Only used by ellipsoid representation. */
  probability?: number;
  /** Material shading style. Default "standard". */
  materialStyle?: MaterialStyleType;
}

export interface Representation {
  readonly type: RepresentationType;
  readonly group: THREE.Group;
  build(data: StructureData, options?: BuildOptions): void;
  /** Get the atom index for a given InstancedMesh intersection, or -1 */
  getAtomIndex(object: THREE.Object3D, instanceId: number): number;
  /** Get both atom indices for a bond mesh intersection, or null */
  getBondAtoms(object: THREE.Object3D, instanceId: number): [number, number] | null;
  /** Apply selection state: highlight added, unhighlight removed */
  syncSelection(added: number[], removed: number[]): void;
  /** Build lightweight proxy meshes for the given selected atom indices (for OutlinePass) */
  buildSelectionProxies(selectedIndices: ReadonlySet<number>): THREE.Object3D[];
  updateMaterial(settings: MaterialSettings): void;
  /** Update atom positions in-place without full rebuild (for trajectory playback) */
  updatePositions?(positions: Float32Array): void;
  dispose(): void;
}
