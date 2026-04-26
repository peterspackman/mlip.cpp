import * as THREE from "three";
import type { StructureObject } from "./StructureObject";

export type PickResult =
  | { type: "atom"; atomIndex: number; entryId?: number }
  | { type: "bond"; atomA: number; atomB: number; entryId?: number }
  | { type: "surface"; entryId: number; surfaceIndex: number };

export interface PickTarget {
  obj: StructureObject;
  entryId: number;
}

export class SelectionManager {
  private raycaster = new THREE.Raycaster();
  private mouse = new THREE.Vector2();
  /** Distance of the last successful pick (Infinity if no hit) */
  lastHitDistance = Infinity;

  constructor() {
    // Enable impostor layer so raycaster can pick impostor meshes
    this.raycaster.layers.enable(31);
  }

  /** Pick from a single structure (backward-compatible) */
  pick(
    event: MouseEvent,
    camera: THREE.Camera,
    canvas: HTMLCanvasElement,
    structureObj: StructureObject | null,
  ): PickResult | null {
    if (!structureObj) return null;
    return this.pickMulti(event, camera, canvas, [{ obj: structureObj, entryId: -1 }]);
  }

  /** Pick from multiple structures, returning the closest hit with entryId */
  pickMulti(
    event: MouseEvent,
    camera: THREE.Camera,
    canvas: HTMLCanvasElement,
    targets: PickTarget[],
  ): PickResult | null {
    if (targets.length === 0) return null;

    const rect = canvas.getBoundingClientRect();
    this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    this.raycaster.setFromCamera(this.mouse, camera);

    // Collect all intersections across all targets with their entryId
    const allHits: { hit: THREE.Intersection; target: PickTarget }[] = [];
    for (const target of targets) {
      const intersects = this.raycaster.intersectObject(target.obj.group, true);
      for (const hit of intersects) {
        allHits.push({ hit, target });
      }
    }

    // Sort by distance
    allHits.sort((a, b) => a.hit.distance - b.hit.distance);

    this.lastHitDistance = Infinity;

    for (const { hit, target } of allHits) {
      if (hit.instanceId === undefined) continue;

      const atomIndex = target.obj.representation.getAtomIndex(
        hit.object,
        hit.instanceId,
      );
      if (atomIndex >= 0) {
        this.lastHitDistance = hit.distance;
        return { type: "atom", atomIndex, entryId: target.entryId };
      }

      const bondAtoms = target.obj.representation.getBondAtoms(
        hit.object,
        hit.instanceId,
      );
      if (bondAtoms) {
        this.lastHitDistance = hit.distance;
        return { type: "bond", atomA: bondAtoms[0], atomB: bondAtoms[1], entryId: target.entryId };
      }
    }

    return null;
  }
}
