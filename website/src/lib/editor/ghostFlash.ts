// Brief "atoms just moved here" cue. Renders a transparent halo around each
// moved atom — same color and roughly the same radius as the atom itself —
// fading to zero over ~0.8 s. The caller supplies per-instance positions,
// colors, and radii so the cue reads as a faded copy of the atoms rather
// than uniform orange blobs.
//
// We use a Group of plain Meshes (one per atom) instead of a single
// InstancedMesh + setColorAt, because:
//   1. Selection counts here are small (≤ a few dozen typically) so the
//      perf gap is irrelevant.
//   2. Per-instance color via setColorAt has been finicky to get working
//      reliably across Three.js versions — invisible-but-no-error states
//      are easy to fall into.
// Sharing a single SphereGeometry across all flashes keeps GPU memory tame.

import * as THREE from 'three'

export interface GhostFlashOptions {
  durationMs?: number
  /** Multiplier on the supplied per-atom radius — slightly >1 so the halo
   *  reads as a corona around the atom, not a replacement. */
  radiusScale?: number
  startOpacity?: number
}

interface ActiveFlash {
  group: THREE.Group
  materials: THREE.MeshBasicMaterial[]
  startTime: number
  duration: number
  startOpacity: number
}

const SHARED_GEOMETRY = new THREE.SphereGeometry(1, 12, 8)

export class GhostFlash {
  private parent: THREE.Object3D
  private requestRender: () => void
  private active: ActiveFlash[] = []
  private rafId: number | null = null
  private opts: Required<GhostFlashOptions>

  constructor(parent: THREE.Object3D, requestRender: () => void, opts: GhostFlashOptions = {}) {
    this.parent = parent
    this.requestRender = requestRender
    this.opts = {
      durationMs: opts.durationMs ?? 800,
      radiusScale: opts.radiusScale ?? 1.18,
      startOpacity: opts.startOpacity ?? 0.55,
    }
  }

  flash(positions: Float32Array, colors: Float32Array, radii: Float32Array): void {
    const n = radii.length
    if (n <= 0) return
    if (positions.length < n * 3 || colors.length < n * 3) return

    const group = new THREE.Group()
    group.frustumCulled = false
    const materials: THREE.MeshBasicMaterial[] = []
    const k = this.opts.radiusScale

    for (let i = 0; i < n; i++) {
      const mat = new THREE.MeshBasicMaterial({
        color: new THREE.Color(colors[i * 3], colors[i * 3 + 1], colors[i * 3 + 2]),
        transparent: true,
        opacity: this.opts.startOpacity,
        depthWrite: false,
      })
      materials.push(mat)
      const mesh = new THREE.Mesh(SHARED_GEOMETRY, mat)
      const r = radii[i] * k
      mesh.scale.set(r, r, r)
      mesh.position.set(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2])
      mesh.frustumCulled = false
      group.add(mesh)
    }
    this.parent.add(group)

    this.active.push({
      group,
      materials,
      startTime: performance.now(),
      duration: this.opts.durationMs,
      startOpacity: this.opts.startOpacity,
    })
    this.tick()
  }

  dispose(): void {
    if (this.rafId !== null) {
      cancelAnimationFrame(this.rafId)
      this.rafId = null
    }
    for (const f of this.active) this.disposeFlash(f)
    this.active = []
  }

  private tick(): void {
    if (this.rafId !== null) return
    const step = () => {
      const now = performance.now()
      let stillRunning = false
      for (let i = this.active.length - 1; i >= 0; i--) {
        const f = this.active[i]
        const t = (now - f.startTime) / f.duration
        if (t >= 1) {
          this.disposeFlash(f)
          this.active.splice(i, 1)
          continue
        }
        const op = f.startOpacity * (1 - t) * (1 - t)
        for (const m of f.materials) m.opacity = op
        stillRunning = true
      }
      this.requestRender()
      if (stillRunning) this.rafId = requestAnimationFrame(step)
      else this.rafId = null
    }
    this.rafId = requestAnimationFrame(step)
  }

  private disposeFlash(f: ActiveFlash): void {
    this.parent.remove(f.group)
    for (const m of f.materials) m.dispose()
    // SHARED_GEOMETRY persists across flashes — do not dispose.
  }
}
