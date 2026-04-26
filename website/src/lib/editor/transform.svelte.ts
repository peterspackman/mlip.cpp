// Modal transform controller (Blender-style).
//
// Lifecycle of one operation:
//   begin('grab', x, y)      → snapshot positions, switch from idle
//   update(x, y)             → recompute delta, mutate store.positions live
//   setAxis('x'|'y'|'z')     → toggle axis constraint, recompute
//   type('1') / type('.')... → numeric input takes over from mouse
//   commit()                 → serialize + push to worker
//   cancel()                 → restore snapshot, no worker call
//
// The controller owns transient state and mutates store.positions directly so
// the existing reactive viewer redraws each frame. The worker only sees the
// final committed positions.

import * as THREE from 'three'
import type { SimulationStore } from '../stores/simulation.svelte'
import { detectBonds } from '../chem/bonds'

export type TransformMode = 'idle' | 'grab' | 'rotate'
export type Axis = 'x' | 'y' | 'z' | 'bond' | null

const AXIS_VEC: Record<'x' | 'y' | 'z', THREE.Vector3> = {
  x: new THREE.Vector3(1, 0, 0),
  y: new THREE.Vector3(0, 1, 0),
  z: new THREE.Vector3(0, 0, 1),
}

function isNumericKey(k: string): boolean {
  return /^[\d.\-]$/.test(k)
}

function parseBuffer(buf: string): number | null {
  if (!buf || buf === '-' || buf === '.') return null
  const v = parseFloat(buf)
  return isNaN(v) ? null : v
}

export class TransformController {
  // Reactive state — read by the overlay component.
  mode = $state<TransformMode>('idle')
  axis = $state<Axis>(null)
  numericBuffer = $state('')
  /** Human-readable summary of the current delta — for the overlay. */
  display = $state('')

  private store: SimulationStore
  private getCamera: () => THREE.Camera | null
  private getCanvas: () => HTMLCanvasElement | null

  // Snapshot taken at begin().
  private snap: Float64Array | null = null
  private snapSelection: number[] = []
  private centroid = new THREE.Vector3()
  private mouseStart = new THREE.Vector2()
  private mouseCurrent = new THREE.Vector2()
  /** Running unwrapped screen-angle since begin (free rotate). Updated every
   *  recompute so wraps at ±π don't cause 360° jumps. */
  private rotateAccumAngle = 0
  /** Last raw screen-angle around the projected pivot — for delta unwrapping. */
  private rotatePrevAngle = 0
  /** Pixels of perpendicular drag → 1 degree of rotation. Tuned to roughly
   *  match Blender's R-drag feel. */
  private static readonly ROTATE_PIX_PER_DEG = 3
  /** True until the first mouse move after begin(). Used to rebase the start
   *  position when the transform was kicked off by keyboard with the mouse
   *  off-canvas — otherwise the first pointermove yields a huge spurious delta. */
  private firstUpdate = false
  /** Cached bond direction (unit vector) when axis === 'bond'. */
  private bondDir = new THREE.Vector3()
  /** Reactive: candidate bonds available for Tab-cycling under axis='bond'. */
  bondCandidates = $state<Array<[number, number]>>([])
  /** Reactive: index into bondCandidates; the active bond. */
  bondIndex = $state(0)

  /** Read by the viewer to draw the constraint axis line in 3D. The origin
   *  matches the actual rotation/translation pivot. */
  get constraintInfo(): { vec: THREE.Vector3; origin: THREE.Vector3 } | null {
    const v = this.constraintVec()
    if (!v) return null
    return { vec: v, origin: this.activePivot().clone() }
  }

  /** When axis='bond', the [atomA, atomB] of the active bond, or null. */
  get activeBond(): [number, number] | null {
    if (this.axis !== 'bond') return null
    return this.bondCandidates[this.bondIndex] ?? null
  }

  constructor(
    store: SimulationStore,
    getCamera: () => THREE.Camera | null,
    getCanvas: () => HTMLCanvasElement | null,
  ) {
    this.store = store
    this.getCamera = getCamera
    this.getCanvas = getCanvas
  }

  get active(): boolean { return this.mode !== 'idle' }

  /** Try to start a transform. Returns false if there's nothing to transform. */
  begin(mode: 'grab' | 'rotate', startX: number, startY: number): boolean {
    if (this.store.selectedAtoms.size === 0) return false
    if (!this.store.positions) return false
    if (this.store.isRunning) return false

    this.mode = mode
    this.axis = null
    this.numericBuffer = ''
    this.snap = new Float64Array(this.store.positions)
    this.snapSelection = [...this.store.selectedAtoms]
    this.computeCentroid()
    this.mouseStart.set(startX, startY)
    this.mouseCurrent.copy(this.mouseStart)
    this.firstUpdate = true
    if (mode === 'rotate') {
      this.rotateAccumAngle = 0
      this.rotatePrevAngle = this.angleAroundPivot(this.mouseStart)
    }
    this.recompute()
    return true
  }

  /**
   * Switch the active transform to a different mode without exiting. Bakes
   * any in-progress delta into a fresh snapshot so the new mode accumulates
   * on top — Blender's behavior when you press G then R mid-grab.
   * Returns false if not active or already in the requested mode.
   */
  switchMode(newMode: 'grab' | 'rotate'): boolean {
    if (!this.active) return false
    if (this.mode === newMode) return false
    if (this.store.positions) {
      this.snap = new Float64Array(this.store.positions)
      this.computeCentroid()  // re-center on the moved positions
    }
    this.mode = newMode
    this.axis = null
    this.numericBuffer = ''
    this.firstUpdate = true
    this.recompute()
    return true
  }

  update(x: number, y: number): void {
    if (!this.active) return
    if (this.firstUpdate) {
      // Keyboard-initiated transform — rebase to wherever the mouse actually is.
      this.mouseStart.set(x, y)
      if (this.mode === 'rotate') {
        this.rotateAccumAngle = 0
        this.rotatePrevAngle = this.angleAroundPivot(this.mouseStart)
      }
      this.firstUpdate = false
    }
    this.mouseCurrent.set(x, y)
    this.recompute()
  }

  setAxis(axis: Exclude<Axis, null>): void {
    if (!this.active) return
    if (axis === 'bond') {
      // (Re)collect candidates every time bond is engaged so Tab cycling works.
      this.collectBondCandidates()
      if (!this.applyActiveBondDir()) return  // no bond context — leave state alone
    }
    this.axis = this.axis === axis ? null : axis
    this.recompute()
  }

  /** Cycle through Tab-available bonds. Direction +1 (Tab) or -1 (Shift+Tab). */
  cycleBond(direction: 1 | -1): boolean {
    if (!this.active || this.axis !== 'bond') return false
    const n = this.bondCandidates.length
    if (n <= 1) return false
    this.bondIndex = (this.bondIndex + direction + n) % n
    this.applyActiveBondDir()
    this.recompute()
    return true
  }

  /** Build the candidate-bond list. Includes:
   *    1. The user-clicked bond (first, if present).
   *    2. Depth-1: bonds with at least one endpoint in the selection.
   *    3. Depth-2: bonds incident to neighbors of the selection — so e.g.
   *       selecting just the H of an -OH gives access to the C-O bond, which
   *       is what you actually want to rotate the H around.
   */
  private collectBondCandidates(): void {
    if (!this.snap || this.snapSelection.length === 0) {
      this.bondCandidates = []
      this.bondIndex = 0
      return
    }
    const sel = new Set(this.snapSelection)
    const seen = new Set<string>()
    const candidates: Array<[number, number]> = []
    const push = (a: number, b: number) => {
      const key = a < b ? `${a}-${b}` : `${b}-${a}`
      if (seen.has(key)) return
      seen.add(key)
      candidates.push([a, b])
    }

    const sb = this.store.selectedBond
    if (sb) push(sb[0], sb[1])

    // detectBonds in chem/bonds.ts is 1-indexed for SDF — shift back here.
    const rawBonds = detectBonds(this.snap, this.store.atomicNumbers)
    const bonds: Array<[number, number]> = rawBonds.map(([a, b]) => [a - 1, b - 1])

    // Depth 1, while collecting selection-neighbor atoms for the depth-2 pass.
    const neighbors = new Set<number>(sel)
    for (const [a, b] of bonds) {
      if (sel.has(a) || sel.has(b)) {
        push(a, b)
        neighbors.add(a)
        neighbors.add(b)
      }
    }

    // Depth 2.
    for (const [a, b] of bonds) {
      if (neighbors.has(a) || neighbors.has(b)) push(a, b)
    }

    this.bondCandidates = candidates
    this.bondIndex = 0
  }

  /** Pivot used by the active transform. For axis='bond' this snaps to a
   *  point on the bond (the anchor = the endpoint NOT in the selection;
   *  midpoint if both in; nearest endpoint if neither) so rotating a single
   *  atom around an external bond actually moves it. Otherwise = centroid. */
  private activePivot(): THREE.Vector3 {
    if (this.axis !== 'bond' || !this.snap) return this.centroid
    const c = this.bondCandidates[this.bondIndex]
    if (!c) return this.centroid
    const [a, b] = c
    const sel = new Set(this.snapSelection)
    const pa = new THREE.Vector3(this.snap[a * 3], this.snap[a * 3 + 1], this.snap[a * 3 + 2])
    const pb = new THREE.Vector3(this.snap[b * 3], this.snap[b * 3 + 1], this.snap[b * 3 + 2])
    const aIn = sel.has(a), bIn = sel.has(b)
    if (aIn && !bIn) return pb
    if (bIn && !aIn) return pa
    if (aIn && bIn) return pa.clone().add(pb).multiplyScalar(0.5)
    const dA = pa.distanceToSquared(this.centroid)
    const dB = pb.distanceToSquared(this.centroid)
    return dA <= dB ? pa : pb
  }

  private applyActiveBondDir(): boolean {
    const c = this.bondCandidates[this.bondIndex]
    if (!c) return false
    return this.setBondDirFromIndices(c[0], c[1])
  }

  private setBondDirFromIndices(i: number, j: number): boolean {
    const p = this.snap!
    if (i * 3 + 2 >= p.length || j * 3 + 2 >= p.length) return false
    this.bondDir.set(
      p[j * 3]     - p[i * 3],
      p[j * 3 + 1] - p[i * 3 + 1],
      p[j * 3 + 2] - p[i * 3 + 2],
    )
    if (this.bondDir.lengthSq() < 1e-12) return false
    this.bondDir.normalize()
    return true
  }

  private constraintVec(): THREE.Vector3 | null {
    if (!this.axis) return null
    if (this.axis === 'bond') return this.bondDir.clone()
    return AXIS_VEC[this.axis].clone()
  }

  type(key: string): void {
    if (!this.active) return
    if (key === 'Backspace') {
      this.numericBuffer = this.numericBuffer.slice(0, -1)
    } else if (isNumericKey(key)) {
      const next = this.numericBuffer + key
      // Only accept partial-valid numeric strings.
      if (next === '-' || next === '.' || next === '-.' || /^-?\d*\.?\d*$/.test(next)) {
        // Reject double minus/dot.
        if ((next.match(/-/g)?.length ?? 0) > 1) return
        if ((next.match(/\./g)?.length ?? 0) > 1) return
        this.numericBuffer = next
      }
    } else {
      return
    }
    this.recompute()
  }

  async commit(): Promise<void> {
    if (!this.active) return
    if (!this.store.positions) { this.reset(); return }
    const finalPositions = new Float64Array(this.store.positions)
    const movedIndices = [...this.snapSelection]
    this.reset()
    // Free-form transform: drop the selection but flash a fading cue at the
    // moved atoms' new positions so the user sees what just changed without
    // the orange outline lingering.
    await this.store.editCommitPositions(finalPositions, new Set())
    if (movedIndices.length > 0) this.store.triggerFlash(movedIndices)
  }

  cancel(): void {
    if (!this.active) return
    if (this.snap) this.store.positions = new Float64Array(this.snap)
    this.reset()
  }

  // --- internals -----------------------------------------------------------

  private reset(): void {
    this.mode = 'idle'
    this.axis = null
    this.numericBuffer = ''
    this.display = ''
    this.snap = null
    this.snapSelection = []
    this.bondCandidates = []
    this.bondIndex = 0
  }

  private computeCentroid(): void {
    const p = this.snap
    if (!p || this.snapSelection.length === 0) {
      this.centroid.set(0, 0, 0)
      return
    }
    let x = 0, y = 0, z = 0
    for (const i of this.snapSelection) {
      x += p[i * 3]
      y += p[i * 3 + 1]
      z += p[i * 3 + 2]
    }
    const n = this.snapSelection.length
    this.centroid.set(x / n, y / n, z / n)
  }

  private recompute(): void {
    if (this.mode === 'grab')   this.recomputeGrab()
    else if (this.mode === 'rotate') this.recomputeRotate()
  }

  private recomputeGrab(): void {
    const camera = this.getCamera()
    const canvas = this.getCanvas()
    if (!camera || !canvas || !this.snap) return

    let delta = new THREE.Vector3()
    const numeric = parseBuffer(this.numericBuffer)

    if (numeric !== null) {
      // Numeric input — needs a constraint vector. Default to X if user hasn't
      // picked one (matches Blender, which defaults to last-used axis).
      const v = this.constraintVec() ?? AXIS_VEC.x
      delta.copy(v).multiplyScalar(numeric)
    } else {
      // Mouse-driven. Cast both mouse positions onto a working plane through
      // the centroid and take the world-space difference.
      const planeNormal = this.workingPlaneNormal()
      const pStart = unprojectToPlane(this.mouseStart, camera, canvas, this.centroid, planeNormal)
      const pCur   = unprojectToPlane(this.mouseCurrent, camera, canvas, this.centroid, planeNormal)
      if (pStart && pCur) {
        delta.subVectors(pCur, pStart)
        const v = this.constraintVec()
        if (v) {
          delta = v.multiplyScalar(delta.dot(v))
        }
      }
    }

    this.applyTranslation(delta)

    const ax = this.axisLabel()
    const num = this.numericBuffer ? `  [${this.numericBuffer}]` : ''
    const dx = delta.x.toFixed(2), dy = delta.y.toFixed(2), dz = delta.z.toFixed(2)
    this.display = `Grab${ax}  Δ = (${dx}, ${dy}, ${dz}) Å${num}`
  }

  private recomputeRotate(): void {
    const camera = this.getCamera()
    const canvas = this.getCanvas()
    if (!camera || !canvas || !this.snap) return

    // Axis: explicit constraint, or camera-forward by default.
    const rotAxis = this.constraintVec() ?? new THREE.Vector3()
    const isFree = rotAxis.lengthSq() < 1e-9
    if (isFree) camera.getWorldDirection(rotAxis).negate()  // toward viewer
    rotAxis.normalize()

    // Update the screen-angle accumulator on every tick so wraps at ±π
    // never cause a 360° jump, and so switching back to free rotate after a
    // constrained drag picks up smoothly.
    const cur = this.angleAroundPivot(this.mouseCurrent)
    let dAng = cur - this.rotatePrevAngle
    if (dAng > Math.PI) dAng -= 2 * Math.PI
    else if (dAng < -Math.PI) dAng += 2 * Math.PI
    this.rotateAccumAngle += dAng
    this.rotatePrevAngle = cur

    let angleRad = 0
    const numeric = parseBuffer(this.numericBuffer)
    if (numeric !== null) {
      angleRad = (numeric * Math.PI) / 180
    } else if (isFree) {
      // Free rotate (axis = view direction): screen-angle around the pivot
      // is the natural mapping — turning the wheel.
      angleRad = this.rotateAccumAngle
    } else {
      // Axis-constrained: drag perpendicular to the projected axis maps to
      // angle. Gain is uniform regardless of distance from the pivot, which
      // is what makes constrained rotation feel consistent across views.
      angleRad = this.computeAxisAngle(rotAxis, camera, canvas)
    }

    this.applyRotation(angleRad, rotAxis)

    const ax = this.axis ? ` around ${this.axisLabel().trim()}` : ' (view)'
    const deg = (angleRad * 180 / Math.PI).toFixed(1)
    const num = this.numericBuffer ? `  [${this.numericBuffer}]` : ''
    this.display = `Rotate${ax}  ${deg}°${num}`
  }

  /** Perpendicular-drag → angle for an axis-constrained rotation. The axis
   *  is projected to the screen; the mouse displacement perpendicular to
   *  the projected axis is scaled to degrees. Falls back to the unwrapped
   *  screen-angle when the axis projects to too few pixels (axis ≈ along
   *  view direction — a degenerate case for perp drag). */
  private computeAxisAngle(axis: THREE.Vector3, camera: THREE.Camera, canvas: HTMLCanvasElement): number {
    const pivot = this.activePivot()
    const sp = this.projectToScreenPx(pivot, camera, canvas)
    const sa = this.projectToScreenPx(pivot.clone().add(axis), camera, canvas)
    const ax = sa.x - sp.x
    const ay = sa.y - sp.y
    const axisLenSq = ax * ax + ay * ay
    if (axisLenSq < 25) {
      // Axis ≈ along view; the projected axis is a dot, perp is undefined.
      // Fall back to free-rotate's accumulated screen-angle.
      return this.rotateAccumAngle
    }
    const inv = 1 / Math.sqrt(axisLenSq)
    // Perpendicular to the projected axis (90° in screen-y-down coords).
    const perpX = -ay * inv
    const perpY =  ax * inv
    const dx = this.mouseCurrent.x - this.mouseStart.x
    const dy = this.mouseCurrent.y - this.mouseStart.y
    const perpPx = dx * perpX + dy * perpY
    // Sign convention: when the axis points away from the camera, the
    // right-hand rotation viewed from the camera matches the screen-coord
    // sign of perpPx. When the axis points toward the camera, flip it.
    const camDir = new THREE.Vector3()
    camera.getWorldDirection(camDir)
    const sign = camDir.dot(axis) >= 0 ? 1 : -1
    return sign * perpPx * (Math.PI / 180) / TransformController.ROTATE_PIX_PER_DEG
  }

  private projectToScreenPx(p: THREE.Vector3, camera: THREE.Camera, canvas: HTMLCanvasElement): { x: number; y: number } {
    const ndc = p.clone().project(camera)
    return {
      x: (ndc.x + 1) * 0.5 * canvas.clientWidth,
      y: (1 - ndc.y) * 0.5 * canvas.clientHeight,
    }
  }

  private axisLabel(): string {
    if (!this.axis) return ''
    if (this.axis === 'bond') {
      const b = this.activeBond
      if (!b) return ' along bond'
      const total = this.bondCandidates.length
      const cyc = total > 1 ? ` (${this.bondIndex + 1}/${total}, tab to cycle)` : ''
      return ` along bond ${b[0]}-${b[1]}${cyc}`
    }
    return ` along ${this.axis.toUpperCase()}`
  }

  private applyTranslation(delta: THREE.Vector3): void {
    if (!this.snap) return
    const next = new Float64Array(this.snap)
    for (const i of this.snapSelection) {
      next[i * 3]     = this.snap[i * 3]     + delta.x
      next[i * 3 + 1] = this.snap[i * 3 + 1] + delta.y
      next[i * 3 + 2] = this.snap[i * 3 + 2] + delta.z
    }
    this.store.positions = next
  }

  private applyRotation(angleRad: number, axis: THREE.Vector3): void {
    if (!this.snap) return
    const next = new Float64Array(this.snap)
    if (Math.abs(angleRad) < 1e-9) {
      this.store.positions = next
      return
    }
    const pivot = this.activePivot()
    const q = new THREE.Quaternion().setFromAxisAngle(axis, angleRad)
    const v = new THREE.Vector3()
    for (const i of this.snapSelection) {
      v.set(this.snap[i * 3], this.snap[i * 3 + 1], this.snap[i * 3 + 2])
      v.sub(pivot).applyQuaternion(q).add(pivot)
      next[i * 3]     = v.x
      next[i * 3 + 1] = v.y
      next[i * 3 + 2] = v.z
    }
    this.store.positions = next
  }

  private workingPlaneNormal(): THREE.Vector3 {
    const camera = this.getCamera()
    const camDir = new THREE.Vector3()
    if (camera) camera.getWorldDirection(camDir)
    if (camDir.lengthSq() < 1e-9) camDir.set(0, 0, 1)

    const axis = this.constraintVec()
    if (!axis) return camDir.normalize()

    // Constrained: plane that contains the axis and is most face-on to camera.
    const cross = new THREE.Vector3().crossVectors(axis, camDir)
    if (cross.lengthSq() < 1e-9) return camDir.normalize()
    return new THREE.Vector3().crossVectors(cross, axis).normalize()
  }

  /** Screen-space angle from the projected rotation pivot (selection
   *  centroid, or the bond endpoint when axis='bond') to a canvas-px point. */
  private angleAroundPivot(p: THREE.Vector2): number {
    const camera = this.getCamera()
    const canvas = this.getCanvas()
    if (!camera || !canvas) return 0
    const sp = this.projectToScreenPx(this.activePivot(), camera, canvas)
    return Math.atan2(p.y - sp.y, p.x - sp.x)
  }
}

function unprojectToPlane(
  px: THREE.Vector2,
  camera: THREE.Camera,
  canvas: HTMLCanvasElement,
  pointOnPlane: THREE.Vector3,
  planeNormal: THREE.Vector3,
): THREE.Vector3 | null {
  const ndc = new THREE.Vector2(
    (px.x / canvas.clientWidth) * 2 - 1,
    -(px.y / canvas.clientHeight) * 2 + 1,
  )
  const raycaster = new THREE.Raycaster()
  raycaster.setFromCamera(ndc, camera)
  const plane = new THREE.Plane(planeNormal.clone().normalize(), -planeNormal.clone().normalize().dot(pointOnPlane))
  const out = new THREE.Vector3()
  const hit = raycaster.ray.intersectPlane(plane, out)
  return hit ? out : null
}
