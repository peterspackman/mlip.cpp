// Reactive store for the MD demo state. Uses Svelte 5 runes — every `$state`
// field is tracked, so components that read these fields rerender when they
// change. Constructed once in App.svelte and passed down through context.

import { Simulation, type Backend, type Thermostat, type Optimizer, type MDStep, type OptStep } from '../worker/simulation'
import type { Lattice } from '../chem/cell'
import { parsePositions, parseAtomicNumbers, serializeXyz } from '../chem/xyz'
import { detectBonds } from '../chem/bonds'
import type { Fragment } from '../editor/fragments'
import { fillHydrogens } from '../editor/hydrogens'
import { getElementByNumber } from '../molview/data/elements'
import * as THREE from 'three'

/** Same scale the ball-and-stick representation uses for atom radii. Keeps
 *  ghost halos visually paired with the atoms they trail. */
const GHOST_ATOM_SCALE = 0.3
import { getMass } from '../chem/elements'
import { computeVibrations, type VibMode, type VibProgress } from '../vib/modes'

export type Mode = 'md' | 'optimize' | 'vib'
export type ModelStatus = 'empty' | 'loading' | 'ready' | 'error'

interface EditSnapshot {
  xyz: string
  lattice: Lattice | null
  bondOverrides: { add: string[]; remove: string[] }
  selectedAtoms: number[]
  selectedBond: [number, number] | null
  selectionGroups: Record<number, number[]>
}

interface EditClipboard {
  atomicNumbers: number[]
  /** Positions stored relative to the selection centroid at copy time, so
   *  the same fragment can be pasted at any anchor without drift. */
  relPositions: Float64Array
}

export class SimulationStore {
  readonly sim: Simulation

  // True once the WASM worker has booted. Structure load + edit only need this;
  // a model is required to compute energies/forces (run / optimize / vib).
  wasmReady = $state(false)

  // Model
  modelStatus = $state<ModelStatus>('empty')
  modelType = $state('')
  modelSource = $state('')
  activeBackend = $state('')
  backendChoice = $state<Backend>(defaultBackend())
  modelError = $state('')

  // Structure
  numAtoms = $state(0)
  atomicNumbers = $state<number[]>([])
  isPeriodic = $state(false)
  lattice = $state<Lattice | null>(null)
  positions = $state<Float64Array | null>(null)
  cell = $state<Float64Array | null>(null)
  currentXyz = $state('')

  // Simulation control
  mode = $state<Mode>('md')
  isRunning = $state(false)
  step = $state(0)
  lastStep = $state<MDStep | null>(null)
  lastOpt = $state<OptStep | null>(null)

  // MD parameters
  temperature = $state(300)
  timestep = $state(1.0)
  thermostat = $state<Thermostat>('none')
  useConservativeForces = $state(true)

  // Optimization parameters
  optimizer = $state<Optimizer>('lbfgs')
  activeOptimizer = $state<Optimizer | null>(null)  // what the worker actually picked
  optimizerForced = $state(false)                    // true when routing overrode the user's pick
  maxOptSteps = $state(100)
  forceThreshold = $state(0.05)
  rattleAmount = $state(0.1)
  optimizationConverged = $state(false)

  // Readouts
  energy = $state(0)
  kineticEnergy = $state(0)
  currentTemperature = $state(0)
  maxForce = $state(0)
  maxStress = $state(0)
  energyDrift = $state(0)
  msPerStep = $state(0)
  energyHistory = $state<number[]>([])

  // Viewer
  viewStyle = $state<'ball+stick' | 'spacefill' | 'wireframe' | 'tube'>('ball+stick')
  /** Scene background color as a `#rrggbb` string. Hex format keeps it in
   *  the same shape as a native `<input type="color">` value. */
  viewerBackground = $state('#1a1a2e')
  wrapPositions = $state(true)
  supercell = $state<[number, number, number]>([2, 2, 2])
  /** When true (default), bonds are re-detected from distances on every
   *  position update — bonds appear/disappear as atoms move (good for MD,
   *  good for spotting an over-stretched bond mid-edit). When false, bonds
   *  are frozen to whatever was detected at structure load — useful in the
   *  editor when you don't want bonds flickering as you nudge atoms. */
  dynamicBonds = $state(true)

  // App mode + editor selection
  appMode = $state<'sim' | 'edit'>('sim')
  /** Selected atom indices (against the *displayed* atom list, which may be
   *  a supercell expansion of the canonical atoms). */
  selectedAtoms = $state<Set<number>>(new Set())
  /** Last-clicked bond as a pair of atom indices, or null. */
  selectedBond = $state<[number, number] | null>(null)
  /** Saved selections, keyed by slot 1..9. Recall via the digit key, save
   *  with shift+digit. Reset when the structure rebuilds. */
  selectionGroups = $state<Record<number, number[]>>({})
  /** Manual bond toggles applied on top of distance-based detection.
   *  Keys are "i-j" with i < j, referencing canonical atom indices.
   *  `add`: force a bond to render even if atoms are too far apart.
   *  `remove`: hide a detected bond. Cleared whenever the atom count changes. */
  bondOverrides = $state<{ add: string[]; remove: string[] }>({ add: [], remove: [] })

  // Vibrational analysis
  vibComputing = $state(false)
  vibProgress = $state<VibProgress | null>(null)
  vibModes = $state<VibMode[]>([])
  vibEquilibrium = $state<Float64Array | null>(null)
  vibError = $state('')
  activeMode = $state<number | null>(null)
  vibAmplitude = $state(0.3)    // max atomic displacement, Å
  vibPlaying = $state(false)
  vibPeriodMs = $state(1500)    // one oscillation = 1.5 s by default
  vibOptimizeFirst = $state(true)
  vibProjectTrRot = $state(true)
  vibShowImaginary = $state(true)
  vibNProjected = $state(0)
  vibOptStep = $state(0)
  vibOptMaxForce = $state(0)

  // Edit history (undo/redo). Each snapshot is the full editor-visible state
  // at a point in time; undo applies a snapshot via setStructure(). Stacks
  // are cleared whenever a fresh structure is loaded from outside the editor.
  canUndo = $state(false)
  canRedo = $state(false)
  private undoStack: EditSnapshot[] = []
  private redoStack: EditSnapshot[] = []
  private undoSuspended = false
  private static readonly UNDO_CAP = 50

  // One-shot signal: counter increments each time we want the viewer to
  // pulse a "moved-atom" cue. Counter pattern keeps $effect re-firing for
  // repeat flashes; the payload is non-reactive so consumers don't get
  // re-triggered by unrelated position updates.
  flashTrigger = $state(0)
  flashPayload: { positions: Float32Array; colors: Float32Array; radii: Float32Array } | null = null
  /** Indices that were the target of the current/last optimize run, captured
   *  at start. Used to flash + clear on stop without losing the pre-relax
   *  selection. */
  private optScope: number[] = []

  /** In-memory clipboard for copy / paste / duplicate. Lives across the
   *  session but not across reloads — molecule editing is short-lived. */
  clipboard = $state<EditClipboard | null>(null)

  private lastStepTime = 0
  private animationFrameId: number | null = null
  private animationStart = 0

  constructor() {
    this.sim = new Simulation()
    this.sim.on((ev) => this.onEvent(ev))
  }

  async initialize() {
    await this.sim.ready()
    await this.sim.init()
    await this.syncParameters()
    this.wasmReady = true
  }

  private onEvent(ev: Parameters<Parameters<Simulation['on']>[0]>[0]) {
    switch (ev.kind) {
      case 'mdStep': {
        const now = performance.now()
        this.msPerStep = this.lastStepTime > 0 ? now - this.lastStepTime : 0
        this.lastStepTime = now
        const s = ev.step
        this.step++
        this.lastStep = s
        this.energy = s.energy
        this.kineticEnergy = s.kineticEnergy
        this.currentTemperature = s.temperature
        this.energyDrift = s.energyDrift
        this.positions = s.positions
        const total = s.energy + s.kineticEnergy
        this.energyHistory = [...this.energyHistory.slice(-99), total]
        break
      }
      case 'optStep': {
        const now = performance.now()
        this.msPerStep = this.lastStepTime > 0 ? now - this.lastStepTime : 0
        this.lastStepTime = now
        const s = ev.step
        this.lastOpt = s
        this.step = s.step
        this.energy = s.energy
        this.maxForce = s.maxForce
        this.maxStress = s.maxStress ?? 0
        this.positions = s.positions
        if (s.cell) this.cell = s.cell
        this.optimizationConverged = s.converged
        this.energyHistory = [...this.energyHistory.slice(-99), s.energy]
        break
      }
      case 'rattled':
        this.positions = ev.positions
        break
      case 'started':
        this.isRunning = true
        break
      case 'stopped':
        this.isRunning = false
        // Sync currentXyz to whatever the worker drove positions to during
        // the run — otherwise XYZ export and the next edit's undo snapshot
        // would still reflect the pre-run geometry, mismatching what's on
        // screen.
        if (this.positions && this.atomicNumbers.length > 0) {
          this.currentXyz = serializeXyz(this.positions, this.atomicNumbers, this.lattice)
        }
        // Flash the relax target atoms at their final positions, then clear
        // the persistent selection. (Only when an opt run actually had a
        // captured scope — MD stops, vib stops, etc. keep selection alone.)
        if (this.optScope.length > 0) {
          const scope = this.optScope
          this.optScope = []
          this.triggerFlash(scope)
          this.selectedAtoms = new Set()
          this.selectedBond = null
        }
        break
      case 'optimizerStarted':
        this.activeOptimizer = ev.optimizer
        this.optimizerForced = ev.forced
        break
      case 'error':
        this.modelError = ev.message
        this.isRunning = false
        // If we were mid-load, surface the error in the model status line.
        if (this.modelStatus === 'loading') this.modelStatus = 'error'
        break
    }
  }

  async loadModel(buffer: ArrayBuffer, source: string) {
    this.modelStatus = 'loading'
    this.modelSource = source
    this.modelError = ''
    try {
      const info = await this.sim.loadModel(buffer, this.backendChoice)
      this.modelType = info.modelType
      this.activeBackend = info.backend
      this.modelStatus = 'ready'
      // loadModel clears the worker's system; if the user already loaded /
      // edited a structure, re-push it so they don't lose it.
      if (this.currentXyz) {
        try {
          await this.sim.setSystem(this.currentXyz)
        } catch {
          /* surfaced via the error event handler */
        }
      }
    } catch (err: any) {
      this.modelStatus = 'error'
      this.modelError = err?.message ?? String(err)
    }
  }

  // Swap the currently loaded model onto a different backend. Called by the
  // UI when the user flips the backend selector after a model is already
  // loaded — the worker re-instantiates using its cached GGUF bytes.
  async switchBackend() {
    if (this.modelStatus !== 'ready') return
    this.modelStatus = 'loading'
    this.modelError = ''
    if (this.isRunning) this.sim.stop()
    try {
      const info = await this.sim.setBackend(this.backendChoice)
      if (info.backend) this.activeBackend = info.backend
      this.modelStatus = 'ready'
    } catch (err: any) {
      this.modelStatus = 'error'
      this.modelError = err?.message ?? String(err)
    }
  }

  async setStructure(xyz: string, lattice: Lattice | null) {
    // Invalidate any vib analysis we have for the previous structure.
    this.clearVibrations()
    const info = await this.sim.setSystem(xyz)
    const oldN = this.atomicNumbers.length
    // Parse atoms/positions client-side — the worker sets them internally but
    // doesn't ship them back over the wire, and the viewer needs them to draw.
    this.atomicNumbers = parseAtomicNumbers(xyz)
    this.positions = new Float64Array(parsePositions(xyz))
    // When the atom count changes (load, delete, add, fragment) we drop
    // every kind of state that was indexed by atom number. Position-only
    // edits — element swap, translate, bond-length, manual bond toggle —
    // keep the same count and preserve selections / saved slots / overrides.
    if (this.atomicNumbers.length !== oldN) {
      this.bondOverrides = { add: [], remove: [] }
      this.selectedAtoms = new Set()
      this.selectedBond = null
      this.selectionGroups = {}
    }
    this.numAtoms = info.numAtoms
    this.isPeriodic = info.isPeriodic
    this.lattice = lattice
    this.currentXyz = xyz
    if (lattice) {
      this.cell = new Float64Array([
        ...lattice.a, ...lattice.b, ...lattice.c,
      ])
    } else {
      this.cell = null
    }
    this.step = 0
    this.lastStepTime = 0
    this.energyHistory = []
    this.energy = 0
    this.kineticEnergy = 0
    this.energyDrift = 0
    this.currentTemperature = 0
    this.optimizationConverged = false
  }

  // ---- editor mutations ---------------------------------------------------
  // All edits read the current canonical structure (asym unit), build the
  // mutated atomicNumbers + positions, serialize to XYZ, and re-push via
  // setStructure() so the worker stays in sync. Selection is cleared on edit
  // because indices may shift (delete) or new atoms may need to be selected.

  private canEdit(): boolean {
    if (this.isRunning) return false
    if (!this.positions || this.atomicNumbers.length === 0) return false
    return true
  }

  async editDeleteAtoms(indices: ReadonlySet<number>) {
    if (!this.canEdit() || indices.size === 0) return
    const n = this.atomicNumbers.length
    const keep: number[] = []
    for (let i = 0; i < n; i++) if (!indices.has(i)) keep.push(i)
    if (keep.length === n) return
    this.pushUndoIfActive()
    const newZ = keep.map((i) => this.atomicNumbers[i])
    const newPos = new Float64Array(keep.length * 3)
    for (let k = 0; k < keep.length; k++) {
      const src = keep[k] * 3
      newPos[k * 3]     = this.positions![src]
      newPos[k * 3 + 1] = this.positions![src + 1]
      newPos[k * 3 + 2] = this.positions![src + 2]
    }
    const xyz = serializeXyz(newPos, newZ, this.lattice)
    this.selectedAtoms = new Set()
    this.selectedBond = null
    await this.setStructure(xyz, this.lattice)
  }

  async editSetElement(indices: ReadonlySet<number>, z: number) {
    if (!this.canEdit() || indices.size === 0) return
    this.pushUndoIfActive()
    const newZ = [...this.atomicNumbers]
    for (const i of indices) {
      if (i >= 0 && i < newZ.length) newZ[i] = z
    }
    const xyz = serializeXyz(this.positions!, newZ, this.lattice)
    const keepSelection = new Set(indices)
    await this.setStructure(xyz, this.lattice)
    this.selectedAtoms = keepSelection
  }

  async editAddAtom(z: number, position: [number, number, number]) {
    if (this.isRunning) return
    this.pushUndoIfActive()
    const oldZ = this.atomicNumbers
    const oldP = this.positions
    const n = oldZ.length
    const newZ = [...oldZ, z]
    const newPos = new Float64Array((n + 1) * 3)
    if (oldP) newPos.set(oldP)
    newPos[n * 3]     = position[0]
    newPos[n * 3 + 1] = position[1]
    newPos[n * 3 + 2] = position[2]
    const xyz = serializeXyz(newPos, newZ, this.lattice)
    await this.setStructure(xyz, this.lattice)
    this.selectedAtoms = new Set([n])
    this.selectedBond = null
  }

  async editTranslate(indices: ReadonlySet<number>, delta: [number, number, number]) {
    if (!this.canEdit() || indices.size === 0) return
    const newPos = new Float64Array(this.positions!)
    for (const i of indices) {
      if (i < 0 || i * 3 + 2 >= newPos.length) continue
      newPos[i * 3]     += delta[0]
      newPos[i * 3 + 1] += delta[1]
      newPos[i * 3 + 2] += delta[2]
    }
    await this.editCommitPositions(newPos, indices)
  }

  /**
   * Change the length of bond (a, b) by translating the connected fragment on
   * b's side along the bond direction. Quick & dirty z-matrix-style: BFS over
   * the bond graph from b, refusing to traverse back through a, and shifts
   * everything reachable. For cyclic structures the fragment loops back
   * through other paths and the geometry distorts — surface this caveat in
   * the UI rather than silently misbehaving.
   *
   * TODO: full internal-coordinate (z-matrix) edits with proper handling of
   * rings, hydrogens, and dependent angles. The "drag the fragment" approach
   * here covers the common acyclic case (alkyl chains, side groups).
   */
  async editSetBondLength(a: number, b: number, newLength: number, side: 'a' | 'b' = 'b') {
    if (!this.canEdit()) return
    const p = this.positions!
    if (a < 0 || b < 0) return
    if (a * 3 + 2 >= p.length || b * 3 + 2 >= p.length) return
    const dx = p[b * 3]     - p[a * 3]
    const dy = p[b * 3 + 1] - p[a * 3 + 1]
    const dz = p[b * 3 + 2] - p[a * 3 + 2]
    const L = Math.hypot(dx, dy, dz)
    if (L < 1e-6 || !Number.isFinite(newLength) || newLength < 0.1) return
    let ux = dx / L, uy = dy / L, uz = dz / L
    const delta = newLength - L

    // Pick the moving side and the anchor.
    const movingFrom = side === 'b' ? b : a
    const anchor = side === 'b' ? a : b
    const sign = side === 'b' ? 1 : -1
    ux *= sign; uy *= sign; uz *= sign

    // BFS over bonds from movingFrom, refusing to cross through `anchor`.
    const adj = new Map<number, number[]>()
    const bonds = detectBonds(p, this.atomicNumbers)
    for (const [a1, b1] of bonds) {
      const i = a1 - 1, j = b1 - 1
      if (!adj.has(i)) adj.set(i, [])
      if (!adj.has(j)) adj.set(j, [])
      adj.get(i)!.push(j)
      adj.get(j)!.push(i)
    }
    const visited = new Set<number>([anchor])
    const queue = [movingFrom]
    visited.add(movingFrom)
    while (queue.length > 0) {
      const cur = queue.shift()!
      for (const n of adj.get(cur) ?? []) {
        if (!visited.has(n)) {
          visited.add(n)
          queue.push(n)
        }
      }
    }
    visited.delete(anchor)

    const newPos = new Float64Array(p)
    for (const i of visited) {
      newPos[i * 3]     += ux * delta
      newPos[i * 3 + 1] += uy * delta
      newPos[i * 3 + 2] += uz * delta
    }
    await this.editCommitPositions(newPos, this.selectedAtoms)
  }

  /**
   * Toggle a bond between canonical atoms a and b. Layered on top of
   * distance-based detection: if the bond is currently visible, this hides it;
   * if it isn't, this forces it on. Driven by the `F` keybind.
   */
  toggleBond(a: number, b: number): void {
    if (a === b || a < 0 || b < 0) return
    if (!this.positions || a >= this.atomicNumbers.length || b >= this.atomicNumbers.length) return
    this.pushUndoIfActive()
    const lo = Math.min(a, b), hi = Math.max(a, b)
    const key = `${lo}-${hi}`

    // Snapshot detected bonds at the canonical positions (no supercell) so
    // toggle decisions match what's drawn in the unit cell.
    const detected = detectBonds(this.positions, this.atomicNumbers)
    const detectedHas = detected.some(([a1, b1]) => {
      const i = a1 - 1, j = b1 - 1
      return (i === lo && j === hi) || (i === hi && j === lo)
    })

    const inAdd = this.bondOverrides.add.includes(key)
    const inRemove = this.bondOverrides.remove.includes(key)
    const visible = (detectedHas && !inRemove) || inAdd

    let nextAdd = this.bondOverrides.add.filter((k) => k !== key)
    let nextRemove = this.bondOverrides.remove.filter((k) => k !== key)
    if (visible) {
      if (detectedHas) nextRemove.push(key)
      // else: the only thing making it visible was nextAdd, which we already cleared.
    } else {
      if (!detectedHas) nextAdd.push(key)
      // else: the only thing hiding it was nextRemove, which we already cleared.
    }
    this.bondOverrides = { add: nextAdd, remove: nextRemove }
  }

  /**
   * Replace the atom at `atomIdx` with a fragment. The fragment's attach atom
   * takes the slot of the replaced atom (so existing indices, including any
   * other selected atoms, remain valid). The rest of the fragment is appended.
   * The fragment is oriented so its local +X aligns with the world "outward"
   * direction — namely (selected atom → its first neighbor) negated. Falls
   * back to +z if the atom has no neighbors.
   */
  async editReplaceWithFragment(atomIdx: number, fragment: Fragment) {
    if (!this.canEdit()) return
    if (atomIdx < 0 || atomIdx >= this.atomicNumbers.length) return
    this.pushUndoIfActive()
    const p = this.positions!
    const anchor = new THREE.Vector3(p[atomIdx * 3], p[atomIdx * 3 + 1], p[atomIdx * 3 + 2])

    // Outward direction = away from the first detected bonded neighbor.
    const outward = new THREE.Vector3(0, 0, 1)
    const bonds = detectBonds(p, this.atomicNumbers)
    for (const [a1, b1] of bonds) {
      const a = a1 - 1, b = b1 - 1
      let neighbor = -1
      if (a === atomIdx) neighbor = b
      else if (b === atomIdx) neighbor = a
      if (neighbor >= 0) {
        outward.set(
          anchor.x - p[neighbor * 3],
          anchor.y - p[neighbor * 3 + 1],
          anchor.z - p[neighbor * 3 + 2],
        )
        break
      }
    }
    if (outward.lengthSq() < 1e-12) outward.set(0, 0, 1)
    outward.normalize()

    const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(1, 0, 0), outward)

    const oldN = this.atomicNumbers.length
    const fragAtoms = fragment.atoms
    const attachIdx = fragment.attachIndex
    const extraCount = fragAtoms.length - 1
    const newZ = [...this.atomicNumbers]
    newZ[atomIdx] = fragAtoms[attachIdx].z

    const newPos = new Float64Array((oldN + extraCount) * 3)
    newPos.set(p)
    // Attach atom: local (0,0,0) → world anchor; the slot already holds the
    // anchor's coords from the original atom, so nothing to overwrite.

    const tmp = new THREE.Vector3()
    let writeIdx = oldN
    for (let i = 0; i < fragAtoms.length; i++) {
      if (i === attachIdx) continue
      const lp = fragAtoms[i].pos
      tmp.set(lp[0], lp[1], lp[2]).applyQuaternion(q)
      newPos[writeIdx * 3]     = anchor.x + tmp.x
      newPos[writeIdx * 3 + 1] = anchor.y + tmp.y
      newPos[writeIdx * 3 + 2] = anchor.z + tmp.z
      newZ.push(fragAtoms[i].z)
      writeIdx++
    }

    const xyz = serializeXyz(newPos, newZ, this.lattice)
    await this.setStructure(xyz, this.lattice)
    this.selectedAtoms = new Set([atomIdx])
    this.selectedBond = null
  }

  /**
   * Replace the current positions wholesale (atom count and species unchanged)
   * and re-push the structure to the worker. Used by the modal transform
   * controller to commit a live drag/rotate.
   */
  // ---- clipboard ---------------------------------------------------------

  /** Copy the selected atoms into the clipboard, normalized to their centroid. */
  editCopy(indices: ReadonlySet<number> = this.selectedAtoms): boolean {
    if (!this.positions || indices.size === 0) return false
    const idx = [...indices].filter((i) => i >= 0 && i < this.atomicNumbers.length)
    if (idx.length === 0) return false
    let cx = 0, cy = 0, cz = 0
    for (const i of idx) {
      cx += this.positions[i * 3]
      cy += this.positions[i * 3 + 1]
      cz += this.positions[i * 3 + 2]
    }
    cx /= idx.length; cy /= idx.length; cz /= idx.length
    const relPositions = new Float64Array(idx.length * 3)
    const atomicNumbers: number[] = []
    for (let k = 0; k < idx.length; k++) {
      const i = idx[k]
      relPositions[k * 3]     = this.positions[i * 3]     - cx
      relPositions[k * 3 + 1] = this.positions[i * 3 + 1] - cy
      relPositions[k * 3 + 2] = this.positions[i * 3 + 2] - cz
      atomicNumbers.push(this.atomicNumbers[i])
    }
    this.clipboard = { atomicNumbers, relPositions }
    return true
  }

  /** Paste the clipboard into the structure. Anchor: current selection's
   *  centroid + small +X offset, or world origin if nothing is selected.
   *  Newly-pasted atoms become the selection so the user can immediately
   *  G-drag them to a precise location. */
  async editPaste(opts: { offset?: [number, number, number] } = {}): Promise<boolean> {
    if (!this.canEdit() || !this.clipboard) return false
    const { atomicNumbers: addZ, relPositions } = this.clipboard
    const oldN = this.atomicNumbers.length
    const addN = addZ.length

    let ax = 0, ay = 0, az = 0
    if (this.selectedAtoms.size > 0 && this.positions) {
      let n = 0
      for (const i of this.selectedAtoms) {
        if (i < 0 || i * 3 + 2 >= this.positions.length) continue
        ax += this.positions[i * 3]
        ay += this.positions[i * 3 + 1]
        az += this.positions[i * 3 + 2]
        n++
      }
      if (n > 0) { ax /= n; ay /= n; az /= n }
    }
    const off = opts.offset ?? [1.5, 0, 0]
    ax += off[0]; ay += off[1]; az += off[2]

    this.pushUndoIfActive()
    const newPos = new Float64Array((oldN + addN) * 3)
    if (this.positions) newPos.set(this.positions)
    for (let k = 0; k < addN; k++) {
      newPos[(oldN + k) * 3]     = ax + relPositions[k * 3]
      newPos[(oldN + k) * 3 + 1] = ay + relPositions[k * 3 + 1]
      newPos[(oldN + k) * 3 + 2] = az + relPositions[k * 3 + 2]
    }
    const newZ = [...this.atomicNumbers, ...addZ]
    const xyz = serializeXyz(newPos, newZ, this.lattice)
    await this.setStructure(xyz, this.lattice)
    const sel = new Set<number>()
    for (let k = 0; k < addN; k++) sel.add(oldN + k)
    this.selectedAtoms = sel
    this.selectedBond = null
    return true
  }

  /** Copy + paste in one shot, leaving the duplicates as the new selection.
   *  Bypasses the persistent clipboard so an outstanding clipboard payload
   *  isn't clobbered by a quick duplicate gesture. */
  async editDuplicate(opts: { offset?: [number, number, number] } = {}): Promise<boolean> {
    if (!this.canEdit() || this.selectedAtoms.size === 0) return false
    const saved = this.clipboard
    const ok = this.editCopy()
    if (!ok) { this.clipboard = saved; return false }
    await this.editBatch(async () => {
      await this.editPaste(opts)
    })
    this.clipboard = saved
    return true
  }

  /**
   * Add hydrogens to satisfy each heavy atom's standard neutral valence.
   * Geometry is extrapolated from existing neighbor directions (linear
   * extension, tetrahedral / methylene fans, etc.) — no charge or formal
   * bond-order perception. If `scope` has any atoms, only those heavy atoms
   * are filled; otherwise all heavy atoms in the structure are filled.
   */
  async editFillHydrogens(scope: ReadonlySet<number> | null = null) {
    if (!this.canEdit()) return
    const { added, count } = fillHydrogens(this.positions!, this.atomicNumbers, scope)
    if (count === 0) return
    this.pushUndoIfActive()
    const oldN = this.atomicNumbers.length
    const newPos = new Float64Array(this.positions!.length + added.length)
    newPos.set(this.positions!)
    newPos.set(added, this.positions!.length)
    const newZ = [...this.atomicNumbers, ...new Array(count).fill(1)]
    const xyz = serializeXyz(newPos, newZ, this.lattice)
    await this.setStructure(xyz, this.lattice)
    // Select the freshly added H's so the user sees what changed and can
    // undo, tweak, or re-relax with them as the focus.
    const sel = new Set<number>()
    for (let k = 0; k < count; k++) sel.add(oldN + k)
    this.selectedAtoms = sel
    this.selectedBond = null
  }

  async editCommitPositions(positions: Float64Array, keepSelection: ReadonlySet<number>) {
    if (this.isRunning) return
    if (positions.length !== this.atomicNumbers.length * 3) return
    this.pushUndoIfActive()
    const xyz = serializeXyz(positions, this.atomicNumbers, this.lattice)
    const sel = new Set(keepSelection)
    await this.setStructure(xyz, this.lattice)
    this.selectedAtoms = sel
  }

  // ---- undo / redo --------------------------------------------------------
  // Snapshot scope: the editor-visible canonical state. Worker / simulation
  // state isn't tracked — undoing back through a relax isn't supported, since
  // optimization positions don't round-trip through currentXyz.

  /** Run a series of edit operations as a single undo step. Used for
   *  multi-atom batch edits (e.g. replacing every selected atom with a
   *  fragment) so the user gets one undo per gesture instead of N. */
  async editBatch(fn: () => Promise<void>) {
    if (this.undoSuspended) {
      await fn()
      return
    }
    if (this.currentXyz) this.pushUndoState()
    this.undoSuspended = true
    try {
      await fn()
    } finally {
      this.undoSuspended = false
    }
  }

  /** Drop all undo/redo history. Called when a fresh structure is loaded
   *  from outside the editor (file, preset, PubChem). */
  clearEditHistory() {
    this.undoStack = []
    this.redoStack = []
    this.canUndo = false
    this.canRedo = false
  }

  async undo() {
    if (this.isRunning || this.undoStack.length === 0) return
    const current = this.snapshotEditState()
    const prev = this.undoStack.pop()!
    this.redoStack.push(current)
    await this.applyEditSnapshot(prev)
    this.canUndo = this.undoStack.length > 0
    this.canRedo = true
  }

  async redo() {
    if (this.isRunning || this.redoStack.length === 0) return
    const current = this.snapshotEditState()
    const next = this.redoStack.pop()!
    this.undoStack.push(current)
    await this.applyEditSnapshot(next)
    this.canUndo = true
    this.canRedo = this.redoStack.length > 0
  }

  private pushUndoIfActive() {
    if (this.undoSuspended) return
    if (!this.currentXyz) return
    this.pushUndoState()
  }

  private pushUndoState() {
    this.undoStack.push(this.snapshotEditState())
    if (this.undoStack.length > SimulationStore.UNDO_CAP) this.undoStack.shift()
    this.redoStack = []
    this.canUndo = true
    this.canRedo = false
  }

  private snapshotEditState(): EditSnapshot {
    return {
      xyz: this.currentXyz,
      lattice: this.lattice,
      bondOverrides: {
        add: [...this.bondOverrides.add],
        remove: [...this.bondOverrides.remove],
      },
      selectedAtoms: [...this.selectedAtoms],
      selectedBond: this.selectedBond ? [this.selectedBond[0], this.selectedBond[1]] : null,
      selectionGroups: Object.fromEntries(
        Object.entries(this.selectionGroups).map(([k, v]) => [k, [...v]]),
      ),
    }
  }

  private async applyEditSnapshot(s: EditSnapshot) {
    // setStructure() may clear bondOverrides / selection / saved slots when
    // the atom count changes; re-apply them after the round-trip so the
    // restored state is identical to the snapshot. Skip the round-trip if
    // the structure didn't change — only bond toggles fall in that bucket,
    // and re-pushing identical xyz needlessly resets step / energyHistory.
    if (s.xyz !== this.currentXyz) {
      await this.setStructure(s.xyz, s.lattice)
    }
    this.bondOverrides = {
      add: [...s.bondOverrides.add],
      remove: [...s.bondOverrides.remove],
    }
    this.selectedAtoms = new Set(s.selectedAtoms)
    this.selectedBond = s.selectedBond ? [s.selectedBond[0], s.selectedBond[1]] : null
    this.selectionGroups = Object.fromEntries(
      Object.entries(s.selectionGroups).map(([k, v]) => [k, [...v]]),
    )
  }

  /** Loader-facing entry point: replaces the structure and clears history.
   *  Edit ops should keep using setStructure() directly. */
  async loadStructure(xyz: string, lattice: Lattice | null) {
    await this.setStructure(xyz, lattice)
    this.clearEditHistory()
  }

  /** Snapshot the current atom positions/colors/radii for a set of indices
   *  and signal the viewer to play a fading halo at each. Capturing inline
   *  means later position updates can't smear into a still-pending flash. */
  triggerFlash(indices: Iterable<number>): void {
    const idx = [...indices]
    if (idx.length === 0 || !this.positions) {
      this.flashPayload = null
      return
    }
    const n = idx.length
    const positions = new Float32Array(n * 3)
    const colors = new Float32Array(n * 3)
    const radii = new Float32Array(n)
    for (let k = 0; k < n; k++) {
      const i = idx[k]
      if (i < 0 || i * 3 + 2 >= this.positions.length) continue
      positions[k * 3]     = this.positions[i * 3]
      positions[k * 3 + 1] = this.positions[i * 3 + 1]
      positions[k * 3 + 2] = this.positions[i * 3 + 2]
      const elem = getElementByNumber(this.atomicNumbers[i])
      colors[k * 3]     = elem.color[0]
      colors[k * 3 + 1] = elem.color[1]
      colors[k * 3 + 2] = elem.color[2]
      radii[k] = elem.covalentRadius * GHOST_ATOM_SCALE
    }
    this.flashPayload = { positions, colors, radii }
    this.flashTrigger++
  }

  // ------------------------------------------------------------------------

  async syncParameters() {
    // Vib mode is a main-thread concept — the worker doesn't know about it, so
    // we leave the worker's mode field alone and just sync the numeric params.
    const workerMode = this.mode === 'vib' ? undefined : this.mode
    await this.sim.setParameters({
      dt: this.timestep,
      temperature: this.temperature,
      mode: workerMode,
      maxOptSteps: this.maxOptSteps,
      forceThreshold: this.forceThreshold,
      thermostat: this.thermostat,
      useConservativeForces: this.useConservativeForces,
      optimizer: this.optimizer,
    })
  }

  start() {
    if (this.mode === 'vib') return
    this.sim.start(1, this.mode, this.rattleAmount)
  }

  /**
   * Editor-side relaxation. Sets mode to 'optimize', optionally restricts
   * the optimization to the selected atoms (everything else is frozen),
   * optionally overrides the algorithm choice, and starts. Hides the mode-
   * shuffle and rattle-zeroing from EditPanel.
   */
  async runOptimize(opts: { onlySelected?: boolean; algorithm?: Optimizer } = {}) {
    if (this.modelStatus !== 'ready') return
    if (this.atomicNumbers.length === 0 || !this.positions) return
    if (this.isRunning) return

    // Snapshot pre-relax geometry so the user can undo back through the run.
    // The 'stopped' event handler refreshes currentXyz when the relax ends,
    // making the post-relax state a proper snapshot for redo.
    this.pushUndoIfActive()

    if (opts.algorithm) this.optimizer = opts.algorithm
    this.mode = 'optimize'
    await this.syncParameters()

    let frozen: number[] | undefined
    if (opts.onlySelected && this.selectedAtoms.size > 0) {
      const sel = this.selectedAtoms
      const n = this.atomicNumbers.length
      frozen = []
      for (let i = 0; i < n; i++) if (!sel.has(i)) frozen.push(i)
      // Relax-selected: only the selection moved, so flash just those.
      this.optScope = [...sel]
    } else if (this.selectedAtoms.size > 0) {
      // Relax-all but the user had a selection — flash that selection so
      // they see what their focus was without lighting up the whole system.
      this.optScope = [...this.selectedAtoms]
    } else {
      // Relax-all with no selection — skip the flash to avoid a visual storm.
      this.optScope = []
    }
    this.sim.start(1, 'optimize', 0, frozen)
  }

  stop() {
    this.sim.stop()
  }

  stepOnce() {
    this.sim.step()
  }

  rattle() {
    this.sim.rattle(this.rattleAmount)
  }

  // ---------- Vibrational analysis ----------

  async computeVibrations(delta: number = 0.01) {
    if (this.vibComputing) return
    if (!this.positions || this.atomicNumbers.length === 0) {
      this.vibError = 'Load a structure first'
      return
    }
    if (this.isRunning) this.stop()
    this.stopModeAnimation()

    this.vibComputing = true
    this.vibError = ''
    this.vibModes = []
    this.activeMode = null

    const masses = new Float64Array(this.atomicNumbers.length)
    for (let i = 0; i < this.atomicNumbers.length; i++) {
      masses[i] = getMass(this.atomicNumbers[i]) || 12.011
    }

    try {
      if (this.vibOptimizeFirst) {
        this.vibProgress = { done: 0, total: this.maxOptSteps, phase: 'optimize' }
        await this.runOptimizeToConvergence()
      }

      this.vibProgress = {
        done: 0,
        total: 3 * this.atomicNumbers.length * 2,
        phase: 'hessian',
      }
      // `predictAt` doubles the FD work vs total — report in predictions, not DOFs.
      const result = await computeVibrations(
        this.sim,
        this.positions!,
        this.atomicNumbers,
        masses,
        {
          delta,
          projectTrRot: this.vibProjectTrRot,
          isPeriodic: this.isPeriodic,
        },
        (p) => {
          // The modes pipeline counts DOFs (each DOF is 2 predictions). Scale
          // for a smoother progress bar.
          this.vibProgress = {
            ...p,
            done: p.done * 2,
            total: p.total * 2,
          }
        },
      )
      this.vibModes = result.modes
      this.vibEquilibrium = result.equilibriumPositions
      this.vibNProjected = result.nProjected
    } catch (err: any) {
      this.vibError = err?.message ?? String(err)
    } finally {
      this.vibComputing = false
      this.vibProgress = null
    }
  }

  // Kick off a FIRE optimization in the worker and resolve when it converges
  // (or hits the max-step cap). The store's normal event handler keeps
  // this.positions in sync as optStep events stream in.
  private runOptimizeToConvergence(): Promise<void> {
    return new Promise((resolve, reject) => {
      let settled = false
      const unsub = this.sim.on((ev) => {
        if (ev.kind === 'optStep') {
          this.vibOptStep = ev.step.step
          this.vibOptMaxForce = ev.step.maxForce
          if (this.vibProgress) {
            this.vibProgress = {
              ...this.vibProgress,
              done: Math.min(ev.step.step, this.vibProgress.total),
            }
          }
          if (ev.step.converged && !settled) {
            settled = true
            unsub()
            resolve()
          }
        } else if (ev.kind === 'stopped' && !settled) {
          // Worker stopped — FIRE hit its max step count without converging.
          settled = true
          unsub()
          resolve()
        } else if (ev.kind === 'error' && !settled) {
          settled = true
          unsub()
          reject(new Error(ev.message))
        }
      })
      this.sim
        .setParameters({ mode: 'optimize', maxOptSteps: this.maxOptSteps, forceThreshold: this.forceThreshold })
        .then(() => {
          this.sim.start(1, 'optimize', 0)
        })
        .catch((err) => {
          if (!settled) {
            settled = true
            unsub()
            reject(err)
          }
        })
    })
  }

  playMode(index: number) {
    if (!this.vibEquilibrium) return
    const mode = this.vibModes[index]
    if (!mode) return
    this.activeMode = index
    this.vibPlaying = true
    this.animationStart = performance.now()
    this.animateStep()
  }

  private animateStep = () => {
    if (!this.vibPlaying || this.activeMode === null || !this.vibEquilibrium) return
    const mode = this.vibModes[this.activeMode]
    if (!mode) return
    const t = (performance.now() - this.animationStart) / this.vibPeriodMs
    const scale = this.vibAmplitude * Math.sin(2 * Math.PI * t)

    const eq = this.vibEquilibrium
    const d = mode.displacement
    const next = new Float64Array(eq.length)
    for (let i = 0; i < eq.length; i++) next[i] = eq[i] + scale * d[i]
    this.positions = next

    this.animationFrameId = requestAnimationFrame(this.animateStep)
  }

  stopModeAnimation() {
    this.vibPlaying = false
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId)
      this.animationFrameId = null
    }
    if (this.vibEquilibrium) {
      // Snap back to the equilibrium geometry so the user doesn't see a
      // half-way displaced molecule once they stop the animation.
      this.positions = new Float64Array(this.vibEquilibrium)
    }
  }

  clearVibrations() {
    this.stopModeAnimation()
    this.vibModes = []
    this.vibEquilibrium = null
    this.activeMode = null
    this.vibError = ''
  }

  dispose() {
    this.stopModeAnimation()
    this.sim.dispose()
  }
}

function defaultBackend(): Backend {
  if (typeof navigator === 'undefined') return 'auto'
  // No WebGPU API exposed → CPU. Covers Firefox (no WebGPU by default) and
  // older Safari. `navigator.gpu` is only present on the main thread in
  // Safari, but defaultBackend runs there so this check is meaningful.
  if (!('gpu' in navigator)) return 'cpu'
  const ua = navigator.userAgent
  // Firefox exposes navigator.gpu (behind or partially enabled in recent
  // versions) but returns a stub/software adapter that's slower than CPU
  // and still triggers the full ASYNCIFY suspend/rewind path. Stay on CPU.
  if (/Firefox/i.test(ua)) return 'cpu'
  // Safari's WebGPU in Web Workers is unreliable / absent, and our WASM
  // module runs inside a Worker. Default Safari to CPU; the user can still
  // flip the toggle manually if a future Safari fixes this.
  const isSafari = /Safari/i.test(ua) && !/Chrome|Chromium|Android/i.test(ua)
  if (isSafari) return 'cpu'
  return 'auto'
}
