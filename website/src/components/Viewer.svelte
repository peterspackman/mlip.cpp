<script lang="ts">
  import { getContext, onMount, onDestroy } from 'svelte'
  import { ViewerStage } from '../lib/molview/Stage'
  import { StructureObject } from '../lib/molview/StructureObject'
  import { SelectionManager } from '../lib/molview/Selection'
  import { createUnitCellGroup, updateUnitCellGroupGeometry } from '../lib/molview/UnitCell'
  import { buildStructureData } from '../lib/molview/adapter'
  import { wrapPositions } from '../lib/chem/cell'
  import { generateSupercell } from '../lib/chem/supercell'
  import { detectBonds } from '../lib/chem/bonds'
  import { TransformController } from '../lib/editor/transform.svelte'
  import { EditorMachine } from '../lib/editor/editorMachine.svelte'
  import { openBondDirection } from '../lib/editor/hydrogens'
  import { GhostFlash } from '../lib/editor/ghostFlash'
  import EditOverlay from './EditOverlay.svelte'
  import EditorHelp from './EditorHelp.svelte'
  import type { SimulationStore } from '../lib/stores/simulation.svelte'
  import { getElementByNumber, symbolToAtomicNumber } from '../lib/molview/data/elements'
  import * as THREE from 'three'
  import { LineSegments2 } from 'three/examples/jsm/lines/LineSegments2.js'
  import { LineSegmentsGeometry } from 'three/examples/jsm/lines/LineSegmentsGeometry.js'
  import { LineMaterial } from 'three/examples/jsm/lines/LineMaterial.js'

  const store = getContext<SimulationStore>('store')

  let container: HTMLDivElement
  let canvasEl: HTMLCanvasElement | null = null
  let stage: ViewerStage | null = null
  let structureObj: StructureObject | null = null
  let unitCellGroup: THREE.Group | null = null
  let axisLine: LineSegments2 | null = null
  let pivotMarker: THREE.Mesh | null = null
  let ghostFlash: GhostFlash | null = null
  const picker = new SelectionManager()

  const AXIS_COLOR = {
    x: 0xff5252,
    y: 0x4caf50,
    z: 0x448aff,
    bond: 0xff9900,
  }
  const transform = new TransformController(
    store,
    () => stage?.camera ?? null,
    () => canvasEl,
  )

  // Drag detection: a pointerdown that moves > DRAG_THRESHOLD before pointerup
  // is a camera gesture, not a pick. Otherwise rotating with the mouse would
  // clear the selection on every release.
  const DRAG_THRESHOLD = 4 // px
  let downX = 0
  let downY = 0
  let dragged = false
  /** True when the latest pointerdown was consumed by a transform commit/cancel.
   *  The matching pointerup must NOT then fall through to pick-on-release (which
   *  would clear the selection if released over empty space). */
  let pointerOwnedByTransform = false

  // Last cursor position (canvas-relative px). Used to seed a transform when
  // it's started by keyboard (G/R) — we don't get a mouse event for that.
  let lastPointer = { x: 0, y: 0 }

  // Box-select tool: pressing 'S' arms a one-shot rectangular marquee. The
  // user click-drags on the canvas to draw a rectangle; on release, atoms
  // whose camera-projected centers fall inside become the selection (modifier
  // keys at start time decide replace / add / toggle). Esc cancels.
  // `boxArmed` is derived from the machine so 'box' is a first-class mode.
  let boxStart: { x: number; y: number } | null = $state(null)
  let boxEnd: { x: number; y: number } | null = $state(null)
  let boxModifier: 'replace' | 'add' | 'toggle' = 'replace'

  let boxRect = $derived.by(() => {
    if (!boxStart || !boxEnd) return null
    const x = Math.min(boxStart.x, boxEnd.x)
    const y = Math.min(boxStart.y, boxEnd.y)
    const w = Math.abs(boxEnd.x - boxStart.x)
    const h = Math.abs(boxEnd.y - boxStart.y)
    return { x, y, w, h }
  })

  function clearBoxScratch(): void {
    boxStart = null
    boxEnd = null
  }

  // INSERT-mode scratch. insertDragging mutes the camera while a bond is being
  // dragged off an existing atom; insertStartAtom is the atom under pointerdown
  // (-1 = empty); insertPointerActive tracks an in-progress insert gesture.
  let insertDragging = $state(false)
  let insertStartAtom = -1
  let insertPointerActive = false

  /** Wipe every per-gesture scratch variable back to its idle value. Called on
   *  every machine transition (via onAnyTransition), so no gesture state can
   *  survive into a different mode. Does NOT touch pointerOwnedByTransform —
   *  that is deliberately set during a transition (commit-on-pointerdown) and
   *  owned by the pointerdown/up pair. */
  function resetGestureScratch(): void {
    insertDragging = false
    insertStartAtom = -1
    insertPointerActive = false
    boxStart = null
    boxEnd = null
    dragged = false
  }

  // The editor mode state machine. It is the SINGLE source of truth for the
  // top-level interaction state; every key/pointer handler routes through it
  // and reads `machine.mode` (never transform.active / boxArmed directly for
  // decisions). Side effects (begin a transform, arm the marquee, run a
  // command) are wired here; onAnyTransition guarantees a clean slate.
  const machine = new EditorMachine({
    onEnterTransform: (kind) => transform.begin(kind, lastPointer.x, lastPointer.y),
    onSwitchTransform: (kind) => transform.switchMode(kind),
    onCommitTransform: () => transform.commit(),
    onCancelTransform: () => transform.cancel(),
    onCommitBox: () => commitBoxSelection(),
    onRunCommand: (buf) => runCommand(buf),
    onAnyTransition: () => resetGestureScratch(),
  })
  let boxArmed = $derived(machine.mode === 'box')

  function commitBoxSelection(): void {
    if (!stage || !canvasEl || !boxStart || !boxEnd) { clearBoxScratch(); return }
    const display = prepareDisplay()
    if (!display) { clearBoxScratch(); return }
    const minX = Math.min(boxStart.x, boxEnd.x)
    const maxX = Math.max(boxStart.x, boxEnd.x)
    const minY = Math.min(boxStart.y, boxEnd.y)
    const maxY = Math.max(boxStart.y, boxEnd.y)
    // Click without drag → ignore (user just tapped, treat as cancel).
    if (maxX - minX < 2 && maxY - minY < 2) { clearBoxScratch(); return }

    const cam = stage.camera
    const w = canvasEl.clientWidth
    const h = canvasEl.clientHeight
    const v = new THREE.Vector3()
    const hits = new Set<number>()
    const n = display.atomicNumbers.length
    for (let i = 0; i < n; i++) {
      v.set(
        display.positions[i * 3],
        display.positions[i * 3 + 1],
        display.positions[i * 3 + 2],
      ).project(cam)
      // Behind the camera in perspective projection — skip.
      if (v.z < -1 || v.z > 1) continue
      const px = (v.x + 1) * 0.5 * w
      const py = (1 - v.y) * 0.5 * h
      if (px >= minX && px <= maxX && py >= minY && py <= maxY) {
        hits.add(canonicalize(i))
      }
    }

    let next: Set<number>
    if (boxModifier === 'add') {
      next = new Set(store.selectedAtoms)
      for (const i of hits) next.add(i)
    } else if (boxModifier === 'toggle') {
      next = new Set(store.selectedAtoms)
      for (const i of hits) {
        if (next.has(i)) next.delete(i)
        else next.add(i)
      }
    } else {
      next = hits
    }
    store.selectedAtoms = next
    store.selectedBond = null
    clearBoxScratch()
  }

  // --- INSERT-mode building ------------------------------------------------

  /** Canonical index of the atom under the cursor, or -1 for empty space. */
  function pickAtomAt(ev: MouseEvent): number {
    if (!stage || !structureObj || !canvasEl) return -1
    const hit = picker.pick(ev, stage.camera, canvasEl, structureObj)
    return hit && hit.type === 'atom' ? canonicalize(hit.atomIndex) : -1
  }

  function atomVec(i: number): THREE.Vector3 {
    const p = store.positions!
    return new THREE.Vector3(p[i * 3], p[i * 3 + 1], p[i * 3 + 2])
  }

  /** Centroid of the current structure, used as the depth reference for
   *  placing free atoms so they land near the molecule and stay visible. */
  function structureCentroid(): THREE.Vector3 {
    const p = store.positions
    const c = new THREE.Vector3()
    if (!p || p.length < 3) return c
    const n = p.length / 3
    for (let i = 0; i < n; i++) { c.x += p[i * 3]; c.y += p[i * 3 + 1]; c.z += p[i * 3 + 2] }
    return c.divideScalar(n)
  }

  /** Project canvas-relative (px) coords onto the screen-parallel plane through
   *  `ref` — the Avogadro trick that turns a 2D point into a 3D one at a
   *  sensible depth. */
  function canvasToWorldPlane(cx: number, cy: number, ref: THREE.Vector3): THREE.Vector3 | null {
    if (!stage || !canvasEl) return null
    const rect = canvasEl.getBoundingClientRect()
    const ndc = new THREE.Vector2(
      (cx / rect.width) * 2 - 1,
      -(cy / rect.height) * 2 + 1,
    )
    const ray = new THREE.Raycaster()
    ray.setFromCamera(ndc, stage.camera)
    const normal = new THREE.Vector3()
    stage.camera.getWorldDirection(normal)
    const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(normal, ref)
    const hit = new THREE.Vector3()
    return ray.ray.intersectPlane(plane, hit) ? hit : null
  }

  /** As above, from a mouse/pointer event's client coordinates. */
  function screenToWorldPlane(clientX: number, clientY: number, ref: THREE.Vector3): THREE.Vector3 | null {
    if (!canvasEl) return null
    const rect = canvasEl.getBoundingClientRect()
    return canvasToWorldPlane(clientX - rect.left, clientY - rect.top, ref)
  }

  /** Paste the clipboard so its centroid lands at the cursor (⌘V). Falls back
   *  to the store's default anchor if the ray can't hit the plane. */
  function pasteAtCursor(): void {
    const p = canvasToWorldPlane(lastPointer.x, lastPointer.y, structureCentroid())
    store.editPaste(p ? { anchor: [p.x, p.y, p.z] } : {})
  }

  /** Where a new atom bonded to `srcIdx` should sit: aim at `target` but clamp
   *  the bond length to the sum of covalent radii; if the drag was tiny, snap
   *  to a valence-open direction instead. */
  function bondedPlacement(srcIdx: number, target: THREE.Vector3): THREE.Vector3 {
    const src = atomVec(srcIdx)
    const bondLen = getElementByNumber(store.atomicNumbers[srcIdx]).covalentRadius
      + getElementByNumber(store.activeElement).covalentRadius
    let dir = target.clone().sub(src)
    if (dir.lengthSq() < 1e-6) {
      const d = openBondDirection(store.positions!, store.atomicNumbers, srcIdx)
      dir = new THREE.Vector3(d[0], d[1], d[2])
    } else {
      dir.normalize()
    }
    return src.add(dir.multiplyScalar(bondLen))
  }

  /** Resolve an INSERT-mode pointer release into an edit. */
  function handleInsertRelease(startAtom: number, wasDrag: boolean, ev: PointerEvent): void {
    if (!wasDrag) {
      if (startAtom >= 0) {
        // Click an atom → recolour it to the active element.
        store.editSetElement(new Set([startAtom]), store.activeElement)
      } else {
        // Click empty → drop a new atom at the cursor.
        const p = screenToWorldPlane(ev.clientX, ev.clientY, structureCentroid())
        if (p) store.editAddAtom(store.activeElement, [p.x, p.y, p.z])
      }
      return
    }
    if (startAtom < 0) return // empty-space drag = camera orbit; nothing to build
    const endAtom = pickAtomAt(ev)
    if (endAtom >= 0 && endAtom !== startAtom) {
      store.toggleBond(startAtom, endAtom) // drag atom → atom: (un)bond
      return
    }
    if (endAtom < 0) {
      // Drag atom → empty: new bonded atom at the release point.
      const raw = screenToWorldPlane(ev.clientX, ev.clientY, atomVec(startAtom))
      if (!raw) return
      const placed = bondedPlacement(startAtom, raw)
      store.editAddAtom(store.activeElement, [placed.x, placed.y, placed.z])
    }
  }

  /** Vim-style command line. Small, extensible verb set. */
  function runCommand(raw: string): void {
    const s = raw.trim()
    if (!s) return
    const parts = s.split(/\s+/)
    const cmd = parts[0].toLowerCase()
    const arg = parts.slice(1).join(' ')
    switch (cmd) {
      case 'e':
      case 'element': {
        const z = symbolToAtomicNumber(arg)
        if (z > 0) store.activeElement = z
        break
      }
      case 'add': {
        const z = arg ? symbolToAtomicNumber(arg) : store.activeElement
        if (z > 0) {
          const c = structureCentroid()
          store.editAddAtom(z, [c.x, c.y, c.z + 1.5])
        }
        break
      }
      case 'fill':
        if (/^h/i.test(arg)) {
          store.editFillHydrogens(store.selectedAtoms.size > 0 ? store.selectedAtoms : null)
        }
        break
      case 'del':
      case 'delete':
        if (store.selectedAtoms.size > 0) store.editDeleteAtoms(store.selectedAtoms)
        break
      case 'relax':
      case 'opt':
        if (store.modelStatus === 'ready' && store.atomicNumbers.length > 0) {
          store.runOptimize({ onlySelected: store.selectedAtoms.size > 0 })
        }
        break
    }
  }

  function onCommandKey(ev: KeyboardEvent): void {
    ev.stopPropagation()
    if (ev.key === 'Enter') { ev.preventDefault(); machine.commit() }
    else if (ev.key === 'Escape') { ev.preventDefault(); machine.escape() }
  }

  function focusOnMount(node: HTMLInputElement) { node.focus() }

  // Visible tool toggle (discoverability — you shouldn't need to know `i`).
  function selectMode() {
    if (machine.mode !== 'normal') machine.escape()
  }
  function drawMode() {
    if (machine.mode === 'insert') return
    if (machine.mode !== 'normal') machine.escape()
    machine.enterInsert()
  }

  // Inputs that force a full rebuild (atom count/types, lattice, wrap, supercell, view style).
  // Position-only changes hit the cheap in-place update path.
  let lastStructureKey = ''
  /** Hash of the bond list the renderer is currently drawing. The cheap path
   *  re-detects bonds each tick; a mismatch triggers a full rebuild so newly
   *  formed / broken bonds appear (or disappear) instead of being stuck on
   *  the cached list from build time. */
  let lastBondsHash = ''
  /** Atom-count + lattice signature of the last `fitToContent`. Rebuilds
   *  caused by view-only changes (style, supercell, bond overrides, dynamic
   *  bond re-detection mid-transform) keep the same signature and don't
   *  refit — otherwise a bond breaking during a rotate would snap the
   *  camera to the entire scene, including the long axis-constraint line. */
  let lastFitKey = ''

  function bondsHash(positions: ArrayLike<number>, atomicNumbers: ArrayLike<number>): string {
    const bonds = detectBonds(positions, atomicNumbers)
    bonds.sort((p, q) => p[0] - q[0] || p[1] - q[1])
    return bonds.map(([a, b]) => `${a}-${b}`).join(',')
  }

  function structureKey(): string {
    return [
      store.atomicNumbers.join(','),
      JSON.stringify(store.lattice),
      store.wrapPositions,
      store.supercell.join('x'),
      store.viewStyle,
      store.dynamicBonds,
      store.bondOverrides.add.join(','),
      store.bondOverrides.remove.join(','),
    ].join('|')
  }

  /** Apply user-toggled bond overrides to a freshly-built StructureData.
   *  Only touches bonds within the canonical cell (atoms < n_canonical) —
   *  image cells in a supercell render whatever the detector produced. */
  function applyBondOverrides(bonds: { atomA: number; atomB: number; order: number }[], n_canonical: number) {
    const adds = new Set(store.bondOverrides.add)
    const removes = new Set(store.bondOverrides.remove)
    if (adds.size === 0 && removes.size === 0) return bonds
    const present = new Set<string>()
    const out: typeof bonds = []
    for (const b of bonds) {
      const inCanonical = b.atomA < n_canonical && b.atomB < n_canonical
      const key = b.atomA < b.atomB ? `${b.atomA}-${b.atomB}` : `${b.atomB}-${b.atomA}`
      if (inCanonical) {
        present.add(key)
        if (removes.has(key)) continue
      }
      out.push(b)
    }
    for (const key of adds) {
      if (present.has(key)) continue
      const [aS, bS] = key.split('-')
      const a = parseInt(aS, 10), b = parseInt(bS, 10)
      if (a < n_canonical && b < n_canonical) {
        out.push({ atomA: a, atomB: b, order: 1 })
      }
    }
    return out
  }

  function prepareDisplay(): { positions: ArrayLike<number>; atomicNumbers: number[] } | null {
    if (!store.positions || store.atomicNumbers.length === 0) return null
    let pos: ArrayLike<number> = store.positions
    if (store.wrapPositions && store.lattice) pos = wrapPositions(pos, store.lattice)
    let atoms = store.atomicNumbers
    const sc = store.supercell
    if (store.lattice && (sc[0] !== 1 || sc[1] !== 1 || sc[2] !== 1)) {
      const sup = generateSupercell(pos, atoms, store.lattice, sc)
      pos = sup.positions
      atoms = sup.atomicNumbers
    }
    return { positions: pos, atomicNumbers: atoms }
  }

  function rebuildStructure() {
    if (!stage) return
    const display = prepareDisplay()
    if (!display) return

    const data = buildStructureData(display.positions, display.atomicNumbers, store.lattice)
    data.bonds = applyBondOverrides(data.bonds, store.atomicNumbers.length)

    if (structureObj) {
      stage.remove(structureObj.group)
      stage.remove(structureObj.selectionProxyGroup)
      structureObj.dispose()
    }
    structureObj = new StructureObject(data, store.viewStyle)
    stage.add(structureObj.group)
    stage.add(structureObj.selectionProxyGroup)
    lastBondsHash = bondsHash(display.positions, display.atomicNumbers)

    // Selection state is owned by the store and cleared there only when the
    // atom count actually changes. A rebuild here may be triggered by a view
    // toggle (style, supercell, bond override, …) that keeps indices valid;
    // we just keep whatever was selected and let the new representation pick
    // it up via syncOutline below.
    syncOutline()

    if (store.lattice && data.unitCell) {
      if (unitCellGroup) {
        updateUnitCellGroupGeometry(unitCellGroup, data.unitCell)
      } else {
        unitCellGroup = createUnitCellGroup(data.unitCell)
        stage.add(unitCellGroup)
      }
    } else if (unitCellGroup) {
      stage.remove(unitCellGroup)
      unitCellGroup = null
    }

    // Refit only when the structure actually changed (atom count or lattice).
    // Skip while a transform / run is in flight — those paths can rebuild for
    // bond-topology reasons mid-motion, and the axis-constraint line in the
    // scene would blow up the bounding box.
    const fitKey = `${store.atomicNumbers.length}|${JSON.stringify(store.lattice)}`
    // Don't refit while drawing — each placed atom changes the count and would
    // otherwise jump the camera out from under the user mid-build.
    if (fitKey !== lastFitKey && machine.mode !== 'transform' && !store.isRunning && machine.mode !== 'insert') {
      stage.fitToContent()
      lastFitKey = fitKey
    }
  }

  function syncOutline() {
    if (!stage || !structureObj) return
    structureObj.applyFullSelection(store.selectedAtoms)
    const proxies = structureObj.selectionProxyGroup.children
    stage.setOutlinedObjects(proxies.length > 0 ? [...proxies] : [])
  }

  function canvasCoords(ev: { clientX: number; clientY: number }): { x: number; y: number } {
    if (!canvasEl) return { x: 0, y: 0 }
    const r = canvasEl.getBoundingClientRect()
    return { x: ev.clientX - r.left, y: ev.clientY - r.top }
  }

  /** Capture-phase pointerdown (registered on window). Only claims INSERT-mode
   *  presses that land on an atom — those become build-drags, so we stop the
   *  event before TrackballControls (bound to the canvas) can start orbiting.
   *  Empty-space presses fall through untouched so the camera still orbits. */
  function onPointerDownCapture(ev: PointerEvent) {
    if (store.appMode !== 'edit' || machine.mode !== 'insert') return
    if (ev.button !== 0 || ev.target !== canvasEl) return
    const atom = pickAtomAt(ev)
    if (atom < 0) return // empty space → let the normal flow orbit
    ev.stopImmediatePropagation()
    ev.preventDefault()
    // Capture the pointer so pointerup fires even if released off-canvas —
    // otherwise a stray release would strand insertDragging (camera stuck muted).
    try { canvasEl?.setPointerCapture(ev.pointerId) } catch { /* invalid id */ }
    if (stage) stage.controls.enabled = false
    insertStartAtom = atom
    insertPointerActive = true
    insertDragging = true
    downX = ev.clientX
    downY = ev.clientY
    dragged = false
  }

  function onPointerDown(ev: PointerEvent) {
    const c = canvasCoords(ev)
    lastPointer = c

    // Move keyboard focus off any text input that's been left focused by a
    // prior interaction (PubChem search, model loader, XYZ editor, …). Without
    // this, editor shortcuts like G/R or shift+1 type into the input instead
    // of reaching the global handler.
    const active = document.activeElement
    if (active instanceof HTMLElement && active !== canvasEl) {
      const tag = active.tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || active.isContentEditable) {
        active.blur()
      }
    }

    // While a transform is active, pointer events drive commit/cancel; don't
    // engage the pick-on-release path.
    if (machine.mode === 'transform') {
      ev.preventDefault()
      pointerOwnedByTransform = true
      if (ev.button === 0) machine.commit()
      else if (ev.button === 2) machine.escape()
      return
    }

    // INSERT mode: left-press starts a build gesture. A press on an atom is a
    // build-drag (bond) so we own the pointer + mute the camera; a press on
    // empty space lets the camera orbit and only *places* on a no-move release.
    if (machine.mode === 'insert') {
      if (ev.button !== 0) return // right-click delete is handled in onContextMenu
      insertStartAtom = pickAtomAt(ev)
      insertPointerActive = true
      downX = ev.clientX
      downY = ev.clientY
      dragged = false
      if (insertStartAtom >= 0) {
        insertDragging = true
        // Disable orbit synchronously — the reactive mute effect runs a tick
        // later, too late to stop TrackballControls grabbing this drag.
        if (stage) stage.controls.enabled = false
        ev.preventDefault()
      }
      return
    }

    // Box-select mode: left-press starts the rectangle.
    if (boxArmed && ev.button === 0) {
      ev.preventDefault()
      pointerOwnedByTransform = true
      boxStart = c
      boxEnd = c
      boxModifier = ev.shiftKey ? 'add' : (ev.metaKey || ev.ctrlKey ? 'toggle' : 'replace')
      return
    }
    if (boxArmed && ev.button === 2) {
      ev.preventDefault()
      machine.escape()
      return
    }

    pointerOwnedByTransform = false
    if (ev.button !== 0) return
    downX = ev.clientX
    downY = ev.clientY
    dragged = false
  }

  function onPointerMove(ev: PointerEvent) {
    const c = canvasCoords(ev)
    lastPointer = c
    if (machine.mode === 'transform') {
      transform.update(c.x, c.y)
      return
    }
    if (boxArmed && boxStart) {
      boxEnd = c
      return
    }
    if (dragged) return
    if (Math.abs(ev.clientX - downX) > DRAG_THRESHOLD ||
        Math.abs(ev.clientY - downY) > DRAG_THRESHOLD) {
      dragged = true
    }
  }

  function canonicalize(displayIdx: number): number {
    const n = store.atomicNumbers.length
    if (n === 0) return displayIdx
    return displayIdx % n
  }

  function onPointerUp(ev: PointerEvent) {
    // INSERT mode owns its own release path (place / set-element / bond).
    if (machine.mode === 'insert') {
      if (!insertPointerActive || ev.button !== 0) return
      insertPointerActive = false
      const startAtom = insertStartAtom
      const wasDrag = dragged
      insertStartAtom = -1
      insertDragging = false
      // Re-enable orbit now the build gesture is over (mirrors the sync disable
      // on pointerdown; the mute effect will keep it consistent afterward).
      if (stage) stage.controls.enabled = true
      handleInsertRelease(startAtom, wasDrag, ev)
      return
    }
    if (boxArmed && boxStart && ev.button === 0) {
      ev.preventDefault()
      pointerOwnedByTransform = false
      machine.commit()
      return
    }
    if (pointerOwnedByTransform) {
      // Commit/cancel ran on the matching pointerdown; swallow the release.
      pointerOwnedByTransform = false
      return
    }
    if (machine.mode === 'transform') return  // a fresh transform was begun via keyboard between down and up
    if (ev.button !== 0) return
    if (dragged) return
    if (!stage || !structureObj || !canvasEl) return

    const hit = picker.pick(ev as unknown as MouseEvent, stage.camera, canvasEl, structureObj)
    const additive = ev.shiftKey
    const toggle = ev.metaKey || ev.ctrlKey

    if (!hit) {
      if (!additive && !toggle) {
        if (store.selectedAtoms.size > 0) store.selectedAtoms = new Set()
        if (store.selectedBond) store.selectedBond = null
      }
      return
    }

    if (hit.type === 'atom') {
      const idx = canonicalize(hit.atomIndex)
      const next = additive || toggle ? new Set(store.selectedAtoms) : new Set<number>()
      if (toggle && next.has(idx)) next.delete(idx)
      else next.add(idx)
      store.selectedAtoms = next
      store.selectedBond = null
    } else if (hit.type === 'bond') {
      // cebuns-style: clicking a bond selects (or toggles) both endpoint atoms
      // AND records the bond so subsequent G/R + B can use its direction.
      const a = canonicalize(hit.atomA)
      const b = canonicalize(hit.atomB)
      const next = additive || toggle ? new Set(store.selectedAtoms) : new Set<number>()
      if (toggle && next.has(a) && next.has(b)) {
        next.delete(a); next.delete(b)
      } else {
        next.add(a); next.add(b)
      }
      store.selectedAtoms = next
      store.selectedBond = [a, b]
    }
  }

  function onContextMenu(ev: MouseEvent) {
    if (machine.mode === 'transform') {
      ev.preventDefault()
      machine.escape()
      return
    }
    // INSERT mode: right-click deletes the atom under the cursor.
    if (machine.mode === 'insert') {
      ev.preventDefault()
      const a = pickAtomAt(ev)
      if (a >= 0) store.editDeleteAtoms(new Set([a]))
      return
    }
  }

  function isTypingTarget(t: EventTarget | null): boolean {
    if (!(t instanceof HTMLElement)) return false
    const tag = t.tagName
    return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || t.isContentEditable
  }

  function onKey(ev: KeyboardEvent) {
    if (store.appMode !== 'edit') return
    if (isTypingTarget(ev.target)) return

    // While the help overlay is open it captures keys: ? / h / Esc close it,
    // everything else is swallowed so nothing acts behind the modal.
    if (store.helpOpen) {
      if (ev.key === '?' || ev.key === 'h' || ev.key === 'Escape') {
        ev.preventDefault()
        store.helpOpen = false
      }
      return
    }

    // ⌘/Ctrl combos (undo, redo, copy, paste, duplicate). Inert during a
    // modal transform — the user has Esc / Enter for that.
    if ((ev.metaKey || ev.ctrlKey) && machine.mode !== 'transform') {
      const k = ev.key.toLowerCase()
      if (k === 'z') {
        ev.preventDefault()
        if (ev.shiftKey) store.redo()
        else store.undo()
        return
      }
      if (k === 'y' && !ev.shiftKey) {
        ev.preventDefault()
        store.redo()
        return
      }
      if (k === 'c' && !ev.shiftKey) {
        if (store.selectedAtoms.size > 0) {
          ev.preventDefault()
          store.editCopy()
        }
        return
      }
      if (k === 'v' && !ev.shiftKey) {
        if (store.clipboard) {
          ev.preventDefault()
          pasteAtCursor()
        }
        return
      }
      if (k === 'd' && !ev.shiftKey) {
        if (store.selectedAtoms.size > 0) {
          ev.preventDefault()
          store.editDuplicate()
        }
        return
      }
    }

    // Modal keys take over while a transform is in flight.
    if (machine.mode === 'transform') {
      if (ev.key === 'Escape') { ev.preventDefault(); machine.escape(); return }
      if (ev.key === 'Enter')  { ev.preventDefault(); machine.commit(); return }
      const k = ev.key.toLowerCase()
      // G / R inside an active transform = switch mode (preserves the in-progress effect).
      if (k === 'g') { ev.preventDefault(); machine.switchTransform('grab'); return }
      if (k === 'r') { ev.preventDefault(); machine.switchTransform('rotate'); return }
      if (k === 'x' || k === 'y' || k === 'z') {
        ev.preventDefault(); transform.setAxis(k); return
      }
      if (k === 'b') {
        ev.preventDefault(); transform.setAxis('bond'); return
      }
      if (ev.key === 'Tab') {
        ev.preventDefault()
        transform.cycleBond(ev.shiftKey ? -1 : 1)
        return
      }
      if (ev.key === 'Backspace') { ev.preventDefault(); transform.type('Backspace'); return }
      if (/^[\d.\-]$/.test(ev.key)) { ev.preventDefault(); transform.type(ev.key); return }
      return
    }

    // Help overlay — reachable from Select and Draw.
    if ((ev.key === '?' || ev.key === 'h') && (machine.mode === 'normal' || machine.mode === 'insert')) {
      ev.preventDefault()
      store.helpOpen = true
      return
    }

    // --- editor mode switches (state machine) ----------------------------
    // From NORMAL, `i` enters INSERT and `:` opens the COMMAND line. In any
    // other modal state (insert / command / box) Esc returns to NORMAL and the
    // NORMAL keymap below is suppressed — INSERT is mouse-driven and COMMAND
    // typing is routed to its own input.
    if (machine.mode !== 'normal') {
      if (ev.key === 'Escape') { ev.preventDefault(); machine.escape(); return }
      return
    }
    if (ev.key === 'i') { ev.preventDefault(); machine.enterInsert(); return }
    if (ev.key === ':') { ev.preventDefault(); machine.enterCommand(); return }

    // Saved selection slots: digit 1-9 recalls, shift+digit saves. Use ev.code
    // so it works regardless of keyboard layout (shift+1 → '!' on US layouts).
    const digitMatch = ev.code.match(/^Digit([1-9])$/)
    if (digitMatch) {
      const slot = parseInt(digitMatch[1], 10)
      if (ev.shiftKey && !ev.ctrlKey && !ev.metaKey && !ev.altKey) {
        ev.preventDefault()
        // Silent no-op when nothing is selected — saving an empty slot looks
        // identical to "didn't save" in the UI and just confuses things.
        if (store.selectedAtoms.size === 0) return
        store.selectionGroups = {
          ...store.selectionGroups,
          [slot]: [...store.selectedAtoms],
        }
        return
      }
      if (!ev.shiftKey && !ev.ctrlKey && !ev.metaKey && !ev.altKey) {
        const saved = store.selectionGroups[slot]
        if (saved && saved.length > 0) {
          ev.preventDefault()
          store.selectedAtoms = new Set(saved)
          store.selectedBond = null
        }
        return
      }
    }

    // Esc while idle stops a running run, so the user doesn't have to chase
    // the Stop button in the panel.
    if (ev.key === 'Escape' && store.isRunning) {
      ev.preventDefault()
      store.stop()
      return
    }

    // Spacebar: toggle a relax. Selection-scoped if anything is selected,
    // otherwise relax all. Stop if a run is already going.
    if (ev.key === ' ' || ev.code === 'Space') {
      ev.preventDefault()
      if (store.isRunning) {
        store.stop()
      } else if (store.modelStatus === 'ready' && store.atomicNumbers.length > 0) {
        store.runOptimize({ onlySelected: store.selectedAtoms.size > 0 })
      }
      return
    }

    // Idle keys.
    if (ev.key === 'g' || ev.key === 'G') {
      ev.preventDefault()
      machine.beginTransform('grab')
    } else if (ev.key === 'r' || ev.key === 'R') {
      ev.preventDefault()
      machine.beginTransform('rotate')
    } else if (ev.key === 'f' || ev.key === 'F') {
      // Bond / unbond between the two selected atoms (or the selected bond).
      let pair: [number, number] | null = null
      if (store.selectedBond) pair = store.selectedBond
      else if (store.selectedAtoms.size === 2) {
        const arr = [...store.selectedAtoms]
        pair = [arr[0], arr[1]]
      }
      if (pair) {
        ev.preventDefault()
        store.toggleBond(pair[0], pair[1])
      }
    } else if (ev.key === 'a' || ev.key === 'A') {
      // Quick "select all".
      if (store.atomicNumbers.length > 0) {
        ev.preventDefault()
        store.selectedAtoms = new Set(store.atomicNumbers.map((_, i) => i))
        store.selectedBond = null
      }
    } else if (ev.key === 's' || ev.key === 'S') {
      // Arm box-select. Drag on the canvas to draw a marquee. Modifier keys
      // at pointerdown decide replace / add / toggle.
      if (store.atomicNumbers.length > 0) {
        ev.preventDefault()
        machine.armBox()
      }
    } else if (ev.key === 'u') {
      // Vim-style undo / redo (alongside ⌘Z / ⌘⇧Z).
      ev.preventDefault()
      store.undo()
    } else if (ev.key === 'U') {
      ev.preventDefault()
      store.redo()
    } else if (ev.key === 'y') {
      // Vim yank — copy the selection.
      if (store.selectedAtoms.size > 0) {
        ev.preventDefault()
        store.editCopy()
      }
    } else if (ev.key === 'p') {
      // Vim paste — drop the clipboard at the cursor.
      if (store.clipboard) {
        ev.preventDefault()
        pasteAtCursor()
      }
    } else if (ev.key === 'Delete' || ev.key === 'Backspace') {
      if (store.selectedAtoms.size > 0) {
        ev.preventDefault()
        store.editDeleteAtoms(store.selectedAtoms)
      }
    }
  }

  function updatePositionsCheap() {
    if (!stage || !structureObj) return
    const display = prepareDisplay()
    if (!display) return
    if (display.atomicNumbers.length !== structureObj.data.atoms.length) {
      // Atom count drifted — fall back to a full rebuild.
      rebuildStructure()
      return
    }
    // With dynamic bonding, re-detect bonds and rebuild if the topology
    // changed (a stretched bond broke, two atoms came into bonding range, …).
    // With dynamic bonding off, the bond list is frozen — what was on screen
    // at the last rebuild stays on screen regardless of the new positions.
    if (store.dynamicBonds) {
      const newBondsHash = bondsHash(display.positions, display.atomicNumbers)
      if (newBondsHash !== lastBondsHash) {
        rebuildStructure()
        return
      }
    }
    const flat = display.positions instanceof Float32Array
      ? display.positions
      : new Float32Array(display.positions as ArrayLike<number>)
    structureObj.updatePositions(flat)
    // Selection proxies are baked at build time — without this, the outline
    // pass keeps drawing the highlight at the pre-move positions while the
    // atoms render at the new ones (a literal "ghost" during a drag or
    // optimization run).
    if (store.selectedAtoms.size > 0) syncOutline()
    stage.invalidateBounds()
    stage.requestRender()
  }

  onMount(() => {
    const canvas = document.createElement('canvas')
    canvas.style.width = '100%'
    canvas.style.height = '100%'
    canvas.style.display = 'block'
    container.appendChild(canvas)
    canvasEl = canvas
    stage = new ViewerStage(canvas)
    stage.setOutlineColor(0xff9900)
    ghostFlash = new GhostFlash(stage.scene, () => stage?.requestRender())
    // Capture-phase interceptor (on window, an ancestor) runs before
    // TrackballControls' own pointerdown on the canvas, so an INSERT-mode atom
    // press can claim the gesture before the camera starts orbiting.
    window.addEventListener('pointerdown', onPointerDownCapture, true)
    canvas.addEventListener('pointerdown', onPointerDown)
    canvas.addEventListener('pointermove', onPointerMove)
    canvas.addEventListener('pointerup', onPointerUp)
    canvas.addEventListener('contextmenu', onContextMenu)
    window.addEventListener('keydown', onKey)
  })

  onDestroy(() => {
    machine.reset()
    window.removeEventListener('pointerdown', onPointerDownCapture, true)
    canvasEl?.removeEventListener('pointerdown', onPointerDown)
    canvasEl?.removeEventListener('pointermove', onPointerMove)
    canvasEl?.removeEventListener('pointerup', onPointerUp)
    canvasEl?.removeEventListener('contextmenu', onContextMenu)
    window.removeEventListener('keydown', onKey)
    structureObj?.dispose()
    structureObj = null
    ghostFlash?.dispose()
    ghostFlash = null
    stage?.dispose()
    stage = null
    canvasEl = null
  })

  // Mirror selection state → OutlinePass whenever the store changes.
  $effect(() => {
    void store.selectedAtoms
    syncOutline()
  })

  // Mute camera controls while the user is in a modal interaction. In INSERT
  // mode the camera stays live for orbiting empty space, but is muted while a
  // bond is being dragged off an atom (insertDragging). COMMAND mode mutes so
  // stray keys/drags don't move the view. TrackballControls keeps its own
  // pointer/keydown listeners on the canvas, so this gating is essential.
  $effect(() => {
    if (!stage) return
    const m = machine.mode
    const muted = m === 'transform' || m === 'box' || m === 'command' || insertDragging
    stage.controls.enabled = !muted
  })

  // Leaving edit mode (or any appMode change away from edit) resets the editor
  // machine so we never resume in a stale modal state.
  $effect(() => {
    if (store.appMode !== 'edit' && machine.mode !== 'normal') machine.reset()
  })

  // Apply the user's viewer background color whenever it changes. Hex string
  // → numeric color for the cebuns Stage API.
  $effect(() => {
    if (!stage) return
    const hex = store.viewerBackground.replace(/^#/, '')
    const n = parseInt(hex, 16)
    if (Number.isFinite(n)) stage.setBackground(n)
  })

  // Pulse a fading "moved here" cue whenever the store bumps flashTrigger.
  // Crucially the only reactive dep here is flashTrigger — the payload and
  // store.positions are read non-reactively, so a normal position stream
  // (drag preview, optimization steps) doesn't keep re-firing the flash.
  $effect(() => {
    void store.flashTrigger
    if (!ghostFlash) return
    const p = store.flashPayload
    if (!p) return
    ghostFlash.flash(p.positions, p.colors, p.radii)
  })

  // Constraint-axis indicator line in 3D — colored line through the selection
  // centroid along the locked axis (red x / green y / blue z / orange bond).
  $effect(() => {
    if (!stage) return
    void transform.mode
    void transform.axis  // touch reactive deps
    syncAxisLine()
  })

  function syncAxisLine() {
    if (!stage) return
    const info = machine.mode === 'transform' ? transform.constraintInfo : null
    if (!info) {
      if (axisLine) {
        stage.remove(axisLine)
        axisLine.geometry.dispose()
        ;(axisLine.material as THREE.Material).dispose()
        axisLine = null
      }
      if (pivotMarker) {
        stage.remove(pivotMarker)
        pivotMarker.geometry.dispose()
        ;(pivotMarker.material as THREE.Material).dispose()
        pivotMarker = null
      }
      return
    }
    const color = transform.axis === 'bond' ? AXIS_COLOR.bond
      : transform.axis === 'x' ? AXIS_COLOR.x
      : transform.axis === 'y' ? AXIS_COLOR.y
      : AXIS_COLOR.z
    const len = 50
    const dir = info.vec.clone().normalize().multiplyScalar(len)
    const a = info.origin.clone().sub(dir)
    const b = info.origin.clone().add(dir)
    const positions = [a.x, a.y, a.z, b.x, b.y, b.z]

    if (axisLine) {
      axisLine.geometry.setPositions(positions)
      const m = axisLine.material as LineMaterial
      m.color.set(color)
      m.needsUpdate = true
    } else {
      const geo = new LineSegmentsGeometry()
      geo.setPositions(positions)
      const mat = new LineMaterial({
        color,
        linewidth: 3,
        transparent: true,
        opacity: 0.95,
        depthTest: false,
        depthWrite: false,
        resolution: new THREE.Vector2(window.innerWidth, window.innerHeight),
      })
      axisLine = new LineSegments2(geo, mat)
      axisLine.computeLineDistances()
      axisLine.renderOrder = 999
      stage.add(axisLine)
    }

    // Pivot marker — small bright sphere at the centroid so the rotation /
    // translation pivot is unmistakable.
    if (!pivotMarker) {
      const g = new THREE.SphereGeometry(0.18, 16, 12)
      const m = new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity: 0.9,
        depthTest: false,
        depthWrite: false,
      })
      pivotMarker = new THREE.Mesh(g, m)
      pivotMarker.renderOrder = 1000
      stage.add(pivotMarker)
    }
    pivotMarker.position.copy(info.origin)
    ;(pivotMarker.material as THREE.MeshBasicMaterial).color.set(color)

    stage.requestRender()
  }

  // Single effect: reads positions every tick (so position-only updates trigger
  // it), then chooses rebuild vs cheap-update based on the structure key.
  $effect(() => {
    if (!stage) return
    void store.positions
    if (store.atomicNumbers.length === 0 || !store.positions) return

    const key = structureKey()
    if (key !== lastStructureKey || !structureObj) {
      lastStructureKey = key
      rebuildStructure()
    } else {
      updatePositionsCheap()
    }
  })
</script>

<div class="viewer" bind:this={container}>
  <!-- Single top-left HUD: tool toggle + mode-aware status / hints. -->
  <EditOverlay
    controller={transform}
    {store}
    mode={machine.mode}
    onselect={selectMode}
    ondraw={drawMode}
    onhelp={() => (store.helpOpen = true)}
  />

  {#if boxArmed && boxRect}
    <div
      class="box-rect"
      style="left: {boxRect.x}px; top: {boxRect.y}px; width: {boxRect.w}px; height: {boxRect.h}px;"
    ></div>
  {/if}

  <EditorHelp open={store.helpOpen} onclose={() => (store.helpOpen = false)} />

  {#if machine.mode === 'command'}
    <div class="cmdline">
      <span class="cmd-prompt">:</span>
      <input
        class="cmd-input"
        type="text"
        bind:value={machine.commandBuffer}
        onkeydown={onCommandKey}
        placeholder="e Fe · add O · fill h · del · relax"
        spellcheck="false"
        autocomplete="off"
        aria-label="Command line"
        use:focusOnMount
      />
    </div>
  {/if}
</div>

<style>
  .viewer {
    flex: 1;
    min-height: 0;
    width: 100%;
    background: var(--bg-primary);
    position: relative;
    overflow: hidden;
  }
  /* Box-select marquee rectangle (the tool toggle + hints live in EditOverlay). */
  .box-rect {
    position: absolute;
    border: 1px solid var(--accent);
    background: color-mix(in srgb, var(--accent) 14%, transparent);
    pointer-events: none;
    z-index: 5;
  }

  .cmdline {
    position: absolute;
    left: 0;
    right: 0;
    bottom: 0;
    display: flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.4rem 0.6rem;
    background: var(--bg-secondary);
    border-top: 1px solid var(--border);
    z-index: 8;
  }
  .cmd-prompt {
    color: var(--accent);
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-weight: 700;
  }
  .cmd-input {
    flex: 1;
    background: transparent;
    border: none;
    outline: none;
    color: var(--text-primary);
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.85rem;
  }
  .cmd-input::placeholder { color: var(--text-secondary); opacity: 0.7; }
</style>
