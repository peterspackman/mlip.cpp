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
  import { GhostFlash } from '../lib/editor/ghostFlash'
  import EditOverlay from './EditOverlay.svelte'
  import type { SimulationStore } from '../lib/stores/simulation.svelte'
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
  let boxArmed = $state(false)
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

  function exitBoxMode(): void {
    boxArmed = false
    boxStart = null
    boxEnd = null
  }

  function commitBoxSelection(): void {
    if (!stage || !canvasEl || !boxStart || !boxEnd) { exitBoxMode(); return }
    const display = prepareDisplay()
    if (!display) { exitBoxMode(); return }
    const minX = Math.min(boxStart.x, boxEnd.x)
    const maxX = Math.max(boxStart.x, boxEnd.x)
    const minY = Math.min(boxStart.y, boxEnd.y)
    const maxY = Math.max(boxStart.y, boxEnd.y)
    // Click without drag → ignore (user just tapped, treat as cancel).
    if (maxX - minX < 2 && maxY - minY < 2) { exitBoxMode(); return }

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
    exitBoxMode()
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
    if (fitKey !== lastFitKey && !transform.active && !store.isRunning) {
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
    if (transform.active) {
      ev.preventDefault()
      pointerOwnedByTransform = true
      if (ev.button === 0) transform.commit()
      else if (ev.button === 2) transform.cancel()
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
      exitBoxMode()
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
    if (transform.active) {
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
    if (boxArmed && boxStart && ev.button === 0) {
      ev.preventDefault()
      pointerOwnedByTransform = false
      commitBoxSelection()
      return
    }
    if (pointerOwnedByTransform) {
      // Commit/cancel ran on the matching pointerdown; swallow the release.
      pointerOwnedByTransform = false
      return
    }
    if (transform.active) return  // a fresh transform was begun via keyboard between down and up
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
    if (!transform.active) return
    ev.preventDefault()
    transform.cancel()
  }

  function isTypingTarget(t: EventTarget | null): boolean {
    if (!(t instanceof HTMLElement)) return false
    const tag = t.tagName
    return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || t.isContentEditable
  }

  function onKey(ev: KeyboardEvent) {
    if (store.appMode !== 'edit') return
    if (isTypingTarget(ev.target)) return

    // ⌘/Ctrl combos (undo, redo, copy, paste, duplicate). Inert during a
    // modal transform — the user has Esc / Enter for that.
    if ((ev.metaKey || ev.ctrlKey) && !transform.active) {
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
          store.editPaste()
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
    if (transform.active) {
      if (ev.key === 'Escape') { ev.preventDefault(); transform.cancel(); return }
      if (ev.key === 'Enter')  { ev.preventDefault(); transform.commit(); return }
      const k = ev.key.toLowerCase()
      // G / R inside an active transform = switch mode (preserves the in-progress effect).
      if (k === 'g') { ev.preventDefault(); transform.switchMode('grab'); return }
      if (k === 'r') { ev.preventDefault(); transform.switchMode('rotate'); return }
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

    // Box-select cancellation.
    if (boxArmed && ev.key === 'Escape') {
      ev.preventDefault()
      exitBoxMode()
      return
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
      transform.begin('grab', lastPointer.x, lastPointer.y)
    } else if (ev.key === 'r' || ev.key === 'R') {
      ev.preventDefault()
      transform.begin('rotate', lastPointer.x, lastPointer.y)
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
        boxArmed = true
        boxStart = null
        boxEnd = null
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
    canvas.addEventListener('pointerdown', onPointerDown)
    canvas.addEventListener('pointermove', onPointerMove)
    canvas.addEventListener('pointerup', onPointerUp)
    canvas.addEventListener('contextmenu', onContextMenu)
    window.addEventListener('keydown', onKey)
  })

  onDestroy(() => {
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

  // Mute camera controls while the user is in a modal interaction (grab /
  // rotate transform OR box-select). TrackballControls keeps its own
  // pointer/keydown listeners on the canvas, so without this they'd also
  // rotate the camera while the user is drawing the marquee.
  $effect(() => {
    if (!stage) return
    stage.controls.enabled = !transform.active && !boxArmed
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
    const info = transform.active ? transform.constraintInfo : null
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
  <EditOverlay controller={transform} {store} />
  {#if boxArmed}
    <div class="box-hint" role="status" aria-live="polite">
      BOX SELECT — drag · shift = add · ⌘/ctrl = toggle · esc cancels
    </div>
    {#if boxRect}
      <div
        class="box-rect"
        style="left: {boxRect.x}px; top: {boxRect.y}px; width: {boxRect.w}px; height: {boxRect.h}px;"
      ></div>
    {/if}
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
  .box-hint {
    position: absolute;
    top: 0.6rem;
    right: 0.6rem;
    padding: 0.3rem 0.55rem;
    background: rgba(15, 17, 24, 0.92);
    color: #ff9900;
    border-radius: 6px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    pointer-events: none;
    z-index: 6;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.45), 0 0 0 1px rgba(255, 153, 0, 0.4);
  }
  .box-rect {
    position: absolute;
    border: 1px solid #ff9900;
    background: rgba(255, 153, 0, 0.12);
    pointer-events: none;
    z-index: 5;
  }
</style>
