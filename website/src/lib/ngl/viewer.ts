import * as NGL from 'ngl'
import { positionsToSdf } from '../chem/sdf'
import { detectBonds, bondsKey, type Bond } from '../chem/bonds'
import { wrapPositions, type Lattice } from '../chem/cell'
import { generateSupercell } from '../chem/supercell'

export type ViewStyle = 'ball+stick' | 'licorice' | 'spacefill' | 'cartoon'

// Imperative NGL wrapper. Kept out of Svelte land so the reactive graph never
// talks to NGL directly — components call setStructure()/updatePositions().
export class Viewer {
  private stage: NGL.Stage | null = null
  private component: any = null
  private unitCell: any = null
  private lastBonds = ''
  private atomicNumbers: number[] = []
  private lattice: Lattice | null = null
  private supercell: [number, number, number] = [1, 1, 1]
  private wrap = false
  private style: ViewStyle = 'ball+stick'
  private drawing = false

  mount(el: HTMLElement) {
    this.stage = new NGL.Stage(el, {
      backgroundColor: this.preferredBg(),
      // Orthographic camera — no perspective-driven near-plane clipping when
      // zooming into a molecule, which is what you want for scientific viz.
      cameraType: 'orthographic',
      // Pull clip planes wide open so small molecules never slice through
      // the near plane; NGL's defaults are tuned for proteins.
      clipNear: 0,
      clipFar: 100,
      clipDist: 0,
      fogNear: 50,
      fogFar: 100,
    })
    window.addEventListener('resize', this.onResize)
  }

  dispose() {
    window.removeEventListener('resize', this.onResize)
    this.stage?.dispose()
    this.stage = null
    this.component = null
    this.unitCell = null
  }

  private onResize = () => this.stage?.handleResize()

  private preferredBg(): string {
    return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? '#1a1a1a' : '#ffffff'
  }

  setStyle(style: ViewStyle) {
    this.style = style
    if (this.component) {
      this.component.removeAllRepresentations?.()
      this.addRepresentation(this.component)
    }
  }

  setWrap(wrap: boolean) {
    this.wrap = wrap
  }

  setSupercell(size: [number, number, number]) {
    this.supercell = size
  }

  private addRepresentation(component: any) {
    const style = this.style
    if (style === 'spacefill') {
      component.addRepresentation('spacefill', { colorScheme: 'element', radiusScale: 1.0 })
    } else if (style === 'licorice') {
      component.addRepresentation('licorice', { colorScheme: 'element', radiusScale: 0.5 })
    } else if (style === 'cartoon') {
      component.addRepresentation('cartoon', { colorScheme: 'element' })
    } else {
      component.addRepresentation('ball+stick', { colorScheme: 'element', radiusScale: 0.5 })
    }
  }

  // Initial structure load (different atom list or first draw).
  async setStructure(
    positions: ArrayLike<number>,
    atomicNumbers: number[],
    lattice: Lattice | null,
  ) {
    if (!this.stage) return
    this.atomicNumbers = [...atomicNumbers]
    this.lattice = lattice
    this.lastBonds = ''
    await this.drawStructure(positions)
    if (lattice) this.drawUnitCell(lattice)
    else this.clearUnitCell()
    this.stage.autoView(0)
  }

  private async drawStructure(positions: ArrayLike<number>) {
    if (!this.stage) return
    if (this.drawing) return
    this.drawing = true
    try {
      const display = this.prepareDisplay(positions)
      const sdf = positionsToSdf(display.positions, display.atomicNumbers)
      this.lastBonds = bondsKey(display.bonds)

      const old = this.component
      // Drop the reference before the await so a stray updatePositions during
      // the load can't touch a half-swapped component.
      this.component = null
      if (old) {
        try {
          this.stage.removeComponent(old)
        } catch {
          /* ignore */
        }
      }
      const next = await this.stage.loadFile(new Blob([sdf], { type: 'text/plain' }), {
        ext: 'sdf',
        defaultRepresentation: false,
      })
      this.component = next
      this.addRepresentation(next)
    } finally {
      this.drawing = false
    }
  }

  private prepareDisplay(positions: ArrayLike<number>): {
    positions: ArrayLike<number>
    atomicNumbers: number[]
    bonds: Bond[]
  } {
    let pos: ArrayLike<number> = positions
    if (this.wrap && this.lattice) {
      pos = wrapPositions(positions, this.lattice)
    }
    let atoms = this.atomicNumbers
    if (this.lattice) {
      const sup = generateSupercell(pos, this.atomicNumbers, this.lattice, this.supercell)
      pos = sup.positions
      atoms = sup.atomicNumbers
    }
    return { positions: pos, atomicNumbers: atoms, bonds: detectBonds(pos, atoms) }
  }

  // Lightweight per-step update: if bonds haven't changed, just move atoms.
  async updatePositions(positions: ArrayLike<number>) {
    if (!this.stage || this.atomicNumbers.length === 0) return
    // Skip while a structure rebuild is in-flight; the rebuild will use the
    // new positions anyway once it finishes.
    if (this.drawing) return

    const display = this.prepareDisplay(positions)
    const key = bondsKey(display.bonds)
    if (key !== this.lastBonds) {
      await this.drawStructure(positions)
      return
    }

    const structure = this.component?.structure
    const store = structure?.atomStore
    // atomStore is populated asynchronously by NGL — its typed arrays may be
    // undefined for a tick after loadFile resolves. Bail rather than crash.
    if (!store || !store.x || !store.y || !store.z) return
    const n = display.atomicNumbers.length
    if (store.count !== n) return
    const p = display.positions
    for (let i = 0; i < n; i++) {
      store.x[i] = p[i * 3]
      store.y[i] = p[i * 3 + 1]
      store.z[i] = p[i * 3 + 2]
    }
    this.component.updateRepresentations({ position: true })
  }

  private drawUnitCell(lat: Lattice) {
    if (!this.stage) return
    this.clearUnitCell()
    const shape = new NGL.Shape('unit-cell')
    const color: [number, number, number] = [0.5, 0.5, 0.5]
    const o: [number, number, number] = [0, 0, 0]
    const a = lat.a as [number, number, number]
    const b = lat.b as [number, number, number]
    const c = lat.c as [number, number, number]
    const ab = add(a, b)
    const ac = add(a, c)
    const bc = add(b, c)
    const abc = add(ab, c)
    const edges: [typeof o, typeof o][] = [
      [o, a], [o, b], [o, c],
      [a, ab], [a, ac],
      [b, ab], [b, bc],
      [c, ac], [c, bc],
      [ab, abc], [ac, abc], [bc, abc],
    ]
    for (const [start, end] of edges) shape.addWideline(start, end, color)
    this.unitCell = this.stage.loadFile(shape as any)
  }

  private clearUnitCell() {
    if (this.unitCell && this.stage) {
      try {
        this.stage.removeComponent(this.unitCell)
      } catch {
        /* ignore */
      }
    }
    this.unitCell = null
  }

  centerView() {
    this.stage?.autoView(400)
  }
}

function add(a: [number, number, number], b: [number, number, number]): [number, number, number] {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
