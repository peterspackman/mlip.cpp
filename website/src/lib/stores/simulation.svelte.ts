// Reactive store for the MD demo state. Uses Svelte 5 runes — every `$state`
// field is tracked, so components that read these fields rerender when they
// change. Constructed once in App.svelte and passed down through context.

import { Simulation, type Backend, type Thermostat, type Optimizer, type MDStep, type OptStep } from '../worker/simulation'
import type { Lattice } from '../chem/cell'
import { parsePositions, parseAtomicNumbers } from '../chem/xyz'
import { getMass } from '../chem/elements'
import { computeVibrations, type VibMode, type VibProgress } from '../vib/modes'

export type Mode = 'md' | 'optimize' | 'vib'
export type ModelStatus = 'empty' | 'loading' | 'ready' | 'error'

export class SimulationStore {
  readonly sim: Simulation

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
  viewStyle = $state<'ball+stick' | 'licorice' | 'spacefill' | 'cartoon'>('ball+stick')
  wrapPositions = $state(true)
  supercell = $state<[number, number, number]>([2, 2, 2])

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
    } catch (err: any) {
      this.modelStatus = 'error'
      this.modelError = err?.message ?? String(err)
    }
  }

  async setStructure(xyz: string, lattice: Lattice | null) {
    // Invalidate any vib analysis we have for the previous structure.
    this.clearVibrations()
    const info = await this.sim.setSystem(xyz)
    // Parse atoms/positions client-side — the worker sets them internally but
    // doesn't ship them back over the wire, and the viewer needs them to draw.
    this.atomicNumbers = parseAtomicNumbers(xyz)
    this.positions = new Float64Array(parsePositions(xyz))
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
  if (typeof navigator !== 'undefined' && /Firefox/i.test(navigator.userAgent)) {
    return 'cpu'
  }
  return 'auto'
}
