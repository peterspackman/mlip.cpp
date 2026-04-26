// Typed RPC wrapper around mdWorker.ts.
//
// The worker speaks a message-based protocol (`{ type, ...payload }` plus
// streamed events like `'mdStep'` and `'modelLoaded'`). This module hides
// that behind method-shaped calls and a small event bus so UI code never
// touches postMessage directly.

export type Backend = 'auto' | 'cpu' | 'webgpu'
export type Thermostat = 'csvr' | 'none'
export type Optimizer = 'lbfgs' | 'fire' | 'cg'

export interface MDStep {
  positions: Float64Array
  forces: Float64Array
  energy: number
  kineticEnergy: number
  temperature: number
  energyDrift: number
  timing: {
    verlet1: number
    systemCreate: number
    predict: number
    verlet2: number
    total: number
  }
}

export interface OptStep {
  positions: Float64Array
  cell: Float64Array | null
  energy: number
  maxForce: number
  maxStress: number
  step: number
  converged: boolean
}

export interface ModelInfo {
  modelType: string
  cutoff: number
  backend: string
}

export interface SystemInfo {
  numAtoms: number
  isPeriodic: boolean
}

export interface Prediction {
  energy: number
  forces: Float64Array
}

export type SimulationEvent =
  | { kind: 'mdStep'; step: MDStep }
  | { kind: 'optStep'; step: OptStep }
  | { kind: 'rattled'; positions: Float64Array }
  | { kind: 'started' }
  | { kind: 'stopped' }
  | { kind: 'optimizerStarted'; optimizer: Optimizer; forced: boolean }
  | { kind: 'error'; message: string }

type Listener = (ev: SimulationEvent) => void

// Events that the worker pushes without an RPC request. Everything else is
// a one-shot request/response keyed by message type.
const STREAM_TYPES = new Set(['mdStep', 'optStep', 'rattled', 'started', 'stopped', 'optimizerStarted', 'error'])

// One-shot response messages keyed by request type → response type.
const RESPONSE_FOR: Record<string, string> = {
  init: 'initialized',
  loadModel: 'modelLoaded',
  setBackend: 'backendSet',
  setSystem: 'systemSet',
  predict: 'prediction',
  setParameters: 'parametersSet',
  predictAt: 'predictAtResult',
}

export class Simulation {
  private worker: Worker
  private listeners = new Set<Listener>()
  private pending = new Map<string, { resolve: (v: any) => void; reject: (e: any) => void }>()
  private readyPromise: Promise<void>

  constructor() {
    this.worker = new Worker(new URL('./mdWorker.ts', import.meta.url), { type: 'module' })
    this.readyPromise = new Promise<void>((resolve) => {
      this.pending.set('ready', { resolve: () => resolve(), reject: () => {} })
    })
    this.worker.onmessage = (e: MessageEvent) => this.onMessage(e)
  }

  private onMessage(e: MessageEvent) {
    const msg = e.data
    if (!msg?.type) return

    // Errors reject any in-flight request AND emit a stream event so the UI
    // can show the message. Without this, a failed loadModel / setSystem /
    // predict hangs forever because the pending promise is keyed to the
    // specific response type.
    if (msg.type === 'error') {
      for (const [key, { reject }] of this.pending) {
        if (key === 'ready') continue
        this.pending.delete(key)
        reject(new Error(msg.message ?? 'Worker error'))
      }
      this.emit({ kind: 'error', message: msg.message })
      return
    }

    if (STREAM_TYPES.has(msg.type)) {
      const event = this.toEvent(msg)
      if (event) this.emit(event)
      return
    }

    // One-shot response: find a pending caller that expects this response type.
    for (const [key, { resolve }] of this.pending) {
      if (RESPONSE_FOR[key] === msg.type || key === msg.type) {
        this.pending.delete(key)
        resolve(msg)
        return
      }
    }
  }

  private toEvent(msg: any): SimulationEvent | null {
    switch (msg.type) {
      case 'mdStep':
        return { kind: 'mdStep', step: msg as MDStep }
      case 'optStep':
        return { kind: 'optStep', step: msg as OptStep }
      case 'rattled':
        return { kind: 'rattled', positions: msg.positions }
      case 'started':
        return { kind: 'started' }
      case 'stopped':
        return { kind: 'stopped' }
      case 'optimizerStarted':
        return { kind: 'optimizerStarted', optimizer: msg.optimizer, forced: msg.forced }
      case 'error':
        return { kind: 'error', message: msg.message }
      default:
        return null
    }
  }

  private emit(ev: SimulationEvent) {
    for (const l of this.listeners) l(ev)
  }

  private request<T>(type: string, payload: any = {}, transfers: Transferable[] = []): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      if (this.pending.has(type)) {
        reject(new Error(`Concurrent ${type} calls are not supported`))
        return
      }
      this.pending.set(type, { resolve, reject })
      this.worker.postMessage({ type, ...payload }, transfers)
    })
  }

  on(listener: Listener): () => void {
    this.listeners.add(listener)
    return () => this.listeners.delete(listener)
  }

  async ready(): Promise<void> {
    return this.readyPromise
  }

  async init(): Promise<{ version: string }> {
    return this.request('init')
  }

  async loadModel(buffer: ArrayBuffer, backend: Backend = 'auto'): Promise<ModelInfo> {
    return this.request('loadModel', { buffer, backend }, [buffer])
  }

  async setBackend(backend: Backend): Promise<{ backend: string }> {
    return this.request('setBackend', { backend })
  }

  async setSystem(xyz: string): Promise<SystemInfo> {
    return this.request('setSystem', { xyz })
  }

  async predict(): Promise<Prediction> {
    return this.request('predict')
  }

  async predictAt(positions: Float64Array): Promise<Prediction> {
    // Send a copy so callers can keep reusing their buffer across FD steps
    // without tripping on the transfer-then-detach semantics.
    const copy = new Float64Array(positions)
    const res = await this.request<any>('predictAt', { positions: copy }, [copy.buffer])
    if (res.error) throw new Error(res.error)
    return { energy: res.energy, forces: res.forces }
  }

  async setParameters(params: {
    dt?: number
    temperature?: number
    mode?: 'md' | 'optimize'
    maxOptSteps?: number
    forceThreshold?: number
    thermostat?: Thermostat
    thermostatTau?: number
    useConservativeForces?: boolean
    optimizer?: Optimizer
  }): Promise<void> {
    await this.request('setParameters', params)
  }

  start(stepsPerFrame = 1, mode: 'md' | 'optimize' = 'md', rattleAmount = 0, frozen?: number[]): void {
    this.worker.postMessage({ type: 'start', stepsPerFrame, mode, rattleAmount, frozen })
  }

  stop(): void {
    this.worker.postMessage({ type: 'stop' })
  }

  step(): void {
    this.worker.postMessage({ type: 'step' })
  }

  rattle(amount: number): void {
    this.worker.postMessage({ type: 'rattle', amount })
  }

  dispose(): void {
    this.worker.terminate()
    this.listeners.clear()
    this.pending.clear()
  }
}
