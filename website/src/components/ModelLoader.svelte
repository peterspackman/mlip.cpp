<script lang="ts">
  import { getContext } from 'svelte'
  import type { SimulationStore } from '../lib/stores/simulation.svelte'
  const store = getContext<SimulationStore>('store')
  let isDragging = $state(false)
  /** When a model is already loaded, hide the loader UI behind a "change"
   *  toggle — most of the time the user isn't swapping models. */
  let showLoaders = $state(false)

  async function loadBuffer(buffer: ArrayBuffer, source: string) {
    showLoaders = false
    await store.loadModel(buffer, source)
  }

  async function handleFile(file: File) {
    if (!/\.gguf$/i.test(file.name)) {
      store.modelError = 'Model file must have .gguf extension'
      return
    }
    const buffer = await file.arrayBuffer()
    await loadBuffer(buffer, file.name)
  }

  async function loadBundled() {
    const url = `${import.meta.env.BASE_URL}pet-mad-xs.gguf`
    try {
      const res = await fetch(url, { cache: 'no-store' })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const fetched = await res.arrayBuffer()
      const magic = new Uint8Array(fetched, 0, 4)
      const isGguf =
        magic[0] === 0x47 && magic[1] === 0x47 && magic[2] === 0x55 && magic[3] === 0x46
      if (!isGguf) {
        throw new Error(
          `fetched bytes don't start with GGUF magic (got ${Array.from(magic)
            .map((b) => b.toString(16).padStart(2, '0'))
            .join(' ')}) — likely a cache/transform issue`,
        )
      }
      const buffer = fetched.slice(0)
      await loadBuffer(buffer, 'pet-mad-xs.gguf')
    } catch (err: any) {
      store.modelStatus = 'error'
      store.modelError = `Failed to fetch bundled model: ${err?.message ?? err}`
    }
  }

  function onDrop(e: DragEvent) {
    e.preventDefault()
    isDragging = false
    const f = e.dataTransfer?.files?.[0]
    if (f) handleFile(f)
  }

  function onInput(e: Event) {
    const target = e.target as HTMLInputElement
    const f = target.files?.[0]
    if (f) handleFile(f)
    target.value = ''
  }

  let isLoading = $derived(store.modelStatus === 'loading')
  let isReady   = $derived(store.modelStatus === 'ready')
  let isError   = $derived(store.modelStatus === 'error')
  // When we have a model, default-collapsed; when none, default-expanded.
  let loadersVisible = $derived(showLoaders || (!isReady && !isLoading))

  /** Best-guess WebGPU availability. 'available' = navigator.gpu present and
   *  the browser actually exposes a real adapter. 'unreliable' = some Firefox /
   *  Safari builds expose navigator.gpu via flags but in practice fall over
   *  inside Web Workers (we run WASM in a worker). 'unavailable' = no API. */
  type GpuStatus = 'available' | 'unreliable' | 'unavailable'
  function detectGpu(): GpuStatus {
    if (typeof navigator === 'undefined') return 'available'
    if (!('gpu' in navigator)) return 'unavailable'
    const ua = navigator.userAgent
    if (/Firefox/i.test(ua)) return 'unreliable'
    const isSafari = /Safari/i.test(ua) && !/Chrome|Chromium|Android/i.test(ua)
    if (isSafari) return 'unreliable'
    return 'available'
  }
  const gpuStatus: GpuStatus = detectGpu()
</script>

<section class="panel-section">
  <h3>Model</h3>

  {#if isReady}
    <div class="status ready">
      <span class="check">✓</span>
      <div class="status-text">
        <div class="line-1">{store.modelSource}</div>
        <div class="line-2">{store.modelType} · {store.activeBackend || '?'} backend</div>
      </div>
      <button class="ghost-link"
        onclick={() => (showLoaders = !showLoaders)}
        disabled={isLoading || store.isRunning}
      >{showLoaders ? 'cancel' : 'change'}</button>
    </div>
  {:else if isLoading}
    <div class="status loading">
      <div class="spinner" aria-hidden="true"></div>
      <span>Loading {store.modelSource}…</span>
    </div>
  {:else if isError}
    <div class="status error">
      <span class="x">!</span>
      <span>{store.modelError}</span>
    </div>
  {:else}
    <p class="why">
      A model predicts energies and forces — load one to enable run / optimize.
    </p>
  {/if}

  {#if loadersVisible}
    <button
      class="primary"
      onclick={loadBundled}
      disabled={isLoading || store.isRunning}
    >
      <span class="primary-icon">⬇</span>
      <span class="primary-body">
        <span class="primary-title">Use bundled model</span>
        <span class="primary-sub">PET-MAD xs · ~16 MB</span>
      </span>
    </button>

    <div class="or"><span>or load your own</span></div>

    <div
      class="model-drop"
      class:dragging={isDragging}
      ondragover={(e) => { e.preventDefault(); isDragging = true }}
      ondragleave={() => { isDragging = false }}
      ondrop={onDrop}
      role="button"
      tabindex="-1"
    >
      <span class="drop-text">
        Drop a <code>.gguf</code> file
      </span>
      <label class="browse-link">
        or browse…
        <input type="file" accept=".gguf" style="display:none" onchange={onInput} />
      </label>
    </div>
  {/if}

  <label class="control-label" class:dim={!isReady && !loadersVisible}>
    <span>Backend</span>
    <select
      bind:value={store.backendChoice}
      onchange={() => { if (isReady) store.switchBackend() }}
      disabled={isLoading || store.isRunning}
    >
      <option value="auto">auto{gpuStatus === 'available' ? ' (prefer webgpu)' : ' (cpu)'}</option>
      <option value="cpu">cpu</option>
      <option value="webgpu" disabled={gpuStatus === 'unavailable'}>
        webgpu{gpuStatus !== 'available' ? ' — unsupported here' : ''}
      </option>
    </select>
  </label>

  {#if gpuStatus === 'unavailable'}
    <p class="gpu-note">
      Your browser doesn't expose WebGPU — calculations will run on CPU.
      For GPU acceleration, use <strong>Chrome</strong> or <strong>Edge</strong>.
    </p>
  {:else if gpuStatus === 'unreliable'}
    <p class="gpu-note">
      WebGPU is flaky in this browser (especially in Web Workers, which we
      use). Defaulting to CPU. For GPU acceleration, use <strong>Chrome</strong> or <strong>Edge</strong>.
    </p>
  {/if}
</section>

<style>
  .panel-section {
    padding: 0.75rem;
    background-color: var(--bg-secondary);
    border-radius: 8px;
    border: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }
  .panel-section h3 {
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin: 0;
  }

  .why {
    margin: 0;
    font-size: 0.74rem;
    color: var(--text-secondary);
    line-height: 1.4;
  }

  .status {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.45rem 0.5rem;
    border-radius: 6px;
    font-size: 0.78rem;
  }
  .status.ready {
    background: color-mix(in srgb, var(--success, #4ade80) 14%, transparent);
    border: 1px solid color-mix(in srgb, var(--success, #4ade80) 30%, transparent);
    color: var(--text-primary);
  }
  .status.ready .check {
    color: var(--success, #4ade80);
    font-weight: 700;
  }
  .status-text {
    flex: 1;
    min-width: 0;
    line-height: 1.25;
  }
  .status-text .line-1 {
    font-weight: 600;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .status-text .line-2 {
    font-size: 0.7rem;
    color: var(--text-secondary);
  }
  .ghost-link {
    background: transparent;
    border: none;
    color: var(--text-secondary);
    font-size: 0.72rem;
    text-decoration: underline;
    text-underline-offset: 2px;
    cursor: pointer;
    padding: 0;
  }
  .ghost-link:hover:not(:disabled) { color: var(--text-primary); }
  .ghost-link:disabled { opacity: 0.4; cursor: default; }

  .status.loading {
    color: var(--text-secondary);
  }
  .spinner {
    width: 12px;
    height: 12px;
    border: 2px solid var(--border);
    border-top-color: var(--text-primary);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  .status.error {
    background: color-mix(in srgb, var(--error, #ef4444) 12%, transparent);
    border: 1px solid color-mix(in srgb, var(--error, #ef4444) 30%, transparent);
    color: var(--text-primary);
    align-items: flex-start;
    line-height: 1.35;
  }
  .status.error .x {
    color: var(--error, #ef4444);
    font-weight: 800;
    margin-top: 0.05rem;
  }

  .primary {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    padding: 0.55rem 0.65rem;
    border-radius: 6px;
    background: var(--accent, #ff9900);
    color: #1a1a1a;
    border: 1px solid transparent;
    cursor: pointer;
    text-align: left;
    font-family: inherit;
    transition: transform 80ms ease, filter 120ms ease;
  }
  .primary:hover:not(:disabled) {
    filter: brightness(1.08);
    transform: translateY(-1px);
  }
  .primary:disabled { opacity: 0.5; cursor: default; }
  .primary-icon {
    font-size: 1.05rem;
    line-height: 1;
  }
  .primary-body {
    display: flex;
    flex-direction: column;
    gap: 0.05rem;
    line-height: 1.2;
  }
  .primary-title {
    font-weight: 700;
    font-size: 0.82rem;
  }
  .primary-sub {
    font-size: 0.66rem;
    opacity: 0.78;
  }

  .or {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: var(--text-secondary);
    font-size: 0.66rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  .or::before, .or::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
  }

  .model-drop {
    border: 1.5px dashed var(--border);
    border-radius: 6px;
    padding: 0.55rem 0.65rem;
    text-align: center;
    transition: background-color 0.15s, border-color 0.15s, color 0.15s;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.4rem;
    color: var(--text-secondary);
    cursor: copy;
    font-size: 0.74rem;
  }
  .model-drop.dragging {
    background-color: color-mix(in srgb, var(--success, #4ade80) 12%, transparent);
    border-color: var(--success, #4ade80);
    border-style: solid;
    color: var(--success, #4ade80);
  }
  .drop-text { white-space: nowrap; }
  code {
    background-color: var(--bg-primary);
    padding: 0.02rem 0.28rem;
    border-radius: 3px;
    font-size: 0.7rem;
    color: var(--text-primary);
  }
  .browse-link {
    font-size: 0.72rem;
    text-decoration: underline;
    text-underline-offset: 2px;
    cursor: pointer;
    opacity: 0.8;
    color: var(--text-secondary);
  }
  .browse-link:hover { opacity: 1; color: var(--text-primary); }

  .control-label {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.4rem;
    font-size: 0.74rem;
    color: var(--text-secondary);
  }
  .control-label.dim { opacity: 0.7; }
  select {
    padding: 0.25rem 0.4rem;
    border: 1px solid var(--border);
    border-radius: 4px;
    background-color: var(--bg-primary);
    color: var(--text-primary);
    font-size: 0.74rem;
  }
  .gpu-note {
    margin: 0;
    padding: 0.4rem 0.5rem;
    background: color-mix(in srgb, var(--accent, #ff9900) 12%, transparent);
    border: 1px solid color-mix(in srgb, var(--accent, #ff9900) 28%, transparent);
    border-radius: 5px;
    font-size: 0.7rem;
    line-height: 1.4;
    color: var(--text-secondary);
  }
  .gpu-note strong {
    color: var(--text-primary);
  }
</style>
