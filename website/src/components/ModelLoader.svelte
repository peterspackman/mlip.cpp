<script lang="ts">
  import { getContext } from 'svelte'
  import type { SimulationStore } from '../lib/stores/simulation.svelte'
  const store = getContext<SimulationStore>('store')
  let isDragging = $state(false)

  async function loadBuffer(buffer: ArrayBuffer, source: string) {
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
      // no-store bypasses the HTTP cache — we want the freshly-built file,
      // not a stale cached copy from a previous convert_models.py run.
      const res = await fetch(url, { cache: 'no-store' })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const fetched = await res.arrayBuffer()
      // Sanity-check the GGUF magic — catches caching / transform layers
      // that might silently hand back a truncated or compressed payload.
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
      // Fresh standalone ArrayBuffer so the transfer into the worker is
      // guaranteed clean (Response.arrayBuffer() can return a buffer with
      // different backing semantics than File.arrayBuffer()).
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
</script>

<section class="panel-section">
  <h3>Model</h3>

  <label class="control-label">
    Backend
    <select
      bind:value={store.backendChoice}
      disabled={store.modelStatus === 'loading' || store.isRunning}
    >
      <option value="auto">auto</option>
      <option value="cpu">cpu</option>
      <option value="webgpu">webgpu</option>
    </select>
  </label>

  <div
    class="model-drop"
    class:dragging={isDragging}
    ondragover={(e) => { e.preventDefault(); isDragging = true }}
    ondragleave={() => { isDragging = false }}
    ondrop={onDrop}
    role="button"
    tabindex="-1"
  >
    <svg class="drop-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
      <path d="M12 3v12m0 0l-4-4m4 4l4-4M4 17v2a2 2 0 002 2h12a2 2 0 002-2v-2" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
    <p>
      Drop a <code>.gguf</code> model here
    </p>
    <label class="browse-link">
      or browse…
      <input type="file" accept=".gguf" style="display:none" onchange={onInput} />
    </label>
  </div>

  <button
    class="bundled-link"
    onclick={loadBundled}
    disabled={store.modelStatus === 'loading' || store.isRunning}
    title="Load the bundled small PET-MAD model"
  >
    use bundled <code>pet-mad-xs.gguf</code>
  </button>

  <div class="model-status">
    {#if store.modelStatus === 'loading'}
      <span>Loading {store.modelSource}…</span>
    {:else if store.modelStatus === 'ready'}
      <span>{store.modelType} · {store.activeBackend || 'backend?'} · {store.modelSource}</span>
    {:else if store.modelStatus === 'error'}
      <span class="error">{store.modelError}</span>
    {:else}
      <span class="muted">No model loaded</span>
    {/if}
  </div>
</section>

<style>
  .panel-section {
    padding: 0.75rem;
    background-color: var(--bg-secondary);
    border-radius: 8px;
    border: 1px solid var(--border);
  }
  .panel-section h3 {
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
  }
  .control-label {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    font-size: 0.8rem;
    color: var(--text-secondary);
    margin-bottom: 0.5rem;
  }
  .model-drop {
    border: 2px dashed var(--border);
    border-radius: 8px;
    padding: 1rem 0.75rem;
    text-align: center;
    transition: background-color 0.15s, border-color 0.15s, color 0.15s;
    margin-bottom: 0.4rem;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.25rem;
    color: var(--text-secondary);
    cursor: copy;
  }
  .model-drop.dragging {
    background-color: color-mix(in srgb, var(--success) 15%, transparent);
    border-color: var(--success);
    border-style: solid;
    color: var(--success);
  }
  .drop-icon {
    width: 28px;
    height: 28px;
    opacity: 0.65;
    transition: opacity 0.15s;
  }
  .model-drop.dragging .drop-icon {
    opacity: 1;
  }
  .model-drop p {
    margin: 0;
    font-size: 0.8rem;
  }
  code {
    background-color: var(--bg-primary);
    padding: 0.05rem 0.3rem;
    border-radius: 3px;
    font-size: 0.75rem;
    color: var(--text-primary);
  }
  .browse-link {
    font-size: 0.72rem;
    text-decoration: underline;
    text-underline-offset: 2px;
    cursor: pointer;
    opacity: 0.8;
  }
  .browse-link:hover {
    opacity: 1;
  }
  .bundled-link {
    align-self: center;
    padding: 0.2rem 0.4rem;
    font-size: 0.72rem;
    color: var(--text-secondary);
    background: transparent;
    border: none;
    cursor: pointer;
    text-decoration: underline;
    text-underline-offset: 2px;
  }
  .bundled-link:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }
  .bundled-link:hover:not(:disabled) {
    color: var(--text-primary);
  }
  .bundled-link code {
    background: transparent;
    padding: 0;
    font-size: 0.72rem;
    color: inherit;
  }
  select {
    padding: 0.4rem;
    border: 1px solid var(--border);
    border-radius: 4px;
    background-color: var(--bg-primary);
    color: var(--text-primary);
    font-size: 0.85rem;
  }
  .model-status {
    font-size: 0.75rem;
    color: var(--text-secondary);
    min-height: 1rem;
  }
  .model-status .error {
    color: var(--error);
  }
  .model-status .muted {
    opacity: 0.6;
  }
</style>
