<script lang="ts">
  import { getContext } from 'svelte'
  import type { SimulationStore } from '../lib/stores/simulation.svelte'
  import { formatFrequency, modeSummary } from '../lib/vib/modes'

  const store = getContext<SimulationStore>('store')

  const ready = $derived(store.modelStatus === 'ready' && store.numAtoms > 0)
  const n3 = $derived(store.numAtoms * 3)
  const visibleModes = $derived(
    store.vibShowImaginary
      ? store.vibModes
      : store.vibModes.filter((m) => !m.imaginary)
  )
  const hiddenImaginaryCount = $derived(
    store.vibModes.filter((m) => m.imaginary).length,
  )

  async function compute() {
    await store.computeVibrations()
  }

  function toggleMode(i: number) {
    if (store.activeMode === i && store.vibPlaying) {
      store.stopModeAnimation()
    } else {
      store.playMode(i)
    }
  }
</script>

<section class="panel-section">
  <h3>Vibrations</h3>

  {#if !store.vibModes.length && !store.vibComputing}
    <label class="opt-toggle">
      <input type="checkbox" bind:checked={store.vibOptimizeFirst} />
      <span>Optimize geometry first</span>
    </label>
    <label class="opt-toggle">
      <input type="checkbox" bind:checked={store.vibProjectTrRot} />
      <span>
        Project out {store.isPeriodic ? 'translations' : 'translations + rotations'}
      </span>
    </label>
    <p class="hint">
      Finite-difference Hessian from forces ({n3 * 2} predictions).
      {#if !store.vibOptimizeFirst}
        Expect imaginary modes if the geometry isn't a minimum.
      {/if}
    </p>
    <button class="primary" onclick={compute} disabled={!ready}>
      Compute modes
    </button>
  {/if}

  {#if store.vibComputing && store.vibProgress}
    <div class="progress">
      <div class="bar">
        <div
          class="fill"
          style="width: {(store.vibProgress.done / Math.max(store.vibProgress.total, 1)) * 100}%"
        ></div>
      </div>
      <p class="hint">
        {#if store.vibProgress.phase === 'optimize'}
          Optimizing · step {store.vibOptStep}
          · max F {store.vibOptMaxForce.toFixed(3)} eV/Å
        {:else if store.vibProgress.phase === 'hessian'}
          Evaluating forces · {store.vibProgress.done}/{store.vibProgress.total}
        {:else if store.vibProgress.phase === 'diagonalize'}
          Diagonalizing…
        {/if}
      </p>
    </div>
  {/if}

  {#if store.vibError}
    <p class="error">{store.vibError}</p>
  {/if}

  {#if store.vibModes.length > 0}
    <div class="slider-row">
      <div class="slider-label">
        <span>Amplitude</span>
        <span class="value">{store.vibAmplitude.toFixed(2)} Å</span>
      </div>
      <input
        type="range"
        min="0.05"
        max="0.8"
        step="0.05"
        bind:value={store.vibAmplitude}
      />
    </div>
    <div class="slider-row">
      <div class="slider-label">
        <span>Period</span>
        <span class="value">{(store.vibPeriodMs / 1000).toFixed(1)} s</span>
      </div>
      <input
        type="range"
        min="300"
        max="3000"
        step="100"
        bind:value={store.vibPeriodMs}
      />
    </div>

    {#if store.vibNProjected > 0}
      <p class="hint caption">
        {store.vibModes.length} vibrational modes
        · {store.vibNProjected} translation{store.vibNProjected === 3 ? '' : '/rotation'} mode{store.vibNProjected === 1 ? '' : 's'} projected out
      </p>
    {:else}
      <p class="hint caption">{store.vibModes.length} modes</p>
    {/if}

    <label class="opt-toggle inline">
      <input type="checkbox" bind:checked={store.vibShowImaginary} />
      <span>
        Show imaginary modes
        {#if hiddenImaginaryCount > 0 && !store.vibShowImaginary}
          ({hiddenImaginaryCount} hidden)
        {/if}
      </span>
    </label>

    <div class="mode-list">
      {#each visibleModes as mode (mode.index)}
        {@const absIdx = store.vibModes.indexOf(mode)}
        <button
          class="mode"
          class:active={store.activeMode === absIdx && store.vibPlaying}
          class:imaginary={mode.imaginary}
          onclick={() => toggleMode(absIdx)}
        >
          <span class="mode-idx">{absIdx + 1}</span>
          <span class="freq">{formatFrequency(mode)}</span>
          <span class="atoms">{modeSummary(mode, store.atomicNumbers)}</span>
        </button>
      {/each}
    </div>

    <div class="actions">
      <button class="ghost" onclick={() => store.stopModeAnimation()} disabled={!store.vibPlaying}>
        Stop
      </button>
      <button class="ghost" onclick={compute}>
        Recompute
      </button>
      <button class="ghost" onclick={() => store.clearVibrations()}>
        Clear
      </button>
    </div>
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
  h3 {
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin: 0;
  }
  .hint {
    font-size: 0.75rem;
    color: var(--text-secondary);
    line-height: 1.4;
    margin: 0;
  }
  .caption {
    font-size: 0.68rem;
    margin-top: -0.1rem;
  }
  .opt-toggle {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.8rem;
    color: var(--text-primary);
    cursor: pointer;
    padding: 0.35rem 0.5rem;
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: 4px;
  }
  .opt-toggle input {
    margin: 0;
    accent-color: var(--accent);
  }
  .opt-toggle.inline {
    padding: 0.25rem 0.5rem;
    font-size: 0.72rem;
    color: var(--text-secondary);
    background: transparent;
    border: none;
  }
  .error {
    font-size: 0.75rem;
    color: var(--error);
  }
  button {
    padding: 0.35rem 0.5rem;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: var(--bg-primary);
    color: var(--text-primary);
    font-size: 0.8rem;
    cursor: pointer;
  }
  button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  button.primary {
    background: var(--accent);
    color: white;
    border-color: var(--accent);
    font-size: 0.85rem;
    padding: 0.5rem;
  }
  button.primary:hover:not(:disabled) {
    background: var(--accent-hover);
  }

  .progress {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }
  .bar {
    height: 4px;
    background: var(--bg-primary);
    border-radius: 2px;
    overflow: hidden;
  }
  .fill {
    height: 100%;
    background: var(--accent);
    transition: width 0.1s;
  }

  /* Slider rows: compact two-line layout — label/value on top, track below. */
  .slider-row {
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
  }
  .slider-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.7rem;
    color: var(--text-secondary);
  }
  .slider-label .value {
    font-variant-numeric: tabular-nums;
    color: var(--text-primary);
    font-weight: 500;
  }
  input[type="range"] {
    -webkit-appearance: none;
    appearance: none;
    width: 100%;
    height: 4px;
    background: var(--border);
    border-radius: 2px;
    outline: none;
    margin: 0;
  }
  input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: var(--accent);
    cursor: pointer;
    border: 2px solid var(--bg-secondary);
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
    transition: transform 0.1s;
  }
  input[type="range"]::-webkit-slider-thumb:hover {
    transform: scale(1.15);
  }
  input[type="range"]::-moz-range-thumb {
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: var(--accent);
    cursor: pointer;
    border: 2px solid var(--bg-secondary);
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
  }

  /* Mode list: compact rows, clear active/imaginary states. */
  .mode-list {
    display: flex;
    flex-direction: column;
    max-height: 260px;
    overflow-y: auto;
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: 6px;
  }
  .mode {
    display: grid;
    grid-template-columns: 1.8rem 1fr auto;
    align-items: baseline;
    gap: 0.5rem;
    padding: 0.35rem 0.6rem;
    border: none;
    background: transparent;
    text-align: left;
    font-variant-numeric: tabular-nums;
    border-bottom: 1px solid var(--border);
    cursor: pointer;
    transition: background-color 0.1s;
  }
  .mode:last-child {
    border-bottom: none;
  }
  .mode-idx {
    font-size: 0.65rem;
    color: var(--text-secondary);
    opacity: 0.6;
  }
  .mode .freq {
    font-size: 0.78rem;
    font-weight: 500;
  }
  .mode .atoms {
    font-size: 0.7rem;
    color: var(--text-secondary);
  }
  .mode:hover {
    background: var(--bg-secondary);
  }
  .mode.active {
    background: var(--accent);
    color: white;
  }
  .mode.active .mode-idx,
  .mode.active .atoms {
    color: rgba(255, 255, 255, 0.85);
    opacity: 1;
  }
  .mode.imaginary .freq {
    color: var(--error);
  }
  .mode.active.imaginary .freq {
    color: #fecaca;
  }

  .actions {
    display: flex;
    gap: 0.25rem;
  }
  .actions .ghost {
    flex: 1;
    font-size: 0.72rem;
    padding: 0.3rem 0.4rem;
    background: transparent;
    border: 1px solid var(--border);
    color: var(--text-secondary);
  }
  .actions .ghost:hover:not(:disabled) {
    background: var(--bg-primary);
    color: var(--text-primary);
  }
</style>
