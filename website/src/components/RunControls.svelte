<script lang="ts">
  import { getContext } from 'svelte'
  import type { SimulationStore } from '../lib/stores/simulation.svelte'

  const store = getContext<SimulationStore>('store')
  const ready = $derived(store.modelStatus === 'ready' && store.numAtoms > 0)

  $effect(() => { store.syncParameters() })

  function toggleRun() {
    if (store.isRunning) store.stop()
    else store.start()
  }
</script>

<section class="panel-section">
  <h3>Run</h3>

  <div class="mode-row">
    <label>
      <input type="radio" bind:group={store.mode} value="md" />
      MD
    </label>
    <label>
      <input type="radio" bind:group={store.mode} value="optimize" />
      Optimize
    </label>
    <label>
      <input type="radio" bind:group={store.mode} value="vib" />
      Modes
    </label>
  </div>

  {#if store.mode !== 'vib'}
    <div class="button-row">
      <button class="run" onclick={toggleRun} disabled={!ready}>
        {store.isRunning ? 'Stop' : 'Start'}
      </button>
      <button onclick={() => store.stepOnce()} disabled={!ready || store.isRunning}>
        Step
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
  .panel-section h3 {
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .mode-row {
    display: flex;
    gap: 1rem;
    font-size: 0.85rem;
  }
  .button-row {
    display: flex;
    gap: 0.5rem;
  }
  button {
    flex: 1;
    padding: 0.5rem;
    border: 1px solid var(--border);
    border-radius: 4px;
    background-color: var(--bg-primary);
    color: var(--text-primary);
    cursor: pointer;
    font-size: 0.85rem;
  }
  button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  button.run {
    background-color: var(--accent);
    color: white;
    border-color: var(--accent);
  }
  button.run:hover:not(:disabled) {
    background-color: var(--accent-hover);
  }
</style>
