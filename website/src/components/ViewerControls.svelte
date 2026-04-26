<script lang="ts">
  import { getContext } from 'svelte'
  import type { SimulationStore } from '../lib/stores/simulation.svelte'

  const store = getContext<SimulationStore>('store')

  // One-click background presets — covers the common "I want a publishable
  // figure" cases (white/light) and the "I'm working" cases (dark).
  const BG_PRESETS: { label: string; value: string }[] = [
    { label: 'midnight', value: '#1a1a2e' },
    { label: 'black',    value: '#000000' },
    { label: 'graphite', value: '#2a2a2a' },
    { label: 'paper',    value: '#f4f1ea' },
    { label: 'white',    value: '#ffffff' },
  ]
</script>

<div class="bar">
  <select bind:value={store.viewStyle}>
    <option value="ball+stick">ball+stick</option>
    <option value="spacefill">spacefill</option>
    <option value="wireframe">wireframe</option>
    <option value="tube">tube</option>
  </select>
  <label class="inline" title="Re-detect bonds from atomic distances each tick. Turn off to freeze the bond list while editing.">
    <input type="checkbox" bind:checked={store.dynamicBonds} />
    dynamic bonds
  </label>
  {#if store.isPeriodic}
    <label class="inline">
      <input type="checkbox" bind:checked={store.wrapPositions} />
      wrap
    </label>
    <label class="inline">
      <span>supercell</span>
      <input type="number" min="1" max="5" step="1" bind:value={store.supercell[0]} />
      ×
      <input type="number" min="1" max="5" step="1" bind:value={store.supercell[1]} />
      ×
      <input type="number" min="1" max="5" step="1" bind:value={store.supercell[2]} />
    </label>
  {/if}

  <div class="bg-row inline" role="group" aria-label="Viewer background">
    <span>bg</span>
    {#each BG_PRESETS as p}
      <button
        type="button"
        class="swatch"
        class:active={store.viewerBackground.toLowerCase() === p.value.toLowerCase()}
        style="background: {p.value};"
        onclick={() => (store.viewerBackground = p.value)}
        title={p.label}
        aria-label={`Background: ${p.label}`}
      ></button>
    {/each}
    <input
      type="color"
      class="picker"
      bind:value={store.viewerBackground}
      title="Custom color"
      aria-label="Custom background color"
    />
  </div>
</div>

<style>
  .bar {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.35rem 0.6rem;
    font-size: 0.75rem;
    color: var(--text-secondary);
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border);
    flex-wrap: wrap;
    flex-shrink: 0;
  }
  .inline {
    display: flex;
    align-items: center;
    gap: 0.35rem;
  }
  select, input[type="number"] {
    padding: 0.15rem 0.3rem;
    border: 1px solid var(--border);
    border-radius: 4px;
    background-color: var(--bg-primary);
    color: var(--text-primary);
    font-size: 0.75rem;
  }
  input[type="number"] {
    width: 2.5rem;
  }
  input[type="checkbox"] {
    margin: 0;
  }
  .bg-row {
    margin-left: auto;
    gap: 0.25rem;
  }
  .swatch {
    width: 0.95rem;
    height: 0.95rem;
    padding: 0;
    border: 1px solid var(--border);
    border-radius: 3px;
    cursor: pointer;
    transition: transform 80ms ease, border-color 80ms ease;
  }
  .swatch:hover { border-color: var(--text-primary); transform: translateY(-1px); }
  .swatch.active {
    border-color: #ff9900;
    box-shadow: 0 0 0 1.5px rgba(255, 153, 0, 0.6);
  }
  .picker {
    width: 1.4rem;
    height: 1.05rem;
    padding: 0;
    border: 1px solid var(--border);
    border-radius: 3px;
    background: transparent;
    cursor: pointer;
  }
  .picker::-webkit-color-swatch-wrapper { padding: 0; }
  .picker::-webkit-color-swatch { border: none; border-radius: 2px; }
</style>
