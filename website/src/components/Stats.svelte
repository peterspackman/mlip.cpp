<script lang="ts">
  import { getContext } from 'svelte'
  import type { SimulationStore } from '../lib/stores/simulation.svelte'

  const store = getContext<SimulationStore>('store')
  const total = $derived(store.mode === 'md' ? store.energy + store.kineticEnergy : store.energy)
</script>

<section class="panel-section">
  <h3>Readout</h3>
  <dl class="readout">
    <dt>Step</dt><dd>{store.step}</dd>
    <dt>Energy</dt><dd>{store.energy.toFixed(4)} eV</dd>
    {#if store.mode === 'md'}
      <dt>Kinetic</dt><dd>{store.kineticEnergy.toFixed(4)} eV</dd>
      <dt>Temperature</dt><dd>{store.currentTemperature.toFixed(1)} K</dd>
      <dt>Total</dt><dd>{total.toFixed(4)} eV</dd>
    {:else}
      <dt>Max force</dt><dd>{store.maxForce.toFixed(4)} eV/Å</dd>
      {#if store.isPeriodic}
        <dt>Max stress</dt><dd>{store.maxStress.toFixed(4)} eV/Å³</dd>
      {/if}
    {/if}
    <dt>Speed</dt><dd>{store.msPerStep.toFixed(0)} ms/step</dd>
  </dl>

  {#if store.lastStep?.timing}
    <p class="breakdown">
      predict: {store.lastStep.timing.predict.toFixed(1)}ms
      · system: {store.lastStep.timing.systemCreate.toFixed(1)}ms
    </p>
  {/if}
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
  .readout {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 0.25rem 0.75rem;
    font-size: 0.85rem;
    margin: 0;
  }
  dt {
    color: var(--text-secondary);
  }
  dd {
    font-variant-numeric: tabular-nums;
    margin: 0;
    font-weight: 500;
  }
  .breakdown {
    margin-top: 0.5rem;
    font-size: 0.7rem;
    color: var(--text-secondary);
    font-variant-numeric: tabular-nums;
  }
</style>
