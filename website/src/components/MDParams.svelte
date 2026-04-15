<script lang="ts">
  import { getContext } from 'svelte'
  import type { SimulationStore } from '../lib/stores/simulation.svelte'
  import Segmented from './Segmented.svelte'

  const store = getContext<SimulationStore>('store')

  const forceMode = $derived<'nc' | 'conservative'>(
    store.useConservativeForces ? 'conservative' : 'nc'
  )
  function setForceMode(v: 'nc' | 'conservative') {
    store.useConservativeForces = v === 'conservative'
  }
</script>

<section class="panel-section">
  <h3>MD Parameters</h3>

  <div class="row">
    <label>
      Temp (K)
      <input
        type="number"
        min="1"
        max="1000"
        step="10"
        bind:value={store.temperature}
        disabled={store.thermostat === 'none'}
        title={store.thermostat === 'none' ? 'NVE — temperature is not controlled' : ''}
      />
    </label>
    <label>
      Timestep (fs)
      <input type="number" min="0.1" max="2.0" step="0.1" bind:value={store.timestep} />
    </label>
  </div>

  <label class="field">
    <span>Thermostat</span>
    <Segmented
      bind:value={store.thermostat}
      options={[
        { value: 'csvr', label: 'CSVR' },
        { value: 'none', label: 'NVE' },
      ]}
    />
  </label>
  <label class="field">
    <span>Forces</span>
    <Segmented
      value={forceMode}
      onchange={setForceMode}
      options={[
        { value: 'nc', label: 'Fast' },
        { value: 'conservative', label: 'Conservative' },
      ]}
    />
  </label>

  <p class="note">
    {#if store.useConservativeForces && store.thermostat === 'none'}
      NVE — total energy should be conserved.
    {:else if store.useConservativeForces}
      Conservative forces with thermostat.
    {:else}
      Non-conservative forces are faster but energy will drift.
    {/if}
  </p>

  {#if store.step > 0}
    <p class="note">
      Drift: <strong>{store.energyDrift.toFixed(4)} eV</strong>
      ({(store.energyDrift * 1000 / Math.max(store.step, 1)).toFixed(3)} meV/step)
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
  }
  .row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.5rem;
  }
  label, .field {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    font-size: 0.75rem;
    color: var(--text-secondary);
  }
  .field > span {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.03em;
  }
  input {
    padding: 0.3rem;
    border: 1px solid var(--border);
    border-radius: 4px;
    background-color: var(--bg-primary);
    color: var(--text-primary);
    font-size: 0.85rem;
  }
  input:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }
  .note {
    font-size: 0.75rem;
    color: var(--text-secondary);
    line-height: 1.3;
  }
</style>
