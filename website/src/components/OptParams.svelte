<script lang="ts">
  import { getContext } from 'svelte'
  import type { SimulationStore } from '../lib/stores/simulation.svelte'
  import Segmented from './Segmented.svelte'

  const store = getContext<SimulationStore>('store')
  const ready = $derived(store.modelStatus === 'ready' && store.numAtoms > 0)
  const cellForced = $derived(store.isPeriodic)  // cell-opt ignores the pick
</script>

<section class="panel-section">
  <h3>Optimization</h3>

  <label class="field">
    <span>Algorithm</span>
    <Segmented
      bind:value={store.optimizer}
      options={[
        { value: 'lbfgs', label: 'L-BFGS' },
        { value: 'cg',    label: 'CG' },
        { value: 'fire',  label: 'FIRE' },
      ]}
    />
  </label>
  <p class="algo-note">
    {#if cellForced}
      Periodic cell optimization uses FIRE regardless of this selection.
    {:else if store.activeOptimizer}
      Running <strong>{store.activeOptimizer === 'lbfgs' ? 'L-BFGS' : store.activeOptimizer === 'cg' ? 'CG' : 'FIRE'}</strong>
      {#if store.optimizerForced}(forced){/if}
    {:else if store.optimizer === 'lbfgs'}
      L-BFGS with max step 0.2 Å, Armijo backtracking line search.
    {:else if store.optimizer === 'cg'}
      Polak–Ribière+ conjugate gradient, Armijo backtracking. Lighter per step than L-BFGS — good cleanup after manual edits.
    {:else}
      FIRE — velocity-based, robust but slower near minima.
    {/if}
  </p>

  <div class="row">
    <label>
      Max Steps
      <input type="number" min="10" max="1000" step="10" bind:value={store.maxOptSteps} />
    </label>
    <label>
      F Tol (eV/Å)
      <input type="number" min="0.001" max="1.0" step="0.01" bind:value={store.forceThreshold} />
    </label>
  </div>

  <div class="row">
    <label>
      Rattle (Å)
      <input type="number" min="0" max="1.0" step="0.05" bind:value={store.rattleAmount} />
    </label>
    <label>
      &nbsp;
      <button onclick={() => store.rattle()} disabled={!ready || store.isRunning}>
        Rattle
      </button>
    </label>
  </div>

  <p class="note">
    Rattle perturbs atom positions to escape local minima before optimizing.
  </p>
  {#if store.optimizationConverged}
    <p class="note success">Converged.</p>
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
  .algo-note {
    margin: 0;
    font-size: 0.7rem;
    color: var(--text-secondary);
    line-height: 1.3;
  }
  input, button {
    padding: 0.3rem;
    border: 1px solid var(--border);
    border-radius: 4px;
    background-color: var(--bg-primary);
    color: var(--text-primary);
    font-size: 0.85rem;
  }
  button {
    cursor: pointer;
  }
  button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  .note {
    font-size: 0.75rem;
    color: var(--text-secondary);
  }
  .note.success {
    color: var(--success);
  }
</style>
