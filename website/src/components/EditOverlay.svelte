<script lang="ts">
  import type { TransformController } from '../lib/editor/transform.svelte'
  import type { SimulationStore } from '../lib/stores/simulation.svelte'

  let { controller, store }: { controller: TransformController; store: SimulationStore } = $props()

  let selCount = $derived(store.selectedAtoms.size)
  let bondLabel = $derived(store.selectedBond
    ? `bond ${store.selectedBond[0]}-${store.selectedBond[1]}`
    : '')
</script>

{#if store.appMode === 'edit'}
  <div class="badge" class:active={controller.active} role="status" aria-live="polite">
    {#if controller.active}
      <div class="line title">
        <span class="mode">{controller.mode === 'grab' ? 'GRAB' : 'ROTATE'}</span>
        {#if controller.axis}
          <span class="axis axis-{controller.axis}">
            {controller.axis === 'bond' ? 'BOND' : controller.axis.toUpperCase()}
          </span>
        {/if}
      </div>
      <div class="line readout">{controller.display}</div>
      <div class="line hints">
        X / Y / Z / B axis · type number · enter ✓ · esc ✗
      </div>
    {:else}
      <div class="line title"><span class="mode mode-idle">EDIT</span></div>
      <div class="line readout">
        {#if selCount === 0 && !bondLabel}
          nothing selected — click an atom
        {:else if bondLabel}
          {bondLabel}
        {:else}
          {selCount} atom{selCount === 1 ? '' : 's'} selected
        {/if}
      </div>
      <div class="line hints">
        G grab · R rotate · Del remove · X/Y/Z/B axis (during G/R)
      </div>
    {/if}
  </div>
{/if}

<style>
  .badge {
    position: absolute;
    top: 0.6rem;
    left: 0.6rem;
    min-width: 12rem;
    max-width: calc(100% - 1.2rem);
    padding: 0.4rem 0.6rem;
    background: rgba(15, 17, 24, 0.78);
    color: #fff;
    border-radius: 6px;
    font-size: 0.74rem;
    font-variant-numeric: tabular-nums;
    pointer-events: none;
    z-index: 5;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.35);
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
    transition: background 120ms ease, box-shadow 120ms ease;
  }
  .badge.active {
    background: rgba(20, 24, 32, 0.92);
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.55), 0 0 0 1px rgba(255, 153, 0, 0.5);
  }
  .line { display: flex; align-items: baseline; gap: 0.4rem; }
  .title { gap: 0.5rem; }
  .mode {
    font-weight: 700;
    font-size: 0.78rem;
    letter-spacing: 0.08em;
    color: #ff9900;
  }
  .mode-idle {
    color: #aab1bd;
    font-weight: 600;
  }
  .axis {
    font-weight: 700;
    font-size: 0.7rem;
    padding: 0.05rem 0.4rem;
    border-radius: 3px;
    letter-spacing: 0.08em;
  }
  .axis-x { background: #ff5252; color: #fff; }
  .axis-y { background: #4caf50; color: #fff; }
  .axis-z { background: #448aff; color: #fff; }
  .axis-bond { background: #ff9900; color: #1a1a1a; }
  .readout {
    font-weight: 600;
    color: #f0f2f5;
  }
  .hints {
    font-size: 0.66rem;
    color: rgba(255, 255, 255, 0.55);
  }
</style>
