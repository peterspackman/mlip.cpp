<script lang="ts">
  import type { TransformController } from '../lib/editor/transform.svelte'
  import type { SimulationStore } from '../lib/stores/simulation.svelte'
  import type { EditorMode } from '../lib/editor/editorMachine.svelte'
  import { getElementByNumber } from '../lib/molview/data/elements'

  interface Props {
    controller: TransformController
    store: SimulationStore
    mode: EditorMode
    onselect: () => void
    ondraw: () => void
    onhelp: () => void
  }
  let { controller, store, mode, onselect, ondraw, onhelp }: Props = $props()

  let selCount = $derived(store.selectedAtoms.size)
  let bondLabel = $derived(store.selectedBond
    ? `bond ${store.selectedBond[0]}–${store.selectedBond[1]}`
    : '')
  let activeSym = $derived(getElementByNumber(store.activeElement).symbol)
</script>

{#if store.appMode === 'edit'}
  <div class="hud">
    <div class="tools" role="group" aria-label="Editor tool">
      <button class:active={mode !== 'insert' && mode !== 'command'} onclick={onselect} title="Select & transform (Esc)">Select</button>
      <button class:active={mode === 'insert'} onclick={ondraw} title="Draw & build atoms (i)">Draw</button>
      <button class="help" onclick={onhelp} title="Shortcuts (?)" aria-label="Shortcuts">?</button>
    </div>

    <div class="status" role="status" aria-live="polite">
      {#if mode === 'transform'}
        <div class="line title">
          <span class="mode">{controller.mode === 'grab' ? 'GRAB' : 'ROTATE'}</span>
          {#if controller.axis}
            <span class="axis axis-{controller.axis}">
              {controller.axis === 'bond' ? 'BOND' : controller.axis.toUpperCase()}
            </span>
          {/if}
        </div>
        <div class="line readout">{controller.display}</div>
        <div class="line hints">X / Y / Z / B axis · type number · ↵ ✓ · esc ✗</div>
      {:else if mode === 'insert'}
        <div class="line title"><span class="mode">DRAW</span><span class="sub">adding {activeSym}</span></div>
        <div class="line hints">click: add · drag atom → space: bond · click atom: set · right-click: delete</div>
      {:else if mode === 'box'}
        <div class="line title"><span class="mode">BOX SELECT</span></div>
        <div class="line hints">drag a rectangle · shift add · ⌘/ctrl toggle · esc cancel</div>
      {:else if mode === 'command'}
        <div class="line title"><span class="mode">COMMAND</span></div>
        <div class="line hints">type a command below · esc cancel</div>
      {:else}
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
          g grab · r rotate · s box · del · <kbd>y</kbd> copy · <kbd>p</kbd> paste · <kbd>?</kbd> help
        </div>
      {/if}
    </div>
  </div>
{/if}

<style>
  .hud {
    position: absolute;
    top: 0.6rem;
    left: 0.6rem;
    max-width: calc(100% - 1.2rem);
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
    z-index: 6;
    pointer-events: none;
  }

  /* Tool toggle — the only interactive part. */
  .tools {
    display: flex;
    gap: 2px;
    padding: 2px;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 6px;
    width: fit-content;
    pointer-events: auto;
  }
  .tools button {
    padding: 0.25rem 0.6rem;
    background: transparent;
    color: var(--text-secondary);
    border: none;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 600;
    cursor: pointer;
  }
  .tools button:hover { color: var(--text-primary); }
  .tools button.active {
    background: var(--accent);
    color: #fff;
  }
  .tools .help { font-weight: 700; padding: 0.25rem 0.5rem; }

  /* Mode-aware status / hints — non-interactive. */
  .status {
    padding: 0.4rem 0.6rem;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 6px;
    font-size: 0.74rem;
    font-variant-numeric: tabular-nums;
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
    width: fit-content;
    max-width: 100%;
  }
  .line { display: flex; align-items: baseline; gap: 0.4rem; }
  .title { gap: 0.5rem; }
  .mode {
    font-weight: 700;
    font-size: 0.74rem;
    letter-spacing: 0.06em;
    color: var(--accent);
  }
  .sub { color: var(--text-secondary); font-size: 0.72rem; }
  .axis {
    font-weight: 700;
    font-size: 0.68rem;
    padding: 0.05rem 0.4rem;
    border-radius: 3px;
    letter-spacing: 0.06em;
    color: #fff;
  }
  .axis-x { background: #ff5252; }
  .axis-y { background: #4caf50; }
  .axis-z { background: #448aff; }
  .axis-bond { background: var(--accent); }
  .readout { font-weight: 600; color: var(--text-primary); }
  .hints { font-size: 0.68rem; color: var(--text-secondary); }
  .hints kbd {
    padding: 0.02rem 0.3rem;
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: 4px;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.64rem;
    color: var(--text-primary);
  }
</style>
