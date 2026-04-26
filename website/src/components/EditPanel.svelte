<script lang="ts">
  import { getContext } from 'svelte'
  import type { SimulationStore } from '../lib/stores/simulation.svelte'
  import { getElementByNumber } from '../lib/molview/data/elements'
  import { FRAGMENTS, FRAGMENT_ORDER } from '../lib/editor/fragments'
  import PeriodicTable from './PeriodicTable.svelte'

  const store = getContext<SimulationStore>('store')

  // Quick palette for the most-used elements. Anything else opens the popover.
  const COMMON: number[] = [1, 6, 7, 8, 9, 15, 16, 17]
  let currentZ = $state(6)
  let ptableOpen = $state(false)

  function rgbStr(c: [number, number, number]): string {
    return `rgb(${Math.round(c[0] * 255)}, ${Math.round(c[1] * 255)}, ${Math.round(c[2] * 255)})`
  }
  function inkFor(c: [number, number, number]): string {
    const y = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
    return y > 0.55 ? '#1a1a1a' : '#fff'
  }

  function atomZ(i: number): number | undefined { return store.atomicNumbers[i] }
  function atomPosition(i: number): [number, number, number] | null {
    const p = store.positions
    if (!p || i * 3 + 2 >= p.length) return null
    return [p[i * 3], p[i * 3 + 1], p[i * 3 + 2]]
  }
  function distance(a: number, b: number): number | null {
    const pa = atomPosition(a), pb = atomPosition(b)
    if (!pa || !pb) return null
    return Math.hypot(pa[0] - pb[0], pa[1] - pb[1], pa[2] - pb[2])
  }
  function angleDeg(i: number, j: number, k: number): number | null {
    const pi = atomPosition(i), pj = atomPosition(j), pk = atomPosition(k)
    if (!pi || !pj || !pk) return null
    const v1 = [pi[0] - pj[0], pi[1] - pj[1], pi[2] - pj[2]]
    const v2 = [pk[0] - pj[0], pk[1] - pj[1], pk[2] - pj[2]]
    const dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
    const l1 = Math.hypot(v1[0], v1[1], v1[2])
    const l2 = Math.hypot(v2[0], v2[1], v2[2])
    if (l1 === 0 || l2 === 0) return null
    return (Math.acos(Math.max(-1, Math.min(1, dot / (l1 * l2)))) * 180) / Math.PI
  }
  function selectionCentroid(): [number, number, number] {
    const sel = [...store.selectedAtoms]
    if (sel.length === 0) return [0, 0, 0]
    let x = 0, y = 0, z = 0, n = 0
    for (const i of sel) {
      const p = atomPosition(i)
      if (!p) continue
      x += p[0]; y += p[1]; z += p[2]; n++
    }
    if (n === 0) return [0, 0, 0]
    return [x / n, y / n, z / n]
  }
  function fmtNum(n: number, digits = 3): string { return n.toFixed(digits) }
  function clearSelection() {
    store.selectedAtoms = new Set()
    store.selectedBond = null
  }

  async function setElementOnSelection() {
    if (store.selectedAtoms.size === 0) return
    await store.editSetElement(store.selectedAtoms, currentZ)
  }
  async function deleteSelection() {
    if (store.selectedAtoms.size === 0) return
    await store.editDeleteAtoms(store.selectedAtoms)
  }
  async function addAtomNearSelection() {
    const c = selectionCentroid()
    await store.editAddAtom(currentZ, [c[0], c[1], c[2] + 1.5])
  }
  async function fillHydrogens() {
    await store.editFillHydrogens(store.selectedAtoms.size > 0 ? store.selectedAtoms : null)
  }
  async function copySelection()      { store.editCopy() }
  async function pasteClipboard()     { await store.editPaste() }
  async function duplicateSelection() { await store.editDuplicate() }

  let selected = $derived([...store.selectedAtoms])
  let runningGuard = $derived(store.isRunning)
  let hasSelection = $derived(selected.length > 0 || !!store.selectedBond)

  const SLOTS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
  function handleSlotClick(slot: number, ev: MouseEvent) {
    if (ev.shiftKey) {
      store.selectionGroups = { ...store.selectionGroups, [slot]: [...store.selectedAtoms] }
      return
    }
    const saved = store.selectionGroups[slot]
    if (!saved || saved.length === 0) return
    store.selectedAtoms = new Set(saved)
    store.selectedBond = null
  }
  function clearSlot(slot: number) {
    const next = { ...store.selectionGroups }
    delete next[slot]
    store.selectionGroups = next
  }

  // Bond-length editing — exposed when 2 atoms or a bond are selected.
  let bondPair = $derived.by((): [number, number] | null => {
    if (store.selectedBond) return store.selectedBond
    if (selected.length === 2) return [selected[0], selected[1]]
    return null
  })
  let bondLength = $derived.by((): number | null => {
    if (!bondPair) return null
    return distance(bondPair[0], bondPair[1])
  })
  let bondInput = $state('')
  let bondMoveSide = $state<'a' | 'b'>('b')
  $effect(() => {
    if (bondLength !== null && document.activeElement?.tagName !== 'INPUT') {
      bondInput = bondLength.toFixed(3)
    }
  })
  async function commitBondLength() {
    if (!bondPair) return
    const v = parseFloat(bondInput)
    if (!Number.isFinite(v) || v < 0.1) return
    await store.editSetBondLength(bondPair[0], bondPair[1], v, bondMoveSide)
  }

  async function applyFragment(key: string) {
    if (selected.length === 0) return
    const frag = FRAGMENTS[key]
    if (!frag) return
    const ordered = [...selected].sort((a, b) => a - b)
    await store.editBatch(async () => {
      for (const i of ordered) await store.editReplaceWithFragment(i, frag)
    })
  }

  let optAlgo = $state<'cg' | 'lbfgs' | 'fire'>('lbfgs')
  function relaxAll()      { store.runOptimize({ algorithm: optAlgo }) }
  function relaxSelected() { store.runOptimize({ algorithm: optAlgo, onlySelected: true }) }
  function stopOpt()       { store.stop() }
  let modelReady = $derived(store.modelStatus === 'ready')

  function exportXyz() {
    if (!store.currentXyz) return
    const blob = new Blob([store.currentXyz], { type: 'chemical/x-xyz' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')
    a.href = url
    a.download = `structure-${stamp}.xyz`
    document.body.appendChild(a)
    a.click()
    a.remove()
    URL.revokeObjectURL(url)
  }

  /** Brief 'Copied' confirmation flag for the copy-XYZ icon. */
  let xyzCopied = $state(false)
  let xyzCopyTimer: number | null = null
  async function copyXyz() {
    if (!store.currentXyz) return
    try {
      await navigator.clipboard.writeText(store.currentXyz)
      xyzCopied = true
      if (xyzCopyTimer !== null) clearTimeout(xyzCopyTimer)
      xyzCopyTimer = window.setTimeout(() => (xyzCopied = false), 1100)
    } catch {
      // Fallback for browsers without secure-context clipboard access.
      const ta = document.createElement('textarea')
      ta.value = store.currentXyz
      ta.style.position = 'fixed'
      ta.style.opacity = '0'
      document.body.appendChild(ta)
      ta.select()
      try { document.execCommand('copy') } catch {}
      ta.remove()
      xyzCopied = true
      if (xyzCopyTimer !== null) clearTimeout(xyzCopyTimer)
      xyzCopyTimer = window.setTimeout(() => (xyzCopied = false), 1100)
    }
  }
</script>

<div class="panel-section">
  <div class="section-header">
    <span class="section-title">Structure</span>
    <div class="header-actions">
      <button class="ghost icon"
        onclick={() => store.undo()}
        disabled={!store.canUndo || runningGuard}
        title="Undo (⌘/Ctrl+Z)">↶</button>
      <button class="ghost icon"
        onclick={() => store.redo()}
        disabled={!store.canRedo || runningGuard}
        title="Redo (⌘/Ctrl+Shift+Z)">↷</button>
      <button class="ghost icon"
        class:flash={xyzCopied}
        onclick={copyXyz}
        disabled={!store.currentXyz}
        title={xyzCopied ? 'Copied!' : 'Copy XYZ to clipboard'}
        aria-label="Copy XYZ to clipboard">{xyzCopied ? '✓' : '⎘'}</button>
      <button class="ghost icon"
        onclick={exportXyz}
        disabled={!store.currentXyz}
        title="Download as extended XYZ">⤓</button>
    </div>
  </div>
  <p class="hint">
    {store.atomicNumbers.length} atom{store.atomicNumbers.length === 1 ? '' : 's'}{store.isPeriodic ? ' · periodic' : ''}
  </p>
</div>

{#if hasSelection}
  <div class="panel-section">
    <div class="section-header">
      <span class="section-title">Selection</span>
      <button class="ghost" onclick={clearSelection}>clear</button>
    </div>

    {#if store.selectedBond && selected.length === 0}
      {@const d = distance(store.selectedBond[0], store.selectedBond[1])}
      <div class="readout">
        <div class="row"><span class="lbl">bond</span><span>{store.selectedBond[0]} ↔ {store.selectedBond[1]}</span></div>
        {#if d !== null}<div class="row"><span class="lbl">length</span><span>{fmtNum(d)} Å</span></div>{/if}
      </div>
    {:else if selected.length === 1}
      {@const i = selected[0]}
      {@const z = atomZ(i)}
      {@const p = atomPosition(i)}
      <div class="readout">
        <div class="row"><span class="lbl">atom</span><span>#{i} {z !== undefined ? getElementByNumber(z).symbol : '?'}</span></div>
        {#if p}
          <div class="row"><span class="lbl">x</span><span>{fmtNum(p[0])}</span></div>
          <div class="row"><span class="lbl">y</span><span>{fmtNum(p[1])}</span></div>
          <div class="row"><span class="lbl">z</span><span>{fmtNum(p[2])}</span></div>
        {/if}
      </div>
    {:else if selected.length === 2}
      {@const d = distance(selected[0], selected[1])}
      <div class="readout">
        <div class="row"><span class="lbl">atoms</span><span>#{selected[0]}, #{selected[1]}</span></div>
        {#if d !== null}<div class="row"><span class="lbl">distance</span><span>{fmtNum(d)} Å</span></div>{/if}
      </div>
    {:else if selected.length === 3}
      {@const a = angleDeg(selected[0], selected[1], selected[2])}
      <div class="readout">
        <div class="row"><span class="lbl">atoms</span><span>#{selected[0]}, #{selected[1]}, #{selected[2]}</span></div>
        {#if a !== null}<div class="row"><span class="lbl">angle</span><span>{fmtNum(a, 1)}°</span></div>{/if}
        <p class="hint">middle atom is the vertex</p>
      </div>
    {:else}
      <div class="readout">
        <div class="row"><span class="lbl">selected</span><span>{selected.length} atoms</span></div>
      </div>
    {/if}
  </div>
{/if}

<div class="panel-section">
  <div class="section-header">
    <span class="section-title">Edit</span>
    <span class="active-elem" title={`Active element — ${getElementByNumber(currentZ).name}`}
      style="--cpk: {rgbStr(getElementByNumber(currentZ).color)}; --ink: {inkFor(getElementByNumber(currentZ).color)};">
      <span class="dot"></span>
      <span class="sym">{getElementByNumber(currentZ).symbol}</span>
    </span>
  </div>

  <div class="elem-row">
    {#each COMMON as z}
      {@const el = getElementByNumber(z)}
      <button class="elem-pill"
        class:active={currentZ === z}
        onclick={() => (currentZ = z)}
        style="--cpk: {rgbStr(el.color)}; --ink: {inkFor(el.color)};"
        title={el.name}
      >{el.symbol}</button>
    {/each}
    <button class="elem-pill more"
      onclick={() => (ptableOpen = true)}
      title="Browse all elements">⋯</button>
  </div>

  <div class="btn-row">
    <button onclick={setElementOnSelection} disabled={runningGuard || !hasSelection}>
      Set to {getElementByNumber(currentZ).symbol}
    </button>
    <button onclick={addAtomNearSelection} disabled={runningGuard}>
      Add {getElementByNumber(currentZ).symbol}
    </button>
    <button
      onclick={fillHydrogens}
      disabled={runningGuard || store.atomicNumbers.length === 0}
      title={hasSelection ? 'Fill missing H’s on selected heavy atoms' : 'Fill missing H’s on every heavy atom'}
    >Add H{hasSelection ? ' (sel)' : ''}</button>
    <button onclick={deleteSelection} disabled={runningGuard || !hasSelection}>Delete</button>
  </div>
  <div class="btn-row">
    <button onclick={copySelection} disabled={runningGuard || !hasSelection} title="Copy selected (⌘/Ctrl+C)">Copy</button>
    <button onclick={pasteClipboard} disabled={runningGuard || !store.clipboard} title="Paste at selection (⌘/Ctrl+V)">Paste</button>
    <button onclick={duplicateSelection} disabled={runningGuard || !hasSelection} title="Duplicate (⌘/Ctrl+D)">Duplicate</button>
  </div>

  {#if runningGuard}
    <p class="hint warn">stop the simulation to edit.</p>
  {/if}
</div>

{#if bondPair && bondLength !== null}
  <div class="panel-section">
    <div class="section-header"><span class="section-title">Bond length</span></div>
    <div class="bond-row">
      <span class="bond-label">{bondPair[0]} ↔ {bondPair[1]}</span>
      <input
        type="number" step="0.01" min="0.1"
        bind:value={bondInput}
        onkeydown={(e) => e.key === 'Enter' && commitBondLength()}
        disabled={runningGuard}
        class="bond-input" />
      <span class="bond-unit">Å</span>
      <button onclick={commitBondLength} disabled={runningGuard}>Set</button>
    </div>
    <div class="side-row">
      <span class="lbl">drag side:</span>
      <label><input type="radio" bind:group={bondMoveSide} value="a" /> {bondPair[0]}</label>
      <label><input type="radio" bind:group={bondMoveSide} value="b" /> {bondPair[1]}</label>
    </div>
  </div>
{/if}

<div class="panel-section">
  <div class="section-header"><span class="section-title">Fragments</span></div>
  <div class="frag-grid">
    {#each FRAGMENT_ORDER as key}
      {@const f = FRAGMENTS[key]}
      <button
        class="frag"
        onclick={() => applyFragment(key)}
        disabled={runningGuard || selected.length === 0}
        title={`replace selected atom${selected.length === 1 ? '' : 's'} with ${f.name}`}
      >{f.symbol}</button>
    {/each}
  </div>
  {#if selected.length === 0}
    <p class="hint">select an atom (typically an H) first; the fragment replaces it, oriented away from its neighbour.</p>
  {/if}
</div>

<div class="panel-section">
  <div class="section-header"><span class="section-title">Optimize</span></div>
  {#if !modelReady}
    <p class="hint warn">load a model in the left panel to relax geometry.</p>
  {:else}
    <div class="opt-row">
      <span class="lbl">algo</span>
      {#each ['cg', 'lbfgs', 'fire'] as algo}
        <label class="algo-pill" class:disabled={store.isRunning}>
          <input type="radio" name="opt-algo" value={algo}
            bind:group={optAlgo}
            disabled={store.isRunning} />
          <span>{algo === 'cg' ? 'CG' : algo === 'lbfgs' ? 'L-BFGS' : 'FIRE'}</span>
        </label>
      {/each}
      <label class="ftol">
        <span class="lbl">F tol</span>
        <input type="number" min="0.001" max="1.0" step="0.01"
          bind:value={store.forceThreshold}
          disabled={store.isRunning} />
      </label>
    </div>
    <div class="btn-row">
      <button onclick={relaxAll} disabled={store.isRunning}>Relax all</button>
      <button
        onclick={relaxSelected}
        disabled={store.isRunning || selected.length === 0}
        title="Freeze every atom not in the current selection during the relax"
      >Relax sel ({selected.length})</button>
      <button class="stop" onclick={stopOpt} disabled={!store.isRunning}>Stop</button>
    </div>
    {#if store.isRunning}
      <p class="hint">
        {store.activeOptimizer ?? optAlgo} · step {store.step} · E {store.energy.toFixed(4)} eV · maxF {store.maxForce.toFixed(3)} eV/Å
      </p>
    {:else if store.step > 0}
      <p class="hint">
        {store.optimizationConverged ? 'converged' : 'stopped'} at {store.step} · E {store.energy.toFixed(4)} eV · maxF {store.maxForce.toFixed(3)} eV/Å
      </p>
    {/if}
  {/if}
</div>

<details class="panel-section panel-collapse">
  <summary>Saved selections</summary>
  <div class="slots">
    {#each SLOTS as slot}
      {@const saved = store.selectionGroups[slot]}
      {@const filled = saved && saved.length > 0}
      <button
        class="slot"
        class:filled
        title={filled
          ? `${saved.length} atoms — click to recall, shift-click to overwrite, alt-click to clear`
          : 'shift-click to save current selection'}
        onclick={(ev) => {
          if (ev.altKey) clearSlot(slot)
          else handleSlotClick(slot, ev)
        }}
      >
        <span class="slot-num">{slot}</span>
        {#if filled}<span class="slot-count">{saved.length}</span>{/if}
      </button>
    {/each}
  </div>
  <p class="hint">key 1–9 recalls · shift+digit saves current selection</p>
</details>

<details class="panel-section panel-collapse">
  <summary>Shortcuts</summary>
  <dl class="shortcuts">
    <dt>G</dt><dd>grab — translate selection</dd>
    <dt>R</dt><dd>rotate around centroid (or bond endpoint w/ B)</dd>
    <dt>X / Y / Z / B</dt><dd>during G/R, lock to axis</dd>
    <dt>Tab</dt><dd>cycle bond axis (after B)</dd>
    <dt>0–9 . −</dt><dd>during G/R, type a precise value</dd>
    <dt>Enter</dt><dd>commit · esc / right-click cancel</dd>
    <dt>A</dt><dd>select all</dd>
    <dt>S</dt><dd>box-select — drag a rectangle (shift add, ⌘/ctrl toggle)</dd>
    <dt>F</dt><dd>toggle bond between two selected atoms (or selected bond)</dd>
    <dt>1–9</dt><dd>recall saved selection · shift saves it</dd>
    <dt>Delete</dt><dd>remove selected atoms</dd>
    <dt>Space</dt><dd>start/stop relax (selection if any, else all)</dd>
    <dt>⌘/Ctrl+C/V/D</dt><dd>copy · paste · duplicate</dd>
    <dt>⌘/Ctrl+Z</dt><dd>undo · shift to redo</dd>
    <dt>Esc</dt><dd>cancel transform · stop a run</dd>
  </dl>
</details>

{#if ptableOpen}
  <div class="ptable-overlay" role="dialog" aria-modal="true"
    onclick={() => (ptableOpen = false)}
    onkeydown={(e) => e.key === 'Escape' && (ptableOpen = false)}
    tabindex="-1">
    <div class="ptable-dialog" onclick={(e) => e.stopPropagation()} role="presentation">
      <PeriodicTable bind:value={currentZ} onpick={() => (ptableOpen = false)} />
    </div>
  </div>
{/if}

<style>
  .panel-section {
    padding: 0.55rem 0.75rem;
    border-bottom: 1px solid var(--border);
  }
  .panel-collapse {
    padding: 0;
  }
  .panel-collapse > summary {
    list-style: none;
    cursor: pointer;
    padding: 0.5rem 0.75rem;
    font-size: 0.74rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-secondary);
    user-select: none;
    display: flex;
    align-items: center;
    gap: 0.4rem;
  }
  .panel-collapse > summary::-webkit-details-marker { display: none; }
  .panel-collapse > summary::before {
    content: '▸';
    font-size: 0.65rem;
    transition: transform 120ms ease;
    color: var(--text-secondary);
  }
  .panel-collapse[open] > summary::before { transform: rotate(90deg); }
  .panel-collapse > :not(summary) {
    padding: 0 0.75rem 0.6rem;
  }

  .section-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem;
    margin-bottom: 0.4rem;
  }
  .section-title {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-secondary);
  }
  .header-actions {
    display: flex;
    gap: 0.2rem;
    align-items: center;
  }
  .hint {
    font-size: 0.7rem;
    color: var(--text-secondary);
    margin: 0.35rem 0 0;
    line-height: 1.4;
  }
  .hint.warn { color: var(--text-primary); }
  .readout {
    display: flex;
    flex-direction: column;
    gap: 0.12rem;
    font-size: 0.78rem;
    font-variant-numeric: tabular-nums;
  }
  .row {
    display: flex;
    justify-content: space-between;
    gap: 0.5rem;
  }
  .lbl { color: var(--text-secondary); }

  button.ghost {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--text-secondary);
    border-radius: 4px;
    padding: 0.1rem 0.4rem;
    font-size: 0.7rem;
    cursor: pointer;
  }
  button.ghost.icon {
    padding: 0.05rem 0.35rem;
    font-size: 0.95rem;
    line-height: 1;
  }
  button.ghost.icon.flash {
    color: #4ade80;
    border-color: #4ade80;
  }
  button.ghost:disabled { opacity: 0.4; cursor: default; }
  button.ghost:not(:disabled):hover { color: var(--text-primary); }

  .active-elem {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    font-size: 0.74rem;
    color: var(--text-primary);
  }
  .active-elem .dot {
    width: 0.7rem;
    height: 0.7rem;
    background: var(--cpk);
    border-radius: 999px;
    border: 1px solid rgba(255, 255, 255, 0.18);
  }
  .active-elem .sym {
    font-weight: 700;
    font-family: ui-sans-serif, system-ui, sans-serif;
  }

  .elem-row {
    display: grid;
    grid-template-columns: repeat(9, 1fr);
    gap: 0.2rem;
    margin-bottom: 0.45rem;
  }
  .elem-pill {
    background: var(--cpk, var(--bg-secondary));
    color: var(--ink, var(--text-primary));
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 0.32rem 0;
    font-size: 0.78rem;
    font-weight: 700;
    line-height: 1;
    cursor: pointer;
    text-align: center;
    transition: transform 80ms ease, border-color 80ms ease;
  }
  .elem-pill.more {
    background: var(--bg-secondary);
    color: var(--text-secondary);
    font-weight: 500;
    font-size: 1rem;
    letter-spacing: 0.1em;
  }
  .elem-pill:hover {
    border-color: rgba(255, 255, 255, 0.65);
    transform: translateY(-1px);
  }
  .elem-pill.active {
    border-color: #ff9900;
    box-shadow: 0 0 0 2px rgba(255, 153, 0, 0.55);
  }

  .btn-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem;
    margin-top: 0.35rem;
  }
  .btn-row button {
    flex: 1 1 auto;
    padding: 0.3rem 0.5rem;
    font-size: 0.76rem;
    background: var(--bg-secondary);
    color: var(--text-primary);
    border: 1px solid var(--border);
    border-radius: 4px;
    cursor: pointer;
  }
  .btn-row button:disabled { opacity: 0.4; cursor: default; }
  .btn-row button:not(:disabled):hover { background: var(--bg-primary); }

  .shortcuts {
    margin: 0;
    display: grid;
    grid-template-columns: auto 1fr;
    column-gap: 0.6rem;
    row-gap: 0.18rem;
    font-size: 0.72rem;
  }
  .shortcuts dt {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-weight: 600;
    color: var(--text-primary);
    white-space: nowrap;
  }
  .shortcuts dd {
    margin: 0;
    color: var(--text-secondary);
  }

  .slots {
    display: grid;
    grid-template-columns: repeat(9, 1fr);
    gap: 0.2rem;
  }
  .slot {
    position: relative;
    aspect-ratio: 1;
    padding: 0;
    background: transparent;
    border: 1px dashed var(--border);
    border-radius: 4px;
    color: var(--text-secondary);
    font-size: 0.68rem;
    cursor: pointer;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    line-height: 1;
  }
  .slot:hover { border-color: var(--text-primary); color: var(--text-primary); }
  .slot.filled {
    border-style: solid;
    background: var(--bg-secondary);
    color: var(--text-primary);
  }
  .slot-num {
    font-weight: 700;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }
  .slot-count {
    font-size: 0.55rem;
    opacity: 0.7;
    margin-top: 0.05rem;
  }

  .bond-row {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    margin-bottom: 0.3rem;
  }
  .bond-label {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.74rem;
    color: var(--text-secondary);
  }
  .bond-input {
    width: 4.5rem;
    padding: 0.2rem 0.3rem;
    background: var(--bg-primary);
    color: var(--text-primary);
    border: 1px solid var(--border);
    border-radius: 4px;
    font-size: 0.78rem;
    font-variant-numeric: tabular-nums;
  }
  .bond-unit { font-size: 0.74rem; color: var(--text-secondary); }
  .bond-row button {
    margin-left: auto;
    padding: 0.2rem 0.6rem;
    font-size: 0.74rem;
    background: var(--bg-secondary);
    color: var(--text-primary);
    border: 1px solid var(--border);
    border-radius: 4px;
    cursor: pointer;
  }
  .bond-row button:disabled { opacity: 0.4; cursor: default; }
  .side-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-size: 0.74rem;
    color: var(--text-secondary);
  }
  .side-row label {
    display: inline-flex;
    align-items: center;
    gap: 0.2rem;
    color: var(--text-primary);
  }

  .frag-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.25rem;
  }
  .frag {
    padding: 0.3rem 0.2rem;
    background: var(--bg-secondary);
    color: var(--text-primary);
    border: 1px solid var(--border);
    border-radius: 4px;
    font-size: 0.74rem;
    font-weight: 600;
    cursor: pointer;
    line-height: 1.1;
  }
  .frag:disabled { opacity: 0.4; cursor: default; }
  .frag:not(:disabled):hover { background: var(--bg-primary); border-color: var(--text-primary); }

  .opt-row {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.4rem 0.5rem;
    margin-bottom: 0.4rem;
    font-size: 0.74rem;
  }
  .algo-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.2rem;
    color: var(--text-primary);
  }
  .algo-pill input { accent-color: var(--text-primary); }
  .algo-pill.disabled { opacity: 0.5; }
  .ftol {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    margin-left: auto;
    color: var(--text-primary);
  }
  .ftol input {
    width: 4rem;
    padding: 0.15rem 0.3rem;
    background: var(--bg-primary);
    color: var(--text-primary);
    border: 1px solid var(--border);
    border-radius: 4px;
    font-size: 0.74rem;
    font-variant-numeric: tabular-nums;
  }
  .btn-row .stop {
    background: transparent;
    color: var(--text-primary);
  }
  .btn-row .stop:not(:disabled):hover {
    border-color: #ff6b6b;
    color: #ff6b6b;
  }

  .ptable-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.55);
    backdrop-filter: blur(4px);
    z-index: 100;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .ptable-dialog {
    max-width: 90vw;
    max-height: 90vh;
  }
</style>
