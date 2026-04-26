<script lang="ts">
  import { getContext } from 'svelte'
  import type { SimulationStore } from '../lib/stores/simulation.svelte'
  import { SAMPLE_MOLECULES, SAMPLE_CRYSTALS, SAMPLE_STRUCTURES } from '../lib/data/samples'
  import { parseLattice } from '../lib/chem/cell'
  import { fetchFromPubChem } from '../lib/chem/pubchem'
  import XyzEditorModal from './XyzEditorModal.svelte'

  const store = getContext<SimulationStore>('store')

  let selectedSample = $state('Ethanol')
  let customXyz = $state(SAMPLE_STRUCTURES['Ethanol'])
  let pubchemQuery = $state('')
  let pubchemLoading = $state(false)
  let pubchemError = $state('')
  let editorOpen = $state(false)

  // Structure load only needs the WASM worker, not a model — editing geometry
  // doesn't require energies. Run / optimize controls keep their own model gate.
  const ready = $derived(store.wasmReady)
  const xyzSummary = $derived(summarise(customXyz))

  function summarise(xyz: string): string {
    const lines = xyz.trim().split('\n')
    const n = parseInt(lines[0])
    if (!Number.isFinite(n)) return '(invalid XYZ)'
    const title = (lines[1] || '').trim() || `${n} atoms`
    return `${n} atoms · ${title.slice(0, 40)}${title.length > 40 ? '…' : ''}`
  }

  async function loadStructure() {
    if (!customXyz.trim()) return
    const lattice = parseLattice(customXyz)
    await store.loadStructure(customXyz, lattice)
  }

  function pickSample(name: string) {
    selectedSample = name
    if (SAMPLE_STRUCTURES[name]) {
      customXyz = SAMPLE_STRUCTURES[name]
      loadStructure()
    }
  }

  function applyEdit(xyz: string) {
    customXyz = xyz
    selectedSample = ''
    editorOpen = false
    loadStructure()
  }

  async function loadPubChem() {
    if (!pubchemQuery.trim()) return
    pubchemLoading = true
    pubchemError = ''
    try {
      const xyz = await fetchFromPubChem(pubchemQuery.trim())
      customXyz = xyz
      selectedSample = ''
      await loadStructure()
    } catch (err: any) {
      pubchemError = err?.message ?? String(err)
    } finally {
      pubchemLoading = false
    }
  }
</script>

<section class="panel-section structure-section">
  <h3>Structure</h3>

  <select
    value={selectedSample}
    onchange={(e) => pickSample((e.target as HTMLSelectElement).value)}
    disabled={!ready}
  >
    <option value="">(custom)</option>
    <optgroup label="Molecules">
      {#each Object.keys(SAMPLE_MOLECULES) as name (name)}
        <option value={name}>{name}</option>
      {/each}
    </optgroup>
    <optgroup label="Crystals">
      {#each Object.keys(SAMPLE_CRYSTALS) as name (name)}
        <option value={name}>{name}</option>
      {/each}
    </optgroup>
  </select>

  <div class="xyz-summary">
    <code>{xyzSummary}</code>
    <button class="edit-link" onclick={() => (editorOpen = true)} disabled={!ready}>Edit…</button>
  </div>

  <button class="load-button" onclick={loadStructure} disabled={!ready || !customXyz.trim()}>
    Load Structure
  </button>

  <div class="pubchem">
    <input
      type="text"
      bind:value={pubchemQuery}
      onkeydown={(e) => e.key === 'Enter' && loadPubChem()}
      placeholder="Search PubChem (e.g. aspirin)"
      disabled={!ready || pubchemLoading}
    />
    <button class="load-button" onclick={loadPubChem} disabled={!ready || !pubchemQuery.trim() || pubchemLoading}>
      {pubchemLoading ? '…' : 'Fetch'}
    </button>
    {#if pubchemError}
      <p class="error">{pubchemError}</p>
    {/if}
  </div>
</section>

<XyzEditorModal
  open={editorOpen}
  initialValue={customXyz}
  onclose={() => (editorOpen = false)}
  onapply={applyEdit}
/>

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
  select, input {
    padding: 0.4rem;
    border: 1px solid var(--border);
    border-radius: 4px;
    background-color: var(--bg-primary);
    color: var(--text-primary);
    font-size: 0.85rem;
    font-family: inherit;
  }
  .xyz-summary {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.4rem;
    padding: 0.4rem 0.5rem;
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: 4px;
  }
  .xyz-summary code {
    font-family: 'SF Mono', 'Menlo', monospace;
    font-size: 0.72rem;
    color: var(--text-secondary);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .edit-link {
    padding: 0;
    background: transparent;
    border: none;
    color: var(--accent);
    font-size: 0.75rem;
    cursor: pointer;
    text-decoration: underline;
    flex-shrink: 0;
  }
  .edit-link:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  .load-button {
    padding: 0.5rem;
    background-color: var(--accent);
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.8rem;
  }
  .load-button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  .load-button:hover:not(:disabled) {
    background-color: var(--accent-hover);
  }
  .pubchem {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }
  .error {
    font-size: 0.75rem;
    color: var(--error);
  }
</style>
