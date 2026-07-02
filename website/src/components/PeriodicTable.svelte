<script lang="ts">
  import { getElementByNumber } from '../lib/molview/data/elements'

  let { value = $bindable(), onpick }: {
    value: number
    onpick?: (z: number) => void
  } = $props()

  // Standard periodic-table layout: each entry says where Z lives in the
  // (row, col) grid. Lanthanides / actinides drop to rows 8 / 9 in the
  // canonical "below the main table" position.
  type Cell = { z: number; row: number; col: number }
  const layout: Cell[] = (() => {
    const out: Cell[] = []
    out.push({ z: 1, row: 1, col: 1 })
    out.push({ z: 2, row: 1, col: 18 })
    out.push({ z: 3, row: 2, col: 1 }, { z: 4, row: 2, col: 2 })
    for (let z = 5; z <= 10; z++) out.push({ z, row: 2, col: z + 8 })
    out.push({ z: 11, row: 3, col: 1 }, { z: 12, row: 3, col: 2 })
    for (let z = 13; z <= 18; z++) out.push({ z, row: 3, col: z })
    for (let z = 19; z <= 36; z++) out.push({ z, row: 4, col: z - 18 })
    for (let z = 37; z <= 54; z++) out.push({ z, row: 5, col: z - 36 })
    out.push({ z: 55, row: 6, col: 1 }, { z: 56, row: 6, col: 2 })
    for (let z = 57; z <= 71; z++) out.push({ z, row: 8, col: z - 54 })
    for (let z = 72; z <= 86; z++) out.push({ z, row: 6, col: z - 68 })
    out.push({ z: 87, row: 7, col: 1 }, { z: 88, row: 7, col: 2 })
    for (let z = 89; z <= 103; z++) out.push({ z, row: 9, col: z - 86 })
    // Period-7 d/p block: Rf(104)…Cn(112) then Nh(113)…Og(118).
    for (let z = 104; z <= 118; z++) out.push({ z, row: 7, col: z - 100 })
    return out
  })()

  function rgbStr(c: [number, number, number]): string {
    return `rgb(${Math.round(c[0] * 255)}, ${Math.round(c[1] * 255)}, ${Math.round(c[2] * 255)})`
  }
  function pickContrastInk(c: [number, number, number]): string {
    // ITU-R BT.601 luma, gives readable text on the CPK background.
    const y = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
    return y > 0.55 ? '#1a1a1a' : '#fff'
  }

  function pick(z: number) {
    value = z
    onpick?.(z)
  }

  // --- Search: filter by symbol prefix or name substring. ------------------
  let query = $state('')

  function matchesQuery(z: number): boolean {
    const q = query.trim().toLowerCase()
    if (!q) return true
    const el = getElementByNumber(z)
    return el.symbol.toLowerCase().startsWith(q) || el.name.toLowerCase().includes(q)
  }

  /** Best match for the current query: exact symbol, then symbol prefix, then
   *  name substring — each scanned in ascending-Z order (layout is Z-sorted). */
  function resolveQuery(): number | null {
    const q = query.trim().toLowerCase()
    if (!q) return null
    for (const c of layout) if (getElementByNumber(c.z).symbol.toLowerCase() === q) return c.z
    for (const c of layout) if (getElementByNumber(c.z).symbol.toLowerCase().startsWith(q)) return c.z
    for (const c of layout) if (getElementByNumber(c.z).name.toLowerCase().includes(q)) return c.z
    return null
  }

  let resolved = $derived(resolveQuery())

  function onSearchKey(e: KeyboardEvent) {
    if (e.key === 'Enter' && resolved != null) {
      e.preventDefault()
      pick(resolved)
    }
  }

  function autofocus(node: HTMLInputElement) {
    node.focus()
  }
</script>

<div class="ptable-wrap">
  <input
    class="ptable-search"
    type="text"
    placeholder="Search element — symbol or name (e.g. Fe, iron)"
    bind:value={query}
    onkeydown={onSearchKey}
    spellcheck="false"
    autocomplete="off"
    aria-label="Search elements"
    use:autofocus
  />
  <div class="ptable" role="grid" aria-label="Periodic table">
    {#each layout as cell (cell.z)}
      {@const el = getElementByNumber(cell.z)}
      <button
        class="cell"
        class:active={value === cell.z}
        class:dim={!matchesQuery(cell.z)}
        class:hit={query.trim() !== '' && resolved === cell.z}
        style="grid-row: {cell.row}; grid-column: {cell.col};
               --cpk: {rgbStr(el.color)}; --ink: {pickContrastInk(el.color)};"
        title={`${el.name} (Z=${cell.z})`}
        onclick={() => pick(cell.z)}
      >
        <span class="z">{cell.z}</span>
        <span class="sym">{el.symbol}</span>
      </button>
    {/each}
  </div>
</div>

<style>
  .ptable-wrap {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
    width: clamp(420px, 60vw, 720px);
  }
  .ptable-search {
    padding: 0.5rem 0.7rem;
    border: 1px solid rgba(255, 255, 255, 0.14);
    border-radius: 8px;
    background: rgba(15, 17, 24, 0.96);
    color: #fff;
    font-size: 0.85rem;
    font-family: ui-sans-serif, system-ui, sans-serif;
    outline: none;
  }
  .ptable-search:focus {
    border-color: #ff9900;
    box-shadow: 0 0 0 2px rgba(255, 153, 0, 0.35);
  }
  .ptable-search::placeholder { color: rgba(255, 255, 255, 0.4); }
  .ptable {
    display: grid;
    grid-template-columns: repeat(18, 1fr);
    grid-auto-rows: 1fr;
    gap: 2px;
    padding: 0.4rem;
    background: rgba(15, 17, 24, 0.96);
    border-radius: 8px;
    box-shadow: 0 8px 28px rgba(0, 0, 0, 0.55), 0 0 0 1px rgba(255, 255, 255, 0.06);
    /* Force square cells via aspect-ratio, so the table doesn't squish. */
    aspect-ratio: 18 / 9.5;
    width: 100%;
  }
  .cell {
    background: var(--cpk);
    color: var(--ink);
    border: 1px solid transparent;
    border-radius: 3px;
    padding: 0;
    font-family: ui-sans-serif, system-ui, sans-serif;
    line-height: 1;
    cursor: pointer;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 1px;
    min-width: 0;
    transition: transform 80ms ease, border-color 80ms ease;
  }
  .cell:hover {
    border-color: rgba(255, 255, 255, 0.65);
    transform: translateY(-1px);
  }
  .cell.active {
    border-color: #ff9900;
    box-shadow: 0 0 0 2px rgba(255, 153, 0, 0.7);
    z-index: 2;
  }
  /* Search: fade non-matches, ring the element Enter would select. */
  .cell.dim {
    opacity: 0.18;
    filter: saturate(0.4);
  }
  .cell.hit {
    border-color: #4ade80;
    box-shadow: 0 0 0 2px rgba(74, 222, 128, 0.8);
    z-index: 3;
  }
  .z {
    font-size: 0.55rem;
    opacity: 0.75;
    font-variant-numeric: tabular-nums;
  }
  .sym {
    font-size: 0.78rem;
    font-weight: 700;
  }
</style>
