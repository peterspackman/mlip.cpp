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
</script>

<div class="ptable" role="grid" aria-label="Periodic table">
  {#each layout as cell (cell.z)}
    {@const el = getElementByNumber(cell.z)}
    <button
      class="cell"
      class:active={value === cell.z}
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

<style>
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
    width: clamp(420px, 60vw, 720px);
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
