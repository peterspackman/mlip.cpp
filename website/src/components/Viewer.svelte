<script lang="ts">
  import { getContext, onMount, onDestroy } from 'svelte'
  import { Viewer as NGLViewer } from '../lib/ngl/viewer'
  import type { SimulationStore } from '../lib/stores/simulation.svelte'

  const store = getContext<SimulationStore>('store')

  let container: HTMLDivElement
  let viewer: NGLViewer | null = null

  onMount(() => {
    viewer = new NGLViewer()
    viewer.mount(container)
  })

  onDestroy(() => {
    viewer?.dispose()
    viewer = null
  })

  $effect(() => { viewer?.setStyle(store.viewStyle) })
  $effect(() => { viewer?.setWrap(store.wrapPositions) })
  $effect(() => { viewer?.setSupercell(store.supercell) })

  let lastStructureKey = ''
  $effect(() => {
    if (!viewer) return
    const key = `${store.atomicNumbers.join(',')}|${JSON.stringify(store.lattice)}`
    if (key === lastStructureKey) return
    if (store.atomicNumbers.length === 0 || !store.positions) return
    lastStructureKey = key
    viewer.setStructure(store.positions, store.atomicNumbers, store.lattice)
  })

  $effect(() => {
    if (!viewer) return
    if (!store.positions) return
    if (store.atomicNumbers.length === 0) return
    // Cheap update path: as long as the structure (atoms + lattice) is the
    // same as what the viewer already holds, just move the atoms. Works for
    // MD, optimize, and vib animation alike — the structure key is stable
    // across all three.
    const key = `${store.atomicNumbers.join(',')}|${JSON.stringify(store.lattice)}`
    if (key === lastStructureKey) {
      viewer.updatePositions(store.positions)
    }
  })
</script>

<div class="viewer" bind:this={container}></div>

<style>
  .viewer {
    flex: 1;
    min-height: 0;
    width: 100%;
    background: var(--bg-primary);
    position: relative;
    overflow: hidden;
  }
</style>
