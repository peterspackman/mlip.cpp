<script lang="ts">
  import { onMount, onDestroy, setContext } from 'svelte'
  import { SimulationStore } from './lib/stores/simulation.svelte'
  import ModelLoader from './components/ModelLoader.svelte'
  import StructureLoader from './components/StructureLoader.svelte'
  import Viewer from './components/Viewer.svelte'
  import ViewerControls from './components/ViewerControls.svelte'
  import RunControls from './components/RunControls.svelte'
  import MDParams from './components/MDParams.svelte'
  import OptParams from './components/OptParams.svelte'
  import Stats from './components/Stats.svelte'
  import EnergyPlot from './components/EnergyPlot.svelte'
  import VibrationsPanel from './components/VibrationsPanel.svelte'

  const store = new SimulationStore()
  setContext('store', store)

  onMount(() => {
    store.initialize()
  })

  onDestroy(() => {
    store.dispose()
  })
</script>

<div class="app">
  <header class="header">
    <div class="container">
      <h1>mlip.js</h1>
      <p class="subtitle">Machine Learning Interatomic Potentials in the Browser</p>
    </div>
  </header>

  <main class="main">
    <div class="container md-layout">
      <aside class="panel panel-left">
        <ModelLoader />
        <StructureLoader />
      </aside>

      <section class="center">
        <div class="viewer-column">
          <ViewerControls />
          <div class="viewer-frame">
            <Viewer />
          </div>
          <EnergyPlot />
        </div>
      </section>

      <aside class="panel panel-right">
        <RunControls />
        {#if store.mode === 'md'}
          <MDParams />
        {:else if store.mode === 'optimize'}
          <OptParams />
        {:else}
          <VibrationsPanel />
        {/if}
        <Stats />
      </aside>
    </div>
  </main>

  <footer class="footer">
    <div class="container">
      <p>
        Powered by <a href="https://github.com/peterspackman/mlip.cpp">mlip.cpp</a>
      </p>
    </div>
  </footer>
</div>
