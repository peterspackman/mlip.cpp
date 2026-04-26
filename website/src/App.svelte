<script lang="ts">
  import { onMount, onDestroy, setContext } from 'svelte'
  import { SimulationStore } from './lib/stores/simulation.svelte'
  import { parseLattice } from './lib/chem/cell'
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
  import EditPanel from './components/EditPanel.svelte'

  const store = new SimulationStore()
  setContext('store', store)

  // Window-level drag-and-drop. Drop a `.gguf` anywhere → load model.
  // Drop a `.xyz` / `.extxyz` anywhere → load structure. Local drop zones
  // (e.g. inside ModelLoader) call preventDefault() in their handler, so
  // we skip when defaultPrevented to avoid double-handling.
  let appDragActive = $state(false)

  function dragHasFiles(e: DragEvent): boolean {
    return Array.from(e.dataTransfer?.types ?? []).includes('Files')
  }

  function onWindowDragEnter(e: DragEvent) {
    if (!dragHasFiles(e)) return
    appDragActive = true
  }
  function onWindowDragLeave(e: DragEvent) {
    // relatedTarget === null only when the cursor leaves the window entirely.
    if (e.relatedTarget === null) appDragActive = false
  }
  function onWindowDragOver(e: DragEvent) {
    if (!dragHasFiles(e)) return
    e.preventDefault()
    if (e.dataTransfer) e.dataTransfer.dropEffect = 'copy'
  }
  async function onWindowDrop(e: DragEvent) {
    appDragActive = false
    // Inner drop targets (ModelLoader's gguf zone) preventDefault before bubbling.
    if (e.defaultPrevented) return
    const file = e.dataTransfer?.files?.[0]
    if (!file) return
    e.preventDefault()
    if (/\.gguf$/i.test(file.name)) {
      const buf = await file.arrayBuffer()
      await store.loadModel(buf, file.name)
    } else if (/\.(xyz|extxyz)$/i.test(file.name)) {
      const text = await file.text()
      const lattice = parseLattice(text)
      await store.loadStructure(text, lattice)
    } else {
      store.modelError = `Unsupported file type: ${file.name}. Drop a .gguf model or .xyz / .extxyz structure.`
    }
  }

  onMount(() => {
    store.initialize()
    window.addEventListener('dragenter', onWindowDragEnter)
    window.addEventListener('dragleave', onWindowDragLeave)
    window.addEventListener('dragover', onWindowDragOver)
    window.addEventListener('drop', onWindowDrop)
  })

  onDestroy(() => {
    window.removeEventListener('dragenter', onWindowDragEnter)
    window.removeEventListener('dragleave', onWindowDragLeave)
    window.removeEventListener('dragover', onWindowDragOver)
    window.removeEventListener('drop', onWindowDrop)
    store.dispose()
  })
</script>

<div class="app">
  <header class="header">
    <div class="container header-inner">
      <div>
        <h1>mlip.js</h1>
        <p class="subtitle">Machine Learning Interatomic Potentials in the Browser</p>
      </div>
      <nav class="tabs" aria-label="App mode">
        <button
          class="tab"
          class:active={store.appMode === 'sim'}
          onclick={() => (store.appMode = 'sim')}
        >Simulate</button>
        <button
          class="tab"
          class:active={store.appMode === 'edit'}
          onclick={() => (store.appMode = 'edit')}
        >Edit</button>
      </nav>
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
        {#if store.appMode === 'sim'}
          <RunControls />
          {#if store.mode === 'md'}
            <MDParams />
          {:else if store.mode === 'optimize'}
            <OptParams />
          {:else}
            <VibrationsPanel />
          {/if}
          <Stats />
        {:else}
          <EditPanel />
        {/if}
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

  {#if appDragActive}
    <div class="app-dropzone" aria-hidden="true">
      <div class="app-dropzone-card">
        <div class="app-dropzone-icon">⬇</div>
        <div class="app-dropzone-title">Drop to load</div>
        <div class="app-dropzone-sub">
          <code>.gguf</code> model · <code>.xyz</code> / <code>.extxyz</code> structure
        </div>
      </div>
    </div>
  {/if}
</div>
