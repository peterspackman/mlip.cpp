<script lang="ts">
  import { getContext, onMount } from 'svelte'
  import type { SimulationStore } from '../lib/stores/simulation.svelte'

  const store = getContext<SimulationStore>('store')

  let canvas: HTMLCanvasElement
  let resizeTick = $state(0)

  onMount(() => {
    const ro = new ResizeObserver(() => { resizeTick++ })
    if (canvas) ro.observe(canvas)
    return () => ro.disconnect()
  })

  $effect(() => {
    resizeTick  // track resize
    const history = store.energyHistory
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const dpr = window.devicePixelRatio || 1
    const w = canvas.clientWidth
    const h = canvas.clientHeight
    if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
      canvas.width = w * dpr
      canvas.height = h * dpr
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, w, h)
    if (history.length < 2) return

    let min = Infinity, max = -Infinity
    for (const v of history) {
      if (v < min) min = v
      if (v > max) max = v
    }
    if (max - min < 1e-6) { max = min + 1e-6 }

    const accent = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() || '#3b82f6'
    ctx.strokeStyle = accent
    ctx.lineWidth = 1.5
    ctx.beginPath()
    const n = history.length
    for (let i = 0; i < n; i++) {
      const x = (i / (n - 1)) * w
      const y = h - ((history[i] - min) / (max - min)) * (h - 4) - 2
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    }
    ctx.stroke()
  })
</script>

<section class="plot">
  <canvas bind:this={canvas}></canvas>
  <span class="label">
    {store.mode === 'md' ? 'Total energy' : 'Potential energy'}
  </span>
</section>

<style>
  .plot {
    position: relative;
    background: var(--bg-primary);
    border-top: 1px solid var(--border);
    width: 100%;
    height: 88px;
    min-height: 88px;
    max-height: 88px;
    flex-shrink: 0;
    flex-grow: 0;
    overflow: hidden;
  }
  canvas {
    position: absolute;
    inset: 0;
    display: block;
    width: 100%;
    height: 100%;
  }
  .label {
    position: absolute;
    top: 0.35rem;
    left: 0.6rem;
    font-size: 0.7rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
    pointer-events: none;
  }
</style>
