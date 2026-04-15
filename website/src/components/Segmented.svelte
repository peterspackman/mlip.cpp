<script lang="ts" generics="T extends string">
  interface Option {
    value: T
    label: string
  }
  interface Props {
    value: T
    options: Option[]
    onchange?: (v: T) => void
  }
  let { value = $bindable(), options, onchange }: Props = $props()

  function pick(v: T) {
    value = v
    onchange?.(v)
  }
</script>

<div class="segmented" role="radiogroup">
  {#each options as opt (opt.value)}
    <button
      type="button"
      class="segment"
      class:active={value === opt.value}
      role="radio"
      aria-checked={value === opt.value}
      onclick={() => pick(opt.value)}
    >
      {opt.label}
    </button>
  {/each}
</div>

<style>
  .segmented {
    display: inline-flex;
    width: 100%;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: var(--bg-primary);
    padding: 2px;
    gap: 2px;
  }
  .segment {
    flex: 1;
    padding: 0.3rem 0.4rem;
    border: none;
    background: transparent;
    color: var(--text-secondary);
    font-size: 0.75rem;
    border-radius: 4px;
    cursor: pointer;
    white-space: nowrap;
    transition: background-color 0.1s, color 0.1s;
  }
  .segment:hover:not(.active) {
    background: color-mix(in srgb, var(--bg-secondary) 60%, transparent);
  }
  .segment.active {
    background: var(--accent);
    color: white;
  }
</style>
