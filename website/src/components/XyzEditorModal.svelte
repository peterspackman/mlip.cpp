<script lang="ts">
  interface Props {
    open: boolean
    initialValue: string
    onclose: () => void
    onapply: (xyz: string) => void
  }
  let { open, initialValue, onclose, onapply }: Props = $props()

  let draft = $state('')

  $effect(() => {
    if (open) draft = initialValue
  })

  function onKey(e: KeyboardEvent) {
    if (e.key === 'Escape') onclose()
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      onapply(draft)
    }
  }
</script>

{#if open}
  <div class="backdrop" onclick={onclose} onkeydown={onKey} role="presentation">
    <div
      class="modal"
      onclick={(e) => e.stopPropagation()}
      onkeydown={(e) => e.stopPropagation()}
      role="dialog"
      aria-modal="true"
      tabindex="-1"
    >
      <header>
        <h3>Edit XYZ</h3>
        <button class="close" onclick={onclose} aria-label="Close">×</button>
      </header>
      <textarea bind:value={draft} onkeydown={onKey} spellcheck="false"></textarea>
      <footer>
        <span class="hint">⌘↵ to apply · Esc to cancel</span>
        <div class="actions">
          <button onclick={onclose}>Cancel</button>
          <button class="primary" onclick={() => onapply(draft)}>Apply</button>
        </div>
      </footer>
    </div>
  </div>
{/if}

<style>
  .backdrop {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.5);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 100;
    padding: 2rem;
  }
  .modal {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 8px;
    width: min(680px, 100%);
    max-height: 100%;
    display: flex;
    flex-direction: column;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
  }
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid var(--border);
  }
  h3 {
    margin: 0;
    font-size: 0.85rem;
    font-weight: 600;
  }
  .close {
    background: transparent;
    border: none;
    font-size: 1.4rem;
    line-height: 1;
    color: var(--text-secondary);
    cursor: pointer;
    padding: 0 0.3rem;
  }
  .close:hover {
    color: var(--text-primary);
  }
  textarea {
    flex: 1;
    min-height: 400px;
    padding: 0.75rem;
    border: none;
    background: var(--bg-primary);
    color: var(--text-primary);
    font-family: 'SF Mono', 'Menlo', monospace;
    font-size: 0.8rem;
    resize: none;
    outline: none;
  }
  footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.5rem 0.75rem;
    border-top: 1px solid var(--border);
    gap: 0.5rem;
  }
  .hint {
    font-size: 0.7rem;
    color: var(--text-secondary);
  }
  .actions {
    display: flex;
    gap: 0.4rem;
  }
  footer button {
    padding: 0.4rem 0.8rem;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: var(--bg-primary);
    color: var(--text-primary);
    font-size: 0.8rem;
    cursor: pointer;
  }
  footer button.primary {
    background: var(--accent);
    color: white;
    border-color: var(--accent);
  }
  footer button.primary:hover {
    background: var(--accent-hover);
  }
</style>
