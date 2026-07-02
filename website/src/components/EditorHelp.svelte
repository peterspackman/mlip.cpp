<script lang="ts">
  // Mode-grouped keyboard/mouse cheat-sheet for the structure editor. Toggled
  // with ? or h; dismissed with Esc, ?, h, or a backdrop click.
  interface Props {
    open: boolean
    onclose: () => void
  }
  let { open, onclose }: Props = $props()

  type Row = { keys: string[]; desc: string }
  type Group = { title: string; rows: Row[] }

  const groups: Group[] = [
    {
      title: 'Global',
      rows: [
        { keys: ['i'], desc: 'Draw / build mode' },
        { keys: ['Esc'], desc: 'Back to Select' },
        { keys: [':'], desc: 'Command line' },
        { keys: ['?', 'h'], desc: 'This help' },
        { keys: ['u'], desc: 'Undo' },
        { keys: ['U'], desc: 'Redo' },
        { keys: ['Space'], desc: 'Relax (selection, else all)' },
      ],
    },
    {
      title: 'Select',
      rows: [
        { keys: ['click'], desc: 'Select atom · shift add · ⌘ toggle' },
        { keys: ['click bond'], desc: 'Select both atoms + bond' },
        { keys: ['g'], desc: 'Grab (move)' },
        { keys: ['r'], desc: 'Rotate' },
        { keys: ['x', 'y', 'z', 'b'], desc: 'Lock axis during g / r' },
        { keys: ['Tab'], desc: 'Cycle bond axis' },
        { keys: ['0-9', '.', '−'], desc: 'Type exact value during g / r' },
        { keys: ['a'], desc: 'Select all' },
        { keys: ['s'], desc: 'Box select' },
        { keys: ['f'], desc: 'Toggle bond (2 atoms)' },
        { keys: ['Del'], desc: 'Delete selection' },
        { keys: ['1-9'], desc: 'Recall selection · shift saves' },
        { keys: ['y'], desc: 'Copy (yank) selection' },
        { keys: ['p'], desc: 'Paste at cursor' },
        { keys: ['⌘C', '⌘V', '⌘D'], desc: 'Copy · paste · duplicate' },
      ],
    },
    {
      title: 'Draw',
      rows: [
        { keys: ['click'], desc: 'Add atom at cursor' },
        { keys: ['click atom'], desc: 'Set atom to active element' },
        { keys: ['drag atom → space'], desc: 'Add bonded atom' },
        { keys: ['drag atom → atom'], desc: 'Toggle bond' },
        { keys: ['right-click'], desc: 'Delete atom' },
      ],
    },
    {
      title: 'Command  :',
      rows: [
        { keys: ['e Fe'], desc: 'Set active element' },
        { keys: ['add O'], desc: 'Add an atom' },
        { keys: ['fill h'], desc: 'Add hydrogens' },
        { keys: ['del'], desc: 'Delete selection' },
        { keys: ['relax'], desc: 'Optimize geometry' },
      ],
    },
  ]
</script>

{#if open}
  <div
    class="backdrop"
    role="button"
    tabindex="-1"
    aria-label="Close help"
    onclick={onclose}
    onkeydown={(e) => e.key === 'Escape' && onclose()}
  >
    <div
      class="modal"
      role="dialog"
      aria-modal="true"
      aria-label="Editor shortcuts"
      tabindex="-1"
      onclick={(e) => e.stopPropagation()}
      onkeydown={(e) => e.stopPropagation()}
    >
      <header>
        <h3>Shortcuts</h3>
        <button class="close" onclick={onclose} aria-label="Close">×</button>
      </header>
      <div class="groups">
        {#each groups as g (g.title)}
          <section>
            <h4 class="section-title">{g.title}</h4>
            <dl>
              {#each g.rows as row (row.desc)}
                <dt>
                  {#each row.keys as k, i (k)}
                    {#if i > 0}<span class="sep">/</span>{/if}
                    <kbd>{k}</kbd>
                  {/each}
                </dt>
                <dd>{row.desc}</dd>
              {/each}
            </dl>
          </section>
        {/each}
      </div>
      <footer><kbd>?</kbd> or <kbd>Esc</kbd> to close</footer>
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
    z-index: 120;
    padding: 2rem;
  }
  .modal {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 8px;
    width: min(720px, 100%);
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
  h3 { margin: 0; font-size: 0.85rem; font-weight: 600; }
  .close {
    background: transparent;
    border: none;
    font-size: 1.4rem;
    line-height: 1;
    color: var(--text-secondary);
    cursor: pointer;
    padding: 0 0.3rem;
  }
  .close:hover { color: var(--text-primary); }
  .groups {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.9rem 1.5rem;
    padding: 0.9rem;
    overflow-y: auto;
  }
  section { break-inside: avoid; }
  .section-title {
    margin: 0 0 0.4rem;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-secondary);
    padding-bottom: 0.3rem;
    border-bottom: 1px solid var(--border);
  }
  dl {
    margin: 0;
    display: grid;
    grid-template-columns: auto 1fr;
    column-gap: 0.6rem;
    row-gap: 0.28rem;
    align-items: baseline;
  }
  dt { text-align: right; white-space: nowrap; }
  dd { margin: 0; font-size: 0.78rem; color: var(--text-secondary); }
  .sep { color: var(--text-secondary); margin: 0 0.15rem; }
  kbd {
    display: inline-block;
    padding: 0.05rem 0.35rem;
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: 4px;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.72rem;
    color: var(--text-primary);
  }
  footer {
    padding: 0.5rem 0.75rem;
    border-top: 1px solid var(--border);
    font-size: 0.74rem;
    color: var(--text-secondary);
  }
</style>
