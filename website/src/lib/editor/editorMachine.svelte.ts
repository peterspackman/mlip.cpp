// Editor mode state machine (Vim-style).
//
// The structure editor has several *modal* interactions that must be mutually
// exclusive — select/transform/box-select/draw/command. Tracking them as
// independent booleans (transform.active, boxArmed, …) is a reliable source of
// transition bugs (enter draw mid-transform, arm box while a command line is
// open, …). This machine makes the top-level state explicit: you can only enter
// a modal state from `normal`, and every exit funnels through commit()/escape()/
// reset() with the matching side effect.
//
// Side effects (begin a transform, arm the marquee, run a command, …) are
// injected as hooks so the machine itself has no Three.js / DOM dependency and
// can be unit-tested in isolation.

export type EditorMode = 'normal' | 'insert' | 'command' | 'transform' | 'box'
export type TransformKind = 'grab' | 'rotate'

export interface EditorMachineHooks {
  onEnterInsert?: () => void
  onExitInsert?: () => void
  onEnterCommand?: () => void
  onRunCommand?: (buffer: string) => void
  onExitCommand?: () => void
  /** Start the transform. Return false to decline (e.g. nothing selected) so
   *  the machine stays in `normal` instead of stranding in `transform`. */
  onEnterTransform?: (kind: TransformKind) => boolean | void
  onSwitchTransform?: (kind: TransformKind) => void
  onCommitTransform?: () => void
  onCancelTransform?: () => void
  onEnterBox?: () => void
  onCommitBox?: () => void
  onCancelBox?: () => void
  /** Runs after every mode change (not intra-state ops like switchTransform).
   *  The single place to wipe per-gesture scratch so no state leaks across
   *  modes. */
  onAnyTransition?: () => void
}

export class EditorMachine {
  /** Current top-level interaction state. */
  mode = $state<EditorMode>('normal')
  /** Buffer for the command line while `mode === 'command'`. */
  commandBuffer = $state('')

  private hooks: EditorMachineHooks

  constructor(hooks: EditorMachineHooks = {}) {
    this.hooks = hooks
  }

  /** True in any non-select state. */
  get isModal(): boolean {
    return this.mode !== 'normal'
  }

  // --- entering modal states (only from normal) ---------------------------

  enterInsert(): boolean {
    if (this.mode !== 'normal') return false
    this.settle('insert')
    this.hooks.onEnterInsert?.()
    return true
  }

  enterCommand(): boolean {
    if (this.mode !== 'normal') return false
    this.commandBuffer = ''
    this.settle('command')
    this.hooks.onEnterCommand?.()
    return true
  }

  beginTransform(kind: TransformKind): boolean {
    if (this.mode !== 'normal') return false
    // Only commit to the transform state if the controller actually started —
    // otherwise (nothing selected, mid-run) we'd strand the machine here.
    if (this.hooks.onEnterTransform?.(kind) === false) return false
    this.settle('transform')
    return true
  }

  armBox(): boolean {
    if (this.mode !== 'normal') return false
    this.settle('box')
    this.hooks.onEnterBox?.()
    return true
  }

  // --- intra-state ops (no mode change, no scratch reset) -----------------

  /** Switch grab <-> rotate without leaving the transform state. */
  switchTransform(kind: TransformKind): boolean {
    if (this.mode !== 'transform') return false
    this.hooks.onSwitchTransform?.(kind)
    return true
  }

  // --- leaving modal states ----------------------------------------------

  /** Confirm the current modal action (Enter / pointer-commit). */
  commit(): boolean {
    switch (this.mode) {
      case 'transform':
        this.hooks.onCommitTransform?.()
        break
      case 'box':
        this.hooks.onCommitBox?.()
        break
      case 'command':
        this.hooks.onRunCommand?.(this.commandBuffer)
        this.hooks.onExitCommand?.()
        this.commandBuffer = ''
        break
      default:
        return false
    }
    this.settle('normal')
    return true
  }

  /** Cancel the current modal action (Esc / right-click). */
  escape(): boolean {
    switch (this.mode) {
      case 'transform':
        this.hooks.onCancelTransform?.()
        break
      case 'box':
        this.hooks.onCancelBox?.()
        break
      case 'insert':
        this.hooks.onExitInsert?.()
        break
      case 'command':
        this.hooks.onExitCommand?.()
        this.commandBuffer = ''
        break
      default:
        return false
    }
    this.settle('normal')
    return true
  }

  /** Hard reset to normal, running the cancel side effect for whatever state
   *  we're in. Used when the editor is torn down / leaves edit mode. */
  reset(): void {
    switch (this.mode) {
      case 'transform': this.hooks.onCancelTransform?.(); break
      case 'box':       this.hooks.onCancelBox?.();       break
      case 'insert':    this.hooks.onExitInsert?.();      break
      case 'command':   this.hooks.onExitCommand?.();     break
    }
    this.commandBuffer = ''
    this.settle('normal')
  }

  /** Commit a mode change: set the mode, then wipe per-gesture scratch. The
   *  single choke point for state changes — nothing else assigns `mode`. */
  private settle(mode: EditorMode): void {
    this.mode = mode
    this.hooks.onAnyTransition?.()
  }
}
