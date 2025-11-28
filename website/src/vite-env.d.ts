/// <reference types="vite/client" />

declare module '*.css' {
  const content: string
  export default content
}

declare module 'ngl' {
  export class Stage {
    constructor(element: HTMLElement, params?: Record<string, any>)
    loadFile(file: string | Blob | File, params?: Record<string, any>): Promise<any>
    removeAllComponents(): void
    autoView(duration?: number): void
    handleResize(): void
    setParameters(params: Record<string, any>): void
    dispose(): void
  }
  export class Shape {
    constructor(name: string)
    addWideline(start: [number, number, number], end: [number, number, number], color: [number, number, number]): void
  }
  export function autoLoad(url: string, params?: Record<string, any>): Promise<any>
}
