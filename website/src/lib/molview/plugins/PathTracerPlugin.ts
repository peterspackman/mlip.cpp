// Stub: only exports the RenderPlugin interface so Stage.ts can type its
// _activePlugin slot. The actual path tracer implementation (which pulls
// in three-gpu-pathtracer) is not vendored — _activePlugin always stays
// null in this build.
import type * as THREE from "three";

export interface RenderPluginContext {
  scene: THREE.Scene;
  camera: THREE.OrthographicCamera | THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  canvas: HTMLCanvasElement;
  controlsTarget: THREE.Vector3;
}

export interface RenderPlugin {
  readonly name: string;
  readonly active: boolean;
  activate(ctx: RenderPluginContext): Promise<void>;
  deactivate(): void;
  dispose(): void;
}
