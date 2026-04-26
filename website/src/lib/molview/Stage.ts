import * as THREE from "three";
import { TrackballControls } from "three/examples/jsm/controls/TrackballControls.js";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { OutlinePass } from "three/examples/jsm/postprocessing/OutlinePass.js";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass.js";
import { GTAOPass } from "three/examples/jsm/postprocessing/GTAOPass.js";
import { ShaderMaterial } from "three";
import { FullScreenQuad } from "three/examples/jsm/postprocessing/Pass.js";
import { DEFAULT_LIGHTING, type LightingSettings } from "./lighting";
import type { UnitCellData } from "./data/types";
import type { RenderPlugin } from "./plugins/PathTracerPlugin";

/** Layer used by impostor meshes — excluded from GTAO normal pass to avoid box artifacts */
export const IMPOSTOR_LAYER = 31;

export type ProjectionMode = "orthographic" | "perspective";
export type DebugPass = "none" | "depth" | "mask" | "edge1" | "edge2" | "ao";

export interface AOSettings {
  enabled: boolean;
  radius: number;
  distanceExponent: number;
  thickness: number;
  scale: number;
  samples: number;
}

export const DEFAULT_AO: AOSettings = {
  enabled: false,
  radius: 0.5,
  distanceExponent: 2,
  thickness: 2,
  scale: 1,
  samples: 8,
};

const TIMING_HISTORY_SIZE = 60;

export interface PassTiming {
  name: string;
  ms: number;
}

export interface FrameTiming {
  total: number;
  passes: PassTiming[];
}


export class ViewerStage {
  scene: THREE.Scene;
  camera: THREE.OrthographicCamera | THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  controls: TrackballControls;

  // Exposed lights for settings panel
  ambientLight: THREE.AmbientLight;
  keyLight: THREE.DirectionalLight;
  fillLight: THREE.DirectionalLight;
  hemisphereLight: THREE.HemisphereLight;

  private animationId: number | null = null;
  private resizeObserver: ResizeObserver;
  private frustumSize = 20;
  private needsRender = true;
  private _projectionMode: ProjectionMode = "orthographic";
  private _sceneBoundsDirty = true;
  private _sceneSphere = new THREE.Sphere();

  // Post-processing
  private composer: EffectComposer;
  private gtaoPass: GTAOPass;
  private outlinePass: OutlinePass;
  private _aoSettings: AOSettings = { ...DEFAULT_AO };
  private _debugPass: DebugPass = "none";
  private _debugQuad: FullScreenQuad;

  // Frame timing
  private _timingEnabled = false;
  private _timingHistory: FrameTiming[] = [];

  // Render plugin (e.g. path tracer) — when active, skips normal composer render
  private _activePlugin: RenderPlugin | null = null;

  /** Called after each render frame — use for overlay updates */
  onAfterRender: (() => void) | null = null;

  constructor(canvas: HTMLCanvasElement) {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x1a1a2e);

    const rect = canvas.parentElement?.getBoundingClientRect() ?? { width: 800, height: 600 };
    const aspect = rect.width / rect.height;

    this.camera = new THREE.OrthographicCamera(
      -this.frustumSize * aspect / 2,
       this.frustumSize * aspect / 2,
       this.frustumSize / 2,
      -this.frustumSize / 2,
      -200,
      2000,
    );
    this.camera.position.set(0, 0, 100);
    this.camera.layers.enable(IMPOSTOR_LAYER);

    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: false,
      preserveDrawingBuffer: true,
    });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setSize(rect.width, rect.height);
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.0;

    this.controls = this.createControls(this.camera, canvas);

    // --- Lighting ---
    this.ambientLight = new THREE.AmbientLight(0xffffff, DEFAULT_LIGHTING.ambientIntensity);
    this.scene.add(this.ambientLight);

    this.hemisphereLight = new THREE.HemisphereLight(
      DEFAULT_LIGHTING.hemisphereSkyColor,
      DEFAULT_LIGHTING.hemisphereGroundColor,
      DEFAULT_LIGHTING.hemisphereIntensity,
    );
    this.scene.add(this.hemisphereLight);

    this.keyLight = new THREE.DirectionalLight(0xffffff, DEFAULT_LIGHTING.keyLightIntensity);
    this.keyLight.position.set(1, 1, 1);
    this.camera.add(this.keyLight);

    this.fillLight = new THREE.DirectionalLight(0xffffff, DEFAULT_LIGHTING.fillLightIntensity);
    this.fillLight.position.set(-1, -0.5, 0.5);
    this.camera.add(this.fillLight);

    // Camera must be in the scene graph for child lights to render
    this.scene.add(this.camera);

    // --- Post-processing ---
    const pixelRatio = this.renderer.getPixelRatio();
    const rtWidth = Math.floor(rect.width * pixelRatio);
    const rtHeight = Math.floor(rect.height * pixelRatio);
    const renderTarget = new THREE.WebGLRenderTarget(rtWidth, rtHeight, {
      depthTexture: new THREE.DepthTexture(rtWidth, rtHeight),
    });
    const resolution = new THREE.Vector2(rect.width, rect.height);
    this.composer = new EffectComposer(this.renderer, renderTarget);
    this.composer.addPass(new RenderPass(this.scene, this.camera));

    this.outlinePass = new OutlinePass(resolution, this.scene, this.camera);
    this.outlinePass.edgeStrength = 8.0;
    this.outlinePass.edgeGlow = 0.3;
    this.outlinePass.edgeThickness = 8.0;
    this.outlinePass.visibleEdgeColor.set(0xff9900);
    this.outlinePass.hiddenEdgeColor.set(0x000000);
    this.outlinePass.pulsePeriod = 0;
    // Normal blending so outline is visible against any background (additive is invisible on white)
    (this.outlinePass as any).overlayMaterial.blending = THREE.NormalBlending;
    this.composer.addPass(this.outlinePass);

    // GTAO after outlines so selection edges don't get darkened
    this.gtaoPass = new GTAOPass(this.scene, this.camera, rtWidth, rtHeight);
    this.gtaoPass.output = GTAOPass.OUTPUT.Default;
    this.gtaoPass.updateGtaoMaterial({
      radius: this._aoSettings.radius,
      distanceExponent: this._aoSettings.distanceExponent,
      thickness: this._aoSettings.thickness,
      scale: this._aoSettings.scale,
      samples: this._aoSettings.samples,
    });
    this.gtaoPass.enabled = this._aoSettings.enabled;
    this.patchGTAOForOrtho();
    this.patchGTAOForImpostors();
    this.composer.addPass(this.gtaoPass);

    this.composer.addPass(new OutputPass());

    // Debug blit quad for visualizing intermediate render targets
    this._debugQuad = new FullScreenQuad(new ShaderMaterial({
      uniforms: { tDiffuse: { value: null } },
      vertexShader: `varying vec2 vUv; void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
      fragmentShader: `uniform sampler2D tDiffuse; varying vec2 vUv; void main() { gl_FragColor = texture2D(tDiffuse, vUv); }`,
    }));

    // Resize handling
    this.resizeObserver = new ResizeObserver(() => this.handleResize());
    if (canvas.parentElement) {
      this.resizeObserver.observe(canvas.parentElement);
    }

    this.startRenderLoop();
  }

  private createControls(
    camera: THREE.Camera,
    canvas: HTMLCanvasElement,
  ): TrackballControls {
    const controls = new TrackballControls(camera, canvas);
    controls.rotateSpeed = 2.0;
    controls.zoomSpeed = 1.2;
    controls.panSpeed = 0.8;
    controls.noRotate = false;
    controls.noPan = false;
    controls.noZoom = false;
    controls.staticMoving = true; // no inertia/damping
    // TrackballControls' default keys (A=rotate, S=zoom, D=pan) collide with
    // editor shortcuts (A = select all, S = box-select). The canvas-level
    // listener fires regardless of window-level preventDefault, so disable
    // the keyboard state machine entirely — we handle camera motion via
    // mouse only.
    controls.keys = ["", "", ""];
    controls.addEventListener("change", () => {
      this.needsRender = true;
    });
    return controls;
  }

  /**
   * Convert ortho camera.zoom changes into dolly (camera movement).
   * TrackballControls scales the frustum for ortho zoom, but we want
   * the camera to physically move closer/farther — this keeps view-space
   * distances correct for impostor depth writes and matches CE trackball.
   */
  private convertZoomToDolly(): void {
    if (!(this.camera instanceof THREE.OrthographicCamera)) return;
    const zoom = this.camera.zoom;
    if (Math.abs(zoom - 1.0) < 1e-6) return;

    // Move camera toward/away from target by the zoom factor
    const target = this.controls.target;
    const dir = new THREE.Vector3().subVectors(this.camera.position, target);
    const dist = dir.length();
    dir.divideScalar(dist); // normalize

    const newDist = dist / zoom;
    this.camera.position.copy(target).addScaledVector(dir, newDist);

    // Reset zoom and adjust frustum size to compensate
    this.camera.zoom = 1.0;
    this.frustumSize /= zoom;
    const parent = this.renderer.domElement.parentElement;
    const rect = parent?.getBoundingClientRect() ?? { width: 800, height: 600 };
    const aspect = rect.width / rect.height;
    this.camera.left   = -this.frustumSize * aspect / 2;
    this.camera.right  =  this.frustumSize * aspect / 2;
    this.camera.top    =  this.frustumSize / 2;
    this.camera.bottom = -this.frustumSize / 2;
    this.camera.updateProjectionMatrix();

    this._sceneBoundsDirty = true;
  }

  /** Signal that the scene has changed and needs re-rendering */
  requestRender(): void {
    this.needsRender = true;
  }

  /** Mark scene bounds dirty (call after modifying scene content in place) */
  invalidateBounds(): void {
    this._sceneBoundsDirty = true;
  }

  /** Recompute near/far from scene bounding sphere and camera distance.
   *  Called every frame so clip planes stay tight as you zoom/rotate. */
  private updateClipPlanes(): void {
    if (this._sceneBoundsDirty) {
      const box = new THREE.Box3().setFromObject(this.scene);
      if (!box.isEmpty()) {
        box.getBoundingSphere(this._sceneSphere);
      }
      this._sceneBoundsDirty = false;
    }

    const radius = this._sceneSphere.radius;
    if (radius < 0.01) return; // no content

    const cameraDist = this.camera.position.distanceTo(this._sceneSphere.center);
    const padding = radius * 1.5;

    if (this.camera instanceof THREE.OrthographicCamera) {
      // Ortho: allow negative near so nothing clips when scene extends
      // behind the camera. Depth is linear so precision is fine.
      this.camera.near = cameraDist - padding;
      this.camera.far = cameraDist + padding;
    } else {
      // Perspective: must have positive near
      this.camera.near = Math.max(0.1, cameraDist - padding);
      this.camera.far = cameraDist + padding;
    }

    this.camera.updateProjectionMatrix();
  }

  add(object: THREE.Object3D): void {
    this.scene.add(object);
    this._sceneBoundsDirty = true;
    this.needsRender = true;
  }

  remove(object: THREE.Object3D): void {
    this.scene.remove(object);
    this._sceneBoundsDirty = true;
    this.needsRender = true;
  }

  /** Set which intermediate pass to visualize (for debugging) */
  setDebugPass(pass: DebugPass): void {
    this._debugPass = pass;
    this.needsRender = true;
  }

  get debugPass(): DebugPass {
    return this._debugPass;
  }

  /** Update ambient occlusion settings */
  updateAO(settings: AOSettings): void {
    this._aoSettings = { ...settings };
    this.gtaoPass.enabled = settings.enabled;
    this.gtaoPass.updateGtaoMaterial({
      radius: settings.radius,
      distanceExponent: settings.distanceExponent,
      thickness: settings.thickness,
      scale: settings.scale,
      samples: settings.samples,
    });
    this.needsRender = true;
  }

  get aoSettings(): AOSettings {
    return this._aoSettings;
  }

  /**
   * Patch the GTAO shader to fix orthographic camera support.
   *
   * The stock Three.js GTAOShader uses `viewDir = normalize(-viewPos.xyz)` which
   * is correct for perspective but wrong for ortho — in ortho the view direction
   * is always constant (0,0,1) in view space regardless of screen position.
   * The perspective formula causes AO intensity to vary with distance from screen
   * center, producing artifacts where front-facing geometry appears overly darkened.
   */
  private patchGTAOForOrtho(): void {
    const mat = (this.gtaoPass as any).gtaoMaterial as ShaderMaterial;
    if (!mat?.fragmentShader) return;

    // Replace the viewDir computation with a camera-type branch
    mat.fragmentShader = mat.fragmentShader.replace(
      "vec3 viewDir = normalize(-viewPos.xyz);",
      `vec3 viewDir;
       #if PERSPECTIVE_CAMERA == 1
         viewDir = normalize(-viewPos.xyz);
       #else
         viewDir = vec3(0.0, 0.0, 1.0);
       #endif`,
    );
    mat.needsUpdate = true;
  }

  /**
   * Patch GTAO's _renderOverride to exclude impostor meshes (layer 31).
   * Impostors use ray-cast shaders; the GTAO normal pass would render
   * their proxy box/quad geometry, producing rectangular AO artifacts.
   */
  private patchGTAOForImpostors(): void {
    const pass = this.gtaoPass as any;
    const orig = pass._renderOverride.bind(pass);
    const camera = this.camera;
    pass._renderOverride = (
      renderer: THREE.WebGLRenderer,
      overrideMaterial: THREE.Material,
      renderTarget: THREE.WebGLRenderTarget,
      clearColor: number,
      clearAlpha: number,
    ) => {
      camera.layers.disable(IMPOSTOR_LAYER);
      orig(renderer, overrideMaterial, renderTarget, clearColor, clearAlpha);
      camera.layers.enable(IMPOSTOR_LAYER);
    };
  }

  /** Set the outline color for selection highlighting */
  setOutlineColor(color: number): void {
    this.outlinePass.visibleEdgeColor.set(color);
    this.outlinePass.hiddenEdgeColor.set(color);
    this.needsRender = true;
  }

  /** Set the objects that should have a selection outline */
  setOutlinedObjects(objects: THREE.Object3D[]): void {
    this.outlinePass.selectedObjects = objects;
    this.needsRender = true;
  }

  setBackground(color: number): void {
    (this.scene.background as THREE.Color).set(color);
    // Sync fog color if fog exists
    if (this.scene.fog instanceof THREE.FogExp2) {
      this.scene.fog.color.set(color);
    }
    this.requestRender();
  }

  setFog(enabled: boolean, density: number): void {
    if (enabled) {
      const bgColor = (this.scene.background as THREE.Color).clone();
      this.scene.fog = new THREE.FogExp2(bgColor.getHex(), density);
    } else {
      this.scene.fog = null;
    }
    this.requestRender();
  }

  lookDownCellAxis(axis: "a" | "b" | "c", unitCell: UnitCellData): void {
    const m = unitCell.matrix;
    // Extract cell vectors: a=[m[0],m[3],m[6]], b=[m[1],m[4],m[7]], c=[m[2],m[5],m[8]]
    const vectors = {
      a: new THREE.Vector3(m[0], m[3], m[6]),
      b: new THREE.Vector3(m[1], m[4], m[7]),
      c: new THREE.Vector3(m[2], m[5], m[8]),
    };
    // Cyclic up: a→b, b→c, c→a
    const upMap = { a: "b", b: "c", c: "a" } as const;

    const lookDir = vectors[axis].clone().normalize();
    const upDir = vectors[upMap[axis]].clone().normalize();
    const target = this.controls.target.clone();
    const currentDist = this.camera.position.distanceTo(target);

    this.camera.position.copy(target).addScaledVector(lookDir, currentDist);
    this.camera.up.copy(upDir);
    this.camera.lookAt(target);
    this.controls.update();
    this.requestRender();
  }

  updateLighting(settings: LightingSettings): void {
    this.ambientLight.intensity = settings.ambientIntensity;
    this.keyLight.intensity = settings.keyLightIntensity;
    this.fillLight.intensity = settings.fillLightIntensity;
    this.hemisphereLight.intensity = settings.hemisphereIntensity;
    this.hemisphereLight.color.set(settings.hemisphereSkyColor);
    this.hemisphereLight.groundColor.set(settings.hemisphereGroundColor);
    this.needsRender = true;
  }

  fitToContent(): void {
    const box = new THREE.Box3().setFromObject(this.scene);
    if (box.isEmpty()) return;

    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);

    this.frustumSize = maxDim * 1.4;

    const parent = this.renderer.domElement.parentElement;
    const rect = parent?.getBoundingClientRect() ?? { width: 800, height: 600 };
    const aspect = rect.width / rect.height;

    if (this.camera instanceof THREE.OrthographicCamera) {
      this.camera.left   = -this.frustumSize * aspect / 2;
      this.camera.right  =  this.frustumSize * aspect / 2;
      this.camera.top    =  this.frustumSize / 2;
      this.camera.bottom = -this.frustumSize / 2;
    } else {
      (this.camera as THREE.PerspectiveCamera).aspect = aspect;
    }
    const cameraDist = maxDim * 2;

    // Preserve current view direction; only reset distance and target
    const lookDir = new THREE.Vector3()
      .subVectors(this.camera.position, this.controls.target)
      .normalize();
    // If direction is degenerate (e.g. first load), fall back to +Z
    if (lookDir.lengthSq() < 0.5) lookDir.set(0, 0, 1);

    this.controls.target.copy(center);
    this.camera.position.copy(center).addScaledVector(lookDir, cameraDist);
    this._sceneBoundsDirty = true;
    this.camera.updateProjectionMatrix();
    this.controls.update();
    this.needsRender = true;
  }

  /**
   * Rotate the camera around the controls target by `angleRadians` about the
   * scene's vertical axis (controls.up). Used by the "Animate Scene" loop.
   */
  rotateCameraAroundTarget(angleRadians: number): void {
    const offset = new THREE.Vector3().subVectors(this.camera.position, this.controls.target);
    const axis = this.controls.object.up.clone().normalize();
    offset.applyAxisAngle(axis, angleRadians);
    this.camera.position.copy(this.controls.target).add(offset);
    this.camera.lookAt(this.controls.target);
    this.controls.update();
    this.needsRender = true;
  }

  /**
   * Move the rotation pivot back to the world origin (0,0,0) without
   * changing zoom or view direction. Mirrors CE's right-click "Reset Origin".
   */
  resetOrigin(): void {
    const offset = new THREE.Vector3().subVectors(this.controls.target, new THREE.Vector3(0, 0, 0));
    this.controls.target.set(0, 0, 0);
    this.camera.position.sub(offset);
    this._sceneBoundsDirty = true;
    this.controls.update();
    this.needsRender = true;
  }

  /**
   * Update the rotation pivot to the centroid of the given atoms
   * without changing camera zoom or orientation. CE-style: always
   * rotate around the center of the current atomic structure.
   */
  setRotationTarget(center: THREE.Vector3): void {
    // Shift camera by the same delta so the view doesn't jump
    const delta = center.clone().sub(this.controls.target);
    this.controls.target.copy(center);
    this.camera.position.add(delta);
    this.controls.update();
    this.needsRender = true;
  }

  /** Set or clear a section clipping plane.
   *  Pass null to disable clipping. */
  setClippingPlane(plane: THREE.Plane | null): void {
    if (plane) {
      this.renderer.clippingPlanes = [plane];
    } else {
      this.renderer.clippingPlanes = [];
    }
    this.renderer.localClippingEnabled = plane !== null;
    this.requestRender();
  }

  /** Look down an arbitrary Miller direction [h,k,l] using unit cell vectors */
  lookDownMillerDirection(h: number, k: number, l: number, unitCell: UnitCellData): void {
    const m = unitCell.matrix;
    const a = new THREE.Vector3(m[0], m[3], m[6]);
    const b = new THREE.Vector3(m[1], m[4], m[7]);
    const c = new THREE.Vector3(m[2], m[5], m[8]);

    // Miller direction in Cartesian: h*a + k*b + l*c
    const dir = a.multiplyScalar(h).add(b.multiplyScalar(k)).add(c.multiplyScalar(l));
    if (dir.lengthSq() < 1e-10) return;
    dir.normalize();

    const target = this.controls.target.clone();
    const currentDist = this.camera.position.distanceTo(target);

    this.camera.position.copy(target).addScaledVector(dir, currentDist);
    this.camera.up.set(0, 1, 0);
    this.camera.lookAt(target);
    this.controls.update();
    this.requestRender();
  }

  /** Capture the current viewport as a PNG data URL */
  captureScreenshot(resolutionMultiplier?: number, transparent?: boolean): string {
    const multiplier = resolutionMultiplier ?? 1;
    if (multiplier === 1 && !transparent) {
      this.composer.render();
      return this.renderer.domElement.toDataURL("image/png");
    }

    // High-res / transparent capture: resize, render, restore
    const canvas = this.renderer.domElement;
    const origWidth = canvas.width;
    const origHeight = canvas.height;
    const newWidth = Math.round(origWidth * multiplier);
    const newHeight = Math.round(origHeight * multiplier);

    const origAlpha = this.renderer.getClearAlpha();
    const origColor = this.renderer.getClearColor(new THREE.Color());

    if (transparent) {
      this.renderer.setClearColor(0x000000, 0);
    }

    this.renderer.setSize(newWidth, newHeight, false);
    this.composer.setSize(newWidth, newHeight);
    this.composer.render();

    const dataUrl = canvas.toDataURL("image/png");

    // Restore original size
    this.renderer.setSize(origWidth, origHeight, false);
    this.composer.setSize(origWidth, origHeight);
    this.renderer.setClearColor(origColor, origAlpha);
    this.requestRender();

    return dataUrl;
  }

  get projectionMode(): ProjectionMode {
    return this._projectionMode;
  }

  setProjection(mode: ProjectionMode): void {
    if (mode === this._projectionMode) return;

    const canvas = this.renderer.domElement;
    const parent = canvas.parentElement;
    const rect = parent?.getBoundingClientRect() ?? { width: 800, height: 600 };
    const aspect = rect.width / rect.height;

    const oldPos = this.camera.position.clone();
    const oldTarget = this.controls.target.clone();
    const oldUp = this.camera.up.clone();

    // Remove old camera from scene (lights are children)
    this.scene.remove(this.camera);

    let newCamera: THREE.OrthographicCamera | THREE.PerspectiveCamera;

    if (mode === "perspective") {
      newCamera = new THREE.PerspectiveCamera(45, aspect, 0.1, 2000);
    } else {
      newCamera = new THREE.OrthographicCamera(
        -this.frustumSize * aspect / 2,
         this.frustumSize * aspect / 2,
         this.frustumSize / 2,
        -this.frustumSize / 2,
        -200,
        2000,
      );
    }

    newCamera.position.copy(oldPos);
    newCamera.up.copy(oldUp);
    newCamera.layers.enable(IMPOSTOR_LAYER);
    newCamera.updateProjectionMatrix();

    // Re-parent lights
    newCamera.add(this.keyLight);
    newCamera.add(this.fillLight);

    this.camera = newCamera;
    this.scene.add(this.camera);

    // Re-attach controls
    this.controls.dispose();
    this.controls = this.createControls(this.camera, canvas);
    this.controls.target.copy(oldTarget);
    this.controls.update();

    // Rebuild composer with new camera
    this.rebuildComposer(rect.width, rect.height);

    this._projectionMode = mode;
    this.needsRender = true;
  }

  /** Rebuild the post-processing pipeline (after camera swap or resize) */
  private rebuildComposer(width: number, height: number): void {
    const selectedObjects = this.outlinePass.selectedObjects;

    // Recreate composer with depth texture
    this.composer.dispose();
    const pixelRatio = this.renderer.getPixelRatio();
    const rtW = Math.floor(width * pixelRatio);
    const rtH = Math.floor(height * pixelRatio);
    const renderTarget = new THREE.WebGLRenderTarget(rtW, rtH, {
      depthTexture: new THREE.DepthTexture(rtW, rtH),
    });
    this.composer = new EffectComposer(this.renderer, renderTarget);

    const resolution = new THREE.Vector2(width, height);
    this.composer.addPass(new RenderPass(this.scene, this.camera));

    this.outlinePass = new OutlinePass(resolution, this.scene, this.camera);
    this.outlinePass.edgeStrength = 8.0;
    this.outlinePass.edgeGlow = 0.3;
    this.outlinePass.edgeThickness = 8.0;
    this.outlinePass.visibleEdgeColor.set(0xff9900);
    this.outlinePass.hiddenEdgeColor.set(0x000000);
    this.outlinePass.pulsePeriod = 0;
    (this.outlinePass as any).overlayMaterial.blending = THREE.NormalBlending;
    this.outlinePass.selectedObjects = selectedObjects;
    this.composer.addPass(this.outlinePass);

    // GTAO after outlines so selection edges don't get darkened
    this.gtaoPass = new GTAOPass(this.scene, this.camera, rtW, rtH);
    this.gtaoPass.output = GTAOPass.OUTPUT.Default;
    this.gtaoPass.updateGtaoMaterial({
      radius: this._aoSettings.radius,
      distanceExponent: this._aoSettings.distanceExponent,
      thickness: this._aoSettings.thickness,
      scale: this._aoSettings.scale,
      samples: this._aoSettings.samples,
    });
    this.gtaoPass.enabled = this._aoSettings.enabled;
    this.gtaoPass.enabled = this._aoSettings.enabled;
    this.patchGTAOForOrtho();
    this.patchGTAOForImpostors();
    this.composer.addPass(this.gtaoPass);

    this.composer.addPass(new OutputPass());
    this.composer.setSize(width, height);
  }

  private handleResize(): void {
    const parent = this.renderer.domElement.parentElement;
    if (!parent) return;

    const rect = parent.getBoundingClientRect();
    const aspect = rect.width / rect.height;

    if (this.camera instanceof THREE.OrthographicCamera) {
      this.camera.left   = -this.frustumSize * aspect / 2;
      this.camera.right  =  this.frustumSize * aspect / 2;
      this.camera.top    =  this.frustumSize / 2;
      this.camera.bottom = -this.frustumSize / 2;
    } else {
      (this.camera as THREE.PerspectiveCamera).aspect = aspect;
    }
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(rect.width, rect.height);
    this.composer.setSize(rect.width, rect.height);
    this.controls.handleResize();
    this.needsRender = true;
  }

  /** Enable/disable per-pass frame timing collection */
  setTimingEnabled(enabled: boolean): void {
    this._timingEnabled = enabled;
    if (!enabled) {
      this._timingHistory = [];
    }
  }

  get timingEnabled(): boolean {
    return this._timingEnabled;
  }

  /** Get the timing history (most recent TIMING_HISTORY_SIZE frames) */
  get timingHistory(): readonly FrameTiming[] {
    return this._timingHistory;
  }

  /** Get averaged timing over the history window */
  get averageTiming(): FrameTiming | null {
    if (this._timingHistory.length === 0) return null;
    const n = this._timingHistory.length;
    const totalAvg = this._timingHistory.reduce((s, f) => s + f.total, 0) / n;

    // Average per-pass: collect all pass names from most recent frame as template
    const latest = this._timingHistory[n - 1];
    const passAvgs: PassTiming[] = latest.passes.map((p, idx) => {
      const sum = this._timingHistory.reduce((s, f) => s + (f.passes[idx]?.ms ?? 0), 0);
      return { name: p.name, ms: sum / n };
    });

    return { total: totalAvg, passes: passAvgs };
  }

  private startRenderLoop(): void {
    const animate = () => {
      this.animationId = requestAnimationFrame(animate);

      // TrackballControls needs update() called each frame
      this.controls.update();
      this.convertZoomToDolly();

      if (this.needsRender) {
        this.updateClipPlanes();

        // When a plugin is active, it owns the render loop — skip composer
        if (!this._activePlugin) {
          if (this._timingEnabled) {
            this.renderWithTiming();
          } else {
            this.composer.render();
          }

          // Debug: blit an intermediate render target to screen
          if (this._debugPass !== "none") {
            const target = this._getDebugRenderTarget();
            if (target) {
              const mat = this._debugQuad.material as ShaderMaterial;
              mat.uniforms.tDiffuse.value = target.texture;
              this.renderer.setRenderTarget(null);
              this._debugQuad.render(this.renderer);
            }
          }
        }

        this.needsRender = false;
        this.onAfterRender?.();
      }
    };
    animate();
  }

  /** Render each composer pass individually, timing each one */
  private renderWithTiming(): void {
    const frameStart = performance.now();
    const passes: PassTiming[] = [];

    const renderer = this.renderer;
    const composer = this.composer;

    // Access internal passes array
    const passList = (composer as any).passes as Array<{ enabled: boolean; renderToScreen: boolean; render(renderer: THREE.WebGLRenderer, writeBuffer: THREE.WebGLRenderTarget, readBuffer: THREE.WebGLRenderTarget, deltaTime: number, maskActive: boolean): void }>;
    const writeBuffer = (composer as any).writeBuffer as THREE.WebGLRenderTarget;
    const readBuffer = (composer as any).readBuffer as THREE.WebGLRenderTarget;

    // Update composer timer for deltaTime
    const composerTimer = (composer as any).timer;
    composerTimer?.update();
    const deltaTime = composerTimer?.getDelta() ?? 0;

    let maskActive = false;

    for (const pass of passList) {
      if (!pass.enabled) continue;

      const passName = pass.constructor.name.replace(/Pass$/, "");
      const t0 = performance.now();

      pass.render(renderer, writeBuffer, readBuffer, deltaTime, maskActive);

      // Force GPU sync for accurate timing (only in timing mode)
      const gl = renderer.getContext();
      gl.finish();

      const t1 = performance.now();
      passes.push({ name: passName, ms: t1 - t0 });

      // Swap buffers (replicate EffectComposer logic)
      if ((pass as any).needsSwap) {
        if (maskActive) {
          const context = renderer.getContext();
          const stencil = renderer.state.buffers.stencil;
          stencil.setFunc(context.NOTEQUAL, 1, 0xffffffff);
          // copy pass would render here
          stencil.setFunc(context.EQUAL, 1, 0xffffffff);
        }
        const tmp = (composer as any).readBuffer;
        (composer as any).readBuffer = (composer as any).writeBuffer;
        (composer as any).writeBuffer = tmp;
      }
    }

    const frameEnd = performance.now();
    const timing: FrameTiming = { total: frameEnd - frameStart, passes };

    this._timingHistory.push(timing);
    if (this._timingHistory.length > TIMING_HISTORY_SIZE) {
      this._timingHistory.shift();
    }
  }

  private _getDebugRenderTarget(): THREE.WebGLRenderTarget | null {
    const op = this.outlinePass as any;
    switch (this._debugPass) {
      case "depth": return op.renderTargetDepthBuffer ?? null;
      case "mask": return op.renderTargetMaskBuffer ?? null;
      case "edge1": return op.renderTargetEdgeBuffer1 ?? null;
      case "edge2": return op.renderTargetEdgeBuffer2 ?? null;
      case "ao": return (this.gtaoPass as any).pdRenderTarget ?? null;
      default: return null;
    }
  }

  /** Activate a render plugin — pauses normal render loop */
  async activatePlugin(plugin: RenderPlugin): Promise<void> {
    if (this._activePlugin) {
      this._activePlugin.deactivate();
    }
    this._activePlugin = plugin;
    await plugin.activate({
      scene: this.scene,
      camera: this.camera,
      renderer: this.renderer,
      canvas: this.renderer.domElement,
      controlsTarget: this.controls.target.clone(),
    });
  }

  /** Deactivate the current render plugin and resume normal rendering */
  deactivatePlugin(): void {
    if (this._activePlugin) {
      this._activePlugin.deactivate();
      this._activePlugin = null;
      this.needsRender = true;
    }
  }

  get activePlugin(): RenderPlugin | null {
    return this._activePlugin;
  }

  dispose(): void {
    if (this._activePlugin) {
      this._activePlugin.dispose();
      this._activePlugin = null;
    }
    if (this.animationId !== null) {
      cancelAnimationFrame(this.animationId);
    }
    this.resizeObserver.disconnect();
    this.controls.dispose();
    this.composer.dispose();
    this._debugQuad.dispose();
    (this._debugQuad.material as ShaderMaterial).dispose();
    this.renderer.dispose();
  }
}
