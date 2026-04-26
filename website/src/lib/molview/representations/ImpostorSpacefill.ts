import * as THREE from "three";
import type { Representation, BuildOptions } from "./types";
import type { StructureData } from "../data/types";
import type { MaterialSettings } from "../lighting";
import { getElementColor } from "../data/colors";
import { getElementByNumber } from "../data/elements";

const PROXY_MATERIAL = new THREE.MeshBasicMaterial({
  transparent: true,
  opacity: 0,
  depthWrite: false,
});

// Billboard margin factor — quad extends this * radius from center.
// Must be > 1.0 to avoid clipping the sphere at the edges.
const BILLBOARD_MARGIN = 1.5;

// --- Shaders ---

const IMPOSTOR_SPHERE_VERT = /* glsl */ `
attribute float aGhost;
attribute float aRadius;

varying vec3 vViewCenter;
varying vec3 vViewPos;
varying float vRadius;
varying float vGhost;
varying vec3 vColor;
// projectionMatrix is only available in vertex shader — pass Z/W coefficients
varying vec4 vProjZW;

void main() {
  #ifdef USE_INSTANCING_COLOR
    vColor = instanceColor;
  #else
    vColor = vec3(0.8);
  #endif
  vGhost = aGhost;
  vRadius = aRadius;

  // Pass projection depth coefficients to fragment shader
  // Packed so fragment can do: clipZ = viewZ * .x + .z, clipW = viewZ * .y + .w
  vProjZW = vec4(
    projectionMatrix[2][2], projectionMatrix[2][3],
    projectionMatrix[3][2], projectionMatrix[3][3]
  );

  // Instance center in world space
  vec4 worldCenter = modelMatrix * instanceMatrix * vec4(0.0, 0.0, 0.0, 1.0);

  // View-space center for fragment ray tracing
  vViewCenter = (viewMatrix * worldCenter).xyz;

  // Billboard: camera right/up from viewMatrix columns
  vec3 camRight = vec3(viewMatrix[0][0], viewMatrix[1][0], viewMatrix[2][0]);
  vec3 camUp    = vec3(viewMatrix[0][1], viewMatrix[1][1], viewMatrix[2][1]);
  float extent = aRadius * ${BILLBOARD_MARGIN.toFixed(1)};
  vec3 worldPos = worldCenter.xyz
    + camRight * position.x * extent
    + camUp    * position.y * extent;

  // View-space position for fragment ray origin
  vec4 viewPos = viewMatrix * vec4(worldPos, 1.0);
  vViewPos = viewPos.xyz;

  gl_Position = projectionMatrix * viewPos;
}
`;

const IMPOSTOR_SPHERE_FRAG = /* glsl */ `
precision highp float;

varying vec3 vViewCenter;
varying vec3 vViewPos;
varying float vRadius;
varying float vGhost;
varying vec3 vColor;
varying vec4 vProjZW;

float calcDepth(float viewZ) {
  // clipZ = viewZ * M[2][2] + M[2][3]
  // clipW = viewZ * M[3][2] + M[3][3]
  float clipZ = viewZ * vProjZW.x + vProjZW.z;
  float clipW = viewZ * vProjZW.y + vProjZW.w;
  return 0.5 + 0.5 * clipZ / clipW;
}

vec3 dither(vec3 color) {
  vec3 d = vec3(dot(vec2(171.0, 231.0), gl_FragCoord.xy));
  d = fract(d / vec3(103.0, 71.0, 97.0)) - 0.5;
  return color + d / 255.0;
}

void main() {
  // Ray in view space
  vec3 rayOrigin;
  vec3 rayDir;
  if (isOrthographic) {
    rayOrigin = vec3(vViewPos.xy, 0.0);
    rayDir = vec3(0.0, 0.0, -1.0);
  } else {
    rayOrigin = vec3(0.0);
    rayDir = normalize(vViewPos);
  }

  // Ray-sphere intersection (rayDir is unit length so a=1)
  vec3 oc = rayOrigin - vViewCenter;
  float b = dot(oc, rayDir);
  float c = dot(oc, oc) - vRadius * vRadius;
  float disc = b * b - c;
  if (disc < 0.0) discard;

  float t = -b - sqrt(disc);
  vec3 hitPos = rayOrigin + t * rayDir;
  vec3 normal = normalize(hitPos - vViewCenter);

  // Blinn-Phong
  vec3 viewDir = normalize(-hitPos);
  vec3 lightDir = normalize(vec3(0.5, 1.0, 0.8));
  float NdotL = max(dot(normal, lightDir), 0.0);
  vec3 halfDir = normalize(lightDir + viewDir);
  float spec = pow(max(dot(normal, halfDir), 0.0), 64.0);

  vec3 color = vColor * (0.35 + 0.55 * NdotL) + vec3(1.0) * spec * 0.3;

  if (vGhost > 0.5) {
    float lum = dot(color, vec3(0.299, 0.587, 0.114));
    vec3 faded = mix(color, vec3(lum), 0.7);
    color = mix(faded, vec3(0.75), 0.5);
  }

  gl_FragColor = vec4(dither(color), 1.0);
  gl_FragDepth = calcDepth(hitPos.z);
}
`;

export class ImpostorSpacefillRepresentation implements Representation {
  readonly type = "spacefill" as const;
  readonly group = new THREE.Group();

  private atomMesh: THREE.InstancedMesh | null = null;
  private material: THREE.ShaderMaterial;
  private _visibleToOriginal: number[] = [];
  private _centers: Float32Array = new Float32Array(0);
  private _radii: Float32Array = new Float32Array(0);
  private ghostAttr: THREE.InstancedBufferAttribute | null = null;

  constructor() {
    this.material = new THREE.ShaderMaterial({
      vertexShader: IMPOSTOR_SPHERE_VERT,
      fragmentShader: IMPOSTOR_SPHERE_FRAG,
      side: THREE.DoubleSide,
      depthWrite: true,
      depthTest: true,
    });
  }

  build(data: StructureData, options?: BuildOptions): void {
    this.dispose();
    if (data.atoms.length === 0) return;

    const co = options?.colorOverride;
    const colorFn = options?.atomColorFn;
    const hidden = options?.hiddenElements;
    const hiddenIdx = options?.hiddenIndices;

    const visibleAtomIndices: number[] = [];
    for (let i = 0; i < data.atoms.length; i++) {
      if (hidden && hidden.has(data.atoms[i].atomicNumber)) continue;
      if (hiddenIdx && hiddenIdx.has(i)) continue;
      visibleAtomIndices.push(i);
    }
    this._visibleToOriginal = visibleAtomIndices;

    if (visibleAtomIndices.length === 0) return;

    const count = visibleAtomIndices.length;

    // Quad geometry (PlaneGeometry 2x2)
    const quadGeo = new THREE.PlaneGeometry(2, 2);
    this.atomMesh = new THREE.InstancedMesh(quadGeo, this.material, count);

    const matrix = new THREE.Matrix4();
    const color = new THREE.Color();
    const radii = new Float32Array(count);
    this._centers = new Float32Array(count * 3);
    this._radii = new Float32Array(count);

    for (let vi = 0; vi < count; vi++) {
      const i = visibleAtomIndices[vi];
      const atom = data.atoms[i];
      const radius = getElementByNumber(atom.atomicNumber).vdwRadius;

      radii[vi] = radius;
      this._radii[vi] = radius;
      this._centers[vi * 3] = atom.position.x;
      this._centers[vi * 3 + 1] = atom.position.y;
      this._centers[vi * 3 + 2] = atom.position.z;

      // Instance matrix: just position (scale handled in shader via aRadius)
      matrix.makeTranslation(atom.position.x, atom.position.y, atom.position.z);
      this.atomMesh.setMatrixAt(vi, matrix);

      if (co) {
        color.setRGB(co[0], co[1], co[2], THREE.SRGBColorSpace);
      } else if (colorFn) {
        const [r, g, b] = colorFn(i);
        color.setRGB(r, g, b, THREE.SRGBColorSpace);
      } else {
        const [r, g, b] = getElementColor(atom.atomicNumber);
        color.setRGB(r, g, b, THREE.SRGBColorSpace);
      }
      this.atomMesh.setColorAt(vi, color);
    }

    // Per-instance radius attribute
    const radiusAttr = new THREE.InstancedBufferAttribute(radii, 1);
    this.atomMesh.geometry.setAttribute("aRadius", radiusAttr);

    // Per-instance ghost attribute
    const ghostArray = new Float32Array(count);
    if (data.ghostStart !== undefined && data.ghostStart >= 0) {
      for (let vi = 0; vi < count; vi++) {
        if (visibleAtomIndices[vi] >= data.ghostStart) {
          ghostArray[vi] = 1.0;
        }
      }
    }
    this.ghostAttr = new THREE.InstancedBufferAttribute(ghostArray, 1);
    this.atomMesh.geometry.setAttribute("aGhost", this.ghostAttr);

    this.atomMesh.instanceMatrix.needsUpdate = true;
    if (this.atomMesh.instanceColor) this.atomMesh.instanceColor.needsUpdate = true;

    // Billboard quads extend beyond geometry bounds via shader — disable frustum culling
    this.atomMesh.frustumCulled = false;
    // Layer 31: visible in main render but excluded from GTAO normal pass
    this.atomMesh.layers.set(31);

    // Custom raycasting using analytical ray-sphere intersection
    this.atomMesh.raycast = this._raycast.bind(this);

    this.group.add(this.atomMesh);
  }

  getAtomIndex(object: THREE.Object3D, instanceId: number): number {
    if (object === this.atomMesh) {
      return this._visibleToOriginal[instanceId] ?? -1;
    }
    return -1;
  }

  getBondAtoms(_object: THREE.Object3D, _instanceId: number): [number, number] | null {
    return null;
  }

  syncSelection(_added: number[], _removed: number[]): void {
    // Handled by OutlinePass via proxy meshes
  }

  buildSelectionProxies(selectedIndices: ReadonlySet<number>): THREE.Object3D[] {
    if (selectedIndices.size === 0 || !this.atomMesh) return [];

    const origToVis = new Map<number, number>();
    for (let vi = 0; vi < this._visibleToOriginal.length; vi++) {
      origToVis.set(this._visibleToOriginal[vi], vi);
    }

    const selectedVis: number[] = [];
    for (const idx of selectedIndices) {
      const vi = origToVis.get(idx);
      if (vi !== undefined) selectedVis.push(vi);
    }
    if (selectedVis.length === 0) return [];

    // Low-poly sphere proxy for OutlinePass
    const proxyGeo = new THREE.SphereGeometry(1, 8, 6);
    const proxy = new THREE.InstancedMesh(proxyGeo, PROXY_MATERIAL, selectedVis.length);
    const mat = new THREE.Matrix4();

    for (let i = 0; i < selectedVis.length; i++) {
      const vi = selectedVis[i];
      const cx = this._centers[vi * 3];
      const cy = this._centers[vi * 3 + 1];
      const cz = this._centers[vi * 3 + 2];
      const r = this._radii[vi];
      mat.makeScale(r, r, r);
      mat.setPosition(cx, cy, cz);
      proxy.setMatrixAt(i, mat);
    }
    proxy.instanceMatrix.needsUpdate = true;
    return [proxy];
  }

  updateMaterial(_settings: MaterialSettings): void {
    // Impostor uses custom shader, no roughness/metalness
  }

  updatePositions(positions: Float32Array): void {
    if (!this.atomMesh) return;
    const arr = this.atomMesh.instanceMatrix.array as Float32Array;
    for (let vi = 0; vi < this._visibleToOriginal.length; vi++) {
      const ai = this._visibleToOriginal[vi];
      const px = positions[ai * 3];
      const py = positions[ai * 3 + 1];
      const pz = positions[ai * 3 + 2];
      const offset = vi * 16;
      arr[offset + 12] = px;
      arr[offset + 13] = py;
      arr[offset + 14] = pz;
      // Update cached centers for raycasting
      this._centers[vi * 3] = px;
      this._centers[vi * 3 + 1] = py;
      this._centers[vi * 3 + 2] = pz;
    }
    this.atomMesh.instanceMatrix.needsUpdate = true;
  }

  dispose(): void {
    if (this.atomMesh) {
      this.atomMesh.geometry.dispose();
      this.group.remove(this.atomMesh);
      this.atomMesh = null;
    }
    this.ghostAttr = null;
    this._visibleToOriginal = [];
    this._centers = new Float32Array(0);
    this._radii = new Float32Array(0);
  }

  /** Analytical ray-sphere raycasting for all instances */
  private _raycast(raycaster: THREE.Raycaster, intersects: THREE.Intersection[]): void {
    if (!this.atomMesh) return;

    const origin = raycaster.ray.origin;
    const direction = raycaster.ray.direction;
    const near = raycaster.near;
    const far = raycaster.far;

    // Transform ray to local space
    const matrixWorld = this.atomMesh.matrixWorld;
    const invMatrix = new THREE.Matrix4().copy(matrixWorld).invert();
    const localOrigin = origin.clone().applyMatrix4(invMatrix);
    const localDir = direction.clone().transformDirection(invMatrix);

    for (let vi = 0; vi < this._visibleToOriginal.length; vi++) {
      const cx = this._centers[vi * 3];
      const cy = this._centers[vi * 3 + 1];
      const cz = this._centers[vi * 3 + 2];
      const r = this._radii[vi];

      const ocx = localOrigin.x - cx;
      const ocy = localOrigin.y - cy;
      const ocz = localOrigin.z - cz;

      const a = localDir.x * localDir.x + localDir.y * localDir.y + localDir.z * localDir.z;
      const b = 2 * (ocx * localDir.x + ocy * localDir.y + ocz * localDir.z);
      const c = ocx * ocx + ocy * ocy + ocz * ocz - r * r;

      const discriminant = b * b - 4 * a * c;
      if (discriminant < 0) continue;

      const t = (-b - Math.sqrt(discriminant)) / (2 * a);
      if (t < near || t > far) continue;

      const hitPoint = new THREE.Vector3(
        localOrigin.x + t * localDir.x,
        localOrigin.y + t * localDir.y,
        localOrigin.z + t * localDir.z,
      );
      hitPoint.applyMatrix4(matrixWorld);

      intersects.push({
        distance: t,
        point: hitPoint,
        object: this.atomMesh,
        instanceId: vi,
        face: null,
        faceIndex: null,
        uv: null,
        normal: null,
      } as unknown as THREE.Intersection);
    }
  }
}
