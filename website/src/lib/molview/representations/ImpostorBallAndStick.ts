import * as THREE from "three";
import type { Representation, BuildOptions } from "./types";
import type { StructureData } from "../data/types";
import type { MaterialSettings } from "../lighting";
import { getElementColor } from "../data/colors";
import { getElementByNumber } from "../data/elements";

const ATOM_SCALE = 0.3;
const BOND_RADIUS = 0.08;
const MIN_ATOM_RADIUS = BOND_RADIUS * 1.5;
const BILLBOARD_MARGIN = 1.5;

const PROXY_MATERIAL = new THREE.MeshBasicMaterial({
  transparent: true,
  opacity: 0,
  depthWrite: false,
});

// --- Sphere impostor shaders (same as ImpostorSpacefill but smaller radii) ---

const SPHERE_VERT = /* glsl */ `
attribute float aGhost;
attribute float aRadius;

varying vec3 vViewCenter;
varying vec3 vViewPos;
varying float vRadius;
varying float vGhost;
varying vec3 vColor;
varying vec4 vProjZW;

void main() {
  #ifdef USE_INSTANCING_COLOR
    vColor = instanceColor;
  #else
    vColor = vec3(0.8);
  #endif
  vGhost = aGhost;
  vRadius = aRadius;

  // Packed so fragment can do: clipZ = viewZ * .x + .z, clipW = viewZ * .y + .w
  vProjZW = vec4(
    projectionMatrix[2][2], projectionMatrix[2][3],
    projectionMatrix[3][2], projectionMatrix[3][3]
  );

  // Instance center in world space
  vec4 worldCenter = modelMatrix * instanceMatrix * vec4(0.0, 0.0, 0.0, 1.0);

  // View-space center (for fragment shader ray tracing)
  vec4 viewCenter = viewMatrix * worldCenter;
  vViewCenter = viewCenter.xyz;

  // Billboard: construct quad in world space using camera basis vectors
  vec3 camRight = vec3(viewMatrix[0][0], viewMatrix[1][0], viewMatrix[2][0]);
  vec3 camUp    = vec3(viewMatrix[0][1], viewMatrix[1][1], viewMatrix[2][1]);
  float extent = aRadius * ${BILLBOARD_MARGIN.toFixed(1)};
  vec3 worldPos = worldCenter.xyz
    + camRight * position.x * extent
    + camUp    * position.y * extent;

  vec4 viewPos = viewMatrix * vec4(worldPos, 1.0);
  vViewPos = viewPos.xyz;

  gl_Position = projectionMatrix * viewPos;
}
`;

const SPHERE_FRAG = /* glsl */ `
precision highp float;

varying vec3 vViewCenter;
varying vec3 vViewPos;
varying float vRadius;
varying float vGhost;
varying vec3 vColor;
varying vec4 vProjZW;

float calcDepth(float viewZ) {
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
  vec3 rayOrigin;
  vec3 rayDir;
  if (isOrthographic) {
    rayOrigin = vec3(vViewPos.xy, 0.0);
    rayDir = vec3(0.0, 0.0, -1.0);
  } else {
    rayOrigin = vec3(0.0);
    rayDir = normalize(vViewPos);
  }

  vec3 oc = rayOrigin - vViewCenter;
  float b = dot(oc, rayDir);
  float c = dot(oc, oc) - vRadius * vRadius;
  float disc = b * b - c;
  if (disc < 0.0) discard;

  float t = -b - sqrt(disc);
  vec3 hitPos = rayOrigin + t * rayDir;
  vec3 normal = normalize(hitPos - vViewCenter);
  vec3 viewDir = normalize(-hitPos);
  vec3 lightDir = normalize(vec3(0.5, 1.0, 0.8));

  float NdotL = max(dot(normal, lightDir), 0.0);
  vec3 halfDir = normalize(lightDir + viewDir);
  float spec = pow(max(dot(normal, halfDir), 0.0), 64.0);

  vec3 color = vColor * (0.35 + 0.55 * NdotL) + vec3(1.0) * spec * 0.3;
  if (vGhost > 0.5) {
    float lum = dot(color, vec3(0.299, 0.587, 0.114));
    vec3 faded = mix(color, vec3(lum), 0.7);
    faded = mix(faded, vec3(0.75), 0.5);
    color = faded * (dot(faded, vec3(0.299, 0.587, 0.114)) > 0.5 ? 1.0 : 0.7);
  }

  gl_FragColor = vec4(dither(color), 1.0);
  // Small bias so spheres win over cylinders at junctions (avoid z-fighting bands)
  gl_FragDepth = calcDepth(hitPos.z) - 0.00001;
}
`;

// --- Cylinder impostor shaders ---

const CYLINDER_VERT = /* glsl */ `
attribute float aGhost;
attribute vec3 aCylinderStart;
attribute vec3 aCylinderEnd;
attribute float aCylinderRadius;

varying vec3 vViewStart;
varying vec3 vViewEnd;
varying vec3 vViewPos;
varying float vCylinderRadius;
varying float vGhost;
varying vec3 vColor;
varying vec4 vProjZW;

void main() {
  #ifdef USE_INSTANCING_COLOR
    vColor = instanceColor;
  #else
    vColor = vec3(0.8);
  #endif
  vGhost = aGhost;
  vCylinderRadius = aCylinderRadius;

  // Packed so fragment can do: clipZ = viewZ * .x + .z, clipW = viewZ * .y + .w
  vProjZW = vec4(
    projectionMatrix[2][2], projectionMatrix[2][3],
    projectionMatrix[3][2], projectionMatrix[3][3]
  );

  // Transform cylinder endpoints to view space
  vViewStart = (viewMatrix * vec4(aCylinderStart, 1.0)).xyz;
  vViewEnd = (viewMatrix * vec4(aCylinderEnd, 1.0)).xyz;

  // Billboard box: instance matrix encodes center + orientation
  vec4 mvPosition = modelViewMatrix * instanceMatrix * vec4(position, 1.0);
  vViewPos = mvPosition.xyz;
  gl_Position = projectionMatrix * mvPosition;
}
`;

const CYLINDER_FRAG = /* glsl */ `
precision highp float;

varying vec3 vViewStart;
varying vec3 vViewEnd;
varying vec3 vViewPos;
varying float vCylinderRadius;
varying float vGhost;
varying vec3 vColor;
varying vec4 vProjZW;

float calcDepth(float viewZ) {
  float clipZ = viewZ * vProjZW.x + vProjZW.z;
  float clipW = viewZ * vProjZW.y + vProjZW.w;
  return 0.5 + 0.5 * clipZ / clipW;
}

// Dither to reduce 8-bit color banding
vec3 dither(vec3 color) {
  vec3 d = vec3(dot(vec2(171.0, 231.0), gl_FragCoord.xy));
  d = fract(d / vec3(103.0, 71.0, 97.0)) - 0.5;
  return color + d / 255.0;
}

void main() {
  vec3 rayOrigin;
  vec3 rayDir;
  if (isOrthographic) {
    rayOrigin = vec3(vViewPos.xy, 0.0);
    rayDir = vec3(0.0, 0.0, -1.0);
  } else {
    rayOrigin = vec3(0.0);
    rayDir = normalize(vViewPos);
  }

  // Ray-cylinder intersection
  vec3 ab = vViewEnd - vViewStart;
  vec3 ao = rayOrigin - vViewStart;
  float abab = dot(ab, ab);
  float abrd = dot(ab, rayDir);
  float abao = dot(ab, ao);

  float A = abab * dot(rayDir, rayDir) - abrd * abrd;
  float B = 2.0 * (abab * dot(ao, rayDir) - abrd * abao);
  float C = abab * dot(ao, ao) - abao * abao - vCylinderRadius * vCylinderRadius * abab;

  float disc = B * B - 4.0 * A * C;
  if (disc < 0.0) discard;

  float sqrtDisc = sqrt(disc);
  float t1 = (-B - sqrtDisc) / (2.0 * A);
  float t2 = (-B + sqrtDisc) / (2.0 * A);

  // Use nearest positive intersection
  float t = t1 > 0.0 ? t1 : t2;
  if (t < 0.0) discard;

  vec3 hitPos = rayOrigin + t * rayDir;

  // Check caps: project hit onto cylinder axis
  float param = dot(hitPos - vViewStart, ab) / abab;
  if (param < 0.0 || param > 1.0) {
    // Try flat cap intersection
    float tCap = -1.0;
    vec3 capNormal;

    // Try start cap
    vec3 nStart = normalize(-ab);
    float denomStart = dot(nStart, rayDir);
    if (abs(denomStart) > 1e-6) {
      float tc = dot(vViewStart - rayOrigin, nStart) / denomStart;
      if (tc > 0.0) {
        vec3 p = rayOrigin + tc * rayDir;
        if (length(p - vViewStart) <= vCylinderRadius) {
          tCap = tc;
          capNormal = nStart;
        }
      }
    }

    // Try end cap (keep nearest)
    vec3 nEnd = normalize(ab);
    float denomEnd = dot(nEnd, rayDir);
    if (abs(denomEnd) > 1e-6) {
      float tc = dot(vViewEnd - rayOrigin, nEnd) / denomEnd;
      if (tc > 0.0 && (tCap < 0.0 || tc < tCap)) {
        vec3 p = rayOrigin + tc * rayDir;
        if (length(p - vViewEnd) <= vCylinderRadius) {
          tCap = tc;
          capNormal = nEnd;
        }
      }
    }

    if (tCap < 0.0) discard;

    hitPos = rayOrigin + tCap * rayDir;
    vec3 viewDir = normalize(-hitPos);
    vec3 lightDir = normalize(vec3(0.5, 1.0, 0.8));
    float NdotL = max(dot(capNormal, lightDir), 0.0);
    vec3 color = vColor * (0.35 + 0.55 * NdotL);
    if (vGhost > 0.5) {
      float lum = dot(color, vec3(0.299, 0.587, 0.114));
      color = mix(color, vec3(lum), 0.7);
      color = mix(color, vec3(0.75), 0.5);
    }

    gl_FragColor = vec4(dither(color), 1.0);
    gl_FragDepth = calcDepth(hitPos.z);
    return;
  }

  // Surface normal at hit point
  vec3 axisPoint = vViewStart + param * ab;
  vec3 normal = normalize(hitPos - axisPoint);

  vec3 viewDir = normalize(-hitPos);
  vec3 lightDir = normalize(vec3(0.5, 1.0, 0.8));
  float NdotL = max(dot(normal, lightDir), 0.0);
  vec3 halfDir = normalize(lightDir + viewDir);
  float spec = pow(max(dot(normal, halfDir), 0.0), 64.0);

  vec3 color = vColor * (0.35 + 0.55 * NdotL) + vec3(1.0) * spec * 0.15;
  if (vGhost > 0.5) {
    float lum = dot(color, vec3(0.299, 0.587, 0.114));
    color = mix(color, vec3(lum), 0.7);
    color = mix(color, vec3(0.75), 0.5);
  }

  gl_FragColor = vec4(dither(color), 1.0);
  gl_FragDepth = calcDepth(hitPos.z);
}
`;

interface BondConnection {
  bondIndex: number;
  isA: boolean;
}

export class ImpostorBallAndStickRepresentation implements Representation {
  readonly type = "ball+stick" as const;
  readonly group = new THREE.Group();

  private atomMesh: THREE.InstancedMesh | null = null;
  private bondMesh: THREE.InstancedMesh | null = null;
  private sphereMaterial: THREE.ShaderMaterial;
  private cylinderMaterial: THREE.ShaderMaterial;

  private _visibleToOriginal: number[] = [];
  private _visibleBonds: StructureData["bonds"] = [];
  private atomBondMap = new Map<number, BondConnection[]>();
  private _centers: Float32Array = new Float32Array(0);
  private _radii: Float32Array = new Float32Array(0);

  private atomGhostAttr: THREE.InstancedBufferAttribute | null = null;
  private bondGhostAttr: THREE.InstancedBufferAttribute | null = null;

  // Cylinder endpoint attributes for updatePositions
  private _cylinderStartAttr: THREE.InstancedBufferAttribute | null = null;
  private _cylinderEndAttr: THREE.InstancedBufferAttribute | null = null;

  constructor() {
    this.sphereMaterial = new THREE.ShaderMaterial({
      vertexShader: SPHERE_VERT,
      fragmentShader: SPHERE_FRAG,
      side: THREE.DoubleSide,
      depthWrite: true,
      depthTest: true,
    });
    this.cylinderMaterial = new THREE.ShaderMaterial({
      vertexShader: CYLINDER_VERT,
      fragmentShader: CYLINDER_FRAG,
      side: THREE.FrontSide,
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

    // Visible atoms
    const visibleAtomIndices: number[] = [];
    const originalToVisible = new Map<number, number>();
    for (let i = 0; i < data.atoms.length; i++) {
      if (hidden && hidden.has(data.atoms[i].atomicNumber)) continue;
      if (hiddenIdx && hiddenIdx.has(i)) continue;
      originalToVisible.set(i, visibleAtomIndices.length);
      visibleAtomIndices.push(i);
    }
    this._visibleToOriginal = visibleAtomIndices;

    // --- Atoms (sphere impostors) ---
    if (visibleAtomIndices.length > 0) {
      const count = visibleAtomIndices.length;
      const quadGeo = new THREE.PlaneGeometry(2, 2);
      this.atomMesh = new THREE.InstancedMesh(quadGeo, this.sphereMaterial, count);

      const matrix = new THREE.Matrix4();
      const color = new THREE.Color();
      const radii = new Float32Array(count);
      this._centers = new Float32Array(count * 3);
      this._radii = new Float32Array(count);

      for (let vi = 0; vi < count; vi++) {
        const i = visibleAtomIndices[vi];
        const atom = data.atoms[i];
        const radius = Math.max(
          getElementByNumber(atom.atomicNumber).covalentRadius * ATOM_SCALE,
          MIN_ATOM_RADIUS,
        );
        radii[vi] = radius;
        this._radii[vi] = radius;
        this._centers[vi * 3] = atom.position.x;
        this._centers[vi * 3 + 1] = atom.position.y;
        this._centers[vi * 3 + 2] = atom.position.z;

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

      this.atomMesh.geometry.setAttribute("aRadius", new THREE.InstancedBufferAttribute(radii, 1));

      const atomGhostArray = new Float32Array(count);
      if (data.ghostStart !== undefined && data.ghostStart >= 0) {
        for (let vi = 0; vi < count; vi++) {
          if (visibleAtomIndices[vi] >= data.ghostStart) atomGhostArray[vi] = 1.0;
        }
      }
      this.atomGhostAttr = new THREE.InstancedBufferAttribute(atomGhostArray, 1);
      this.atomMesh.geometry.setAttribute("aGhost", this.atomGhostAttr);

      this.atomMesh.instanceMatrix.needsUpdate = true;
      if (this.atomMesh.instanceColor) this.atomMesh.instanceColor.needsUpdate = true;
      this.atomMesh.frustumCulled = false;
      // Layer 31: visible in main render but excluded from GTAO normal pass
      this.atomMesh.layers.set(31);
      this.atomMesh.raycast = this._raycastSpheres.bind(this);
      this.group.add(this.atomMesh);
    }

    // --- Bonds (cylinder impostor proxies using oriented boxes) ---
    this.atomBondMap.clear();
    const visibleBonds: { bond: typeof data.bonds[0]; originalIndex: number }[] = [];
    for (let i = 0; i < data.bonds.length; i++) {
      const bond = data.bonds[i];
      if (hidden) {
        if (hidden.has(data.atoms[bond.atomA].atomicNumber) ||
            hidden.has(data.atoms[bond.atomB].atomicNumber)) continue;
      }
      visibleBonds.push({ bond, originalIndex: i });
    }

    if (visibleBonds.length > 0) {
      // Two half-cylinders per bond
      const bondCount = visibleBonds.length * 2;
      const boxGeo = new THREE.BoxGeometry(1, 1, 1);
      this.bondMesh = new THREE.InstancedMesh(boxGeo, this.cylinderMaterial, bondCount);

      const matrix = new THREE.Matrix4();
      const color = new THREE.Color();
      const start = new THREE.Vector3();
      const end = new THREE.Vector3();
      const midpoint = new THREE.Vector3();
      const direction = new THREE.Vector3();
      const up = new THREE.Vector3(0, 1, 0);
      const quaternion = new THREE.Quaternion();

      const cylinderStarts = new Float32Array(bondCount * 3);
      const cylinderEnds = new Float32Array(bondCount * 3);
      const cylinderRadii = new Float32Array(bondCount);

      for (let vi = 0; vi < visibleBonds.length; vi++) {
        const { bond } = visibleBonds[vi];
        const atomA = data.atoms[bond.atomA];
        const atomB = data.atoms[bond.atomB];

        if (!this.atomBondMap.has(bond.atomA)) this.atomBondMap.set(bond.atomA, []);
        this.atomBondMap.get(bond.atomA)!.push({ bondIndex: vi, isA: true });
        if (!this.atomBondMap.has(bond.atomB)) this.atomBondMap.set(bond.atomB, []);
        this.atomBondMap.get(bond.atomB)!.push({ bondIndex: vi, isA: false });

        start.set(atomA.position.x, atomA.position.y, atomA.position.z);
        end.set(atomB.position.x, atomB.position.y, atomB.position.z);
        midpoint.addVectors(start, end).multiplyScalar(0.5);

        const fullLength = start.distanceTo(end);
        const halfLength = fullLength / 2;
        direction.subVectors(end, start).normalize();
        quaternion.setFromUnitVectors(up, direction);

        // Half A
        const posA = new THREE.Vector3().addVectors(start, midpoint).multiplyScalar(0.5);
        matrix.makeRotationFromQuaternion(quaternion);
        matrix.scale(new THREE.Vector3(BOND_RADIUS * 3, halfLength, BOND_RADIUS * 3));
        matrix.setPosition(posA);
        this.bondMesh.setMatrixAt(vi * 2, matrix);

        cylinderStarts[vi * 6] = start.x; cylinderStarts[vi * 6 + 1] = start.y; cylinderStarts[vi * 6 + 2] = start.z;
        cylinderEnds[vi * 6] = midpoint.x; cylinderEnds[vi * 6 + 1] = midpoint.y; cylinderEnds[vi * 6 + 2] = midpoint.z;
        cylinderRadii[vi * 2] = BOND_RADIUS;

        if (co) {
          color.setRGB(co[0], co[1], co[2], THREE.SRGBColorSpace);
        } else if (colorFn) {
          const [rA, gA, bA] = colorFn(bond.atomA);
          color.setRGB(rA, gA, bA, THREE.SRGBColorSpace);
        } else {
          const [rA, gA, bA] = getElementColor(atomA.atomicNumber);
          color.setRGB(rA, gA, bA, THREE.SRGBColorSpace);
        }
        this.bondMesh.setColorAt(vi * 2, color);

        // Half B
        const posB = new THREE.Vector3().addVectors(midpoint, end).multiplyScalar(0.5);
        matrix.makeRotationFromQuaternion(quaternion);
        matrix.scale(new THREE.Vector3(BOND_RADIUS * 3, halfLength, BOND_RADIUS * 3));
        matrix.setPosition(posB);
        this.bondMesh.setMatrixAt(vi * 2 + 1, matrix);

        cylinderStarts[vi * 6 + 3] = midpoint.x; cylinderStarts[vi * 6 + 4] = midpoint.y; cylinderStarts[vi * 6 + 5] = midpoint.z;
        cylinderEnds[vi * 6 + 3] = end.x; cylinderEnds[vi * 6 + 4] = end.y; cylinderEnds[vi * 6 + 5] = end.z;
        cylinderRadii[vi * 2 + 1] = BOND_RADIUS;

        if (co) {
          color.setRGB(co[0], co[1], co[2], THREE.SRGBColorSpace);
        } else if (colorFn) {
          const [rB, gB, bB] = colorFn(bond.atomB);
          color.setRGB(rB, gB, bB, THREE.SRGBColorSpace);
        } else {
          const [rB, gB, bB] = getElementColor(atomB.atomicNumber);
          color.setRGB(rB, gB, bB, THREE.SRGBColorSpace);
        }
        this.bondMesh.setColorAt(vi * 2 + 1, color);
      }

      this._visibleBonds = visibleBonds.map(vb => vb.bond);

      this._cylinderStartAttr = new THREE.InstancedBufferAttribute(cylinderStarts, 3);
      this.bondMesh.geometry.setAttribute("aCylinderStart", this._cylinderStartAttr);
      this._cylinderEndAttr = new THREE.InstancedBufferAttribute(cylinderEnds, 3);
      this.bondMesh.geometry.setAttribute("aCylinderEnd", this._cylinderEndAttr);
      this.bondMesh.geometry.setAttribute("aCylinderRadius", new THREE.InstancedBufferAttribute(cylinderRadii, 1));

      const bondGhostArray = new Float32Array(bondCount);
      if (data.ghostStart !== undefined && data.ghostStart >= 0) {
        for (let vi = 0; vi < visibleBonds.length; vi++) {
          const { bond } = visibleBonds[vi];
          if (bond.atomA >= data.ghostStart) bondGhostArray[vi * 2] = 1.0;
          if (bond.atomB >= data.ghostStart) bondGhostArray[vi * 2 + 1] = 1.0;
        }
      }
      this.bondGhostAttr = new THREE.InstancedBufferAttribute(bondGhostArray, 1);
      this.bondMesh.geometry.setAttribute("aGhost", this.bondGhostAttr);

      this.bondMesh.instanceMatrix.needsUpdate = true;
      if (this.bondMesh.instanceColor) this.bondMesh.instanceColor.needsUpdate = true;
      this.bondMesh.frustumCulled = false;
      // Layer 31: visible in main render but excluded from GTAO normal pass
      this.bondMesh.layers.set(31);
      this.group.add(this.bondMesh);
    }
  }

  getAtomIndex(object: THREE.Object3D, instanceId: number): number {
    if (object === this.atomMesh) return this._visibleToOriginal[instanceId] ?? -1;
    return -1;
  }

  getBondAtoms(object: THREE.Object3D, instanceId: number): [number, number] | null {
    if (object === this.bondMesh) {
      const bondIndex = Math.floor(instanceId / 2);
      if (bondIndex < this._visibleBonds.length) {
        const bond = this._visibleBonds[bondIndex];
        return [bond.atomA, bond.atomB];
      }
    }
    return null;
  }

  syncSelection(_added: number[], _removed: number[]): void {}

  buildSelectionProxies(selectedIndices: ReadonlySet<number>): THREE.Object3D[] {
    const proxies: THREE.Object3D[] = [];
    if (selectedIndices.size === 0) return proxies;

    const origToVis = new Map<number, number>();
    for (let vi = 0; vi < this._visibleToOriginal.length; vi++) {
      origToVis.set(this._visibleToOriginal[vi], vi);
    }

    // Atom proxies (low-poly spheres)
    if (this.atomMesh) {
      const selectedVis: number[] = [];
      for (const idx of selectedIndices) {
        const vi = origToVis.get(idx);
        if (vi !== undefined) selectedVis.push(vi);
      }
      if (selectedVis.length > 0) {
        const proxyGeo = new THREE.SphereGeometry(1, 8, 6);
        const proxy = new THREE.InstancedMesh(proxyGeo, PROXY_MATERIAL, selectedVis.length);
        const mat = new THREE.Matrix4();
        for (let i = 0; i < selectedVis.length; i++) {
          const vi = selectedVis[i];
          const r = this._radii[vi];
          mat.makeScale(r, r, r);
          mat.setPosition(this._centers[vi * 3], this._centers[vi * 3 + 1], this._centers[vi * 3 + 2]);
          proxy.setMatrixAt(i, mat);
        }
        proxy.instanceMatrix.needsUpdate = true;
        proxies.push(proxy);
      }
    }

    // Bond proxies
    if (this.bondMesh) {
      const selectedBondInstances: number[] = [];
      for (const atomIdx of selectedIndices) {
        const connections = this.atomBondMap.get(atomIdx);
        if (!connections) continue;
        for (const { bondIndex, isA } of connections) {
          selectedBondInstances.push(isA ? bondIndex * 2 : bondIndex * 2 + 1);
        }
      }
      if (selectedBondInstances.length > 0) {
        const unique = [...new Set(selectedBondInstances)];
        const cylinderGeo = new THREE.CylinderGeometry(1, 1, 1, 6);
        const proxy = new THREE.InstancedMesh(cylinderGeo, PROXY_MATERIAL, unique.length);
        const mat = new THREE.Matrix4();
        for (let i = 0; i < unique.length; i++) {
          this.bondMesh.getMatrixAt(unique[i], mat);
          proxy.setMatrixAt(i, mat);
        }
        proxy.instanceMatrix.needsUpdate = true;
        proxies.push(proxy);
      }
    }

    return proxies;
  }

  updateMaterial(_settings: MaterialSettings): void {}

  updatePositions(positions: Float32Array): void {
    // Update atom positions
    if (this.atomMesh) {
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
        this._centers[vi * 3] = px;
        this._centers[vi * 3 + 1] = py;
        this._centers[vi * 3 + 2] = pz;
      }
      this.atomMesh.instanceMatrix.needsUpdate = true;
    }

    // Recompute bond cylinder endpoints and proxy transforms
    if (this.bondMesh && this._visibleBonds.length > 0 && this._cylinderStartAttr && this._cylinderEndAttr) {
      const starts = this._cylinderStartAttr.array as Float32Array;
      const ends = this._cylinderEndAttr.array as Float32Array;
      const start = new THREE.Vector3();
      const end = new THREE.Vector3();
      const midpoint = new THREE.Vector3();
      const direction = new THREE.Vector3();
      const up = new THREE.Vector3(0, 1, 0);
      const quaternion = new THREE.Quaternion();
      const matrix = new THREE.Matrix4();

      for (let vi = 0; vi < this._visibleBonds.length; vi++) {
        const bond = this._visibleBonds[vi];
        start.set(positions[bond.atomA * 3], positions[bond.atomA * 3 + 1], positions[bond.atomA * 3 + 2]);
        end.set(positions[bond.atomB * 3], positions[bond.atomB * 3 + 1], positions[bond.atomB * 3 + 2]);
        midpoint.addVectors(start, end).multiplyScalar(0.5);

        const fullLength = start.distanceTo(end);
        const halfLength = fullLength / 2;
        direction.subVectors(end, start).normalize();
        quaternion.setFromUnitVectors(up, direction);

        // Half A
        starts[vi * 6] = start.x; starts[vi * 6 + 1] = start.y; starts[vi * 6 + 2] = start.z;
        ends[vi * 6] = midpoint.x; ends[vi * 6 + 1] = midpoint.y; ends[vi * 6 + 2] = midpoint.z;

        const posA = new THREE.Vector3().addVectors(start, midpoint).multiplyScalar(0.5);
        matrix.makeRotationFromQuaternion(quaternion);
        matrix.scale(new THREE.Vector3(BOND_RADIUS * 3, halfLength, BOND_RADIUS * 3));
        matrix.setPosition(posA);
        this.bondMesh.setMatrixAt(vi * 2, matrix);

        // Half B
        starts[vi * 6 + 3] = midpoint.x; starts[vi * 6 + 4] = midpoint.y; starts[vi * 6 + 5] = midpoint.z;
        ends[vi * 6 + 3] = end.x; ends[vi * 6 + 4] = end.y; ends[vi * 6 + 5] = end.z;

        const posB = new THREE.Vector3().addVectors(midpoint, end).multiplyScalar(0.5);
        matrix.makeRotationFromQuaternion(quaternion);
        matrix.scale(new THREE.Vector3(BOND_RADIUS * 3, halfLength, BOND_RADIUS * 3));
        matrix.setPosition(posB);
        this.bondMesh.setMatrixAt(vi * 2 + 1, matrix);
      }

      this._cylinderStartAttr.needsUpdate = true;
      this._cylinderEndAttr.needsUpdate = true;
      this.bondMesh.instanceMatrix.needsUpdate = true;
    }
  }

  dispose(): void {
    if (this.atomMesh) {
      this.atomMesh.geometry.dispose();
      this.group.remove(this.atomMesh);
      this.atomMesh = null;
    }
    if (this.bondMesh) {
      this.bondMesh.geometry.dispose();
      this.group.remove(this.bondMesh);
      this.bondMesh = null;
    }
    this.atomGhostAttr = null;
    this.bondGhostAttr = null;
    this._cylinderStartAttr = null;
    this._cylinderEndAttr = null;
    this.atomBondMap.clear();
    this._visibleToOriginal = [];
    this._visibleBonds = [];
    this._centers = new Float32Array(0);
    this._radii = new Float32Array(0);
  }

  private _raycastSpheres(raycaster: THREE.Raycaster, intersects: THREE.Intersection[]): void {
    if (!this.atomMesh) return;

    const origin = raycaster.ray.origin;
    const direction = raycaster.ray.direction;
    const near = raycaster.near;
    const far = raycaster.far;

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
