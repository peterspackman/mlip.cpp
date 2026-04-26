import * as THREE from "three";

export type MaterialStyleType = "standard" | "toon" | "gooch" | "iridescent" | "matcap" | "hatching";

/**
 * GLSL snippet for ghost atom styling.
 * Desaturated, faded toward a light tone with a 2-step toon ramp — fully opaque.
 */
const GHOST_FADE_GLSL = /* glsl */ `
vec3 applyGhostFade(vec3 color) {
  float lum = dot(color, vec3(0.299, 0.587, 0.114));
  vec3 grey = vec3(lum);
  vec3 faded = mix(color, grey, 0.7);
  faded = mix(faded, vec3(0.75), 0.5);
  float toonLum = dot(faded, vec3(0.299, 0.587, 0.114));
  float step = toonLum > 0.5 ? 1.0 : 0.7;
  return faded * step;
}
`;

// --- Shared textures ---

/** Build a 3-tone gradient map for MeshToonMaterial cel-shading */
let _gradientMap: THREE.DataTexture | null = null;
function gradientMap(): THREE.DataTexture {
  if (!_gradientMap) {
    const data = new Uint8Array([80, 160, 255]);
    _gradientMap = new THREE.DataTexture(data, 3, 1, THREE.RedFormat);
    _gradientMap.minFilter = THREE.NearestFilter;
    _gradientMap.magFilter = THREE.NearestFilter;
    _gradientMap.needsUpdate = true;
  }
  return _gradientMap;
}

/** Generate a clay/ceramic matcap texture procedurally */
let _matcapTexture: THREE.DataTexture | null = null;
function matcapTexture(): THREE.DataTexture {
  if (_matcapTexture) return _matcapTexture;
  const size = 256;
  const data = new Uint8Array(size * size * 4);
  // Light direction (upper-right)
  const lx = 0.4, ly = 0.6, lz = 0.7;
  const llen = Math.sqrt(lx * lx + ly * ly + lz * lz);
  const lnx = lx / llen, lny = ly / llen, lnz = lz / llen;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const nx = (x / (size - 1)) * 2 - 1;
      const ny = ((size - 1 - y) / (size - 1)) * 2 - 1;
      const r2 = nx * nx + ny * ny;
      const idx = (y * size + x) * 4;

      if (r2 > 1.0) {
        // Outside sphere — use silhouette color
        data[idx] = 100; data[idx + 1] = 90; data[idx + 2] = 82; data[idx + 3] = 255;
        continue;
      }

      const nz = Math.sqrt(1 - r2);
      const diff = Math.max(0, nx * lnx + ny * lny + nz * lnz);

      // Specular (Blinn-Phong half-vector with view = [0,0,1])
      const hx = lnx, hy = lny, hz = lnz + 1;
      const hlen = Math.sqrt(hx * hx + hy * hy + hz * hz);
      const spec = Math.pow(Math.max(0, (nx * hx + ny * hy + nz * hz) / hlen), 40);

      // Rim light
      const rimLight = Math.pow(1 - nz, 3) * 0.12;

      const shade = 0.22 + diff * 0.58 + spec * 0.25 + rimLight;
      // Warm clay tint
      data[idx] = Math.min(255, Math.round(Math.min(1, shade * 1.04) * 255));
      data[idx + 1] = Math.min(255, Math.round(Math.min(1, shade * 0.97) * 255));
      data[idx + 2] = Math.min(255, Math.round(Math.min(1, shade * 0.88) * 255));
      data[idx + 3] = 255;
    }
  }

  _matcapTexture = new THREE.DataTexture(data, size, size, THREE.RGBAFormat);
  _matcapTexture.magFilter = THREE.LinearFilter;
  _matcapTexture.minFilter = THREE.LinearFilter;
  _matcapTexture.needsUpdate = true;
  return _matcapTexture;
}

// --- Ghost fade patch for built-in materials (Standard, Toon, Physical, Matcap) ---

function patchGhostFade(material: THREE.Material, cacheKey: string): void {
  const mat = material as any;
  mat.customProgramCacheKey = () => cacheKey;
  mat.onBeforeCompile = (shader: any) => {
    shader.vertexShader = shader.vertexShader.replace(
      "#include <common>",
      `#include <common>
       attribute float aGhost;
       varying float vGhost;`,
    );
    shader.vertexShader = shader.vertexShader.replace(
      "#include <begin_vertex>",
      `#include <begin_vertex>
       vGhost = aGhost;`,
    );
    shader.fragmentShader = shader.fragmentShader.replace(
      "#include <common>",
      `#include <common>
       varying float vGhost;
       ${GHOST_FADE_GLSL}`,
    );
    shader.fragmentShader = shader.fragmentShader.replace(
      "#include <opaque_fragment>",
      `#include <opaque_fragment>
       if (vGhost > 0.5) {
         gl_FragColor.rgb = applyGhostFade(gl_FragColor.rgb);
       }`,
    );
  };
}

// --- Gooch shader (custom ShaderMaterial for atoms/bonds) ---

const GOOCH_VERTEX = /* glsl */ `
varying vec3 vNormal;
varying vec3 vViewPosition;
varying vec3 vInstanceColor;
varying float vGhost;

attribute float aGhost;

void main() {
  vGhost = aGhost;

  #ifdef USE_INSTANCING_COLOR
    vInstanceColor = instanceColor.rgb;
  #else
    vInstanceColor = vec3(0.8);
  #endif

  vec4 mvPosition = modelViewMatrix * instanceMatrix * vec4(position, 1.0);
  vViewPosition = -mvPosition.xyz;

  mat3 normalMat = transpose(inverse(mat3(modelViewMatrix * instanceMatrix)));
  vNormal = normalize(normalMat * normal);

  gl_Position = projectionMatrix * mvPosition;
}
`;

const GOOCH_FRAGMENT = /* glsl */ `
varying vec3 vNormal;
varying vec3 vViewPosition;
varying vec3 vInstanceColor;
varying float vGhost;

uniform vec3 uBaseColor;
uniform bool uUseVertexColor;

${GHOST_FADE_GLSL}

void main() {
  vec3 normal = normalize(vNormal);
  vec3 viewDir = normalize(vViewPosition);
  vec3 lightDir = normalize(vec3(0.5, 1.0, 0.8));

  float NdotL = dot(normal, lightDir);
  float t = (NdotL + 1.0) * 0.5;

  vec3 surfaceColor = uUseVertexColor ? vInstanceColor : uBaseColor;
  vec3 coolColor = surfaceColor * 0.3 + vec3(0.0, 0.0, 0.15);
  vec3 warmColor = surfaceColor * 0.95 + vec3(0.05, 0.02, 0.0);

  vec3 gooch = mix(coolColor, warmColor, t);

  vec3 halfDir = normalize(lightDir + viewDir);
  float spec = pow(max(dot(normal, halfDir), 0.0), 64.0);

  float edgeFactor = 1.0 - max(dot(normalize(vNormal), viewDir), 0.0);
  float edge = smoothstep(0.2, 0.6, edgeFactor);

  vec3 color = gooch + vec3(1.0) * spec * 0.3;
  color = mix(color, color * 0.4, edge * 0.25);

  if (vGhost > 0.5) {
    color = applyGhostFade(color);
  }

  gl_FragColor = vec4(color, 1.0);
}
`;

// --- Hatching shader (custom ShaderMaterial for atoms/bonds) ---

const HATCHING_VERTEX = /* glsl */ `
varying vec3 vNormal;
varying vec3 vViewPosition;
varying vec3 vColor;
varying float vGhost;

attribute float aGhost;

void main() {
  vGhost = aGhost;

  #ifdef USE_INSTANCING_COLOR
    vColor = instanceColor.rgb;
  #else
    vColor = vec3(0.8);
  #endif

  vec4 mvPosition = modelViewMatrix * instanceMatrix * vec4(position, 1.0);
  vViewPosition = -mvPosition.xyz;

  mat3 normalMat = transpose(inverse(mat3(modelViewMatrix * instanceMatrix)));
  vNormal = normalize(normalMat * normal);

  gl_Position = projectionMatrix * mvPosition;
}
`;

/** Shared GLSL for screen-space cross-hatching */
const HATCHING_CORE_GLSL = /* glsl */ `
vec3 applyHatching(vec3 surfaceColor, float shade) {
  // Screen-space line coordinates
  vec2 uv = gl_FragCoord.xy / 5.0;

  float lineW = 0.35;

  // Layer 1: 45-degree diagonal — always present, gives base texture
  float h1 = abs(fract((uv.x + uv.y) * 0.5) - 0.5) * 2.0;
  float line1 = smoothstep(lineW - 0.15, lineW + 0.15, h1);

  // Layer 2: opposite diagonal for medium shadows
  float h2 = abs(fract((uv.x - uv.y) * 0.5) - 0.5) * 2.0;
  float line2 = smoothstep(lineW - 0.15, lineW + 0.15, h2);

  // Layer 3: horizontal for deep shadows
  float h3 = abs(fract(uv.y * 0.7) - 0.5) * 2.0;
  float line3 = smoothstep(lineW - 0.15, lineW + 0.15, h3);

  // Always hatch — density increases in shadow
  float paper = line1;
  if (shade < 0.6) paper = min(paper, line2);
  if (shade < 0.3) paper = min(paper, line3);

  // Colored ink on tinted paper
  vec3 inkColor = surfaceColor * 0.2;
  vec3 paperColor = mix(vec3(0.96), surfaceColor * 0.7 + vec3(0.3), 0.15);
  return mix(inkColor, paperColor, paper);
}
`;

const HATCHING_FRAGMENT = /* glsl */ `
varying vec3 vNormal;
varying vec3 vViewPosition;
varying vec3 vColor;
varying float vGhost;

${GHOST_FADE_GLSL}
${HATCHING_CORE_GLSL}

void main() {
  vec3 normal = normalize(vNormal);
  vec3 lightDir = normalize(vec3(0.5, 1.0, 0.8));

  float shade = dot(normal, lightDir) * 0.5 + 0.5;
  shade = shade * 0.8 + 0.2;

  vec3 color = applyHatching(vColor, shade);

  if (vGhost > 0.5) {
    color = applyGhostFade(color);
  }

  gl_FragColor = vec4(color, 1.0);
}
`;

// --- Factory functions ---

export interface StyledMaterials {
  /** Material for atom/bond meshes (InstancedMesh with instanceColor) */
  atom: THREE.Material;
  /** Material for bond meshes (same type, possibly different params) */
  bond: THREE.Material;
  /** Whether this style responds to roughness/metalness updates */
  supportsPhysical: boolean;
}

/**
 * Create atom + bond materials for a given style.
 * All materials come pre-patched for ghost fade.
 */
export function createStyledMaterials(
  style: MaterialStyleType,
  atomRoughness = 0.35,
  atomMetalness = 0.0,
): StyledMaterials {
  switch (style) {
    case "standard": {
      const atom = new THREE.MeshStandardMaterial({ roughness: atomRoughness, metalness: atomMetalness });
      patchGhostFade(atom, "ghost-standard");
      const bond = new THREE.MeshStandardMaterial({ roughness: atomRoughness + 0.15, metalness: atomMetalness });
      patchGhostFade(bond, "ghost-standard");
      return { atom, bond, supportsPhysical: true };
    }

    case "toon": {
      const gm = gradientMap();
      const atom = new THREE.MeshToonMaterial({ gradientMap: gm });
      patchGhostFade(atom, "ghost-toon");
      const bond = new THREE.MeshToonMaterial({ gradientMap: gm });
      patchGhostFade(bond, "ghost-toon");
      return { atom, bond, supportsPhysical: false };
    }

    case "gooch": {
      const mkGooch = () => new THREE.ShaderMaterial({
        vertexShader: GOOCH_VERTEX,
        fragmentShader: GOOCH_FRAGMENT,
        uniforms: {
          uBaseColor: { value: new THREE.Color(0.8, 0.8, 0.8) },
          uUseVertexColor: { value: true },
        },
      });
      return { atom: mkGooch(), bond: mkGooch(), supportsPhysical: false };
    }

    case "iridescent": {
      const atom = new THREE.MeshPhysicalMaterial({
        roughness: 0.15,
        metalness: 0.0,
        clearcoat: 1.0,
        clearcoatRoughness: 0.05,
        iridescence: 1.0,
        iridescenceIOR: 1.5,
        iridescenceThicknessRange: [200, 600],
        sheen: 0.3,
        sheenRoughness: 0.3,
        sheenColor: new THREE.Color(0.4, 0.2, 0.8),
      });
      patchGhostFade(atom, "ghost-iridescent");
      const bond = new THREE.MeshPhysicalMaterial({
        roughness: 0.2,
        metalness: 0.0,
        clearcoat: 0.8,
        clearcoatRoughness: 0.1,
        iridescence: 0.7,
        iridescenceIOR: 1.4,
        iridescenceThicknessRange: [300, 500],
      });
      patchGhostFade(bond, "ghost-iridescent");
      return { atom, bond, supportsPhysical: true };
    }

    case "matcap": {
      const mc = matcapTexture();
      const atom = new THREE.MeshMatcapMaterial({ matcap: mc });
      patchGhostFade(atom, "ghost-matcap");
      const bond = new THREE.MeshMatcapMaterial({ matcap: mc });
      patchGhostFade(bond, "ghost-matcap");
      return { atom, bond, supportsPhysical: false };
    }

    case "hatching": {
      const mkHatch = () => new THREE.ShaderMaterial({
        vertexShader: HATCHING_VERTEX,
        fragmentShader: HATCHING_FRAGMENT,
        uniforms: {},
      });
      return { atom: mkHatch(), bond: mkHatch(), supportsPhysical: false };
    }
  }
}

// --- Surface colormap uniforms for per-pixel property coloring ---

export interface SurfaceColormapUniforms {
  uColormap: { value: THREE.DataTexture | null };
  uRangeMin: { value: number };
  uRangeMax: { value: number };
  uMidpoint: { value: number };
  uDiverging: { value: boolean };
}

export function createSurfaceColormapUniforms(): SurfaceColormapUniforms {
  return {
    uColormap: { value: null },
    uRangeMin: { value: 0 },
    uRangeMax: { value: 1 },
    uMidpoint: { value: 0 },
    uDiverging: { value: false },
  };
}

/** GLSL colormap lookup function — shared between shader injection and custom surface shaders */
const COLORMAP_LOOKUP_GLSL = /* glsl */ `
vec3 colormapLookup(float scalar) {
  float t;
  if (uDiverging) {
    float belowSpan = uMidpoint - uRangeMin;
    float aboveSpan = uRangeMax - uMidpoint;
    if (scalar <= uMidpoint) {
      t = belowSpan > 0.0 ? 0.5 * clamp((scalar - uRangeMin) / belowSpan, 0.0, 1.0) : 0.5;
    } else {
      t = 0.5 + (aboveSpan > 0.0 ? 0.5 * clamp((scalar - uMidpoint) / aboveSpan, 0.0, 1.0) : 0.5);
    }
  } else {
    float rng = uRangeMax - uRangeMin;
    t = rng > 0.0 ? clamp((scalar - uRangeMin) / rng, 0.0, 1.0) : 0.5;
  }
  return texture2D(uColormap, vec2(t, 0.5)).rgb;
}
`;

/**
 * Patch a built-in Three.js material (Standard, Toon, Physical, Matcap) to sample
 * a 1D colormap texture per-pixel using an interpolated scalar attribute.
 */
function patchSurfaceColormap(material: THREE.Material, uniforms: SurfaceColormapUniforms): void {
  const mat = material as any;
  const baseKey = mat.type ?? "unknown";
  mat.customProgramCacheKey = () => `surface-cmap-${baseKey}`;
  mat.onBeforeCompile = (shader: any) => {
    Object.assign(shader.uniforms, uniforms);

    shader.vertexShader = shader.vertexShader.replace(
      "#include <common>",
      `#include <common>
       attribute float aScalar;
       varying float vScalar;`,
    );
    shader.vertexShader = shader.vertexShader.replace(
      "#include <begin_vertex>",
      `#include <begin_vertex>
       vScalar = aScalar;`,
    );

    shader.fragmentShader = shader.fragmentShader.replace(
      "#include <common>",
      `#include <common>
       varying float vScalar;
       uniform sampler2D uColormap;
       uniform float uRangeMin;
       uniform float uRangeMax;
       uniform float uMidpoint;
       uniform bool uDiverging;
       ${COLORMAP_LOOKUP_GLSL}`,
    );

    shader.fragmentShader = shader.fragmentShader.replace(
      "#include <color_fragment>",
      `diffuseColor.rgb = colormapLookup(vScalar);`,
    );
  };
}

// --- Custom surface shaders (Gooch + Hatching) with colormap support ---

const CUSTOM_SURFACE_VERTEX = /* glsl */ `
varying vec3 vNormal;
varying vec3 vViewPosition;
varying float vScalar;

#ifdef USE_COLORMAP
attribute float aScalar;
#endif

void main() {
  #ifdef USE_COLORMAP
    vScalar = aScalar;
  #else
    vScalar = 0.0;
  #endif

  vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
  vViewPosition = -mvPosition.xyz;
  vNormal = normalize(normalMatrix * normal);
  gl_Position = projectionMatrix * mvPosition;
}
`;

const GOOCH_SURFACE_FRAGMENT = /* glsl */ `
varying vec3 vNormal;
varying vec3 vViewPosition;
varying float vScalar;

uniform float uOpacity;
uniform vec3 uBaseColor;

#ifdef USE_COLORMAP
uniform sampler2D uColormap;
uniform float uRangeMin;
uniform float uRangeMax;
uniform float uMidpoint;
uniform bool uDiverging;
${COLORMAP_LOOKUP_GLSL}
#endif

void main() {
  vec3 surfaceColor;
  #ifdef USE_COLORMAP
    surfaceColor = colormapLookup(vScalar);
  #else
    surfaceColor = uBaseColor;
  #endif

  vec3 normal = normalize(vNormal);
  vec3 viewDir = normalize(vViewPosition);
  vec3 lightDir = normalize(vec3(0.5, 1.0, 0.8));

  float NdotL = dot(normal, lightDir);
  float t = (NdotL + 1.0) * 0.5;

  vec3 coolColor = surfaceColor * 0.3 + vec3(0.0, 0.0, 0.15);
  vec3 warmColor = surfaceColor * 0.95 + vec3(0.05, 0.02, 0.0);

  vec3 gooch = mix(coolColor, warmColor, t);

  vec3 halfDir = normalize(lightDir + viewDir);
  float spec = pow(max(dot(normal, halfDir), 0.0), 64.0);

  float edgeFactor = 1.0 - max(dot(normalize(vNormal), viewDir), 0.0);
  float edge = smoothstep(0.2, 0.6, edgeFactor);

  vec3 color = gooch + vec3(1.0) * spec * 0.3;
  color = mix(color, color * 0.4, edge * 0.25);

  gl_FragColor = vec4(color, uOpacity);
}
`;

const HATCHING_SURFACE_FRAGMENT = /* glsl */ `
varying vec3 vNormal;
varying vec3 vViewPosition;
varying float vScalar;

uniform float uOpacity;
uniform vec3 uBaseColor;

#ifdef USE_COLORMAP
uniform sampler2D uColormap;
uniform float uRangeMin;
uniform float uRangeMax;
uniform float uMidpoint;
uniform bool uDiverging;
${COLORMAP_LOOKUP_GLSL}
#endif

${HATCHING_CORE_GLSL}

void main() {
  vec3 surfaceColor;
  #ifdef USE_COLORMAP
    surfaceColor = colormapLookup(vScalar);
  #else
    surfaceColor = uBaseColor;
  #endif

  vec3 normal = normalize(vNormal);
  vec3 lightDir = normalize(vec3(0.5, 1.0, 0.8));
  float shade = dot(normal, lightDir) * 0.5 + 0.5;
  shade = shade * 0.8 + 0.2;

  vec3 color = applyHatching(surfaceColor, shade);
  gl_FragColor = vec4(color, uOpacity);
}
`;

/** Helper to create a custom surface ShaderMaterial (Gooch or Hatching) with optional colormap */
function createCustomSurfaceMaterial(
  fragmentShader: string,
  cmapUniforms: SurfaceColormapUniforms | undefined,
  opacity: number,
  solidColor?: THREE.Color,
): THREE.ShaderMaterial {
  const defines: Record<string, string> = {};
  const uniforms: Record<string, { value: unknown }> = {
    uOpacity: { value: opacity },
    uBaseColor: { value: solidColor ?? new THREE.Color(0.6, 0.6, 0.8) },
  };
  if (cmapUniforms) {
    defines.USE_COLORMAP = "";
    Object.assign(uniforms, cmapUniforms);
  }
  return new THREE.ShaderMaterial({
    vertexShader: CUSTOM_SURFACE_VERTEX,
    fragmentShader,
    side: THREE.DoubleSide,
    transparent: true,
    depthWrite: true,
    depthTest: true,
    defines,
    uniforms,
  });
}

/**
 * Create a surface material for a given style.
 *
 * For property-colored surfaces, pass `colormapUniforms` — the material will
 * sample a 1D colormap texture per-pixel using an interpolated scalar attribute.
 *
 * For solid-colored surfaces (e.g. orbital isosurfaces), pass `solidColor`.
 */
export function createSurfaceMaterial(
  style: MaterialStyleType,
  options?: {
    colormapUniforms?: SurfaceColormapUniforms;
    opacity?: number;
    solidColor?: THREE.Color;
  },
): THREE.Material {
  const opacity = options?.opacity ?? 1.0;
  const cmapUniforms = options?.colormapUniforms;

  switch (style) {
    case "standard": {
      const mat = new THREE.MeshStandardMaterial({
        color: options?.solidColor ?? 0xffffff,
        side: THREE.DoubleSide,
        transparent: true,
        depthWrite: true,
        depthTest: true,
        opacity,
        roughness: 0.4,
        metalness: 0.0,
      });
      if (cmapUniforms) patchSurfaceColormap(mat, cmapUniforms);
      return mat;
    }

    case "toon": {
      const mat = new THREE.MeshToonMaterial({
        color: options?.solidColor ?? 0xffffff,
        side: THREE.DoubleSide,
        transparent: true,
        depthWrite: true,
        depthTest: true,
        opacity,
        gradientMap: gradientMap(),
      });
      if (cmapUniforms) patchSurfaceColormap(mat, cmapUniforms);
      return mat;
    }

    case "gooch":
      return createCustomSurfaceMaterial(GOOCH_SURFACE_FRAGMENT, cmapUniforms, opacity, options?.solidColor);

    case "iridescent": {
      const mat = new THREE.MeshPhysicalMaterial({
        color: options?.solidColor ?? 0xffffff,
        side: THREE.DoubleSide,
        transparent: true,
        depthWrite: true,
        depthTest: true,
        opacity,
        roughness: 0.15,
        metalness: 0.0,
        clearcoat: 0.6,
        clearcoatRoughness: 0.1,
        iridescence: 0.8,
        iridescenceIOR: 1.4,
        iridescenceThicknessRange: [200, 500],
      });
      if (cmapUniforms) patchSurfaceColormap(mat, cmapUniforms);
      return mat;
    }

    case "matcap": {
      const mat = new THREE.MeshMatcapMaterial({
        matcap: matcapTexture(),
        color: options?.solidColor ?? 0xffffff,
        side: THREE.DoubleSide,
        transparent: true,
        depthWrite: true,
        depthTest: true,
        opacity,
      });
      if (cmapUniforms) patchSurfaceColormap(mat, cmapUniforms);
      return mat;
    }

    case "hatching":
      return createCustomSurfaceMaterial(HATCHING_SURFACE_FRAGMENT, cmapUniforms, opacity, options?.solidColor);
  }
}

/**
 * Update material physical properties (roughness/metalness).
 * No-op for material types that don't support these properties.
 */
export function updateStyledMaterial(materials: StyledMaterials, roughness: number, metalness: number): void {
  if (!materials.supportsPhysical) return;
  const atom = materials.atom as THREE.MeshStandardMaterial;
  const bond = materials.bond as THREE.MeshStandardMaterial;
  atom.roughness = roughness;
  atom.metalness = metalness;
  bond.roughness = roughness + 0.15;
  bond.metalness = metalness;
}
