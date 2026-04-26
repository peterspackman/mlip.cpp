export interface LightingSettings {
  ambientIntensity: number;
  keyLightIntensity: number;
  fillLightIntensity: number;
  hemisphereIntensity: number;
  hemisphereSkyColor: number;
  hemisphereGroundColor: number;
}

export interface MaterialSettings {
  roughness: number;
  metalness: number;
  clearcoat: number;
}

export const DEFAULT_LIGHTING: LightingSettings = {
  ambientIntensity: 0.3,
  keyLightIntensity: 1.5,
  fillLightIntensity: 0.6,
  hemisphereIntensity: 0.6,
  hemisphereSkyColor: 0xc8d8f0,
  hemisphereGroundColor: 0x504030,
};

export const DEFAULT_MATERIAL: MaterialSettings = {
  roughness: 0.35,
  metalness: 0.0,
  clearcoat: 0.0,
};

/** Global line width in pixels for illustration lines (measurements, contacts, cell edges) */
export const DEFAULT_LINE_WIDTH = 2;
