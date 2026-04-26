import { getElementByNumber } from "./elements";

export { JMOL_COLORS_255 } from "./jmolColors";

/**
 * Get the active element colour as [r, g, b] in 0-1 sRGB range.
 * Reflects any user overrides applied via the periodic-table editor.
 * When using with Three.js Color.setRGB(), pass THREE.SRGBColorSpace
 * as the colorSpace argument so the values are correctly linearized.
 */
export function getElementColor(atomicNumber: number): [number, number, number] {
  return getElementByNumber(atomicNumber).color;
}

/** Get the active element colour as a hex integer (e.g., 0xFF0D0D for oxygen). */
export function getElementColorHex(atomicNumber: number): number {
  const [r, g, b] = getElementByNumber(atomicNumber).color;
  return (Math.round(r * 255) << 16) | (Math.round(g * 255) << 8) | Math.round(b * 255);
}
