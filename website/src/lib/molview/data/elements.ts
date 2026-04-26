export interface ElementData {
  name: string;
  symbol: string;
  atomicNumber: number;
  covalentRadius: number;
  vdwRadius: number;
  mass: number;
  /** sRGB colour, components in 0–1 (Jmol palette by default) */
  color: [number, number, number];
}

/** Subset of element fields editable by the user (mass + identity are immutable). */
export type ElementOverride = Partial<Pick<ElementData, "covalentRadius" | "vdwRadius" | "color">>;

type ElementBase = Omit<ElementData, "color">;
const ELEMENTS: ElementBase[] = [
  { name: "dummy", symbol: "X", atomicNumber: 0, covalentRadius: 0.2, vdwRadius: 1.0, mass: 0 },
  { name: "hydrogen", symbol: "H", atomicNumber: 1, covalentRadius: 0.23, vdwRadius: 1.09, mass: 1.00794 },
  { name: "helium", symbol: "He", atomicNumber: 2, covalentRadius: 1.50, vdwRadius: 1.40, mass: 4.002602 },
  { name: "lithium", symbol: "Li", atomicNumber: 3, covalentRadius: 1.28, vdwRadius: 1.82, mass: 6.941 },
  { name: "beryllium", symbol: "Be", atomicNumber: 4, covalentRadius: 0.96, vdwRadius: 2.00, mass: 9.012182 },
  { name: "boron", symbol: "B", atomicNumber: 5, covalentRadius: 0.83, vdwRadius: 2.00, mass: 10.811 },
  { name: "carbon", symbol: "C", atomicNumber: 6, covalentRadius: 0.68, vdwRadius: 1.70, mass: 12.0107 },
  { name: "nitrogen", symbol: "N", atomicNumber: 7, covalentRadius: 0.68, vdwRadius: 1.55, mass: 14.0067 },
  { name: "oxygen", symbol: "O", atomicNumber: 8, covalentRadius: 0.68, vdwRadius: 1.52, mass: 15.9994 },
  { name: "fluorine", symbol: "F", atomicNumber: 9, covalentRadius: 0.64, vdwRadius: 1.47, mass: 18.998403 },
  { name: "neon", symbol: "Ne", atomicNumber: 10, covalentRadius: 1.50, vdwRadius: 1.54, mass: 20.1797 },
  { name: "sodium", symbol: "Na", atomicNumber: 11, covalentRadius: 1.66, vdwRadius: 2.27, mass: 22.98977 },
  { name: "magnesium", symbol: "Mg", atomicNumber: 12, covalentRadius: 1.41, vdwRadius: 1.73, mass: 24.305 },
  { name: "aluminium", symbol: "Al", atomicNumber: 13, covalentRadius: 1.21, vdwRadius: 2.00, mass: 26.981538 },
  { name: "silicon", symbol: "Si", atomicNumber: 14, covalentRadius: 1.20, vdwRadius: 2.10, mass: 28.0855 },
  { name: "phosphorus", symbol: "P", atomicNumber: 15, covalentRadius: 1.05, vdwRadius: 1.80, mass: 30.973761 },
  { name: "sulfur", symbol: "S", atomicNumber: 16, covalentRadius: 1.02, vdwRadius: 1.80, mass: 32.065 },
  { name: "chlorine", symbol: "Cl", atomicNumber: 17, covalentRadius: 0.99, vdwRadius: 1.75, mass: 35.453 },
  { name: "argon", symbol: "Ar", atomicNumber: 18, covalentRadius: 1.51, vdwRadius: 1.88, mass: 39.948 },
  { name: "potassium", symbol: "K", atomicNumber: 19, covalentRadius: 2.03, vdwRadius: 2.75, mass: 39.0983 },
  { name: "calcium", symbol: "Ca", atomicNumber: 20, covalentRadius: 1.76, vdwRadius: 2.00, mass: 40.078 },
  { name: "scandium", symbol: "Sc", atomicNumber: 21, covalentRadius: 1.70, vdwRadius: 2.00, mass: 44.95591 },
  { name: "titanium", symbol: "Ti", atomicNumber: 22, covalentRadius: 1.60, vdwRadius: 2.00, mass: 47.867 },
  { name: "vanadium", symbol: "V", atomicNumber: 23, covalentRadius: 1.53, vdwRadius: 2.00, mass: 50.9415 },
  { name: "chromium", symbol: "Cr", atomicNumber: 24, covalentRadius: 1.39, vdwRadius: 2.00, mass: 51.9961 },
  { name: "manganese", symbol: "Mn", atomicNumber: 25, covalentRadius: 1.61, vdwRadius: 2.00, mass: 54.938049 },
  { name: "iron", symbol: "Fe", atomicNumber: 26, covalentRadius: 1.52, vdwRadius: 2.00, mass: 55.845 },
  { name: "cobalt", symbol: "Co", atomicNumber: 27, covalentRadius: 1.26, vdwRadius: 2.00, mass: 58.9332 },
  { name: "nickel", symbol: "Ni", atomicNumber: 28, covalentRadius: 1.24, vdwRadius: 1.63, mass: 58.6934 },
  { name: "copper", symbol: "Cu", atomicNumber: 29, covalentRadius: 1.32, vdwRadius: 1.40, mass: 63.546 },
  { name: "zinc", symbol: "Zn", atomicNumber: 30, covalentRadius: 1.22, vdwRadius: 1.39, mass: 65.409 },
  { name: "gallium", symbol: "Ga", atomicNumber: 31, covalentRadius: 1.22, vdwRadius: 1.87, mass: 69.723 },
  { name: "germanium", symbol: "Ge", atomicNumber: 32, covalentRadius: 1.17, vdwRadius: 2.00, mass: 72.64 },
  { name: "arsenic", symbol: "As", atomicNumber: 33, covalentRadius: 1.21, vdwRadius: 1.85, mass: 74.9216 },
  { name: "selenium", symbol: "Se", atomicNumber: 34, covalentRadius: 1.22, vdwRadius: 1.90, mass: 78.96 },
  { name: "bromine", symbol: "Br", atomicNumber: 35, covalentRadius: 1.21, vdwRadius: 1.85, mass: 79.904 },
  { name: "krypton", symbol: "Kr", atomicNumber: 36, covalentRadius: 1.50, vdwRadius: 2.02, mass: 83.798 },
  { name: "rubidium", symbol: "Rb", atomicNumber: 37, covalentRadius: 2.20, vdwRadius: 2.00, mass: 85.4678 },
  { name: "strontium", symbol: "Sr", atomicNumber: 38, covalentRadius: 1.95, vdwRadius: 2.00, mass: 87.62 },
  { name: "yttrium", symbol: "Y", atomicNumber: 39, covalentRadius: 1.90, vdwRadius: 2.00, mass: 88.90585 },
  { name: "zirconium", symbol: "Zr", atomicNumber: 40, covalentRadius: 1.75, vdwRadius: 2.00, mass: 91.224 },
  { name: "niobium", symbol: "Nb", atomicNumber: 41, covalentRadius: 1.64, vdwRadius: 2.00, mass: 92.90638 },
  { name: "molybdenum", symbol: "Mo", atomicNumber: 42, covalentRadius: 1.54, vdwRadius: 2.00, mass: 95.94 },
  { name: "technetium", symbol: "Tc", atomicNumber: 43, covalentRadius: 1.47, vdwRadius: 2.00, mass: 98.0 },
  { name: "ruthenium", symbol: "Ru", atomicNumber: 44, covalentRadius: 1.46, vdwRadius: 2.00, mass: 101.07 },
  { name: "rhodium", symbol: "Rh", atomicNumber: 45, covalentRadius: 1.45, vdwRadius: 2.00, mass: 102.9055 },
  { name: "palladium", symbol: "Pd", atomicNumber: 46, covalentRadius: 1.39, vdwRadius: 1.63, mass: 106.42 },
  { name: "silver", symbol: "Ag", atomicNumber: 47, covalentRadius: 1.45, vdwRadius: 1.72, mass: 107.8682 },
  { name: "cadmium", symbol: "Cd", atomicNumber: 48, covalentRadius: 1.44, vdwRadius: 1.58, mass: 112.411 },
  { name: "indium", symbol: "In", atomicNumber: 49, covalentRadius: 1.42, vdwRadius: 1.93, mass: 114.818 },
  { name: "tin", symbol: "Sn", atomicNumber: 50, covalentRadius: 1.39, vdwRadius: 2.17, mass: 118.71 },
  { name: "antimony", symbol: "Sb", atomicNumber: 51, covalentRadius: 1.39, vdwRadius: 2.00, mass: 121.76 },
  { name: "tellurium", symbol: "Te", atomicNumber: 52, covalentRadius: 1.47, vdwRadius: 2.06, mass: 127.6 },
  { name: "iodine", symbol: "I", atomicNumber: 53, covalentRadius: 1.40, vdwRadius: 1.98, mass: 126.90447 },
  { name: "xenon", symbol: "Xe", atomicNumber: 54, covalentRadius: 1.50, vdwRadius: 2.16, mass: 131.293 },
  { name: "caesium", symbol: "Cs", atomicNumber: 55, covalentRadius: 2.44, vdwRadius: 2.00, mass: 132.90545 },
  { name: "barium", symbol: "Ba", atomicNumber: 56, covalentRadius: 2.15, vdwRadius: 2.00, mass: 137.327 },
  { name: "lanthanum", symbol: "La", atomicNumber: 57, covalentRadius: 2.07, vdwRadius: 2.00, mass: 138.9055 },
  { name: "cerium", symbol: "Ce", atomicNumber: 58, covalentRadius: 2.04, vdwRadius: 2.00, mass: 140.116 },
  { name: "praseodymium", symbol: "Pr", atomicNumber: 59, covalentRadius: 2.03, vdwRadius: 2.00, mass: 140.90765 },
  { name: "neodymium", symbol: "Nd", atomicNumber: 60, covalentRadius: 2.01, vdwRadius: 2.00, mass: 144.24 },
  { name: "promethium", symbol: "Pm", atomicNumber: 61, covalentRadius: 1.99, vdwRadius: 2.00, mass: 145.0 },
  { name: "samarium", symbol: "Sm", atomicNumber: 62, covalentRadius: 1.98, vdwRadius: 2.00, mass: 150.36 },
  { name: "europium", symbol: "Eu", atomicNumber: 63, covalentRadius: 1.98, vdwRadius: 2.00, mass: 151.964 },
  { name: "gadolinium", symbol: "Gd", atomicNumber: 64, covalentRadius: 1.96, vdwRadius: 2.00, mass: 157.25 },
  { name: "terbium", symbol: "Tb", atomicNumber: 65, covalentRadius: 1.94, vdwRadius: 2.00, mass: 158.92534 },
  { name: "dysprosium", symbol: "Dy", atomicNumber: 66, covalentRadius: 1.92, vdwRadius: 2.00, mass: 162.5 },
  { name: "holmium", symbol: "Ho", atomicNumber: 67, covalentRadius: 1.92, vdwRadius: 2.00, mass: 164.93032 },
  { name: "erbium", symbol: "Er", atomicNumber: 68, covalentRadius: 1.89, vdwRadius: 2.00, mass: 167.259 },
  { name: "thulium", symbol: "Tm", atomicNumber: 69, covalentRadius: 1.90, vdwRadius: 2.00, mass: 168.93421 },
  { name: "ytterbium", symbol: "Yb", atomicNumber: 70, covalentRadius: 1.87, vdwRadius: 2.00, mass: 173.04 },
  { name: "lutetium", symbol: "Lu", atomicNumber: 71, covalentRadius: 1.87, vdwRadius: 2.00, mass: 174.967 },
  { name: "hafnium", symbol: "Hf", atomicNumber: 72, covalentRadius: 1.75, vdwRadius: 2.00, mass: 178.49 },
  { name: "tantalum", symbol: "Ta", atomicNumber: 73, covalentRadius: 1.70, vdwRadius: 2.00, mass: 180.9479 },
  { name: "tungsten", symbol: "W", atomicNumber: 74, covalentRadius: 1.62, vdwRadius: 2.00, mass: 183.84 },
  { name: "rhenium", symbol: "Re", atomicNumber: 75, covalentRadius: 1.51, vdwRadius: 2.00, mass: 186.207 },
  { name: "osmium", symbol: "Os", atomicNumber: 76, covalentRadius: 1.44, vdwRadius: 2.00, mass: 190.23 },
  { name: "iridium", symbol: "Ir", atomicNumber: 77, covalentRadius: 1.41, vdwRadius: 2.00, mass: 192.217 },
  { name: "platinum", symbol: "Pt", atomicNumber: 78, covalentRadius: 1.36, vdwRadius: 1.72, mass: 195.078 },
  { name: "gold", symbol: "Au", atomicNumber: 79, covalentRadius: 1.50, vdwRadius: 1.66, mass: 196.96655 },
  { name: "mercury", symbol: "Hg", atomicNumber: 80, covalentRadius: 1.32, vdwRadius: 1.55, mass: 200.59 },
  { name: "thallium", symbol: "Tl", atomicNumber: 81, covalentRadius: 1.45, vdwRadius: 1.96, mass: 204.3833 },
  { name: "lead", symbol: "Pb", atomicNumber: 82, covalentRadius: 1.46, vdwRadius: 2.02, mass: 207.2 },
  { name: "bismuth", symbol: "Bi", atomicNumber: 83, covalentRadius: 1.48, vdwRadius: 2.00, mass: 208.98038 },
  { name: "polonium", symbol: "Po", atomicNumber: 84, covalentRadius: 1.40, vdwRadius: 2.00, mass: 290.0 },
  { name: "astatine", symbol: "At", atomicNumber: 85, covalentRadius: 1.21, vdwRadius: 2.00, mass: 210.0 },
  { name: "radon", symbol: "Rn", atomicNumber: 86, covalentRadius: 1.50, vdwRadius: 2.00, mass: 222.0 },
  { name: "francium", symbol: "Fr", atomicNumber: 87, covalentRadius: 2.60, vdwRadius: 2.00, mass: 223.0 },
  { name: "radium", symbol: "Ra", atomicNumber: 88, covalentRadius: 2.21, vdwRadius: 2.00, mass: 226.0 },
  { name: "actinium", symbol: "Ac", atomicNumber: 89, covalentRadius: 2.15, vdwRadius: 2.00, mass: 227.0 },
  { name: "thorium", symbol: "Th", atomicNumber: 90, covalentRadius: 2.06, vdwRadius: 2.00, mass: 232.0381 },
  { name: "protactinium", symbol: "Pa", atomicNumber: 91, covalentRadius: 2.00, vdwRadius: 2.00, mass: 231.03588 },
  { name: "uranium", symbol: "U", atomicNumber: 92, covalentRadius: 1.96, vdwRadius: 1.86, mass: 238.02891 },
  { name: "neptunium", symbol: "Np", atomicNumber: 93, covalentRadius: 1.90, vdwRadius: 2.00, mass: 237.0 },
  { name: "plutonium", symbol: "Pu", atomicNumber: 94, covalentRadius: 1.87, vdwRadius: 2.00, mass: 244.0 },
  { name: "americium", symbol: "Am", atomicNumber: 95, covalentRadius: 1.80, vdwRadius: 2.00, mass: 243.0 },
  { name: "curium", symbol: "Cm", atomicNumber: 96, covalentRadius: 1.69, vdwRadius: 2.00, mass: 247.0 },
  { name: "berkelium", symbol: "Bk", atomicNumber: 97, covalentRadius: 1.54, vdwRadius: 2.00, mass: 247.0 },
  { name: "californium", symbol: "Cf", atomicNumber: 98, covalentRadius: 1.83, vdwRadius: 2.00, mass: 251.0 },
  { name: "einsteinium", symbol: "Es", atomicNumber: 99, covalentRadius: 1.50, vdwRadius: 2.00, mass: 252.0 },
  { name: "fermium", symbol: "Fm", atomicNumber: 100, covalentRadius: 1.50, vdwRadius: 2.00, mass: 257.0 },
  { name: "mendelevium", symbol: "Md", atomicNumber: 101, covalentRadius: 1.50, vdwRadius: 2.00, mass: 258.0 },
  { name: "nobelium", symbol: "No", atomicNumber: 102, covalentRadius: 1.50, vdwRadius: 2.00, mass: 259.0 },
  { name: "lawrencium", symbol: "Lr", atomicNumber: 103, covalentRadius: 1.50, vdwRadius: 2.00, mass: 262.0 },
];

import { JMOL_COLORS_NORMALIZED } from "./jmolColors";

const _overrides = new Map<number, ElementOverride>();
const _changeListeners = new Set<() => void>();

function _baseColor(z: number): [number, number, number] {
  return JMOL_COLORS_NORMALIZED[z] ?? JMOL_COLORS_NORMALIZED[0];
}

function _merge(base: ElementBase): ElementData {
  const ov = _overrides.get(base.atomicNumber);
  return {
    ...base,
    color: ov?.color ?? _baseColor(base.atomicNumber),
    covalentRadius: ov?.covalentRadius ?? base.covalentRadius,
    vdwRadius: ov?.vdwRadius ?? base.vdwRadius,
  };
}

const symbolMap = new Map<string, ElementBase>();
for (const el of ELEMENTS) {
  symbolMap.set(el.symbol.toLowerCase(), el);
}

export function getElement(symbol: string): ElementData {
  return _merge(symbolMap.get(symbol.toLowerCase()) ?? ELEMENTS[0]);
}

export function getElementByNumber(atomicNumber: number): ElementData {
  return _merge(ELEMENTS[atomicNumber] ?? ELEMENTS[0]);
}

export function symbolToAtomicNumber(symbol: string): number {
  return getElement(symbol).atomicNumber;
}

export function covalentRadius(symbol: string): number {
  return getElement(symbol).covalentRadius;
}

export function vdwRadius(symbol: string): number {
  return getElement(symbol).vdwRadius;
}

/** All element records (merged with current overrides). Z 0..103. */
export function allElements(): ElementData[] {
  return ELEMENTS.map(_merge);
}

/** Default record (no overrides applied) for a single element. */
export function defaultElement(atomicNumber: number): ElementData {
  const base = ELEMENTS[atomicNumber] ?? ELEMENTS[0];
  return { ...base, color: _baseColor(base.atomicNumber) };
}

export function getElementOverrides(): Map<number, ElementOverride> {
  return new Map(_overrides);
}

/** Replace the entire override map (used when restoring from persisted state). */
export function setElementOverrides(map: Map<number, ElementOverride>): void {
  _overrides.clear();
  for (const [z, ov] of map) _overrides.set(z, { ...ov });
  _notify();
}

export function applyElementOverride(atomicNumber: number, partial: ElementOverride): void {
  const cur = _overrides.get(atomicNumber) ?? {};
  _overrides.set(atomicNumber, { ...cur, ...partial });
  _notify();
}

export function resetElementOverride(atomicNumber: number): void {
  if (_overrides.delete(atomicNumber)) _notify();
}

export function resetAllElementOverrides(): void {
  if (_overrides.size === 0) return;
  _overrides.clear();
  _notify();
}

/** Subscribe to override-set changes. Returns an unsubscribe function. */
export function onElementsChanged(listener: () => void): () => void {
  _changeListeners.add(listener);
  return () => _changeListeners.delete(listener);
}

function _notify(): void {
  for (const l of _changeListeners) l();
}
