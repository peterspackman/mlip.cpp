// Element data from OCC (occ/core/element.h)
// Contains atomic number, symbol, name, covalent radius, van der Waals radius, and mass

export interface ElementData {
  atomicNumber: number
  symbol: string
  name: string
  covalentRadius: number  // Angstroms
  vdwRadius: number       // Angstroms
  mass: number            // atomic mass units
}

// Element data indexed by atomic number (0 = dummy)
export const ELEMENTS: ElementData[] = [
  { atomicNumber: 0, symbol: 'Xx', name: 'Dummy', covalentRadius: 0.0, vdwRadius: 0.0, mass: 0.0 },
  { atomicNumber: 1, symbol: 'H', name: 'hydrogen', covalentRadius: 0.23, vdwRadius: 1.20, mass: 1.00794 },
  { atomicNumber: 2, symbol: 'He', name: 'helium', covalentRadius: 1.50, vdwRadius: 1.40, mass: 4.002602 },
  { atomicNumber: 3, symbol: 'Li', name: 'lithium', covalentRadius: 1.28, vdwRadius: 1.82, mass: 6.941 },
  { atomicNumber: 4, symbol: 'Be', name: 'beryllium', covalentRadius: 0.96, vdwRadius: 1.53, mass: 9.012182 },
  { atomicNumber: 5, symbol: 'B', name: 'boron', covalentRadius: 0.83, vdwRadius: 1.92, mass: 10.811 },
  { atomicNumber: 6, symbol: 'C', name: 'carbon', covalentRadius: 0.68, vdwRadius: 1.70, mass: 12.0107 },
  { atomicNumber: 7, symbol: 'N', name: 'nitrogen', covalentRadius: 0.68, vdwRadius: 1.55, mass: 14.0067 },
  { atomicNumber: 8, symbol: 'O', name: 'oxygen', covalentRadius: 0.68, vdwRadius: 1.52, mass: 15.9994 },
  { atomicNumber: 9, symbol: 'F', name: 'fluorine', covalentRadius: 0.64, vdwRadius: 1.47, mass: 18.998403 },
  { atomicNumber: 10, symbol: 'Ne', name: 'neon', covalentRadius: 1.50, vdwRadius: 1.54, mass: 20.1797 },
  { atomicNumber: 11, symbol: 'Na', name: 'sodium', covalentRadius: 1.66, vdwRadius: 2.27, mass: 22.98977 },
  { atomicNumber: 12, symbol: 'Mg', name: 'magnesium', covalentRadius: 1.41, vdwRadius: 1.73, mass: 24.305 },
  { atomicNumber: 13, symbol: 'Al', name: 'aluminium', covalentRadius: 1.21, vdwRadius: 1.84, mass: 26.981538 },
  { atomicNumber: 14, symbol: 'Si', name: 'silicon', covalentRadius: 1.20, vdwRadius: 2.10, mass: 28.0855 },
  { atomicNumber: 15, symbol: 'P', name: 'phosphorus', covalentRadius: 1.05, vdwRadius: 1.80, mass: 30.973761 },
  { atomicNumber: 16, symbol: 'S', name: 'sulfur', covalentRadius: 1.02, vdwRadius: 1.80, mass: 32.065 },
  { atomicNumber: 17, symbol: 'Cl', name: 'chlorine', covalentRadius: 0.99, vdwRadius: 1.75, mass: 35.453 },
  { atomicNumber: 18, symbol: 'Ar', name: 'argon', covalentRadius: 1.51, vdwRadius: 1.88, mass: 39.948 },
  { atomicNumber: 19, symbol: 'K', name: 'potassium', covalentRadius: 2.03, vdwRadius: 2.75, mass: 39.0983 },
  { atomicNumber: 20, symbol: 'Ca', name: 'calcium', covalentRadius: 1.76, vdwRadius: 2.31, mass: 40.078 },
  { atomicNumber: 21, symbol: 'Sc', name: 'scandium', covalentRadius: 1.70, vdwRadius: 2.16, mass: 44.95591 },
  { atomicNumber: 22, symbol: 'Ti', name: 'titanium', covalentRadius: 1.60, vdwRadius: 1.87, mass: 47.867 },
  { atomicNumber: 23, symbol: 'V', name: 'vanadium', covalentRadius: 1.53, vdwRadius: 1.79, mass: 50.9415 },
  { atomicNumber: 24, symbol: 'Cr', name: 'chromium', covalentRadius: 1.39, vdwRadius: 1.89, mass: 51.9961 },
  { atomicNumber: 25, symbol: 'Mn', name: 'manganese', covalentRadius: 1.61, vdwRadius: 1.97, mass: 54.938049 },
  { atomicNumber: 26, symbol: 'Fe', name: 'iron', covalentRadius: 1.52, vdwRadius: 1.94, mass: 55.845 },
  { atomicNumber: 27, symbol: 'Co', name: 'cobalt', covalentRadius: 1.26, vdwRadius: 1.92, mass: 58.9332 },
  { atomicNumber: 28, symbol: 'Ni', name: 'nickel', covalentRadius: 1.24, vdwRadius: 1.84, mass: 58.6934 },
  { atomicNumber: 29, symbol: 'Cu', name: 'copper', covalentRadius: 1.32, vdwRadius: 1.86, mass: 63.546 },
  { atomicNumber: 30, symbol: 'Zn', name: 'zinc', covalentRadius: 1.22, vdwRadius: 2.10, mass: 65.409 },
  { atomicNumber: 31, symbol: 'Ga', name: 'gallium', covalentRadius: 1.22, vdwRadius: 1.87, mass: 69.723 },
  { atomicNumber: 32, symbol: 'Ge', name: 'germanium', covalentRadius: 1.17, vdwRadius: 2.11, mass: 72.64 },
  { atomicNumber: 33, symbol: 'As', name: 'arsenic', covalentRadius: 1.21, vdwRadius: 1.85, mass: 74.9216 },
  { atomicNumber: 34, symbol: 'Se', name: 'selenium', covalentRadius: 1.22, vdwRadius: 1.90, mass: 78.96 },
  { atomicNumber: 35, symbol: 'Br', name: 'bromine', covalentRadius: 1.21, vdwRadius: 1.85, mass: 79.904 },
  { atomicNumber: 36, symbol: 'Kr', name: 'krypton', covalentRadius: 1.50, vdwRadius: 2.02, mass: 83.798 },
  { atomicNumber: 37, symbol: 'Rb', name: 'rubidium', covalentRadius: 2.20, vdwRadius: 3.03, mass: 85.4678 },
  { atomicNumber: 38, symbol: 'Sr', name: 'strontium', covalentRadius: 1.95, vdwRadius: 2.49, mass: 87.62 },
  { atomicNumber: 39, symbol: 'Y', name: 'yttrium', covalentRadius: 1.90, vdwRadius: 2.19, mass: 88.90585 },
  { atomicNumber: 40, symbol: 'Zr', name: 'zirconium', covalentRadius: 1.75, vdwRadius: 1.86, mass: 91.224 },
  { atomicNumber: 41, symbol: 'Nb', name: 'niobium', covalentRadius: 1.64, vdwRadius: 2.07, mass: 92.90638 },
  { atomicNumber: 42, symbol: 'Mo', name: 'molybdenum', covalentRadius: 1.54, vdwRadius: 2.09, mass: 95.94 },
  { atomicNumber: 43, symbol: 'Tc', name: 'technetium', covalentRadius: 1.47, vdwRadius: 2.09, mass: 98.0 },
  { atomicNumber: 44, symbol: 'Ru', name: 'ruthenium', covalentRadius: 1.46, vdwRadius: 2.07, mass: 101.07 },
  { atomicNumber: 45, symbol: 'Rh', name: 'rhodium', covalentRadius: 1.45, vdwRadius: 1.95, mass: 102.9055 },
  { atomicNumber: 46, symbol: 'Pd', name: 'palladium', covalentRadius: 1.39, vdwRadius: 2.02, mass: 106.42 },
  { atomicNumber: 47, symbol: 'Ag', name: 'silver', covalentRadius: 1.45, vdwRadius: 2.03, mass: 107.8682 },
  { atomicNumber: 48, symbol: 'Cd', name: 'cadmium', covalentRadius: 1.44, vdwRadius: 2.30, mass: 112.411 },
  { atomicNumber: 49, symbol: 'In', name: 'indium', covalentRadius: 1.42, vdwRadius: 1.93, mass: 114.818 },
  { atomicNumber: 50, symbol: 'Sn', name: 'tin', covalentRadius: 1.39, vdwRadius: 2.17, mass: 118.71 },
  { atomicNumber: 51, symbol: 'Sb', name: 'antimony', covalentRadius: 1.39, vdwRadius: 2.06, mass: 121.76 },
  { atomicNumber: 52, symbol: 'Te', name: 'tellurium', covalentRadius: 1.47, vdwRadius: 2.06, mass: 127.6 },
  { atomicNumber: 53, symbol: 'I', name: 'iodine', covalentRadius: 1.40, vdwRadius: 1.98, mass: 126.90447 },
  { atomicNumber: 54, symbol: 'Xe', name: 'xenon', covalentRadius: 1.50, vdwRadius: 2.16, mass: 131.293 },
  { atomicNumber: 55, symbol: 'Cs', name: 'caesium', covalentRadius: 2.44, vdwRadius: 3.43, mass: 132.90545 },
  { atomicNumber: 56, symbol: 'Ba', name: 'barium', covalentRadius: 2.15, vdwRadius: 2.68, mass: 137.327 },
  { atomicNumber: 57, symbol: 'La', name: 'lanthanum', covalentRadius: 2.07, vdwRadius: 2.40, mass: 138.9055 },
  { atomicNumber: 58, symbol: 'Ce', name: 'cerium', covalentRadius: 2.04, vdwRadius: 2.35, mass: 140.116 },
  { atomicNumber: 59, symbol: 'Pr', name: 'praseodymium', covalentRadius: 2.39, vdwRadius: 2.00, mass: 140.90765 },
  { atomicNumber: 60, symbol: 'Nd', name: 'neodymium', covalentRadius: 2.01, vdwRadius: 2.29, mass: 144.24 },
  { atomicNumber: 61, symbol: 'Pm', name: 'promethium', covalentRadius: 1.99, vdwRadius: 2.36, mass: 145.0 },
  { atomicNumber: 62, symbol: 'Sm', name: 'samarium', covalentRadius: 1.98, vdwRadius: 2.29, mass: 150.36 },
  { atomicNumber: 63, symbol: 'Eu', name: 'europium', covalentRadius: 1.98, vdwRadius: 2.33, mass: 151.964 },
  { atomicNumber: 64, symbol: 'Gd', name: 'gadolinium', covalentRadius: 1.96, vdwRadius: 2.37, mass: 157.25 },
  { atomicNumber: 65, symbol: 'Tb', name: 'terbium', covalentRadius: 1.94, vdwRadius: 2.21, mass: 158.92534 },
  { atomicNumber: 66, symbol: 'Dy', name: 'dysprosium', covalentRadius: 1.92, vdwRadius: 2.29, mass: 162.5 },
  { atomicNumber: 67, symbol: 'Ho', name: 'holmium', covalentRadius: 1.92, vdwRadius: 2.16, mass: 164.93032 },
  { atomicNumber: 68, symbol: 'Er', name: 'erbium', covalentRadius: 1.89, vdwRadius: 2.35, mass: 167.259 },
  { atomicNumber: 69, symbol: 'Tm', name: 'thulium', covalentRadius: 1.90, vdwRadius: 2.27, mass: 168.93421 },
  { atomicNumber: 70, symbol: 'Yb', name: 'ytterbium', covalentRadius: 1.87, vdwRadius: 2.42, mass: 173.04 },
  { atomicNumber: 71, symbol: 'Lu', name: 'lutetium', covalentRadius: 1.87, vdwRadius: 2.21, mass: 174.967 },
  { atomicNumber: 72, symbol: 'Hf', name: 'hafnium', covalentRadius: 1.75, vdwRadius: 2.12, mass: 178.49 },
  { atomicNumber: 73, symbol: 'Ta', name: 'tantalum', covalentRadius: 1.70, vdwRadius: 2.17, mass: 180.9479 },
  { atomicNumber: 74, symbol: 'W', name: 'tungsten', covalentRadius: 1.62, vdwRadius: 2.10, mass: 183.84 },
  { atomicNumber: 75, symbol: 'Re', name: 'rhenium', covalentRadius: 1.51, vdwRadius: 2.17, mass: 186.207 },
  { atomicNumber: 76, symbol: 'Os', name: 'osmium', covalentRadius: 1.44, vdwRadius: 2.16, mass: 190.23 },
  { atomicNumber: 77, symbol: 'Ir', name: 'iridium', covalentRadius: 1.41, vdwRadius: 2.02, mass: 192.217 },
  { atomicNumber: 78, symbol: 'Pt', name: 'platinum', covalentRadius: 1.36, vdwRadius: 2.09, mass: 195.078 },
  { atomicNumber: 79, symbol: 'Au', name: 'gold', covalentRadius: 1.50, vdwRadius: 2.17, mass: 196.96655 },
  { atomicNumber: 80, symbol: 'Hg', name: 'mercury', covalentRadius: 1.32, vdwRadius: 2.09, mass: 200.59 },
  { atomicNumber: 81, symbol: 'Tl', name: 'thallium', covalentRadius: 1.45, vdwRadius: 1.96, mass: 204.3833 },
  { atomicNumber: 82, symbol: 'Pb', name: 'lead', covalentRadius: 1.46, vdwRadius: 2.02, mass: 207.2 },
  { atomicNumber: 83, symbol: 'Bi', name: 'bismuth', covalentRadius: 1.48, vdwRadius: 2.07, mass: 208.98038 },
  { atomicNumber: 84, symbol: 'Po', name: 'polonium', covalentRadius: 1.40, vdwRadius: 1.97, mass: 290.0 },
  { atomicNumber: 85, symbol: 'At', name: 'astatine', covalentRadius: 1.21, vdwRadius: 2.02, mass: 210.0 },
  { atomicNumber: 86, symbol: 'Rn', name: 'radon', covalentRadius: 1.50, vdwRadius: 2.20, mass: 222.0 },
  { atomicNumber: 87, symbol: 'Fr', name: 'francium', covalentRadius: 2.60, vdwRadius: 3.48, mass: 223.0 },
  { atomicNumber: 88, symbol: 'Ra', name: 'radium', covalentRadius: 2.21, vdwRadius: 2.83, mass: 226.0 },
  { atomicNumber: 89, symbol: 'Ac', name: 'actinium', covalentRadius: 2.15, vdwRadius: 2.60, mass: 227.0 },
  { atomicNumber: 90, symbol: 'Th', name: 'thorium', covalentRadius: 2.06, vdwRadius: 2.37, mass: 232.0381 },
  { atomicNumber: 91, symbol: 'Pa', name: 'protactinium', covalentRadius: 2.43, vdwRadius: 2.00, mass: 231.03588 },
  { atomicNumber: 92, symbol: 'U', name: 'uranium', covalentRadius: 1.96, vdwRadius: 2.40, mass: 238.02891 },
  { atomicNumber: 93, symbol: 'Np', name: 'neptunium', covalentRadius: 1.90, vdwRadius: 2.21, mass: 237.0 },
  { atomicNumber: 94, symbol: 'Pu', name: 'plutonium', covalentRadius: 1.87, vdwRadius: 2.43, mass: 244.0 },
  { atomicNumber: 95, symbol: 'Am', name: 'americium', covalentRadius: 1.80, vdwRadius: 2.44, mass: 243.0 },
  { atomicNumber: 96, symbol: 'Cm', name: 'curium', covalentRadius: 1.69, vdwRadius: 2.45, mass: 247.0 },
  { atomicNumber: 97, symbol: 'Bk', name: 'berkelium', covalentRadius: 1.54, vdwRadius: 2.44, mass: 247.0 },
  { atomicNumber: 98, symbol: 'Cf', name: 'californium', covalentRadius: 1.83, vdwRadius: 2.45, mass: 251.0 },
  { atomicNumber: 99, symbol: 'Es', name: 'einsteinium', covalentRadius: 1.50, vdwRadius: 2.45, mass: 252.0 },
  { atomicNumber: 100, symbol: 'Fm', name: 'fermium', covalentRadius: 1.50, vdwRadius: 2.45, mass: 257.0 },
  { atomicNumber: 101, symbol: 'Md', name: 'mendelevium', covalentRadius: 1.50, vdwRadius: 2.46, mass: 258.0 },
  { atomicNumber: 102, symbol: 'No', name: 'nobelium', covalentRadius: 1.50, vdwRadius: 2.46, mass: 259.0 },
  { atomicNumber: 103, symbol: 'Lr', name: 'lawrencium', covalentRadius: 1.50, vdwRadius: 2.00, mass: 262.0 },
]

// Lookup maps for fast access
const symbolToZ: Map<string, number> = new Map()
const nameToZ: Map<string, number> = new Map()

// Build lookup maps
ELEMENTS.forEach(el => {
  symbolToZ.set(el.symbol, el.atomicNumber)
  symbolToZ.set(el.symbol.toLowerCase(), el.atomicNumber)
  symbolToZ.set(el.symbol.toUpperCase(), el.atomicNumber)
  nameToZ.set(el.name.toLowerCase(), el.atomicNumber)
})

// Get element by atomic number
export function getElement(atomicNumber: number): ElementData | undefined {
  if (atomicNumber >= 0 && atomicNumber < ELEMENTS.length) {
    return ELEMENTS[atomicNumber]
  }
  return undefined
}

// Get atomic number from symbol (case insensitive)
export function getAtomicNumber(symbol: string): number {
  return symbolToZ.get(symbol) ?? symbolToZ.get(symbol.trim()) ?? 0
}

// Get atomic number from name (case insensitive)
export function getAtomicNumberFromName(name: string): number {
  return nameToZ.get(name.toLowerCase().trim()) ?? 0
}

// Get symbol from atomic number
export function getSymbol(atomicNumber: number): string {
  return ELEMENTS[atomicNumber]?.symbol ?? 'Xx'
}

// Get covalent radius in Angstroms
export function getCovalentRadius(atomicNumber: number): number {
  return ELEMENTS[atomicNumber]?.covalentRadius ?? 0.77
}

// Get van der Waals radius in Angstroms
export function getVdwRadius(atomicNumber: number): number {
  return ELEMENTS[atomicNumber]?.vdwRadius ?? 1.70
}

// Get atomic mass in atomic mass units
export function getMass(atomicNumber: number): number {
  return ELEMENTS[atomicNumber]?.mass ?? 0.0
}
