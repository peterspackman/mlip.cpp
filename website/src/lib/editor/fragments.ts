// Predefined functional-group fragments. Each is described in a local frame
// where:
//   • the attach atom (the one that bonds into the host) sits at the origin
//   • the +X axis is the "outward" direction along which the fragment extends
//     away from the host
// Placement (see store.editReplaceWithFragment) rotates the local frame so
// local +X aligns with the world outward direction (atom-being-replaced minus
// its neighbour), then translates the attach atom onto the anchor position.

export interface FragmentAtom {
  z: number
  pos: [number, number, number]  // Å, local frame
}

export interface Fragment {
  name: string
  symbol: string
  atoms: FragmentAtom[]
  attachIndex: number
}

// Common bond lengths (Å) and angles (degrees).
const OH_LEN  = 0.97, OHX_ANG = 104.5
const NH_LEN  = 1.01, NHX_ANG = 107.0
const CH_LEN  = 1.09, TET_ANG = 109.47
const CN_TRIPLE = 1.16
const CC_DOUBLE = 1.34
const CO_DOUBLE = 1.22

const deg = (d: number) => (d * Math.PI) / 180

/** Position of an atom whose bond to the attach atom makes angle `bondAngle`
 *  with the host-bond direction (= -X), at distance `r`, optionally rotated
 *  by `phi` around the X axis to spread tetrahedrally / trigonally. */
function offset(r: number, bondAngle: number, phi: number): [number, number, number] {
  // Bond from attach to this atom makes (180 - bondAngle) with +X.
  const fromX = deg(180 - bondAngle)
  const x = r * Math.cos(fromX)
  const ringR = r * Math.sin(fromX)
  return [x, ringR * Math.cos(phi), ringR * Math.sin(phi)]
}

export const FRAGMENTS: Record<string, Fragment> = {
  H:  { name: 'hydrogen', symbol: 'H',  atoms: [{ z: 1,  pos: [0, 0, 0] }], attachIndex: 0 },
  F:  { name: 'fluorine', symbol: 'F',  atoms: [{ z: 9,  pos: [0, 0, 0] }], attachIndex: 0 },
  Cl: { name: 'chlorine', symbol: 'Cl', atoms: [{ z: 17, pos: [0, 0, 0] }], attachIndex: 0 },
  Br: { name: 'bromine',  symbol: 'Br', atoms: [{ z: 35, pos: [0, 0, 0] }], attachIndex: 0 },
  I:  { name: 'iodine',   symbol: 'I',  atoms: [{ z: 53, pos: [0, 0, 0] }], attachIndex: 0 },

  OH: {
    name: 'hydroxyl',
    symbol: '-OH',
    atoms: [
      { z: 8, pos: [0, 0, 0] },
      { z: 1, pos: offset(OH_LEN, OHX_ANG, 0) },
    ],
    attachIndex: 0,
  },

  NH2: {
    name: 'amine',
    symbol: '-NH₂',
    atoms: [
      { z: 7, pos: [0, 0, 0] },
      { z: 1, pos: offset(NH_LEN, NHX_ANG, deg(60)) },
      { z: 1, pos: offset(NH_LEN, NHX_ANG, deg(-60)) },
    ],
    attachIndex: 0,
  },

  CH3: {
    name: 'methyl',
    symbol: '-CH₃',
    atoms: [
      { z: 6, pos: [0, 0, 0] },
      { z: 1, pos: offset(CH_LEN, TET_ANG, 0) },
      { z: 1, pos: offset(CH_LEN, TET_ANG, deg(120)) },
      { z: 1, pos: offset(CH_LEN, TET_ANG, deg(240)) },
    ],
    attachIndex: 0,
  },

  CN: {
    name: 'cyano',
    symbol: '-C≡N',
    atoms: [
      { z: 6, pos: [0, 0, 0] },
      { z: 7, pos: [CN_TRIPLE, 0, 0] },
    ],
    attachIndex: 0,
  },

  CHO: {
    name: 'formyl',
    symbol: '-CHO',
    atoms: [
      // sp2 C; substituents at 120° around it.
      { z: 6, pos: [0, 0, 0] },
      // C=O (double bond) at +120° from host (= -X), so at +60° from +X.
      { z: 8, pos: [CO_DOUBLE * Math.cos(deg(60)),  CO_DOUBLE * Math.sin(deg(60)), 0] },
      // C-H at -120° from host, so -60° from +X.
      { z: 1, pos: [CH_LEN    * Math.cos(deg(60)), -CH_LEN    * Math.sin(deg(60)), 0] },
    ],
    attachIndex: 0,
  },

  // Vinyl: -CH=CH2 — sp2 chain, useful for "what does adding a vinyl do".
  Vinyl: {
    name: 'vinyl',
    symbol: '-CH=CH₂',
    atoms: [
      { z: 6, pos: [0, 0, 0] },
      // C=C at +60° from +X (sp2 substituent at 120° from host).
      { z: 6, pos: [CC_DOUBLE * Math.cos(deg(60)), CC_DOUBLE * Math.sin(deg(60)), 0] },
      // H on attach C at -60° from +X.
      { z: 1, pos: [CH_LEN * Math.cos(deg(60)), -CH_LEN * Math.sin(deg(60)), 0] },
      // Two H's on far C — 120° around it, in the same plane.
      // Far-C position: (CC_DOUBLE * 0.5, CC_DOUBLE * 0.866, 0).
      ...(() => {
        const cx = CC_DOUBLE * Math.cos(deg(60))
        const cy = CC_DOUBLE * Math.sin(deg(60))
        // far C's "host" is the attach C, in direction back toward origin.
        // The two new H's are at ±60° from the C=C axis on the far side.
        const ax = -Math.cos(deg(60)), ay = -Math.sin(deg(60))  // unit, far→attach
        // Rotate ax,ay by ±120° to get H directions.
        const cos120 = Math.cos(deg(120)), sin120 = Math.sin(deg(120))
        const h1x = cx + CH_LEN * (ax * cos120 - ay * sin120)
        const h1y = cy + CH_LEN * (ax * sin120 + ay * cos120)
        const h2x = cx + CH_LEN * (ax * cos120 + ay * sin120)
        const h2y = cy + CH_LEN * (-ax * sin120 + ay * cos120)
        return [
          { z: 1, pos: [h1x, h1y, 0] as [number, number, number] },
          { z: 1, pos: [h2x, h2y, 0] as [number, number, number] },
        ]
      })(),
    ],
    attachIndex: 0,
  },
}

/** Display order in the UI palette. */
export const FRAGMENT_ORDER: string[] = [
  'H', 'F', 'Cl', 'Br', 'I',
  'OH', 'NH2', 'CH3', 'CN', 'CHO', 'Vinyl',
]
