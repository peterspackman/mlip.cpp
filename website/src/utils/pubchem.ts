// PubChem API utilities for fetching molecular structures

// Convert SDF/MOL format to XYZ
function sdfToXyz(sdf: string): string {
  const lines = sdf.split('\n')

  // Find the counts line (4th line in V2000, contains atom/bond counts)
  // Format: aaabbblllfffcccsssxxxrrrpppiiimmmvvvvvv
  const countsLineIdx = 3
  const countsLine = lines[countsLineIdx]
  const numAtoms = parseInt(countsLine.substring(0, 3).trim())

  if (isNaN(numAtoms) || numAtoms <= 0) {
    throw new Error('Invalid SDF format: could not parse atom count')
  }

  const atoms: { symbol: string, x: number, y: number, z: number }[] = []

  // Parse atom block (starts at line 4, 0-indexed)
  for (let i = 0; i < numAtoms; i++) {
    const line = lines[countsLineIdx + 1 + i]
    if (!line) continue

    // V2000 format: x(10) y(10) z(10) symbol(3) ...
    const x = parseFloat(line.substring(0, 10).trim())
    const y = parseFloat(line.substring(10, 20).trim())
    const z = parseFloat(line.substring(20, 30).trim())
    const symbol = line.substring(31, 34).trim()

    if (!isNaN(x) && !isNaN(y) && !isNaN(z) && symbol) {
      atoms.push({ symbol, x, y, z })
    }
  }

  if (atoms.length === 0) {
    throw new Error('No atoms found in SDF file')
  }

  // Build XYZ string
  let xyz = `${atoms.length}\n`
  xyz += `From PubChem\n`
  for (const atom of atoms) {
    xyz += `${atom.symbol}    ${atom.x.toFixed(6)}    ${atom.y.toFixed(6)}    ${atom.z.toFixed(6)}\n`
  }

  return xyz
}

/**
 * Fetch 3D structure from PubChem by compound name
 * @param name Compound name (e.g., "aspirin", "caffeine", "paracetamol")
 * @returns XYZ format string
 */
export async function fetchFromPubChem(name: string): Promise<string> {
  // First, get the CID from the compound name
  const searchUrl = `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/${encodeURIComponent(name)}/cids/JSON`
  const searchResponse = await fetch(searchUrl)

  if (!searchResponse.ok) {
    throw new Error(`Compound "${name}" not found on PubChem`)
  }

  const searchData = await searchResponse.json()
  const cid = searchData.IdentifierList?.CID?.[0]

  if (!cid) {
    throw new Error(`Could not find CID for "${name}"`)
  }

  // Fetch the 3D SDF
  const sdfUrl = `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/${cid}/SDF?record_type=3d`
  const sdfResponse = await fetch(sdfUrl)

  if (!sdfResponse.ok) {
    throw new Error(`3D structure not available for "${name}"`)
  }

  const sdf = await sdfResponse.text()
  return sdfToXyz(sdf)
}
