import { useEffect, useRef, useState, useCallback } from 'react'
import * as NGL from 'ngl'
import './MolecularDynamics.css'

// Sample structures for MD - molecules and crystals
// Crystals use extended XYZ format with Lattice= and pbc= in comment line
const SAMPLE_MOLECULES: Record<string, string> = {
  'Water': `3
Water
O     0.000000     0.000000     0.117489
H     0.000000     0.756950    -0.469957
H     0.000000    -0.756950    -0.469957`,
  'Methane': `5
Methane
C     0.000000     0.000000     0.000000
H     0.629118     0.629118     0.629118
H    -0.629118    -0.629118     0.629118
H    -0.629118     0.629118    -0.629118
H     0.629118    -0.629118    -0.629118`,
  'Ethanol': `9
Ethanol
C    -0.001193    -0.004555     0.009236
C     1.519736    -0.001568    -0.012413
O     2.032422     1.326098    -0.087629
H    -0.394952     1.007606    -0.074891
H    -0.376887    -0.547259    -0.861972
H    -0.435219    -0.483282     0.891082
H     1.894949    -0.539891     0.862637
H     1.898649    -0.518854    -0.898756
H     1.685063     1.800579     0.682628`,
  'Dichloroethane': `8
1,2-Dichloroethane
C     0.000000     0.000000     0.000000
C     1.524000     0.000000     0.000000
Cl   -0.799000     1.524000     0.000000
Cl    2.323000    -1.524000     0.000000
H    -0.360000    -0.514000     0.891000
H    -0.360000    -0.514000    -0.891000
H     1.884000     0.514000     0.891000
H     1.884000     0.514000    -0.891000`,
  'Ethylene Glycol': `10
Ethylene glycol
C     0.000000     0.000000     0.000000
C     1.524000     0.000000     0.000000
O    -0.524000     1.343000     0.000000
O     2.048000    -1.343000     0.000000
H    -0.360000    -0.514000     0.891000
H    -0.360000    -0.514000    -0.891000
H     1.884000     0.514000     0.891000
H     1.884000     0.514000    -0.891000
H    -0.161000     1.861000     0.748000
H     1.685000    -1.861000     0.748000`,
}

// Crystal structures in extended XYZ format (unit cells)
const SAMPLE_CRYSTALS: Record<string, string> = {
  'Silicon': `8
Lattice="5.43 0.0 0.0 0.0 5.43 0.0 0.0 0.0 5.43" pbc="T T T"
Si    0.00000    0.00000    0.00000
Si    2.71500    2.71500    0.00000
Si    2.71500    0.00000    2.71500
Si    0.00000    2.71500    2.71500
Si    1.35750    1.35750    1.35750
Si    4.07250    4.07250    1.35750
Si    4.07250    1.35750    4.07250
Si    1.35750    4.07250    4.07250`,
  'MgO': `2
Lattice="4.212 0.0 0.0 0.0 4.212 0.0 0.0 0.0 4.212" pbc="T T T"
Mg    0.00000    0.00000    0.00000
O     2.10600    2.10600    2.10600`,
  'Urea': `16
Lattice="5.582 0.0 0.0 0.0 5.582 0.0 0.0 0.0 4.686" pbc="T T T"
C     0.00000    2.83100    1.55628
H     1.37587    4.20687    1.32520
H     0.80400    3.63500    0.13205
N     0.81136    3.64236    0.87105
O     0.00000    2.83100    2.82017
H    -1.37587    1.45513    1.32520
H    -0.80400    2.02700    0.13205
N    -0.81136    2.01964    0.87105
C     2.83100    0.00000    3.15972
H     1.45513    1.37587    3.39080
H     2.02700    0.80400    4.58395
N     2.01964    0.81136    3.84495
O     2.83100    0.00000    1.89583
H     4.20687   -1.37587    3.39080
H     3.63500   -0.80400    4.58395
N     3.64236   -0.81136    3.84495`,
}

// Combined for lookup
const SAMPLE_STRUCTURES: Record<string, string> = {
  ...SAMPLE_MOLECULES,
  ...SAMPLE_CRYSTALS,
}

// Covalent radii in Angstroms for bond detection
const COVALENT_RADII: Record<number, number> = {
  1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57,
  12: 1.41, 14: 1.11, 15: 1.07, 16: 1.05, 17: 1.02
}

// Detect bonds based on distance and covalent radii
function detectBonds(positions: number[], atomicNumbers: number[]): [number, number][] {
  const numAtoms = atomicNumbers.length
  const bonds: [number, number][] = []
  const tolerance = 0.4  // Angstroms tolerance

  for (let i = 0; i < numAtoms; i++) {
    const ri = COVALENT_RADII[atomicNumbers[i]] || 0.77
    const xi = positions[i * 3]
    const yi = positions[i * 3 + 1]
    const zi = positions[i * 3 + 2]

    for (let j = i + 1; j < numAtoms; j++) {
      const rj = COVALENT_RADII[atomicNumbers[j]] || 0.77
      const xj = positions[j * 3]
      const yj = positions[j * 3 + 1]
      const zj = positions[j * 3 + 2]

      const dx = xi - xj
      const dy = yi - yj
      const dz = zi - zj
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz)

      // Bond if distance < sum of covalent radii + tolerance
      if (dist < ri + rj + tolerance) {
        bonds.push([i + 1, j + 1])  // 1-indexed for SDF
      }
    }
  }

  return bonds
}

// Generate supercell positions for periodic visualization
// Returns expanded positions and atomic numbers for a 3x3x3 supercell (-1 to +1)
function generateSupercell(
  positions: number[],
  atomicNumbers: number[],
  lattice: { a: number[], b: number[], c: number[] }
): { positions: number[], atomicNumbers: number[] } {
  const numAtoms = atomicNumbers.length
  const supercellPositions: number[] = []
  const supercellAtomicNumbers: number[] = []

  // Generate 3x3x3 supercell (from -1 to +1 in each direction)
  for (let na = -1; na <= 1; na++) {
    for (let nb = -1; nb <= 1; nb++) {
      for (let nc = -1; nc <= 1; nc++) {
        // Translation vector for this cell
        const tx = na * lattice.a[0] + nb * lattice.b[0] + nc * lattice.c[0]
        const ty = na * lattice.a[1] + nb * lattice.b[1] + nc * lattice.c[1]
        const tz = na * lattice.a[2] + nb * lattice.b[2] + nc * lattice.c[2]

        // Add translated atoms
        for (let i = 0; i < numAtoms; i++) {
          supercellPositions.push(
            positions[i * 3] + tx,
            positions[i * 3 + 1] + ty,
            positions[i * 3 + 2] + tz
          )
          supercellAtomicNumbers.push(atomicNumbers[i])
        }
      }
    }
  }

  return { positions: supercellPositions, atomicNumbers: supercellAtomicNumbers }
}

// Convert positions array to SDF/MOL format for NGL (better element support)
function positionsToSdf(positions: number[], atomicNumbers: number[]): string {
  const numAtoms = atomicNumbers.length
  const elements: Record<number, string> = {
    1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 12: 'Mg', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl'
  }

  const bonds = detectBonds(positions, atomicNumbers)

  let sdf = '\n'  // molecule name (blank)
  sdf += '     RDKit          3D\n'  // program/timestamp line
  sdf += '\n'  // comment line

  // Counts line: aaabbblllfffcccsssxxxrrrpppiiimmmvvvvvv
  const atomCount = String(numAtoms).padStart(3)
  const bondCount = String(bonds.length).padStart(3)
  sdf += `${atomCount}${bondCount}  0  0  0  0  0  0  0  0999 V2000\n`

  // Atom block: x, y, z, symbol, mass diff, charge, etc.
  for (let i = 0; i < numAtoms; i++) {
    const element = elements[atomicNumbers[i]] || 'C'
    const x = positions[i * 3].toFixed(4).padStart(10)
    const y = positions[i * 3 + 1].toFixed(4).padStart(10)
    const z = positions[i * 3 + 2].toFixed(4).padStart(10)
    const sym = element.padEnd(3)
    sdf += `${x}${y}${z} ${sym} 0  0  0  0  0  0  0  0  0  0  0  0\n`
  }

  // Bond block: atom1 atom2 type stereo
  for (const [a1, a2] of bonds) {
    sdf += `${String(a1).padStart(3)}${String(a2).padStart(3)}  1  0\n`
  }

  sdf += 'M  END\n'
  return sdf
}

// Parse XYZ to get atomic numbers
function parseXyzAtomicNumbers(xyz: string): number[] {
  const elementToZ: Record<string, number> = {
    'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Mg': 12, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17
  }

  const lines = xyz.trim().split('\n')
  const numAtoms = parseInt(lines[0])
  const atomicNumbers: number[] = []

  for (let i = 0; i < numAtoms; i++) {
    const parts = lines[i + 2].trim().split(/\s+/)
    const element = parts[0]
    atomicNumbers.push(elementToZ[element] || 6)
  }

  return atomicNumbers
}

// Parse lattice vectors from extended XYZ comment line
// Returns null if no lattice info, or {a, b, c} vectors
function parseLattice(xyz: string): { a: number[], b: number[], c: number[] } | null {
  const lines = xyz.trim().split('\n')
  if (lines.length < 2) return null

  const commentLine = lines[1]
  const latticeMatch = commentLine.match(/Lattice="([^"]+)"/)
  if (!latticeMatch) return null

  const values = latticeMatch[1].split(/\s+/).map(v => parseFloat(v))
  if (values.length !== 9) return null

  // Lattice vectors: [a1 a2 a3 b1 b2 b3 c1 c2 c3]
  return {
    a: [values[0], values[1], values[2]],
    b: [values[3], values[4], values[5]],
    c: [values[6], values[7], values[8]]
  }
}

interface MDState {
  isInitialized: boolean
  isLoadingModel: boolean
  isModelLoaded: boolean
  isRunning: boolean
  modelType: string
  error: string
  step: number
  energy: number
  kineticEnergy: number
  temperature: number
  maxForce: number
  msPerStep: number
  optimizationConverged: boolean
}

export default function MolecularDynamics() {
  const containerRef = useRef<HTMLDivElement>(null)
  const stageRef = useRef<NGL.Stage | null>(null)
  const componentRef = useRef<any>(null)
  const unitCellRef = useRef<any>(null)  // NGL shape component for unit cell
  const workerRef = useRef<Worker | null>(null)
  const atomicNumbersRef = useRef<number[]>([])
  const latticeRef = useRef<{ a: number[], b: number[], c: number[] } | null>(null)

  const [state, setState] = useState<MDState>({
    isInitialized: false,
    isLoadingModel: false,
    isModelLoaded: false,
    isRunning: false,
    modelType: '',
    error: '',
    step: 0,
    energy: 0,
    kineticEnergy: 0,
    temperature: 0,
    maxForce: 0,
    msPerStep: 0,
    optimizationConverged: false,
  })

  const lastStepTimeRef = useRef<number>(0)
  const [energyHistory, setEnergyHistory] = useState<number[]>([])

  const [targetTemperature, setTargetTemperature] = useState(300)
  const [timestep, setTimestep] = useState(1.0)
  const [selectedStructure, setSelectedStructure] = useState('Ethanol')
  const [customXyz, setCustomXyz] = useState(SAMPLE_STRUCTURES['Ethanol'])
  const [mode, setMode] = useState<'md' | 'optimize'>('md')
  const [maxOptSteps, setMaxOptSteps] = useState(100)
  const [forceThreshold, setForceThreshold] = useState(0.05)

  // Initialize NGL Stage
  useEffect(() => {
    if (!containerRef.current) return

    const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches
    stageRef.current = new NGL.Stage(containerRef.current, {
      backgroundColor: isDark ? '#1a1a1a' : '#ffffff',
      quality: 'high',
    })

    const handleResize = () => stageRef.current?.handleResize()
    window.addEventListener('resize', handleResize)

    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')
    const handleTheme = (e: MediaQueryListEvent) => {
      stageRef.current?.setParameters({ backgroundColor: e.matches ? '#1a1a1a' : '#ffffff' })
    }
    mediaQuery.addEventListener('change', handleTheme)

    return () => {
      window.removeEventListener('resize', handleResize)
      mediaQuery.removeEventListener('change', handleTheme)
      stageRef.current?.dispose()
    }
  }, [])

  // Initialize worker
  useEffect(() => {
    workerRef.current = new Worker(
      new URL('../workers/mdWorker.ts', import.meta.url),
      { type: 'module' }
    )

    workerRef.current.onmessage = (e) => {
      const msg = e.data

      switch (msg.type) {
        case 'ready':
          // Worker is ready, initialize
          workerRef.current?.postMessage({ type: 'init' })
          break

        case 'initialized':
          setState(s => ({ ...s, isInitialized: true }))
          // Auto-load bundled PET-MAD model
          loadBundledModel()
          break

        case 'modelLoaded':
          setState(s => ({
            ...s,
            isLoadingModel: false,
            isModelLoaded: true,
            modelType: msg.modelType,
          }))
          break

        case 'systemSet':
          // System ready for MD
          break

        case 'mdStep':
          {
            const now = performance.now()
            const msPerStep = lastStepTimeRef.current > 0 ? now - lastStepTimeRef.current : 0
            lastStepTimeRef.current = now
            // For MD, track total energy (potential + kinetic)
            const totalEnergy = msg.energy + msg.kineticEnergy
            setEnergyHistory(h => [...h.slice(-99), totalEnergy])  // Keep last 100 points
            setState(s => ({
              ...s,
              step: s.step + 1,
              energy: msg.energy,
              kineticEnergy: msg.kineticEnergy,
              temperature: msg.temperature,
              msPerStep,
            }))
            // Update visualization
            updateVisualization(msg.positions)
          }
          break

        case 'optStep':
          {
            const now = performance.now()
            const msPerStep = lastStepTimeRef.current > 0 ? now - lastStepTimeRef.current : 0
            lastStepTimeRef.current = now
            // For optimization, track potential energy
            setEnergyHistory(h => [...h.slice(-99), msg.energy])  // Keep last 100 points
            setState(s => ({
              ...s,
              step: msg.step,
              energy: msg.energy,
              maxForce: msg.maxForce,
              msPerStep,
              optimizationConverged: msg.converged,
            }))
            // Update visualization
            updateVisualization(msg.positions)
          }
          break

        case 'started':
          setState(s => ({ ...s, isRunning: true }))
          break

        case 'stopped':
          setState(s => ({ ...s, isRunning: false }))
          break

        case 'error':
          setState(s => ({ ...s, error: msg.message, isRunning: false }))
          break
      }
    }

    return () => {
      workerRef.current?.terminate()
    }
  }, [])

  // Load bundled PET-MAD model
  const loadBundledModel = async () => {
    if (!workerRef.current) return

    setState(s => ({ ...s, isLoadingModel: true }))
    try {
      const response = await fetch(`${import.meta.env.BASE_URL}pet-mad.gguf`)
      if (!response.ok) {
        throw new Error(`Failed to fetch model: ${response.status}`)
      }
      const buffer = await response.arrayBuffer()
      workerRef.current.postMessage(
        { type: 'loadModel', buffer },
        [buffer]
      )
    } catch (err: any) {
      setState(s => ({
        ...s,
        isLoadingModel: false,
        error: `Failed to load bundled model: ${err.message}`
      }))
    }
  }

  // Update visualization with new positions (in-place update, no flashing)
  const updateVisualization = useCallback((positions: number[]) => {
    if (!stageRef.current || !componentRef.current || atomicNumbersRef.current.length === 0) return

    const structure = componentRef.current.structure
    if (!structure || !structure.atomStore) return

    const atomStore = structure.atomStore
    const numAtoms = atomicNumbersRef.current.length

    // For periodic structures, we have 27 copies (3x3x3 supercell)
    const isPeriodic = latticeRef.current !== null
    const expectedAtoms = isPeriodic ? numAtoms * 27 : numAtoms

    // Check if atom count matches
    if (atomStore.count !== expectedAtoms) return

    if (isPeriodic && latticeRef.current) {
      // Update all 27 copies of each atom
      const { a, b, c } = latticeRef.current
      let atomIdx = 0
      for (let na = -1; na <= 1; na++) {
        for (let nb = -1; nb <= 1; nb++) {
          for (let nc = -1; nc <= 1; nc++) {
            const tx = na * a[0] + nb * b[0] + nc * c[0]
            const ty = na * a[1] + nb * b[1] + nc * c[1]
            const tz = na * a[2] + nb * b[2] + nc * c[2]

            for (let i = 0; i < numAtoms; i++) {
              atomStore.x[atomIdx] = positions[i * 3] + tx
              atomStore.y[atomIdx] = positions[i * 3 + 1] + ty
              atomStore.z[atomIdx] = positions[i * 3 + 2] + tz
              atomIdx++
            }
          }
        }
      }
    } else {
      // Non-periodic: direct update
      for (let i = 0; i < numAtoms; i++) {
        atomStore.x[i] = positions[i * 3]
        atomStore.y[i] = positions[i * 3 + 1]
        atomStore.z[i] = positions[i * 3 + 2]
      }
    }

    // Rebuild the structure to reflect new positions
    structure.refreshPosition()
    componentRef.current.rebuildRepresentations()
  }, [])

  // Create unit cell visualization using NGL Shape
  const createUnitCellShape = useCallback((lattice: { a: number[], b: number[], c: number[] }) => {
    if (!stageRef.current) return

    // Remove existing unit cell
    if (unitCellRef.current) {
      (stageRef.current as any).removeComponent(unitCellRef.current)
      unitCellRef.current = null
    }

    const { a, b, c } = lattice

    // Calculate the 8 corners of the unit cell (at origin)
    const corners = [
      [0, 0, 0],
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
      [1, 1, 0],
      [1, 0, 1],
      [0, 1, 1],
      [1, 1, 1]
    ].map(([na, nb, nc]) => [
      na * a[0] + nb * b[0] + nc * c[0],
      na * a[1] + nb * b[1] + nc * c[1],
      na * a[2] + nb * b[2] + nc * c[2]
    ])

    // Create shape with unit cell edges
    const shape = new NGL.Shape('unitcell')

    // Define the 12 edges of a cube
    const edges = [
      [0, 1], [0, 2], [0, 3], // from origin
      [1, 4], [1, 5], // from (1,0,0)
      [2, 4], [2, 6], // from (0,1,0)
      [3, 5], [3, 6], // from (0,0,1)
      [4, 7], [5, 7], [6, 7] // to (1,1,1)
    ]

    // Add thick lines for each edge (orange color)
    edges.forEach(([i, j]) => {
      shape.addWideline(
        corners[i] as [number, number, number],
        corners[j] as [number, number, number],
        [1, 0.5, 0] // orange
      )
    })

    // Add the shape to the stage
    const shapeComp = (stageRef.current as any).addComponentFromObject(shape)
    shapeComp.addRepresentation('buffer', {
      linewidth: 4,
      opacity: 0.9
    })
    unitCellRef.current = shapeComp
  }, [])

  // Load a new structure (creates new component)
  const loadStructureVisualization = useCallback((positions: number[], atomicNumbers: number[]) => {
    if (!stageRef.current) return

    // For periodic structures, generate 3x3x3 supercell for visualization
    let displayPositions = positions
    let displayAtomicNumbers = atomicNumbers
    if (latticeRef.current) {
      const supercell = generateSupercell(positions, atomicNumbers, latticeRef.current)
      displayPositions = supercell.positions
      displayAtomicNumbers = supercell.atomicNumbers
    }

    const sdf = positionsToSdf(displayPositions, displayAtomicNumbers)
    const blob = new Blob([sdf], { type: 'text/plain' })

    stageRef.current.removeAllComponents()
    unitCellRef.current = null  // Clear unit cell ref since we removed all components

    stageRef.current.loadFile(blob, { ext: 'sdf', defaultRepresentation: false })
      .then((component: any) => {
        componentRef.current = component
        component.addRepresentation('ball+stick', {
          colorScheme: 'element',
          radiusScale: 0.8,
          bondScale: 0.3,
        })

        // Add unit cell visualization if we have lattice data
        if (latticeRef.current) {
          createUnitCellShape(latticeRef.current)
        }

        // Small delay to let DOM settle, then resize and auto-view
        setTimeout(() => {
          stageRef.current?.handleResize()
          stageRef.current?.autoView()
        }, 50)
      })
  }, [createUnitCellShape])

  // Set structure
  const setStructure = useCallback((xyz: string) => {
    if (!workerRef.current || !stageRef.current) return

    // Parse atomic numbers for visualization
    atomicNumbersRef.current = parseXyzAtomicNumbers(xyz)

    // Parse lattice for periodic structures
    latticeRef.current = parseLattice(xyz)

    // Send to worker
    workerRef.current.postMessage({ type: 'setSystem', xyz })

    // Load initial visualization
    const lines = xyz.trim().split('\n')
    const numAtoms = parseInt(lines[0])
    const positions: number[] = []

    for (let i = 0; i < numAtoms; i++) {
      const parts = lines[i + 2].trim().split(/\s+/)
      positions.push(parseFloat(parts[1]), parseFloat(parts[2]), parseFloat(parts[3]))
    }

    // Load new structure visualization
    loadStructureVisualization(positions, atomicNumbersRef.current)

    // Clear energy history and reset state
    setEnergyHistory([])
    setState(s => ({ ...s, step: 0, energy: 0, kineticEnergy: 0, temperature: 0, maxForce: 0, msPerStep: 0, optimizationConverged: false }))
  }, [loadStructureVisualization])

  // Handle sample structure selection
  const handleSampleSelect = (name: string) => {
    setSelectedStructure(name)
    if (SAMPLE_STRUCTURES[name]) {
      setCustomXyz(SAMPLE_STRUCTURES[name])
    }
  }

  // Handle loading the current XYZ
  const loadCurrentStructure = () => {
    if (customXyz.trim()) {
      setStructure(customXyz)
    }
  }

  // Update parameters
  useEffect(() => {
    workerRef.current?.postMessage({
      type: 'setParameters',
      dt: timestep,
      temperature: targetTemperature,
      mode,
      maxOptSteps,
      forceThreshold,
    })
  }, [timestep, targetTemperature, mode, maxOptSteps, forceThreshold])

  // Control functions
  const startSimulation = () => {
    // Reset step counter, energy history, and convergence flag
    setEnergyHistory([])
    setState(s => ({ ...s, step: 0, optimizationConverged: false }))
    lastStepTimeRef.current = 0
    workerRef.current?.postMessage({ type: 'start', stepsPerFrame: 1, mode })
  }

  const stopMD = () => {
    workerRef.current?.postMessage({ type: 'stop' })
  }

  return (
    <div className="md-simulation">
      {/* Left panel - Structure and parameters */}
      <div className="panel panel-left">
        <div className="panel-section">
          <h3>Structure</h3>
          <select
            value={selectedStructure}
            onChange={e => handleSampleSelect(e.target.value)}
            className="structure-select"
            disabled={!state.isModelLoaded}
          >
            <optgroup label="Molecules">
              {Object.keys(SAMPLE_MOLECULES).map(name => (
                <option key={name} value={name}>{name}</option>
              ))}
            </optgroup>
            <optgroup label="Crystals">
              {Object.keys(SAMPLE_CRYSTALS).map(name => (
                <option key={name} value={name}>{name}</option>
              ))}
            </optgroup>
          </select>
          <textarea
            value={customXyz}
            onChange={e => {
              setCustomXyz(e.target.value)
              setSelectedStructure('')
            }}
            placeholder="Paste XYZ format..."
            className="xyz-input"
            rows={8}
            disabled={!state.isModelLoaded}
          />
          <button
            onClick={loadCurrentStructure}
            className="load-button"
            disabled={!state.isModelLoaded || !customXyz.trim()}
          >
            Load Structure
          </button>
        </div>

        <div className="panel-section">
          <div className="mode-tabs">
            <button
              className={`mode-tab ${mode === 'md' ? 'active' : ''}`}
              onClick={() => setMode('md')}
            >
              MD
            </button>
            <button
              className={`mode-tab ${mode === 'optimize' ? 'active' : ''}`}
              onClick={() => setMode('optimize')}
            >
              Optimize
            </button>
          </div>

          {state.isModelLoaded && (
            <div className="model-info">
              Model: <strong>PET-MAD v1.1.0</strong>
            </div>
          )}

          {mode === 'md' ? (
            <>
              <div className="params-row">
                <div className="control-group">
                  <label>Temp (K)</label>
                  <input
                    type="number"
                    value={targetTemperature}
                    onChange={e => setTargetTemperature(Number(e.target.value))}
                    min={1}
                    max={1000}
                    step={10}
                    className="number-input"
                  />
                </div>
                <div className="control-group">
                  <label>Timestep (fs)</label>
                  <input
                    type="number"
                    value={timestep}
                    onChange={e => setTimestep(Number(e.target.value))}
                    min={0.1}
                    max={2.0}
                    step={0.1}
                    className="number-input"
                  />
                </div>
              </div>
              <p className="nc-note">
                * Using non-conservative forces. Total energy will drift.
              </p>
            </>
          ) : (
            <>
              <div className="params-row">
                <div className="control-group">
                  <label>Max Steps</label>
                  <input
                    type="number"
                    value={maxOptSteps}
                    onChange={e => setMaxOptSteps(Number(e.target.value))}
                    min={10}
                    max={1000}
                    step={10}
                    className="number-input"
                  />
                </div>
                <div className="control-group">
                  <label>F Tol (eV/Å)</label>
                  <input
                    type="number"
                    value={forceThreshold}
                    onChange={e => setForceThreshold(Number(e.target.value))}
                    min={0.001}
                    max={1.0}
                    step={0.01}
                    className="number-input"
                  />
                </div>
              </div>
              <p className="nc-note">
                FIRE geometry optimization using gradient forces.
              </p>
            </>
          )}
        </div>

        {state.error && (
          <div className="error-message">
            {state.error}
          </div>
        )}
      </div>

      {/* Center - Viewer */}
      <div className="viewer-center">
        <div ref={containerRef} className="ngl-container">
          {/* Loading indicator - small overlay in corner, doesn't block viewer */}
          {(!state.isInitialized || state.isLoadingModel) && (
            <div className="loading-indicator">
              <div className="spinner-small" />
              <span>{!state.isInitialized ? 'Initializing...' : 'Loading model...'}</span>
            </div>
          )}
          {state.isInitialized && !state.isLoadingModel && !state.isModelLoaded && (
            <div className="placeholder">
              Failed to load model
            </div>
          )}
          {state.isModelLoaded && atomicNumbersRef.current.length === 0 && (
            <div className="placeholder">
              Select a structure to begin
            </div>
          )}
          {/* Energy plot overlay */}
          {energyHistory.length > 1 && (
            <div className="energy-plot">
              <svg viewBox="0 0 1000 80" preserveAspectRatio="none">
                {(() => {
                  const data = energyHistory
                  const minE = Math.min(...data)
                  const maxE = Math.max(...data)
                  const range = maxE - minE || 1
                  const padding = range * 0.1
                  // Use first energy as top of chart, so decreasing energy goes down
                  const firstE = data[0]
                  const lastE = data[data.length - 1]
                  // Set y range based on data extent with padding
                  const yMax = Math.max(firstE, maxE) + padding
                  const yMin = Math.min(lastE, minE) - padding
                  const yRange = yMax - yMin || 1

                  // Generate path - each step is 1/100th of the width (max 100 points shown)
                  // y: high energy at top (y=0), low energy at bottom (y=80)
                  const maxSteps = 100
                  const svgWidth = 1000  // viewBox width
                  const stepWidth = svgWidth / maxSteps

                  const points = data.map((e, i) => {
                    const x = i * stepWidth
                    const y = ((yMax - e) / yRange) * 80
                    return `${x},${y}`
                  }).join(' ')

                  // Calculate dot positions
                  const dots = data.map((e, i) => ({
                    x: i * stepWidth,
                    y: ((yMax - e) / yRange) * 80
                  }))

                  return (
                    <>
                      <polyline
                        points={points}
                        fill="none"
                        stroke="rgba(59, 130, 246, 0.9)"
                        strokeWidth="2.5"
                      />
                      {/* Show dot at current position */}
                      {dots.length > 0 && (
                        <circle
                          cx={dots[dots.length - 1].x}
                          cy={dots[dots.length - 1].y}
                          r="8"
                          fill="rgba(59, 130, 246, 1)"
                        />
                      )}
                      <text x="10" y="18" className="energy-label">
                        {mode === 'md' ? 'Total E' : 'E'} (eV)
                      </text>
                      <text x="10" y="70" className="energy-value">
                        {data[data.length - 1]?.toFixed(3)}
                      </text>
                    </>
                  )
                })()}
              </svg>
            </div>
          )}
        </div>

        <div className="stats-panel">
          <div className="stat">
            <span className="stat-label">Step</span>
            <span className="stat-value">{state.step}</span>
          </div>
          <div className="stat">
            <span className="stat-label">Energy</span>
            <span className="stat-value">{state.energy.toFixed(4)} eV</span>
          </div>
          {mode === 'md' ? (
            <>
              <div className="stat">
                <span className="stat-label">Kinetic</span>
                <span className="stat-value">{state.kineticEnergy.toFixed(4)} eV</span>
              </div>
              <div className="stat">
                <span className="stat-label">Temperature</span>
                <span className="stat-value">{state.temperature.toFixed(1)} K</span>
              </div>
              <div className="stat">
                <span className="stat-label">Total</span>
                <span className="stat-value">{(state.energy + state.kineticEnergy).toFixed(4)} eV</span>
              </div>
            </>
          ) : (
            <div className="stat">
              <span className="stat-label">Max Force</span>
              <span className="stat-value">{state.maxForce?.toFixed(4) ?? '—'} eV/Å</span>
            </div>
          )}
          <div className="stat">
            <span className="stat-label">Speed</span>
            <span className="stat-value">{state.msPerStep.toFixed(0)} ms/step</span>
          </div>
          <button
            onClick={state.isRunning ? stopMD : startSimulation}
            className={`play-button ${state.isRunning ? 'stop' : 'start'}`}
            disabled={!state.isModelLoaded || atomicNumbersRef.current.length === 0}
          >
            {state.isRunning ? 'Stop' : (mode === 'md' ? 'Start' : 'Optimize')}
          </button>
        </div>
      </div>

    </div>
  )
}
