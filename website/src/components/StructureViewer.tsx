import { useEffect, useRef, useState, useCallback } from 'react'
import * as NGL from 'ngl'
import './StructureViewer.css'

// Sample XYZ structures for demo
const SAMPLE_STRUCTURES: Record<string, string> = {
  'Water': `3
Water molecule
O     0.000000     0.000000     0.117489
H     0.000000     0.756950    -0.469957
H     0.000000    -0.756950    -0.469957`,
  'Ethanol': `9
Ethanol molecule
C    -0.001193    -0.004555     0.009236
C     1.519736    -0.001568    -0.012413
O     2.032422     1.326098    -0.087629
H    -0.394952     1.007606    -0.074891
H    -0.376887    -0.547259    -0.861972
H    -0.435219    -0.483282     0.891082
H     1.894949    -0.539891     0.862637
H     1.898649    -0.518854    -0.898756
H     1.685063     1.800579     0.682628`,
  'Benzene': `12
Benzene molecule
C     1.391500     0.000000     0.000000
C     0.695750     1.205074     0.000000
C    -0.695750     1.205074     0.000000
C    -1.391500     0.000000     0.000000
C    -0.695750    -1.205074     0.000000
C     0.695750    -1.205074     0.000000
H     2.479500     0.000000     0.000000
H     1.239750     2.147073     0.000000
H    -1.239750     2.147073     0.000000
H    -2.479500     0.000000     0.000000
H    -1.239750    -2.147073     0.000000
H     1.239750    -2.147073     0.000000`,
}

// Convert XYZ to PDB format for NGL
function xyzToPdb(xyz: string): string {
  const lines = xyz.trim().split('\n')
  const numAtoms = parseInt(lines[0])
  let pdb = ''

  for (let i = 0; i < numAtoms; i++) {
    const parts = lines[i + 2].trim().split(/\s+/)
    const element = parts[0]
    const x = parseFloat(parts[1])
    const y = parseFloat(parts[2])
    const z = parseFloat(parts[3])

    const atomNum = (i + 1).toString().padStart(5, ' ')
    const atomName = element.padEnd(4, ' ')
    const xStr = x.toFixed(3).padStart(8, ' ')
    const yStr = y.toFixed(3).padStart(8, ' ')
    const zStr = z.toFixed(3).padStart(8, ' ')

    pdb += `ATOM  ${atomNum} ${atomName} MOL A   1    ${xStr}${yStr}${zStr}  1.00  0.00           ${element.padEnd(2, ' ')}\n`
  }
  pdb += 'END\n'
  return pdb
}

interface ViewerState {
  isLoading: boolean
  error: string
  currentStructure: string
}

export default function StructureViewer() {
  const containerRef = useRef<HTMLDivElement>(null)
  const stageRef = useRef<NGL.Stage | null>(null)
  const componentRef = useRef<any>(null)

  const [state, setState] = useState<ViewerState>({
    isLoading: false,
    error: '',
    currentStructure: '',
  })
  const [representation, setRepresentation] = useState('ball+stick')
  const [colorScheme, setColorScheme] = useState('element')
  const [customXyz, setCustomXyz] = useState('')

  // Initialize NGL Stage
  useEffect(() => {
    if (!containerRef.current) return

    const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches
    stageRef.current = new NGL.Stage(containerRef.current, {
      backgroundColor: isDark ? '#1a1a1a' : '#ffffff',
      quality: 'high',
    })

    const handleResize = () => {
      stageRef.current?.handleResize()
    }

    window.addEventListener('resize', handleResize)

    // Watch for theme changes
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')
    const handleThemeChange = (e: MediaQueryListEvent) => {
      stageRef.current?.setParameters({
        backgroundColor: e.matches ? '#1a1a1a' : '#ffffff',
      })
    }
    mediaQuery.addEventListener('change', handleThemeChange)

    return () => {
      window.removeEventListener('resize', handleResize)
      mediaQuery.removeEventListener('change', handleThemeChange)
      stageRef.current?.dispose()
    }
  }, [])

  // Load structure
  const loadStructure = useCallback(async (xyz: string, name: string) => {
    if (!stageRef.current) return

    setState(s => ({ ...s, isLoading: true, error: '' }))

    try {
      // Clear existing structure
      stageRef.current.removeAllComponents()
      componentRef.current = null

      // Convert to PDB and load
      const pdb = xyzToPdb(xyz)
      const blob = new Blob([pdb], { type: 'text/plain' })

      const component = await stageRef.current.loadFile(blob, {
        ext: 'pdb',
        name: name,
        defaultRepresentation: false,
      })

      componentRef.current = component
      updateRepresentation()
      // Small delay to let DOM settle
      setTimeout(() => {
        stageRef.current?.handleResize()
        stageRef.current?.autoView()
      }, 50)

      setState(s => ({ ...s, isLoading: false, currentStructure: name }))
    } catch (err: any) {
      setState(s => ({
        ...s,
        isLoading: false,
        error: `Failed to load structure: ${err.message}`,
      }))
    }
  }, [])

  // Update representation
  const updateRepresentation = useCallback(() => {
    if (!componentRef.current) return

    componentRef.current.removeAllRepresentations()

    const params: any = {
      colorScheme: colorScheme,
    }

    if (representation === 'ball+stick') {
      params.radiusScale = 0.8
      params.bondScale = 0.3
    } else if (representation === 'spacefill') {
      params.radiusScale = 1.0
    } else if (representation === 'licorice') {
      params.bondScale = 0.5
    }

    componentRef.current.addRepresentation(representation, params)
    stageRef.current?.autoView()
  }, [representation, colorScheme])

  // Update representation when settings change
  useEffect(() => {
    updateRepresentation()
  }, [representation, colorScheme, updateRepresentation])

  // Handle file upload
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    try {
      const text = await file.text()
      setCustomXyz(text)
      loadStructure(text, file.name)
    } catch (err: any) {
      setState(s => ({ ...s, error: `Failed to read file: ${err.message}` }))
    }
  }

  // Load custom XYZ
  const loadCustom = () => {
    if (customXyz.trim()) {
      loadStructure(customXyz, 'Custom')
    }
  }

  return (
    <div className="structure-viewer">
      <div className="viewer-controls">
        <div className="control-group">
          <label>Sample Structures</label>
          <div className="sample-buttons">
            {Object.keys(SAMPLE_STRUCTURES).map(name => (
              <button
                key={name}
                className={`sample-button ${state.currentStructure === name ? 'active' : ''}`}
                onClick={() => loadStructure(SAMPLE_STRUCTURES[name], name)}
              >
                {name}
              </button>
            ))}
          </div>
        </div>

        <div className="control-group">
          <label>Upload XYZ File</label>
          <input
            type="file"
            accept=".xyz"
            onChange={handleFileUpload}
            className="file-input"
          />
        </div>

        <div className="control-group">
          <label>Or paste XYZ data</label>
          <textarea
            value={customXyz}
            onChange={e => setCustomXyz(e.target.value)}
            placeholder="Paste XYZ format data here..."
            className="xyz-input"
            rows={4}
          />
          <button onClick={loadCustom} className="load-button">
            Load Custom
          </button>
        </div>

        <div className="control-group">
          <label>Representation</label>
          <select
            value={representation}
            onChange={e => setRepresentation(e.target.value)}
            className="select-input"
          >
            <option value="ball+stick">Ball & Stick</option>
            <option value="spacefill">Spacefill</option>
            <option value="licorice">Licorice</option>
            <option value="line">Line</option>
          </select>
        </div>

        <div className="control-group">
          <label>Color Scheme</label>
          <select
            value={colorScheme}
            onChange={e => setColorScheme(e.target.value)}
            className="select-input"
          >
            <option value="element">Element</option>
            <option value="chainname">Chain</option>
            <option value="residueindex">Residue</option>
          </select>
        </div>

        <button
          onClick={() => stageRef.current?.autoView(500)}
          className="reset-button"
        >
          Reset View
        </button>
      </div>

      <div className="viewer-container">
        <div ref={containerRef} className="ngl-container">
          {state.isLoading && (
            <div className="loading-overlay">
              <div className="spinner" />
              Loading structure...
            </div>
          )}
          {state.error && (
            <div className="error-overlay">
              {state.error}
            </div>
          )}
          {!state.currentStructure && !state.isLoading && !state.error && (
            <div className="placeholder">
              Select a sample structure or upload an XYZ file to begin
            </div>
          )}
        </div>

        {state.currentStructure && (
          <div className="structure-info">
            <strong>{state.currentStructure}</strong>
          </div>
        )}
      </div>
    </div>
  )
}
