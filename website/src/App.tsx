import MolecularDynamics from './components/MolecularDynamics'
import './App.css'

function App() {
  return (
    <div className="app">
      <header className="header">
        <div className="container">
          <h1>mlip.js</h1>
          <p className="subtitle">
            Machine Learning Interatomic Potentials in the Browser
          </p>
        </div>
      </header>

      <main className="main">
        <div className="container">
          <MolecularDynamics />
        </div>
      </main>

      <footer className="footer">
        <div className="container">
          <p>
            Powered by <a href="https://github.com/peterspackman/mlip.cpp">mlip.cpp</a> |{' '}
            <a href="https://github.com/peterspackman/mlip.cpp">GitHub</a>
          </p>
        </div>
      </footer>
    </div>
  )
}

export default App
