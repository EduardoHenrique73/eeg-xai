import { BrowserRouter, Route, Routes } from 'react-router-dom'
import { Layout } from './components/Layout'
import { ProtectedRoute } from './components/ProtectedRoute'
import { Dashboard } from './pages/Dashboard'
import { GestaoPacientes } from './pages/GestaoPacientes'
import { Login } from './pages/Login'
import { VisualizadorClinico } from './pages/VisualizadorClinico'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<Login />} />

        <Route element={<ProtectedRoute />}>
          <Route element={<Layout />}>
            <Route path="/" element={<Dashboard />} />
            <Route path="/pacientes" element={<GestaoPacientes />} />
            <Route path="/pacientes/:pacienteId/exame" element={<VisualizadorClinico />} />
          </Route>
        </Route>
      </Routes>
    </BrowserRouter>
  )
}

export default App
