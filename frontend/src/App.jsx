import { useState } from 'react'
import axios from 'axios'
import {
  AlertTriangle,
  CheckCircle,
  UploadCloud,
  Cpu,
  Activity,
  ShieldAlert,
  Database,
  RefreshCw,
  Lock,
  LogOut
} from 'lucide-react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
} from 'recharts'

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [authLoading, setAuthLoading] = useState(false)

  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [chartData, setChartData] = useState([])

  // --- Auth Handlers ---
  const handleLogin = async (e) => {
    e.preventDefault()
    setAuthLoading(true)
    // Simulate auth handshake delay
    await new Promise(resolve => setTimeout(resolve, 1200))
    setAuthLoading(false)
    setIsAuthenticated(true)
  }

  const handleLogout = () => {
    setIsAuthenticated(false)
    resetDashboard()
  }

  // --- Dashboard Handlers ---
  const handleFileUpload = async () => {
    setLoading(true)

    // Simulate a brief network delay for file parsing realism
    await new Promise(resolve => setTimeout(resolve, 800))

    try {
      // The exact same edge-case test batch
      const payload = [
        { Component_ID: "COMP_001", Lot_ID: "LOT_01", Leakage_0h_uA: 5.2, Leakage_24h_uA: 5.4 },
        { Component_ID: "COMP_002", Lot_ID: "LOT_01", Leakage_0h_uA: 5.1, Leakage_24h_uA: 5.3 },
        { Component_ID: "COMP_003", Lot_ID: "LOT_01", Leakage_0h_uA: 5.3, Leakage_24h_uA: 5.5 },
        { Component_ID: "COMP_004", Lot_ID: "LOT_01", Leakage_0h_uA: 5.2, Leakage_24h_uA: 5.2 },
        { Component_ID: "COMP_005", Lot_ID: "LOT_01", Leakage_0h_uA: 5.0, Leakage_24h_uA: 5.4 },
        { Component_ID: "COMP_BROKEN", Lot_ID: "LOT_01", Leakage_0h_uA: 5.1, Leakage_24h_uA: 25.4 }
      ]

      const response = await axios.post('http://127.0.0.1:8000/analyze_batch', payload)
      setResults(response.data)

      // Format data for the Recharts Line Graph
      const formattedChartData = [
        { time: '0h (Start)' },
        { time: '24h (Measured)' },
        { time: '168h (Predicted)' }
      ]

      payload.forEach(comp => {
        const prediction = response.data.results.find(r => r.Component_ID === comp.Component_ID)
        formattedChartData[0][comp.Component_ID] = comp.Leakage_0h_uA
        formattedChartData[1][comp.Component_ID] = comp.Leakage_24h_uA
        formattedChartData[2][comp.Component_ID] = prediction ? prediction.Predicted_168h_uA : null
      })

      setChartData(formattedChartData)
    } catch (error) {
      console.error("Error calling backend:", error)
      alert("System Error: Ensure the FastAPI backend is running and reachable on port 8000.")
    }
    setLoading(false)
  }

  const resetDashboard = () => {
    setResults(null)
    setChartData([])
  }

  // ==========================================
  // RENDER: LOGIN SCREEN
  // ==========================================
  if (!isAuthenticated) {
    return (
      <div className="min-h-screen bg-slate-950 flex items-center justify-center p-4 font-sans selection:bg-blue-500/30">
        <div className="max-w-md w-full animate-in fade-in zoom-in duration-700">
          <div className="text-center mb-10">
            <div className="mx-auto bg-blue-900/20 w-20 h-20 rounded-full flex items-center justify-center border border-blue-500/30 mb-6 shadow-[0_0_40px_-10px_rgba(37,99,235,0.4)]">
              <ShieldAlert className="text-blue-400 w-10 h-10" />
            </div>
            <h1 className="text-4xl font-extrabold text-slate-100 tracking-tight">AeroGuard AI</h1>
            <p className="text-blue-400/80 font-medium tracking-wide text-sm uppercase mt-3">Mission Assurance Authentication</p>
          </div>

          <form onSubmit={handleLogin} className="bg-slate-900/60 p-8 rounded-2xl border border-slate-800 shadow-2xl backdrop-blur-md">
            <div className="space-y-5">
              <div>
                <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Inspector ID</label>
                <div className="relative">
                  <input
                    type="text"
                    defaultValue="ISRO-QA-7749"
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-4 py-3 text-slate-300 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all font-mono"
                    required
                  />
                </div>
              </div>
              <div>
                <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Security Clearance Key</label>
                <div className="relative">
                  <input
                    type="password"
                    defaultValue="••••••••••••"
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-4 py-3 text-slate-300 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all"
                    required
                  />
                </div>
              </div>

              <button
                type="submit"
                disabled={authLoading}
                className="w-full relative group overflow-hidden bg-blue-600 hover:bg-blue-500 text-white mt-4 px-8 py-3.5 rounded-lg font-bold transition-all duration-300 disabled:opacity-70 disabled:cursor-not-allowed shadow-[0_0_30px_-10px_rgba(37,99,235,0.5)] border border-blue-400/50"
              >
                <div className="flex items-center justify-center gap-2 relative z-10">
                  {authLoading ? (
                    <RefreshCw className="animate-spin w-5 h-5 text-white" />
                  ) : (
                    <Lock className="w-5 h-5" />
                  )}
                  <span className="tracking-wide">
                    {authLoading ? "VERIFYING CREDENTIALS..." : "INITIALIZE SESSION"}
                  </span>
                </div>
              </button>
            </div>
          </form>

          <p className="text-center text-slate-600 text-xs mt-8 font-mono">
            SECURE CONNECTION • ENCRYPTED END-TO-END
          </p>
        </div>
      </div>
    )
  }

  // ==========================================
  // RENDER: MAIN DASHBOARD
  // ==========================================
  return (
    <div className="min-h-screen bg-slate-950 text-slate-300 p-4 md:p-8 font-sans selection:bg-blue-500/30">
      <div className="max-w-7xl mx-auto space-y-6">

        {/* Header Section */}
        <header className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-slate-900/50 p-6 rounded-2xl border border-slate-800 shadow-2xl backdrop-blur-sm animate-in fade-in slide-in-from-top-4 duration-500">
          <div className="flex items-center gap-4">
            <div className="bg-blue-600/20 p-3 rounded-xl border border-blue-500/30">
              <ShieldAlert className="text-blue-400 w-8 h-8" />
            </div>
            <div>
              <h1 className="text-3xl font-extrabold text-slate-100 tracking-tight">AeroGuard AI</h1>
              <p className="text-blue-400/80 font-medium tracking-wide text-sm uppercase mt-1">Mission Assurance & Anomaly Command</p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className="flex items-center gap-3 bg-slate-950/50 px-4 py-2 rounded-lg border border-slate-800">
              <Activity className="text-emerald-400 w-4 h-4 animate-pulse" />
              <span className="text-sm font-mono text-slate-400">STATUS: <span className="text-emerald-400 font-bold">NOMINAL</span></span>
            </div>
            <button
              onClick={handleLogout}
              className="flex items-center gap-2 bg-red-500/10 hover:bg-red-500/20 text-red-400 border border-red-500/30 px-4 py-2 rounded-lg transition-colors text-sm font-semibold"
            >
              <LogOut className="w-4 h-4" /> End Session
            </button>
          </div>
        </header>

        {/* Main Content Area */}
        {!results ? (
          <div className="flex flex-col items-center justify-center min-h-[60vh] bg-slate-900/30 rounded-2xl border border-slate-800/50 border-dashed p-10 animate-in fade-in duration-700">
            <div className="max-w-md w-full text-center space-y-6">
              <div className="mx-auto w-24 h-24 bg-slate-800/50 rounded-full flex items-center justify-center border border-slate-700 mb-6">
                <Database className="text-blue-500 w-10 h-10" />
              </div>
              <h2 className="text-2xl font-bold text-slate-200">Awaiting Telemetry Data</h2>
              <p className="text-slate-400">Upload the latest 24-hour component burn-in batch data (CSV) to initiate IQR scaling and predictive drift analysis.</p>

              <button
                onClick={handleFileUpload}
                disabled={loading}
                className="w-full relative group overflow-hidden bg-blue-600 hover:bg-blue-500 text-white px-8 py-4 rounded-xl font-bold transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed shadow-[0_0_40px_-10px_rgba(37,99,235,0.5)] border border-blue-400/50"
              >
                <div className="flex items-center justify-center gap-3 relative z-10">
                  {loading ? (
                    <RefreshCw className="animate-spin w-6 h-6 text-white" />
                  ) : (
                    <UploadCloud className="w-6 h-6 group-hover:-translate-y-1 transition-transform" />
                  )}
                  <span className="tracking-wide">
                    {loading ? "PROCESSING BATCH DATA..." : "SIMULATE BATCH UPLOAD"}
                  </span>
                </div>
                <div className="absolute inset-0 h-full w-full bg-gradient-to-r from-blue-600 via-blue-400 to-blue-600 opacity-0 group-hover:opacity-20 transition-opacity duration-300 blur-xl"></div>
              </button>
            </div>
          </div>
        ) : (
          <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-700">

            {/* Top Metrics Row */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800 shadow-xl flex items-center gap-4">
                <div className="p-4 bg-slate-800/50 rounded-xl text-slate-400">
                  <Cpu className="w-8 h-8" />
                </div>
                <div>
                  <p className="text-slate-500 text-sm font-medium uppercase tracking-wider mb-1">Batch Size</p>
                  <p className="text-3xl font-bold text-slate-200">{results
