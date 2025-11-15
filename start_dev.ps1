# PowerShell script to start backend server, frontend, and SSH tunnel
# Usage: .\start_dev.ps1

$ErrorActionPreference = "Stop"

# Configuration
$REMOTE_HOST = "46.101.158.230"
$REMOTE_PORT = "5565"
$LOCAL_SERVER_PORT = "5565"
$LOCAL_FRONTEND_PORT = "5174"
$SSH_TUNNEL_PORT = "5565"
# SSH user (can be set via SSH_USER environment variable)
$SSH_USER = if ($env:SSH_USER) { $env:SSH_USER } else { "" }
$SSH_TARGET = if ($SSH_USER) { "${SSH_USER}@${REMOTE_HOST}" } else { $REMOTE_HOST }

# Function to cleanup on exit
function Cleanup {
    Write-Host "`nStopping services..." -ForegroundColor Yellow
    
    # Kill backend server
    if ($global:BACKEND_PROCESS -and !$global:BACKEND_PROCESS.HasExited) {
        Write-Host "Stopping backend server (PID: $($global:BACKEND_PROCESS.Id))..."
        Stop-Process -Id $global:BACKEND_PROCESS.Id -Force -ErrorAction SilentlyContinue
    }
    
    # Kill frontend
    if ($global:FRONTEND_PROCESS -and !$global:FRONTEND_PROCESS.HasExited) {
        Write-Host "Stopping frontend (PID: $($global:FRONTEND_PROCESS.Id))..."
        Stop-Process -Id $global:FRONTEND_PROCESS.Id -Force -ErrorAction SilentlyContinue
    }
    
    # Kill SSH tunnel
    if ($global:SSH_PROCESS -and !$global:SSH_PROCESS.HasExited) {
        Write-Host "Stopping SSH tunnel (PID: $($global:SSH_PROCESS.Id))..."
        Stop-Process -Id $global:SSH_PROCESS.Id -Force -ErrorAction SilentlyContinue
    }
    
    Write-Host "All services stopped." -ForegroundColor Green
    exit 0
}

# Trap Ctrl+C
[Console]::TreatControlCAsInput = $false
$null = Register-ObjectEvent -InputObject ([System.Console]) -EventName CancelKeyPress -Action {
    Cleanup
}

# Check if port is in use
function Test-Port {
    param([int]$Port)
    
    $connection = Test-NetConnection -ComputerName "127.0.0.1" -Port $Port -WarningAction SilentlyContinue -InformationLevel Quiet
    return $connection
}

Write-Host "Starting development environment..." -ForegroundColor Green

# Check ports
Write-Host "Checking ports..."
if (Test-Port -Port $LOCAL_SERVER_PORT) {
    Write-Host "Error: Port $LOCAL_SERVER_PORT is already in use" -ForegroundColor Red
    exit 1
}
if (Test-Port -Port $LOCAL_FRONTEND_PORT) {
    Write-Host "Error: Port $LOCAL_FRONTEND_PORT is already in use" -ForegroundColor Red
    exit 1
}

# Get project root directory
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $SCRIPT_DIR

# Start backend server
Write-Host "`nStarting backend server on 127.0.0.1:$LOCAL_SERVER_PORT..." -ForegroundColor Green
$backendStartInfo = New-Object System.Diagnostics.ProcessStartInfo
$backendStartInfo.FileName = "python"
$backendStartInfo.Arguments = "-m src.perudo.web.main"
$backendStartInfo.UseShellExecute = $false
$backendStartInfo.RedirectStandardOutput = $true
$backendStartInfo.RedirectStandardError = $true
$backendStartInfo.WorkingDirectory = $SCRIPT_DIR
$global:BACKEND_PROCESS = [System.Diagnostics.Process]::Start($backendStartInfo)
$global:BACKEND_PROCESS.StandardOutput | Out-File -FilePath "backend.log" -Append
$global:BACKEND_PROCESS.StandardError | Out-File -FilePath "backend.log" -Append
Write-Host "Backend server started (PID: $($global:BACKEND_PROCESS.Id))"

# Wait for backend to start
Start-Sleep -Seconds 3

# Check if backend is running
if ($global:BACKEND_PROCESS.HasExited) {
    Write-Host "Error: Backend server failed to start" -ForegroundColor Red
    Get-Content backend.log
    exit 1
}

# Start frontend
Write-Host "`nStarting frontend on localhost:$LOCAL_FRONTEND_PORT..." -ForegroundColor Green
$frontendStartInfo = New-Object System.Diagnostics.ProcessStartInfo
$frontendStartInfo.FileName = "npm"
$frontendStartInfo.Arguments = "run dev"
$frontendStartInfo.UseShellExecute = $false
$frontendStartInfo.RedirectStandardOutput = $true
$frontendStartInfo.RedirectStandardError = $true
$frontendStartInfo.WorkingDirectory = Join-Path $SCRIPT_DIR "frontend"
$global:FRONTEND_PROCESS = [System.Diagnostics.Process]::Start($frontendStartInfo)
$global:FRONTEND_PROCESS.StandardOutput | Out-File -FilePath (Join-Path $SCRIPT_DIR "frontend.log") -Append
$global:FRONTEND_PROCESS.StandardError | Out-File -FilePath (Join-Path $SCRIPT_DIR "frontend.log") -Append
Write-Host "Frontend started (PID: $($global:FRONTEND_PROCESS.Id))"

# Wait for frontend to start
Start-Sleep -Seconds 3

# Check if frontend is running
if ($global:FRONTEND_PROCESS.HasExited) {
    Write-Host "Error: Frontend failed to start" -ForegroundColor Red
    Get-Content frontend.log
    Stop-Process -Id $global:BACKEND_PROCESS.Id -Force -ErrorAction SilentlyContinue
    exit 1
}

# Start SSH tunnel
Write-Host "`nStarting SSH tunnel to ${SSH_TARGET}:${REMOTE_PORT}..." -ForegroundColor Green
Write-Host "SSH tunnel will forward remote port $REMOTE_PORT to local port $SSH_TUNNEL_PORT"
$sshStartInfo = New-Object System.Diagnostics.ProcessStartInfo
$sshStartInfo.FileName = "ssh"
$sshStartInfo.Arguments = "-N -L ${SSH_TUNNEL_PORT}:127.0.0.1:${REMOTE_PORT} ${SSH_TARGET}"
$sshStartInfo.UseShellExecute = $false
$sshStartInfo.RedirectStandardOutput = $true
$sshStartInfo.RedirectStandardError = $true
$global:SSH_PROCESS = [System.Diagnostics.Process]::Start($sshStartInfo)
$global:SSH_PROCESS.StandardOutput | Out-File -FilePath "ssh_tunnel.log" -Append
$global:SSH_PROCESS.StandardError | Out-File -FilePath "ssh_tunnel.log" -Append
Write-Host "SSH tunnel started (PID: $($global:SSH_PROCESS.Id))"

# Wait for SSH tunnel to establish
Start-Sleep -Seconds 2

# Check if SSH tunnel is running
if ($global:SSH_PROCESS.HasExited) {
    Write-Host "Warning: SSH tunnel may have failed to start" -ForegroundColor Yellow
    Get-Content ssh_tunnel.log
}

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "All services started successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "Backend server: http://127.0.0.1:$LOCAL_SERVER_PORT" -ForegroundColor Green
Write-Host "Frontend:       http://localhost:$LOCAL_FRONTEND_PORT" -ForegroundColor Green
Write-Host "SSH tunnel:     ${SSH_TARGET}:${REMOTE_PORT} -> localhost:${SSH_TUNNEL_PORT}" -ForegroundColor Green
Write-Host "`nPress Ctrl+C to stop all services" -ForegroundColor Yellow
Write-Host ""

# Wait for all processes
try {
    while ($true) {
        Start-Sleep -Seconds 1
        
        # Check if any process has exited
        if ($global:BACKEND_PROCESS.HasExited -or $global:FRONTEND_PROCESS.HasExited -or $global:SSH_PROCESS.HasExited) {
            Write-Host "One of the services has stopped. Cleaning up..." -ForegroundColor Yellow
            Cleanup
        }
    }
} catch {
    Cleanup
}

