# push_to_github.ps1 — initialize the repo and push MCU-RAG to GitHub.
#
# Usage (from D:\pixel-rag):
#   .\push_to_github.ps1 -RepoUrl https://github.com/<your-username>/mcu-rag.git
#
# Prereqs:
#   1. Git installed:        winget install Git.Git   (click "Yes" on the UAC prompt)
#   2. An empty repo created on GitHub (no README), OR install GitHub CLI and let it create one:
#                            winget install GitHub.cli ; gh auth login ; gh repo create mcu-rag --public --source . --push
#
# On the first push, Git Credential Manager opens a browser to sign in to GitHub — complete it once.

param(
    [Parameter(Mandatory = $true)]
    [string]$RepoUrl,
    [string]$Branch = "main"
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Error "git is not installed. Run:  winget install Git.Git   then reopen PowerShell and retry."
}

# Identify the committer if not already configured.
if (-not (git config user.email)) { git config user.email "r.siddharthgargi@gmail.com" }
if (-not (git config user.name))  { git config user.name  "Siddharth Gargi" }

if (-not (Test-Path ".git")) { git init }

git add .
git commit -m "MCU-RAG: local pixel-RAG for embedded datasheets (UI + CLI + Colab notebook)"
git branch -M $Branch

if (git remote | Select-String -Quiet "^origin$") {
    git remote set-url origin $RepoUrl
} else {
    git remote add origin $RepoUrl
}

git push -u origin $Branch
Write-Host "`nPushed to $RepoUrl ($Branch)" -ForegroundColor Green
