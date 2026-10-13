<# Deterministic sample-level backfit and preregistration-gated analysis. #>
param(
    [string]$Repo = 'C:\Users\Admin\PycharmProjects\FS_MS',
    [string]$Manifest = 'G:\FS_microstates_study\vg_microstates\vg_microstate_manifest_mat_paper_20260723.csv',
    [string]$Hierarchy = 'G:\FS_microstates_study\vg_microstates\timestamped_hierarchy_20260801',
    [string]$Backfits = 'G:\FS_microstates_study\vg_microstates\timestamped_backfits_20260801',
    [string]$AutonomicNative = 'G:\FS_microstates_study\vg_microstates\focused_analysis_20260730_feedback_rawcounts\data\autonomic_native_ii_standardised.csv.gz',
    [string]$ValidationWindows = 'G:\FS_microstates_study\vg_microstates\focused_analysis_20260730_feedback_rawcounts\data\directional_peak_backfit_windows.csv',
    [string]$Output = 'C:\Users\Admin\OneDrive - NHS\FS_MS\publication_timestamped_prereg',
    [string]$Review = 'C:\Users\Admin\OneDrive - NHS\FS_MS\for_review_timestamped_prereg',
    [switch]$RunConfirmatory,
    [string]$PreregistrationId = ''
)

$ErrorActionPreference = 'Stop'
$Python = Join-Path $Repo '.venv\Scripts\python.exe'
$Index = Join-Path $Backfits 'transition_index.csv'
$Labels = "{'A','F','C','G','B','E'}"
$Mutex = [System.Threading.Mutex]::new($false, 'Global\FSMS_TIMESTAMPED_TRANSITIONS')
$HasMutex = $false

try {
    try { $HasMutex = $Mutex.WaitOne(0) } catch [System.Threading.AbandonedMutexException] { $HasMutex = $true }
    if (-not $HasMutex) { throw 'Another timestamped transition pipeline is running.' }
    $RequiredPaths = @($Repo,$Manifest,$AutonomicNative,$Python)
    if (-not $RunConfirmatory) { $RequiredPaths += $ValidationWindows }
    foreach ($Path in $RequiredPaths) {
        if (-not (Test-Path -LiteralPath $Path)) { throw "Missing required path: $Path" }
    }
    if ($RunConfirmatory -and -not $PreregistrationId.Trim()) {
        throw 'RunConfirmatory is locked until PreregistrationId is supplied.'
    }
    Set-Location -LiteralPath $Repo
    & $Python -B analyse_timestamped_transition_dynamics.py --self-check `
        --transition-index placeholder --autonomic-native placeholder --manifest placeholder --out-dir placeholder
    if ($LASTEXITCODE -ne 0) { throw 'Timestamped Python self-check failed.' }

    if (-not $RunConfirmatory) {
        $GlobalSolution = Join-Path $Hierarchy 'global\global_solution.mat'
        if (-not (Test-Path -LiteralPath $GlobalSolution)) {
            & matlab -batch "labels=$Labels; addpath('$($Repo.Replace("'", "''"))'); run_vg_microstates('$($Manifest.Replace("'", "''"))','$($Hierarchy.Replace("'", "''"))','global_manual_labels',labels,'global_only',true,'interactive_global_labelling',false,'require_template_labels',false,'run_backfit',false,'run_topoplots',false,'run_tanova',false,'compact_results_mat',true);"
            if ($LASTEXITCODE -ne 0) { throw 'Deterministic global fit failed.' }
        }
        & matlab -batch "labels=$Labels; addpath('$($Repo.Replace("'", "''"))'); run_vg_microstates('$($Manifest.Replace("'", "''"))','$($Hierarchy.Replace("'", "''"))','global_manual_labels',labels,'interactive_global_labelling',false,'spm_prior_weight',1,'run_backfit',false,'run_participant_condition_fits',false,'run_global_backfit',false,'run_cardiac_phase',false,'run_legacy_analyses',false,'run_topoplots',false,'run_tanova',false,'compact_results_mat',true);"
        if ($LASTEXITCODE -ne 0) { throw 'Participant-level hierarchy recovery failed.' }

        & matlab -batch "addpath('$($Repo.Replace("'", "''"))'); recover_timestamped_transition_backfits('$($Manifest.Replace("'", "''"))','$($Hierarchy.Replace("'", "''"))','$($Backfits.Replace("'", "''"))','validation_windows_csv','$($ValidationWindows.Replace("'", "''"))');"
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $Index)) { throw 'Exact transition export failed.' }
    } elseif (-not (Test-Path -LiteralPath $Index)) {
        throw "Frozen transition index is missing: $Index"
    }

    $Arguments = @('-B','analyse_timestamped_transition_dynamics.py',
        '--transition-index',$Index,'--autonomic-native',$AutonomicNative,'--manifest',$Manifest,
        '--out-dir',$Output,'--review-dir',$Review)
    if ($RunConfirmatory) { $Arguments += @('--run-confirmatory','--preregistration-id',$PreregistrationId) }
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw 'Timestamped transition analysis failed.' }
} finally {
    if ($HasMutex) { $Mutex.ReleaseMutex() }
    $Mutex.Dispose()
}
