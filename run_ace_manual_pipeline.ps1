<# Focused clean run from converted MAT files. Resumes completed stages. #>
param(
    [string]$Repo = 'C:\Users\Admin\PycharmProjects\FS_MS',
    [string]$Manifest = 'G:\FS_microstates_study\vg_microstates\vg_microstate_manifest_mat_paper_20260723.csv',
    [string]$Results = 'G:\FS_microstates_study\vg_microstates\focused_hierarchy_final',
    [string]$Analysis = 'G:\FS_microstates_study\vg_microstates\focused_analysis_final',
    [string]$OptimisedAnalysis = '',
    [string]$PublicationAnalysis = '',
    [string[]]$ManualLabels = @()
)

$OptimisedAnalysis = if ($OptimisedAnalysis) { $OptimisedAnalysis } else { Join-Path $Analysis 'optimised' }
$PublicationAnalysis = if ($PublicationAnalysis) { $PublicationAnalysis } else { Join-Path (Split-Path -Parent $Analysis) 'publication_final' }
$ErrorActionPreference = 'Stop'
$Python = Join-Path $Repo '.venv\Scripts\python.exe'
$EcgQc = 'G:\FS_microstates_study\heppy_output_spmms_20260724\ecg_qc'
$OriginalEcgQc = 'G:\FS_microstates_study\heppy_output\ecg_qc'
$Demographics = Join-Path $Repo 'analysis_database.csv'
$PeakWindows = Join-Path $Analysis 'data\directional_peak_backfit_windows.csv'
$Audit = Join-Path $Analysis 'pipeline_audit.json'
$Log = Join-Path $Analysis 'pipeline_run.log'
$Mutex = [System.Threading.Mutex]::new($false, 'Global\FSMS_FOCUSED_PIPELINE')
$HasMutex = $false
$State = [ordered]@{started_utc=[DateTime]::UtcNow.ToString('o'); results=$Results; analysis=$Analysis; optimised_analysis=$OptimisedAnalysis; publication_analysis=$PublicationAnalysis; stages=[ordered]@{}}

function Test-Outputs([string[]]$Paths) {
    foreach ($Path in $Paths) {
        if (-not (Test-Path -LiteralPath $Path) -or (Get-Item -LiteralPath $Path).Length -eq 0) { return $false }
    }
    return $true
}

function Save-State([string]$Name, [string]$Status) {
    $State.stages[$Name] = [ordered]@{status=$Status; utc=[DateTime]::UtcNow.ToString('o')}
    $State | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $Audit -Encoding utf8
}

function Invoke-Stage([string]$Name, [string[]]$Outputs, [scriptblock]$Action, [switch]$Force) {
    if (-not $Force -and (Test-Outputs $Outputs)) { Write-Host "[SKIP] $Name"; Save-State $Name 'skipped'; return }
    Write-Host "[RUN ] $Name"; Save-State $Name 'running'; & $Action
    if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit status $LASTEXITCODE" }
    if (-not (Test-Outputs $Outputs)) { throw "$Name completed without required outputs" }
    Write-Host "[DONE] $Name"; Save-State $Name 'completed'
}

function Quote-Matlab([string]$Value) { $Value.Replace("'", "''") }

try {
    New-Item -ItemType Directory -Force -Path $Results, $Analysis, $OptimisedAnalysis, (Join-Path $Analysis 'data') | Out-Null
    Start-Transcript -LiteralPath $Log -Append | Out-Null
    try { $HasMutex = $Mutex.WaitOne(0) } catch [System.Threading.AbandonedMutexException] { $HasMutex = $true }
    if (-not $HasMutex) { throw 'Another focused FS_MS pipeline is running.' }
    foreach ($Path in @($Repo,$Manifest,$Python,$EcgQc,$OriginalEcgQc,$Demographics)) {
        if (-not (Test-Path -LiteralPath $Path)) { throw "Missing required path: $Path" }
    }
    Set-Location -LiteralPath $Repo
    $MatRepo = Quote-Matlab $Repo; $MatManifest = Quote-Matlab $Manifest; $MatResults = Quote-Matlab $Results

    Invoke-Stage 'self_checks' @((Join-Path $Analysis 'self_checks.ok')) {
        & $Python -B analyse_focused_fs_ms.py --self-check
        if ($LASTEXITCODE -ne 0) { return }
        & $Python -B analyse_optimised_transition_model.py --self-check
        if ($LASTEXITCODE -ne 0) { return }
        & $Python -B build_publication_fs_ms.py --self-check
        if ($LASTEXITCODE -ne 0) { return }
        & matlab -batch "addpath('$MatRepo'); vg=fullfile('$MatRepo','external','VG-microstates'); addpath(vg); setup_microstate_paths(vg); test_spm_mix_prior;"
        if ($LASTEXITCODE -eq 0) { 'ok' | Set-Content -LiteralPath (Join-Path $Analysis 'self_checks.ok') }
    }

    $GlobalSolution = Join-Path $Results 'global\global_solution.mat'
    Invoke-Stage 'global_fit' @($GlobalSolution) {
        & matlab -batch "addpath('$MatRepo'); run_vg_microstates('$MatManifest','$MatResults','global_only',true,'interactive_global_labelling',false,'require_template_labels',false,'run_backfit',false,'run_topoplots',false,'run_tanova',false,'compact_results_mat',true);"
    }

    $LabelCsv = Join-Path $Results 'global\global_manual_labels.csv'
    if (-not (Test-Path -LiteralPath $LabelCsv)) {
        $Preview = Join-Path $Results 'global\global_centers_for_manual_labelling.png'
        if (-not (Test-Path -LiteralPath $Preview)) {
            & matlab -batch "addpath('$MatRepo'); export_global_manual_preview('$MatResults','$(Quote-Matlab $Preview)');"
            if ($LASTEXITCODE -ne 0) { throw 'Manual-label preview failed.' }
        }
        if ($ManualLabels.Count -eq 0) {
            Start-Process -FilePath $Preview
            do { $ManualLabels = @((Read-Host 'Labels for fitted States 1..6, comma-separated') -split '[,;/\s]+' | Where-Object { $_ } | ForEach-Object {$_.ToUpperInvariant()}) }
            until ($ManualLabels.Count -eq 6 -and @($ManualLabels | Sort-Object -Unique).Count -eq 6 -and
                   (@($ManualLabels | Sort-Object) -join ',') -eq 'A,B,C,E,F,G')
            for ($i=0; $i -lt 6; $i++) { Write-Host "State $($i+1) = $($ManualLabels[$i])" }
            if ((Read-Host 'Continue with this mapping? [y/N]') -notmatch '^(?i:y|yes)$') { throw 'Manual labels not confirmed.' }
        }
        if ($ManualLabels.Count -ne 6 -or (@($ManualLabels | Sort-Object) -join ',') -ne 'A,B,C,E,F,G') {
            throw 'ManualLabels must be the six unique labels A,B,C,E,F,G in fitted-state order.'
        }
    } else {
        $ManualLabels = @(Import-Csv -LiteralPath $LabelCsv | Sort-Object {[int]$_.state_index} | ForEach-Object {$_.label})
    }

    $HierarchyOutputs = @(
        $LabelCsv,
        (Join-Path $Results 'hierarchical_fit_summary.csv'),
        (Join-Path $Results 'participant_condition_state_backfit_metrics.csv'),
        (Join-Path $Results 'participant_condition_preictal_time_metrics.csv'),
        (Join-Path $Results 'participant_condition_transition_metrics.csv')
    )
    Invoke-Stage 'hierarchy_and_backfit' $HierarchyOutputs {
        $MatLabels = '{' + (($ManualLabels | ForEach-Object {"'$(Quote-Matlab $_)'"}) -join ',') + '}'
        & matlab -batch "labels=$MatLabels; addpath('$MatRepo'); run_vg_microstates('$MatManifest','$MatResults','global_manual_labels',labels,'interactive_global_labelling',false,'spm_prior_weight',1,'run_backfit',true,'run_participant_condition_fits',false,'run_global_backfit',false,'run_cardiac_phase',false,'run_legacy_analyses',false,'run_topoplots',false,'run_tanova',false,'backfit_time_bin_s',10,'backfit_time_window_s',1200,'compact_results_mat',true);"
    }

    Invoke-Stage 'peak_backfit_windows' @($PeakWindows) {
        & matlab -batch "addpath('$MatRepo'); export_directional_peak_windows('$MatResults','$MatManifest','$(Quote-Matlab $PeakWindows)');"
    } -Force

    Invoke-Stage 'retained_analysis' @((Join-Path $Analysis 'RETAINED_RESULTS.md'),
        (Join-Path $Analysis 'RETAINED_AUDIT.json'),
        (Join-Path $Analysis 'data\retained_summary.json'),
        (Join-Path $Analysis 'data\ecg_cache_schema.json'),(Join-Path $Analysis 'data\analysis_schema.json'),
        (Join-Path $Analysis 'data\hr_participant_period_cells.csv'),
        (Join-Path $Analysis 'data\hr_bayesian_cells.csv'),
        (Join-Path $Analysis 'data\coverage_bayesian_cells.csv'),
        (Join-Path $Analysis 'data\autonomic_level_participant_cells.csv'),
        (Join-Path $Analysis 'data\autonomic_level_bayesian_cells.csv'),
        (Join-Path $Analysis 'data\autonomic_level_bayesian_contrasts.csv'),
        (Join-Path $Analysis 'data\c_origin_js_recovery.csv'),
        (Join-Path $Analysis 'data\c_origin_js_recovery_paired_events.csv'),
        (Join-Path $Analysis 'data\autonomic_peak_microstate_common_windows.csv.gz'),
        (Join-Path $Analysis 'data\autonomic_native_ii_standardised.csv.gz'),
        (Join-Path $Analysis 'data\state_reference_qc.csv')) {
        & $Python -B analyse_focused_fs_ms.py --results-dir $Results --manifest $Manifest `
            --ecg-qc-dir $EcgQc --original-ecg-qc-dir $OriginalEcgQc --demographics-csv $Demographics `
            --peak-windows-csv $PeakWindows --out-dir $Analysis --reuse-ecg --retained-only
    }

    Invoke-Stage 'optimised_transition_model' @(
        (Join-Path $OptimisedAnalysis 'AUDIT.json'),
        (Join-Path $OptimisedAnalysis 'RESULTS.md'),
        (Join-Path $OptimisedAnalysis 'data\summary.json'),
        (Join-Path $OptimisedAnalysis 'data\hierarchical_multinomial_posterior.nc'),
        (Join-Path $OptimisedAnalysis 'data\planned_early_late_contrasts.csv'),
        (Join-Path $OptimisedAnalysis 'data\matched_pseudo_difference_in_differences.csv'),
        (Join-Path $OptimisedAnalysis 'data\confirmatory_fit_diagnostics.json'),
        (Join-Path $OptimisedAnalysis 'data\posterior_predictive_checks.csv'),
        (Join-Path $OptimisedAnalysis 'data\simulation_recovery.csv'),
        (Join-Path $OptimisedAnalysis 'data\dynamic_loop_model_comparison.csv')) {
        & $Python -B analyse_optimised_transition_model.py `
            --peak-windows-csv $PeakWindows `
            --common-windows-csv (Join-Path $Analysis 'data\autonomic_peak_microstate_common_windows.csv.gz') `
            --native-windows-csv (Join-Path $Analysis 'data\autonomic_native_ii_standardised.csv.gz') `
            --reference-qc-csv (Join-Path $Analysis 'data\state_reference_qc.csv') `
            --out-dir $OptimisedAnalysis
    }

    Invoke-Stage 'publication_analysis' @(
        (Join-Path $PublicationAnalysis 'RESULTS.md'),
        (Join-Path $PublicationAnalysis 'PUBLICATION_AUDIT.json'),
        (Join-Path $PublicationAnalysis 'ANALYSIS_DECISIONS.csv'),
        (Join-Path $PublicationAnalysis 'tables\matched_pseudo_difference_in_differences.csv'),
        (Join-Path $PublicationAnalysis 'tables\confirmatory_fit_diagnostics.json'),
        (Join-Path $PublicationAnalysis 'figures\figure_03_transition_trajectories.png'),
        (Join-Path $PublicationAnalysis 'figures\figure_09_fs_kappa_js_cpi_3d.png'),
        (Join-Path $PublicationAnalysis 'figures\figure_10_es_kappa_js_cpi_3d.png')) {
        & $Python -B build_publication_fs_ms.py --focused-dir $Analysis `
            --optimised-dir $OptimisedAnalysis --out-dir $PublicationAnalysis --overwrite
    }
    $State.result='completed'
} catch {
    $State.result='failed'; $State.error=$_.Exception.ToString(); Write-Error $_ -ErrorAction Continue
} finally {
    $State.finished_utc=[DateTime]::UtcNow.ToString('o'); $State | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $Audit -Encoding utf8
    try { Stop-Transcript | Out-Null } catch {}
    if ($HasMutex) { $Mutex.ReleaseMutex() }; $Mutex.Dispose()
}
if ($State.result -ne 'completed') { exit 1 }
