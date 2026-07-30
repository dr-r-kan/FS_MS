<# Focused clean run from converted MAT files. Resumes completed stages. #>
param(
    [string]$Repo = 'C:\Users\Admin\PycharmProjects\FS_MS',
    [string]$Manifest = 'G:\FS_microstates_study\vg_microstates\vg_microstate_manifest_mat_paper_20260723.csv',
    [string]$Results = 'G:\FS_microstates_study\vg_microstates\focused_hierarchy_final',
    [string]$Analysis = 'G:\FS_microstates_study\vg_microstates\focused_analysis_final',
    [string[]]$ManualLabels = @()
)

$ErrorActionPreference = 'Stop'
$Python = Join-Path $Repo '.venv\Scripts\python.exe'
$EcgQc = 'G:\FS_microstates_study\heppy_output_spmms_20260724\ecg_qc'
$OriginalEcgQc = 'G:\FS_microstates_study\heppy_output\ecg_qc'
$Demographics = Join-Path $Repo 'analysis_database.csv'
$PeakWindows = Join-Path $Analysis 'data\directional_peak_backfit_windows.csv'
$Summary = Join-Path $Analysis 'data\summary.json'
$Audit = Join-Path $Analysis 'pipeline_audit.json'
$Log = Join-Path $Analysis 'pipeline_run.log'
$Mutex = [System.Threading.Mutex]::new($false, 'Global\FSMS_FOCUSED_PIPELINE')
$HasMutex = $false
$State = [ordered]@{started_utc=[DateTime]::UtcNow.ToString('o'); results=$Results; analysis=$Analysis; stages=[ordered]@{}}

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
    New-Item -ItemType Directory -Force -Path $Results, $Analysis, (Join-Path $Analysis 'data'), (Join-Path $Analysis 'figures') | Out-Null
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
        & $Python -B plot_focused_fs_ms.py --self-check
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

    Invoke-Stage 'focused_analysis' @($Summary,(Join-Path $Analysis 'RESULTS.md'),(Join-Path $Analysis 'AUDIT.txt'),
        (Join-Path $Analysis 'data\ecg_cache_schema.json'),(Join-Path $Analysis 'data\analysis_schema.json'),
        (Join-Path $Analysis 'data\hr_participant_period_cells.csv'),
        (Join-Path $Analysis 'data\hr_bayesian_cells.csv'),
        (Join-Path $Analysis 'data\coverage_bayesian_cells.csv'),
        (Join-Path $Analysis 'data\autonomic_level_participant_cells.csv'),
        (Join-Path $Analysis 'data\autonomic_level_bayesian_cells.csv'),
        (Join-Path $Analysis 'data\autonomic_level_bayesian_contrasts.csv'),
        (Join-Path $Analysis 'data\c_origin_preictal_windows.csv'),
        (Join-Path $Analysis 'data\c_origin_late_preictal.csv'),
        (Join-Path $Analysis 'data\c_origin_primary_bayesian.csv'),
        (Join-Path $Analysis 'data\c_origin_subsidiary_bayesian.csv'),
        (Join-Path $Analysis 'data\c_origin_dd_paired_values.csv'),
        (Join-Path $Analysis 'data\c_origin_js_recovery.csv'),
        (Join-Path $Analysis 'data\c_origin_js_recovery_paired_events.csv'),
        (Join-Path $Analysis 'data\fs_autonomic_c_origin_correlations.csv'),
        (Join-Path $Analysis 'data\bayesian_predictive_flow.csv'),
        (Join-Path $Analysis 'data\preictal_kappa_js_participant.csv'),
        (Join-Path $Analysis 'data\preictal_kappa_js_lines.csv'),
        (Join-Path $Analysis 'data\autonomic_peak_microstate_common_windows.csv.gz'),
        (Join-Path $Analysis 'data\posterior_fs_c_origin_c_log_kappa_autonomic_association.npz'),
        (Join-Path $Analysis 'data\posterior_fs_ii_versus_last_minute_flow_autonomic_to_c_coverage.npz'),
        (Join-Path $Analysis 'data\raw_count_c_origin_bayesian.csv'),
        (Join-Path $Analysis 'data\raw_count_c_origin_event_means.csv'),
        (Join-Path $Analysis 'data\c_origin_route_destination_bayesian.csv'),
        (Join-Path $Analysis 'data\c_origin_destination_participant_means.csv'),
        (Join-Path $Analysis 'data\all_state_raw_count_specificity_bayesian.csv'),
        (Join-Path $Analysis 'data\c_reference_threshold_sensitivity_bayesian.csv'),
        (Join-Path $Analysis 'data\c_origin_preictal_binned_trajectory_bayesian.csv'),
        (Join-Path $Analysis 'data\preictal_physiology_trajectory_bayesian.csv'),
        (Join-Path $Analysis 'data\c_origin_recovery_matched_bayesian.csv'),
        (Join-Path $Analysis 'data\matched_interictal_pseudo_onset_bayesian.csv'),
        (Join-Path $Analysis 'data\matched_interictal_pseudo_onsets.csv'),
        (Join-Path $Analysis 'data\feedback_analysis_summary.json')) {
        & $Python -B analyse_focused_fs_ms.py --results-dir $Results --manifest $Manifest `
            --ecg-qc-dir $EcgQc --original-ecg-qc-dir $OriginalEcgQc --demographics-csv $Demographics `
            --peak-windows-csv $PeakWindows --out-dir $Analysis --reuse-ecg
    }

    Invoke-Stage 'focused_plots' @(
        (Join-Path $Analysis 'PLOT_AUDIT.json'),
        (Join-Path $Analysis 'figures\dense_rr_hr.png'),
        (Join-Path $Analysis 'figures\microstate_coverage.png'),
        (Join-Path $Analysis 'figures\robust_autonomic_interictal_preictal.png'),
        (Join-Path $Analysis 'figures\c_origin_primary.png'),
        (Join-Path $Analysis 'figures\c_origin_subsidiary_raw.png'),
        (Join-Path $Analysis 'figures\c_origin_js_recovery.png'),
        (Join-Path $Analysis 'figures\fs_autonomic_c_origin_correlations.png'),
        (Join-Path $Analysis 'figures\bayesian_predictive_flow.png'),
        (Join-Path $Analysis 'figures\preictal_kappa_vs_js.png'),
        (Join-Path $Analysis 'figures\c_origin_final_minute_boxplots.png'),
        (Join-Path $Analysis 'figures\c_origin_js_recovery_boxplots.png'),
        (Join-Path $Analysis 'figures\fs_cpi_kappa_regression.png'),
        (Join-Path $Analysis 'figures\fs_csi_to_c_regression.png'),
        (Join-Path $Analysis 'figures\raw_count_c_origin.png'),
        (Join-Path $Analysis 'figures\c_origin_route_destinations.png'),
        (Join-Path $Analysis 'figures\all_state_specificity.png'),
        (Join-Path $Analysis 'figures\preictal_physiology_trajectory.png'),
        (Join-Path $Analysis 'figures\c_origin_js_kappa_state_space.png'),
        (Join-Path $Analysis 'figures\matched_pseudo_onsets.png'),
        (Join-Path $Analysis 'figures\measured_loop_diagram.png')) {
        & $Python -B plot_focused_fs_ms.py --analysis-dir $Analysis
    } -Force
    $State.result='completed'
} catch {
    $State.result='failed'; $State.error=$_.Exception.ToString(); Write-Error $_ -ErrorAction Continue
} finally {
    $State.finished_utc=[DateTime]::UtcNow.ToString('o'); $State | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $Audit -Encoding utf8
    try { Stop-Transcript | Out-Null } catch {}
    if ($HasMutex) { $Mutex.ReleaseMutex() }; $Mutex.Dispose()
}
if ($State.result -ne 'completed') { exit 1 }
