<# Focused clean run from converted MAT files. Resumes completed stages. #>
param(
    [string]$Repo = 'C:\Users\Admin\PycharmProjects\FS_MS',
    [string]$Manifest = 'G:\FS_microstates_study\vg_microstates\vg_microstate_manifest_mat_paper_20260723.csv',
    [string]$Results = 'G:\FS_microstates_study\vg_microstates\focused_hierarchy_final',
    [string]$Analysis = 'G:\FS_microstates_study\vg_microstates\focused_analysis_final',
    [string]$PublicationAnalysis = '',
    [string]$ReviewAnalysis = '',
    [string[]]$ManualLabels = @(),
    [string]$TimestampedHierarchy = 'G:\FS_microstates_study\vg_microstates\timestamped_hierarchy_20260801',
    [string]$TimestampedBackfits = 'G:\FS_microstates_study\vg_microstates\timestamped_backfits_20260801',
    [switch]$RunConfirmatory,
    [string]$PreregistrationId = ''
)

if ($RunConfirmatory) { throw 'Use run_timestamped_transition_pipeline.ps1 -RunConfirmatory directly; the legacy runner regenerates upstream inputs.' }

$PublicationAnalysis = if ($PublicationAnalysis) { $PublicationAnalysis } else { Join-Path (Split-Path -Parent $Analysis) 'publication_final' }
$ReviewAnalysis = if ($ReviewAnalysis) { $ReviewAnalysis } else { Join-Path (Split-Path -Parent $PublicationAnalysis) 'for_review' }
$ErrorActionPreference = 'Stop'
$Python = Join-Path $Repo '.venv\Scripts\python.exe'
$EcgQc = 'G:\FS_microstates_study\heppy_output_spmms_20260724\ecg_qc'
$OriginalEcgQc = 'G:\FS_microstates_study\heppy_output\ecg_qc'
$Demographics = Join-Path $Repo 'analysis_database.csv'
$PeakWindows = Join-Path $Analysis 'data\directional_peak_backfit_windows.csv'
$TimestampedAnalysis = Join-Path $Repo 'analyse_timestamped_transition_dynamics.py'
$TimestampedRunner = Join-Path $Repo 'run_timestamped_transition_pipeline.ps1'
$AutonomicNative = Join-Path $Analysis 'data\autonomic_native_ii_standardised.csv.gz'
$Audit = Join-Path $Analysis 'pipeline_audit.json'
$Log = Join-Path $Analysis 'pipeline_run.log'
$Mutex = [System.Threading.Mutex]::new($false, 'Global\FSMS_FOCUSED_PIPELINE')
$HasMutex = $false
$State = [ordered]@{started_utc=[DateTime]::UtcNow.ToString('o'); results=$Results; analysis=$Analysis; publication_analysis=$PublicationAnalysis; review_analysis=$ReviewAnalysis; stages=[ordered]@{}}

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
    New-Item -ItemType Directory -Force -Path $Results, $Analysis, (Join-Path $Analysis 'data') | Out-Null
    Start-Transcript -LiteralPath $Log -Append | Out-Null
    try { $HasMutex = $Mutex.WaitOne(0) } catch [System.Threading.AbandonedMutexException] { $HasMutex = $true }
    if (-not $HasMutex) { throw 'Another focused FS_MS pipeline is running.' }
    foreach ($Path in @($Repo,$Manifest,$Python,$EcgQc,$OriginalEcgQc,$Demographics,$TimestampedAnalysis,$TimestampedRunner)) {
        if (-not (Test-Path -LiteralPath $Path)) { throw "Missing required path: $Path" }
    }
    if ($RunConfirmatory -and -not $PreregistrationId.Trim()) { throw 'RunConfirmatory requires PreregistrationId.' }
    Set-Location -LiteralPath $Repo
    $MatRepo = Quote-Matlab $Repo; $MatManifest = Quote-Matlab $Manifest; $MatResults = Quote-Matlab $Results

    Invoke-Stage 'self_checks' @((Join-Path $Analysis 'self_checks.ok'),
        (Join-Path $Analysis 'mechanistic_self_checks.ok')) {
        & $Python -B analyse_focused_fs_ms.py --self-check
        if ($LASTEXITCODE -ne 0) { return }
        & $Python -B analyse_global_transition_exploration.py --self-check
        if ($LASTEXITCODE -ne 0) { return }
        & $Python -B $TimestampedAnalysis --self-check --transition-index placeholder `
            --autonomic-native placeholder --manifest placeholder --out-dir placeholder
        if ($LASTEXITCODE -ne 0) { return }
        & matlab -batch "addpath('$MatRepo'); vg=fullfile('$MatRepo','external','VG-microstates'); addpath(vg); setup_microstate_paths(vg); test_spm_mix_prior;"
        if ($LASTEXITCODE -eq 0) {
            'ok' | Set-Content -LiteralPath (Join-Path $Analysis 'self_checks.ok')
            'ok' | Set-Content -LiteralPath (Join-Path $Analysis 'mechanistic_self_checks.ok')
        }
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

    Invoke-Stage 'global_first_publication_analysis' @(
        (Join-Path $PublicationAnalysis 'RESULTS.md'),
        (Join-Path $PublicationAnalysis 'PUBLICATION_AUDIT.json'),
        (Join-Path $PublicationAnalysis 'ANALYSIS_DECISIONS.csv'),
        (Join-Path $PublicationAnalysis 'tables\matched_pseudo_difference_in_differences.csv'),
        (Join-Path $PublicationAnalysis 'tables\manifold_loadings.csv'),
        (Join-Path $PublicationAnalysis 'tables\stepwise_logistic_models.csv'),
        (Join-Path $PublicationAnalysis 'tables\state_exploratory_associations.csv'),
        (Join-Path $PublicationAnalysis 'tables\subgroup_contrasts.csv'),
        (Join-Path $PublicationAnalysis 'tables\demographic_summary.csv'),
        (Join-Path $PublicationAnalysis 'figures\figure_01_global_transition_trajectories.png'),
        (Join-Path $PublicationAnalysis 'figures\figure_02_js_kappa_csi_cpi_manifold.png'),
        (Join-Path $PublicationAnalysis 'figures\figure_03_stepwise_logistic.png'),
        (Join-Path $PublicationAnalysis 'figures\figure_04_exploratory_state_drilldown.png'),
        (Join-Path $PublicationAnalysis 'figures\figure_05_matched_pseudo_sensitivity.png'),
        (Join-Path $PublicationAnalysis 'figures\figure_06_within_cohort_comparisons.png'),
        (Join-Path $ReviewAnalysis 'README.md')) {
        & $Python -B analyse_global_transition_exploration.py `
            --peak-windows-csv $PeakWindows `
            --common-windows-csv (Join-Path $Analysis 'data\autonomic_peak_microstate_common_windows.csv.gz') `
            --focused-dir $Analysis --manifest $Manifest --demographics-csv $Demographics `
            --out-dir $PublicationAnalysis --review-dir $ReviewAnalysis --overwrite
    } -Force

    $TimestampedOut = Join-Path $PublicationAnalysis 'timestamped_transition'
    $TimestampedReview = Join-Path $ReviewAnalysis 'timestamped_transition'
    $TimestampedIndex = Join-Path $TimestampedBackfits 'transition_index.csv'
    $TimestampedOutputs = @(
        $TimestampedIndex,
        (Join-Path $TimestampedOut 'ANALYSIS_AUDIT.json'),
        (Join-Path $TimestampedOut 'OSF_PREREGISTRATION_BLURBS.md'),
        (Join-Path $TimestampedOut 'POST_REGISTRATION_RUNBOOK.md'),
        (Join-Path $TimestampedOut 'PREREGISTRATION_LOCK.json'),
        (Join-Path $TimestampedOut 'PREREGISTRATION_LOCK.sha256'),
        (Join-Path $TimestampedOut 'tables\timestamped_window_metrics.csv.gz'),
        (Join-Path $TimestampedOut 'tables\baseline_B_tensor_precision_summary.csv'),
        (Join-Path $TimestampedOut 'tables\confirmatory_analysis_plan_WITHHELD.csv'),
        (Join-Path $TimestampedOut 'figures\figure_01_peri_ictal_manifold.png'),
        (Join-Path $TimestampedOut 'figures\figure_02_matrix_js_kappa_trajectories.png'),
        (Join-Path $TimestampedOut 'figures\figure_03_baseline_B_tensor_precision.png'),
        (Join-Path $TimestampedOut 'figures\figure_04_autonomic_state_trajectories.png'),
        (Join-Path $TimestampedReview 'SHA256SUMS.csv'))
    if ($RunConfirmatory) {
        $TimestampedOutputs += (Join-Path $TimestampedOut 'tables\confirmatory_results.csv')
        $TimestampedOutputs += (Join-Path $TimestampedOut 'CONFIRMATORY_EXECUTION.json')
    }
    Invoke-Stage 'timestamped_transition_tensor' $TimestampedOutputs {
        $TimestampedArguments = @{
            Repo = $Repo; Manifest = $Manifest; Hierarchy = $TimestampedHierarchy
            Backfits = $TimestampedBackfits; AutonomicNative = $AutonomicNative
            ValidationWindows = $PeakWindows; Output = $TimestampedOut; Review = $TimestampedReview
        }
        if ($RunConfirmatory) {
            & $TimestampedRunner @TimestampedArguments -RunConfirmatory -PreregistrationId $PreregistrationId
        } else {
            & $TimestampedRunner @TimestampedArguments -Prepare
        }
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
