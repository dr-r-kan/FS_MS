function outputs = run_vg_microstates(manifest_csv, output_dir, varargin)
%RUN_VG_MICROSTATES Headless K=6 VG-microstates hierarchy for a manifest of EEG files.
%
%   outputs = run_vg_microstates(manifest_csv, output_dir, ...)
%
% Fits the global K=6 VB-GMM from the pooled GFP-peak maps, labels the global states, then fits the
% group, condition and participant levels (parent-prior chain) and, optionally, participant-condition
% fits and backfit metric tables. There is no interactive step, no hard-coded path and no external
% Python call: manifest_csv and output_dir are required.
%
% LABELLING (name 'labelling_mode')
%   'prespecified' (default when no explicit labels are supplied) - pre-specified rule
%                  template_prespecified_v2 (fsms/matlab/template_prespecified_v2.m): exact optimal
%                  one-to-one assignment of the six global states to classes {A,B,C,E,F,G} maximising
%                  summed |r| with MetaMaps_2023_06. State names are nominal: assigned |r| < 0.5 only sets
%                  labelling_quality = 'below_threshold' in the audit; TECHNICAL failures (e.g. no usable
%                  channel match) stop the run with global/LABELLING_REJECTED.json. Implemented as: pass 1 (global fit only, no template
%                  alignment) -> labelling -> pass 2 (the full hierarchy, which resumes from the cached
%                  global fit and the saved labels). The global fit is identical to a single-pass run.
%   'explicit'     - labels given in 'global_manual_labels' (fitted-state order), as before.
%   'interactive'  - legacy terminal prompt (needs a MATLAB terminal; not usable with -batch).
%   'template_auto'- vendored automatic template alignment (any K of 7 classes; NOT pre-specified).
%
% Name-value options (study defaults shown; every analysis not needed for the transition study is off)
%   K_candidates 6 (the paper analysis is locked to K=6), spm_prior_weight 1, global_only false,
%   run_backfit false, run_participant_condition_fits false, run_global_backfit false,
%   run_cardiac_phase false, run_topoplots false, run_tanova false, tanova_permutations 5000 (only used
%   when run_tanova), compact_results_mat true, redo false (true recomputes every cache IN PLACE),
%   template_file '' (vendored MetaMaps), extra_hierarchy_args {} (extra name-value pairs for
%   fit_microstate_hierarchical_dataset; tests/diagnostics only, recorded by run_study_hierarchy),
%   save_plots false (centre/candidate figures are not needed headless; numerics are unaffected).

    if nargin < 2 || isempty(manifest_csv) || isempty(output_dir)
        error('FSMS:RunVgArguments', 'run_vg_microstates(manifest_csv, output_dir, ...) needs both arguments.');
    end
    p = inputParser;
    addRequired(p, 'manifest_csv', @(x) ischar(x) || isstring(x));
    addRequired(p, 'output_dir', @(x) ischar(x) || isstring(x));
    addParameter(p, 'K_candidates', 6, @isnumeric);
    addParameter(p, 'labelling_mode', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'global_manual_labels', {}, @(x) iscell(x) || isstring(x));
    addParameter(p, 'interactive_global_labelling', false, @islogical);
    addParameter(p, 'relabel_global_states', false, @islogical);
    addParameter(p, 'spm_prior_weight', 1, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'global_only', false, @islogical);
    addParameter(p, 'ecg_qc_dir', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'run_backfit', false, @islogical);
    addParameter(p, 'run_participant_condition_fits', false, @islogical);
    addParameter(p, 'run_global_backfit', false, @islogical);
    addParameter(p, 'run_cardiac_phase', false, @islogical);
    addParameter(p, 'run_legacy_analyses', false, @islogical);
    addParameter(p, 'preictal_window_refit_only', false, @islogical);
    addParameter(p, 'window_refit_length_s', 60, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'window_refit_step_s', 30, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'window_refit_time_window_s', 1200, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'window_refit_min_peaks', 30, @(x) isnumeric(x) && isscalar(x) && x >= 2);
    addParameter(p, 'window_refit_min_state_peaks', 5, @(x) isnumeric(x) && isscalar(x) && x >= 1);
    addParameter(p, 'window_refit_start_record', 1, @(x) isnumeric(x) && isscalar(x) && x >= 1);
    addParameter(p, 'window_refit_max_records', Inf, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'window_refit_csv', 'preictal_window_refit_maps.csv', @(x) ischar(x) || isstring(x));
    addParameter(p, 'compact_results_mat', true, @islogical);
    addParameter(p, 'redo', false, @islogical);
    addParameter(p, 'require_template_labels', true, @islogical);
    addParameter(p, 'template_file', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'run_topoplots', false, @islogical);
    addParameter(p, 'run_tanova', false, @islogical);
    addParameter(p, 'tanova_permutations', 5000, @isnumeric);
    addParameter(p, 'backfit_time_bin_s', 1, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'backfit_time_window_s', 1200, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'extra_hierarchy_args', {}, @iscell);
    addParameter(p, 'save_plots', false, @islogical);
    parse(p, manifest_csv, output_dir, varargin{:});
    cfg = p.Results;
    if ~isequal(double(cfg.K_candidates(:)'), 6)
        error('FSMS:KNotSix', 'The paper analysis requires the locked K=6 hierarchy.');
    end
    if cfg.run_legacy_analyses
        error('FSMS:LegacyAnalysesRemoved', ['run_legacy_analyses is no longer supported: the legacy Python ' ...
            'analysis scripts it called do not exist in this repository.']);
    end
    env = fsms_paths();
    manifest_csv = char(string(cfg.manifest_csv));
    output_dir = char(string(cfg.output_dir));
    fsms_assert_unrestricted(manifest_csv, 'reading the manifest');
    fsms_assert_unrestricted(output_dir, 'writing hierarchy output');
    if ~isfile(manifest_csv)
        error('FSMS:ManifestMissing', 'Manifest not found: %s', manifest_csv);
    end
    if ~isfolder(output_dir), mkdir(output_dir); end

    mode = char(string(cfg.labelling_mode));
    explicit_labels = cellstr(string(cfg.global_manual_labels(:)));
    if isempty(mode)
        if ~isempty(explicit_labels)
            mode = 'explicit';
        elseif cfg.interactive_global_labelling || cfg.relabel_global_states
            mode = 'interactive';
        else
            mode = 'prespecified';
        end
    end
    if ~any(strcmp(mode, {'prespecified', 'explicit', 'interactive', 'template_auto'}))
        error('FSMS:LabellingMode', 'Unknown labelling_mode "%s".', mode);
    end
    if strcmp(mode, 'explicit') && isempty(explicit_labels)
        error('FSMS:LabellingMode', 'labelling_mode explicit needs global_manual_labels.');
    end

    common = struct( ...
        'K_candidates', 6, ...
        'spm_prior_weight', double(cfg.spm_prior_weight), ...
        'ecg_qc_dir', char(cfg.ecg_qc_dir), ...
        'preictal_window_refit_only', logical(cfg.preictal_window_refit_only), ...
        'window_refit_length_s', double(cfg.window_refit_length_s), ...
        'window_refit_step_s', double(cfg.window_refit_step_s), ...
        'window_refit_time_window_s', double(cfg.window_refit_time_window_s), ...
        'window_refit_min_peaks', double(cfg.window_refit_min_peaks), ...
        'window_refit_min_state_peaks', double(cfg.window_refit_min_state_peaks), ...
        'window_refit_start_record', double(cfg.window_refit_start_record), ...
        'window_refit_max_records', double(cfg.window_refit_max_records), ...
        'window_refit_csv', char(cfg.window_refit_csv), ...
        'compact_results_mat', logical(cfg.compact_results_mat), ...
        'run_backfit', logical(cfg.run_backfit), ...
        'run_participant_condition_fits', logical(cfg.run_participant_condition_fits), ...
        'run_global_backfit', logical(cfg.run_global_backfit), ...
        'run_cardiac_phase', logical(cfg.run_cardiac_phase), ...
        'backfit_time_bin_s', double(cfg.backfit_time_bin_s), ...
        'backfit_time_window_s', double(cfg.backfit_time_window_s), ...
        'resume', ~logical(cfg.redo), ...
        'save_plots', logical(cfg.save_plots), ...
        'infer_localizer_dir', false);
    template_args = {};
    if ~isempty(char(string(cfg.template_file)))
        template_args = {'template_file', char(string(cfg.template_file))};
    end
    extra = cfg.extra_hierarchy_args;

    labelling = struct();
    switch mode
        case 'prespecified'
            if cfg.redo
                error('FSMS:RedoNotSupported', ['redo is not supported with the pre-specified labelling ' ...
                    '(stale child caches would be mixed with a refitted global solution). Use a fresh output_dir.']);
            end
            % Pass 1: pooled GFP peaks + global K=6 fit. No template alignment (template_file empty) so the
            % cached global solution keeps the raw fitted-state order the labels will refer to.
            solution_file = fullfile(output_dir, 'global', 'global_solution.mat');
            labels_mat = fullfile(output_dir, 'global', 'global_manual_labels.mat');
            if ~isfile(solution_file) || ~isfile(labels_mat)
                fprintf('Pass 1/2: global K=6 fit without template alignment...\n');
                run_hierarchy(manifest_csv, output_dir, common, ...
                    [{'global_only', true, 'template_file', '', 'require_template_labels', false, ...
                      'global_manual_labels', {}, 'interactive_global_labelling', false}, extra]);
            end
            labelling = prespecified_global_labels(output_dir, 'template_file', char(string(cfg.template_file)));
            fprintf('Pass 2/2: hierarchy with the pre-specified labels...\n');
            [HResults, results_mat] = run_hierarchy(manifest_csv, output_dir, common, ...
                [{'global_only', logical(cfg.global_only), 'require_template_labels', logical(cfg.require_template_labels), ...
                  'global_manual_labels', {}, 'interactive_global_labelling', false, ...
                  'relabel_global_states', false}, template_args, extra]);
            verify_prespecified_labels(output_dir, labelling.labels);
        case 'explicit'
            [HResults, results_mat] = run_hierarchy(manifest_csv, output_dir, common, ...
                [{'global_only', logical(cfg.global_only), 'require_template_labels', logical(cfg.require_template_labels), ...
                  'global_manual_labels', explicit_labels, 'interactive_global_labelling', false, ...
                  'relabel_global_states', logical(cfg.relabel_global_states)}, template_args, extra]);
        case 'interactive'
            [HResults, results_mat] = run_hierarchy(manifest_csv, output_dir, common, ...
                [{'global_only', logical(cfg.global_only), 'require_template_labels', logical(cfg.require_template_labels), ...
                  'global_manual_labels', {}, 'interactive_global_labelling', true, ...
                  'relabel_global_states', logical(cfg.relabel_global_states)}, template_args, extra]);
        case 'template_auto'
            [HResults, results_mat] = run_hierarchy(manifest_csv, output_dir, common, ...
                [{'global_only', logical(cfg.global_only), 'require_template_labels', logical(cfg.require_template_labels), ...
                  'global_manual_labels', {}, 'interactive_global_labelling', false, ...
                  'relabel_global_states', false}, template_args, extra]);
    end

    omnibus = struct();
    if cfg.run_topoplots || cfg.run_tanova
        omnibus = analyse_hierarchical_outputs(results_mat, ...
            'output_dir', fullfile(output_dir, 'omnibus_analysis'), ...
            'run_topoplots', logical(cfg.run_topoplots), ...
            'run_tanova', logical(cfg.run_tanova), ...
            'semiology_manifest_csv', manifest_csv, ...
            'tanova_permutations', double(cfg.tanova_permutations));
    end
    outputs = struct('HResults', HResults, 'results_mat', results_mat, 'omnibus', omnibus, ...
        'labelling_mode', mode, 'labelling', labelling, 'env', env);
end

function [HResults, results_mat] = run_hierarchy(manifest_csv, output_dir, common, overrides)
    args = [struct_to_args(common), overrides];
    [HResults, results_mat] = fit_microstate_hierarchical_csv(manifest_csv, 'output_dir', output_dir, args{:});
end

function args = struct_to_args(S)
    names = fieldnames(S);
    args = cell(1, 2 * numel(names));
    for i = 1:numel(names)
        args{2 * i - 1} = names{i};
        args{2 * i} = S.(names{i});
    end
end

function verify_prespecified_labels(output_dir, labels)
% Pass 2 must have applied exactly the pre-specified labels (via the saved-label path).
    S = load(fullfile(output_dir, 'global', 'global_solution.mat'), 'Fit');
    fit_labels = fsms_global_fit_labels(output_dir);
    if ~isequal(cellstr(fit_labels(:)), cellstr(string(labels(:))))
        error('FSMS:LabelsNotApplied', 'Hierarchy fitted-state labels (%s) differ from the pre-specified labels (%s).', ...
            strjoin(fit_labels, ','), strjoin(string(labels), ','));
    end
    source = '';
    if isfield(S.Fit.template_alignment, 'manual_label_source')
        source = char(string(S.Fit.template_alignment.manual_label_source));
    end
    if ~strcmp(source, 'saved_manual_labels')
        error('FSMS:LabelsNotApplied', ['Pass 2 did not take the saved pre-specified labels ' ...
            '(manual_label_source = "%s").'], source);
    end
end
