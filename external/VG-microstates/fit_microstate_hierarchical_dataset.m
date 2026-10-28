function [HResults, results_mat] = fit_microstate_hierarchical_dataset(input_path, varargin)
% FIT_MICROSTATE_HIERARCHICAL_DATASET Dataset-wise hierarchical VB-GMM microstates.
% By default, runs off local LEMON
%
% Pipeline:
%   preprocess/filter -> GFP peaks -> global VB-GMM -> group/condition/
%   participant VB-GMM fits with parent component-mean priors.
%
% Example:
%   H = fit_microstate_hierarchical_dataset();

    setup_microstate_paths();
    if nargin >= 1 && isstruct(input_path) && isfield(input_path, 'command')
        [HResults, results_mat] = run_recovery_command(input_path);
        return;
    end
    DEFAULT_FOLDER = fullfile(getenv('HOME'), 'EEG', 'LEMON');

    if nargin < 1 || isempty(input_path)
        input_path = DEFAULT_FOLDER;
    elseif (ischar(input_path) || isstring(input_path)) && strcmpi(strtrim(char(input_path)), 'LEMON')
        input_path = DEFAULT_FOLDER;
    end

    t0 = tic;
    util = microstate_utilities();
    repo_cfg = util.load_config();
    path_defaults = repo_cfg.paths;
    pre_defaults = repo_cfg.preprocessing;
    hier_defaults = repo_cfg.hierarchical;
    max_child_maps_default = double(hier_defaults.max_global_maps);
    if isfield(hier_defaults, 'max_child_maps')
        max_child_maps_default = double(hier_defaults.max_child_maps);
    end
    spm_prior_weight_default = 1;
    if isfield(hier_defaults, 'spm_prior_weight')
        spm_prior_weight_default = double(hier_defaults.spm_prior_weight);
    end

    p = inputParser;
    addRequired(p, 'input_path', @(x) ischar(x) || isstring(x));
    addParameter(p, 'output_dir', path_defaults.hierarchical_output_dir, @(x) ischar(x) || isstring(x));
    addParameter(p, 'template_file', path_defaults.template_file, @(x) ischar(x) || isstring(x));
    addParameter(p, 'template_label_override', {}, @(x) iscell(x) || isstring(x));
    addParameter(p, 'global_manual_labels', {}, @(x) iscell(x) || isstring(x));
    addParameter(p, 'interactive_global_labelling', false, @islogical);
    addParameter(p, 'relabel_global_states', false, @islogical);
    addParameter(p, 'require_template_labels', false, @islogical);
    addParameter(p, 'template_reference_K', 7, @(x) isnumeric(x) && isscalar(x) && x >= 2);
    addParameter(p, 'resume', true, @islogical);
    addParameter(p, 'K_candidates', double(hier_defaults.K_candidates(:)'), @isnumeric);
    addParameter(p, 'criterion', char(hier_defaults.criterion), @(x) ischar(x) || isstring(x));
    addParameter(p, 'apply_average_reference', logical(pre_defaults.apply_average_reference), @islogical);
    addParameter(p, 'spatial_filter', char(pre_defaults.spatial_filter), @(x) ischar(x) || isstring(x));
    addParameter(p, 'spatial_filter_neighbours', double(pre_defaults.spatial_filter_neighbours), @isnumeric);
    addParameter(p, 'spatial_filter_strength', double(pre_defaults.spatial_filter_strength), @isnumeric);
    addParameter(p, 'filter_band', double(pre_defaults.filter_band(:)'), @(x) isempty(x) || isnumeric(x));
    addParameter(p, 'use_scalp_channels', true, @islogical);
    addParameter(p, 'scalp_channel_max_radius', 0.60, @(x) isnumeric(x) && isscalar(x) && x > 0 && x <= 1);
    addParameter(p, 'exclude_channels', {'PO9', 'PO10'}, @(x) iscell(x) || isstring(x));
    addParameter(p, 'interpolate_missing_channels', true, @islogical);
    addParameter(p, 'localizer_dir', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'infer_localizer_dir', true, @islogical);
    addParameter(p, 'gfp_peak_min_distance', 3, @isnumeric);
    addParameter(p, 'gfp_peak_threshold_schedule', [0.50 0.60 0.70 0.80 0.90], @isnumeric);
    addParameter(p, 'min_gfp_peaks_per_file', 20, @isnumeric);
    addParameter(p, 'reject_gfp_peak_outliers', logical(pre_defaults.reject_gfp_peak_outliers), @islogical);
    addParameter(p, 'gfp_outlier_mad_multiplier', double(pre_defaults.gfp_outlier_mad_multiplier), @isnumeric);
    addParameter(p, 'max_maps_per_file', [], @(x) isempty(x) || isnumeric(x));
    addParameter(p, 'max_global_maps', double(hier_defaults.max_global_maps), @(x) isempty(x) || isnumeric(x));
    addParameter(p, 'max_child_maps', max_child_maps_default, @(x) isempty(x) || isnumeric(x));
    addParameter(p, 'spm_prior_weight', spm_prior_weight_default, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'global_only', false, @islogical);
    addParameter(p, 'run_participant_condition_fits', true, @islogical);
    addParameter(p, 'run_backfit', true, @islogical);
    addParameter(p, 'run_global_backfit', true, @islogical);
    addParameter(p, 'run_cardiac_phase', true, @islogical);
    addParameter(p, 'state_covariance_metrics_csv', 'hierarchical_state_covariance_metrics.csv', @(x) ischar(x) || isstring(x));
    addParameter(p, 'backfit_state_metrics_csv', 'participant_condition_state_backfit_metrics.csv', @(x) ischar(x) || isstring(x));
    addParameter(p, 'backfit_pairwise_metrics_csv', 'participant_condition_state_pairwise_backfit_metrics.csv', @(x) ischar(x) || isstring(x));
    addParameter(p, 'backfit_record_summary_csv', 'participant_condition_record_backfit_summary.csv', @(x) ischar(x) || isstring(x));
    addParameter(p, 'backfit_preictal_time_metrics_csv', 'participant_condition_preictal_time_metrics.csv', @(x) ischar(x) || isstring(x));
    addParameter(p, 'backfit_transition_metrics_csv', 'participant_condition_transition_metrics.csv', @(x) ischar(x) || isstring(x));
    addParameter(p, 'global_backfit_state_metrics_csv', 'global_state_backfit_metrics.csv', @(x) ischar(x) || isstring(x));
    addParameter(p, 'global_backfit_pairwise_metrics_csv', 'global_state_pairwise_backfit_metrics.csv', @(x) ischar(x) || isstring(x));
    addParameter(p, 'global_backfit_record_summary_csv', 'global_record_backfit_summary.csv', @(x) ischar(x) || isstring(x));
    addParameter(p, 'global_backfit_preictal_time_metrics_csv', 'global_preictal_time_metrics.csv', @(x) ischar(x) || isstring(x));
    addParameter(p, 'global_backfit_transition_metrics_csv', 'global_transition_metrics.csv', @(x) ischar(x) || isstring(x));
    addParameter(p, 'global_c_cardiac_phase_metrics_csv', 'global_ace_cardiac_phase_metrics.csv', @(x) ischar(x) || isstring(x));
    addParameter(p, 'participant_ce_cardiac_phase_metrics_csv', 'participant_ace_cardiac_phase_metrics.csv', @(x) ischar(x) || isstring(x));
    addParameter(p, 'ecg_qc_dir', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'preictal_morphology_only', false, @islogical);
    addParameter(p, 'preictal_morphology_csv', 'preictal_state_morphology_10s.csv', @(x) ischar(x) || isstring(x));
    addParameter(p, 'preictal_window_refit_only', false, @islogical);
    addParameter(p, 'window_refit_length_s', 60, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'window_refit_step_s', 30, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'window_refit_time_window_s', 1200, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'window_refit_min_peaks', 30, @(x) isnumeric(x) && isscalar(x) && x >= 2);
    addParameter(p, 'window_refit_min_state_peaks', 5, @(x) isnumeric(x) && isscalar(x) && x >= 1);
    addParameter(p, 'window_refit_csv', 'preictal_window_refit_maps.csv', @(x) ischar(x) || isstring(x));
    addParameter(p, 'window_refit_start_record', 1, @(x) isnumeric(x) && isscalar(x) && x >= 1);
    addParameter(p, 'window_refit_max_records', Inf, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'channel_positions_csv', 'common_channel_positions.csv', @(x) ischar(x) || isstring(x));
    addParameter(p, 'compact_results_mat', false, @islogical);
    addParameter(p, 'backfit_time_bin_s', 60, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'backfit_time_window_s', Inf, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'run_backfit_axis_dynamics', false, @islogical);
    addParameter(p, 'backfit_axis_dynamics_csv', 'participant_condition_state_axis_dynamics.csv', @(x) ischar(x) || isstring(x));
    addParameter(p, 'backfit_axis_min_samples', 16, @isnumeric);
    addParameter(p, 'random_seed', 1, @isnumeric);
    addParameter(p, 'save_plots', true, @islogical);
    addParameter(p, 'verbose', true, @islogical);
    if is_parameter_name(input_path, p)
        varargin = [{input_path}, varargin];
        input_path = DEFAULT_FOLDER;
    end
    parse(p, input_path, varargin{:});
    cfg = p.Results;

    cfg.input_path = char(cfg.input_path);
    cfg.output_dir = util.resolve_path(cfg.output_dir, util.project_root());
    cfg.template_file = util.resolve_path(cfg.template_file, util.project_root());
    cfg.template_label_override = cellstr(string(cfg.template_label_override));
    cfg.global_manual_labels = cellstr(string(cfg.global_manual_labels));
    cfg.criterion = lower(char(cfg.criterion));
    cfg.spatial_filter = char(cfg.spatial_filter);
    cfg.exclude_channels = cellstr(string(cfg.exclude_channels));
    cfg.localizer_dir = resolve_localizer_dir(cfg.localizer_dir, cfg.input_path, util, cfg.infer_localizer_dir);
    cfg.state_covariance_metrics_csv = char(cfg.state_covariance_metrics_csv);
    cfg.backfit_state_metrics_csv = char(cfg.backfit_state_metrics_csv);
    cfg.backfit_pairwise_metrics_csv = char(cfg.backfit_pairwise_metrics_csv);
    cfg.backfit_record_summary_csv = char(cfg.backfit_record_summary_csv);
    cfg.backfit_preictal_time_metrics_csv = char(cfg.backfit_preictal_time_metrics_csv);
    cfg.backfit_transition_metrics_csv = char(cfg.backfit_transition_metrics_csv);
    cfg.global_backfit_state_metrics_csv = char(cfg.global_backfit_state_metrics_csv);
    cfg.global_backfit_pairwise_metrics_csv = char(cfg.global_backfit_pairwise_metrics_csv);
    cfg.global_backfit_record_summary_csv = char(cfg.global_backfit_record_summary_csv);
    cfg.global_backfit_preictal_time_metrics_csv = char(cfg.global_backfit_preictal_time_metrics_csv);
    cfg.global_backfit_transition_metrics_csv = char(cfg.global_backfit_transition_metrics_csv);
    cfg.global_c_cardiac_phase_metrics_csv = char(cfg.global_c_cardiac_phase_metrics_csv);
    cfg.participant_ce_cardiac_phase_metrics_csv = char(cfg.participant_ce_cardiac_phase_metrics_csv);
    cfg.ecg_qc_dir = char(cfg.ecg_qc_dir);
    cfg.preictal_morphology_csv = char(cfg.preictal_morphology_csv);
    cfg.window_refit_csv = char(cfg.window_refit_csv);
    cfg.channel_positions_csv = char(cfg.channel_positions_csv);
    cfg.backfit_axis_dynamics_csv = char(cfg.backfit_axis_dynamics_csv);
    cfg.K_candidates = unique(round(double(cfg.K_candidates(:)')));
    cfg.K_candidates = cfg.K_candidates(cfg.K_candidates >= 2);
    if isempty(cfg.K_candidates)
        error('K_candidates must contain at least one value >= 2.');
    end
    rng(cfg.random_seed, 'twister');

    ensure_dir(cfg.output_dir);
    dirs = make_output_dirs(cfg.output_dir);

    [spm_ok, spm_info] = util.ensure_spm_mix('', path_defaults.spm_mixture_paths, cfg.verbose);
    if ~spm_ok
        error('SPM spm_mix not found. Checked: %s', strjoin(spm_info.attempted, ', '));
    end
    if exist('spm_mix_prior', 'file') ~= 2
        error('Current SPM_MS spm_mix_prior.m is required for hierarchical fitting.');
    end

    manifest = read_or_build_manifest(cfg.input_path);
    if isempty(cfg.localizer_dir) && cfg.infer_localizer_dir
        cfg.localizer_dir = infer_localizer_dir_from_manifest(manifest);
    end
    writetable(manifest, fullfile(cfg.output_dir, 'normalised_input_manifest.csv'));
    if cfg.verbose
        fprintf('\nHierarchical VB-GMM microstates\n');
        fprintf('Input rows: %d\nOutput: %s\n', height(manifest), cfg.output_dir);
        fprintf('K candidates: %s\n', mat2str(cfg.K_candidates));
        fprintf('Excluded channels: %s\n', strjoin(cfg.exclude_channels, ', '));
        fprintf('Channel interpolation: %s\n', onoff(cfg.interpolate_missing_channels));
        if ~isempty(cfg.localizer_dir)
            fprintf('Digitized localizers: %s\n', cfg.localizer_dir);
        end
        fprintf('\n');
    end

    if cfg.preictal_window_refit_only
        self_check_preictal_window_refit();
        solution_file = fullfile(dirs.global, 'global_solution.mat');
        if ~isfile(solution_file)
            error('Cached global fit is required for preictal_window_refit_only: %s', solution_file);
        end
        cached = load(solution_file, 'common_labels', 'common_chanlocs');
        common_labels = cached.common_labels;
        common_chanlocs = cached.common_chanlocs;
        common_can = util.canonical_channel_labels(common_labels);
        common_pos = util.positions_from_chanlocs(common_chanlocs, numel(common_labels));
        refits = compile_preictal_window_refit_table(manifest, dirs.peaks, dirs.participants, ...
            common_can, cfg);
        refit_csv = output_csv_path(cfg.output_dir, cfg.window_refit_csv);
        positions_csv = output_csv_path(cfg.output_dir, cfg.channel_positions_csv);
        writetable(refits, refit_csv);
        writetable(table((1:numel(common_can))', string(common_can(:)), common_pos(:,1), common_pos(:,2), common_pos(:,3), ...
            'VariableNames', {'channel_index','channel_label','x','y','z'}), positions_csv);
        [~, refit_stem] = fileparts(refit_csv);
        results_mat = fullfile(cfg.output_dir, sprintf('%s_results.mat', refit_stem));
        HResults = struct('source', 'preictal_window_refit_only', 'window_refit_csv', refit_csv, ...
            'channel_positions_csv', positions_csv, 'n_rows', height(refits));
        if cfg.verbose
            fprintf('Preictal window-refit rows: %d\n%s\n', height(refits), refit_csv);
        end
        return;
    end

    if cfg.preictal_morphology_only
        self_check_preictal_morphology();
        solution_file = fullfile(dirs.global, 'global_solution.mat');
        if ~isfile(solution_file)
            error('Cached global fit is required for preictal_morphology_only: %s', solution_file);
        end
        cached = load(solution_file, 'Fit', 'common_labels', 'common_chanlocs');
        global_fit = cached.Fit;
        common_labels = cached.common_labels;
        common_chanlocs = cached.common_chanlocs;
        common_can = util.canonical_channel_labels(common_labels);
        common_pos = util.positions_from_chanlocs(common_chanlocs, numel(common_labels));
        morphology = compile_preictal_morphology_table(manifest, global_fit, dirs.participants, ...
            common_can, common_pos, cfg, util);
        morphology_csv = output_csv_path(cfg.output_dir, cfg.preictal_morphology_csv);
        positions_csv = output_csv_path(cfg.output_dir, cfg.channel_positions_csv);
        writetable(morphology, morphology_csv);
        writetable(table((1:numel(common_can))', string(common_can(:)), common_pos(:,1), common_pos(:,2), common_pos(:,3), ...
            'VariableNames', {'channel_index','channel_label','x','y','z'}), positions_csv);
        results_mat = fullfile(cfg.output_dir, 'hierarchical_microstate_results.mat');
        HResults = struct('source', 'preictal_morphology_only', 'morphology_csv', morphology_csv, ...
            'channel_positions_csv', positions_csv, 'n_rows', height(morphology));
        if cfg.verbose
            fprintf('Preictal morphology rows: %d\n%s\n', height(morphology), morphology_csv);
        end
        return;
    end

    common_metadata_file = fullfile(cfg.output_dir, 'common_channel_metadata.mat');
    peak_cache_complete = true;
    for cache_i = 1:height(manifest)
        cache_file = fullfile(dirs.peaks, sprintf('%03d_%s_%s_gfp_peaks.mat', cache_i, ...
            clean_key(manifest.participant(cache_i)), clean_key(manifest.condition(cache_i))));
        peak_cache_complete = peak_cache_complete && isfile(cache_file);
    end
    using_complete_peak_cache = cfg.resume && peak_cache_complete && isfile(common_metadata_file);
    if using_complete_peak_cache
        cached_common = load(common_metadata_file, 'common_labels', 'common_can', 'common_chanlocs', 'common_pos');
        common_labels = cached_common.common_labels;
        common_can = cached_common.common_can;
        common_chanlocs = cached_common.common_chanlocs;
        common_pos = cached_common.common_pos;
        meta = repmat(struct(), height(manifest), 1);
        if cfg.verbose
            fprintf('Reusing complete prior-independent GFP peak and common-channel cache.\n');
        end
    else
        meta = inspect_files(manifest, cfg, util);
        [common_labels, common_can, common_chanlocs, common_pos] = common_channel_set(meta, cfg);
        save(common_metadata_file, 'common_labels', 'common_can', 'common_chanlocs', 'common_pos', '-v7.3');
    end
    if numel(common_labels) < 8
        error('Only %d common channels remain after filtering/exclusion.', numel(common_labels));
    end
    writetable(table((1:numel(common_labels))', string(common_labels(:)), ...
        'VariableNames', {'index', 'label'}), fullfile(cfg.output_dir, 'common_channels.csv'));
    if cfg.verbose
        fprintf('Common channels: %d\n\n', numel(common_labels));
    end

    peak_records = cell(height(manifest), 1);
    pooled_maps_parts = cell(height(manifest), 1);
    pooled_gfp_parts = cell(height(manifest), 1);
    pooled_row_parts = cell(height(manifest), 1);
    for i = 1:height(manifest)
        if cfg.verbose
            fprintf('[%d/%d] GFP peaks: %s %s\n', i, height(manifest), manifest.participant(i), manifest.condition(i));
        end
        peak_file = fullfile(dirs.peaks, sprintf('%03d_%s_%s_gfp_peaks.mat', i, clean_key(manifest.participant(i)), clean_key(manifest.condition(i))));
        rec = [];
        if cfg.resume && isfile(peak_file)
            cached = load(peak_file, 'rec');
            if isfield(cached, 'rec')
                rec = cached.rec;
            end
        end
        if isempty(rec)
            if using_complete_peak_cache
                error('Complete GFP cache entry is unreadable: %s', peak_file);
            end
            rec = extract_file_peaks(manifest.file_path(i), meta(i), common_can, common_pos, cfg, util);
            save(peak_file, 'rec', 'cfg', '-v7.3');
        end
        peak_records{i} = rec;

        n = size(rec.maps_norm, 1);
        pooled_maps_parts{i} = double(rec.maps_norm);
        pooled_gfp_parts{i} = double(rec.gfp_peak(:));
        rows = table();
        rows.file_index = repmat(i, n, 1);
        rows.participant = repmat(manifest.participant(i), n, 1);
        rows.condition = repmat(manifest.condition(i), n, 1);
        rows.group = repmat(manifest.group(i), n, 1);
        rows.file_path = repmat(manifest.file_path(i), n, 1);
        rows.peak_sample = rec.peak_sample(:);
        rows.gfp = double(rec.gfp_peak(:));
        pooled_row_parts{i} = rows;
    end
    pooled_maps = vertcat(pooled_maps_parts{:});
    pooled_gfp = vertcat(pooled_gfp_parts{:});
    pooled_rows = vertcat(pooled_row_parts{:});
    clear pooled_maps_parts pooled_gfp_parts pooled_row_parts
    pooled_manifest_file = fullfile(cfg.output_dir, 'pooled_gfp_peak_manifest.csv');
    if ~cfg.resume || ~isfile(pooled_manifest_file)
        writetable(pooled_rows, pooled_manifest_file);
    end

    fit_idx = deterministic_subsample(size(pooled_maps, 1), cfg.max_global_maps);
    global_maps = pooled_maps(fit_idx, :);
    global_gfp = pooled_gfp(fit_idx);
    if cfg.verbose && numel(fit_idx) < size(pooled_maps, 1)
        fprintf('Global fit uses deterministic cap: %d/%d peaks\n', numel(fit_idx), size(pooled_maps, 1));
    end

    if cfg.verbose
        fprintf('\nGlobal VB-GMM fit...\n');
    end
    global_fit = load_cached_fit(dirs.global, 'global', cfg, common_labels, common_chanlocs);
    if isempty(global_fit)
        global_fit = fit_spm_vb_peak_maps(global_maps, global_gfp, cfg.K_candidates, cfg, 'global', [], 'none');
    end
    [manual_labels, global_labels_changed, manual_label_source] = resolve_global_manual_labels( ...
        global_fit, cfg, dirs.global, common_chanlocs);
    if isempty(manual_labels)
        global_fit = attach_template_alignment_if_available(global_fit, cfg, common_labels, common_chanlocs);
        global_labels_changed = false;
    else
        global_fit = apply_manual_global_labels_by_state_order(global_fit, manual_labels, cfg, common_labels, common_chanlocs);
        global_fit.template_alignment.manual_label_source = manual_label_source;
    end
    global_rows = pooled_rows(fit_idx, :);
    global_rows.cluster_label = assign_by_abs_correlation(global_maps, global_fit.centers);
    save_fit_bundle(dirs.global, 'global', global_fit, global_rows, common_labels, common_chanlocs, cfg);
    if global_labels_changed && cfg.verbose
        fprintf(['Global labels changed: reusing GFP preprocessing and fitted maps; ' ...
            'label-dependent backfit tables will be rebuilt.\n']);
    end

    K = global_fit.K_estimated;
    child_cfg = cfg;
    child_cfg.K_candidates = K;

    group_nodes = empty_nodes(size(global_fit.centers, 2));
    condition_nodes = empty_nodes(size(global_fit.centers, 2));
    participant_nodes = empty_nodes(size(global_fit.centers, 2));
    participant_condition_nodes = empty_nodes(size(global_fit.centers, 2));
    group_summary = table();
    condition_summary = table();
    participant_summary = table();
    participant_condition_summary = table();
    if cfg.global_only
        if cfg.verbose
            fprintf('Global-only mode: skipping group, condition, participant, and backfit fits.\n');
        end
    else
        [group_nodes, group_summary] = fit_group_nodes(pooled_maps, pooled_gfp, pooled_rows, global_fit, child_cfg, dirs.groups, common_labels, common_chanlocs);
        [condition_nodes, condition_summary] = fit_condition_nodes(pooled_maps, pooled_gfp, pooled_rows, global_fit, child_cfg, dirs.conditions, common_labels, common_chanlocs);
        [participant_nodes, participant_summary] = fit_participant_nodes(pooled_maps, pooled_gfp, pooled_rows, manifest, global_fit, group_nodes, child_cfg, dirs.participants, common_labels, common_chanlocs);
        if cfg.run_participant_condition_fits
            [participant_condition_nodes, participant_condition_summary] = fit_participant_condition_nodes( ...
                pooled_maps, pooled_gfp, pooled_rows, manifest, global_fit, group_nodes, condition_nodes, ...
                participant_nodes, child_cfg, dirs.participant_conditions, common_labels, common_chanlocs);
        elseif cfg.verbose
            fprintf('Focused mode: skipping participant-condition fits; backfit will use participant templates.\n');
        end
        assert_fixed_hierarchy_k(K, group_nodes, condition_nodes, participant_nodes, participant_condition_nodes);
    end

    hierarchy_summary = [ ...
        make_summary_row('global', "global", "", "", "", global_fit, height(manifest), size(global_maps, 1), dirs.global); ...
        group_summary; condition_summary; participant_summary; participant_condition_summary];
    writetable(hierarchy_summary, fullfile(cfg.output_dir, 'hierarchical_fit_summary.csv'));
    state_covariance_metrics = compile_state_covariance_metrics(global_fit, group_nodes, condition_nodes, participant_nodes, participant_condition_nodes, size(global_maps, 1), dirs.global);
    state_covariance_metrics_csv = output_csv_path(cfg.output_dir, cfg.state_covariance_metrics_csv);
    if ~isempty(state_covariance_metrics)
        writetable(state_covariance_metrics, state_covariance_metrics_csv);
    end

    backfit_state_metrics = table();
    backfit_pairwise_metrics = table();
    backfit_record_summary = table();
    backfit_axis_dynamics = table();
    backfit_preictal_time_metrics = table();
    backfit_transition_metrics = table();
    participant_ce_cardiac_phase_metrics = table();
    backfit_state_metrics_csv = '';
    backfit_pairwise_metrics_csv = '';
    backfit_record_summary_csv = '';
    backfit_axis_dynamics_csv = '';
    backfit_preictal_time_metrics_csv = '';
    backfit_transition_metrics_csv = '';
    participant_ce_cardiac_phase_metrics_csv = '';
    global_backfit_state_metrics = table();
    global_backfit_pairwise_metrics = table();
    global_backfit_record_summary = table();
    global_backfit_preictal_time_metrics = table();
    global_backfit_transition_metrics = table();
    global_c_cardiac_phase_metrics = table();
    global_backfit_state_metrics_csv = '';
    global_backfit_pairwise_metrics_csv = '';
    global_backfit_record_summary_csv = '';
    global_backfit_preictal_time_metrics_csv = '';
    global_backfit_transition_metrics_csv = '';
    global_c_cardiac_phase_metrics_csv = '';
    if cfg.run_backfit && ~cfg.global_only
        backfit_state_metrics_csv = output_csv_path(cfg.output_dir, cfg.backfit_state_metrics_csv);
        backfit_pairwise_metrics_csv = output_csv_path(cfg.output_dir, cfg.backfit_pairwise_metrics_csv);
        backfit_record_summary_csv = output_csv_path(cfg.output_dir, cfg.backfit_record_summary_csv);
        backfit_axis_dynamics_csv = output_csv_path(cfg.output_dir, cfg.backfit_axis_dynamics_csv);
        backfit_preictal_time_metrics_csv = output_csv_path(cfg.output_dir, cfg.backfit_preictal_time_metrics_csv);
        backfit_transition_metrics_csv = output_csv_path(cfg.output_dir, cfg.backfit_transition_metrics_csv);
        participant_ce_cardiac_phase_metrics_csv = output_csv_path(cfg.output_dir, cfg.participant_ce_cardiac_phase_metrics_csv);
        if cfg.verbose
            fprintf('\nBackfitting participant templates to full records...\n');
        end
        participant_cache = {backfit_state_metrics_csv, backfit_pairwise_metrics_csv, backfit_record_summary_csv, ...
            backfit_preictal_time_metrics_csv, backfit_transition_metrics_csv, participant_ce_cardiac_phase_metrics_csv};
        if cfg.run_backfit_axis_dynamics, participant_cache{end+1} = backfit_axis_dynamics_csv; end
        if cfg.resume && ~global_labels_changed && all(cellfun(@isfile, participant_cache))
            backfit_state_metrics = readtable(backfit_state_metrics_csv, 'TextType', 'string');
            backfit_pairwise_metrics = readtable(backfit_pairwise_metrics_csv, 'TextType', 'string');
            backfit_record_summary = readtable(backfit_record_summary_csv, 'TextType', 'string');
            backfit_preictal_time_metrics = readtable(backfit_preictal_time_metrics_csv, 'TextType', 'string');
            backfit_transition_metrics = readtable(backfit_transition_metrics_csv, 'TextType', 'string');
            participant_ce_cardiac_phase_metrics = readtable(participant_ce_cardiac_phase_metrics_csv, 'TextType', 'string');
            if cfg.run_backfit_axis_dynamics
                backfit_axis_dynamics = readtable(backfit_axis_dynamics_csv, 'TextType', 'string');
            end
            if cfg.verbose, fprintf('Reused cached participant backfit metrics.\n'); end
        else
            [backfit_state_metrics, backfit_pairwise_metrics, backfit_record_summary, backfit_axis_dynamics, backfit_preictal_time_metrics, backfit_transition_metrics, participant_ce_cardiac_phase_metrics] = ...
                compile_backfit_metric_tables(manifest, participant_nodes, participant_condition_nodes, common_can, common_pos, cfg, util);
            if ~isempty(backfit_state_metrics), writetable(backfit_state_metrics, backfit_state_metrics_csv); end
            if ~isempty(backfit_pairwise_metrics), writetable(backfit_pairwise_metrics, backfit_pairwise_metrics_csv); end
            if ~isempty(backfit_record_summary), writetable(backfit_record_summary, backfit_record_summary_csv); end
            if cfg.run_backfit_axis_dynamics && ~isempty(backfit_axis_dynamics), writetable(backfit_axis_dynamics, backfit_axis_dynamics_csv); end
            if ~isempty(backfit_preictal_time_metrics), writetable(backfit_preictal_time_metrics, backfit_preictal_time_metrics_csv); end
            if ~isempty(backfit_transition_metrics), writetable(backfit_transition_metrics, backfit_transition_metrics_csv); end
            if ~isempty(participant_ce_cardiac_phase_metrics), writetable(participant_ce_cardiac_phase_metrics, participant_ce_cardiac_phase_metrics_csv); end
        end

        if cfg.run_global_backfit
            global_backfit_state_metrics_csv = output_csv_path(cfg.output_dir, cfg.global_backfit_state_metrics_csv);
            global_backfit_pairwise_metrics_csv = output_csv_path(cfg.output_dir, cfg.global_backfit_pairwise_metrics_csv);
            global_backfit_record_summary_csv = output_csv_path(cfg.output_dir, cfg.global_backfit_record_summary_csv);
            global_backfit_preictal_time_metrics_csv = output_csv_path(cfg.output_dir, cfg.global_backfit_preictal_time_metrics_csv);
            global_backfit_transition_metrics_csv = output_csv_path(cfg.output_dir, cfg.global_backfit_transition_metrics_csv);
            global_c_cardiac_phase_metrics_csv = output_csv_path(cfg.output_dir, cfg.global_c_cardiac_phase_metrics_csv);
            if cfg.verbose
                fprintf('\nBackfitting the global template to full records...\n');
            end
            global_cache = {global_backfit_state_metrics_csv, global_backfit_pairwise_metrics_csv, global_backfit_record_summary_csv, ...
                global_backfit_preictal_time_metrics_csv, global_backfit_transition_metrics_csv, global_c_cardiac_phase_metrics_csv};
            if cfg.resume && ~global_labels_changed && all(cellfun(@isfile, global_cache))
                global_backfit_state_metrics = readtable(global_backfit_state_metrics_csv, 'TextType', 'string');
                global_backfit_pairwise_metrics = readtable(global_backfit_pairwise_metrics_csv, 'TextType', 'string');
                global_backfit_record_summary = readtable(global_backfit_record_summary_csv, 'TextType', 'string');
                global_backfit_preictal_time_metrics = readtable(global_backfit_preictal_time_metrics_csv, 'TextType', 'string');
                global_backfit_transition_metrics = readtable(global_backfit_transition_metrics_csv, 'TextType', 'string');
                global_c_cardiac_phase_metrics = readtable(global_c_cardiac_phase_metrics_csv, 'TextType', 'string');
                if cfg.verbose, fprintf('Reused cached global backfit metrics.\n'); end
            else
                global_backfit_cfg = cfg;
                global_backfit_cfg.run_backfit_axis_dynamics = false;
                [global_backfit_state_metrics, global_backfit_pairwise_metrics, global_backfit_record_summary, ~, global_backfit_preictal_time_metrics, global_backfit_transition_metrics, global_c_cardiac_phase_metrics] = ...
                    compile_backfit_metric_tables(manifest, participant_nodes, participant_condition_nodes, ...
                        common_can, common_pos, global_backfit_cfg, util, global_fit, "global");
                if ~isempty(global_backfit_state_metrics), writetable(global_backfit_state_metrics, global_backfit_state_metrics_csv); end
                if ~isempty(global_backfit_pairwise_metrics), writetable(global_backfit_pairwise_metrics, global_backfit_pairwise_metrics_csv); end
                if ~isempty(global_backfit_record_summary), writetable(global_backfit_record_summary, global_backfit_record_summary_csv); end
                if ~isempty(global_backfit_preictal_time_metrics), writetable(global_backfit_preictal_time_metrics, global_backfit_preictal_time_metrics_csv); end
                if ~isempty(global_backfit_transition_metrics), writetable(global_backfit_transition_metrics, global_backfit_transition_metrics_csv); end
                if ~isempty(global_c_cardiac_phase_metrics), writetable(global_c_cardiac_phase_metrics, global_c_cardiac_phase_metrics_csv); end
            end
        end
    end

    HResults = struct();
    HResults.source = 'fit_microstate_hierarchical_dataset';
    HResults.created = char(datetime('now', 'Format', 'yyyyMMdd''T''HHmmss'));
    HResults.cfg = cfg;
    HResults.manifest = manifest;
    HResults.common_labels = common_labels;
    HResults.common_channel_labels = common_labels;
    HResults.common_chanlocs = common_chanlocs;
    HResults.common_pos = common_pos;
    HResults.selected_K = K;
    HResults.global = build_node('global', "", "all", "", "global", global_fit, size(global_maps, 1), dirs.global);
    HResults.groups = group_nodes;
    HResults.conditions = condition_nodes;
    HResults.participants = participant_nodes;
    HResults.participant_level = participant_nodes;
    HResults.participant_conditions = participant_condition_nodes;
    HResults.participant_condition_level = participant_condition_nodes;
    HResults.hierarchy_summary = hierarchy_summary;
    HResults.state_covariance_metrics = state_covariance_metrics;
    HResults.state_covariance_metrics_csv = state_covariance_metrics_csv;
    HResults.peak_records = peak_records;
    HResults.backfit_state_metrics = backfit_state_metrics;
    HResults.backfit_state_metrics_csv = backfit_state_metrics_csv;
    HResults.backfit_pairwise_metrics = backfit_pairwise_metrics;
    HResults.backfit_pairwise_metrics_csv = backfit_pairwise_metrics_csv;
    HResults.backfit_record_summary = backfit_record_summary;
    HResults.backfit_record_summary_csv = backfit_record_summary_csv;
    HResults.backfit_axis_dynamics = backfit_axis_dynamics;
    HResults.backfit_axis_dynamics_csv = backfit_axis_dynamics_csv;
    HResults.backfit_preictal_time_metrics = backfit_preictal_time_metrics;
    HResults.backfit_preictal_time_metrics_csv = backfit_preictal_time_metrics_csv;
    HResults.backfit_transition_metrics = backfit_transition_metrics;
    HResults.backfit_transition_metrics_csv = backfit_transition_metrics_csv;
    HResults.participant_ce_cardiac_phase_metrics = participant_ce_cardiac_phase_metrics;
    HResults.participant_ce_cardiac_phase_metrics_csv = participant_ce_cardiac_phase_metrics_csv;
    HResults.global_backfit_state_metrics = global_backfit_state_metrics;
    HResults.global_backfit_state_metrics_csv = global_backfit_state_metrics_csv;
    HResults.global_backfit_pairwise_metrics = global_backfit_pairwise_metrics;
    HResults.global_backfit_pairwise_metrics_csv = global_backfit_pairwise_metrics_csv;
    HResults.global_backfit_record_summary = global_backfit_record_summary;
    HResults.global_backfit_record_summary_csv = global_backfit_record_summary_csv;
    HResults.global_backfit_preictal_time_metrics = global_backfit_preictal_time_metrics;
    HResults.global_backfit_preictal_time_metrics_csv = global_backfit_preictal_time_metrics_csv;
    HResults.global_backfit_transition_metrics = global_backfit_transition_metrics;
    HResults.global_backfit_transition_metrics_csv = global_backfit_transition_metrics_csv;
    HResults.global_c_cardiac_phase_metrics = global_c_cardiac_phase_metrics;
    HResults.global_c_cardiac_phase_metrics_csv = global_c_cardiac_phase_metrics_csv;
    HResults.runtime_s = toc(t0);

    results_mat = fullfile(cfg.output_dir, 'hierarchical_microstate_results.mat');
    if cfg.compact_results_mat
        payload = struct('HResults', compact_hresults_for_save(HResults));
        save(results_mat, '-struct', 'payload', '-v7');
    else
        save(results_mat, 'HResults', '-v7.3');
    end

    if cfg.verbose
        fprintf('\nDone in %.1f s\nResults: %s\n', HResults.runtime_s, results_mat);
        if ~isempty(backfit_state_metrics_csv)
            fprintf('Backfit state metrics CSV: %s\n', backfit_state_metrics_csv);
            fprintf('Backfit pairwise metrics CSV: %s\n', backfit_pairwise_metrics_csv);
            fprintf('Backfit record summary CSV: %s\n', backfit_record_summary_csv);
            fprintf('Participant transition metrics CSV: %s\n', backfit_transition_metrics_csv);
            if cfg.run_backfit_axis_dynamics && ~isempty(backfit_axis_dynamics_csv)
                fprintf('Backfit axis dynamics CSV: %s\n', backfit_axis_dynamics_csv);
            end
        end
        if ~isempty(state_covariance_metrics_csv)
            fprintf('State covariance metrics CSV: %s\n', state_covariance_metrics_csv);
        end
        if ~isempty(global_backfit_state_metrics_csv)
            fprintf('Global backfit state metrics CSV: %s\n', global_backfit_state_metrics_csv);
            fprintf('Global backfit pairwise metrics CSV: %s\n', global_backfit_pairwise_metrics_csv);
            fprintf('Global backfit record summary CSV: %s\n', global_backfit_record_summary_csv);
            fprintf('Global transition metrics CSV: %s\n', global_backfit_transition_metrics_csv);
            fprintf('Global C cardiac-phase metrics CSV: %s\n', global_c_cardiac_phase_metrics_csv);
        end
        if ~isempty(participant_ce_cardiac_phase_metrics_csv)
            fprintf('Participant C/E cardiac-phase metrics CSV: %s\n', participant_ce_cardiac_phase_metrics_csv);
        end
        print_alignment_line('Global', global_fit);
    end
end

function dirs = make_output_dirs(root_dir)
    dirs = struct();
    dirs.root = root_dir;
    dirs.peaks = fullfile(root_dir, 'gfp_peaks');
    dirs.global = fullfile(root_dir, 'global');
    dirs.groups = fullfile(root_dir, 'groups');
    dirs.conditions = fullfile(root_dir, 'conditions');
    dirs.participants = fullfile(root_dir, 'participants');
    dirs.participant_conditions = fullfile(root_dir, 'participant_conditions');
    names = fieldnames(dirs);
    for i = 1:numel(names)
        ensure_dir(dirs.(names{i}));
    end
end

function tf = is_parameter_name(value, parser)
    tf = false;
    if ~(ischar(value) || (isstring(value) && isscalar(value)))
        return;
    end
    tf = any(strcmpi(strtrim(char(value)), parser.Parameters));
end

function localizer_dir = resolve_localizer_dir(localizer_dir, input_path, util, infer_when_empty)
    if nargin < 4 || isempty(infer_when_empty)
        infer_when_empty = true;
    end
    localizer_dir = char(string(localizer_dir));
    if isempty(localizer_dir)
        if infer_when_empty
            localizer_dir = infer_localizer_dir(input_path);
        end
    else
        base = char(input_path);
        if isfile(base)
            base = fileparts(base);
        end
        localizer_dir = util.resolve_path(localizer_dir, base);
    end
    if ~isfolder(localizer_dir)
        localizer_dir = '';
    end
end

function localizer_dir = infer_localizer_dir(input_path)
    input_path = char(input_path);
    if isfile(input_path)
        input_path = fileparts(input_path);
    end
    candidates = string({ ...
        fullfile(input_path, 'EEG_Localizer_BIDS_ID'), ...
        fullfile(fileparts(input_path), 'EEG_Localizer_BIDS_ID'), ...
        fullfile(input_path, 'EEG_MPILMBB_LEMON', 'EEG_Localizer_BIDS_ID')});
    localizer_dir = '';
    for i = 1:numel(candidates)
        if isfolder(candidates(i))
            localizer_dir = char(candidates(i));
            return;
        end
    end
end

function localizer_dir = infer_localizer_dir_from_manifest(manifest)
    localizer_dir = '';
    if ~istable(manifest) || ~ismember('file_path', manifest.Properties.VariableNames)
        return;
    end
    for i = 1:height(manifest)
        localizer_dir = infer_localizer_dir(fileparts(char(manifest.file_path(i))));
        if ~isempty(localizer_dir)
            return;
        end
    end
end

function manifest = read_or_build_manifest(input_path)
    input_path = char(input_path);
    if isfolder(input_path)
        files = discover_eeg_files(input_path);
        if isempty(files)
            error('No supported EEG files found under %s.', input_path);
        end
        file_path = strings(numel(files), 1);
        participant = strings(numel(files), 1);
        condition = strings(numel(files), 1);
        group = repmat("all", numel(files), 1);
        for i = 1:numel(files)
            file_path(i) = string(fullfile(files(i).folder, files(i).name));
            participant(i) = infer_participant(file_path(i), i);
            condition(i) = infer_condition(file_path(i));
        end
    elseif isfile(input_path)
        opts = detectImportOptions(input_path, 'FileType', 'text', 'TextType', 'string', ...
            'Delimiter', manifest_delimiter(input_path));
        opts.VariableNamingRule = 'preserve';
        T = readtable(input_path, opts);
        names = normalise_names(T.Properties.VariableNames);
        file_col = find_first(names, ["filepath", "file_path", "file", "path", "filename"]);
        if isempty(file_col)
            error('Manifest must contain a file_path column.');
        end
        file_path = string(T{:, file_col});
        base_dir = fileparts(input_path);
        for i = 1:numel(file_path)
            if ~is_absolute_path(file_path(i))
                file_path(i) = string(fullfile(base_dir, file_path(i)));
            end
            if ~isfile(file_path(i))
                local_path = fullfile(base_dir, path_basename(file_path(i)));
                if isfile(local_path)
                    file_path(i) = string(local_path);
                end
            end
        end
        p_col = find_first(names, ["participant", "subject", "sub", "id", "subjectid", "participantid"]);
        c_col = find_first(names, ["condition", "dreaming", "state", "task", "eyes", "session"]);
        g_col = find_first(names, ["group", "diagnosis", "cohort", "class"]);
        participant = column_or_infer(T, p_col, file_path, "participant");
        condition = column_or_default(T, c_col, "condition", numel(file_path));
        group = column_or_default(T, g_col, "all", numel(file_path));
    else
        error('Input path not found: %s', input_path);
    end

    [file_path, ord] = sort(file_path);
    participant = participant(ord);
    condition = condition(ord);
    group = group(ord);
    manifest = table(participant(:), condition(:), group(:), file_path(:), ...
        'VariableNames', {'participant', 'condition', 'group', 'file_path'});
end

function delimiter = manifest_delimiter(input_path)
    [~, ~, ext] = fileparts(char(input_path));
    if strcmpi(ext, '.tsv')
        delimiter = '\t';
    else
        delimiter = ',';
    end
end

function files = discover_eeg_files(input_path)
    eeg_extensions = {'.set', '.mat', '.fif', '.fiff', '.edf', '.bdf', '.gdf', '.cnt', '.vhdr', '.mff'};
    scan_roots = eeg_scan_roots(input_path);
    files = [];
    for i = 1:numel(scan_roots)
        root = char(scan_roots(i));
        for j = 1:numel(eeg_extensions)
            files = [files; dir(fullfile(root, '**', ['*' eeg_extensions{j}]))]; %#ok<AGROW>
        end
    end
    if isempty(files)
        return;
    end

    paths = string(fullfile({files.folder}', {files.name}'));
    skip = non_eeg_path_mask(paths);
    files = files(~skip);
    paths = paths(~skip);
    [~, keep] = unique(paths, 'stable');
    files = files(keep);
end

function roots = eeg_scan_roots(input_path)
    dirs = dir(input_path);
    dirs = dirs([dirs.isdir]);
    dirs = dirs(~ismember({dirs.name}, {'.', '..'}));
    names = lower(string({dirs.name}));
    eeg_dirs = contains(names, 'eeg');
    if ~any(eeg_dirs)
        roots = string(input_path);
        return;
    end

    roots = strings(nnz(eeg_dirs), 1);
    hit = find(eeg_dirs);
    for i = 1:numel(hit)
        roots(i) = string(fullfile(dirs(hit(i)).folder, dirs(hit(i)).name));
    end
end

function skip = non_eeg_path_mask(paths)
    p = lower(paths);
    skip = contains(p, [filesep 'behaviour']) | contains(p, [filesep 'behavior']) | ...
        contains(p, [filesep 'cognitive']) | contains(p, [filesep 'medical']) | ...
        contains(p, [filesep 'personality']) | contains(p, [filesep 'data_availability']) | ...
        contains(p, [filesep 'eeg_localizer']) | contains(p, [filesep 'localizer']);
end

function participant = column_or_infer(T, col, file_path, fallback)
    if isempty(col)
        participant = strings(numel(file_path), 1);
        for i = 1:numel(file_path)
            participant(i) = infer_participant(file_path(i), i);
        end
        return;
    end
    participant = string(T{:, col});
    bad = ismissing(participant) | strlength(strtrim(participant)) == 0;
    for i = find(bad(:))'
        participant(i) = infer_participant(file_path(i), i);
    end
    participant(strlength(strtrim(participant)) == 0) = fallback;
end

function values = column_or_default(T, col, default_value, n)
    if isempty(col)
        values = repmat(string(default_value), n, 1);
    else
        values = string(T{:, col});
        values(ismissing(values) | strlength(strtrim(values)) == 0) = string(default_value);
    end
end

function names = normalise_names(names_in)
    names = lower(regexprep(string(names_in), '[^a-zA-Z0-9]', ''));
end

function idx = find_first(names, choices)
    idx = [];
    choices = lower(regexprep(string(choices), '[^a-zA-Z0-9]', ''));
    for i = 1:numel(choices)
        hit = find(names == choices(i), 1, 'first');
        if ~isempty(hit)
            idx = hit;
            return;
        end
    end
end

function tf = is_absolute_path(pth)
    pth = char(pth);
    tf = startsWith(pth, filesep) || ~isempty(regexp(pth, '^[A-Za-z]:[\\/]', 'once'));
end

function name = path_basename(pth)
    parts = regexp(char(pth), '[\\/]', 'split');
    name = parts{end};
end

function pid = infer_participant(file_path, row_index)
    [parent, stem] = fileparts(char(file_path));
    hit = regexp(stem, '(sub-[A-Za-z0-9]+)', 'match', 'once');
    if isempty(hit)
        [~, hit] = fileparts(parent);
    end
    if isempty(hit)
        hit = sprintf('row%03d', row_index);
    end
    pid = string(hit);
end

function condition = infer_condition(file_path)
    [~, stem] = fileparts(char(file_path));
    s = upper(stem);
    if contains(s, '_EC') || contains(s, 'EYESCLOSED') || contains(s, 'CLOSED')
        condition = "EC";
    elseif contains(s, '_EO') || contains(s, 'EYESOPEN') || contains(s, 'OPEN')
        condition = "EO";
    else
        condition = "condition";
    end
end

function meta = inspect_files(manifest, cfg, util)
    meta = repmat(struct('file_path', "", 'labels', {{}}, 'canonical', {{}}, ...
        'keep_index', [], 'chanlocs', [], 'pos', [], 'has_digitized_chanlocs', false), height(manifest), 1);
    for i = 1:height(manifest)
        [data, ~, chanlocs, labels, pos, chanloc_source] = load_eeg_file(manifest.file_path(i), cfg, util);
        n = size(data, 1);
        keep = true(n, 1);
        if cfg.use_scalp_channels
            keep = util.scalp_channel_mask(chanlocs, n, cfg.scalp_channel_max_radius);
        end
        can = util.canonical_channel_labels(labels);
        exclude_can = util.canonical_channel_labels(cfg.exclude_channels);
        keep = keep & ~ismember(can(:), exclude_can(:));
        if nnz(keep) < 8
            error('Too few channels remain for %s.', manifest.file_path(i));
        end
        meta(i).file_path = manifest.file_path(i);
        meta(i).labels = labels(keep);
        meta(i).canonical = can(keep);
        meta(i).keep_index = find(keep);
        if ~isempty(chanlocs)
            meta(i).chanlocs = chanlocs(keep);
        end
        if ~isempty(pos)
            meta(i).pos = pos(keep, :);
        end
        meta(i).has_digitized_chanlocs = strcmp(chanloc_source, 'localizer');
    end
end

function [common_labels, common_can, common_chanlocs, common_pos] = common_channel_set(meta, cfg)
    n_by_file = arrayfun(@(m) numel(m.canonical), meta);
    if cfg.interpolate_missing_channels
        max_n = max(n_by_file);
        ref_candidates = find(n_by_file == max_n);
        digitized = [meta(ref_candidates).has_digitized_chanlocs];
        if any(digitized)
            ref = ref_candidates(find(digitized, 1, 'first'));
        else
            ref = ref_candidates(1);
        end
        common_can = meta(ref).canonical(:);
    else
        common_can = meta(1).canonical(:);
        for i = 2:numel(meta)
            common_can = common_can(ismember(common_can, meta(i).canonical));
        end
        if isempty(common_can)
            error('No common channels across files. Rerun with ''interpolate_missing_channels'', true to use the densest montage and fill missing channels.');
        end
        max_n = max(n_by_file);
        if max_n >= 32 && numel(common_can) < 0.5 * max_n
            error('Strict common-channel intersection kept only %d/%d channels. Rerun with ''interpolate_missing_channels'', true.', numel(common_can), max_n);
        end
        ref = 1;
    end
    common_labels = cell(numel(common_can), 1);
    common_chanlocs = [];
    common_pos = [];
    idx_first = zeros(numel(common_can), 1);
    for c = 1:numel(common_can)
        idx_first(c) = find(strcmp(meta(ref).canonical, common_can{c}), 1, 'first');
        common_labels{c} = meta(ref).labels{idx_first(c)};
    end
    if ~isempty(meta(ref).chanlocs)
        common_chanlocs = meta(ref).chanlocs(idx_first);
    end
    if ~isempty(meta(ref).pos)
        common_pos = meta(ref).pos(idx_first, :);
    end
end

function rec = extract_file_peaks(file_path, meta, common_can, common_pos, cfg, util)
    [data, sfreq, chanlocs, labels, pos] = load_eeg_file(file_path, cfg, util);
    can = util.canonical_channel_labels(labels);
    target_data = nan(numel(common_can), size(data, 2));
    idx = nan(numel(common_can), 1);
    for c = 1:numel(common_can)
        local = find(strcmp(can, common_can{c}), 1, 'first');
        if ~isempty(local)
            idx(c) = local;
            target_data(c, :) = double(data(local, :));
        end
    end
    missing = find(~isfinite(idx));
    if ~isempty(missing)
        if ~cfg.interpolate_missing_channels
            error('Missing channel %s in %s.', common_can{missing(1)}, file_path);
        end
        target_data = fill_missing_channels(target_data, missing, pos, idx, common_pos, file_path);
    end
    data = target_data;
    good_t = all(isfinite(data), 1);
    data = data(:, good_t);
    Sim = struct();
    Sim.X_noisy = data;
    Sim.X_clean = data;
    Sim.sfreq = sfreq;
    Sim.chanlocs = target_chanlocs(chanlocs, idx, common_can);
    Sim.channel_labels = common_can(:)';
    Sim.pos = common_pos;
    Sim.preprocessing = struct( ...
        'apply_average_reference', cfg.apply_average_reference, ...
        'filter_band', cfg.filter_band, ...
        'spatial_filter', cfg.spatial_filter, ...
        'spatial_filter_neighbours', cfg.spatial_filter_neighbours, ...
        'spatial_filter_strength', cfg.spatial_filter_strength, ...
        'gfp_peak_min_distance', cfg.gfp_peak_min_distance, ...
        'gfp_peak_threshold_schedule', cfg.gfp_peak_threshold_schedule, ...
        'reject_gfp_peak_outliers', cfg.reject_gfp_peak_outliers, ...
        'gfp_outlier_mad_multiplier', cfg.gfp_outlier_mad_multiplier, ...
        'min_peak_count_after_gfp_rejection', cfg.min_gfp_peaks_per_file);
    [maps_norm, peak_sample, gfp_vec, ~, ~, maps_raw, preproc_info] = util.preprocess_maps(Sim);
    gfp_peak = gfp_vec(peak_sample(:));
    keep = deterministic_subsample(size(maps_norm, 1), cfg.max_maps_per_file);

    rec = struct();
    rec.file_path = char(file_path);
    rec.common_labels = Sim.channel_labels;
    rec.sfreq = sfreq;
    rec.maps_norm = single(maps_norm(keep, :));
    rec.maps_raw = single(maps_raw(keep, :));
    rec.peak_sample = peak_sample(keep);
    rec.gfp_peak = gfp_peak(keep);
    rec.n_peaks_raw = numel(peak_sample);
    rec.n_peaks_used = numel(keep);
    rec.preprocessing_info = preproc_info;
    rec.n_interpolated_channels = numel(missing);
    rec.interpolated_channel_labels = common_can(missing);
    rec.excluded_channel_labels = setdiff(meta.labels, Sim.channel_labels, 'stable');
end

function chanlocs_out = target_chanlocs(chanlocs, idx, common_can)
    if isempty(chanlocs) || any(~isfinite(idx))
        chanlocs_out = [];
    else
        chanlocs_out = chanlocs(idx);
    end
    if isempty(chanlocs_out)
        chanlocs_out = repmat(struct('labels', ''), 1, numel(common_can));
        for i = 1:numel(common_can)
            chanlocs_out(i).labels = common_can{i};
        end
    end
end

function data = fill_missing_channels(data, missing, source_pos, direct_idx, target_pos, file_path)
    if isempty(source_pos) || isempty(target_pos)
        error('Cannot interpolate missing channels in %s without channel positions.', file_path);
    end
    available = find(isfinite(direct_idx));
    if numel(available) < 4
        error('Cannot interpolate missing channels in %s with fewer than 4 observed target channels.', file_path);
    end
    source_xyz = source_pos(direct_idx(available), :);
    if size(source_xyz, 1) ~= numel(available) || size(target_pos, 1) < max(missing)
        error('Cannot interpolate missing channels in %s because channel positions are incomplete.', file_path);
    end
    for i = 1:numel(missing)
        t = missing(i);
        d = sqrt(sum((source_xyz - target_pos(t, :)).^2, 2));
        [ds, ord] = sort(d, 'ascend');
        m = min(6, numel(ord));
        ord = ord(1:m);
        ds = ds(1:m);
        if ds(1) <= eps
            w = zeros(m, 1);
            w(1) = 1;
        else
            w = 1 ./ (ds.^2 + eps);
            w = w ./ sum(w);
        end
        data(t, :) = w' * data(available(ord), :);
    end
end

function [data, sfreq, chanlocs, labels, pos, chanloc_source] = load_eeg_file(file_path, cfg, util)
    file_path = char(file_path);
    if ~isfile(file_path)
        error('EEG file not found: %s', file_path);
    end
    [~, ~, ext] = fileparts(file_path);
    ext = lower(ext);
    chanlocs = [];
    chanloc_source = 'none';
    sfreq = 250;
    switch ext
        case '.set'
            if exist('pop_loadset', 'file') ~= 2
                error('EEGLAB pop_loadset not found on MATLAB path.');
            end
            EEG = pop_loadset(file_path);
            data = double(EEG.data);
            sfreq = double(EEG.srate);
            chanlocs = EEG.chanlocs;
            chanloc_source = 'file';
        case '.mat'
            S = load(file_path);
            if isfield(S, 'EEG')
                data = double(S.EEG.data);
                if isfield(S.EEG, 'srate'), sfreq = double(S.EEG.srate); end
                if isfield(S.EEG, 'chanlocs')
                    chanlocs = S.EEG.chanlocs;
                    chanloc_source = 'file';
                end
            elseif isfield(S, 'eeg_data')
                data = double(S.eeg_data);
            elseif isfield(S, 'data')
                data = double(S.data);
            else
                error('No EEG data field found in %s.', file_path);
            end
            if isfield(S, 'sfreq'), sfreq = double(S.sfreq); end
            if isfield(S, 'srate'), sfreq = double(S.srate); end
            if isfield(S, 'chanlocs')
                chanlocs = S.chanlocs;
                chanloc_source = 'file';
            end
        otherwise
            [data, sfreq, chanlocs] = load_external_eeg_file(file_path, ext);
            chanloc_source = 'file';
    end
    data = squeeze(data);
    if ~ismatrix(data)
        data = reshape(data, size(data, 1), []);
    end
    if size(data, 1) > size(data, 2) && size(data, 2) < 128
        data = data';
    end
    [chanlocs, used_sidecar] = apply_chanloc_sidecar(chanlocs, file_path, size(data, 1), util);
    if used_sidecar
        chanloc_source = 'sidecar';
    end
    if ~isempty(chanlocs)
        [chanlocs, used_localizer] = apply_localizer_chanlocs(chanlocs, file_path, cfg, util);
        if used_localizer
            chanloc_source = 'localizer';
        end
    end
    labels = util.channel_labels_from_chanlocs(chanlocs, size(data, 1));
    pos = util.positions_from_chanlocs(chanlocs, size(data, 1));
end

function [data, sfreq, chanlocs] = load_external_eeg_file(file_path, ext)
    errors = {};
    if exist('ft_read_header', 'file') == 2 && exist('ft_read_data', 'file') == 2
        try
            [data, sfreq, chanlocs] = load_fieldtrip_file(file_path);
            return;
        catch ME
            errors{end+1} = sprintf('FieldTrip: %s', ME.message);
        end
    end
    if any(strcmpi(ext, {'.fif', '.fiff'})) && exist('fiff_setup_read_raw', 'file') == 2
        try
            [data, sfreq, chanlocs] = load_mne_fif_file(file_path);
            return;
        catch ME
            errors{end+1} = sprintf('MNE-MATLAB: %s', ME.message);
        end
    end
    if exist('pop_fileio', 'file') == 2
        try
            [data, sfreq, chanlocs] = load_eeglab_import(file_path, @pop_fileio);
            return;
        catch ME
            errors{end+1} = sprintf('EEGLAB pop_fileio: %s', ME.message);
        end
    end
    if exist('pop_biosig', 'file') == 2
        try
            [data, sfreq, chanlocs] = load_eeglab_import(file_path, @pop_biosig);
            return;
        catch ME
            errors{end+1} = sprintf('EEGLAB pop_biosig: %s', ME.message);
        end
    end

    detail = '';
    if ~isempty(errors)
        detail = sprintf(' Reader errors: %s', strjoin(errors, ' | '));
    end
    error(['Unsupported EEG extension or missing reader for %s.%s ', ...
        'Install FieldTrip, EEGLAB FileIO/Biosig, or MNE-MATLAB, or convert to .set while preserving EEG.chanlocs.'], ...
        ext, detail);
end

function [data, sfreq, chanlocs] = load_fieldtrip_file(file_path)
    hdr = ft_read_header(file_path);
    data = double(ft_read_data(file_path));
    sfreq = field_value_or_default(hdr, 'Fs', 250);
    labels = field_value_or_default(hdr, 'label', {});
    labels = cellstr(string(labels(:)));
    keep = fieldtrip_eeg_channel_mask(hdr, numel(labels));
    if any(keep) && numel(keep) == numel(labels)
        data = data(keep, :, :);
        labels = labels(keep);
    end
    chanlocs = chanlocs_from_fieldtrip_header(hdr, labels);
end

function keep = fieldtrip_eeg_channel_mask(hdr, n)
    keep = true(n, 1);
    if n == 0 || ~isfield(hdr, 'chantype') || numel(hdr.chantype) < n
        return;
    end
    types = lower(string(hdr.chantype(:)));
    eeg = types(1:n) == "eeg" | contains(types(1:n), "eeg");
    if any(eeg)
        keep = eeg;
    end
end

function chanlocs = chanlocs_from_fieldtrip_header(hdr, labels)
    chanlocs = blank_chanlocs(labels);
    pos = nan(numel(labels), 3);
    if isfield(hdr, 'elec') && ~isempty(hdr.elec)
        pos = labelled_positions(hdr.elec, labels);
    end
    chanlocs = apply_chanloc_positions(chanlocs, pos);
end

function [data, sfreq, chanlocs] = load_mne_fif_file(file_path)
    raw = fiff_setup_read_raw(file_path);
    [data, ~] = fiff_read_raw_segment(raw, raw.first_samp, raw.last_samp);
    sfreq = double(raw.info.sfreq);
    labels = mne_channel_labels(raw.info);
    keep = mne_eeg_channel_mask(raw.info, numel(labels));
    chanlocs_all = chanlocs_from_mne_info(raw.info, labels);
    if any(keep)
        data = data(keep, :);
        labels = labels(keep); %#ok<NASGU>
        chanlocs = chanlocs_all(keep);
    else
        chanlocs = chanlocs_all;
    end
end

function keep = mne_eeg_channel_mask(info, n)
    keep = true(n, 1);
    if ~isfield(info, 'chs') || numel(info.chs) < n || exist('fiff_define_constants', 'file') ~= 2
        return;
    end
    FIFF = fiff_define_constants();
    if ~isfield(FIFF, 'FIFFV_EEG_CH')
        return;
    end
    kinds = nan(n, 1);
    for i = 1:n
        if isfield(info.chs(i), 'kind') && ~isempty(info.chs(i).kind)
            kinds(i) = double(info.chs(i).kind);
        end
    end
    eeg = kinds == double(FIFF.FIFFV_EEG_CH);
    if any(eeg)
        keep = eeg;
    end
end

function labels = mne_channel_labels(info)
    if isfield(info, 'ch_names') && ~isempty(info.ch_names)
        labels = cellstr(string(info.ch_names(:)));
        return;
    end
    n = numel(info.chs);
    labels = cell(n, 1);
    for i = 1:n
        if isfield(info.chs(i), 'ch_name') && ~isempty(info.chs(i).ch_name)
            labels{i} = char(info.chs(i).ch_name);
        else
            labels{i} = sprintf('Ch%03d', i);
        end
    end
end

function chanlocs = chanlocs_from_mne_info(info, labels)
    chanlocs = blank_chanlocs(labels);
    pos = nan(numel(labels), 3);
    if isfield(info, 'chs')
        n = min(numel(labels), numel(info.chs));
        for i = 1:n
            if isfield(info.chs(i), 'loc') && numel(info.chs(i).loc) >= 3
                pos(i, :) = double(info.chs(i).loc(1:3));
            end
        end
    end
    chanlocs = apply_chanloc_positions(chanlocs, pos);
end

function [data, sfreq, chanlocs] = load_eeglab_import(file_path, loader)
    EEG = loader(file_path);
    data = double(EEG.data);
    sfreq = field_value_or_default(EEG, 'srate', 250);
    if isfield(EEG, 'chanlocs') && ~isempty(EEG.chanlocs)
        chanlocs = EEG.chanlocs;
    else
        chanlocs = blank_chanlocs({});
    end
end

function chanlocs = blank_chanlocs(labels)
    labels = cellstr(string(labels(:)));
    chanlocs = repmat(struct('labels', '', 'X', [], 'Y', [], 'Z', []), 1, numel(labels));
    for i = 1:numel(labels)
        chanlocs(i).labels = labels{i};
    end
end

function chanlocs = apply_chanloc_positions(chanlocs, pos)
    n = min(numel(chanlocs), size(pos, 1));
    for i = 1:n
        xyz = double(pos(i, :));
        if numel(xyz) == 3 && all(isfinite(xyz)) && norm(xyz) > eps
            chanlocs(i).X = xyz(1);
            chanlocs(i).Y = xyz(2);
            chanlocs(i).Z = xyz(3);
        end
    end
end

function pos = labelled_positions(geom, labels)
    pos = nan(numel(labels), 3);
    geom_labels = field_value_or_default(geom, 'label', {});
    geom_labels = cellstr(string(geom_labels(:)));
    coords = [];
    for field_name = {'chanpos', 'elecpos', 'pnt'}
        if isfield(geom, field_name{1}) && size(geom.(field_name{1}), 2) >= 3
            coords = double(geom.(field_name{1})(:, 1:3));
            break;
        end
    end
    if isempty(coords)
        return;
    end
    if isempty(geom_labels) && size(coords, 1) == numel(labels)
        pos = coords(1:numel(labels), :);
        return;
    end
    can_geom = simple_canonical_labels(geom_labels);
    can_labels = simple_canonical_labels(labels);
    for i = 1:numel(labels)
        idx = find(strcmp(can_geom, can_labels{i}), 1, 'first');
        if ~isempty(idx) && idx <= size(coords, 1)
            pos(i, :) = coords(idx, :);
        end
    end
end

function can = simple_canonical_labels(labels)
    labels = cellstr(string(labels(:)));
    can = cell(size(labels));
    for i = 1:numel(labels)
        s = lower(strtrim(labels{i}));
        s = regexprep(s, '\s+', '');
        s = regexprep(s, '^eeg', '');
        can{i} = regexprep(s, '[^a-z0-9]', '');
    end
end

function value = field_value_or_default(S, field_name, default_value)
    value = default_value;
    if isstruct(S) && isfield(S, field_name) && ~isempty(S.(field_name))
        value = S.(field_name);
    end
end

function [chanlocs_out, used_sidecar] = apply_chanloc_sidecar(chanlocs_in, file_path, n_channels, util)
    chanlocs_out = chanlocs_in;
    used_sidecar = false;

    sidecar = load_chanloc_sidecar(file_path);
    if isempty(sidecar) || ~has_usable_chanloc_positions(sidecar, numel(sidecar), util)
        return;
    end

    current_labels = util.channel_labels_from_chanlocs(chanlocs_in, n_channels);
    side_labels = util.channel_labels_from_chanlocs(sidecar, numel(sidecar));
    current_can = util.canonical_channel_labels(current_labels);
    side_can = util.canonical_channel_labels(side_labels);

    if isempty(chanlocs_out) || numel(chanlocs_out) < n_channels
        chanlocs_out = blank_chanlocs(current_labels);
    end
    chanlocs_out = ensure_xyz_fields(chanlocs_out);

    n_hit = 0;
    for i = 1:n_channels
        j = find(strcmp(side_can, current_can{i}), 1, 'first');
        if isempty(j) && numel(sidecar) == n_channels
            j = i;
        end
        if isempty(j)
            continue;
        end
        [chanlocs_out(i), ok] = copy_chanloc_position(sidecar(j), chanlocs_out(i));
        if ok
            n_hit = n_hit + 1;
        end
    end
    used_sidecar = n_hit >= max(4, ceil(0.5 * n_channels));
    if ~used_sidecar
        chanlocs_out = chanlocs_in;
    end
end

function tf = has_usable_chanloc_positions(chanlocs, n_channels, util)
    tf = false;
    if isempty(chanlocs)
        return;
    end
    pos = util.positions_from_chanlocs(chanlocs, n_channels);
    good = all(isfinite(pos), 2);
    if sum(good) < 4
        return;
    end
    tf = norm(max(pos(good, :), [], 1) - min(pos(good, :), [], 1)) > eps;
end

function chanlocs = load_chanloc_sidecar(file_path)
    chanlocs = [];
    [folder, stem] = fileparts(file_path);
    candidates = { ...
        fullfile(folder, [stem '_chanlocs.mat']), ...
        fullfile(folder, [stem '_montage.mat'])};
    for i = 1:numel(candidates)
        if ~isfile(candidates{i})
            continue;
        end
        S = load(candidates{i});
        if isfield(S, 'chanlocs') && ~isempty(S.chanlocs)
            chanlocs = S.chanlocs;
            return;
        end
        if isfield(S, 'labels') && isfield(S, 'pos')
            chanlocs = blank_chanlocs(cellstr(string(S.labels)));
            chanlocs = apply_chanloc_positions(chanlocs, double(S.pos));
            return;
        end
    end
end

function [dst, ok] = copy_chanloc_position(src, dst)
    for field_name = {'X', 'Y', 'Z'}
        f = field_name{1};
        if isfield(src, f) && ~isempty(src.(f))
            dst.(f) = src.(f);
        end
    end
    xyz = [field_double(dst, 'X'), field_double(dst, 'Y'), field_double(dst, 'Z')];
    ok = all(isfinite(xyz)) && norm(xyz) > eps;
    if ok
        for field_name = {'theta', 'radius', 'sph_theta', 'sph_phi', 'sph_radius'}
            f = field_name{1};
            if isfield(dst, f)
                dst.(f) = [];
            end
        end
    end
end

function chanlocs = ensure_xyz_fields(chanlocs)
    for field_name = {'X', 'Y', 'Z'}
        f = field_name{1};
        if ~isfield(chanlocs, f)
            [chanlocs.(f)] = deal([]);
        end
    end
end

function [chanlocs_out, used_localizer] = apply_localizer_chanlocs(chanlocs_in, file_path, cfg, util)
    chanlocs_out = chanlocs_in;
    used_localizer = false;
    if ~isfield(cfg, 'localizer_dir') || isempty(cfg.localizer_dir) || ~isfolder(cfg.localizer_dir)
        return;
    end
    participant = regexp(char(file_path), 'sub-[A-Za-z0-9]+', 'match', 'once');
    if isempty(participant)
        return;
    end
    loc_file = fullfile(cfg.localizer_dir, participant, [participant '.mat']);
    if ~isfile(loc_file)
        return;
    end

    try
        S = load(loc_file, 'Channel');
    catch
        return;
    end
    if ~isfield(S, 'Channel') || isempty(S.Channel)
        return;
    end

    [loc_labels, loc_xyz] = localizer_channel_locations(S.Channel);
    loc_can = util.canonical_channel_labels(loc_labels);
    loc_xyz = orient_localizer_superior_axis(loc_xyz, loc_can);
    labels = util.channel_labels_from_chanlocs(chanlocs_in, numel(chanlocs_in));
    can = util.canonical_channel_labels(labels);
    hit = false(numel(chanlocs_in), 1);

    for i = 1:numel(chanlocs_in)
        idx = find(strcmp(loc_can, can{i}), 1, 'first');
        if isempty(idx) || any(~isfinite(loc_xyz(idx, :)))
            continue;
        end
        chanlocs_out(i).labels = labels{i};
        chanlocs_out(i).X = loc_xyz(idx, 1);
        chanlocs_out(i).Y = loc_xyz(idx, 2);
        chanlocs_out(i).Z = loc_xyz(idx, 3);
        chanlocs_out = set_topoplot_location_fields(chanlocs_out, i);
        hit(i) = true;
    end

    if nnz(hit) < max(4, ceil(0.5 * numel(chanlocs_in)))
        chanlocs_out = chanlocs_in;
        return;
    end
    used_localizer = true;
end

function xyz = orient_localizer_superior_axis(xyz, loc_can)
    anchors = {'cz', 'fcz', 'cpz', 'fz', 'pz'};
    idx = [];
    for i = 1:numel(anchors)
        hit = find(strcmp(loc_can, anchors{i}), 1, 'first');
        if ~isempty(hit)
            idx(end+1) = hit; %#ok<AGROW>
        end
    end
    z = xyz(idx, 3);
    z = z(isfinite(z));
    if ~isempty(z) && median(z) < 0
        xyz(:, 3) = -xyz(:, 3);
    end
end

function [labels, xyz] = localizer_channel_locations(Channel)
    labels = cell(1, numel(Channel));
    xyz = nan(numel(Channel), 3);
    for i = 1:numel(Channel)
        labels{i} = localizer_label(Channel(i).Name);
        if isfield(Channel(i), 'Loc') && ~isempty(Channel(i).Loc)
            loc = double(Channel(i).Loc);
            loc = loc(:, 1);
            if numel(loc) >= 3
                xyz(i, :) = loc(1:3)';
            end
        end
    end
end

function label = localizer_label(name)
    parts = regexp(char(name), '_', 'split');
    label = parts{end};
end

function chanlocs = set_topoplot_location_fields(chanlocs, idx)
    xyz = [field_double(chanlocs(idx), 'X'), field_double(chanlocs(idx), 'Y'), field_double(chanlocs(idx), 'Z')];
    if all(isfinite(xyz)) && norm(xyz) > eps
        [az, el, sph_r] = cart2sph(xyz(1), xyz(2), xyz(3));
        theta = atan2d(xyz(1), xyz(2));
        radius = 0.5 - rad2deg(el) / 180;
        chanlocs(idx).theta = theta;
        chanlocs(idx).radius = radius;
        chanlocs(idx).sph_theta = rad2deg(az);
        chanlocs(idx).sph_phi = rad2deg(el);
        chanlocs(idx).sph_radius = sph_r;
    end
end

function value = field_double(S, field_name)
    value = NaN;
    if isfield(S, field_name) && ~isempty(S.(field_name))
        tmp = double(S.(field_name));
        value = tmp(1);
    end
end

function [nodes, summary] = fit_group_nodes(pooled_maps, pooled_gfp, pooled_rows, global_fit, cfg, out_root, common_labels, common_chanlocs)
    levels = meaningful_levels(pooled_rows.group, "all");
    nodes = empty_nodes(size(global_fit.centers, 2));
    summary = table();
    for i = 1:numel(levels)
        level = levels(i);
        mask = pooled_rows.group == level;
        fit_idx = capped_mask_indices(mask, cfg.max_child_maps);
        out_dir = fullfile(out_root, clean_key(level));
        ensure_dir(out_dir);
        Fit = load_cached_fit(out_dir, 'group', cfg, common_labels, common_chanlocs);
        if isempty(Fit)
            Fit = fit_spm_vb_peak_maps(pooled_maps(fit_idx, :), pooled_gfp(fit_idx), cfg.K_candidates, cfg, ['group_' char(level)], global_fit.centers, 'global_k6');
        end
        Fit = align_fit_to_reference(Fit, global_fit);
        rows = pooled_rows(fit_idx, :);
        rows.cluster_label = assign_by_abs_correlation(pooled_maps(fit_idx, :), Fit.centers);
        save_fit_bundle(out_dir, 'group', Fit, rows, common_labels, common_chanlocs, cfg);
        nodes(end+1) = build_node('group', "", level, "", level, Fit, numel(fit_idx), out_dir); %#ok<AGROW>
        summary = [summary; make_summary_row('group', level, level, "", "", Fit, numel(unique(rows.file_index)), numel(fit_idx), out_dir)]; %#ok<AGROW>
        print_alignment_line(['Group ' char(level)], Fit);
    end
end

function [nodes, summary] = fit_condition_nodes(pooled_maps, pooled_gfp, pooled_rows, global_fit, cfg, out_root, common_labels, common_chanlocs)
    levels = meaningful_levels(pooled_rows.condition, "condition");
    nodes = empty_nodes(size(global_fit.centers, 2));
    summary = table();
    for i = 1:numel(levels)
        level = levels(i);
        mask = pooled_rows.condition == level;
        fit_idx = capped_mask_indices(mask, cfg.max_child_maps);
        out_dir = fullfile(out_root, clean_key(level));
        ensure_dir(out_dir);
        Fit = load_cached_fit(out_dir, 'condition', cfg, common_labels, common_chanlocs);
        if isempty(Fit)
            Fit = fit_spm_vb_peak_maps(pooled_maps(fit_idx, :), pooled_gfp(fit_idx), cfg.K_candidates, cfg, ['condition_' char(level)], global_fit.centers, 'global_k6');
        end
        Fit = align_fit_to_reference(Fit, global_fit);
        rows = pooled_rows(fit_idx, :);
        rows.cluster_label = assign_by_abs_correlation(pooled_maps(fit_idx, :), Fit.centers);
        save_fit_bundle(out_dir, 'condition', Fit, rows, common_labels, common_chanlocs, cfg);
        nodes(end+1) = build_node('condition', "", "", level, level, Fit, numel(fit_idx), out_dir); %#ok<AGROW>
        summary = [summary; make_summary_row('condition', level, "", level, "", Fit, numel(unique(rows.file_index)), numel(fit_idx), out_dir)]; %#ok<AGROW>
        print_alignment_line(['Condition ' char(level)], Fit);
    end
end

function [nodes, summary] = fit_participant_nodes(pooled_maps, pooled_gfp, pooled_rows, manifest, global_fit, group_nodes, cfg, out_root, common_labels, common_chanlocs)
    levels = unique(pooled_rows.participant, 'stable');
    nodes = empty_nodes(size(global_fit.centers, 2));
    summary = table();
    for i = 1:numel(levels)
        participant = levels(i);
        mask = pooled_rows.participant == participant;
        fit_idx = capped_mask_indices(mask, cfg.max_child_maps);
        row_mask = manifest.participant == participant;
        group_value = manifest.group(find(row_mask, 1, 'first'));
        [prior_maps, prior_source] = parent_prior_maps(global_fit, group_nodes, group_value);
        out_dir = fullfile(out_root, clean_key(participant));
        ensure_dir(out_dir);
        Fit = load_cached_fit(out_dir, 'participant', cfg, common_labels, common_chanlocs);
        if isempty(Fit)
            Fit = fit_spm_vb_peak_maps(pooled_maps(fit_idx, :), pooled_gfp(fit_idx), cfg.K_candidates, cfg, ['participant_' char(participant)], prior_maps, prior_source);
        end
        Fit = align_fit_to_reference(Fit, global_fit);
        rows = pooled_rows(fit_idx, :);
        rows.cluster_label = assign_by_abs_correlation(pooled_maps(fit_idx, :), Fit.centers);
        save_fit_bundle(out_dir, 'participant', Fit, rows, common_labels, common_chanlocs, cfg);
        nodes(end+1) = build_node('participant', participant, group_value, "", participant, Fit, numel(fit_idx), out_dir); %#ok<AGROW>
        summary = [summary; make_summary_row('participant', participant, group_value, "", participant, Fit, numel(unique(rows.file_index)), numel(fit_idx), out_dir)]; %#ok<AGROW>
        print_alignment_line(['Participant ' char(participant)], Fit);
    end
end

function [nodes, summary] = fit_participant_condition_nodes(pooled_maps, pooled_gfp, pooled_rows, manifest, global_fit, group_nodes, condition_nodes, participant_nodes, cfg, out_root, common_labels, common_chanlocs)
    key = string(pooled_rows.participant) + "|" + string(pooled_rows.condition);
    levels = unique(key, 'stable');
    nodes = empty_nodes(size(global_fit.centers, 2));
    summary = table();
    for i = 1:numel(levels)
        parts = split(levels(i), "|");
        participant = parts(1);
        condition = parts(2);
        mask = pooled_rows.participant == participant & pooled_rows.condition == condition;
        if ~any(mask)
            continue;
        end
        fit_idx = capped_mask_indices(mask, cfg.max_child_maps);

        row_mask = manifest.participant == participant & manifest.condition == condition;
        if ~any(row_mask)
            row_mask = manifest.participant == participant;
        end
        group_value = "all";
        if any(row_mask)
            group_value = manifest.group(find(row_mask, 1, 'first'));
        end

        g = find_node(group_nodes, "", group_value, "");
        if ~isempty(g)
            prior_maps = g.fit.centers;
            prior_source = 'group_k6';
        else
            prior_maps = global_fit.centers;
            prior_source = 'global_k6';
        end
        c = find_node(condition_nodes, "", "", condition);
        if ~isempty(c)
            prior_maps = c.fit.centers;
            prior_source = 'condition_k6';
        end
        p = find_node(participant_nodes, participant, "", "");
        if ~isempty(p)
            prior_maps = p.fit.centers;
            prior_source = 'participant_k6';
        end

        out_dir = fullfile(out_root, clean_key(participant), clean_key(condition));
        ensure_dir(out_dir);
        fit_name = ['participant_condition_' char(participant) '_' char(condition)];
        Fit = load_cached_fit(out_dir, 'participant_condition', cfg, common_labels, common_chanlocs);
        if isempty(Fit)
            Fit = fit_spm_vb_peak_maps(pooled_maps(fit_idx, :), pooled_gfp(fit_idx), cfg.K_candidates, cfg, fit_name, prior_maps, prior_source);
        end
        Fit = align_fit_to_reference(Fit, global_fit);
        rows = pooled_rows(fit_idx, :);
        rows.cluster_label = assign_by_abs_correlation(pooled_maps(fit_idx, :), Fit.centers);
        save_fit_bundle(out_dir, 'participant_condition', Fit, rows, common_labels, common_chanlocs, cfg);

        node_name = participant + "_" + condition;
        nodes(end+1) = build_node('participant_condition', participant, group_value, condition, node_name, Fit, numel(fit_idx), out_dir); %#ok<AGROW>
        summary = [summary; make_summary_row('participant_condition', node_name, group_value, condition, participant, Fit, numel(unique(rows.file_index)), numel(fit_idx), out_dir)]; %#ok<AGROW>
        print_alignment_line(['Participant-condition ' char(participant) ' ' char(condition)], Fit);
    end
end

function [prior_maps, prior_source] = parent_prior_maps(global_fit, group_nodes, group_value)
    prior_maps = global_fit.centers;
    prior_source = 'global_k6';
    g = find_node(group_nodes, "", group_value, "");
    if ~isempty(g)
        prior_maps = g.fit.centers;
        prior_source = 'group_k6';
    end
end

function rec = find_node(nodes, participant, group, condition)
    rec = [];
    for i = 1:numel(nodes)
        if (strlength(participant) == 0 || nodes(i).participant == participant) && ...
                (strlength(group) == 0 || nodes(i).group == group) && ...
                (strlength(condition) == 0 || nodes(i).condition == condition)
            rec = nodes(i);
            return;
        end
    end
end

function out = output_csv_path(output_dir, csv_name)
    out = char(string(csv_name));
    if isempty(out)
        out = output_dir;
    elseif ~is_absolute_path(out)
        out = fullfile(output_dir, out);
    end
end

function T = compile_state_covariance_metrics(global_fit, group_nodes, condition_nodes, participant_nodes, participant_condition_nodes, global_n_maps, global_fit_path)
    rows = {state_covariance_rows_for_fit("global", "global", "all", "", "", global_fit, global_n_maps, global_fit_path)};
    rows = append_node_covariance_rows(rows, group_nodes);
    rows = append_node_covariance_rows(rows, condition_nodes);
    rows = append_node_covariance_rows(rows, participant_nodes);
    rows = append_node_covariance_rows(rows, participant_condition_nodes);
    T = vertcat_nonempty(rows);
    if ~isempty(T)
        T = sortrows(T, {'level', 'group', 'condition', 'participant', 'template_label', 'state_index'});
    end
end

function rows = append_node_covariance_rows(rows, nodes)
    for i = 1:numel(nodes)
        if isfield(nodes(i), 'fit') && isstruct(nodes(i).fit) && isfield(nodes(i).fit, 'centers') && ~isempty(nodes(i).fit.centers)
            rows{end+1, 1} = state_covariance_rows_for_fit(nodes(i).level, nodes(i).name, ...
                nodes(i).group, nodes(i).condition, nodes(i).participant, nodes(i).fit, nodes(i).n_maps, nodes(i).fit_path); %#ok<AGROW>
        end
    end
end

function T = state_covariance_rows_for_fit(level, name, group, condition, participant, Fit, n_maps, fit_path)
    K = size(Fit.centers, 1);
    [state_labels, state_order, template_corr] = state_labels_for_metrics(Fit);
    summary = selected_mix_summary(Fit);
    feature_dim = double(get_field_or(summary, 'feature_dim', NaN));
    free_energy = double(get_field_or(summary, 'free_energy', NaN));
    priors = state_vector(get_field_or(summary, 'priors', []), K);
    traces = state_vector(get_field_or(summary, 'covariance_traces', []), K);
    logdets = state_vector(get_field_or(summary, 'covariance_logdets', []), K);
    cluster_weights = state_vector(get_field_or(Fit, 'cluster_weights', []), K);
    trace_per_feature = traces ./ max(feature_dim, eps);
    trace_per_feature(~isfinite(trace_per_feature)) = NaN;
    logdet_per_feature = logdets ./ max(feature_dim, eps);
    logdet_per_feature(~isfinite(logdet_per_feature)) = NaN;

    rows = cell(numel(state_order), 1);
    for j = 1:numel(state_order)
        k = state_order(j);
        rows{j} = table(string(level), string(name), string(group), string(condition), string(participant), ...
            double(k), string(state_labels{k}), double(K), double(n_maps), double(feature_dim), double(free_energy), ...
            double(priors(k)), double(cluster_weights(k)), double(traces(k)), double(trace_per_feature(k)), ...
            double(logdets(k)), double(logdet_per_feature(k)), double(template_corr(k)), string(fit_path), ...
            'VariableNames', {'level', 'name', 'group', 'condition', 'participant', ...
            'state_index', 'template_label', 'K_estimated', 'n_maps', 'feature_dim', 'free_energy', ...
            'spm_prior', 'cluster_weight', 'covariance_trace', 'covariance_trace_per_feature', ...
            'covariance_logdet', 'covariance_logdet_per_feature', 'template_match_abs_correlation', 'fit_path'});
    end
    T = vertcat(rows{:});
end

function summary = selected_mix_summary(Fit)
    if isfield(Fit, 'selected_spm_mix_model') && isstruct(Fit.selected_spm_mix_model)
        summary = Fit.selected_spm_mix_model;
    else
        summary = empty_mix_summary();
    end
end

function v = state_vector(x, K)
    v = nan(K, 1);
    x = double(x(:));
    n = min(K, numel(x));
    if n > 0
        v(1:n) = x(1:n);
    end
end

function value = get_field_or(S, field_name, fallback)
    if isstruct(S) && isfield(S, field_name) && ~isempty(S.(field_name))
        value = S.(field_name);
    else
        value = fallback;
    end
end

function [T_state, T_pairwise, T_record, T_axis, T_time, T_transition, T_cardiac] = compile_backfit_metric_tables(manifest, participant_nodes, participant_condition_nodes, common_can, common_pos, cfg, util, forced_fit, forced_source)
    if nargin < 8
        forced_fit = [];
    end
    if nargin < 9
        forced_source = "";
    end
    state_rows = {};
    pair_rows = {};
    record_rows = {};
    axis_rows = {};
    time_rows = {};
    transition_rows = {};
    cardiac_rows = cell(height(manifest),1);
    collect_cardiac = cfg.run_cardiac_phase;
    cardiac_labels = ["A", "C", "E"];
    manifest = attach_cardiac_metadata(manifest, cfg.input_path);
    for i = 1:height(manifest)
        if cfg.verbose && (i == 1 || mod(i, 25) == 0 || i == height(manifest))
            fprintf('Backfit record %d/%d\n', i, height(manifest));
        end
        participant = manifest.participant(i);
        condition = manifest.condition(i);
        if isempty(forced_fit)
            node = find_node(participant_condition_nodes, participant, "", condition);
            if isempty(node)
                node = find_node(participant_nodes, participant, "", "");
            end
            if isempty(node)
                warning('No participant-condition or participant fit found for %s %s; backfit skipped.', participant, condition);
                continue;
            end
            Results = node.fit;
            result_source = node.level;
        else
            Results = forced_fit;
            result_source = forced_source;
        end
        try
            [Sim_backfit, gfp, X_metric] = prepare_backfit_record(manifest.file_path(i), common_can, common_pos, cfg, util);
            Results.name = sprintf('backfit_%s_%s', char(manifest.participant(i)), char(manifest.condition(i)));
            backfit = backfit_microstate_timecourse(Sim_backfit, Results);
            if ~isstruct(backfit) || ~isfield(backfit, 'ok') || ~backfit.ok
                warning('Backfit failed for %s: %s', manifest.file_path(i), backfit.message);
                continue;
            end
            [Ts, Tp, Tr, Ttrans] = summarise_backfit_record(backfit, Results, gfp, Sim_backfit.sfreq, ...
                manifest.participant(i), manifest.condition(i), manifest.group(i), manifest.file_path(i), result_source);
            state_rows{end+1, 1} = Ts; %#ok<AGROW>
            pair_rows{end+1, 1} = Tp; %#ok<AGROW>
            record_rows{end+1, 1} = Tr; %#ok<AGROW>
            transition_rows{end+1, 1} = Ttrans; %#ok<AGROW>
            if collect_cardiac
                cardiac_fit = Results;
                cardiac_backfit = backfit;
                cardiac_source = result_source;
                if isempty(forced_fit)
                    participant_node = find_node(participant_nodes, participant, "", "");
                    if isempty(participant_node)
                        warning('No participant fit found for cardiac-phase analysis of %s.', participant);
                        cardiac_fit = [];
                    else
                        cardiac_fit = participant_node.fit;
                        cardiac_source = participant_node.level;
                        cardiac_fit.name = sprintf('cardiac_backfit_%s', char(participant));
                        cardiac_backfit = backfit_microstate_timecourse(Sim_backfit, cardiac_fit);
                    end
                end
                if ~isempty(cardiac_fit) && isstruct(cardiac_backfit) && isfield(cardiac_backfit, 'ok') && cardiac_backfit.ok
                    cardiac_rows{i, 1} = summarise_cardiac_phase(cardiac_backfit, cardiac_fit, Sim_backfit.sfreq, ...
                        participant, condition, manifest.group(i), manifest.file_path(i), cardiac_source, manifest(i,:), cardiac_labels, cfg.ecg_qc_dir);
                end
            end
            if any(strcmpi(char(condition), {'preict', 'preictal'}))
                Tt = summarise_preictal_time_bins(backfit, Results, gfp, X_metric, Sim_backfit.sfreq, ...
                    participant, condition, manifest.group(i), manifest.file_path(i), result_source, ...
                    cfg.backfit_time_bin_s, cfg.backfit_time_window_s);
                time_rows{end+1, 1} = Tt; %#ok<AGROW>
            end
            if cfg.run_backfit_axis_dynamics
                Ta = summarise_backfit_axis_dynamics(backfit, Results, X_metric, Sim_backfit.sfreq, ...
                    manifest.participant(i), manifest.condition(i), manifest.group(i), manifest.file_path(i), ...
                    cfg.backfit_axis_min_samples, result_source);
                axis_rows{end+1, 1} = Ta; %#ok<AGROW>
            end
        catch ME
            warning('Backfit metric extraction failed for %s: %s', manifest.file_path(i), ME.message);
        end
    end

    T_state = vertcat_nonempty(state_rows);
    T_pairwise = vertcat_nonempty(pair_rows);
    T_record = vertcat_nonempty(record_rows);
    T_axis = vertcat_nonempty(axis_rows);
    T_time = vertcat_nonempty(time_rows);
    T_transition = vertcat_nonempty(transition_rows);
    T_cardiac = vertcat_nonempty(cardiac_rows);
    if ~isempty(T_state)
        T_state = sortrows(T_state, {'participant', 'condition', 'backfit_method', 'template_label', 'state_index'});
    end
    if ~isempty(T_pairwise)
        T_pairwise = sortrows(T_pairwise, {'participant', 'condition', 'backfit_method', 'state_i_label', 'state_j_label'});
    end
    if ~isempty(T_record)
        T_record = sortrows(T_record, {'participant', 'condition', 'backfit_method'});
    end
    if ~isempty(T_axis)
        T_axis = sortrows(T_axis, {'participant', 'condition', 'template_label', 'state_index'});
    end
    if ~isempty(T_time)
        T_time = sortrows(T_time, {'participant', 'file_path', 'backfit_result_source', 'template_label', 'bin_from_ictus'});
    end
    if ~isempty(T_transition)
        T_transition = sortrows(T_transition, {'participant', 'condition', 'backfit_method', 'state_from_label', 'state_to_label'});
    end
    if ~isempty(T_cardiac)
        T_cardiac = sortrows(T_cardiac, {'participant', 'condition', 'file_path', 'template_label'});
    end
end

function T = vertcat_nonempty(rows)
    rows = rows(~cellfun(@isempty, rows));
    if isempty(rows)
        T = table();
    else
        T = vertcat(rows{:});
    end
end

function manifest = attach_cardiac_metadata(manifest, input_path)
    manifest.source_raw_fif = strings(height(manifest),1);
    manifest.crop_start_seconds = nan(height(manifest),1);
    manifest.crop_end_seconds = nan(height(manifest),1);
    if ~isfile(input_path), return; end
    opts = detectImportOptions(input_path,'FileType','text','TextType','string','Delimiter',manifest_delimiter(input_path));
    opts.VariableNamingRule = 'preserve'; M = readtable(input_path,opts);
    names = normalise_names(M.Properties.VariableNames);
    file_col = find_first(names,["filepath","file_path","file","path","filename"]);
    raw_col = find_first(names,["sourcerawfif","source_raw_fif"]);
    start_col = find_first(names,["cropstartseconds","crop_start_seconds"]);
    end_col = find_first(names,["cropendseconds","crop_end_seconds"]);
    if isempty(file_col) || isempty(raw_col) || isempty(start_col), return; end
    source_path = lower(replace(string(M{:,file_col}),'/','\'));
    target_path = lower(replace(string(manifest.file_path),'/','\'));
    [found,location] = ismember(target_path,source_path);
    manifest.source_raw_fif(found) = string(M{location(found),raw_col});
    manifest.crop_start_seconds(found) = str2double(string(M{location(found),start_col}));
    if ~isempty(end_col), manifest.crop_end_seconds(found) = str2double(string(M{location(found),end_col})); end
end

function T = summarise_cardiac_phase(backfit, Results, sfreq, participant, condition, group, file_path, result_source, metadata, target_labels, ecg_qc_dir)
    [labels,~,~] = state_labels_for_metrics(Results);
    label_strings = upper(strtrim(string(labels)));
    assignments = double(backfit.hard.assignments(:));
    n = numel(assignments); duration_s = n/sfreq; systolic = false(n,1); rpeaks = [];
    source_raw = string(metadata.source_raw_fif);
    crop_start = double(metadata.crop_start_seconds);
    status = "ok";
    if strlength(source_raw)==0 || ~isfinite(crop_start)
        status = "missing_source_or_crop_metadata";
    else
        [raw_dir,base] = fileparts(char(source_raw));
        base = regexprep(base,'_pp_raw(?:_keepcfa)?$','');
        if strlength(string(ecg_qc_dir)) > 0
            rpeak_csv = fullfile(char(ecg_qc_dir), [base '_rpeaks_corrected.csv']);
        else
            rpeak_csv = fullfile(fileparts(raw_dir),'ecg_qc',[base '_rpeaks_corrected.csv']);
        end
        if ~isfile(rpeak_csv)
            status = "missing_corrected_rpeaks";
        else
            R = readtable(rpeak_csv,'TextType','string');
            if ~ismember('rpeak_time_s',R.Properties.VariableNames)
                status = "invalid_corrected_rpeaks";
            else
                rpeaks = str2double(string(R.rpeak_time_s)) - crop_start;
                rpeaks = rpeaks(isfinite(rpeaks) & rpeaks > -10 & rpeaks < duration_s + 10);
                for r = 1:numel(rpeaks)
                    first = max(1,floor((rpeaks(r)+.180)*sfreq)+1);
                    last = min(n,ceil((rpeaks(r)+.500)*sfreq));
                    if last >= first, systolic(first:last)=true; end
                end
                if ~any(systolic), status = "no_systolic_samples"; end
            end
        end
    end
    rpeak_times_text = strjoin(compose('%.6f',rpeaks),';');
    systolic_s = sum(systolic)/sfreq; diastolic_s = sum(~systolic)/sfreq;
    rows = cell(numel(target_labels),1);
    for target_index = 1:numel(target_labels)
        state_index = find(label_strings == upper(string(target_labels(target_index))),1);
        if isempty(state_index), continue; end
        onsets = assignments == state_index & [true; assignments(1:end-1) ~= state_index];
        onset_times = (find(onsets)-1)/sfreq;
        systolic_count = sum(onsets & systolic); diastolic_count = sum(onsets & ~systolic);
        systolic_rate = systolic_count/max(systolic_s,eps); diastolic_rate = diastolic_count/max(diastolic_s,eps);
        rate_ratio = systolic_rate/diastolic_rate;
        log_rate_ratio = log((systolic_count+.5)/max(systolic_s,eps))-log((diastolic_count+.5)/max(diastolic_s,eps));
        available = status == "ok";
        state_systolic_s = systolic_s; state_diastolic_s = diastolic_s;
        if ~available
            systolic_count=NaN; diastolic_count=NaN; state_systolic_s=NaN; state_diastolic_s=NaN;
            systolic_rate=NaN; diastolic_rate=NaN; rate_ratio=NaN; log_rate_ratio=NaN;
        end
        row = table(string(participant),string(condition),string(group),string(file_path),string(result_source), ...
            "hard",available,double(state_index),label_strings(state_index),double(sfreq),double(n),double(duration_s),double(numel(rpeaks)), ...
            double(systolic_count),double(diastolic_count),double(state_systolic_s),double(state_diastolic_s), ...
            double(systolic_rate),double(diastolic_rate),double(rate_ratio),double(log_rate_ratio),status,source_raw,double(crop_start), ...
            'VariableNames',{'participant','condition','group','file_path','backfit_result_source','backfit_method', ...
            'cardiac_phase_available','state_index','template_label','sfreq','n_samples','duration_s','n_rpeaks','systolic_occurrences', ...
            'diastolic_occurrences','systolic_exposure_s','diastolic_exposure_s','systolic_occurrence_rate_hz', ...
            'diastolic_occurrence_rate_hz','systolic_to_diastolic_rate_ratio','log_systolic_to_diastolic_rate_ratio', ...
            'status','source_raw_fif','crop_start_seconds'});
        row.state_occurrence_times_s = string(strjoin(compose('%.6f',onset_times),';'));
        row.rpeak_times_relative_s = string(rpeak_times_text);
        row.systolic_window_start_ms = 180;
        row.systolic_window_end_ms = 500;
        rows{target_index} = row;
    end
    T = vertcat_nonempty(rows);
end

function [Sim_backfit, gfp, X_metric] = prepare_backfit_record(file_path, common_can, common_pos, cfg, util)
    [data, sfreq, ~, labels, pos] = load_eeg_file(file_path, cfg, util);
    can = util.canonical_channel_labels(labels);
    target_data = nan(numel(common_can), size(data, 2));
    idx = nan(numel(common_can), 1);
    for c = 1:numel(common_can)
        local = find(strcmp(can, common_can{c}), 1, 'first');
        if ~isempty(local)
            idx(c) = local;
            target_data(c, :) = double(data(local, :));
        end
    end
    missing = find(~isfinite(idx));
    if ~isempty(missing)
        if ~cfg.interpolate_missing_channels
            error('Missing channel %s in %s.', common_can{missing(1)}, file_path);
        end
        target_data = fill_missing_channels(target_data, missing, pos, idx, common_pos, file_path);
    end
    good_t = all(isfinite(target_data), 1);
    target_data = target_data(:, good_t);

    Sim_backfit = struct();
    Sim_backfit.X_noisy = target_data;
    Sim_backfit.sfreq = sfreq;
    Sim_backfit.channel_labels = common_can(:)';
    Sim_backfit.chanlocs = [];
    Sim_backfit.pos = common_pos;
    Sim_backfit.preprocessing = struct( ...
        'apply_average_reference', cfg.apply_average_reference, ...
        'filter_band', cfg.filter_band, ...
        'spatial_filter', cfg.spatial_filter, ...
        'spatial_filter_neighbours', cfg.spatial_filter_neighbours, ...
        'spatial_filter_strength', cfg.spatial_filter_strength, ...
        'gfp_peak_min_distance', cfg.gfp_peak_min_distance, ...
        'gfp_peak_threshold_schedule', cfg.gfp_peak_threshold_schedule);
    X_metric = preprocess_backfit_metric_matrix(Sim_backfit, util);
    gfp = std(X_metric, 0, 1, 'omitnan')';
end

function X = preprocess_backfit_metric_matrix(Sim_backfit, util)
    X = double(Sim_backfit.X_noisy);
    if isfield(Sim_backfit.preprocessing, 'apply_average_reference') && Sim_backfit.preprocessing.apply_average_reference
        X = X - mean(X, 1, 'omitnan');
    end
    [X, ~] = util.apply_spatial_filter(X, Sim_backfit);
    if isfield(Sim_backfit.preprocessing, 'filter_band') && ~isempty(Sim_backfit.preprocessing.filter_band)
        X = util.bandpass_filter(X, Sim_backfit.sfreq, Sim_backfit.preprocessing.filter_band);
    end
end

function [T_state, T_pairwise, T_record, T_transition] = summarise_backfit_record(backfit, Results, gfp, sfreq, participant, condition, group, file_path, result_source)
    [state_labels, state_order, template_corr] = state_labels_for_metrics(Results);
    K = size(Results.centers, 1);
    n_samples = min(numel(gfp), backfit.n_samples);
    gfp = gfp(1:n_samples);

    mix_available = isfield(backfit, 'mixture') && isstruct(backfit.mixture) && ...
        isfield(backfit.mixture, 'available') && backfit.mixture.available;
    modes = { ...
        struct('name', 'hard', 'available', true, 'weights', backfit.hard.weights, 'assignments', backfit.hard.assignments), ...
        struct('name', 'gaussian_mixture', 'available', mix_available, 'weights', get_backfit_weights(backfit, 'mixture'), 'assignments', get_backfit_assignments(backfit, 'mixture'))};

    state_rows = {};
    pair_rows = {};
    record_rows = {};
    transition_rows = cell(numel(modes), 1);
    for m = 1:numel(modes)
        spec = modes{m};
        weights = double(spec.weights);
        assignments = double(spec.assignments(:));
        gfp_mode = gfp;
        if spec.available
            if size(weights, 1) > n_samples
                weights = weights(1:n_samples, :);
            elseif size(weights, 1) < n_samples
                gfp_mode = gfp(1:size(weights, 1));
            else
                gfp_mode = gfp;
            end
            if numel(assignments) < size(weights, 1)
                [~, assignments] = max(weights, [], 2);
            else
                assignments = assignments(1:size(weights, 1));
            end
        else
            weights = nan(n_samples, K);
            assignments = nan(n_samples, 1);
            gfp_mode = gfp;
        end
        n_mode = size(weights, 1);
        duration_mode_s = n_mode / max(sfreq, eps);

        Wn = normalize_weight_matrix_rows(weights);
        if spec.available && strcmpi(spec.name, 'hard')
            record_differential_entropy_bits = NaN;
            record_shannon_entropy_bits = shannon_entropy_bits_from_state_distribution(mean(Wn, 1, 'omitnan'));
        elseif spec.available
            record_differential_entropy_bits = joint_differential_entropy_bits_from_weight_matrix(Wn);
            record_shannon_entropy_bits = NaN;
        else
            record_differential_entropy_bits = NaN;
            record_shannon_entropy_bits = NaN;
        end

        record_rows{end+1, 1} = table( ...
            string(participant), string(condition), string(group), string(file_path), ...
            string(result_source), string(spec.name), logical(spec.available), ...
            double(K), double(n_mode), double(sfreq), double(duration_mode_s), ...
            double(record_differential_entropy_bits), double(record_shannon_entropy_bits), ...
            'VariableNames', {'participant', 'condition', 'group', 'file_path', ...
            'backfit_result_source', 'backfit_method', 'backfit_available', ...
            'K_estimated', 'n_samples', 'sfreq', 'duration_s', ...
            'record_differential_entropy_bits', 'record_shannon_entropy_bits'}); %#ok<AGROW>

        for j = 1:numel(state_order)
            k = state_order(j);
            wk = weights(:, k);
            wk(~isfinite(wk)) = 0;
            if ~spec.available
                occupancy = NaN;
                pct_present = NaN;
                mean_gfp = NaN;
                occurrence_count = NaN;
                occurrence_rate_hz = NaN;
            elseif sum(wk, 'omitnan') > eps
                occupancy = mean(wk, 'omitnan');
                pct_present = 100 * mean(assignments == k, 'omitnan');
                mean_gfp = sum(wk .* gfp_mode, 'omitnan') / sum(wk, 'omitnan');
                occurrence_count = count_state_occurrences(assignments, k);
                occurrence_rate_hz = occurrence_count / max(duration_mode_s, eps);
            else
                occupancy = 0;
                pct_present = 0;
                mean_gfp = NaN;
                occurrence_count = 0;
                occurrence_rate_hz = 0;
            end
            state_rows{end+1, 1} = table( ...
                string(participant), string(condition), string(group), string(file_path), ...
                string(result_source), string(spec.name), logical(spec.available), ...
                double(k), string(state_labels{k}), double(K), double(n_mode), double(sfreq), double(duration_mode_s), ...
                double(occupancy), double(pct_present), double(occupancy), double(mean_gfp), ...
                double(occurrence_count), double(occurrence_rate_hz), double(template_corr(k)), ...
                'VariableNames', {'participant', 'condition', 'group', 'file_path', ...
                'backfit_result_source', 'backfit_method', 'backfit_available', ...
                'state_index', 'template_label', 'K_estimated', 'n_samples', 'sfreq', 'duration_s', ...
                'occupancy', 'percentage_record_present', 'mean_quantity', 'mean_gfp', ...
                'occurrence_count', 'occurrence_rate_hz', 'template_match_abs_correlation'}); %#ok<AGROW>
        end

        if spec.available
            transition_rows{m, 1} = transition_metric_rows(assignments, state_labels, state_order, ...
                participant, condition, group, file_path, result_source, spec.name, sfreq);
            for a = 1:(numel(state_order) - 1)
                k1 = state_order(a);
                for b = (a + 1):numel(state_order)
                    k2 = state_order(b);
                    if strcmpi(spec.name, 'hard')
                        [mi_bits, nmi_bits] = binary_mutual_information_bits(weights(:, k1) > 0.5, weights(:, k2) > 0.5);
                    else
                        [mi_bits, nmi_bits] = normalized_quantity_mutual_information_bits(weights(:, k1), weights(:, k2));
                    end
                    pair_rows{end+1, 1} = table( ...
                        string(participant), string(condition), string(group), string(file_path), ...
                        string(result_source), string(spec.name), logical(spec.available), ...
                        double(K), double(n_mode), double(sfreq), double(duration_mode_s), ...
                        double(k1), string(state_labels{k1}), double(k2), string(state_labels{k2}), ...
                        double(mi_bits), double(nmi_bits), ...
                        'VariableNames', {'participant', 'condition', 'group', 'file_path', ...
                        'backfit_result_source', 'backfit_method', 'backfit_available', ...
                        'K_estimated', 'n_samples', 'sfreq', 'duration_s', ...
                        'state_i_index', 'state_i_label', 'state_j_index', 'state_j_label', ...
                        'mutual_information_bits', 'normalized_mutual_information'}); %#ok<AGROW>
                end
            end
        end
    end

    T_state = vertcat_nonempty(state_rows);
    T_pairwise = vertcat_nonempty(pair_rows);
    T_record = vertcat_nonempty(record_rows);
    T_transition = vertcat_nonempty(transition_rows);
end

function T = transition_metric_rows(assignments, labels, state_order, participant, condition, group, file_path, result_source, method, sfreq)
    assignments = round(double(assignments(:)));
    K = numel(labels);
    valid = isfinite(assignments) & assignments >= 1 & assignments <= K;
    dwell = accumarray(assignments(valid), 1, [K 1]);
    from = assignments(1:end-1);
    to = assignments(2:end);
    changes = valid(1:end-1) & valid(2:end) & from ~= to;
    counts = accumarray([from(changes), to(changes)], 1, [K K]);
    outgoing = sum(counts, 2);
    n_valid = sum(valid);
    n_changes = sum(counts, 'all');
    rows = cell(numel(state_order) * (numel(state_order) - 1), 1);
    row = 0;
    for a = 1:numel(state_order)
        i = state_order(a);
        for b = 1:numel(state_order)
            j = state_order(b);
            if i == j, continue; end
            row = row + 1;
            empirical = counts(i, j) / outgoing(i);
            expected = dwell(j) / (n_valid - dwell(i));
            if outgoing(i) == 0, empirical = NaN; end
            if n_valid == dwell(i), expected = NaN; end
            rows{row} = table(string(participant), string(condition), string(group), string(file_path), ...
                string(result_source), string(method), true, double(sfreq), double(n_valid), double(n_changes), ...
                double(i), string(labels{i}), double(j), string(labels{j}), double(dwell(i)), double(dwell(j)), ...
                double(outgoing(i)), double(counts(i, j)), double(empirical), double(expected), double(empirical - expected), ...
                'VariableNames', {'participant','condition','group','file_path','backfit_result_source', ...
                'backfit_method','backfit_available','sfreq','valid_samples','total_state_changes', ...
                'state_from_index','state_from_label','state_to_index','state_to_label','state_from_samples', ...
                'state_to_samples','outgoing_transitions','transition_count','empirical_probability', ...
                'expected_probability','excess_probability'});
        end
    end
    T = vertcat(rows{:});
end

function T = summarise_preictal_time_bins(backfit, Results, gfp, X_metric, sfreq, participant, condition, group, file_path, result_source, bin_s, window_s)
    T = table();
    if ~isfield(backfit, 'hard') || ~isfield(backfit.hard, 'weights') || isempty(backfit.hard.weights)
        return;
    end
    [labels, state_order, template_corr] = state_labels_for_metrics(Results);
    maps = normalize_maps(double(X_metric)');
    centers = normalize_maps(double(Results.centers));
    map_gmd = sqrt(2 * (1 - min(1, abs(maps * centers'))));
    samples_per_bin = max(1, round(double(bin_s) * double(sfreq)));
    mixture_available = isfield(backfit, 'mixture') && isstruct(backfit.mixture) && ...
        isfield(backfit.mixture, 'available') && backfit.mixture.available;
    modes = {struct('name', "hard", 'weights', backfit.hard.weights, ...
            'assignments', get_backfit_assignments(backfit, 'hard')), ...
        struct('name', "gaussian_mixture", 'weights', get_backfit_weights(backfit, 'mixture'), ...
            'assignments', get_backfit_assignments(backfit, 'mixture'))};
    max_bins = min(ceil(numel(gfp) / samples_per_bin), ceil(double(window_s) / double(bin_s)));
    rows = cell(2 * max_bins * numel(state_order), 1);
    row = 0;
    for m = 1:numel(modes)
        if modes{m}.name == "gaussian_mixture" && ~mixture_available, continue; end
        weights = double(modes{m}.weights);
        assignments = round(double(modes{m}.assignments(:)));
        if numel(assignments) < size(weights, 1)
            [~, assignments] = max(weights, [], 2);
        end
        n = min([size(weights, 1), numel(gfp), numel(assignments)]);
        if n < 2, continue; end
        weights = weights(1:n, :);
        gfp_mode = double(gfp(1:n));
        n_bins = min(ceil(n / samples_per_bin), ceil(double(window_s) / double(bin_s)));
        for bin_from_ictus = 1:n_bins
            bin_end = n - (bin_from_ictus - 1) * samples_per_bin;
            bin_start = max(1, bin_end - samples_per_bin + 1);
            idx = bin_start:bin_end;
            centre_sample = (bin_start + bin_end) / 2;
            minutes_to_ictus = -(n - centre_sample) / double(sfreq) / 60;
            normalised_preictal_time = (centre_sample - 1) / max(n - 1, 1);
            sequence = assignments(idx);
            sequence = sequence(isfinite(sequence) & sequence >= 1 & sequence <= size(weights, 2));
            if ~isempty(sequence)
                sequence = sequence(:);
                sequence = sequence([true; diff(sequence) ~= 0]);
            end
            transition_counts = zeros(size(weights, 2));
            for q = 1:max(0, numel(sequence) - 1)
                transition_counts(sequence(q), sequence(q + 1)) = transition_counts(sequence(q), sequence(q + 1)) + 1;
            end
            collapsed_sequence = strjoin(string(sequence(:)'), ',');
            transition_count_matrix_json = string(jsonencode(transition_counts));
            for j = 1:numel(state_order)
                k = state_order(j);
                wk = weights(idx, k);
                valid = isfinite(wk) & isfinite(gfp_mode(idx));
                coverage = mean(wk, 'omitnan');
                if any(valid) && sum(wk(valid), 'omitnan') > eps
                    mean_gfp = sum(wk(valid) .* gfp_mode(idx(valid)), 'omitnan') / sum(wk(valid), 'omitnan');
                    gmd_values = map_gmd(idx(valid), k);
                    mean_topographic_gmd = sum(wk(valid) .* gmd_values, 'omitnan') / sum(wk(valid), 'omitnan');
                else
                    mean_gfp = NaN;
                    mean_topographic_gmd = NaN;
                end
                occurrence_count = NaN;
                mean_dwell_s = NaN;
                if modes{m}.name == "hard"
                    active = wk > .5;
                    runs = active_run_lengths(active);
                    occurrence_count = numel(runs);
                    if ~isempty(runs), mean_dwell_s = mean(runs) / double(sfreq); end
                end
                row = row + 1;
                rows{row, 1} = table(string(participant), string(condition), string(group), string(file_path), ... %#ok<AGROW>
                    string(result_source), modes{m}.name, double(k), string(labels{k}), double(template_corr(k)), ...
                    double(bin_from_ictus), double((bin_start - 1) / sfreq), double(bin_end / sfreq), ...
                    double(minutes_to_ictus), double(normalised_preictal_time), double(numel(idx) / sfreq), ...
                    double(coverage), double(mean_gfp), double(occurrence_count), double(mean_dwell_s), double(mean_topographic_gmd), ...
                    collapsed_sequence, transition_count_matrix_json, ...
                    'VariableNames', {'participant','condition','group','file_path','backfit_result_source', ...
                    'backfit_method','state_index','template_label','template_match_abs_correlation', ...
                    'bin_from_ictus','bin_start_s','bin_end_s','minutes_to_ictus','normalised_preictal_time', ...
                    'bin_duration_s','coverage','mean_gfp','occurrence_count','mean_dwell_s','mean_topographic_gmd', ...
                    'collapsed_sequence','transition_count_matrix_json'});
            end
        end
    end
    if row > 0, T = vertcat(rows{1:row}); end
end

function T = compile_preictal_window_refit_table(manifest, peak_root, participant_root, common_can, cfg)
    keep_index = find(strcmpi(string(manifest.condition), "preict") | strcmpi(string(manifest.condition), "preictal"));
    start_record = floor(double(cfg.window_refit_start_record));
    if start_record > numel(keep_index)
        T = table();
        return;
    end
    keep_index = keep_index(start_record:end);
    if isfinite(cfg.window_refit_max_records)
        keep_index = keep_index(1:min(numel(keep_index), floor(cfg.window_refit_max_records)));
    end
    cache_root = fullfile(cfg.output_dir, 'window_refit_events');
    ensure_dir(cache_root);
    rows = cell(numel(keep_index), 1);

    for j = 1:numel(keep_index)
        i = keep_index(j);
        participant = manifest.participant(i);
        if cfg.verbose
            fprintf('[%d/%d] Window refit: %s\n', j, numel(keep_index), manifest.file_path(i));
        end
        peak_file = fullfile(peak_root, sprintf('%03d_%s_%s_gfp_peaks.mat', i, ...
            clean_key(participant), clean_key(manifest.condition(i))));
        participant_file = fullfile(participant_root, clean_key(participant), 'participant_solution.mat');
        if ~isfile(peak_file) || ~isfile(participant_file)
            warning('Missing peak or participant cache for %s.', manifest.file_path(i));
            continue;
        end
        duration_s = exact_record_duration_s(manifest.file_path(i));
        if ~isfinite(duration_s) || duration_s <= 0
            warning('Missing exact record duration for %s.', manifest.file_path(i));
            continue;
        end
        peak_cached = load(peak_file, 'rec');
        participant_cached = load(participant_file, 'Fit');
        rec = peak_cached.rec;
        participant_fit = participant_cached.Fit;
        signature = window_refit_signature(cfg, peak_file, participant_file, participant_fit.K_estimated);
        cache_file = fullfile(cache_root, sprintf('%03d_%s_w%g_s%g_p%d.mat', i, clean_key(participant), ...
            cfg.window_refit_length_s, cfg.window_refit_step_s, round(cfg.window_refit_min_peaks)));
        part = table();
        if cfg.resume && isfile(cache_file)
            cached = load(cache_file, 'part', 'signature');
            if isfield(cached, 'part') && isfield(cached, 'signature') && isequaln(cached.signature, signature)
                part = cached.part;
            end
        end
        if isempty(part)
            part = refit_one_preictal_event(rec, participant_fit, duration_s, ...
                manifest(i, :), common_can, cfg);
            save(cache_file, 'part', 'signature', '-v7.3');
        end
        rows{j} = part;
    end
    T = vertcat_nonempty(rows);
    if ~isempty(T)
        T = sortrows(T, {'participant','file_path','template_label','window_index_from_ictus'});
    end
end

function duration_s = exact_record_duration_s(file_path)
    duration_s = NaN;
    file_path = char(string(file_path));
    if ~isfile(file_path), return; end
    info = whos('-file', file_path);
    data_info = info(strcmp({info.name}, 'data'));
    if isempty(data_info) || isempty(data_info(1).size), return; end
    n_samples = max(double(data_info(1).size));
    rate = load(file_path, 'sfreq', 'srate');
    if isfield(rate, 'sfreq') && isfinite(double(rate.sfreq)) && double(rate.sfreq) > 0
        sfreq = double(rate.sfreq);
    elseif isfield(rate, 'srate') && isfinite(double(rate.srate)) && double(rate.srate) > 0
        sfreq = double(rate.srate);
    else
        return;
    end
    duration_s = n_samples / sfreq;
end

function signature = window_refit_signature(cfg, peak_file, participant_file, K)
    peak_info = dir(peak_file);
    participant_info = dir(participant_file);
    signature = struct('schema_version', 3, 'window_length_s', double(cfg.window_refit_length_s), ...
        'window_step_s', double(cfg.window_refit_step_s), ...
        'time_window_s', double(cfg.window_refit_time_window_s), ...
        'min_peaks', double(cfg.window_refit_min_peaks), ...
        'min_state_peaks', double(cfg.window_refit_min_state_peaks), ...
        'prior_mode', 'spm_mix_prior_component_means', ...
        'spm_prior_weight', double(cfg.spm_prior_weight), 'K', double(K), ...
        'spm_ms_commit', '0dbde1146f6777412dce19edfa4962cd45157fca', ...
        'peak_bytes', double(peak_info.bytes), 'peak_datenum', double(peak_info.datenum), ...
        'participant_bytes', double(participant_info.bytes), ...
        'participant_datenum', double(participant_info.datenum));
end

function T = refit_one_preictal_event(rec, participant_fit, duration_s, manifest_row, common_can, cfg)
    centers = normalize_maps(double(participant_fit.centers));
    K = size(centers, 1);
    if K ~= 6
        error('Window refitting requires the locked K=6 participant solution, found K=%d.', K);
    end
    [labels, ~, template_corr] = state_labels_for_metrics(participant_fit);
    label_values = upper(strtrim(string(labels)));
    if numel(label_values) ~= K || numel(unique(label_values)) ~= K || ...
            any(~ismember(label_values, ["A","B","C","D","E","F","G"])) || ...
            ~all(ismember(["A","C","E"], label_values))
        error('Participant labels must be six unique A-G labels including A, C and E.');
    end
    target_labels = label_values(:)';
    target_index = nan(numel(target_labels), 1);
    for s = 1:numel(target_labels)
        hit = find(label_values == target_labels(s), 1, 'first');
        if ~isempty(hit)
            target_index(s) = hit;
        end
    end
    sfreq = double(rec.sfreq);
    n_samples = max(1, round(double(duration_s) * sfreq));
    bounds = window_refit_bounds(n_samples, sfreq, cfg.window_refit_length_s, ...
        cfg.window_refit_step_s, cfg.window_refit_time_window_s);
    amp_names = matlab.lang.makeUniqueStrings(cellstr(matlab.lang.makeValidName("amp_" + string(common_can(:)'))));
    ref_amp_names = matlab.lang.makeUniqueStrings(cellstr(matlab.lang.makeValidName("ref_amp_" + string(common_can(:)'))));
    parent_map_sha256 = sha256_array(centers);
    prior_mode = "spm_mix_prior_component_means";
    spm_ms_commit = "0dbde1146f6777412dce19edfa4962cd45157fca";
    spm_version = string(spm('Ver'));
    spm_commit = "04db1be762077630772ee660a470a0fd63d4ff43";
    if isempty(bounds)
        empty_map = array2table(nan(1, numel(common_can)), 'VariableNames', amp_names);
        rows = cell(numel(target_labels), 1);
        for s = 1:numel(target_labels)
            prior_corr = NaN;
            prior_map = nan(1, numel(common_can));
            if isfinite(target_index(s))
                prior_corr = template_corr(target_index(s));
                prior_map = centers(target_index(s), :);
            end
            rows{s} = [table(string(manifest_row.participant), string(manifest_row.condition), ...
                string(manifest_row.group), string(manifest_row.file_path), "participant", "window_refit", ...
                "participant_full_k6", prior_mode, double(cfg.spm_prior_weight), parent_map_sha256, ...
                spm_ms_commit, spm_version, spm_commit, double(K), ...
                double(target_index(s)), NaN, target_labels(s), double(prior_corr), ...
                NaN, NaN, NaN, NaN, NaN, double(cfg.window_refit_length_s), 0, NaN, ...
                "insufficient_record_duration", "insufficient_record_duration", ...
                'VariableNames', {'participant','condition','group','file_path','backfit_result_source','fit_kind', ...
                'prior_source','prior_mode','spm_prior_weight','parent_map_sha256','spm_ms_commit','spm_version','spm_commit', ...
                'K','prior_state_index','window_state_index','template_label','template_match_abs_correlation', ...
                'window_match_abs_correlation','window_index_from_ictus','window_start_s','window_end_s', ...
                'minutes_to_ictus','window_duration_s','window_peak_count','state_peak_count','fit_status','fit_message'}), ...
                empty_map, array2table(prior_map, 'VariableNames', ref_amp_names)];
        end
        T = vertcat(rows{:});
        return;
    end
    rows = cell(height(bounds) * numel(target_labels), 1);
    row = 0;
    peak_sample = double(rec.peak_sample(:));
    peak_maps = double(rec.maps_norm);
    peak_gfp = double(rec.gfp_peak(:));
    local_cfg = cfg;
    local_cfg.K_candidates = K;
    local_cfg.save_plots = false;

    for w = 1:height(bounds)
        in_window = peak_sample >= bounds.window_start_sample(w) & peak_sample <= bounds.window_end_sample(w);
        maps = peak_maps(in_window, :);
        gfp = peak_gfp(in_window);
        window_fit = [];
        fit_message = "";
        if size(maps, 1) >= cfg.window_refit_min_peaks
            try
                fit_name = sprintf('window_%s_%03d', char(manifest_row.participant), w);
                window_fit = fit_spm_vb_peak_maps(maps, gfp, K, local_cfg, fit_name, centers, 'participant_full_k6');
            catch ME
                fit_message = string(ME.message);
            end
        else
            fit_message = "insufficient_window_peaks";
        end

        assignment = [];
        signed_corr = nan(K, K);
        if ~isempty(window_fit)
            recovery = microstate_partial_alignment(centers, window_fit.centers, ...
                'threshold', 0, 'polarity', true);
            assignment = recovery.match_assignment;
            signed_corr = normalize_maps(double(window_fit.centers)) * centers';
        end

        for s = 1:numel(target_labels)
            prior_idx = target_index(s);
            prior_corr = NaN;
            prior_map = nan(1, numel(common_can));
            if isfinite(prior_idx)
                prior_corr = template_corr(prior_idx);
                prior_map = centers(prior_idx, :);
            end
            estimated_idx = NaN;
            match_corr = NaN;
            state_peaks = NaN;
            mean_map = nan(1, numel(common_can));
            status = "fit_failed";
            state_message = fit_message;
            if ~isfinite(prior_idx)
                status = "invalid_participant_reference";
                state_message = "participant solution lacks canonical " + target_labels(s) + " label";
            elseif fit_message == "insufficient_window_peaks"
                status = "insufficient_window_peaks";
            elseif ~isempty(window_fit)
                hit = find(assignment(:, 2) == prior_idx, 1, 'first');
                if isempty(hit)
                    status = "state_unmatched";
                else
                    estimated_idx = assignment(hit, 1);
                    match_corr = abs(signed_corr(estimated_idx, prior_idx));
                    state_peaks = sum(double(window_fit.labels(:)) == estimated_idx);
                    if state_peaks < cfg.window_refit_min_state_peaks
                        status = "insufficient_state_peaks";
                    else
                        polarity = sign(signed_corr(estimated_idx, prior_idx));
                        if polarity == 0 || ~isfinite(polarity), polarity = 1; end
                        mean_map = normalize_maps(polarity * double(window_fit.centers(estimated_idx, :)));
                        status = "ok";
                    end
                end
            end
            meta = table(string(manifest_row.participant), string(manifest_row.condition), ...
                string(manifest_row.group), string(manifest_row.file_path), "participant", "window_refit", ...
                "participant_full_k6", prior_mode, double(cfg.spm_prior_weight), parent_map_sha256, ...
                spm_ms_commit, spm_version, spm_commit, double(K), ...
                double(prior_idx), double(estimated_idx), target_labels(s), ...
                double(prior_corr), double(match_corr), double(w), ...
                double(bounds.window_start_s(w)), double(bounds.window_end_s(w)), ...
                double(bounds.minutes_to_ictus(w)), double(cfg.window_refit_length_s), ...
                double(size(maps, 1)), double(state_peaks), status, fit_message, ...
                'VariableNames', {'participant','condition','group','file_path','backfit_result_source','fit_kind', ...
                'prior_source','prior_mode','spm_prior_weight','parent_map_sha256','spm_ms_commit','spm_version','spm_commit', ...
                'K','prior_state_index','window_state_index','template_label','template_match_abs_correlation', ...
                'window_match_abs_correlation','window_index_from_ictus','window_start_s','window_end_s', ...
                'minutes_to_ictus','window_duration_s','window_peak_count','state_peak_count','fit_status','fit_message'});
            meta.fit_message = state_message;
            row = row + 1;
            rows{row} = [meta, array2table(mean_map, 'VariableNames', amp_names), ...
                array2table(prior_map, 'VariableNames', ref_amp_names)];
        end
    end
    T = vertcat(rows{1:row});
end

function hash = sha256_array(values)
    if isempty(values)
        hash = "none";
        return;
    end
    bytes = typecast(double(values(:)), 'uint8');
    digest = java.security.MessageDigest.getInstance('SHA-256');
    digest.update(bytes);
    hash = string(lower(reshape(dec2hex(typecast(digest.digest(), 'uint8'), 2)', 1, [])));
end

function bounds = window_refit_bounds(n_samples, sfreq, window_s, step_s, time_window_s)
    max_end_to = min(double(time_window_s) - double(window_s), double(n_samples) / double(sfreq) - double(window_s));
    if max_end_to < 0
        bounds = table();
        return;
    end
    end_to = (0:double(step_s):max_end_to)';
    window_end_sample = n_samples - round(end_to * sfreq);
    window_start_sample = window_end_sample - round(double(window_s) * sfreq) + 1;
    valid = window_start_sample >= 1 & window_end_sample <= n_samples;
    end_to = end_to(valid);
    window_start_sample = window_start_sample(valid);
    window_end_sample = window_end_sample(valid);
    window_start_s = (window_start_sample - 1) / sfreq;
    window_end_s = window_end_sample / sfreq;
    minutes_to_ictus = -(end_to + double(window_s) / 2) / 60;
    bounds = table(window_start_sample, window_end_sample, window_start_s, window_end_s, minutes_to_ictus);
end

function self_check_preictal_window_refit()
    rng(19, 'twister');
    prior = normalize_maps(randn(6, 27));
    order = [4 1 6 2 5 3];
    signs = [-1 1 -1 1 1 -1]';
    estimated = prior(order, :) .* signs;
    recovery = microstate_partial_alignment(prior, estimated, 'threshold', 0, 'polarity', true);
    assert(size(recovery.match_assignment, 1) == 6, 'Window-refit one-to-one matching self-check failed.');
    signed = normalize_maps(estimated) * prior';
    for i = 1:6
        hit = find(recovery.match_assignment(:, 2) == i, 1, 'first');
        est = recovery.match_assignment(hit, 1);
        aligned = normalize_maps(sign(signed(est, i)) * estimated(est, :));
        assert(abs(aligned * prior(i, :)' - 1) < 1e-10, 'Window-refit polarity self-check failed.');
    end
    bounds = window_refit_bounds(1200 * 256, 256, 60, 30, 1200);
    assert(height(bounds) == 39 && max(bounds.window_end_sample) == 1200 * 256, ...
        'Window-refit timing self-check failed.');
    x = linspace(-19.5, -0.5, 39)';
    y = 0.1 + 0.02 * x;
    beta = [ones(size(x)), x] \ y;
    assert(abs(beta(2) - 0.02) < 1e-12, 'Window-refit slope self-check failed.');
    drift = normalize_maps([1 -1 zeros(1, 25)]);
    target = 3;
    prior(target, :) = normalize_maps(prior(target, :) - (prior(target, :) * drift') * drift);
    recovered_projection = nan(size(x));
    for w = 1:numel(x)
        true_maps = prior;
        true_maps(target, :) = normalize_maps(prior(target, :) + 0.01 * x(w) * drift);
        permutation = circshift(1:6, w - 1);
        polarity = (-1) .^ ((1:6)' + w);
        estimated = true_maps(permutation, :) .* polarity;
        recovery = microstate_partial_alignment(prior, estimated, 'threshold', 0, 'polarity', true);
        signed = normalize_maps(estimated) * prior';
        hit = find(recovery.match_assignment(:, 2) == target, 1, 'first');
        est = recovery.match_assignment(hit, 1);
        aligned = normalize_maps(sign(signed(est, target)) * estimated(est, :));
        recovered_projection(w) = aligned * drift';
    end
    drift_beta = [ones(size(x)), x] \ recovered_projection;
    assert(drift_beta(2) > 0, ...
        'Combined drift/permutation/polarity self-check recovered the wrong beta direction.');
end

function T = compile_preictal_morphology_table(manifest, global_fit, participant_root, common_can, common_pos, cfg, util)
    keep = strcmpi(string(manifest.condition), "preict") | strcmpi(string(manifest.condition), "preictal");
    preictal = manifest(keep, :);
    rows = cell(2 * height(preictal), 1);
    row = 0;
    for i = 1:height(preictal)
        if cfg.verbose
            fprintf('[%d/%d] Preictal morphology: %s\n', i, height(preictal), preictal.file_path(i));
        end
        try
            [Sim, ~, X_metric] = prepare_backfit_record(preictal.file_path(i), common_can, common_pos, cfg, util);
        catch ME
            warning('Preictal morphology preprocessing failed for %s: %s', preictal.file_path(i), ME.message);
            continue;
        end
        fits = {global_fit, "global"};
        participant_dir = fullfile(participant_root, clean_key(preictal.participant(i)));
        participant_fit = load_cached_fit(participant_dir, 'participant', cfg, {}, []);
        if ~isempty(participant_fit)
            fits(end+1, :) = {participant_fit, "participant"}; %#ok<AGROW>
        end
        for f = 1:size(fits, 1)
            Fit = fits{f, 1};
            source = fits{f, 2};
            try
                Fit.name = sprintf('morphology_%s_%s', char(source), char(preictal.participant(i)));
                backfit = backfit_microstate_timecourse(Sim, Fit);
                if ~isstruct(backfit) || ~isfield(backfit, 'ok') || ~backfit.ok
                    continue;
                end
                part = summarise_preictal_morphology_bins(backfit, Fit, X_metric, Sim.sfreq, ...
                    preictal.participant(i), preictal.condition(i), preictal.group(i), preictal.file_path(i), ...
                    source, common_can, cfg.backfit_time_bin_s, cfg.backfit_time_window_s);
                if ~isempty(part)
                    row = row + 1;
                    rows{row} = part;
                end
            catch ME
                warning('Preictal morphology backfit failed for %s (%s): %s', preictal.file_path(i), source, ME.message);
            end
        end
    end
    T = vertcat_nonempty(rows(1:row));
    if ~isempty(T)
        T = sortrows(T, {'participant','file_path','backfit_result_source','template_label','bin_from_ictus'});
    end
end

function T = summarise_preictal_morphology_bins(backfit, Results, X_metric, sfreq, participant, condition, group, file_path, result_source, common_can, bin_s, window_s)
    T = table();
    if ~isfield(backfit, 'hard') || ~isfield(backfit.hard, 'weights') || isempty(backfit.hard.weights)
        return;
    end
    weights = double(backfit.hard.weights);
    n = min(size(weights, 1), size(X_metric, 2));
    if n < 2, return; end
    weights = weights(1:n, :);
    maps = normalize_maps(double(X_metric(:, 1:n))');
    centers = normalize_maps(double(Results.centers));
    [labels, ~, template_corr] = state_labels_for_metrics(Results);
    targets = find(ismember(upper(string(labels)), ["C", "E"]));
    if isempty(targets), return; end
    samples_per_bin = max(1, round(double(bin_s) * double(sfreq)));
    n_bins = min(ceil(n / samples_per_bin), ceil(double(window_s) / double(bin_s)));
    amp_names = matlab.lang.makeUniqueStrings(cellstr(matlab.lang.makeValidName("amp_" + string(common_can(:)'))));
    rows = cell(n_bins * numel(targets), 1);
    row = 0;
    for bin_from_ictus = 1:n_bins
        bin_end = n - (bin_from_ictus - 1) * samples_per_bin;
        bin_start = max(1, bin_end - samples_per_bin + 1);
        idx = bin_start:bin_end;
        centre_sample = (bin_start + bin_end) / 2;
        for j = 1:numel(targets)
            k = targets(j);
            wk = weights(idx, k);
            polarity = sign(maps(idx, :) * centers(k, :)');
            polarity(polarity == 0 | ~isfinite(polarity)) = 1;
            valid = isfinite(wk) & wk > .5 & all(isfinite(maps(idx, :)), 2);
            state_samples = sum(valid);
            mean_map = nan(1, numel(common_can));
            if state_samples >= 8
                aligned = maps(idx(valid), :) .* polarity(valid);
                mean_map = normalize_maps(mean(aligned, 1, 'omitnan'));
            end
            meta = table(string(participant), string(condition), string(group), string(file_path), string(result_source), ...
                "hard", double(k), string(labels{k}), double(template_corr(k)), double(bin_from_ictus), ...
                double((bin_start - 1) / sfreq), double(bin_end / sfreq), ...
                double(-(n - centre_sample) / double(sfreq) / 60), double((centre_sample - 1) / max(n - 1, 1)), ...
                double(numel(idx) / sfreq), double(state_samples), double(mean(wk, 'omitnan')), ...
                'VariableNames', {'participant','condition','group','file_path','backfit_result_source','backfit_method', ...
                'state_index','template_label','template_match_abs_correlation','bin_from_ictus','bin_start_s','bin_end_s', ...
                'minutes_to_ictus','normalised_preictal_time','bin_duration_s','state_samples','coverage'});
            row = row + 1;
            rows{row} = [meta, array2table(mean_map, 'VariableNames', amp_names)];
        end
    end
    if row > 0, T = vertcat(rows{1:row}); end
end

function self_check_preictal_morphology()
    maps = normalize_maps([1 -1 0; -1 1 0]);
    center = maps(1, :);
    polarity = sign(maps * center');
    aligned = maps .* polarity;
    recovered = normalize_maps(mean(aligned, 1));
    assert(norm(recovered - center) < 1e-12, 'Preictal morphology polarity self-check failed.');
end

function T_axis = summarise_backfit_axis_dynamics(backfit, Results, X_metric, sfreq, participant, condition, group, file_path, min_samples, result_source)
    [state_labels, state_order, template_corr] = state_labels_for_metrics(Results);
    K = size(Results.centers, 1);
    rows = {};

    if ~isfield(backfit, 'hard') || ~isfield(backfit.hard, 'assignments') || isempty(backfit.hard.assignments)
        T_axis = table();
        return;
    end

    assignments = double(backfit.hard.assignments(:));
    n_samples = min([size(X_metric, 2), numel(assignments), backfit.n_samples]);
    if n_samples < 2
        T_axis = table();
        return;
    end

    X_metric = double(X_metric(:, 1:n_samples));
    assignments = assignments(1:n_samples);
    maps_norm = normalize_maps(X_metric');
    centers = normalize_maps(Results.centers);
    min_samples = max(2, round(double(min_samples)));

    for j = 1:numel(state_order)
        k = state_order(j);
        active = assignments == k & isfinite(assignments);
        n_active = sum(active);
        active_fraction = n_active / n_samples;
        active_duration_s = n_active / max(sfreq, eps);
        run_lengths = active_run_lengths(active);
        active_run_count = numel(run_lengths);
        mean_run_s = mean_or_nan(run_lengths) / max(sfreq, eps);
        median_run_s = median_or_nan(run_lengths) / max(sfreq, eps);

        axis_projection_rms = NaN;
        axis_projection_std = NaN;
        axis_projection_mean_abs = NaN;
        axis_zero_crossing_rate_hz = NaN;
        axis_energy_fraction = NaN;
        pca_pc1_variance_pct = NaN;
        pca_pc2_variance_pct = NaN;
        pca_pc1_abs_corr_with_state_axis = NaN;
        angular_deviation_mean_deg = NaN;
        angular_deviation_sd_deg = NaN;
        angular_deviation_median_deg = NaN;
        perpendicular_energy_fraction = NaN;
        residual_pc1_variance_pct = NaN;
        residual_pc2_variance_pct = NaN;
        residual_phase_resultant_length = NaN;
        precession_signed_mean_deg_per_s = NaN;
        precession_mean_abs_deg_per_s = NaN;
        precession_median_abs_deg_per_s = NaN;
        precession_abs_cycles_per_s = NaN;
        precession_net_cycles_per_s = NaN;
        precession_directionality_index = NaN;
        precession_positive_fraction = NaN;
        precession_negative_fraction = NaN;

        if n_active >= min_samples && k <= size(centers, 1) && size(maps_norm, 2) == size(centers, 2)
            axis_norm = centers(k, :);
            axis_col = axis_norm(:);
            axis_signal_full = X_metric' * axis_col;
            axis_signal = axis_signal_full(active);
            axis_projection_rms = sqrt(mean(axis_signal.^2, 'omitnan'));
            axis_projection_std = std(axis_signal, 0, 'omitnan');
            axis_projection_mean_abs = mean(abs(axis_signal), 'omitnan');
            axis_zero_crossing_rate_hz = zero_crossing_rate_active(axis_signal_full, active, sfreq);

            maps_k = maps_norm(active, :);
            proj = maps_k * axis_col;
            proj_abs = abs(proj);
            axis_energy_fraction = mean(clamp_unit(proj).^2, 'omitnan');
            angles = acosd(clamp_unit(proj_abs));
            angular_deviation_mean_deg = mean(angles, 'omitnan');
            angular_deviation_sd_deg = std(angles, 0, 'omitnan');
            angular_deviation_median_deg = median(angles, 'omitnan');
            perpendicular_energy_fraction = mean(max(0, 1 - proj_abs.^2), 'omitnan');

            Xc = maps_k - mean(maps_k, 1, 'omitnan');
            [pca_pc1_variance_pct, pca_pc2_variance_pct, pca_pc1_abs_corr_with_state_axis] = ...
                pca_axis_summary(Xc, axis_col);

            signs = sign(proj);
            signs(signs == 0) = 1;
            maps_aligned = maps_k .* signs;
            residual_active = maps_aligned - proj_abs * axis_norm;
            Rc = residual_active - mean(residual_active, 1, 'omitnan');
            [residual_pc1_variance_pct, residual_pc2_variance_pct, ~, residual_basis] = ...
                pca_axis_summary(Rc, axis_col);

            if size(residual_basis, 2) >= 2
                proj_full = maps_norm * axis_col;
                signs_full = sign(proj_full);
                signs_full(signs_full == 0) = 1;
                maps_aligned_full = maps_norm .* signs_full;
                residual_full = maps_aligned_full - abs(proj_full) * axis_norm;
                phase = atan2(residual_full * residual_basis(:, 2), residual_full * residual_basis(:, 1));
                residual_phase_resultant_length = abs(mean(exp(1i * phase(active)), 'omitnan'));
                [precession_signed_mean_deg_per_s, precession_mean_abs_deg_per_s, ...
                    precession_median_abs_deg_per_s, precession_abs_cycles_per_s, ...
                    precession_net_cycles_per_s, precession_directionality_index, ...
                    precession_positive_fraction, precession_negative_fraction] = ...
                    precession_metrics_active(phase, active, sfreq);
            end
        end

        rows{end+1, 1} = table( ...
            string(participant), string(condition), string(group), string(file_path), ...
            string(result_source), "hard", true, double(k), string(state_labels{k}), ...
            double(K), double(n_samples), double(sfreq), double(n_active), double(active_fraction), ...
            double(active_duration_s), double(active_run_count), double(mean_run_s), double(median_run_s), ...
            double(axis_projection_rms), double(axis_projection_std), double(axis_projection_mean_abs), ...
            double(axis_zero_crossing_rate_hz), double(axis_energy_fraction), ...
            double(pca_pc1_variance_pct), double(pca_pc2_variance_pct), double(pca_pc1_abs_corr_with_state_axis), ...
            double(angular_deviation_mean_deg), double(angular_deviation_sd_deg), double(angular_deviation_median_deg), ...
            double(perpendicular_energy_fraction), double(residual_pc1_variance_pct), double(residual_pc2_variance_pct), ...
            double(residual_phase_resultant_length), double(precession_signed_mean_deg_per_s), ...
            double(precession_mean_abs_deg_per_s), double(precession_median_abs_deg_per_s), ...
            double(precession_abs_cycles_per_s), double(precession_net_cycles_per_s), ...
            double(precession_directionality_index), double(precession_positive_fraction), ...
            double(precession_negative_fraction), double(template_corr(k)), ...
            'VariableNames', {'participant', 'condition', 'group', 'file_path', ...
            'backfit_result_source', 'backfit_method', 'backfit_available', ...
            'state_index', 'template_label', 'K_estimated', 'n_samples', 'sfreq', ...
            'n_active_samples', 'active_fraction', 'active_duration_s', 'active_run_count', ...
            'mean_active_run_duration_s', 'median_active_run_duration_s', ...
            'axis_projection_rms', 'axis_projection_std', 'axis_projection_mean_abs', ...
            'axis_zero_crossing_rate_hz', 'axis_energy_fraction', ...
            'pca_pc1_variance_pct', 'pca_pc2_variance_pct', 'pca_pc1_abs_corr_with_state_axis', ...
            'angular_deviation_mean_deg', 'angular_deviation_sd_deg', 'angular_deviation_median_deg', ...
            'perpendicular_energy_fraction', 'residual_pc1_variance_pct', 'residual_pc2_variance_pct', ...
            'residual_phase_resultant_length', 'precession_signed_mean_deg_per_s', ...
            'precession_mean_abs_deg_per_s', 'precession_median_abs_deg_per_s', ...
            'precession_abs_cycles_per_s', 'precession_net_cycles_per_s', ...
            'precession_directionality_index', 'precession_positive_fraction', ...
            'precession_negative_fraction', ...
            'template_match_abs_correlation'}); %#ok<AGROW>
    end

    T_axis = vertcat_nonempty(rows);
end

function lengths = active_run_lengths(mask)
    mask = logical(mask(:));
    d = diff([false; mask; false]);
    starts = find(d == 1);
    stops = find(d == -1) - 1;
    lengths = stops - starts + 1;
end

function x = mean_or_nan(v)
    v = double(v(:));
    v = v(isfinite(v));
    if isempty(v)
        x = NaN;
    else
        x = mean(v);
    end
end

function x = median_or_nan(v)
    v = double(v(:));
    v = v(isfinite(v));
    if isempty(v)
        x = NaN;
    else
        x = median(v);
    end
end

function x = clamp_unit(x)
    x = max(-1, min(1, double(x)));
end

function [pc1_pct, pc2_pct, pc1_axis_corr, basis] = pca_axis_summary(Xc, axis_col)
    pc1_pct = NaN;
    pc2_pct = NaN;
    pc1_axis_corr = NaN;
    basis = zeros(size(Xc, 2), 0);
    if size(Xc, 1) < 2 || isempty(Xc)
        return;
    end
    Xc(~isfinite(Xc)) = 0;
    [~, S, V] = svd(Xc, 'econ');
    latent = diag(S).^2 ./ max(1, size(Xc, 1) - 1);
    total = sum(latent);
    if total <= eps || isempty(V)
        return;
    end
    basis = V;
    pc1_pct = 100 * latent(1) / total;
    if numel(latent) >= 2
        pc2_pct = 100 * latent(2) / total;
    end
    pc1_axis_corr = abs(dot(V(:, 1), axis_col));
end

function rate_hz = zero_crossing_rate_active(signal, active, sfreq)
    idx = find(active(:));
    if numel(idx) < 2
        rate_hz = NaN;
        return;
    end
    a = double(signal(idx(1:end-1)));
    b = double(signal(idx(2:end)));
    adjacent = idx(2:end) == idx(1:end-1) + 1;
    valid = adjacent & isfinite(a) & isfinite(b) & a ~= 0 & b ~= 0;
    flips = (a < 0 & b > 0) | (a > 0 & b < 0);
    active_duration_s = numel(idx) / max(sfreq, eps);
    rate_hz = sum(flips(:) & valid(:)) / max(active_duration_s, eps);
end

function [signed_mean_deg_s, mean_abs_deg_s, median_abs_deg_s, abs_cycles_s, net_cycles_s, directionality, positive_fraction, negative_fraction] = precession_metrics_active(phase, active, sfreq)
    idx = find(active(:));
    if numel(idx) < 2
        signed_mean_deg_s = NaN;
        mean_abs_deg_s = NaN;
        median_abs_deg_s = NaN;
        abs_cycles_s = NaN;
        net_cycles_s = NaN;
        directionality = NaN;
        positive_fraction = NaN;
        negative_fraction = NaN;
        return;
    end
    p1 = phase(idx(1:end-1));
    p2 = phase(idx(2:end));
    adjacent = idx(2:end) == idx(1:end-1) + 1;
    dphi = angle(exp(1i * (p2 - p1)));
    dphi = dphi(adjacent & isfinite(dphi));
    if isempty(dphi)
        signed_mean_deg_s = NaN;
        mean_abs_deg_s = NaN;
        median_abs_deg_s = NaN;
        abs_cycles_s = NaN;
        net_cycles_s = NaN;
        directionality = NaN;
        positive_fraction = NaN;
        negative_fraction = NaN;
        return;
    end
    abs_dphi = abs(dphi);
    duration_s = numel(idx) / max(sfreq, eps);
    signed_mean_deg_s = mean(dphi) * sfreq * 180 / pi;
    mean_abs_deg_s = mean(abs_dphi) * sfreq * 180 / pi;
    median_abs_deg_s = median(abs_dphi) * sfreq * 180 / pi;
    abs_cycles_s = sum(abs_dphi) / (2 * pi) / max(duration_s, eps);
    net_cycles_s = sum(dphi) / (2 * pi) / max(duration_s, eps);
    directionality = abs(sum(dphi)) / max(sum(abs_dphi), eps);
    positive_fraction = mean(dphi > 0);
    negative_fraction = mean(dphi < 0);
end

function weights = get_backfit_weights(backfit, mode)
    weights = [];
    if isfield(backfit, mode) && isfield(backfit.(mode), 'weights')
        weights = backfit.(mode).weights;
    end
end

function assignments = get_backfit_assignments(backfit, mode)
    assignments = [];
    if isfield(backfit, mode) && isfield(backfit.(mode), 'assignments')
        assignments = backfit.(mode).assignments;
    end
end

function [state_labels, state_order, template_corr] = state_labels_for_metrics(Results)
    K = size(Results.centers, 1);
    state_labels = arrayfun(@(k) sprintf('state_%02d', k), 1:K, 'UniformOutput', false);
    template_corr = nan(K, 1);
    if isfield(Results, 'template_alignment') && isstruct(Results.template_alignment)
        A = Results.template_alignment;
        if isfield(A, 'labels') && numel(A.labels) >= K
            state_labels = cellstr(string(A.labels(:)));
        end
        if isfield(A, 'correlations') && numel(A.correlations) >= K
            template_corr = abs(double(A.correlations(:)));
        end
    end
    [~, state_order] = sort(lower(string(state_labels)));
end

function n = count_state_occurrences(assignments, state_idx)
    assignments = double(assignments(:));
    assignments = assignments(isfinite(assignments));
    if isempty(assignments)
        n = NaN;
        return;
    end
    hit = assignments == state_idx;
    n = double(hit(1)) + sum(hit(2:end) & ~hit(1:end-1));
end

function W = normalize_weight_matrix_rows(W)
    W = double(W);
    W(~isfinite(W)) = 0;
    row_sum = sum(W, 2);
    valid = row_sum > eps;
    W(valid, :) = W(valid, :) ./ row_sum(valid);
    W(~valid, :) = 0;
end

function h_bits = joint_differential_entropy_bits_from_weight_matrix(W)
    W = normalize_weight_matrix_rows(W);
    if isempty(W) || size(W, 1) < 16
        h_bits = NaN;
        return;
    end
    d = size(W, 2);
    n_bins = min(5, max(3, round(size(W, 1) .^ (1 / max(d + 1, 2)))));
    edges = linspace(0, 1, n_bins + 1);
    bin_subs = zeros(size(W));
    for j = 1:d
        bin_subs(:, j) = discretize(min(max(W(:, j), 0), 1), edges);
    end
    valid = all(bin_subs >= 1 & bin_subs <= n_bins, 2);
    bin_subs = bin_subs(valid, :);
    if size(bin_subs, 1) < 16
        h_bits = NaN;
        return;
    end
    strides = [1, cumprod(repmat(n_bins, 1, d - 1))];
    linear_idx = 1 + sum((bin_subs - 1) .* strides, 2);
    counts = accumarray(linear_idx, 1, [n_bins ^ d, 1]);
    p = counts / sum(counts);
    p = p(p > 0);
    h_bits = -sum(p .* log2(p)) + d * log2(edges(2) - edges(1));
end

function h_bits = shannon_entropy_bits_from_state_distribution(p)
    p = double(p(:));
    p(~isfinite(p)) = 0;
    if sum(p) <= eps
        h_bits = NaN;
        return;
    end
    p = p / sum(p);
    p = p(p > 0);
    h_bits = -sum(p .* log2(p));
end

function [mi_bits, nmi] = binary_mutual_information_bits(x, y)
    x = logical(x(:));
    y = logical(y(:));
    n = min(numel(x), numel(y));
    if n == 0
        mi_bits = NaN;
        nmi = NaN;
        return;
    end
    x = x(1:n);
    y = y(1:n);
    joint = zeros(2, 2);
    for xi = 0:1
        for yi = 0:1
            joint(xi + 1, yi + 1) = mean((x == logical(xi)) & (y == logical(yi)));
        end
    end
    [mi_bits, nmi] = mutual_information_from_joint(joint);
end

function [mi_bits, nmi] = normalized_quantity_mutual_information_bits(x, y)
    x = normalize_single_quantity_trace(x);
    y = normalize_single_quantity_trace(y);
    valid = isfinite(x) & isfinite(y);
    x = x(valid);
    y = y(valid);
    if numel(x) < 16 || max(x) - min(x) <= eps || max(y) - min(y) <= eps
        mi_bits = NaN;
        nmi = NaN;
        return;
    end
    n_bins = min(16, max(6, round(sqrt(numel(x)) / 2)));
    joint = histcounts2(min(max(x, 0), 1), min(max(y, 0), 1), linspace(0, 1, n_bins + 1), linspace(0, 1, n_bins + 1));
    [mi_bits, nmi] = mutual_information_from_joint(joint);
end

function [mi_bits, nmi] = mutual_information_from_joint(joint_counts)
    total = sum(joint_counts, 'all');
    if total <= 0
        mi_bits = NaN;
        nmi = NaN;
        return;
    end
    pxy = joint_counts / total;
    px = sum(pxy, 2);
    py = sum(pxy, 1);
    mi_bits = 0;
    for i = 1:size(pxy, 1)
        for j = 1:size(pxy, 2)
            if pxy(i, j) > 0
                mi_bits = mi_bits + pxy(i, j) * log2(pxy(i, j) / max(px(i) * py(j), eps));
            end
        end
    end
    nmi = mi_bits / max(sqrt(discrete_entropy_bits(px) * discrete_entropy_bits(py(:))), eps);
end

function h = discrete_entropy_bits(p)
    p = double(p(:));
    p = p(isfinite(p) & p > 0);
    if isempty(p)
        h = 0;
    else
        h = -sum(p .* log2(p));
    end
end

function x = normalize_single_quantity_trace(x)
    x = double(x(:));
    x(~isfinite(x)) = 0;
    x = x - min(x);
    xmax = max(x);
    if xmax > eps
        x = x ./ xmax;
    else
        x(:) = 0;
    end
end

function levels = meaningful_levels(values, default_value)
    levels = unique(string(values), 'stable');
    levels = levels(~ismissing(levels) & strlength(strtrim(levels)) > 0);
    levels = levels(levels ~= string(default_value));
    levels = levels(:)';
end

function Fit = fit_spm_vb_peak_maps(maps, gfp, K_candidates, cfg, fit_name, prior_maps, prior_source)
    if nargin < 7 || isempty(prior_source)
        prior_source = 'unspecified';
    end
    t0 = tic;
    X = normalize_maps(double(maps));
    gfp = double(gfp(:));
    if numel(gfp) ~= size(X, 1)
        gfp = ones(size(X, 1), 1);
    end
    ok = all(isfinite(X), 2) & isfinite(gfp) & gfp > 0;
    X = X(ok, :);
    gfp = gfp(ok);
    K_candidates = unique(round(double(K_candidates(:)')));
    K_candidates = K_candidates(K_candidates >= 2 & K_candidates < size(X, 1));
    if isempty(K_candidates)
        error('No valid K candidates for %s with %d maps.', fit_name, size(X, 1));
    end

    prior_maps = normalize_maps(double(prior_maps));
    nK = numel(K_candidates);
    free_energy = -inf(nK, 1);
    silhouette_vals = nan(nK, 1);
    gev_vals = nan(nK, 1);
    wss_vals = inf(nK, 1);
    centers_by_K = cell(nK, 1);
    labels_by_K = cell(nK, 1);
    prior_beta_by_K = cell(nK, 1);
    summaries = repmat(empty_mix_summary(), nK, 1);

    for iK = 1:nK
        K = K_candidates(iK);
        try
            prior_K = ensure_k_prior_maps(prior_maps, X, K);
            [features, feature_info, prior_features] = spm_features(X, prior_K);
            if size(features, 1) ~= size(X, 1)
                error('Feature observation count differs from the %d real GFP-peak maps for %s.', size(X, 1), fit_name);
            end
            if isempty(prior_K)
                evalc('mix = spm_mix(features, K, 0);');
            else
                mix = spm_mix_prior(features, K, 0, ...
                    struct('means', prior_features, 'mean_weight', cfg.spm_prior_weight));
                beta_0 = double(mix.prior.beta_0(:));
                if numel(beta_0) ~= K || any(abs(beta_0 - cfg.spm_prior_weight) > 100 * eps(max(1, cfg.spm_prior_weight)))
                    error('SPM mean-prior weights were not retained for %s K=%d.', fit_name, K);
                end
                prior_beta_by_K{iK} = beta_0;
            end
            if ~valid_mix(mix, K)
                continue;
            end
            labels0 = assign_mix(features, mix);
            centers0 = recover_centers(X, labels0, K, prior_K);
            centers = refine_centers(X, centers0, prior_K, 20);
            labels = assign_by_abs_correlation(X, centers);
            metrics = topo_metrics(X, labels, centers, gfp);

            free_energy(iK) = double(real(mix.fm));
            silhouette_vals(iK) = metrics.silhouette;
            gev_vals(iK) = metrics.gev;
            wss_vals(iK) = metrics.wss;
            centers_by_K{iK} = centers;
            labels_by_K{iK} = labels;
            summaries(iK) = summarise_mix(mix, K, size(features, 2), feature_info);
        catch ME
            warning('VB-GMM fit failed for %s K=%d: %s', fit_name, K, ME.message);
        end
    end

    valid = isfinite(free_energy) & ~cellfun(@isempty, centers_by_K);
    if ~any(valid)
        error('All VB-GMM fits failed for %s.', fit_name);
    end
    selection_payload = struct('K_candidates', K_candidates(:), ...
        'free_energy_vals', free_energy(:), ...
        'silhouette_vals', silhouette_vals(:), ...
        'gev_vals', gev_vals(:), ...
        'calinski_harabasz_vals', nan(nK, 1), ...
        'spm_mix_model_summaries', summaries);
    [K_est, best_score, score_by_k, selection_details] = select_spm_vb_k_by_criterion(selection_payload, cfg.criterion);
    if ~isfinite(K_est)
        [~, best_idx] = max(gev_vals);
        K_est = K_candidates(best_idx);
        best_score = gev_vals(best_idx);
        score_by_k = gev_vals;
        selection_details = struct('criterion', 'gev_fallback');
    else
        best_idx = find(K_candidates == K_est, 1, 'first');
    end

    model_comparison = table(K_candidates(:), free_energy(:), silhouette_vals(:), gev_vals(:), wss_vals(:), score_by_k(:), ...
        'VariableNames', {'K', 'free_energy', 'silhouette', 'gev', 'wss', 'selection_score'});

    Fit = struct();
    Fit.name = char(fit_name);
    Fit.method = 'spm_vb_hierarchical';
    Fit.criterion = cfg.criterion;
    Fit.K_candidates = K_candidates;
    Fit.K_estimated = K_est;
    Fit.best_index = best_idx;
    Fit.best_criterion_value = best_score;
    Fit.centers = centers_by_K{best_idx};
    Fit.labels = labels_by_K{best_idx};
    Fit.cluster_weights = cluster_weights(Fit.labels, K_est);
    Fit.maps_nc = single(X);
    Fit.backfit_peak_labels = double(Fit.labels(:));
    Fit.preprocessing = struct( ...
        'apply_average_reference', cfg.apply_average_reference, ...
        'filter_band', cfg.filter_band, ...
        'gfp_peak_min_distance', cfg.gfp_peak_min_distance, ...
        'gfp_peak_threshold_schedule', cfg.gfp_peak_threshold_schedule);
    Fit.free_energy_vals = free_energy;
    Fit.silhouette_vals = silhouette_vals;
    Fit.gev_vals = gev_vals;
    Fit.within_ss = wss_vals;
    Fit.selection_score_by_k = score_by_k;
    Fit.selection_details = selection_details;
    Fit.model_comparison = model_comparison;
    Fit.spm_mix_model_summaries = summaries;
    Fit.selected_spm_mix_model = summaries(best_idx);
    Fit.centers_by_K = centers_by_K;
    Fit.labels_by_K = labels_by_K;
    Fit.n_maps = size(X, 1);
    Fit.real_observation_count = size(X, 1);
    Fit.n_channels = size(X, 2);
    Fit.prior_n_maps = size(prior_maps, 1);
    Fit.prior_source = char(prior_source);
    Fit.prior_state_index = 1:Fit.K_estimated;
    Fit.parent_map_sha256 = sha256_array(prior_maps);
    if isempty(prior_maps)
        Fit.prior_mode = 'none';
    else
        Fit.prior_mode = 'spm_mix_prior_component_means';
    end
    Fit.spm_prior_weight = cfg.spm_prior_weight;
    Fit.prior_beta_0 = prior_beta_by_K{best_idx};
    Fit.spm_ms_commit = '0dbde1146f6777412dce19edfa4962cd45157fca';
    Fit.spm_version = spm('Ver');
    Fit.spm_commit = '04db1be762077630772ee660a470a0fd63d4ff43';
    if Fit.n_maps ~= size(X, 1)
        error('Child observation count changed during hierarchical fitting.');
    end
    Fit.runtime = toc(t0);
end

function [output, detail] = run_recovery_command(request)
% Narrow public bridge for deterministic recovery from preserved peak caches.
    command = lower(strtrim(char(string(request.command))));
    detail = '';
    util = microstate_utilities();
    switch command
        case 'fit_global_maps'
            required = {'maps','gfp','cfg','common_labels','common_chanlocs','manual_labels'};
            for i = 1:numel(required)
                if ~isfield(request, required{i})
                    error('Recovery request is missing %s.', required{i});
                end
            end
            cfg = request.cfg;
            cfg.K_candidates = 6;
            cfg.spm_prior_weight = 1;
            if ~isfield(cfg, 'criterion'), cfg.criterion = 'free_energy_elbow'; end
            if ~isfield(cfg, 'template_reference_K'), cfg.template_reference_K = 7; end
            if ~isfield(cfg, 'template_file') || ~isfile(cfg.template_file)
                repo_cfg = util.load_config();
                cfg.template_file = repo_cfg.paths.template_file;
            end
            repo_cfg = util.load_config();
            [spm_ok, spm_info] = util.ensure_spm_mix('', repo_cfg.paths.spm_mixture_paths, false);
            if ~spm_ok
                error('SPM spm_mix not found. Checked: %s', strjoin(spm_info.attempted, ', '));
            end
            rng(1, 'twister');
            output = fit_spm_vb_peak_maps(request.maps, request.gfp, 6, cfg, ...
                'recovered_global', [], 'none');
            output = apply_manual_global_labels_by_state_order(output, request.manual_labels, cfg, ...
                request.common_labels, request.common_chanlocs);
            output.template_alignment.manual_label_source = 'recovered_authoritative_mapping';
        case 'extract_peak_record'
            required = {'file_path','common_can','common_pos','cfg'};
            for i = 1:numel(required)
                if ~isfield(request, required{i})
                    error('Recovery request is missing %s.', required{i});
                end
            end
            manifest = table(string(request.file_path), 'VariableNames', {'file_path'});
            meta = inspect_files(manifest, request.cfg, util);
            output = extract_file_peaks(request.file_path, meta(1), request.common_can, ...
                request.common_pos, request.cfg, util);
        case 'timestamped_transitions'
            required = {'file_path','common_can','common_pos','cfg','fit'};
            for i = 1:numel(required)
                if ~isfield(request, required{i})
                    error('Recovery request is missing %s.', required{i});
                end
            end
            [Sim, ~, X_metric] = prepare_backfit_record(request.file_path, request.common_can, ...
                request.common_pos, request.cfg, util);
            validate_full = isfield(request, 'validate_full_backfit') && request.validate_full_backfit;
            sfreq = double(Sim.sfreq);
            if ~validate_full
                Sim.X_noisy = [];
            end
            gfp = sqrt(mean((double(X_metric) - mean(double(X_metric), 1)).^2, 1));
            min_dist = max(1, round(double(request.cfg.gfp_peak_min_distance)));
            peak_sample = exact_local_peak_finder(gfp, min_dist);
            peak_maps = util.normalize_maps(double(X_metric(:, peak_sample))');
            centers = util.normalize_maps(double(request.fit.centers));
            [~, peak_assignment] = max(abs(peak_maps * centers'), [], 2);
            changes = find(peak_assignment(2:end) ~= peak_assignment(1:end-1));
            transition_sample = floor((peak_sample(changes) + peak_sample(changes + 1)) / 2) + 1;
            output = struct( ...
                'file_path', char(request.file_path), ...
                'sfreq', sfreq, ...
                'n_samples', double(size(X_metric, 2)), ...
                'peak_sample', uint32(peak_sample(:)), ...
                'peak_assignment', uint8(peak_assignment(:)), ...
                'transition_sample', uint32(transition_sample(:)), ...
                'from_state', uint8(peak_assignment(changes)), ...
                'to_state', uint8(peak_assignment(changes + 1)));
            output.full_backfit_validation = false;
            if validate_full
                backfit = backfit_microstate_timecourse(Sim, request.fit);
                if ~backfit.ok || ~strcmp(backfit.hard.mode, 'gfp_peaks_interpolated')
                    error('Full backfit validation failed for %s: %s', request.file_path, backfit.message);
                end
                full_assignment = uint8(backfit.hard.assignments(:));
                full_sample = uint32(find(full_assignment(2:end) ~= full_assignment(1:end-1)) + 1);
                if ~isequal(full_sample(:), output.transition_sample(:)) || ...
                        ~isequal(full_assignment(double(full_sample) - 1), output.from_state(:)) || ...
                        ~isequal(full_assignment(double(full_sample)), output.to_state(:))
                    error('Compact transition export disagrees with the full hard backfit for %s.', request.file_path);
                end
                output.full_backfit_validation = true;
            end
        otherwise
            error('Unknown recovery command: %s', command);
    end
end

function idx = exact_local_peak_finder(x, min_dist)
% Identical selection rule to backfit_microstate_timecourse.
    x = double(x(:)');
    idx = [];
    if numel(x) < 3, return; end
    cand = find(x(2:end-1) >= x(1:end-2) & x(2:end-1) > x(3:end)) + 1;
    if isempty(cand)
        [~, idx] = max(x);
        return;
    end
    [~, order] = sort(x(cand), 'descend');
    cand = cand(order);
    taken = false(size(x));
    keep = false(size(cand));
    for i = 1:numel(cand)
        c = cand(i);
        bounds = max(1, c - min_dist):min(numel(x), c + min_dist);
        if any(taken(bounds)), continue; end
        keep(i) = true;
        taken(bounds) = true;
    end
    idx = sort(cand(keep), 'ascend');
end

function prior_K = ensure_k_prior_maps(prior_maps, X, K)
    if isempty(prior_maps)
        prior_K = [];
        return;
    end
    prior_K = normalize_maps(prior_maps);
    if size(prior_K, 1) ~= K
        error('A hierarchical parent must contain exactly K=%d maps; received %d.', K, size(prior_K, 1));
    end
    if size(prior_K, 2) ~= size(X, 2)
        error('Parent and child maps must have the same channel count (%d versus %d).', size(prior_K, 2), size(X, 2));
    end
end

function [features, info, seed_features] = spm_features(X, seed_maps)
    if nargin < 2 || isempty(seed_maps)
        seed_maps = [];
    else
        seed_maps = normalize_maps(seed_maps);
    end
    X = normalize_maps(X);
    x_mean = mean(X, 1);
    Xc = X - x_mean;
    if isempty(seed_maps)
        seed_Xc = zeros(0, size(X, 2));
    else
        seed_Xc = seed_maps - x_mean;
    end
    [~, S, V] = svd(Xc, 'econ');
    latent = diag(S).^2 ./ max(1, size(Xc, 1) - 1);
    total = sum(latent);
    if total <= eps
        linear = Xc(:, 1:min(size(Xc, 2), 2));
        seed_linear = seed_Xc(:, 1:size(linear, 2));
        rank_est = size(linear, 2);
        var_pct = 0;
    else
        rank_est = nnz(latent > max(size(Xc)) * eps(max(latent)));
        cumv = cumsum(latent) ./ total;
        nd = find(cumv >= 0.999, 1, 'first');
        if isempty(nd), nd = rank_est; end
        nd = max(1, min([nd, rank_est, size(Xc, 1) - 1, size(Xc, 2) - 1, 8]));
        linear = Xc * V(:, 1:nd);
        seed_linear = seed_Xc * V(:, 1:nd);
        var_pct = 100 * cumv(nd);
    end
    sd = std(linear, 0, 1);
    sd(sd < eps) = 1;
    linear = linear ./ sd;
    seed_linear = seed_linear ./ sd;

    Y = normalize_rows(linear);
    seed_Y = normalize_rows(seed_linear);
    features = projective_features(Y);
    seed_features = projective_features(seed_Y);
    feature_mean = mean(features, 1);
    features = features - feature_mean;
    seed_features = seed_features - feature_mean;
    fsd = std(features, 0, 1);
    keep = fsd > 10 * eps;
    features = features(:, keep);
    seed_features = seed_features(:, keep);
    fsd = fsd(keep);
    if isempty(features)
        features = linear;
        seed_features = seed_linear;
        fsd = std(features, 0, 1);
    end
    fsd(fsd < eps) = 1;
    features = features ./ fsd;
    seed_features = seed_features ./ fsd;
    info = struct('rank_est', rank_est, 'linear_dims', size(linear, 2), ...
        'feature_dim', size(features, 2), 'variance_explained_pct', var_pct);
end

function features = projective_features(Y)
    D = size(Y, 2);
    features = zeros(size(Y, 1), D * (D + 1) / 2);
    col = 0;
    for a = 1:D
        for b = a:D
            col = col + 1;
            if a == b, scale = 1; else, scale = sqrt(2); end
            features(:, col) = scale * Y(:, a) .* Y(:, b);
        end
    end
end

function tf = valid_mix(mix, K)
    tf = isstruct(mix) && isfield(mix, 'state') && numel(mix.state) >= K && ...
        isfield(mix, 'fm') && isfinite(double(real(mix.fm)));
end

function labels = assign_mix(X, mix)
    K = mix.m;
    logp = -inf(size(X, 1), K);
    for k = 1:K
        mu = double(real(mix.state(k).m(:)'));
        C = double(real(mix.state(k).C));
        if isfield(mix.state(k), 'prior') && ~isempty(mix.state(k).prior)
            prior = double(real(mix.state(k).prior));
        else
            prior = 1 / K;
        end
        logp(:, k) = log(prior + eps) + log_gaussian(X, mu, C);
    end
    [~, labels] = max(logp, [], 2);
end

function y = log_gaussian(X, mu, C)
    D = size(X, 2);
    mu = mu(1:min(numel(mu), D));
    if numel(mu) < D
        mu = [mu zeros(1, D - numel(mu))];
    end
    C = coerce_cov(C, D);
    C = (C + C') / 2;
    ridge = max(1e-6, mean(diag(C), 'omitnan') * 1e-6);
    [R, p] = chol(C + ridge * eye(D));
    tries = 0;
    while p ~= 0 && tries < 5
        ridge = ridge * 10;
        [R, p] = chol(C + ridge * eye(D));
        tries = tries + 1;
    end
    if p ~= 0
        C = diag(max(var(X, 0, 1), 1e-6));
        R = chol(C);
    end
    Z = bsxfun(@minus, X, mu) / R;
    q = sum(Z.^2, 2);
    logdet = 2 * sum(log(diag(R)));
    y = -0.5 * (D * log(2*pi) + logdet + q);
end

function C = coerce_cov(C, D)
    if isscalar(C)
        C = C * eye(D);
    elseif isvector(C)
        v = C(:);
        C = diag(v(1:min(D, numel(v))));
    end
    if size(C, 1) ~= D || size(C, 2) ~= D
        out = eye(D);
        n = min([D, size(C, 1), size(C, 2)]);
        out(1:n, 1:n) = C(1:n, 1:n);
        C = out;
    end
end

function centers = recover_centers(X, labels, K, fallback)
    X = normalize_maps(X);
    fallback = normalize_maps(fallback);
    centers = zeros(K, size(X, 2));
    for k = 1:K
        idx = labels == k;
        if any(idx)
            Xk = X(idx, :);
            if size(Xk, 1) == 1
                c = Xk(1, :);
            else
                [~, ~, V] = svd(Xk, 'econ');
                c = V(:, 1)';
                if dot(c, mean(Xk, 1)) < 0
                    c = -c;
                end
            end
        elseif size(fallback, 1) >= k
            c = fallback(k, :);
        else
            c = X(mod(k - 1, size(X, 1)) + 1, :);
        end
        centers(k, :) = c;
    end
    centers = normalize_maps(centers);
end

function centers = refine_centers(X, centers, fallback, max_iter)
    duplicate_threshold = 0.90;
    centers = normalize_maps(centers);
    labels = assign_by_abs_correlation(X, centers);
    [labels, centers] = repair_duplicate_centers(X, labels, centers, fallback, duplicate_threshold);
    for it = 1:max_iter
        old = labels;
        old_centers = centers;
        centers = recover_centers(X, labels, size(centers, 1), fallback);
        labels = assign_by_abs_correlation(X, centers);
        [labels, centers] = repair_duplicate_centers(X, labels, centers, fallback, duplicate_threshold);
        if isequal(old, labels) && max(abs(centers(:) - old_centers(:))) < 1e-8
            break;
        end
    end
end

function [labels, centers] = repair_duplicate_centers(X, labels, centers, fallback, threshold)
    K = size(centers, 1);
    for attempt = 1:K
        pairs = duplicate_center_pairs(centers, threshold);
        if isempty(pairs)
            return;
        end
        pair = pairs(1, 1:2);
        counts = accumarray(labels(:), 1, [K, 1], @sum, 0);
        if counts(pair(1)) <= counts(pair(2))
            replace_idx = pair(1);
        else
            replace_idx = pair(2);
        end
        keep_idx = setdiff(1:K, replace_idx);
        sims = abs(normalize_maps(X) * normalize_maps(centers(keep_idx, :))');
        novelty = 1 - max(sims, [], 2);
        [~, seed_idx] = max(novelty);
        centers(replace_idx, :) = normalize_maps(X(seed_idx, :));
        labels = assign_by_abs_correlation(X, centers);
        centers = recover_centers(X, labels, K, fallback);
    end
end

function pairs = duplicate_center_pairs(centers, threshold)
    centers = normalize_maps(centers);
    signed_corr = centers * centers';
    pairs = [];
    for i = 1:(size(centers, 1) - 1)
        for j = (i + 1):size(centers, 1)
            if abs(signed_corr(i, j)) >= threshold
                pairs(end+1, :) = [i, j, signed_corr(i, j), abs(signed_corr(i, j))]; %#ok<AGROW>
            end
        end
    end
end

function metrics = topo_metrics(X, labels, centers, gfp)
    X = normalize_maps(X);
    centers = normalize_maps(centers);
    sim = abs(X * centers');
    [maxsim, ~] = max(sim, [], 2);
    w = double(gfp(:)).^2;
    if numel(w) ~= size(X, 1) || sum(w) <= eps
        w = ones(size(X, 1), 1);
    end
    metrics = struct();
    metrics.wss = sum(w .* (1 - maxsim.^2));
    metrics.gev = sum(w .* maxsim.^2) / sum(w);
    metrics.silhouette = silhouette_centroid(X, labels, centers);
end

function sil = silhouette_centroid(X, labels, centers)
    K = size(centers, 1);
    if K < 2 || size(X, 1) <= K
        sil = NaN;
        return;
    end
    dist = 1 - abs(X * centers');
    own = labels(:);
    a = dist(sub2ind(size(dist), (1:size(dist, 1))', own));
    dist(sub2ind(size(dist), (1:size(dist, 1))', own)) = Inf;
    b = min(dist, [], 2);
    vals = (b - a) ./ max(a, b);
    vals(~isfinite(vals)) = NaN;
    sil = mean(vals, 'omitnan');
end

function w = cluster_weights(labels, K)
    w = zeros(1, K);
    for k = 1:K
        w(k) = mean(labels == k);
    end
    w = w ./ max(sum(w), eps);
end

function s = empty_mix_summary()
    s = struct('K_candidate', NaN, 'feature_dim', NaN, 'free_energy', NaN, ...
        'priors', [], 'means', [], 'covariance_traces', [], 'covariance_logdets', [], 'feature_info', struct());
end

function s = summarise_mix(mix, K, feature_dim, feature_info)
    s = empty_mix_summary();
    s.K_candidate = K;
    s.feature_dim = feature_dim;
    s.free_energy = double(real(mix.fm));
    s.priors = nan(K, 1);
    s.means = nan(K, feature_dim);
    s.covariance_traces = nan(K, 1);
    s.covariance_logdets = nan(K, 1);
    s.feature_info = feature_info;
    for k = 1:K
        if isfield(mix.state(k), 'prior')
            s.priors(k) = double(real(mix.state(k).prior));
        end
        if isfield(mix.state(k), 'm')
            m = double(real(mix.state(k).m(:)'));
            s.means(k, 1:min(feature_dim, numel(m))) = m(1:min(feature_dim, numel(m)));
        end
        if isfield(mix.state(k), 'C')
            C = coerce_cov(double(real(mix.state(k).C)), feature_dim);
            s.covariance_traces(k) = trace(C);
            s.covariance_logdets(k) = safe_logdet(C);
        end
    end
end

function val = safe_logdet(C)
    val = NaN;
    if isempty(C) || any(~isfinite(C(:)))
        return;
    end
    C = (C + C') / 2;
    try
        [R, p] = chol(C);
        if p == 0
            val = 2 * sum(log(diag(R)));
        end
    catch
    end
end

function Fit = attach_template_alignment_if_available(Fit, cfg, common_labels, common_chanlocs)
    if isempty(cfg.template_file) || ~isfile(cfg.template_file)
        if cfg.require_template_labels
            error('Template labelling is required, but the template file was not found: %s', cfg.template_file);
        end
        return;
    end
    try
        alignment = align_microstates_to_template(Fit.centers, cfg.template_file, ...
            'estimated_channel_labels', common_labels, ...
            'estimated_chanlocs', common_chanlocs, ...
            'template_K', cfg.template_reference_K, ...
            'label_override', cfg.template_label_override, ...
            'strong_threshold', 0.5);
        Fit.template_alignment = alignment;
        Fit.centers = alignment.aligned_maps;
    catch ME
        if cfg.require_template_labels
            error('Template alignment failed for %s: %s', Fit.name, ME.message);
        end
        Fit.template_alignment = struct('labels', {{}}, 'mean_correlation', NaN, ...
            'n_strong_matches', NaN, 'message', ME.message);
    end
    if cfg.require_template_labels && ...
            (~isfield(Fit, 'template_alignment') || ...
             ~isfield(Fit.template_alignment, 'labels') || ...
             numel(Fit.template_alignment.labels) ~= Fit.K_estimated)
        error('Template alignment for %s did not provide exactly K=%d labels.', Fit.name, Fit.K_estimated);
    end
end

function save_fit_bundle(out_dir, prefix, Fit, rows, common_labels, common_chanlocs, cfg)
    ensure_dir(out_dir);
    save(fullfile(out_dir, [prefix '_solution.mat']), 'Fit', 'rows', 'common_labels', 'common_chanlocs', 'cfg', '-v7.3');
    write_matrix_csv(fullfile(out_dir, [prefix '_centers.csv']), Fit.centers);
    writetable(Fit.model_comparison, fullfile(out_dir, [prefix '_model_comparison.csv']));
    if istable(rows) && ~isempty(rows)
        writetable(rows, fullfile(out_dir, [prefix '_gfp_peak_manifest.csv']));
    end
    if cfg.save_plots
        plot_centers(Fit, common_chanlocs, fullfile(out_dir, [prefix '_centers.png']));
        plot_candidate_solutions(Fit, common_labels, common_chanlocs, cfg, fullfile(out_dir, [prefix '_candidate_solutions.png']));
    end
end

function plot_centers(Fit, chanlocs, out_file)
    try
        centers = Fit.centers;
        [plot_chanlocs, keep_idx] = prepare_fit_topoplot_chanlocs(chanlocs, size(centers, 2));
        if ~isempty(keep_idx)
            centers_plot = centers(:, keep_idx);
        else
            centers_plot = centers;
        end
        K = size(centers, 1);
        fig = figure('Visible', 'off', 'Color', 'white', 'Position', [100 100 max(900, 230*K) 300]);
        n_cols = K;
        n_rows = 1;
        for k = 1:K
            subplot(n_rows, n_cols, k);
            if ~isempty(plot_chanlocs) && exist('topoplot', 'file') == 2
                topoplot(centers_plot(k, :), plot_chanlocs, 'electrodes', 'off', 'numcontour', 6);
            else
                imagesc(centers(k, :)); axis tight; colorbar;
            end
            label = sprintf('State %d', k);
            if isfield(Fit, 'template_alignment') && isfield(Fit.template_alignment, 'labels') && numel(Fit.template_alignment.labels) >= k
                label = sprintf('%s r=%.2f', Fit.template_alignment.labels{k}, Fit.template_alignment.correlations(k));
            end
            title(label, 'Interpreter', 'none', 'Color', [0 0 0], 'FontWeight', 'bold');
        end
        st = sgtitle(Fit.name, 'Interpreter', 'none');
        set(st, 'Color', [0 0 0], 'FontWeight', 'bold');
        saveas(fig, out_file);
        close(fig);
    catch ME
        warning('Plot failed for %s: %s', out_file, ME.message);
    end
end

function plot_candidate_solutions(Fit, common_labels, chanlocs, cfg, out_file)
    try
        if ~isfield(Fit, 'centers_by_K') || numel(Fit.centers_by_K) < 2
            return;
        end
        K_candidates = double(Fit.K_candidates(:));
        valid_idx = find(~cellfun(@isempty, Fit.centers_by_K(:)));
        if numel(valid_idx) < 2
            return;
        end

        first_centers = Fit.centers_by_K{valid_idx(1)};
        n_channels = size(first_centers, 2);
        [plot_chanlocs, keep_idx] = prepare_fit_topoplot_chanlocs(chanlocs, n_channels);

        rows = repmat(struct('K', NaN, 'maps', [], 'labels', {{}}, 'correlations', []), numel(valid_idx), 1);
        for r = 1:numel(valid_idx)
            iK = valid_idx(r);
            K = K_candidates(iK);
            centers = normalize_maps(Fit.centers_by_K{iK});
            [maps, labels, correlations] = align_candidate_centers_for_plot(centers, K, cfg, common_labels, chanlocs);
            rows(r).K = K;
            rows(r).maps = maps;
            rows(r).labels = labels;
            rows(r).correlations = correlations;
        end

        col_labels = string(cellstr(('A':'G')'))';

        n_rows = numel(rows);
        n_cols = numel(col_labels);
        row_maps = cell(n_rows, 1);
        row_corr = cell(n_rows, 1);
        for r = 1:n_rows
            maps = nan(n_cols, n_channels);
            correlations = nan(n_cols, 1);
            for k = 1:size(rows(r).maps, 1)
                col = find(col_labels == string(rows(r).labels{k}), 1, 'first');
                if isempty(col)
                    continue;
                end
                maps(col, :) = rows(r).maps(k, :);
                if numel(rows(r).correlations) >= k
                    correlations(col) = rows(r).correlations(k);
                end
            end
            row_maps{r} = maps;
            row_corr{r} = correlations;
        end

        all_maps = vertcat(row_maps{:});
        clim = max(abs(all_maps(isfinite(all_maps))));
        if isempty(clim) || ~isfinite(clim) || clim <= eps
            clim = 1;
        end

        fig = figure('Visible', 'off', 'Color', 'white', ...
            'Position', [80 80 max(900, 180 * n_cols) max(420, 170 * n_rows)]);
        colormap(fig, 'jet');
        for r = 1:n_rows
            selected_row = rows(r).K == double(Fit.K_estimated);
            row_color = [0 0 0];
            row_label = sprintf('K=%d', rows(r).K);
            if selected_row
                row_color = [0 0.45 0];
                row_label = sprintf('K=%d\nBEST', rows(r).K);
            end
            for c = 1:n_cols
                ax = subplot(n_rows, n_cols, (r - 1) * n_cols + c);
                map = row_maps{r}(c, :);
                if all(~isfinite(map))
                    axis(ax, 'off');
                    text(ax, 0.5, 0.5, '-', 'Units', 'normalized', ...
                        'HorizontalAlignment', 'center', 'FontSize', 14, 'Color', [0.55 0.55 0.55]);
                else
                    if ~isempty(keep_idx)
                        map_plot = map(keep_idx);
                    else
                        map_plot = map;
                    end
                    if ~isempty(plot_chanlocs) && exist('topoplot', 'file') == 2
                        topoplot(map_plot, plot_chanlocs, 'electrodes', 'off', 'numcontour', 6, 'maplimits', [-clim clim]);
                    else
                        imagesc(map); axis tight; caxis([-clim clim]); colorbar;
                    end
                    axis(ax, 'off');
                    if isfinite(row_corr{r}(c))
                        text(ax, 0.5, -0.08, sprintf('r=%.2f', row_corr{r}(c)), ...
                            'Units', 'normalized', 'HorizontalAlignment', 'center', ...
                            'FontSize', 8, 'Color', [0.25 0.25 0.25]);
                    end
                end
                if r == 1
                    title(ax, char(col_labels(c)), 'Interpreter', 'none', 'FontWeight', 'bold');
                end
                if c == 1
                    text(ax, -0.34, 0.5, row_label, 'Units', 'normalized', ...
                        'HorizontalAlignment', 'center', 'FontWeight', 'bold', ...
                        'Color', row_color, 'Interpreter', 'none');
                end
            end
        end
        st = sgtitle(sprintf('%s candidate K solutions (%s)', Fit.name, Fit.criterion), 'Interpreter', 'none');
        set(st, 'Color', [0 0 0], 'FontWeight', 'bold');
        saveas(fig, out_file);
        close(fig);
    catch ME
        warning('Candidate-solution plot failed for %s: %s', out_file, ME.message);
    end
end

function [maps, labels, correlations] = align_candidate_centers_for_plot(centers, K, cfg, common_labels, chanlocs)
    maps = normalize_maps(centers);
    labels = arrayfun(@(k) sprintf('State %d', k), 1:size(maps, 1), 'UniformOutput', false);
    correlations = nan(size(maps, 1), 1);
    if isempty(cfg.template_file) || ~isfile(cfg.template_file)
        return;
    end

    label_override = {};
    if numel(cfg.template_label_override) == K
        label_override = cfg.template_label_override;
    end
    try
        alignment = align_microstates_to_template(maps, cfg.template_file, ...
            'estimated_channel_labels', common_labels, ...
            'estimated_chanlocs', chanlocs, ...
            'template_K', cfg.template_reference_K, ...
            'label_override', label_override, ...
            'strong_threshold', 0.5);
        maps = alignment.aligned_maps;
        labels = alignment.labels;
        correlations = alignment.correlations;
    catch
    end
end

function [chanlocs_out, keep_idx] = prepare_fit_topoplot_chanlocs(chanlocs, n_channels)
    chanlocs_out = [];
    keep_idx = [];
    if isempty(chanlocs)
        return;
    end

    n = min(n_channels, numel(chanlocs));
    chanlocs_out = chanlocs(1:n);
    keep = false(1, n);
    for i = 1:n
        chanlocs_out = set_mne_topoplot_location_fields(chanlocs_out, i);
        theta = field_double(chanlocs_out(i), 'theta');
        radius = field_double(chanlocs_out(i), 'radius');
        keep(i) = isfinite(theta) && isfinite(radius) && radius >= 0 && radius <= 1;
    end
    keep_idx = find(keep);
    chanlocs_out = chanlocs_out(keep_idx);
end

function chanlocs = set_mne_topoplot_location_fields(chanlocs, idx)
    xyz = [field_double(chanlocs(idx), 'X'), field_double(chanlocs(idx), 'Y'), field_double(chanlocs(idx), 'Z')];
    if all(isfinite(xyz)) && norm(xyz) > eps
        [az, el, sph_r] = cart2sph(xyz(1), xyz(2), xyz(3));
        theta = atan2d(xyz(1), xyz(2));
        radius = 0.5 - rad2deg(el) / 180;
        chanlocs(idx).theta = theta;
        chanlocs(idx).radius = radius;
        chanlocs(idx).sph_theta = rad2deg(az);
        chanlocs(idx).sph_phi = rad2deg(el);
        chanlocs(idx).sph_radius = sph_r;
    end
end

function write_matrix_csv(file_path, X)
    T = array2table(double(X));
    names = cell(1, size(X, 2));
    for i = 1:size(X, 2)
        names{i} = sprintf('ch_%03d', i);
    end
    T.Properties.VariableNames = names;
    writetable(T, file_path);
end

function row = make_summary_row(level, name, group, condition, participant, Fit, n_units, n_maps, fit_path)
    [mean_corr, strong] = alignment_score(Fit);
    prior_source = string(field_value_or_default(Fit, 'prior_source', ''));
    prior_mode = string(field_value_or_default(Fit, 'prior_mode', ''));
    spm_prior_weight = double(field_value_or_default(Fit, 'spm_prior_weight', NaN));
    prior_state_index = string(strjoin(string(field_value_or_default(Fit, 'prior_state_index', [])), ';'));
    prior_beta_0 = string(strjoin(compose('%.12g', double(field_value_or_default(Fit, 'prior_beta_0', []))), ';'));
    parent_map_sha256 = string(field_value_or_default(Fit, 'parent_map_sha256', ''));
    spm_ms_commit = string(field_value_or_default(Fit, 'spm_ms_commit', ''));
    spm_version = string(field_value_or_default(Fit, 'spm_version', ''));
    spm_commit = string(field_value_or_default(Fit, 'spm_commit', ''));
    real_observation_count = double(field_value_or_default(Fit, 'real_observation_count', n_maps));
    row = table(string(level), string(name), string(group), string(condition), string(participant), ...
        double(Fit.K_estimated), double(n_units), double(n_maps), real_observation_count, ...
        prior_source, prior_mode, spm_prior_weight, prior_state_index, prior_beta_0, parent_map_sha256, ...
        spm_ms_commit, spm_version, spm_commit, double(mean_corr), double(strong), string(fit_path), ...
        'VariableNames', {'level', 'name', 'group', 'condition', 'participant', 'K_estimated', 'n_units', 'n_maps', ...
        'real_observation_count','prior_source','prior_mode','spm_prior_weight','prior_state_index','prior_beta_0', ...
        'parent_map_sha256','spm_ms_commit','spm_version','spm_commit','template_mean_corr','template_strong_matches','fit_path'});
end

function [mean_corr, strong] = alignment_score(Fit)
    mean_corr = NaN;
    strong = NaN;
    if isfield(Fit, 'template_alignment')
        A = Fit.template_alignment;
        if isfield(A, 'mean_correlation'), mean_corr = A.mean_correlation; end
        if isfield(A, 'n_strong_matches'), strong = A.n_strong_matches; end
    end
end

function node = build_node(level, participant, group, condition, name, Fit, n_maps, fit_path)
    node = struct();
    node.level = string(level);
    node.participant = string(participant);
    node.group = string(group);
    node.condition = string(condition);
    node.name = string(name);
    node.centers = Fit.centers;
    node.K_estimated = Fit.K_estimated;
    node.n_maps = n_maps;
    node.fit = Fit;
    if isfield(Fit, 'template_alignment') && isfield(Fit.template_alignment, 'labels')
        node.canonical_state_labels = Fit.template_alignment.labels;
        node.template_alignment = Fit.template_alignment;
    else
        node.canonical_state_labels = {};
        node.template_alignment = struct();
    end
    node.fit_path = string(fit_path);
end

function [labels, changed, source] = resolve_global_manual_labels(Fit, cfg, out_dir, common_chanlocs)
    labels = {};
    source = '';
    old_labels = {};
    if isfield(Fit, 'template_alignment') && isfield(Fit.template_alignment, 'labels')
        old_labels = cellstr(string(Fit.template_alignment.labels(:)));
    end

    label_file = fullfile(out_dir, 'global_manual_labels.mat');
    explicit = cellstr(string(cfg.global_manual_labels(:)));
    if ~cfg.relabel_global_states && ~isempty(explicit)
        labels = validate_global_labels(explicit, size(Fit.centers, 1));
        source = 'command_line';
    elseif ~cfg.relabel_global_states && cfg.resume && isfile(label_file)
        cached = load(label_file, 'labels', 'reference_centers');
        if isfield(cached, 'labels') && isfield(cached, 'reference_centers') && ...
                same_state_maps(Fit.centers, cached.reference_centers)
            labels = validate_global_labels(cached.labels, size(Fit.centers, 1));
            source = 'saved_manual_labels';
            if cfg.verbose
                fprintf('Reusing saved manual global labels: %s\n', strjoin(labels, ', '));
            end
        elseif cfg.verbose
            fprintf('Ignoring stale manual-label cache because the global maps changed.\n');
        end
    end

    if isempty(labels) && (cfg.interactive_global_labelling || cfg.relabel_global_states)
        preview = Fit;
        if isfield(preview, 'template_alignment')
            preview = rmfield(preview, 'template_alignment');
        end
        preview_file = fullfile(out_dir, 'global_centers_for_manual_labelling.png');
        plot_centers(preview, common_chanlocs, preview_file);
        labels = prompt_for_global_labels(size(Fit.centers, 1), preview_file);
        source = 'interactive_terminal';
    end

    changed = ~isempty(labels) && ~same_labels(labels, old_labels);
    if isempty(labels) || strcmp(source, 'saved_manual_labels')
        return;
    end
    reference_centers = Fit.centers;
    selected_utc = string(datetime('now', 'TimeZone', 'UTC', 'Format', 'yyyy-MM-dd''T''HH:mm:ssXXX'));
    label_source = source;
    save(label_file, 'labels', 'reference_centers', 'selected_utc', 'label_source', '-v7');
    writetable(table((1:numel(labels))', string(labels(:)), repmat(selected_utc, numel(labels), 1), ...
        repmat(string(source), numel(labels), 1), ...
        'VariableNames', {'state_index','label','selected_utc','source'}), ...
        fullfile(out_dir, 'global_manual_labels.csv'));
end

function labels = prompt_for_global_labels(K, preview_file)
    [check, valid] = parse_global_labels('F E G B A C', 6);
    assert(valid && isequal(check, {'F';'E';'G';'B';'A';'C'}), 'Manual-label parser self-check failed.');
    fprintf('\nMANUAL GLOBAL MICROSTATE LABELLING REQUIRED\n');
    fprintf('Open this unlabelled State 1..%d figure:\n  %s\n', K, preview_file);
    fprintf('Enter one unique A-G label per fitted state, in State 1..%d order.\n', K);
    fprintf('Example: F,E,G,B,A,C   (type QUIT to stop without saving)\n\n');
    if ispc
        try
            winopen(preview_file);
        catch
        end
    end
    while true
        try
            raw = input(sprintf('Labels for States 1..%d: ', K), 's');
        catch ME
            error('Interactive labelling needs a MATLAB terminal with stdin: %s', ME.message);
        end
        if strcmpi(strtrim(raw), 'QUIT')
            error('Stopped before manual global labels were saved.');
        end
        [labels, valid] = parse_global_labels(raw, K);
        if ~valid
            fprintf('Invalid entry. Supply %d unique A-G labels including A, C and E.\n', K);
            continue;
        end
        fprintf('Selected mapping: %s\n', strjoin(compose('State %d=%s', (1:K)', string(labels)), ', '));
        confirm = input('Save this mapping and continue? [y/N]: ', 's');
        if strcmpi(strtrim(confirm), 'y') || strcmpi(strtrim(confirm), 'yes')
            return;
        end
    end
end

function C = compact_hresults_for_save(H)
    C = H;
    embedded = {'state_covariance_metrics','peak_records','backfit_state_metrics', ...
        'backfit_pairwise_metrics','backfit_record_summary','backfit_axis_dynamics', ...
        'backfit_preictal_time_metrics','backfit_transition_metrics', ...
        'participant_ce_cardiac_phase_metrics','global_backfit_state_metrics', ...
        'global_backfit_pairwise_metrics','global_backfit_record_summary', ...
        'global_backfit_preictal_time_metrics','global_backfit_transition_metrics', ...
        'global_c_cardiac_phase_metrics','participant_level','participant_condition_level'};
    for i = 1:numel(embedded)
        if isfield(C, embedded{i}), C = rmfield(C, embedded{i}); end
    end
    node_fields = {'global','groups','conditions','participants','participant_conditions'};
    for i = 1:numel(node_fields)
        field = node_fields{i};
        if isfield(C, field) && isstruct(C.(field)) && isfield(C.(field), 'fit')
            C.(field) = rmfield(C.(field), 'fit');
        end
    end
    C.storage_mode = 'compact_hierarchy_with_csv_backfit_tables';
end

function [labels, valid] = parse_global_labels(raw, K)
    labels = regexp(upper(strtrim(char(raw))), '[,;/\s]+', 'split');
    labels = labels(~cellfun('isempty', labels));
    labels = labels(:);
    valid = numel(labels) == K && numel(unique(labels)) == K && ...
        all(~cellfun('isempty', regexp(labels, '^[A-G]$', 'once'))) && ...
        all(ismember({'A','C','E'}, labels));
    if ~valid
        labels = {};
    end
end

function labels = validate_global_labels(labels, K)
    [labels, valid] = parse_global_labels(strjoin(cellstr(string(labels(:))), ','), K);
    if ~valid
        error('Global manual labels must contain %d unique A-G labels including A, C and E.', K);
    end
end

function tf = same_labels(a, b)
    tf = numel(a) == numel(b) && all(upper(strtrim(string(a(:)))) == upper(strtrim(string(b(:)))));
end

function tf = same_state_maps(a, b)
    if ~isequal(size(a), size(b))
        tf = false;
        return;
    end
    a = normalize_maps(double(a));
    b = normalize_maps(double(b));
    tf = all(abs(sum(a .* b, 2)) > 1 - 1e-8);
end

function Fit = apply_manual_global_labels_by_state_order(Fit, labels, cfg, common_labels, common_chanlocs)
    labels = cellstr(string(labels(:)));
    if isempty(labels)
        return;
    end
    K = size(Fit.centers, 1);
    if numel(labels) ~= K || numel(unique(lower(string(labels)))) ~= K
        error('global_manual_labels must contain exactly %d unique labels.', K);
    end
    alignment = align_microstates_to_template(Fit.centers, cfg.template_file, ...
        'estimated_channel_labels', common_labels, ...
        'estimated_chanlocs', common_chanlocs, ...
        'template_K', cfg.template_reference_K, ...
        'label_override', labels, ...
        'strong_threshold', 0.5);
    alignment.manual_label_override = true;
    alignment.manual_label_note = 'Authoritative labels assigned by global fitted-state order.';
    Fit.template_alignment = alignment;
    Fit.centers = alignment.aligned_maps;
end

function Fit = align_fit_to_reference(Fit, reference_fit)
    reference = normalize_maps(double(reference_fit.centers));
    estimated = normalize_maps(double(Fit.centers));
    K = size(reference, 1);
    if size(estimated, 1) ~= K
        error('Reference alignment requires equal K; reference=%d fitted=%d.', K, size(estimated, 1));
    end
    recovery = microstate_partial_alignment(reference, estimated, 'threshold', 0, 'polarity', true);
    assignment = recovery.match_assignment;
    if size(assignment, 1) ~= K || numel(unique(assignment(:, 1))) ~= K || numel(unique(assignment(:, 2))) ~= K
        error('Reference alignment failed to obtain a one-to-one K=%d assignment.', K);
    end
    signed = estimated * reference';
    centers = nan(size(estimated));
    correlations = nan(K, 1);
    polarity = ones(K, 1);
    old_to_new = nan(K, 1);
    for r = 1:size(assignment, 1)
        old = assignment(r, 1);
        new = assignment(r, 2);
        s = sign(signed(old, new));
        if s == 0 || ~isfinite(s), s = 1; end
        centers(new, :) = s * estimated(old, :);
        correlations(new) = abs(signed(old, new));
        polarity(new) = s;
        old_to_new(old) = new;
    end
    Fit.centers = centers;
    if isfield(Fit, 'labels'), Fit.labels = remap_state_indices_local(Fit.labels, old_to_new); end
    if isfield(Fit, 'backfit_peak_labels'), Fit.backfit_peak_labels = remap_state_indices_local(Fit.backfit_peak_labels, old_to_new); end
    if isfield(Fit, 'cluster_weights'), Fit.cluster_weights = reorder_state_rows(Fit.cluster_weights, old_to_new); end
    if isfield(Fit, 'centers_by_K') && numel(Fit.centers_by_K) >= Fit.best_index
        Fit.centers_by_K{Fit.best_index} = centers;
    end
    if isfield(Fit, 'selected_spm_mix_model')
        Fit.selected_spm_mix_model = reorder_mix_summary(Fit.selected_spm_mix_model, old_to_new);
    end
    if isfield(Fit, 'spm_mix_model_summaries') && numel(Fit.spm_mix_model_summaries) >= Fit.best_index
        Fit.spm_mix_model_summaries(Fit.best_index) = Fit.selected_spm_mix_model;
    end
    [labels, ~, reference_corr] = state_labels_for_metrics(reference_fit);
    Fit.template_alignment = struct('labels', {labels}, 'correlations', correlations, ...
        'reference_correlations', reference_corr, 'polarity', polarity, ...
        'aligned_maps', centers, 'old_order_to_new_order', old_to_new, ...
        'reference_state_indices', (1:K)', 'mean_correlation', mean(correlations), ...
        'median_correlation', median(correlations), 'min_correlation', min(correlations), ...
        'n_strong_matches', sum(correlations >= 0.5), 'strong_threshold', 0.5, ...
        'manual_label_override', true, 'alignment_source', 'global_manual_reference');
end

function labels = remap_state_indices_local(labels, old_to_new)
    original = labels;
    for old = 1:numel(old_to_new)
        labels(original == old) = old_to_new(old);
    end
end

function values = reorder_state_rows(values, old_to_new)
    original = values;
    values = nan(size(original));
    if isvector(original)
        for old = 1:numel(old_to_new), values(old_to_new(old)) = original(old); end
    else
        for old = 1:numel(old_to_new), values(old_to_new(old), :) = original(old, :); end
    end
end

function summary = reorder_mix_summary(summary, old_to_new)
    fields = {'priors','means','covariance_traces','covariance_logdets'};
    for f = 1:numel(fields)
        name = fields{f};
        if isfield(summary, name) && ~isempty(summary.(name))
            summary.(name) = reorder_state_rows(summary.(name), old_to_new);
        end
    end
end

function Fit = load_cached_fit(out_dir, prefix, cfg, ~, ~)
    Fit = [];
    solution_file = fullfile(out_dir, [prefix '_solution.mat']);
    if ~cfg.resume || ~isfile(solution_file)
        return;
    end
    cached = load(solution_file, 'Fit');
    if isfield(cached, 'Fit')
        Fit = cached.Fit;
        if cfg.verbose
            fprintf('Resuming completed fit: %s\n', solution_file);
        end
    end
end

function nodes = empty_nodes(n_channels)
    nodes = repmat(build_node("", "", "", "", "", struct('centers', zeros(1, n_channels), 'K_estimated', 0), 0, ""), 0, 1);
end

function assert_fixed_hierarchy_k(K, varargin)
    for a = 1:numel(varargin)
        nodes = varargin{a};
        for i = 1:numel(nodes)
            if double(nodes(i).K_estimated) ~= double(K) || size(nodes(i).centers, 1) ~= double(K)
                error('Hierarchy node %s has K=%d; all nodes must retain global K=%d.', ...
                    char(nodes(i).name), double(nodes(i).K_estimated), double(K));
            end
        end
    end
end

function print_alignment_line(label, Fit)
    if ~isfield(Fit, 'template_alignment')
        return;
    end
    A = Fit.template_alignment;
    if isfield(A, 'mean_correlation') && isfinite(A.mean_correlation)
        fprintf('%s template fit: mean r=%.3f, strong=%d/%d\n', label, A.mean_correlation, A.n_strong_matches, Fit.K_estimated);
    end
end

function labels = assign_by_abs_correlation(maps, centers)
    X = normalize_maps(double(maps));
    C = normalize_maps(double(centers));
    [~, labels] = max(abs(X * C'), [], 2);
end

function Xn = normalize_maps(X)
    X = double(X);
    if isempty(X)
        Xn = X;
        return;
    end
    X = X - mean(X, 2, 'omitnan');
    denom = sqrt(sum(X.^2, 2));
    denom(~isfinite(denom) | denom <= eps) = 1;
    Xn = X ./ denom;
end

function Xn = normalize_rows(X)
    denom = sqrt(sum(X.^2, 2));
    denom(~isfinite(denom) | denom <= eps) = 1;
    Xn = X ./ denom;
end

function idx = deterministic_subsample(N, maxN)
    if isempty(maxN) || ~isfinite(maxN) || N <= maxN
        idx = (1:N)';
        return;
    end
    idx = unique(round(linspace(1, N, round(maxN))))';
    if numel(idx) > maxN
        idx = idx(1:maxN);
    end
end

function idx = capped_mask_indices(mask, maxN)
    idx = find(mask);
    keep = deterministic_subsample(numel(idx), maxN);
    idx = idx(keep);
end

function ensure_dir(pth)
    if ~exist(pth, 'dir')
        mkdir(pth);
    end
end

function s = onoff(tf)
    if tf
        s = 'ON';
    else
        s = 'OFF';
    end
end

function key = clean_key(value)
    key = char(string(value));
    key = regexprep(key, '[^A-Za-z0-9_-]', '_');
    if isempty(key)
        key = 'empty';
    end
end
