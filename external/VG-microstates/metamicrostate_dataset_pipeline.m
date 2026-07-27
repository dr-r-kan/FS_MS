function [MResults, output_csv] = metamicrostate_dataset_pipeline(manifest_csv, varargin)
% METAMICROSTATE_DATASET_PIPELINE
%
% A dataset-level EEG microstate pipeline that first builds a common-channel
% GFP-peak bank, then fits a hierarchical microstate model over those pooled
% peaks, and finally derives participant-driven meta-microstates.
%
% Required input CSV columns:
%   file_path
%
% Optional input CSV columns:
%   participant, condition, group
%
% If participant is missing, empty, or uninformative (e.g. all rows say
% "EEG"), participant IDs are inferred from file names such as
% sub-010300_EC.set -> sub-010300.
%
% The pipeline does five things:
%   1. Calls analyze_single_eeg_file on every CSV row when requested, while
%      independently caching all GFP-peak maps on a common channel set.
%   2. Fits a hierarchical GFP-peak model:
%         global -> group (if present) -> participant -> participant_condition
%         -> file
%      using parent templates as empirical-Bayes priors.
%   3. Writes conditioned participant solutions and per-row/file solutions
%      back to the manifest, so each participant/condition row has a fitted
%      microstate result.
%   4. Builds global meta-microstates from the participant-level templates,
%      treating those participant templates as pseudo-GFP peaks.
%   5. When conditions are present, builds condition-specific population
%      meta-microstate solutions from the participant-conditioned templates.
%
% Core output:
%   <output_dir>/cluster_solution_manifest.csv
%   with columns participant, condition, group, file_path,
%   cluster_solution_file, gfp_peak_cache_file, local_K, status.
%   Hierarchical summaries are written under hierarchical_gfp_clusters and
%   condition-level meta summaries are written under meta_clusters/subsets.
%
% Example:
%   [R, csv] = metamicrostate_dataset_pipeline('conditioned_lemon_sets.csv', ...
%       'output_dir', 'lemon_meta_microstates', ...
%       'K_candidates', 2:10, ...
%       'method', 'spm_vb', ...
%       'criterion', 'elbow_sil_combined', ...
%       'template_file', 'MetaMaps_2023_06.set');
%
% Notes:
%   - Polarity is treated axially: maps with opposite sign are equivalent.
%   - Template alignment, if possible, uses a genuine optimal subset/permutation
%     assignment against the supplied template maps. It does not assume that a
%     K=4 solution corresponds to template states 1:4.
%   - By default, GFP peaks are extracted on each file's observed scalp
%     channels and then remapped onto the densest scalp montage available in
%     the dataset by direct channel matching plus peak-level IDW
%     interpolation for missing channels.
%   - The per-file call to analyze_single_eeg_file is retained for compatibility
%     with the existing single-file pipeline. The common-channel per-file fit is
%     separately saved because analyze_single_eeg_file does not expose the GFP
%     peaks it used internally.
%
% Requirements:
%   - EEGLAB for .set files.
%   - analyze_single_eeg_file.m on the MATLAB path if call_analyze_single=true.
%   - Signal Processing Toolbox is useful for Butterworth filtering, but the
%     script will continue without filtering if butter/filtfilt are unavailable.

    setup_microstate_paths();
    t0 = tic;
    p = inputParser;
    addRequired(p, 'manifest_csv', @(x) ischar(x) || isstring(x));

    addParameter(p, 'output_dir', 'meta_microstate_output', @(x) ischar(x) || isstring(x));
    addParameter(p, 'method', 'spm_vb', @(x) ischar(x) || isstring(x));
    addParameter(p, 'criterion', 'elbow_sil_combined', @(x) ischar(x) || isstring(x));
    addParameter(p, 'K_candidates', 4:7, @(x) isnumeric(x) && isvector(x));
    addParameter(p, 'meta_K_candidates', [], @(x) isempty(x) || (isnumeric(x) && isvector(x)));
    addParameter(p, 'pooled_K_candidates', [], @(x) isempty(x) || (isnumeric(x) && isvector(x)));

    addParameter(p, 'call_analyze_single', true, @(x) islogical(x) && isscalar(x));
    addParameter(p, 'fail_on_analyze_single_error', false, @(x) islogical(x) && isscalar(x));
    addParameter(p, 'save_json', false, @(x) islogical(x) && isscalar(x));
    addParameter(p, 'json_dir', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'plot_dir', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'export_backfit_state_metrics', true, @(x) islogical(x) && isscalar(x));
    addParameter(p, 'backfit_state_metrics_csv', 'participant_condition_state_backfit_metrics.csv', @(x) ischar(x) || isstring(x));
    addParameter(p, 'backfit_pairwise_metrics_csv', 'participant_condition_state_pairwise_backfit_metrics.csv', @(x) ischar(x) || isstring(x));
    addParameter(p, 'backfit_record_summary_csv', 'participant_condition_record_backfit_summary.csv', @(x) ischar(x) || isstring(x));

    addParameter(p, 'template_file', 'MetaMaps_2023_06.set', @(x) ischar(x) || isstring(x));
    addParameter(p, 'align_to_template', true, @(x) islogical(x) && isscalar(x));
    addParameter(p, 'strong_template_corr', 0.65, @(x) isnumeric(x) && isscalar(x) && x >= 0 && x <= 1);

    addParameter(p, 'apply_average_reference', true, @(x) islogical(x) && isscalar(x));
    addParameter(p, 'filter_band', [2 20], @(x) isempty(x) || (isnumeric(x) && numel(x) == 2));
    addParameter(p, 'use_scalp_channels', true, @(x) islogical(x) && isscalar(x));
    addParameter(p, 'channel_policy', 'intersect', @(x) ischar(x) || isstring(x));
    addParameter(p, 'interpolate_missing_peak_channels', true, @(x) islogical(x) && isscalar(x));
    addParameter(p, 'peak_interpolation_method', 'idw', @(x) ischar(x) || isstring(x));
    addParameter(p, 'peak_interpolation_neighbours', 6, @(x) isnumeric(x) && isscalar(x) && x >= 1);
    addParameter(p, 'peak_interpolation_min_source_channels', 8, @(x) isnumeric(x) && isscalar(x) && x >= 3);
    addParameter(p, 'sfreq', 250, @(x) isnumeric(x) && isscalar(x) && x > 0);

    addParameter(p, 'gfp_peak_min_distance_samples', 3, @(x) isnumeric(x) && isscalar(x) && x >= 1);
    addParameter(p, 'gfp_peak_quantile_schedule', [0.50 0.60 0.70 0.80 0.90], @(x) isnumeric(x) && isvector(x));
    addParameter(p, 'min_gfp_peaks_per_file', 20, @(x) isnumeric(x) && isscalar(x) && x >= 1);
    addParameter(p, 'max_maps_per_file', 1500, @(x) isempty(x) || (isnumeric(x) && isscalar(x) && x > 0));
    addParameter(p, 'max_global_maps', 40000, @(x) isempty(x) || (isnumeric(x) && isscalar(x) && x > 0));

    addParameter(p, 'use_template_prior_global', true, @(x) islogical(x) && isscalar(x));
    addParameter(p, 'canonical_prior_weight_global', NaN, @(x) isnumeric(x) && isscalar(x) && x >= 0);
    addParameter(p, 'prior_weight_group', NaN, @(x) isnumeric(x) && isscalar(x) && x >= 0);
    addParameter(p, 'prior_weight_condition', NaN, @(x) isnumeric(x) && isscalar(x) && x >= 0);
    addParameter(p, 'prior_weight_participant', NaN, @(x) isnumeric(x) && isscalar(x) && x >= 0);
    addParameter(p, 'prior_weight_file', NaN, @(x) isnumeric(x) && isscalar(x) && x >= 0);

    addParameter(p, 'reject_gfp_peak_outliers_individual', true, @(x) islogical(x) && isscalar(x));
    addParameter(p, 'gfp_outlier_mad_multiplier_individual', 6, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'reject_gfp_peak_outliers_population', true, @(x) islogical(x) && isscalar(x));
    addParameter(p, 'gfp_outlier_mad_multiplier_population', 6, @(x) isnumeric(x) && isscalar(x) && x > 0);

    addParameter(p, 'n_initialisations', 50, @(x) isnumeric(x) && isscalar(x) && x >= 1);
    addParameter(p, 'max_iter', 300, @(x) isnumeric(x) && isscalar(x) && x >= 1);
    addParameter(p, 'tol', 1e-7, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'random_seed', 1, @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'use_parfor', true, @(x) islogical(x) && isscalar(x));
    addParameter(p, 'n_workers', 8, @(x) isnumeric(x) && isscalar(x) && x >= 1);
    addParameter(p, 'force_recompute', true, @(x) islogical(x) && isscalar(x));
    addParameter(p, 'force_recompute_peaks', false, @(x) islogical(x) && isscalar(x));
    addParameter(p, 'force_recompute_clusters', false, @(x) islogical(x) && isscalar(x));
    addParameter(p, 'infer_participant_from_path', true, @(x) islogical(x) && isscalar(x));
    addParameter(p, 'verbose', true, @(x) islogical(x) && isscalar(x));
    parse(p, manifest_csv, varargin{:});

    cfg = p.Results;
    cfg.manifest_csv = char(cfg.manifest_csv);
    cfg.output_dir = char(cfg.output_dir);
    cfg.method = lower(char(cfg.method));
    cfg.criterion = lower(char(cfg.criterion));
    cfg.template_file = char(cfg.template_file);
    cfg.channel_policy = lower(char(cfg.channel_policy));
    cfg.peak_interpolation_method = lower(char(cfg.peak_interpolation_method));
    cfg.json_dir = char(cfg.json_dir);
    cfg.plot_dir = char(cfg.plot_dir);
    cfg.backfit_state_metrics_csv = char(cfg.backfit_state_metrics_csv);
    cfg.backfit_pairwise_metrics_csv = char(cfg.backfit_pairwise_metrics_csv);
    cfg.backfit_record_summary_csv = char(cfg.backfit_record_summary_csv);
    cfg.force_recompute_peaks = cfg.force_recompute || cfg.force_recompute_peaks;
    cfg.force_recompute_clusters = cfg.force_recompute || cfg.force_recompute_clusters;
    util = microstate_utilities();
    repo_cfg = util.load_config();
    hier_defaults = struct();
    if isfield(repo_cfg, 'hierarchical') && isstruct(repo_cfg.hierarchical)
        hier_defaults = repo_cfg.hierarchical;
    end
    cfg.canonical_prior_weight_global = fill_from_hier_defaults(cfg.canonical_prior_weight_global, hier_defaults, 'canonical_prior_weight_global', 0);
    cfg.prior_weight_group = fill_from_hier_defaults(cfg.prior_weight_group, hier_defaults, 'prior_weight_group', 0);
    cfg.prior_weight_condition = fill_from_hier_defaults(cfg.prior_weight_condition, hier_defaults, 'prior_weight_condition', 0);
    cfg.prior_weight_participant = fill_from_hier_defaults(cfg.prior_weight_participant, hier_defaults, 'prior_weight_participant', 0);
    cfg.prior_weight_file = fill_from_hier_defaults(cfg.prior_weight_file, hier_defaults, 'prior_weight_file', 0);
    cfg.use_template_prior_global = logical(cfg.use_template_prior_global);
    if ~strcmp(cfg.peak_interpolation_method, 'idw')
        error('peak_interpolation_method must currently be ''idw''.');
    end
    cfg.K_candidates = unique(round(cfg.K_candidates(:)'));
    cfg.K_candidates = cfg.K_candidates(cfg.K_candidates >= 2);
    if isempty(cfg.K_candidates)
        error('K_candidates must contain at least one integer >= 2.');
    end
    if isempty(cfg.meta_K_candidates)
        cfg.meta_K_candidates = cfg.K_candidates;
    else
        cfg.meta_K_candidates = unique(round(cfg.meta_K_candidates(:)'));
    end
    if isempty(cfg.pooled_K_candidates)
        cfg.pooled_K_candidates = cfg.K_candidates;
    else
        cfg.pooled_K_candidates = unique(round(cfg.pooled_K_candidates(:)'));
    end
    if isempty(cfg.json_dir)
        cfg.json_dir = fullfile(cfg.output_dir, 'single_json');
    end
    if isempty(cfg.plot_dir)
        cfg.plot_dir = fullfile(cfg.output_dir, 'single_plots');
    end

    rng(cfg.random_seed, 'twister');
    ensure_dir(cfg.output_dir);
    dirs = struct();
    dirs.single = fullfile(cfg.output_dir, 'single_analyze_outputs'); ensure_dir(dirs.single);
    dirs.cache = fullfile(cfg.output_dir, 'gfp_peak_cache'); ensure_dir(dirs.cache);
    dirs.file_solutions = fullfile(cfg.output_dir, 'per_file_solutions'); ensure_dir(dirs.file_solutions);
    dirs.meta = fullfile(cfg.output_dir, 'meta_clusters'); ensure_dir(dirs.meta);
    dirs.pooled = fullfile(cfg.output_dir, 'pooled_gfp_clusters'); ensure_dir(dirs.pooled);
    dirs.hierarchy = fullfile(cfg.output_dir, 'hierarchical_gfp_clusters'); ensure_dir(dirs.hierarchy);
    dirs.hierarchy_groups = fullfile(dirs.hierarchy, 'groups'); ensure_dir(dirs.hierarchy_groups);
    dirs.hierarchy_conditions = fullfile(dirs.hierarchy, 'conditions'); ensure_dir(dirs.hierarchy_conditions);
    dirs.hierarchy_participants = fullfile(dirs.hierarchy, 'participants'); ensure_dir(dirs.hierarchy_participants);
    dirs.hierarchy_participant_conditions = fullfile(dirs.hierarchy, 'participant_conditions'); ensure_dir(dirs.hierarchy_participant_conditions);
    dirs.meta_subsets = fullfile(dirs.meta, 'subsets'); ensure_dir(dirs.meta_subsets);
    dirs.pooled_subsets = fullfile(dirs.pooled, 'subsets'); ensure_dir(dirs.pooled_subsets);
    dirs.qc = fullfile(cfg.output_dir, 'qc'); ensure_dir(dirs.qc);
    if cfg.save_json
        ensure_dir(cfg.json_dir);
        ensure_dir(cfg.plot_dir);
    end
    cfg.parallel_pool_ready = maybe_start_parallel_pool(cfg);

    if cfg.verbose
        fprintf('\n========================================\n');
        fprintf('EEG Meta-Microstate dataset pipeline\n');
        fprintf('========================================\n');
        fprintf('CSV:          %s\n', cfg.manifest_csv);
        fprintf('Output:       %s\n', cfg.output_dir);
        fprintf('Method:       %s\n', cfg.method);
        fprintf('Criterion:    %s\n', cfg.criterion);
        fprintf('K candidates: %s\n', mat2str(cfg.K_candidates));
        fprintf('Hierarchy:    global -> group -> participant -> participant_condition -> file\n');
        fprintf('Peak interpolation: %s\n', ternary_string(cfg.interpolate_missing_peak_channels, 'enabled', 'disabled'));
        if cfg.use_parfor
            fprintf('Parallel K fits: %s\n', ternary_string(cfg.parallel_pool_ready, 'enabled', 'requested but unavailable'));
        end
        fprintf('========================================\n\n');
    end

    manifest = read_and_standardise_manifest(cfg.manifest_csv, cfg);
    writetable(manifest, fullfile(cfg.output_dir, 'normalised_input_manifest.csv'));
    writetable(manifest, fullfile(cfg.output_dir, 'manifest_resolved.csv'));

    if cfg.verbose
        fprintf('1. Inspecting channel sets for %d files...\n', height(manifest));
    end
    file_meta = inspect_channel_sets(manifest, cfg);
    [common_labels, common_index_by_file, common_chanlocs, common_pos] = define_common_channels(file_meta, cfg);
    channel_table = table((1:numel(common_labels))', common_labels(:), 'VariableNames', {'common_index','label'});
    writetable(channel_table, fullfile(cfg.output_dir, 'common_channels.csv'));

    if cfg.verbose
        fprintf('   Target scalp/channel set: %d channels\n', numel(common_labels));
    end

    output_rows = initialise_output_rows(height(manifest));
    file_fits = cell(height(manifest), 1);
    record_refs = cell(height(manifest), 1);
    analysis_cache = cell(height(manifest), 1);
    solution_files = cell(height(manifest), 1);

    for i = 1:height(manifest)
        row = manifest(i, :);
        row_key = make_row_key(i, row.participant{1}, row.condition{1}, row.group{1}, row.file_path{1});
        if cfg.verbose
            fprintf('\n[%d/%d] %s | %s | %s\n', i, height(manifest), row.participant{1}, row.condition{1}, row.file_path{1});
        end

        cache_file = fullfile(dirs.cache, [row_key '_gfp_peaks.mat']);
        solution_file = fullfile(dirs.file_solutions, [row_key '_cluster_solution.mat']);
        analysis_file = fullfile(dirs.single, [row_key '_analyze_single_result.mat']);
        solution_files{i} = solution_file;
        analysis_cache{i} = analysis_file;

        AnalysisResults = [];
        json_file = '';
        status = "ok";
        error_message = "";

        try
            if cfg.call_analyze_single
                try
                    if cfg.force_recompute || ~isfile(analysis_file)
                        if exist('analyze_single_eeg_file', 'file') ~= 2
                            warning('analyze_single_eeg_file is not on the MATLAB path. Continuing with local common-channel fit only.');
                        else
                            if cfg.verbose
                                fprintf('   Running analyze_single_eeg_file...\n');
                            end
                            analyze_args = {'method', cfg.method, ...
                                'criterion', cfg.criterion, ...
                                'K_candidates', cfg.K_candidates, ...
                                'save_json', cfg.save_json, ...
                                'json_dir', cfg.json_dir, ...
                                'plot_dir', cfg.plot_dir, ...
                                'align_template', cfg.align_to_template, ...
                                'template_file', cfg.template_file, ...
                                'use_scalp_channels', cfg.use_scalp_channels, ...
                                'apply_average_reference', cfg.apply_average_reference, ...
                                'filter_band', cfg.filter_band, ...
                                'reject_gfp_peak_outliers', cfg.reject_gfp_peak_outliers_individual, ...
                                'gfp_outlier_mad_multiplier', cfg.gfp_outlier_mad_multiplier_individual, ...
                                'verbose', false};
                            [AnalysisResults, json_file] = analyze_single_eeg_file(row.file_path{1}, analyze_args{:});
                            save(analysis_file, 'AnalysisResults', 'json_file', 'row', 'cfg', '-v7.3');
                        end
                    else
                        S = load(analysis_file, 'AnalysisResults', 'json_file');
                        if isfield(S, 'AnalysisResults'), AnalysisResults = S.AnalysisResults; end
                        if isfield(S, 'json_file'), json_file = S.json_file; end
                    end
                catch ME_analyze
                    if cfg.fail_on_analyze_single_error
                        rethrow(ME_analyze);
                    end
                    AnalysisResults = [];
                    json_file = '';
                    warning('analyze_single_eeg_file failed for %s; continuing with dataset pipeline local fit only.\n%s', ...
                        row.file_path{1}, ME_analyze.message);
                end
            end

            rec = [];
            need_peak_refresh = cfg.force_recompute_peaks || ~isfile(cache_file);
            if ~need_peak_refresh
                S = load(cache_file, 'rec');
                rec = S.rec;
                need_peak_refresh = peak_cache_needs_refresh(rec, common_labels, cfg);
            end
            if need_peak_refresh
                if cfg.verbose
                    fprintf('   Extracting and caching GFP peaks...\n');
                end
                rec = extract_file_gfp_maps(row.file_path{1}, common_index_by_file{i}, common_labels, common_chanlocs, common_pos, cfg);
                rec.cache_file = cache_file;
                save(cache_file, 'rec', 'row', 'cfg', '-v7.3');
            end
            record_refs{i} = rec;

            output_rows.participant{i} = row.participant{1};
            output_rows.condition{i} = row.condition{1};
            output_rows.group{i} = row.group{1};
            output_rows.file_path{i} = row.file_path{1};
            output_rows.cluster_solution_file{i} = solution_file;
            output_rows.gfp_peak_cache_file{i} = cache_file;
            output_rows.analyze_single_result_file{i} = analysis_file;
            output_rows.json_file{i} = json_file;
            output_rows.local_K(i) = NaN;
            output_rows.n_gfp_peaks(i) = rec.n_peaks_raw;
            output_rows.n_gfp_peaks_used(i) = size(rec.maps_norm, 1);
            output_rows.n_target_channels(i) = rec.n_target_channels;
            output_rows.n_observed_channels(i) = rec.n_observed_channels;
            output_rows.n_interpolated_channels(i) = rec.n_interpolated_channels;
            output_rows.interpolated_channel_fraction(i) = rec.interpolated_fraction;
            output_rows.status{i} = char(status);
            output_rows.error_message{i} = char(error_message);

            write_checkpoint_manifest(output_rows, fullfile(cfg.output_dir, 'cluster_solution_manifest_partial.csv'));

        catch ME
            status = "failed";
            error_message = string(ME.message);
            warning('File failed: %s\n%s', row.file_path{1}, ME.getReport());
            output_rows.participant{i} = row.participant{1};
            output_rows.condition{i} = row.condition{1};
            output_rows.group{i} = row.group{1};
            output_rows.file_path{i} = row.file_path{1};
            output_rows.cluster_solution_file{i} = solution_file;
            output_rows.gfp_peak_cache_file{i} = cache_file;
            output_rows.analyze_single_result_file{i} = analysis_file;
            output_rows.json_file{i} = json_file;
            output_rows.local_K(i) = NaN;
            output_rows.n_gfp_peaks(i) = NaN;
            output_rows.n_gfp_peaks_used(i) = NaN;
            output_rows.n_target_channels(i) = NaN;
            output_rows.n_observed_channels(i) = NaN;
            output_rows.n_interpolated_channels(i) = NaN;
            output_rows.interpolated_channel_fraction(i) = NaN;
            output_rows.status{i} = char(status);
            output_rows.error_message{i} = char(error_message);
        end
    end

    output_csv = fullfile(cfg.output_dir, 'cluster_solution_manifest.csv');
    writetable(output_rows, output_csv);

    if cfg.verbose
        fprintf('\n2. Building pooled GFP bank and fitting hierarchical microstates...\n');
    end
    [pooled_maps, pooled_gfp, pooled_rows, population_filter] = pool_cached_gfp_maps(record_refs, manifest, cfg);
    [template_prior_maps, template_prior_info] = maybe_load_template_prior(cfg, common_labels, common_chanlocs, common_pos);

    pooled_fit = fit_hierarchical_map_set(pooled_maps, pooled_gfp, cfg.pooled_K_candidates, cfg, 'global_gfp_peaks', ...
        'primary_prior_maps', template_prior_maps, ...
        'primary_prior_weight', cfg.use_template_prior_global * cfg.canonical_prior_weight_global);
    if cfg.align_to_template
        pooled_fit = attach_template_alignment(pooled_fit, cfg.template_file, common_labels, common_chanlocs, common_pos, cfg);
    end
    pooled_fit.template_prior_info = template_prior_info;
    pooled_rows.cluster_label = assign_by_abs_correlation(pooled_maps, pooled_fit.centers);
    writetable(pooled_rows, fullfile(dirs.pooled, 'pooled_gfp_peak_manifest.csv'));
    save(fullfile(dirs.pooled, 'pooled_gfp_microstate_solution.mat'), 'pooled_fit', 'pooled_rows', 'population_filter', 'common_labels', 'common_chanlocs', 'common_pos', 'cfg', '-v7.3');
    write_matrix_csv(fullfile(dirs.pooled, 'pooled_gfp_microstate_centers.csv'), pooled_fit.centers);
    writetable(pooled_fit.model_comparison, fullfile(dirs.pooled, 'pooled_gfp_model_comparison.csv'));
    plot_center_grid(pooled_fit.centers, common_chanlocs, fullfile(dirs.pooled, 'pooled_gfp_microstate_centers.png'), 'Global pooled GFP microstates');

    group_levels = meaningful_factor_levels(manifest.group, "all");
    condition_levels = meaningful_factor_levels(manifest.condition, "condition");
    has_groups = ~isempty(group_levels);
    has_conditions = ~isempty(condition_levels);

    hierarchy_summary = table();
    hierarchy_nodes = struct('global', [], 'groups', [], 'conditions', [], 'participant_level', [], ...
        'participant_conditions', [], 'files', []);

    global_node_dir = fullfile(dirs.hierarchy, 'global');
    ensure_dir(global_node_dir);
    save_named_fit_bundle(global_node_dir, pooled_fit, pooled_rows, struct('level', 'global'), common_labels, common_chanlocs, common_pos, cfg, 'global_gfp');
    hierarchy_summary = [hierarchy_summary; make_hierarchy_summary_row('global', 'all', '', '', pooled_fit, global_node_dir, size(pooled_maps, 1), height(manifest))]; %#ok<AGROW>

    group_nodes = repmat(empty_hierarchy_fit_record(), 0, 1);
    for g = 1:numel(group_levels)
        group_value = group_levels(g);
        mask = string(pooled_rows.group) == group_value;
        group_dir = fullfile(dirs.hierarchy_groups, sanitize_key_piece(group_value));
        ensure_dir(group_dir);
        Fit = fit_hierarchical_map_set(pooled_maps(mask, :), pooled_gfp(mask), cfg.pooled_K_candidates, cfg, ...
            sprintf('group_%s', sanitize_key_piece(group_value)), ...
            'primary_prior_maps', pooled_fit.centers, ...
            'primary_prior_weight', cfg.prior_weight_group);
        if cfg.align_to_template
            Fit = attach_template_alignment(Fit, cfg.template_file, common_labels, common_chanlocs, common_pos, cfg);
        end
        rows_subset = pooled_rows(mask, :);
        rows_subset.cluster_label = assign_by_abs_correlation(pooled_maps(mask, :), Fit.centers);
        save_named_fit_bundle(group_dir, Fit, rows_subset, struct('level', 'group', 'group', char(group_value)), common_labels, common_chanlocs, common_pos, cfg, 'group_gfp');
        group_nodes(end+1, 1) = build_hierarchy_fit_record('group', '', char(group_value), '', Fit, group_dir, find(mask)); %#ok<AGROW>
        hierarchy_summary = [hierarchy_summary; make_hierarchy_summary_row('group', 'group', char(group_value), '', Fit, group_dir, sum(mask), numel(unique(pooled_rows.file_index(mask))))]; %#ok<AGROW>
    end

    condition_nodes = repmat(empty_hierarchy_fit_record(), 0, 1);
    for c = 1:numel(condition_levels)
        condition_value = condition_levels(c);
        mask = string(pooled_rows.condition) == condition_value;
        condition_dir = fullfile(dirs.hierarchy_conditions, sanitize_key_piece(condition_value));
        ensure_dir(condition_dir);
        Fit = fit_hierarchical_map_set(pooled_maps(mask, :), pooled_gfp(mask), cfg.pooled_K_candidates, cfg, ...
            sprintf('condition_%s', sanitize_key_piece(condition_value)), ...
            'primary_prior_maps', pooled_fit.centers, ...
            'primary_prior_weight', cfg.prior_weight_condition);
        if cfg.align_to_template
            Fit = attach_template_alignment(Fit, cfg.template_file, common_labels, common_chanlocs, common_pos, cfg);
        end
        rows_subset = pooled_rows(mask, :);
        rows_subset.cluster_label = assign_by_abs_correlation(pooled_maps(mask, :), Fit.centers);
        save_named_fit_bundle(condition_dir, Fit, rows_subset, struct('level', 'condition', 'condition', char(condition_value)), common_labels, common_chanlocs, common_pos, cfg, 'condition_gfp');
        condition_nodes(end+1, 1) = build_hierarchy_fit_record('condition', '', '', char(condition_value), Fit, condition_dir, find(mask)); %#ok<AGROW>
        hierarchy_summary = [hierarchy_summary; make_hierarchy_summary_row('condition', 'condition', '', char(condition_value), Fit, condition_dir, sum(mask), numel(unique(pooled_rows.file_index(mask))))]; %#ok<AGROW>
    end

    participant_values = unique(strtrim(string(manifest.participant)));
    participant_values = participant_values(strlength(participant_values) > 0 & ~ismissing(participant_values));
    participant_nodes = repmat(empty_hierarchy_fit_record(), 0, 1);
    for p_idx = 1:numel(participant_values)
        participant_value = participant_values(p_idx);
        row_mask = string(manifest.participant) == participant_value;
        peak_mask = string(pooled_rows.participant) == participant_value;
        if ~any(peak_mask)
            continue;
        end
        participant_group = char(string(manifest.group(find(row_mask, 1, 'first'))));
        [primary_maps, primary_name] = select_primary_parent_maps(participant_group, '', group_nodes, condition_nodes, pooled_fit.centers);
        participant_dir = fullfile(dirs.hierarchy_participants, sanitize_key_piece(participant_value));
        ensure_dir(participant_dir);
        Fit = fit_hierarchical_map_set(pooled_maps(peak_mask, :), pooled_gfp(peak_mask), cfg.pooled_K_candidates, cfg, ...
            sprintf('participant_%s', sanitize_key_piece(participant_value)), ...
            'primary_prior_maps', primary_maps, ...
            'primary_prior_weight', cfg.prior_weight_participant);
        if cfg.align_to_template
            Fit = attach_template_alignment(Fit, cfg.template_file, common_labels, common_chanlocs, common_pos, cfg);
        end
        Fit.parent_node = primary_name;
        rows_subset = pooled_rows(peak_mask, :);
        rows_subset.cluster_label = assign_by_abs_correlation(pooled_maps(peak_mask, :), Fit.centers);
        save_named_fit_bundle(participant_dir, Fit, rows_subset, struct('level', 'participant', 'participant', char(participant_value), 'group', participant_group), common_labels, common_chanlocs, common_pos, cfg, 'participant_gfp');
        participant_nodes(end+1, 1) = build_hierarchy_fit_record('participant', char(participant_value), participant_group, '', Fit, participant_dir, find(peak_mask)); %#ok<AGROW>
        hierarchy_summary = [hierarchy_summary; make_hierarchy_summary_row('participant', 'participant', participant_group, '', Fit, participant_dir, sum(peak_mask), sum(row_mask))]; %#ok<AGROW>
    end

    participant_condition_nodes = repmat(empty_hierarchy_fit_record(), 0, 1);
    if has_conditions
        [participant_condition_specs, participant_condition_keys] = build_participant_condition_specs(manifest);
        for pc = 1:numel(participant_condition_specs)
            spec = participant_condition_specs(pc);
            peak_mask = string(pooled_rows.participant) == spec.participant & string(pooled_rows.condition) == spec.condition;
            if ~any(peak_mask)
                continue;
            end
            participant_fit = find_hierarchy_fit_record(participant_nodes, char(spec.participant), '', '');
            condition_fit = find_hierarchy_fit_record(condition_nodes, '', '', char(spec.condition));
            participant_prior_maps = pooled_fit.centers;
            participant_parent_key = 'global';
            if ~isempty(participant_fit)
                participant_prior_maps = participant_fit.fit.centers;
                participant_parent_key = participant_fit.key;
            end
            condition_prior_maps = pooled_fit.centers;
            condition_parent_key = 'global';
            if ~isempty(condition_fit)
                condition_prior_maps = condition_fit.fit.centers;
                condition_parent_key = condition_fit.key;
            end
            participant_condition_dir = fullfile(dirs.hierarchy_participant_conditions, participant_condition_keys{pc});
            ensure_dir(participant_condition_dir);
            Fit = fit_hierarchical_map_set(pooled_maps(peak_mask, :), pooled_gfp(peak_mask), cfg.K_candidates, cfg, ...
                sprintf('participant_condition_%s', participant_condition_keys{pc}), ...
                'primary_prior_maps', participant_prior_maps, ...
                'primary_prior_weight', cfg.prior_weight_participant, ...
                'secondary_prior_maps', condition_prior_maps, ...
                'secondary_prior_weight', 0.5 * cfg.prior_weight_condition);
            if cfg.align_to_template
                Fit = attach_template_alignment(Fit, cfg.template_file, common_labels, common_chanlocs, common_pos, cfg);
            end
            Fit.parent_node = participant_parent_key;
            Fit.secondary_parent_node = condition_parent_key;
            rows_subset = pooled_rows(peak_mask, :);
            rows_subset.cluster_label = assign_by_abs_correlation(pooled_maps(peak_mask, :), Fit.centers);
            save_named_fit_bundle(participant_condition_dir, Fit, rows_subset, spec, common_labels, common_chanlocs, common_pos, cfg, 'participant_condition_gfp');
            participant_condition_nodes(end+1, 1) = build_hierarchy_fit_record('participant_condition', char(spec.participant), char(spec.group), char(spec.condition), Fit, participant_condition_dir, find(peak_mask)); %#ok<AGROW>
            hierarchy_summary = [hierarchy_summary; make_hierarchy_summary_row('participant_condition', 'participant_condition', char(spec.group), char(spec.condition), Fit, participant_condition_dir, sum(peak_mask), spec.n_rows)]; %#ok<AGROW>
        end
    end

    file_nodes = repmat(empty_hierarchy_fit_record(), 0, 1);
    for i = 1:height(manifest)
        if ~strcmpi(char(string(output_rows.status{i})), 'ok')
            continue;
        end
        peak_mask = pooled_rows.file_index == i;
        if ~any(peak_mask)
            output_rows.status{i} = 'failed';
            output_rows.error_message{i} = 'No pooled GFP peaks were retained for this row.';
            continue;
        end
        row = manifest(i, :);
        participant_value = char(string(row.participant{1}));
        group_value = char(string(row.group{1}));
        condition_value = char(string(row.condition{1}));
        if has_conditions
            parent_fit = find_hierarchy_fit_record(participant_condition_nodes, participant_value, '', condition_value);
        else
            parent_fit = find_hierarchy_fit_record(participant_nodes, participant_value, '', '');
        end
        if isempty(parent_fit)
            parent_fit = find_hierarchy_fit_record(participant_nodes, participant_value, '', '');
        end
        parent_prior_maps = pooled_fit.centers;
        parent_prior_key = 'global';
        if ~isempty(parent_fit)
            parent_prior_maps = parent_fit.fit.centers;
            parent_prior_key = parent_fit.key;
        end
        secondary_maps = [];
        if has_conditions
            condition_fit = find_hierarchy_fit_record(condition_nodes, '', '', condition_value);
            if ~isempty(condition_fit)
                secondary_maps = condition_fit.fit.centers;
            end
        end
        try
            Fit = fit_hierarchical_map_set(pooled_maps(peak_mask, :), pooled_gfp(peak_mask), cfg.K_candidates, cfg, ...
                sprintf('file_%03d', i), ...
                'primary_prior_maps', parent_prior_maps, ...
                'primary_prior_weight', cfg.prior_weight_file, ...
                'secondary_prior_maps', secondary_maps, ...
                'secondary_prior_weight', 0.25 * cfg.prior_weight_condition);
            if cfg.align_to_template
                Fit = attach_template_alignment(Fit, cfg.template_file, common_labels, common_chanlocs, common_pos, cfg);
            end
            Fit.parent_node = parent_prior_key;
            Fit.preprocessing = struct('apply_average_reference', cfg.apply_average_reference, 'filter_band', cfg.filter_band);
            Fit.maps_nc = rec.maps_norm;
            Fit.idx_peaks = rec.peak_sample(:);
            if isfield(Fit, 'labels') && ~isempty(Fit.labels)
                Fit.backfit_peak_labels = double(Fit.labels(1:min(numel(Fit.labels), size(rec.maps_norm, 1))));
            else
                Fit.backfit_peak_labels = [];
            end
            Fit.backfit_support = struct( ...
                'common_labels', {common_labels}, ...
                'common_chanlocs', common_chanlocs, ...
                'common_pos', common_pos, ...
                'channel_remap_spec', common_index_by_file{i}, ...
                'peak_sample', rec.peak_sample(:), ...
                'scalp_channel_labels', {file_meta(i).labels}, ...
                'scalp_chanlocs', file_meta(i).chanlocs, ...
                'scalp_pos', file_meta(i).pos, ...
                'source_file', row.file_path{1});
            rows_subset = pooled_rows(peak_mask, :);
            rows_subset.cluster_label = assign_by_abs_correlation(pooled_maps(peak_mask, :), Fit.centers);
            save_named_fit_bundle(fileparts(solution_files{i}), Fit, rows_subset, struct('level', 'file', 'row_index', i), common_labels, common_chanlocs, common_pos, cfg, sprintf('file_%03d_hierarchy', i));
            FileFit = Fit; %#ok<NASGU>
            save(solution_files{i}, 'FileFit', 'rows_subset', 'row', 'cfg', '-v7.3');
            file_fits{i} = Fit;
            file_nodes(end+1, 1) = build_hierarchy_fit_record('file', participant_value, group_value, condition_value, Fit, solution_files{i}, find(peak_mask)); %#ok<AGROW>
            output_rows.local_K(i) = Fit.K_estimated;
            output_rows.status{i} = 'ok';
            output_rows.error_message{i} = '';
            hierarchy_summary = [hierarchy_summary; make_hierarchy_summary_row('file', 'file', group_value, condition_value, Fit, solution_files{i}, sum(peak_mask), 1)]; %#ok<AGROW>
        catch ME_file
            output_rows.status{i} = 'failed';
            output_rows.error_message{i} = ME_file.message;
            warning('Hierarchical file fit failed for row %d (%s): %s', i, row.file_path{1}, ME_file.message);
        end
    end
    writetable(output_rows, output_csv);
    writetable(hierarchy_summary, fullfile(cfg.output_dir, 'hierarchical_fit_summary.csv'));

    if cfg.verbose
        fprintf('3. Building participant-driven global metamicrostates...\n');
    end
    [participant_bank, participant_bank_weights, participant_bank_rows] = hierarchy_fit_records_to_bank(participant_nodes);
    if isempty(participant_bank)
        error('No participant-level solutions were available for global metamicrostate fitting.');
    end
    meta_fit = fit_hierarchical_map_set(participant_bank, participant_bank_weights, cfg.meta_K_candidates, cfg, 'participant_metamicrostates', ...
        'primary_prior_maps', pooled_fit.centers, ...
        'primary_prior_weight', cfg.prior_weight_participant);
    if cfg.align_to_template
        meta_fit = attach_template_alignment(meta_fit, cfg.template_file, common_labels, common_chanlocs, common_pos, cfg);
    end
    meta_assign = assign_by_abs_correlation(participant_bank, meta_fit.centers);
    participant_bank_rows.meta_state = meta_assign(:);
    writetable(participant_bank_rows, fullfile(dirs.meta, 'meta_template_assignments.csv'));
    save(fullfile(dirs.meta, 'meta_microstate_solution.mat'), 'meta_fit', 'participant_bank', 'participant_bank_rows', 'common_labels', 'common_chanlocs', 'common_pos', 'cfg', '-v7.3');
    write_matrix_csv(fullfile(dirs.meta, 'meta_microstate_centers.csv'), meta_fit.centers);
    writetable(meta_fit.model_comparison, fullfile(dirs.meta, 'meta_model_comparison.csv'));
    plot_center_grid(meta_fit.centers, common_chanlocs, fullfile(dirs.meta, 'meta_microstate_centers.png'), 'Participant-derived meta-microstates');

    meta_subset_summary = table();
    if has_conditions && ~isempty(participant_condition_nodes)
        [pc_bank, pc_bank_weights, pc_bank_rows] = hierarchy_fit_records_to_bank(participant_condition_nodes);
        meta_subset_summary = run_conditioned_meta_fits(pc_bank_rows, pc_bank, pc_bank_weights, condition_levels, condition_nodes, ...
            cfg.meta_K_candidates, cfg, dirs.meta_subsets, common_labels, common_chanlocs, common_pos);
        writetable(meta_subset_summary, fullfile(dirs.meta, 'meta_subset_comparison.csv'));
    else
        writetable(meta_subset_summary, fullfile(dirs.meta, 'meta_subset_comparison.csv'));
    end

    pooled_subset_summary = hierarchy_summary;
    HResults = build_hresults_from_hierarchy(manifest, pooled_fit, group_nodes, condition_nodes, participant_nodes, participant_condition_nodes, file_nodes, meta_fit, common_labels, common_chanlocs, common_pos, cfg);
    backfit_state_metrics = table();
    backfit_state_metrics_csv = '';
    backfit_pairwise_metrics = table();
    backfit_pairwise_metrics_csv = '';
    backfit_record_summary = table();
    backfit_record_summary_csv = '';
    if cfg.export_backfit_state_metrics
        [backfit_state_metrics, backfit_pairwise_metrics, backfit_record_summary] = compile_backfit_metric_tables(output_rows, cfg);
        backfit_state_metrics_csv = resolve_backfit_metrics_csv_path(cfg);
        if ~isempty(backfit_state_metrics)
            writetable(backfit_state_metrics, backfit_state_metrics_csv);
        end
        backfit_pairwise_metrics_csv = resolve_backfit_pairwise_metrics_csv_path(cfg);
        if ~isempty(backfit_pairwise_metrics)
            writetable(backfit_pairwise_metrics, backfit_pairwise_metrics_csv);
        end
        backfit_record_summary_csv = resolve_backfit_record_summary_csv_path(cfg);
        if ~isempty(backfit_record_summary)
            writetable(backfit_record_summary, backfit_record_summary_csv);
        end
    end

    MResults = struct();
    MResults.config = cfg;
    MResults.output_csv = output_csv;
    MResults.manifest = manifest;
    MResults.common_labels = common_labels;
    MResults.meta_fit = meta_fit;
    MResults.pooled_fit = pooled_fit;
    MResults.meta_subset_summary = meta_subset_summary;
    MResults.pooled_subset_summary = pooled_subset_summary;
    MResults.file_fits = file_fits;
    MResults.population_filter = population_filter;
    MResults.hierarchy_summary = hierarchy_summary;
    MResults.participant_bank_rows = participant_bank_rows;
    MResults.runtime_s = toc(t0);
    MResults.HResults = HResults;
    MResults.backfit_state_metrics = backfit_state_metrics;
    MResults.backfit_state_metrics_csv = backfit_state_metrics_csv;
    MResults.backfit_pairwise_metrics = backfit_pairwise_metrics;
    MResults.backfit_pairwise_metrics_csv = backfit_pairwise_metrics_csv;
    MResults.backfit_record_summary = backfit_record_summary;
    MResults.backfit_record_summary_csv = backfit_record_summary_csv;
    save(fullfile(cfg.output_dir, 'meta_microstate_dataset_results.mat'), 'MResults', 'HResults', '-v7.3');
    save(fullfile(cfg.output_dir, 'hierarchical_microstate_results.mat'), 'HResults', 'MResults', '-v7.3');

    if cfg.verbose
        fprintf('\nDone.\n');
        fprintf('Output CSV: %s\n', output_csv);
        if ~isempty(backfit_state_metrics_csv)
            fprintf('Backfit state metrics CSV: %s\n', backfit_state_metrics_csv);
        end
        if ~isempty(backfit_pairwise_metrics_csv)
            fprintf('Backfit pairwise metrics CSV: %s\n', backfit_pairwise_metrics_csv);
        end
        if ~isempty(backfit_record_summary_csv)
            fprintf('Backfit record summary CSV: %s\n', backfit_record_summary_csv);
        end
        fprintf('Meta solution: %s\n', fullfile(dirs.meta, 'meta_microstate_solution.mat'));
        fprintf('Pooled GFP solution: %s\n', fullfile(dirs.pooled, 'pooled_gfp_microstate_solution.mat'));
        fprintf('Runtime: %.1f s\n', MResults.runtime_s);
    end
end

% -------------------------------------------------------------------------
% Manifest and filesystem helpers
% -------------------------------------------------------------------------
function manifest = read_and_standardise_manifest(manifest_csv, cfg)
    if ~isfile(manifest_csv)
        error('Manifest CSV not found: %s', manifest_csv);
    end
    opts = detectImportOptions(manifest_csv, 'FileType', 'text', 'Delimiter', ',', 'TextType', 'string');
    opts.VariableNamingRule = 'preserve';
    T = readtable(manifest_csv, opts);

    names = normalise_header_tokens(T.Properties.VariableNames);
    file_col = find_first_col(names, normalise_header_tokens(["file_path", "filepath", "file", "path", "filename"]));
    if isempty(file_col)
        error('CSV must contain a file_path column. Available columns: %s', strjoin(string(T.Properties.VariableNames), ', '));
    end
    p_col = find_first_col(names, normalise_header_tokens(["participant", "subject", "sub", "id", "participant_id", "subject_id"]));
    c_col = find_first_col(names, normalise_header_tokens(["condition", "state", "task", "eyes", "session"]));
    g_col = find_first_col(names, normalise_header_tokens(["group", "diagnosis", "cohort", "class"]));

    file_path = string(T{:, T.Properties.VariableNames{file_col}});
    csv_dir = fileparts(which_or_absolute(manifest_csv));
    for i = 1:numel(file_path)
        file_path(i) = string(resolve_manifest_path(char(file_path(i)), csv_dir));
    end

    if isempty(p_col)
        participant = strings(numel(file_path), 1);
    else
        participant = string(T{:, T.Properties.VariableNames{p_col}});
    end
    if isempty(c_col)
        condition = repmat("condition", numel(file_path), 1);
    else
        condition = string(T{:, T.Properties.VariableNames{c_col}});
    end
    if isempty(g_col)
        group = repmat("all", numel(file_path), 1);
    else
        group = string(T{:, T.Properties.VariableNames{g_col}});
    end

    bad_p = ismissing(participant) | strlength(strtrim(participant)) == 0;
    participant(bad_p) = arrayfun(@(i) "row" + string(i), find(bad_p), 'UniformOutput', true);
    if cfg.infer_participant_from_path
        unique_p = unique(participant);
        uninformative = numel(unique_p) == 1 && any(strcmpi(unique_p, ["eeg", "subject", "participant", "all", "unknown"]));
        if uninformative || numel(unique_p) < round(0.25 * numel(participant))
            for i = 1:numel(participant)
                participant(i) = string(infer_participant_id_from_path(char(file_path(i)), i));
            end
        end
    end

    condition(ismissing(condition) | strlength(strtrim(condition)) == 0) = "condition";
    group(ismissing(group) | strlength(strtrim(group)) == 0) = "all";

    manifest = table(cellstr(participant), cellstr(condition), cellstr(group), cellstr(file_path), ...
        'VariableNames', {'participant', 'condition', 'group', 'file_path'});
end

function value = fill_from_hier_defaults(value, defaults, field_name, fallback)
    if isnumeric(value) && isscalar(value) && ~isnan(value)
        return;
    end
    if isstruct(defaults) && isfield(defaults, field_name) && ~isempty(defaults.(field_name))
        value = double(defaults.(field_name));
    else
        value = double(fallback);
    end
end

function levels = meaningful_factor_levels(values, default_value)
    levels = unique(strtrim(string(values)));
    levels = levels(strlength(levels) > 0 & ~ismissing(levels));
    levels = levels(levels ~= string(default_value));
    levels = levels(:)';
end

function [prior_maps, info] = maybe_load_template_prior(cfg, common_labels, common_chanlocs, common_pos)
    prior_maps = [];
    info = struct('ok', false, 'message', '', 'n_maps', 0);
    if ~cfg.use_template_prior_global || isempty(cfg.template_file) || ~isfile(cfg.template_file)
        info.message = 'template_prior_disabled';
        return;
    end
    try
        [template_maps, ~, template_channel_labels, template_chanlocs] = load_template_maps(cfg.template_file);
        prior_maps = remap_template_to_common(template_maps, template_channel_labels, template_chanlocs, common_labels, common_pos);
        prior_maps = normalize_maps(prior_maps);
        info.ok = true;
        info.message = 'ok';
        info.n_maps = size(prior_maps, 1);
    catch ME
        prior_maps = [];
        info.ok = false;
        info.message = ME.message;
        warning('Global template prior could not be loaded: %s', ME.message);
    end
end

function Fit = fit_hierarchical_map_set(maps, gfp, K_candidates, cfg, fit_name, varargin)
    p = inputParser;
    addRequired(p, 'maps', @isnumeric);
    addRequired(p, 'gfp', @isnumeric);
    addRequired(p, 'K_candidates', @isnumeric);
    addRequired(p, 'cfg', @isstruct);
    addRequired(p, 'fit_name', @(x) ischar(x) || isstring(x));
    addParameter(p, 'primary_prior_maps', [], @isnumeric);
    addParameter(p, 'primary_prior_weight', 0, @(x) isnumeric(x) && isscalar(x) && x >= 0);
    addParameter(p, 'secondary_prior_maps', [], @isnumeric);
    addParameter(p, 'secondary_prior_weight', 0, @(x) isnumeric(x) && isscalar(x) && x >= 0);
    parse(p, maps, gfp, K_candidates, cfg, fit_name, varargin{:});

    X = double(p.Results.maps);
    w = double(p.Results.gfp(:));
    if isempty(w) || numel(w) ~= size(X, 1)
        w = ones(size(X, 1), 1);
    end

    prior_specs = {};
    [X_aug, w_aug, prior_specs] = append_prior_maps(X, w, p.Results.primary_prior_maps, p.Results.primary_prior_weight, 'primary', prior_specs);
    [X_aug, w_aug, prior_specs] = append_prior_maps(X_aug, w_aug, p.Results.secondary_prior_maps, p.Results.secondary_prior_weight, 'secondary', prior_specs);

    Fit = fit_microstate_map_set(X_aug, w_aug, K_candidates, cfg, char(p.Results.fit_name));
    Fit.fit_stage = 'hierarchical_empirical_bayes';
    Fit.observed_n_maps = size(X, 1);
    Fit.observed_weight_sum = sum(w.^2);
    Fit.augmented_n_maps = size(X_aug, 1);
    Fit.prior_specs = prior_specs;
end

function [X_out, w_out, prior_specs] = append_prior_maps(X_in, w_in, prior_maps, total_weight, label, prior_specs)
    X_out = X_in;
    w_out = w_in;
    if nargin < 6 || isempty(prior_specs)
        prior_specs = {};
    end
    prior_maps = double(prior_maps);
    total_weight = double(total_weight);
    if isempty(prior_maps) || total_weight <= 0
        return;
    end
    prior_maps = normalize_maps(prior_maps);
    per_map_weight = sqrt(total_weight / max(1, size(prior_maps, 1)));
    prior_weights = repmat(per_map_weight, size(prior_maps, 1), 1);
    X_out = [X_out; prior_maps]; %#ok<AGROW>
    w_out = [w_out; prior_weights]; %#ok<AGROW>
    prior_specs{end+1} = struct('label', char(label), 'n_maps', size(prior_maps, 1), 'total_weight', total_weight); %#ok<AGROW>
end

function save_named_fit_bundle(out_dir, Fit, rows_subset, spec, common_labels, common_chanlocs, common_pos, cfg, prefix)
    ensure_dir(out_dir);
    mat_file = fullfile(out_dir, [prefix '_solution.mat']);
    centers_file = fullfile(out_dir, [prefix '_centers.csv']);
    model_file = fullfile(out_dir, [prefix '_model_comparison.csv']);
    save(mat_file, 'Fit', 'rows_subset', 'spec', 'common_labels', 'common_chanlocs', 'common_pos', 'cfg', '-v7.3');
    write_matrix_csv(centers_file, Fit.centers);
    writetable(Fit.model_comparison, model_file);
    if istable(rows_subset) && ~isempty(rows_subset)
        writetable(rows_subset, fullfile(out_dir, [prefix '_manifest.csv']));
    end
    plot_center_grid(Fit.centers, common_chanlocs, fullfile(out_dir, [prefix '_centers.png']), prefix);
end

function row = make_hierarchy_summary_row(level, subset_kind, group_value, condition_value, Fit, fit_path, n_maps, n_units)
    row = table(string(level), string(subset_kind), string(group_value), string(condition_value), ...
        double(Fit.K_estimated), double(n_maps), double(n_units), string(fit_path), string('ok'), ...
        'VariableNames', {'level', 'subset_kind', 'group', 'condition', 'K_estimated', 'n_maps', 'n_units', 'fit_path', 'status'});
end

function rec = empty_hierarchy_fit_record()
    rec = struct('level', '', 'participant', '', 'group', '', 'condition', '', 'key', '', ...
        'fit', struct(), 'fit_path', '', 'peak_index', []);
end

function rec = build_hierarchy_fit_record(level, participant, group, condition, Fit, fit_path, peak_index)
    rec = empty_hierarchy_fit_record();
    rec.level = char(string(level));
    rec.participant = char(string(participant));
    rec.group = char(string(group));
    rec.condition = char(string(condition));
    rec.key = char(strjoin([string(rec.level), string(rec.participant), string(rec.group), string(rec.condition)], '|'));
    rec.fit = Fit;
    rec.fit_path = char(string(fit_path));
    rec.peak_index = peak_index(:);
end

function rec = find_hierarchy_fit_record(records, participant, group, condition)
    rec = [];
    if isempty(records)
        return;
    end
    mask = true(numel(records), 1);
    if ~isempty(participant)
        mask = mask & strcmp({records.participant}', char(string(participant)));
    end
    if ~isempty(group)
        mask = mask & strcmp({records.group}', char(string(group)));
    end
    if ~isempty(condition)
        mask = mask & strcmp({records.condition}', char(string(condition)));
    end
    idx = find(mask, 1, 'first');
    if ~isempty(idx)
        rec = records(idx);
    end
end

function [primary_maps, primary_name] = select_primary_parent_maps(group_value, ~, group_nodes, ~, global_maps)
    primary_maps = global_maps;
    primary_name = 'global';
    if isempty(group_value) || isempty(group_nodes)
        return;
    end
    group_fit = find_hierarchy_fit_record(group_nodes, '', char(string(group_value)), '');
    if ~isempty(group_fit)
        primary_maps = group_fit.fit.centers;
        primary_name = group_fit.key;
    end
end

function [specs, keys] = build_participant_condition_specs(manifest)
    keys = {};
    specs = repmat(struct('participant', "", 'condition', "", 'group', "", 'n_rows', 0, 'level', 'participant_condition'), 0, 1);
    combo = strcat(string(manifest.participant), "__", string(manifest.condition));
    [uniq_combo, ~, idx] = unique(combo, 'stable');
    for i = 1:numel(uniq_combo)
        mask = idx == i;
        spec = struct();
        spec.participant = string(manifest.participant(find(mask, 1, 'first')));
        spec.condition = string(manifest.condition(find(mask, 1, 'first')));
        spec.group = string(manifest.group(find(mask, 1, 'first')));
        spec.n_rows = sum(mask);
        spec.level = 'participant_condition';
        specs(end+1, 1) = spec; %#ok<AGROW>
        keys{end+1, 1} = char(sanitize_key_piece(sprintf('%s__%s', spec.participant, spec.condition))); %#ok<AGROW>
    end
end

function [bank_maps, bank_weights, bank_rows] = hierarchy_fit_records_to_bank(records)
    bank_maps = [];
    bank_weights = [];
    bank_rows = table();
    for i = 1:numel(records)
        Fit = records(i).fit;
        if ~isstruct(Fit) || ~isfield(Fit, 'centers') || isempty(Fit.centers)
            continue;
        end
        K = size(Fit.centers, 1);
        node_weight = max(1, get_field_or_default(Fit, 'observed_n_maps', get_field_or_default(Fit, 'n_maps', K)));
        bank_maps = [bank_maps; normalize_maps(Fit.centers)]; %#ok<AGROW>
        bank_weights = [bank_weights; repmat(sqrt(node_weight / max(1, K)), K, 1)]; %#ok<AGROW>
        rows = table();
        rows.record_index = repmat(i, K, 1);
        rows.level = repmat(string(records(i).level), K, 1);
        rows.participant = repmat(string(records(i).participant), K, 1);
        rows.group = repmat(string(records(i).group), K, 1);
        rows.condition = repmat(string(records(i).condition), K, 1);
        rows.local_state = (1:K)';
        rows.fit_weight_proxy = repmat(sqrt(node_weight / max(1, K)), K, 1);
        rows.fit_path = repmat(string(records(i).fit_path), K, 1);
        bank_rows = [bank_rows; rows]; %#ok<AGROW>
    end
end

function summary = run_conditioned_meta_fits(bank_rows, bank_maps, bank_weights, condition_levels, condition_nodes, K_candidates, cfg, out_root, common_labels, common_chanlocs, common_pos)
    summary = table();
    for c = 1:numel(condition_levels)
        condition_value = condition_levels(c);
        mask = string(bank_rows.condition) == condition_value;
        if ~any(mask)
            continue;
        end
        out_dir = fullfile(out_root, ['condition_' sanitize_key_piece(condition_value)]);
        ensure_dir(out_dir);
        try
            condition_fit = find_hierarchy_fit_record(condition_nodes, '', '', char(condition_value));
            prior_maps = [];
            if ~isempty(condition_fit)
                prior_maps = condition_fit.fit.centers;
            end
            Fit = fit_hierarchical_map_set(bank_maps(mask, :), bank_weights(mask), K_candidates, cfg, ...
                sprintf('condition_meta_%s', sanitize_key_piece(condition_value)), ...
                'primary_prior_maps', prior_maps, ...
                'primary_prior_weight', cfg.prior_weight_condition);
            if cfg.align_to_template
                Fit = attach_template_alignment(Fit, cfg.template_file, common_labels, common_chanlocs, common_pos, cfg);
            end
            rows_subset = bank_rows(mask, :);
            rows_subset.meta_state = assign_by_abs_correlation(bank_maps(mask, :), Fit.centers);
            save_named_fit_bundle(out_dir, Fit, rows_subset, struct('level', 'condition_meta', 'condition', char(condition_value)), common_labels, common_chanlocs, common_pos, cfg, 'condition_meta');
            row = table(string(condition_value), double(Fit.K_estimated), double(sum(mask)), string(fullfile(out_dir, 'condition_meta_solution.mat')), string('ok'), ...
                'VariableNames', {'condition', 'K_estimated', 'n_rows', 'fit_file', 'status'});
        catch ME
            row = table(string(condition_value), NaN, double(sum(mask)), string(fullfile(out_dir, 'condition_meta_solution.mat')), string(ME.message), ...
                'VariableNames', {'condition', 'K_estimated', 'n_rows', 'fit_file', 'status'});
            warning('Conditioned meta fit failed for %s: %s', condition_value, ME.message);
        end
        summary = [summary; row]; %#ok<AGROW>
    end
end

function HResults = build_hresults_from_hierarchy(manifest, pooled_fit, group_nodes, condition_nodes, participant_nodes, participant_condition_nodes, file_nodes, meta_fit, common_labels, common_chanlocs, common_pos, cfg)
    HResults = struct();
    HResults.source = 'metamicrostate_dataset_pipeline';
    HResults.created = datestr(now, 30);
    HResults.cfg = cfg;
    HResults.manifest = manifest;
    HResults.common_channel_labels = common_labels;
    HResults.common_labels = common_labels;
    HResults.common_chanlocs = common_chanlocs;
    HResults.common_pos = common_pos;
    HResults.selected_K = pooled_fit.K_estimated;
    HResults.global = build_pipeline_node('', 'all', '', 'global', 'global', pooled_fit.centers, pooled_fit.n_maps, '', pooled_fit);
    HResults.groups = hierarchy_records_to_nodes(group_nodes);
    HResults.conditions = hierarchy_records_to_nodes(condition_nodes);
    HResults.participant_level = hierarchy_records_to_nodes(participant_nodes);
    if isempty(participant_condition_nodes)
        HResults.participant_conditions = HResults.participant_level;
        HResults.participants = HResults.participant_level;
    else
        HResults.participant_conditions = hierarchy_records_to_nodes(participant_condition_nodes);
        HResults.participants = HResults.participant_conditions;
    end
    HResults.files = hierarchy_records_to_nodes(file_nodes);
    HResults.meta_fit = meta_fit;
    HResults.group_conditions = repmat(build_pipeline_node('', '', '', 'group_condition', '', zeros(1, size(pooled_fit.centers, 2)), 0, '', struct()), 0, 1);
end

function nodes = hierarchy_records_to_nodes(records)
    nodes = repmat(build_pipeline_node('', '', '', 'empty', '', zeros(1, 1), 0, '', struct()), 0, 1);
    for i = 1:numel(records)
        Fit = records(i).fit;
        if ~isstruct(Fit) || ~isfield(Fit, 'centers') || isempty(Fit.centers)
            continue;
        end
        n_maps = get_field_or_default(Fit, 'observed_n_maps', get_field_or_default(Fit, 'n_maps', size(Fit.centers, 1)));
        nodes(end+1, 1) = build_pipeline_node(records(i).participant, records(i).group, records(i).condition, ...
            records(i).level, records(i).key, Fit.centers, n_maps, records(i).fit_path, Fit); %#ok<AGROW>
    end
end

function subset_summary = run_subset_cluster_fits(rows_table, maps, K_candidates, cfg, out_root, fit_kind, label_column, common_labels, common_chanlocs, common_pos)
    specs = build_subset_specs(rows_table);
    subset_summary = table();
    for i = 1:numel(specs)
        spec = specs(i);
        subset_rows = rows_table(spec.mask, :);
        subset_maps = maps(spec.mask, :);
        subset_gfp = get_subset_gfp(rows_table, spec.mask);

        out_dir = fullfile(out_root, spec.subset_key);
        ensure_dir(out_dir);

        row = make_subset_summary_row(spec, subset_rows, fit_kind, out_dir);
        try
            fit_name = sprintf('%s_%s', fit_kind, spec.subset_key);
            fit = fit_microstate_map_set(subset_maps, subset_gfp, K_candidates, cfg, fit_name);
            if cfg.align_to_template
                fit = attach_template_alignment(fit, cfg.template_file, common_labels, common_chanlocs, common_pos, cfg);
            end

            assigned = assign_by_abs_correlation(subset_maps, fit.centers);
            subset_rows.(label_column) = assigned(:);

            if strcmp(fit_kind, 'meta')
                writetable(subset_rows, fullfile(out_dir, 'meta_template_assignments.csv'));
                write_matrix_csv(fullfile(out_dir, 'meta_microstate_centers.csv'), fit.centers);
                writetable(fit.model_comparison, fullfile(out_dir, 'meta_model_comparison.csv'));
                save(fullfile(out_dir, 'meta_microstate_solution.mat'), 'fit', 'subset_rows', 'spec', 'common_labels', 'common_chanlocs', 'common_pos', 'cfg', '-v7.3');
                plot_center_grid(fit.centers, common_chanlocs, fullfile(out_dir, 'meta_microstate_centers.png'), sprintf('Meta-microstates %s', spec.label));
            else
                writetable(subset_rows, fullfile(out_dir, 'pooled_gfp_peak_manifest.csv'));
                write_matrix_csv(fullfile(out_dir, 'pooled_gfp_microstate_centers.csv'), fit.centers);
                writetable(fit.model_comparison, fullfile(out_dir, 'pooled_gfp_model_comparison.csv'));
                save(fullfile(out_dir, 'pooled_gfp_microstate_solution.mat'), 'fit', 'subset_rows', 'spec', 'common_labels', 'common_chanlocs', 'common_pos', 'cfg', '-v7.3');
                plot_center_grid(fit.centers, common_chanlocs, fullfile(out_dir, 'pooled_gfp_microstate_centers.png'), sprintf('Pooled GFP-peak microstates %s', spec.label));
            end

            row.status = "ok";
            row.error_message = "";
            row.K_estimated = fit.K_estimated;
            row.n_maps = fit.n_maps;
            row.n_rows_used = height(subset_rows);
            row.fit_file = string(fullfile(out_dir, subset_fit_filename(fit_kind)));
            row.centers_file = string(fullfile(out_dir, subset_centers_filename(fit_kind)));
            row.comparison_file = string(fullfile(out_dir, subset_model_comparison_filename(fit_kind)));
        catch ME
            row.status = "failed";
            row.error_message = string(ME.message);
            row.K_estimated = NaN;
            row.n_maps = NaN;
            row.n_rows_used = height(subset_rows);
            row.fit_file = string(fullfile(out_dir, subset_fit_filename(fit_kind)));
            row.centers_file = string(fullfile(out_dir, subset_centers_filename(fit_kind)));
            row.comparison_file = string(fullfile(out_dir, subset_model_comparison_filename(fit_kind)));
            warning('Subset fit failed for %s: %s', spec.label, ME.message);
        end
        subset_summary = [subset_summary; struct2table(row)]; %#ok<AGROW>
    end
end

function specs = build_subset_specs(T)
    group_vals = unique_nonempty_strings(T.group);
    condition_vals = unique_nonempty_strings(T.condition);
    specs = struct('subset_key', {}, 'subset_kind', {}, 'group', {}, 'condition', {}, 'label', {}, 'mask', {}, 'n_rows', {}, 'n_files', {});

    specs(end+1) = make_subset_spec(T, "all", "all", "all", true(height(T), 1), "all");

    if numel(group_vals) > 1
        for i = 1:numel(group_vals)
            g = group_vals(i);
            mask = string(T.group) == g;
            specs(end+1) = make_subset_spec(T, "group", g, "", mask, "group"); %#ok<AGROW>
        end
    end

    if numel(condition_vals) > 1
        for i = 1:numel(condition_vals)
            c = condition_vals(i);
            mask = string(T.condition) == c;
            specs(end+1) = make_subset_spec(T, "condition", "", c, mask, "condition"); %#ok<AGROW>
        end
    end

    if numel(group_vals) > 1 && numel(condition_vals) > 1
        for i = 1:numel(group_vals)
            for j = 1:numel(condition_vals)
                g = group_vals(i);
                c = condition_vals(j);
                mask = string(T.group) == g & string(T.condition) == c;
                if any(mask)
                    specs(end+1) = make_subset_spec(T, "interaction", g, c, mask, "interaction"); %#ok<AGROW>
                end
            end
        end
    end
end

function spec = make_subset_spec(T, subset_kind, group_value, condition_value, mask, subset_kind_label)
    if subset_kind == "all"
        key = 'all';
        label = 'all files';
    elseif subset_kind == "group"
        key = ['group_' sanitize_key_piece(group_value)];
        label = sprintf('group=%s', string(group_value));
    elseif subset_kind == "condition"
        key = ['condition_' sanitize_key_piece(condition_value)];
        label = sprintf('condition=%s', string(condition_value));
    else
        key = ['interaction_' sanitize_key_piece(group_value) '__' sanitize_key_piece(condition_value)];
        label = sprintf('group=%s | condition=%s', string(group_value), string(condition_value));
    end

    spec = struct();
    spec.subset_key = char(key);
    spec.subset_kind = subset_kind_label;
    spec.group = string(group_value);
    spec.condition = string(condition_value);
    spec.label = label;
    spec.mask = mask(:);
    spec.n_rows = sum(mask);
    if any(strcmpi(T.Properties.VariableNames, 'file_index'))
        spec.n_files = numel(unique(T.file_index(mask)));
    else
        spec.n_files = NaN;
    end
end

function vals = unique_nonempty_strings(x)
    vals = unique(strtrim(string(x)));
    vals = vals(strlength(vals) > 0 & ~ismissing(vals));
    vals = vals(:)';
end

function gfp = get_subset_gfp(rows_table, mask)
    if any(strcmpi(rows_table.Properties.VariableNames, 'gfp_effective'))
        gfp = double(rows_table.gfp_effective(mask));
    elseif any(strcmpi(rows_table.Properties.VariableNames, 'fit_weight_proxy'))
        gfp = double(rows_table.fit_weight_proxy(mask));
    elseif any(strcmpi(rows_table.Properties.VariableNames, 'gfp'))
        gfp = double(rows_table.gfp(mask));
    else
        gfp = ones(sum(mask), 1);
    end
end

function row = make_subset_summary_row(spec, subset_rows, fit_kind, out_dir)
    row = struct();
    row.fit_kind = string(fit_kind);
    row.subset_key = string(spec.subset_key);
    row.subset_kind = string(spec.subset_kind);
    row.group = string(spec.group);
    row.condition = string(spec.condition);
    row.label = string(spec.label);
    row.n_rows = spec.n_rows;
    row.n_files = spec.n_files;
    row.n_rows_used = height(subset_rows);
    row.K_estimated = NaN;
    row.n_maps = NaN;
    row.status = "pending";
    row.error_message = "";
    row.fit_file = string(fullfile(out_dir, subset_fit_filename(fit_kind)));
    row.centers_file = string(fullfile(out_dir, subset_centers_filename(fit_kind)));
    row.comparison_file = string(fullfile(out_dir, subset_model_comparison_filename(fit_kind)));
end

function fn = subset_fit_filename(fit_kind)
    if strcmp(fit_kind, 'meta')
        fn = 'meta_microstate_solution.mat';
    else
        fn = 'pooled_gfp_microstate_solution.mat';
    end
end

function fn = subset_centers_filename(fit_kind)
    if strcmp(fit_kind, 'meta')
        fn = 'meta_microstate_centers.csv';
    else
        fn = 'pooled_gfp_microstate_centers.csv';
    end
end

function fn = subset_model_comparison_filename(fit_kind)
    if strcmp(fit_kind, 'meta')
        fn = 'meta_model_comparison.csv';
    else
        fn = 'pooled_gfp_model_comparison.csv';
    end
end

function piece = sanitize_key_piece(value)
    piece = regexprep(lower(string(value)), '[^a-zA-Z0-9]+', '_');
    piece = regexprep(piece, '_+', '_');
    piece = regexprep(piece, '^_|_$', '');
    if strlength(piece) == 0
        piece = "subset";
    end
    piece = char(piece);
end

function HResults = build_pipeline_hresults(manifest, file_fits, meta_fit, template_bank, template_bank_rows, common_labels, common_chanlocs, common_pos, cfg)
    HResults = struct();
    HResults.source = 'metamicrostate_dataset_pipeline';
    HResults.created = datestr(now, 30);
    HResults.cfg = cfg;
    HResults.manifest = manifest;
    HResults.common_channel_labels = common_labels;
    HResults.common_labels = common_labels;
    HResults.common_chanlocs = common_chanlocs;
    HResults.common_pos = common_pos;
    HResults.meta_template_bank_rows = template_bank_rows;
    HResults.global = build_pipeline_node('global', 'global', '', '', '', meta_fit.centers, size(template_bank_rows, 1), '', meta_fit);

    file_nodes = repmat(build_pipeline_node('', '', '', '', '', meta_fit.centers, 0, '', meta_fit), numel(file_fits), 1);
    for i = 1:numel(file_fits)
        if isempty(file_fits{i}) || ~isstruct(file_fits{i}) || ~isfield(file_fits{i}, 'centers')
            centers = meta_fit.centers;
            fit = [];
            n_maps = NaN;
        else
            centers = aggregate_file_template_centers(i, template_bank, template_bank_rows, meta_fit);
            fit = file_fits{i};
            n_maps = get_field_or_default(fit, 'n_maps', size(centers, 1));
        end
        file_nodes(i) = build_pipeline_node(manifest.participant{i}, manifest.group{i}, manifest.condition{i}, ...
            'participant_condition', sprintf('%s__%s__%s', manifest.participant{i}, manifest.group{i}, manifest.condition{i}), ...
            centers, n_maps, manifest.file_path{i}, fit);
    end

    HResults.files = file_nodes;
    HResults.participants = file_nodes;
    HResults.participant_conditions = file_nodes;
    HResults.group_conditions = build_group_condition_nodes(file_nodes, manifest, meta_fit);
end

function centers = sign_align_and_normalise_centers(centers, ref)
    centers = normalize_maps(centers);
    ref = normalize_maps(ref);
    K = min(size(centers, 1), size(ref, 1));
    for k = 1:K
        if all(isfinite(centers(k, :))) && all(isfinite(ref(k, :))) && dot(centers(k, :), ref(k, :)) < 0
            centers(k, :) = -centers(k, :);
        end
    end
end

function node = build_pipeline_node(participant, group, condition, level, name, centers, n_maps, source_file, fit)
    node = struct();
    node.participant = to_char_scalar(participant);
    node.group = to_char_scalar(group);
    node.condition = to_char_scalar(condition);
    node.name = to_char_scalar(name);
    node.level = to_char_scalar(level);
    node.centers = normalize_maps(double(centers));
    node.n_maps = double(n_maps);
    node.inherited = false;
    node.source_file_path = to_char_scalar(source_file);
    node.K_estimated = NaN;
    node.model_comparison = [];
    node.template_alignment = [];
    if nargin >= 9 && ~isempty(fit)
        if isfield(fit, 'K_estimated')
            node.K_estimated = fit.K_estimated;
        end
        if isfield(fit, 'model_comparison')
            node.model_comparison = fit.model_comparison;
        end
        if isfield(fit, 'template_alignment')
            node.template_alignment = fit.template_alignment;
        end
    end
end

function centers = aggregate_file_template_centers(file_index, template_bank, template_bank_rows, meta_fit)
    K = size(meta_fit.centers, 1);
    centers = meta_fit.centers;
    if isempty(template_bank_rows) || ~any(template_bank_rows.file_index == file_index)
        return;
    end
    file_mask = template_bank_rows.file_index == file_index;
    for k = 1:K
        state_mask = file_mask & template_bank_rows.meta_state == k;
        if ~any(state_mask)
            continue;
        end
        weights = ones(sum(state_mask), 1);
        if ismember('template_abs_corr', template_bank_rows.Properties.VariableNames)
            weights = abs(template_bank_rows.template_abs_corr(state_mask));
        end
        centers(k, :) = weighted_average_maps(template_bank(state_mask, :), weights, meta_fit.centers(k, :));
    end
    centers = sign_align_and_normalise_centers(centers, meta_fit.centers);
end

function row = weighted_average_maps(rows, weights, fallback)
    if isempty(rows)
        row = fallback;
        return;
    end
    if isempty(weights)
        weights = ones(size(rows, 1), 1);
    end
    weights = double(weights(:));
    weights(~isfinite(weights) | weights < 0) = 0;
    if sum(weights) <= eps
        row = mean(rows, 1, 'omitnan');
    else
        row = sum(double(rows) .* weights, 1, 'omitnan') ./ max(eps, sum(weights));
    end
    if isempty(row) || all(~isfinite(row))
        row = fallback;
    end
    row = normalize_maps(double(row));
end

function nodes = build_group_condition_nodes(file_nodes, manifest, meta_fit)
    nodes = repmat(build_pipeline_node('', '', '', '', '', meta_fit.centers, 0, '', struct()), 0, 1);
    if isempty(file_nodes)
        return;
    end
    group_vals = unique_nonempty_strings(manifest.group);
    condition_vals = unique_nonempty_strings(manifest.condition);
    idx = 0;
    for g = 1:numel(group_vals)
        for c = 1:numel(condition_vals)
            mask = string(manifest.group) == group_vals(g) & string(manifest.condition) == condition_vals(c);
            if ~any(mask)
                continue;
            end
            idx = idx + 1;
            centers_stack = cat(3, file_nodes(mask).centers);
            centers = mean(centers_stack, 3, 'omitnan');
            centers = sign_align_and_normalise_centers(centers, meta_fit.centers);
            nodes(idx, 1) = build_pipeline_node('', group_vals(g), condition_vals(c), 'group_condition', ...
                sprintf('%s__%s', group_vals(g), condition_vals(c)), centers, sum([file_nodes(mask).n_maps]), '', struct('K_estimated', size(centers, 1)));
        end
    end
end

function value = get_field_or_default(S, field_name, default_value)
    if isstruct(S) && isfield(S, field_name)
        value = S.(field_name);
    else
        value = default_value;
    end
end

function text = to_char_scalar(value)
    if isstring(value) || ischar(value)
        text = char(string(value));
    elseif isempty(value)
        text = '';
    else
        text = char(string(value));
    end
end

function idx = find_first_col(names, aliases)
    idx = [];
    for a = 1:numel(aliases)
        hit = find(names == aliases(a), 1, 'first');
        if ~isempty(hit)
            idx = hit;
            return;
        end
    end
end

function names = normalise_header_tokens(names)
    names = lower(regexprep(strtrim(string(names)), '[^a-zA-Z0-9]+', '_'));
    names = regexprep(names, '_+', '_');
    names = regexprep(names, '^_|_$', '');
end

function pth = resolve_manifest_path(pth, csv_dir)
    pth = strtrim(pth);
    if is_absolute_path(pth)
        return;
    end
    pth = fullfile(csv_dir, pth);
end

function tf = is_absolute_path(pth)
    tf = startsWith(pth, filesep) || ~isempty(regexp(pth, '^[A-Za-z]:[\\/]', 'once'));
end

function pth = which_or_absolute(pth)
    w = which(pth);
    if ~isempty(w)
        pth = w;
    end
end

function pid = infer_participant_id_from_path(pth, row_index)
    [~, name, ~] = fileparts(pth);
    patterns = {'sub-[A-Za-z0-9]+', 'suj[_-]?[A-Za-z0-9]+', 'subj[_-]?[A-Za-z0-9]+', 'subject[_-]?[A-Za-z0-9]+', '[A-Za-z]+[_-]?\d{3,}'};
    pid = '';
    for i = 1:numel(patterns)
        hit = regexp(name, patterns{i}, 'match', 'once');
        if ~isempty(hit)
            pid = hit;
            break;
        end
    end
    if isempty(pid)
        parts = regexp(name, '[_\-]+', 'split');
        if ~isempty(parts) && ~isempty(parts{1})
            pid = parts{1};
        else
            pid = sprintf('row%04d', row_index);
        end
    end
end

function ensure_dir(d)
    d = char(d);
    if ~exist(d, 'dir')
        mkdir(d);
    end
end

function key = make_row_key(i, participant, condition, group, file_path)
    [~, nm, ~] = fileparts(file_path);
    key = sprintf('%04d_%s_%s_%s_%s', i, participant, group, condition, nm);
    key = regexprep(key, '[^A-Za-z0-9_\-]+', '_');
    key = regexprep(key, '_+', '_');
    if numel(key) > 180
        key = key(1:180);
    end
end

function output_rows = initialise_output_rows(n)
    output_rows = table();
    output_rows.participant = repmat({''}, n, 1);
    output_rows.condition = repmat({''}, n, 1);
    output_rows.group = repmat({''}, n, 1);
    output_rows.file_path = repmat({''}, n, 1);
    output_rows.cluster_solution_file = repmat({''}, n, 1);
    output_rows.gfp_peak_cache_file = repmat({''}, n, 1);
    output_rows.analyze_single_result_file = repmat({''}, n, 1);
    output_rows.json_file = repmat({''}, n, 1);
    output_rows.local_K = nan(n, 1);
    output_rows.n_gfp_peaks = nan(n, 1);
    output_rows.n_gfp_peaks_used = nan(n, 1);
    output_rows.n_target_channels = nan(n, 1);
    output_rows.n_observed_channels = nan(n, 1);
    output_rows.n_interpolated_channels = nan(n, 1);
    output_rows.interpolated_channel_fraction = nan(n, 1);
    output_rows.status = repmat({'pending'}, n, 1);
    output_rows.error_message = repmat({''}, n, 1);
end

function write_checkpoint_manifest(T, fn)
    try
        writetable(T, fn);
    catch
    end
end

function out_csv = resolve_backfit_metrics_csv_path(cfg)
    out_csv = cfg.backfit_state_metrics_csv;
    if isempty(out_csv)
        out_csv = fullfile(cfg.output_dir, 'participant_condition_state_backfit_metrics.csv');
        return;
    end
    if ~is_absolute_path_local(out_csv)
        out_csv = fullfile(cfg.output_dir, out_csv);
    end
end

function out_csv = resolve_backfit_pairwise_metrics_csv_path(cfg)
    out_csv = cfg.backfit_pairwise_metrics_csv;
    if isempty(out_csv)
        out_csv = fullfile(cfg.output_dir, 'participant_condition_state_pairwise_backfit_metrics.csv');
        return;
    end
    if ~is_absolute_path_local(out_csv)
        out_csv = fullfile(cfg.output_dir, out_csv);
    end
end

function out_csv = resolve_backfit_record_summary_csv_path(cfg)
    out_csv = cfg.backfit_record_summary_csv;
    if isempty(out_csv)
        out_csv = fullfile(cfg.output_dir, 'participant_condition_record_backfit_summary.csv');
        return;
    end
    if ~is_absolute_path_local(out_csv)
        out_csv = fullfile(cfg.output_dir, out_csv);
    end
end

function [T_state_metrics, T_pairwise_metrics, T_record_summary] = compile_backfit_metric_tables(output_rows, cfg)
    T_state_metrics = table();
    T_pairwise_metrics = table();
    T_record_summary = table();
    if ~istable(output_rows) || height(output_rows) == 0
        return;
    end

    state_rows = cell(height(output_rows), 1);
    pair_rows = cell(height(output_rows), 1);
    record_rows = cell(height(output_rows), 1);
    state_row_count = 0;
    pair_row_count = 0;
    record_row_count = 0;
    for i = 1:height(output_rows)
        if ~ismember('status', output_rows.Properties.VariableNames) || ~strcmpi(char(string(output_rows.status{i})), 'ok')
            continue;
        end
        eeg_file = char(string(output_rows.file_path{i}));
        if isempty(eeg_file) || ~isfile(eeg_file)
            continue;
        end
        try
            [Results, backfit_source] = load_backfit_result_for_row(output_rows, i);
            if isempty(Results) || ~isstruct(Results) || ~isfield(Results, 'centers')
                continue;
            end
            [file_table, pairwise_table, record_table] = summarise_single_file_backfit_metrics( ...
                eeg_file, Results, ...
                char(string(output_rows.participant{i})), ...
                char(string(output_rows.condition{i})), ...
                char(string(output_rows.group{i})), ...
                cfg, backfit_source);
            if ~isempty(file_table)
                state_row_count = state_row_count + 1;
                state_rows{state_row_count} = file_table;
            end
            if ~isempty(pairwise_table)
                pair_row_count = pair_row_count + 1;
                pair_rows{pair_row_count} = pairwise_table;
            end
            if ~isempty(record_table)
                record_row_count = record_row_count + 1;
                record_rows{record_row_count} = record_table;
            end
        catch ME
            warning('Backfit state-metric extraction failed for %s: %s', eeg_file, ME.message);
        end
    end

    state_rows = state_rows(1:state_row_count);
    if ~isempty(state_rows)
        T_state_metrics = vertcat(state_rows{:});
        if ismember('template_label', T_state_metrics.Properties.VariableNames)
            T_state_metrics = sortrows(T_state_metrics, {'participant', 'condition', 'backfit_method', 'template_label', 'state_index'});
        end
    end

    pair_rows = pair_rows(1:pair_row_count);
    if ~isempty(pair_rows)
        T_pairwise_metrics = vertcat(pair_rows{:});
        T_pairwise_metrics = sortrows(T_pairwise_metrics, {'participant', 'condition', 'backfit_method', 'state_i_label', 'state_j_label'});
    end

    record_rows = record_rows(1:record_row_count);
    if ~isempty(record_rows)
        T_record_summary = vertcat(record_rows{:});
        T_record_summary = sortrows(T_record_summary, {'participant', 'condition', 'backfit_method'});
    end
end

function [Results, source_name] = load_backfit_result_for_row(output_rows, row_idx)
    Results = [];
    source_name = '';
    if ismember('analyze_single_result_file', output_rows.Properties.VariableNames)
        result_file = char(string(output_rows.analyze_single_result_file{row_idx}));
        if ~isempty(result_file) && isfile(result_file)
            S = load(result_file, 'AnalysisResults');
            if isfield(S, 'AnalysisResults') && isstruct(S.AnalysisResults) && isfield(S.AnalysisResults, 'centers')
                Results = S.AnalysisResults;
                source_name = 'analyze_single';
                return;
            end
        end
    end
    if ismember('cluster_solution_file', output_rows.Properties.VariableNames)
        cluster_file = char(string(output_rows.cluster_solution_file{row_idx}));
        if ~isempty(cluster_file) && isfile(cluster_file)
            S = load(cluster_file, 'FileFit');
            if isfield(S, 'FileFit') && isstruct(S.FileFit) && isfield(S.FileFit, 'centers')
                Results = S.FileFit;
                source_name = 'hierarchical_file_fit';
            end
        end
    end
end

function [T_file, T_pairwise, T_record] = summarise_single_file_backfit_metrics(eeg_file, Results, participant, condition, group, cfg, backfit_source)
    T_file = table();
    T_pairwise = table();
    T_record = table();
    if ~isfield(Results, 'centers') || isempty(Results.centers)
        return;
    end

    [Sim_backfit, gfp] = prepare_full_record_for_state_metrics(eeg_file, Results, cfg);
    if isempty(gfp) || ~isstruct(Sim_backfit) || ~isfield(Sim_backfit, 'X_noisy') || isempty(Sim_backfit.X_noisy)
        return;
    end

    backfit = [];
    if isfield(Results, 'backfit_timecourse') && isstruct(Results.backfit_timecourse) && ...
            isfield(Results.backfit_timecourse, 'ok') && Results.backfit_timecourse.ok
        backfit = Results.backfit_timecourse;
    end
    if isempty(backfit)
        backfit = backfit_microstate_timecourse(Sim_backfit, Results);
    end
    if ~isstruct(backfit) || ~isfield(backfit, 'ok') || ~backfit.ok
        return;
    end

    [state_labels, state_order, template_corr] = state_labels_for_metrics(Results);
    duration_s = size(Sim_backfit.X_noisy, 2) / max(Sim_backfit.sfreq, eps);
    n_samples = min(size(gfp, 1), backfit.n_samples);
    gfp = gfp(1:n_samples);

    rows = {};
    pair_rows = {};
    record_rows = {};
    row_count = 0;
    pair_row_count = 0;
    record_row_count = 0;
    mix_available = isfield(backfit, 'mixture') && isstruct(backfit.mixture) && ...
        isfield(backfit.mixture, 'available') && backfit.mixture.available;
    mix_weights = [];
    mix_assignments = [];
    if mix_available
        mix_weights = backfit.mixture.weights;
        mix_assignments = backfit.mixture.assignments;
    end
    mode_specs = { ...
        struct('name', 'hard', 'available', true, 'weights', backfit.hard.weights, 'assignments', backfit.hard.assignments), ...
        struct('name', 'gaussian_mixture', 'available', mix_available, 'weights', mix_weights, 'assignments', mix_assignments)};

    for m = 1:numel(mode_specs)
        spec = mode_specs{m};
        n_samples_mode = n_samples;
        gfp_mode = gfp;
        weights = spec.weights;
        assignments = spec.assignments;
        if spec.available
            if size(weights, 1) > n_samples_mode
                weights = weights(1:n_samples_mode, :);
            elseif size(weights, 1) < n_samples_mode
                n_samples_mode = size(weights, 1);
                gfp_mode = gfp_mode(1:n_samples_mode);
            end
            if isempty(assignments)
                [~, assignments] = max(weights, [], 2);
            else
                assignments = assignments(:);
                assignments = assignments(1:min(numel(assignments), size(weights, 1)));
            end
            if numel(assignments) < size(weights, 1)
                [~, assignments] = max(weights, [], 2);
            end
        else
            weights = nan(n_samples_mode, size(Results.centers, 1));
            assignments = nan(n_samples_mode, 1);
        end

        record_differential_entropy_bits = NaN;
        record_shannon_entropy_bits = NaN;
        if spec.available
            weights_for_record = normalize_weight_matrix_rows(weights);
            if strcmpi(char(spec.name), 'hard')
                record_shannon_entropy_bits = shannon_entropy_bits_from_state_distribution(mean(weights_for_record, 1, 'omitnan'));
            else
                record_differential_entropy_bits = joint_differential_entropy_bits_from_weight_matrix(weights_for_record);
            end
        end

        record_row_count = record_row_count + 1;
        record_rows{record_row_count, 1} = table( ...
            string(participant), string(condition), string(group), string(eeg_file), ...
            string(backfit_source), string(char(spec.name)), logical(spec.available), ...
            double(Results.K_estimated), double(n_samples_mode), double(Sim_backfit.sfreq), double(duration_s), ...
            double(record_differential_entropy_bits), double(record_shannon_entropy_bits), ...
            'VariableNames', {'participant', 'condition', 'group', 'file_path', ...
            'backfit_result_source', 'backfit_method', 'backfit_available', ...
            'K_estimated', 'n_samples', 'sfreq', 'duration_s', ...
            'record_differential_entropy_bits', 'record_shannon_entropy_bits'});

        for j = 1:numel(state_order)
            k = state_order(j);
            wk = double(weights(:, k));
            wk(~isfinite(wk)) = 0;
            occupancy = mean(wk, 'omitnan');
            percentage_record_present = 100 * mean(assignments == k, 'omitnan');
            mean_quantity = occupancy;
            if spec.available && sum(wk, 'omitnan') > eps
                mean_gfp = sum(wk .* gfp_mode, 'omitnan') / sum(wk, 'omitnan');
            else
                mean_gfp = NaN;
            end
            if spec.available
                occurrence_count = count_state_occurrences(assignments, k);
                occurrence_rate_hz = occurrence_count / max(duration_s, eps);
            else
                occupancy = NaN;
                percentage_record_present = NaN;
                mean_quantity = NaN;
                occurrence_count = NaN;
                occurrence_rate_hz = NaN;
                record_differential_entropy_bits = NaN;
                record_shannon_entropy_bits = NaN;
            end

            row_count = row_count + 1;
            rows{row_count, 1} = table( ...
                string(participant), string(condition), string(group), string(eeg_file), ...
                string(backfit_source), ...
                string(char(spec.name)), logical(spec.available), ...
                double(k), string(state_labels{k}), double(Results.K_estimated), ...
                double(n_samples_mode), double(Sim_backfit.sfreq), double(duration_s), ...
                double(occupancy), double(percentage_record_present), double(mean_quantity), ...
                double(mean_gfp), double(occurrence_count), double(occurrence_rate_hz), ...
                double(template_corr(k)), ...
                'VariableNames', {'participant', 'condition', 'group', 'file_path', ...
                'backfit_result_source', 'backfit_method', 'backfit_available', 'state_index', 'template_label', 'K_estimated', ...
                'n_samples', 'sfreq', 'duration_s', ...
                'occupancy', 'percentage_record_present', 'mean_quantity', 'mean_gfp', ...
                'occurrence_count', 'occurrence_rate_hz', ...
                'template_match_abs_correlation'});
        end

        if spec.available
            for a = 1:(numel(state_order) - 1)
                k1 = state_order(a);
                x = double(weights(:, k1));
                for b = (a + 1):numel(state_order)
                    k2 = state_order(b);
                    y = double(weights(:, k2));
                    if strcmpi(char(spec.name), 'hard')
                        [mi_bits, nmi_bits] = binary_mutual_information_bits(x > 0.5, y > 0.5);
                    else
                        [mi_bits, nmi_bits] = normalized_quantity_mutual_information_bits(x, y);
                    end
                    pair_row_count = pair_row_count + 1;
                    pair_rows{pair_row_count, 1} = table( ...
                        string(participant), string(condition), string(group), string(eeg_file), ...
                        string(backfit_source), string(char(spec.name)), logical(spec.available), ...
                        double(Results.K_estimated), double(n_samples_mode), double(Sim_backfit.sfreq), double(duration_s), ...
                        double(k1), string(state_labels{k1}), double(k2), string(state_labels{k2}), ...
                        double(mi_bits), double(nmi_bits), ...
                        'VariableNames', {'participant', 'condition', 'group', 'file_path', ...
                        'backfit_result_source', 'backfit_method', 'backfit_available', ...
                        'K_estimated', 'n_samples', 'sfreq', 'duration_s', ...
                        'state_i_index', 'state_i_label', 'state_j_index', 'state_j_label', ...
                        'mutual_information_bits', 'normalized_mutual_information'});
                end
            end
        end
    end

    if ~isempty(rows)
        T_file = vertcat(rows{:});
    end
    if ~isempty(pair_rows)
        T_pairwise = vertcat(pair_rows{:});
    end
    if ~isempty(record_rows)
        T_record = vertcat(record_rows{:});
    end
end

function [Sim_backfit, gfp] = prepare_full_record_for_state_metrics(eeg_file, Results, cfg)
    [eeg_data, sfreq, chanlocs, labels, pos] = load_eeg_matrix(eeg_file, cfg.sfreq);
    eeg_data = double(eeg_data);
    Sim_backfit = struct();
    if isempty(eeg_data)
        gfp = [];
        return;
    end

    if isempty(labels)
        labels = channel_labels_from_chanlocs(chanlocs, size(eeg_data, 1));
    end

    pre = struct();
    if isfield(Results, 'preprocessing') && isstruct(Results.preprocessing)
        pre = Results.preprocessing;
    else
        pre.apply_average_reference = cfg.apply_average_reference;
        pre.filter_band = cfg.filter_band;
    end

    Sim_backfit.X_noisy = eeg_data;
    Sim_backfit.sfreq = sfreq;
    Sim_backfit.chanlocs = chanlocs;
    Sim_backfit.channel_labels = labels;
    Sim_backfit.pos = pos;
    Sim_backfit.preprocessing = pre;

    remapped = prepare_record_matrix_for_backfit_metrics(Sim_backfit, Results);
    gfp = std(remapped, 0, 1, 'omitnan')';
end

function X_fit = prepare_record_matrix_for_backfit_metrics(Sim_backfit, Results)
    util = microstate_utilities();
    X_fit = double(Sim_backfit.X_noisy);
    if isfield(Results, 'backfit_support') && isstruct(Results.backfit_support) && ...
            isfield(Results.backfit_support, 'channel_remap_spec') && ~isempty(Results.backfit_support.channel_remap_spec)
        X_fit = remap_full_record_for_metrics(X_fit, Results.backfit_support.channel_remap_spec);
    elseif size(X_fit, 1) ~= size(Results.centers, 2)
        if ~isempty(Sim_backfit.chanlocs)
            scalp_mask = scalp_channel_mask(Sim_backfit.channel_labels, Sim_backfit.chanlocs, Sim_backfit.pos);
            if any(scalp_mask) && nnz(scalp_mask) == size(Results.centers, 2)
                X_fit = X_fit(scalp_mask, :);
            end
        end
    end
    if size(X_fit, 1) ~= size(Results.centers, 2)
        error('Backfit metric preparation could not match record channels (%d) to result channels (%d).', ...
            size(X_fit, 1), size(Results.centers, 2));
    end
    if isfield(Sim_backfit.preprocessing, 'apply_average_reference') && Sim_backfit.preprocessing.apply_average_reference
        X_fit = X_fit - mean(X_fit, 1, 'omitnan');
    end
    if isfield(Sim_backfit.preprocessing, 'filter_band') && ~isempty(Sim_backfit.preprocessing.filter_band)
        X_fit = util.bandpass_filter(X_fit, Sim_backfit.sfreq, Sim_backfit.preprocessing.filter_band);
    end
end

function X_target = remap_full_record_for_metrics(eeg_data, spec)
    source_idx = double(spec.source_data_index(:));
    source_subset = eeg_data(source_idx, :);
    n_target = double(spec.n_target_channels);
    X_target = nan(n_target, size(source_subset, 2));
    direct = double(spec.direct_local_index(:));
    observed = isfinite(direct);
    if any(observed)
        X_target(observed, :) = source_subset(direct(observed), :);
    end
    missing = find(~observed);
    for j = 1:numel(missing)
        t = missing(j);
        local_idx = double(spec.interpolation_source_local_index{t});
        weights = double(spec.interpolation_weights{t});
        X_target(t, :) = weights(:)' * source_subset(local_idx, :);
    end
end

function [state_labels, state_order, template_corr] = state_labels_for_metrics(Results)
    K = size(Results.centers, 1);
    state_labels = arrayfun(@(k) sprintf('state_%02d', k), 1:K, 'UniformOutput', false);
    template_corr = nan(K, 1);
    if isfield(Results, 'template_alignment') && isstruct(Results.template_alignment) && ~isempty(Results.template_alignment)
        ta = Results.template_alignment;
        if isfield(ta, 'labels') && numel(ta.labels) >= K
            state_labels = cellstr(string(ta.labels(:)));
        elseif isfield(ta, 'assigned_labels') && numel(ta.assigned_labels) >= K
            state_labels = cellstr(string(ta.assigned_labels(:)));
        end
        if isfield(ta, 'correlations') && numel(ta.correlations) >= K
            template_corr = abs(double(ta.correlations(:)));
        elseif isfield(ta, 'assigned_abs_corr') && numel(ta.assigned_abs_corr) >= K
            template_corr = double(ta.assigned_abs_corr(:));
        end
    end
    [~, state_order] = sort(lower(string(state_labels)));
end

function n = count_state_occurrences(assignments, state_idx)
    if isempty(assignments)
        n = NaN;
        return;
    end
    assignments = double(assignments(:));
    valid = isfinite(assignments);
    assignments = assignments(valid);
    if isempty(assignments)
        n = NaN;
        return;
    end
    state_mask = assignments == state_idx;
    n = double(state_mask(1)) + sum(state_mask(2:end) & ~state_mask(1:end-1));
end

function h = occupancy_entropy_bits(p)
    p = double(p);
    if ~isfinite(p) || p <= 0
        h = 0;
        return;
    end
    h = -p * log2(max(p, eps));
end

function W = normalize_weight_matrix_rows(W)
    W = double(W);
    if isempty(W)
        return;
    end
    W(~isfinite(W)) = 0;
    row_sum = sum(W, 2);
    valid = row_sum > eps;
    W(valid, :) = W(valid, :) ./ row_sum(valid);
    W(~valid, :) = 0;
end

function h_bits = joint_differential_entropy_bits_from_weight_matrix(W)
    W = normalize_weight_matrix_rows(W);
    if isempty(W) || size(W, 1) < 16 || size(W, 2) < 1
        h_bits = NaN;
        return;
    end

    valid = all(isfinite(W), 2);
    W = W(valid, :);
    if size(W, 1) < 16
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
    mask = p > 0;
    discrete_h_bits = -sum(p(mask) .* log2(p(mask)));
    bin_width = edges(2) - edges(1);
    h_bits = discrete_h_bits + d * log2(bin_width);
end

function h_bits = shannon_entropy_bits_from_state_distribution(p)
    p = double(p(:));
    p(~isfinite(p)) = 0;
    total = sum(p);
    if total <= eps
        h_bits = NaN;
        return;
    end
    p = p / total;
    mask = p > 0;
    h_bits = -sum(p(mask) .* log2(p(mask)));
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
    px = sum(joint, 2);
    py = sum(joint, 1);
    mi_bits = 0;
    for i = 1:2
        for j = 1:2
            pij = joint(i, j);
            if pij > 0
                mi_bits = mi_bits + pij * log2(pij / max(px(i) * py(j), eps));
            end
        end
    end
    hx = discrete_entropy_bits(px);
    hy = discrete_entropy_bits(py(:));
    nmi = normalized_mutual_information_from_components(mi_bits, hx, hy);
end

function [mi_bits, nmi] = normalized_quantity_mutual_information_bits(x, y)
    x_norm = normalize_single_quantity_trace(x);
    y_norm = normalize_single_quantity_trace(y);
    valid = isfinite(x_norm) & isfinite(y_norm);
    x_norm = x_norm(valid);
    y_norm = y_norm(valid);
    if numel(x_norm) < 16 || max(x_norm) - min(x_norm) <= eps || max(y_norm) - min(y_norm) <= eps
        mi_bits = NaN;
        nmi = NaN;
        return;
    end

    n_bins = min(16, max(6, round(sqrt(numel(x_norm)) / 2)));
    x_edges = linspace(0, 1, n_bins + 1);
    y_edges = linspace(0, 1, n_bins + 1);
    joint_counts = histcounts2(min(max(x_norm, 0), 1), min(max(y_norm, 0), 1), x_edges, y_edges);
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
            pij = pxy(i, j);
            if pij > 0
                mi_bits = mi_bits + pij * log2(pij / max(px(i) * py(j), eps));
            end
        end
    end
    hx = discrete_entropy_bits(px);
    hy = discrete_entropy_bits(py(:));
    nmi = normalized_mutual_information_from_components(mi_bits, hx, hy);
end

function h = discrete_entropy_bits(p)
    p = double(p(:));
    p = p(isfinite(p) & p > 0);
    if isempty(p)
        h = 0;
        return;
    end
    h = -sum(p .* log2(p));
end

function nmi = normalized_mutual_information_from_components(mi_bits, hx, hy)
    denom = sqrt(max(hx, 0) * max(hy, 0));
    if denom <= eps
        nmi = NaN;
    else
        nmi = mi_bits / denom;
    end
end

function x_norm = normalize_single_quantity_trace(x)
    x = double(x(:));
    x(~isfinite(x)) = 0;
    if isempty(x)
        x_norm = x;
        return;
    end
    x = x - min(x);
    xmax = max(x);
    if xmax <= eps
        x_norm = zeros(size(x));
    else
        x_norm = x ./ xmax;
    end
end

function tf = is_absolute_path_local(pth)
    pth = char(string(pth));
    tf = startsWith(pth, filesep) || ~isempty(regexp(pth, '^[A-Za-z]:[\\/]', 'once'));
end

function ok = maybe_start_parallel_pool(cfg)
    ok = false;
    if ~isfield(cfg, 'use_parfor') || ~cfg.use_parfor
        return;
    end
    if exist('parpool', 'file') ~= 2 || exist('gcp', 'file') ~= 2
        warning('Parallel Computing Toolbox functions are unavailable. Falling back to serial K fits.');
        return;
    end
    try
        p_pool = gcp('nocreate');
        if isempty(p_pool)
            max_workers = max(1, maxNumCompThreads());
            n_workers = min(max(1, round(cfg.n_workers)), max_workers);
            parpool('local', n_workers);
        end
        ok = true;
    catch ME
        warning('Could not start/use parallel pool: %s. Falling back to serial K fits.', ME.message);
        ok = false;
    end
end

function tf = peak_cache_needs_refresh(rec, common_labels, cfg)
    tf = isempty(rec) || ~isstruct(rec) || ~isfield(rec, 'maps_norm') || ~isfield(rec, 'gfp') || ...
        size(rec.maps_norm, 2) ~= numel(common_labels);
    if tf
        return;
    end
    if ~isfield(rec, 'common_labels') || numel(rec.common_labels) ~= numel(common_labels) || ...
            any(~strcmp(cellstr(rec.common_labels(:)), cellstr(common_labels(:))))
        tf = true;
        return;
    end
    needed_fields = {'gfp_effective', 'n_target_channels', 'n_observed_channels', ...
        'n_interpolated_channels', 'interpolated_fraction', 'interpolation_enabled'};
    for i = 1:numel(needed_fields)
        if ~isfield(rec, needed_fields{i})
            tf = true;
            return;
        end
    end
    tf = false;
end

function tf = cluster_solution_needs_refresh(FileFit, rec, common_labels)
    tf = isempty(FileFit) || ~isstruct(FileFit) || ~isfield(FileFit, 'centers') || ...
        size(FileFit.centers, 2) ~= numel(common_labels);
    if tf
        return;
    end
    if isfield(rec, 'maps_norm') && size(rec.maps_norm, 2) ~= size(FileFit.centers, 2)
        tf = true;
        return;
    end
    if ~isfield(FileFit, 'backfit_support') || ~isstruct(FileFit.backfit_support) || ...
            ~isfield(FileFit.backfit_support, 'channel_remap_spec') || isempty(FileFit.backfit_support.channel_remap_spec)
        tf = true;
        return;
    end
    if ~isfield(FileFit, 'preprocessing') || ~isstruct(FileFit.preprocessing)
        tf = true;
        return;
    end
    needs_mixture_payload = isfield(FileFit, 'requested_method') && strcmpi(char(string(FileFit.requested_method)), 'spm_vb');
    if needs_mixture_payload
        tf = ~isfield(FileFit, 'maps_nc') || isempty(FileFit.maps_nc) || ...
            ~isfield(FileFit, 'backfit_peak_labels') || isempty(FileFit.backfit_peak_labels) || ...
            numel(FileFit.backfit_peak_labels) ~= size(FileFit.maps_nc, 1) || ...
            ~isfield(FileFit, 'idx_peaks') || isempty(FileFit.idx_peaks) || ...
            numel(FileFit.idx_peaks) ~= size(FileFit.maps_nc, 1) || ...
            ~isfield(FileFit.backfit_support, 'peak_sample') || isempty(FileFit.backfit_support.peak_sample);
        return;
    end
    tf = false;
end

function out = ternary_string(cond, true_str, false_str)
    if cond
        out = true_str;
    else
        out = false_str;
    end
end

% -------------------------------------------------------------------------
% Channel inspection/loading
% -------------------------------------------------------------------------
function file_meta = inspect_channel_sets(manifest, cfg)
    n = height(manifest);
    file_meta = repmat(struct('file_path', '', 'sfreq', NaN, 'n_channels', NaN, 'labels', {{}}, ...
        'chanlocs', [], 'pos', [], 'original_index', [], 'canonical_labels', {{}}, ...
        'n_channels_after_filter', NaN), n, 1);
    for i = 1:n
        fp = manifest.file_path{i};
        if ~isfile(fp)
            error('EEG file not found: %s', fp);
        end
        [sfreq, labels, chanlocs, pos, n_channels] = load_eeg_info(fp, cfg.sfreq);
        file_meta(i).file_path = fp;
        file_meta(i).sfreq = sfreq;
        file_meta(i).labels = labels(:);
        file_meta(i).chanlocs = chanlocs;
        file_meta(i).pos = pos;
        file_meta(i).n_channels = n_channels;
    end
end

function [sfreq, labels, chanlocs, pos, n_channels] = load_eeg_info(eeg_file, fallback_sfreq)
    [~, ~, ext] = fileparts(eeg_file);
    ext = lower(ext);
    chanlocs = [];
    labels = {};
    pos = [];
    sfreq = fallback_sfreq;
    if strcmp(ext, '.set')
        if exist('pop_loadset', 'file') ~= 2
            error('EEGLAB pop_loadset is required for .set files. Add EEGLAB to the MATLAB path.');
        end
        [filepath, filename, extension] = fileparts(eeg_file);
        try
            EEG = pop_loadset('filename', [filename extension], 'filepath', filepath, 'loadmode', 'info');
        catch
            EEG = pop_loadset(eeg_file);
        end
        n_channels = numel(EEG.chanlocs);
        if isfield(EEG, 'nbchan') && ~isempty(EEG.nbchan)
            n_channels = EEG.nbchan;
        end
        sfreq = EEG.srate;
        chanlocs = EEG.chanlocs;
        labels = channel_labels_from_chanlocs(chanlocs, n_channels);
        pos = pos_from_chanlocs(chanlocs, n_channels);
    elseif strcmp(ext, '.mat')
        S = load(eeg_file);
        [data, sfreq, chanlocs, labels, pos] = extract_mat_data_and_meta(S, fallback_sfreq);
        n_channels = size(data, 1);
        if isempty(labels)
            labels = arrayfun(@(i) sprintf('Ch%d', i), 1:n_channels, 'UniformOutput', false)';
        end
    else
        error('Unsupported EEG file format: %s. Supported: .set and .mat', ext);
    end
end

function [eeg_data, sfreq, chanlocs, labels, pos] = load_eeg_matrix(eeg_file, fallback_sfreq)
    [~, ~, ext] = fileparts(eeg_file);
    ext = lower(ext);
    chanlocs = [];
    labels = {};
    pos = [];
    sfreq = fallback_sfreq;
    if strcmp(ext, '.set')
        if exist('pop_loadset', 'file') ~= 2
            error('EEGLAB pop_loadset is required for .set files. Add EEGLAB to the MATLAB path.');
        end
        EEG = pop_loadset(eeg_file);
        eeg_data = double(EEG.data);
        if ndims(eeg_data) == 3
            eeg_data = reshape(eeg_data, size(eeg_data, 1), []);
        end
        sfreq = EEG.srate;
        chanlocs = EEG.chanlocs;
        labels = channel_labels_from_chanlocs(chanlocs, size(eeg_data, 1));
        pos = pos_from_chanlocs(chanlocs, size(eeg_data, 1));
    elseif strcmp(ext, '.mat')
        S = load(eeg_file);
        [eeg_data, sfreq, chanlocs, labels, pos] = extract_mat_data_and_meta(S, fallback_sfreq);
        eeg_data = double(eeg_data);
        if ndims(eeg_data) == 3
            eeg_data = reshape(eeg_data, size(eeg_data, 1), []);
        end
    else
        error('Unsupported EEG file format: %s. Supported: .set and .mat', ext);
    end
end

function [data, sfreq, chanlocs, labels, pos] = extract_mat_data_and_meta(S, fallback_sfreq)
    if isfield(S, 'eeg_data')
        data = S.eeg_data;
    elseif isfield(S, 'data')
        data = S.data;
    elseif isfield(S, 'EEG') && isfield(S.EEG, 'data')
        data = S.EEG.data;
    else
        error('Could not find EEG data in .mat file. Expected eeg_data, data, or EEG.data.');
    end
    if isfield(S, 'sfreq')
        sfreq = S.sfreq;
    elseif isfield(S, 'srate')
        sfreq = S.srate;
    elseif isfield(S, 'EEG') && isfield(S.EEG, 'srate')
        sfreq = S.EEG.srate;
    else
        sfreq = fallback_sfreq;
    end
    if isfield(S, 'chanlocs')
        chanlocs = S.chanlocs;
    elseif isfield(S, 'EEG') && isfield(S.EEG, 'chanlocs')
        chanlocs = S.EEG.chanlocs;
    else
        chanlocs = [];
    end
    if isfield(S, 'channel_labels')
        labels = cellstr(S.channel_labels);
    elseif isfield(S, 'ch_labels')
        labels = cellstr(S.ch_labels);
    elseif ~isempty(chanlocs)
        labels = channel_labels_from_chanlocs(chanlocs, size(data, 1));
    else
        labels = {};
    end
    if isfield(S, 'pos')
        pos = S.pos;
    elseif ~isempty(chanlocs)
        pos = pos_from_chanlocs(chanlocs, size(data, 1));
    else
        pos = [];
    end
end

function labels = channel_labels_from_chanlocs(chanlocs, n_channels)
    labels = cell(n_channels, 1);
    for i = 1:n_channels
        labels{i} = sprintf('Ch%d', i);
    end
    if ~isempty(chanlocs) && numel(chanlocs) >= n_channels && isfield(chanlocs, 'labels')
        for i = 1:n_channels
            if ~isempty(chanlocs(i).labels)
                labels{i} = char(chanlocs(i).labels);
            end
        end
    end
end

function pos = pos_from_chanlocs(chanlocs, n_channels)
    pos = nan(n_channels, 3);
    if isempty(chanlocs) || numel(chanlocs) < n_channels
        return;
    end
    for i = 1:n_channels
        if isfield(chanlocs, 'X') && ~isempty(chanlocs(i).X) && isfield(chanlocs, 'Y') && ~isempty(chanlocs(i).Y)
            pos(i, 1) = double(chanlocs(i).X);
            pos(i, 2) = double(chanlocs(i).Y);
            if isfield(chanlocs, 'Z') && ~isempty(chanlocs(i).Z)
                pos(i, 3) = double(chanlocs(i).Z);
            else
                pos(i, 3) = 0;
            end
        end
    end
end

function [common_labels, common_index_by_file, common_chanlocs, common_pos] = define_common_channels(file_meta, cfg)
    n_files = numel(file_meta);
    for i = 1:n_files
        file_meta(i) = preprocess_file_meta_channels(file_meta(i), cfg);
    end

    if cfg.interpolate_missing_peak_channels
        ref_idx = select_target_channel_reference(file_meta);
        common_labels = file_meta(ref_idx).labels(:);
        common_chanlocs = file_meta(ref_idx).chanlocs;
        common_pos = file_meta(ref_idx).pos;
        common_can = file_meta(ref_idx).canonical_labels(:);
        if isempty(common_can)
            error('Reference channel set is empty after scalp-channel filtering.');
        end
        common_index_by_file = cell(n_files, 1);
        for i = 1:n_files
            common_index_by_file{i} = build_channel_remap_spec(file_meta(i), common_can, common_pos, cfg);
        end
    else
        canon_sets = cell(n_files, 1);
        for i = 1:n_files
            canon_sets{i} = file_meta(i).canonical_labels(:);
        end

        common_can = canon_sets{1};
        for i = 2:n_files
            common_can = intersect(common_can, canon_sets{i}, 'stable');
        end
        if isempty(common_can)
            error('No common channels across files.');
        end
        if strcmp(cfg.channel_policy, 'strict')
            base = canon_sets{1};
            for i = 2:n_files
                if numel(base) ~= numel(canon_sets{i}) || any(~strcmp(base(:), canon_sets{i}(:)))
                    error('channel_policy=strict, but file %d has a non-identical channel set.', i);
                end
            end
            common_can = base;
        elseif ~strcmp(cfg.channel_policy, 'intersect')
            error('channel_policy must be intersect or strict when interpolation is disabled.');
        end

        first_labels = file_meta(1).labels;
        first_can = file_meta(1).canonical_labels;
        common_labels = cell(numel(common_can), 1);
        for c = 1:numel(common_can)
            idx = find(strcmp(first_can, common_can{c}), 1, 'first');
            common_labels{c} = first_labels{idx};
        end
        if ~isempty(file_meta(1).chanlocs)
            common_chanlocs = file_meta(1).chanlocs(match_labels_or_die(first_can, common_can));
        else
            common_chanlocs = [];
        end
        if ~isempty(file_meta(1).pos)
            common_pos = file_meta(1).pos(match_labels_or_die(first_can, common_can), :);
        else
            common_pos = [];
        end
        common_index_by_file = cell(n_files, 1);
        for i = 1:n_files
            common_index_by_file{i} = build_channel_remap_spec(file_meta(i), common_can, common_pos, cfg);
            if common_index_by_file{i}.n_interpolated_channels > 0
                error('Interpolation was disabled, but file %d is missing target channels after filtering.', i);
            end
        end
    end
end

function meta = preprocess_file_meta_channels(meta, cfg)
    labels = meta.labels(:);
    chanlocs = meta.chanlocs;
    pos = meta.pos;
    keep = true(numel(labels), 1);
    if cfg.use_scalp_channels
        keep = scalp_channel_mask(labels, chanlocs, pos);
    end
    meta.original_index = find(keep);
    meta.labels = labels(keep);
    if ~isempty(chanlocs)
        meta.chanlocs = chanlocs(keep);
    else
        meta.chanlocs = [];
    end
    if ~isempty(pos)
        meta.pos = pos(keep, :);
    else
        meta.pos = [];
    end
    meta.canonical_labels = canonical_channel_labels(meta.labels);
    meta.n_channels_after_filter = numel(meta.labels);
end

function ref_idx = select_target_channel_reference(file_meta)
    n_files = numel(file_meta);
    score = -inf(n_files, 1);
    for i = 1:n_files
        n_channels = numel(file_meta(i).labels);
        if isempty(file_meta(i).pos)
            n_pos = 0;
        else
            n_pos = nnz(all(isfinite(file_meta(i).pos), 2) & sqrt(sum(file_meta(i).pos.^2, 2)) > 0);
        end
        score(i) = 10000 * n_channels + n_pos;
    end
    [~, ref_idx] = max(score);
end

function spec = build_channel_remap_spec(file_meta, target_can, target_pos, cfg)
    source_can = file_meta.canonical_labels(:);
    source_idx = file_meta.original_index(:);
    n_target = numel(target_can);
    direct_local_index = nan(n_target, 1);
    for c = 1:n_target
        hit = find(strcmp(source_can, target_can{c}), 1, 'first');
        if ~isempty(hit)
            direct_local_index(c) = hit;
        end
    end

    spec = struct();
    spec.source_data_index = source_idx;
    spec.direct_local_index = direct_local_index;
    spec.target_labels = target_can(:);
    spec.n_target_channels = n_target;
    spec.n_observed_channels = nnz(isfinite(direct_local_index));
    spec.n_interpolated_channels = n_target - spec.n_observed_channels;
    spec.observed_fraction = spec.n_observed_channels / max(1, n_target);
    spec.interpolated_fraction = 1 - spec.observed_fraction;
    spec.interpolation_enabled = cfg.interpolate_missing_peak_channels && spec.n_interpolated_channels > 0;
    spec.interpolation_weights = cell(n_target, 1);
    spec.interpolation_source_local_index = cell(n_target, 1);

    if spec.n_interpolated_channels == 0
        return;
    end
    if ~cfg.interpolate_missing_peak_channels
        return;
    end
    if isempty(target_pos) || size(target_pos, 1) ~= n_target
        error('Target channel positions are required for peak interpolation.');
    end
    source_pos = file_meta.pos;
    if isempty(source_pos) || size(source_pos, 1) ~= numel(source_idx)
        error('Source channel positions are required for peak interpolation: %s', file_meta.file_path);
    end
    valid_source = all(isfinite(source_pos), 2) & sqrt(sum(source_pos.^2, 2)) > 0;
    if nnz(valid_source) < cfg.peak_interpolation_min_source_channels
        error('Only %d source scalp channels with valid positions are available for interpolation in %s.', ...
            nnz(valid_source), file_meta.file_path);
    end

    source_pos_valid = source_pos(valid_source, :);
    valid_local_index = find(valid_source);
    missing = find(~isfinite(direct_local_index));
    for j = 1:numel(missing)
        t = missing(j);
        target_xyz = target_pos(t, :);
        if ~all(isfinite(target_xyz)) || sqrt(sum(target_xyz.^2)) <= 0
            error('Target channel %d lacks a valid 3D position required for interpolation.', t);
        end
        [local_idx, weights] = build_interpolation_weights(source_pos_valid, target_xyz, cfg);
        spec.interpolation_source_local_index{t} = valid_local_index(local_idx);
        spec.interpolation_weights{t} = weights;
    end
end

function idx = match_labels_or_die(source_can, target_can)
    idx = zeros(numel(target_can), 1);
    for c = 1:numel(target_can)
        hit = find(strcmp(source_can, target_can{c}), 1, 'first');
        if isempty(hit)
            error('Internal channel matching failure for channel %s.', target_can{c});
        end
        idx(c) = hit;
    end
end

function labels = canonical_channel_labels(labels_in)
    labels = cellstr(labels_in(:));
    for i = 1:numel(labels)
        s = upper(strtrim(labels{i}));
        s = regexprep(s, '[^A-Z0-9]', '');
        labels{i} = s;
    end
end

function keep = scalp_channel_mask(labels, chanlocs, pos)
    n = numel(labels);
    keep = true(n, 1);
    bad_patterns = {'ECG','EKG','EOG','HEOG','VEOG','EMG','EXG','GSR','RESP','TRIG','TRIGGER','STI','STATUS','PHOTO','AUX','MISC','REF'};
    for i = 1:n
        lab = upper(strtrim(labels{i}));
        for b = 1:numel(bad_patterns)
            if contains(lab, bad_patterns{b})
                keep(i) = false;
                break;
            end
        end
    end
    if ~isempty(pos) && size(pos, 1) == n
        has_pos = all(isfinite(pos), 2) & sqrt(sum(pos.^2, 2)) > 0;
        if nnz(has_pos & keep) >= 16
            keep = keep & has_pos;
        end
    elseif ~isempty(chanlocs) && numel(chanlocs) >= n
        has_xy = false(n, 1);
        for i = 1:n
            has_xy(i) = isfield(chanlocs, 'X') && ~isempty(chanlocs(i).X) && isfinite(double(chanlocs(i).X));
        end
        if nnz(has_xy & keep) >= 16
            keep = keep & has_xy;
        end
    end
end

% -------------------------------------------------------------------------
% GFP peak extraction
% -------------------------------------------------------------------------
function rec = extract_file_gfp_maps(eeg_file, chan_spec, common_labels, common_chanlocs, common_pos, cfg)
    [eeg_data, sfreq, ~, ~, ~] = load_eeg_matrix(eeg_file, cfg.sfreq);
    eeg_data = eeg_data(chan_spec.source_data_index, :);
    eeg_data = double(eeg_data);
    good_t = all(isfinite(eeg_data), 1);
    eeg_data = eeg_data(:, good_t);
    if isempty(eeg_data) || size(eeg_data, 2) < 10
        error('No usable EEG samples after finite-value filtering.');
    end
    if cfg.apply_average_reference
        eeg_data = eeg_data - mean(eeg_data, 1, 'omitnan');
    end
    eeg_data = apply_temporal_filter(eeg_data, sfreq, cfg.filter_band);
    if cfg.apply_average_reference
        eeg_data = eeg_data - mean(eeg_data, 1, 'omitnan');
    end

    gfp = std(eeg_data, 0, 1, 'omitnan');
    peak_idx = find_gfp_peaks(gfp, cfg.gfp_peak_min_distance_samples, cfg.gfp_peak_quantile_schedule, cfg.min_gfp_peaks_per_file);
    if isempty(peak_idx)
        error('No GFP peaks found.');
    end
    source_maps = eeg_data(:, peak_idx)';
    maps = remap_peak_maps_to_target(source_maps, chan_spec);
    gfp_peak = gfp(peak_idx)';
    n_raw = size(maps, 1);

    valid = all(isfinite(maps), 2) & isfinite(gfp_peak) & gfp_peak > 0;
    maps = maps(valid, :);
    gfp_peak = gfp_peak(valid);
    peak_idx = peak_idx(valid);

    if cfg.reject_gfp_peak_outliers_individual && numel(gfp_peak) >= max(20, cfg.min_gfp_peaks_per_file)
        keep = robust_upper_outlier_mask(gfp_peak, cfg.gfp_outlier_mad_multiplier_individual, cfg.min_gfp_peaks_per_file);
        maps = maps(keep, :);
        gfp_peak = gfp_peak(keep);
        peak_idx = peak_idx(keep);
    end

    if ~isempty(cfg.max_maps_per_file) && size(maps, 1) > cfg.max_maps_per_file
        idx = deterministic_subsample(size(maps, 1), cfg.max_maps_per_file);
        maps = maps(idx, :);
        gfp_peak = gfp_peak(idx);
        peak_idx = peak_idx(idx);
    end

    maps_norm = normalize_maps(maps);
    interpolation_weight_scale = max(0, chan_spec.observed_fraction);
    rec = struct();
    rec.eeg_file = eeg_file;
    rec.sfreq = sfreq;
    rec.common_labels = common_labels;
    rec.common_chanlocs = common_chanlocs;
    rec.common_pos = common_pos;
    rec.peak_sample = peak_idx(:);
    rec.gfp = gfp_peak(:);
    rec.gfp_effective = gfp_peak(:) * sqrt(interpolation_weight_scale);
    rec.maps = single(maps);
    rec.maps_norm = single(maps_norm);
    rec.n_peaks_raw = n_raw;
    rec.n_peaks_used = size(maps_norm, 1);
    rec.n_target_channels = chan_spec.n_target_channels;
    rec.n_observed_channels = chan_spec.n_observed_channels;
    rec.n_interpolated_channels = chan_spec.n_interpolated_channels;
    rec.observed_fraction = chan_spec.observed_fraction;
    rec.interpolated_fraction = chan_spec.interpolated_fraction;
    rec.interpolation_weight_scale = interpolation_weight_scale;
    rec.interpolation_enabled = chan_spec.interpolation_enabled;
end

function maps = remap_peak_maps_to_target(source_maps, chan_spec)
    n_peaks = size(source_maps, 1);
    n_target = chan_spec.n_target_channels;
    maps = nan(n_peaks, n_target);
    direct = chan_spec.direct_local_index;
    observed = isfinite(direct);
    if any(observed)
        maps(:, observed) = source_maps(:, direct(observed));
    end
    missing = find(~observed);
    for j = 1:numel(missing)
        t = missing(j);
        local_idx = chan_spec.interpolation_source_local_index{t};
        weights = chan_spec.interpolation_weights{t};
        if isempty(local_idx) || isempty(weights)
            continue;
        end
        maps(:, t) = source_maps(:, local_idx) * weights;
    end
end

function [local_idx, weights] = build_interpolation_weights(source_pos, target_xyz, cfg)
    d = sqrt(sum((source_pos - target_xyz).^2, 2));
    [ds, ord] = sort(d, 'ascend');
    m = min(cfg.peak_interpolation_neighbours, numel(ord));
    ord = ord(1:m);
    ds = ds(1:m);
    if isempty(ds)
        local_idx = [];
        weights = [];
        return;
    end
    if ds(1) <= eps
        weights = zeros(m, 1);
        weights(1) = 1;
    else
        weights = 1 ./ (ds.^2 + eps);
        weights = weights ./ sum(weights);
    end
    local_idx = ord(:);
    weights = weights(:);
end

function data = apply_temporal_filter(data, sfreq, band)
    if isempty(band)
        return;
    end
    band = double(band(:)');
    if numel(band) ~= 2 || band(1) <= 0 || band(2) >= sfreq/2 || band(1) >= band(2)
        warning('Invalid filter_band [%s] for sfreq %.3f. Skipping filter.', num2str(band), sfreq);
        return;
    end
    if exist('butter', 'file') ~= 2 || exist('filtfilt', 'file') ~= 2
        warning('butter/filtfilt unavailable. Skipping temporal filtering.');
        return;
    end
    if size(data, 2) < round(3 * sfreq / band(1))
        warning('Recording too short for stable filtering. Skipping temporal filtering.');
        return;
    end
    try
        [b, a] = butter(2, band ./ (sfreq/2), 'bandpass');
        data = filtfilt(b, a, data')';
    catch ME
        warning(ME.identifier, 'Temporal filtering failed (%s). Continuing unfiltered.', ME.message);
    end
end

function peak_idx = find_gfp_peaks(gfp, min_dist, q_schedule, min_peaks)
    gfp = double(gfp(:)');
    peak_idx = [];
    q_schedule = sort(q_schedule(:)', 'ascend');
    for q = q_schedule
        thr = quantile(gfp(isfinite(gfp)), q);
        if exist('findpeaks', 'file') == 2
            try
                [~, locs] = findpeaks(gfp, 'MinPeakDistance', round(min_dist), 'MinPeakHeight', thr);
            catch
                locs = local_peak_finder(gfp, min_dist, thr);
            end
        else
            locs = local_peak_finder(gfp, min_dist, thr);
        end
        peak_idx = locs(:)';
        if numel(peak_idx) >= min_peaks
            return;
        end
    end
    if isempty(peak_idx)
        peak_idx = local_peak_finder(gfp, min_dist, -Inf);
    end
end

function locs = local_peak_finder(x, min_dist, thr)
    x = x(:)';
    candidates = find(x(2:end-1) > x(1:end-2) & x(2:end-1) >= x(3:end) & x(2:end-1) >= thr) + 1;
    if isempty(candidates)
        locs = [];
        return;
    end
    [~, order] = sort(x(candidates), 'descend');
    candidates = candidates(order);
    chosen = [];
    for i = 1:numel(candidates)
        c = candidates(i);
        if isempty(chosen) || all(abs(c - chosen) >= min_dist)
            chosen(end+1) = c; %#ok<AGROW>
        end
    end
    locs = sort(chosen);
end

function keep = robust_upper_outlier_mask(x, mad_mult, min_keep)
    x = double(x(:));
    med = median(x, 'omitnan');
    madv = median(abs(x - med), 'omitnan') * 1.4826;
    if ~isfinite(madv) || madv <= eps
        thr = quantile(x, 0.995);
    else
        thr = med + mad_mult * madv;
    end
    keep = isfinite(x) & x <= thr;
    if nnz(keep) < min_keep
        [~, ord] = sort(x, 'ascend');
        keep = false(size(x));
        keep(ord(1:min(numel(x), min_keep))) = true;
    end
end

function idx = deterministic_subsample(N, maxN)
    if isempty(maxN) || N <= maxN
        idx = (1:N)';
        return;
    end
    idx = unique(round(linspace(1, N, maxN)))';
    if numel(idx) > maxN
        idx = idx(1:maxN);
    end
end

% -------------------------------------------------------------------------
% Topographic microstate clustering
% -------------------------------------------------------------------------
function Fit = fit_microstate_map_set(maps, gfp, K_candidates, cfg, fit_name)
    X = normalize_maps(double(maps));
    gfp = double(gfp(:));
    if numel(gfp) ~= size(X, 1)
        gfp = ones(size(X, 1), 1);
    end
    valid = all(isfinite(X), 2) & isfinite(gfp) & gfp > 0;
    X = X(valid, :);
    gfp = gfp(valid);
    if size(X, 1) < min(K_candidates)
        error('Not enough maps (%d) for requested K candidates.', size(X, 1));
    end
    K_candidates = K_candidates(K_candidates <= max(2, size(X, 1) - 1));
    K_candidates = K_candidates(K_candidates >= 2);
    if isempty(K_candidates)
        error('No valid K candidates for %d maps.', size(X, 1));
    end

    fits = cell(numel(K_candidates), 1);
    metrics = repmat(empty_metrics_struct(), numel(K_candidates), 1);
    fit_name_code = sum(double(char(fit_name)));
    if should_parallelise_k_fits(cfg, K_candidates)
        parfor k_i = 1:numel(K_candidates)
            K = K_candidates(k_i);
            best = best_topographic_kmeans_across_restarts(X, gfp, K, cfg, fit_name_code);
            fits{k_i} = best;
            metrics(k_i) = best.metrics;
        end
    else
        for k_i = 1:numel(K_candidates)
            K = K_candidates(k_i);
            best = best_topographic_kmeans_across_restarts(X, gfp, K, cfg, fit_name_code);
            fits{k_i} = best;
            metrics(k_i) = best.metrics;
        end
    end

    model_comparison = struct2table(metrics);
    [K_est, best_idx] = select_K(K_candidates, cfg.criterion, model_comparison);
    best_fit = fits{best_idx};

    Fit = struct();
    Fit.name = fit_name;
    Fit.method = 'polarity_invariant_topographic_kmeans';
    Fit.requested_method = cfg.method;
    Fit.criterion = cfg.criterion;
    Fit.K_candidates = K_candidates;
    Fit.K_estimated = K_est;
    Fit.best_index = best_idx;
    Fit.centers = best_fit.centers;
    Fit.labels = best_fit.labels;
    Fit.metrics = best_fit.metrics;
    Fit.model_comparison = model_comparison;
    Fit.n_maps = size(X, 1);
    Fit.n_channels = size(X, 2);
end

function tf = should_parallelise_k_fits(cfg, K_candidates)
    tf = isfield(cfg, 'use_parfor') && cfg.use_parfor && ...
        isfield(cfg, 'parallel_pool_ready') && cfg.parallel_pool_ready && ...
        numel(K_candidates) > 1;
end

function best = best_topographic_kmeans_across_restarts(X, gfp, K, cfg, fit_name_code)
    best = [];
    for r = 1:cfg.n_initialisations
        seed = cfg.random_seed + 1000 * K + r + fit_name_code;
        tmp = topographic_kmeans(X, gfp, K, cfg.max_iter, cfg.tol, seed);
        if isempty(best) || tmp.metrics.wss < best.metrics.wss
            best = tmp;
        end
    end
end

function fit = topographic_kmeans(X, gfp, K, max_iter, tol, seed)
    rng(seed, 'twister');
    X = normalize_maps(X);
    N = size(X, 1);
    centres = initialise_centres_plus_plus(X, K);
    labels = zeros(N, 1);
    prev_obj = Inf;
    for it = 1:max_iter
        sim = abs(X * centres');
        [~, labels] = max(sim, [], 2);
        centres = update_centres_by_pc(X, labels, centres, K);
        sim = abs(X * centres');
        maxsim = max(sim, [], 2);
        w = gfp(:).^2;
        obj = sum(w .* (1 - maxsim.^2));
        if isfinite(prev_obj) && abs(prev_obj - obj) <= tol * max(1, prev_obj)
            break;
        end
        prev_obj = obj;
    end
    metrics = compute_metrics(X, labels, centres, gfp, K);
    fit = struct('centers', centres, 'labels', labels, 'metrics', metrics, 'iterations', it);
end

function centres = initialise_centres_plus_plus(X, K)
    N = size(X, 1);
    centres = zeros(K, size(X, 2));
    first = randi(N);
    centres(1, :) = X(first, :);
    d2 = ones(N, 1);
    for k = 2:K
        sim = abs(X * centres(1:k-1, :)');
        d2 = min(d2, 1 - max(sim, [], 2).^2);
        d2(~isfinite(d2) | d2 < 0) = 0;
        if sum(d2) <= eps
            idx = randi(N);
        else
            cs = cumsum(d2 ./ sum(d2));
            idx = find(cs >= rand(), 1, 'first');
        end
        centres(k, :) = X(idx, :);
    end
    centres = normalize_maps(centres);
end

function centres = update_centres_by_pc(X, labels, old_centres, K)
    centres = old_centres;
    N = size(X, 1);
    for k = 1:K
        idx = find(labels == k);
        if isempty(idx)
            centres(k, :) = X(randi(N), :);
            continue;
        end
        Xk = X(idx, :);
        if size(Xk, 1) == 1
            c = Xk(1, :);
        else
            try
                [~, ~, V] = svd(Xk, 'econ');
                c = V(:, 1)';
            catch
                c = mean(Xk, 1);
            end
        end
        if dot(c, mean(Xk, 1)) < 0
            c = -c;
        end
        centres(k, :) = c;
    end
    centres = normalize_maps(centres);
end

function metrics = compute_metrics(X, labels, centres, gfp, K)
    X = normalize_maps(X);
    centres = normalize_maps(centres);
    sim = abs(X * centres');
    [maxsim, ~] = max(sim, [], 2);
    w = double(gfp(:)).^2;
    if numel(w) ~= size(X, 1) || sum(w) <= eps
        w = ones(size(X, 1), 1);
    end
    wss = sum(w .* (1 - maxsim.^2));
    gev = sum(w .* maxsim.^2) / sum(w);
    residual_variance = wss / sum(w);
    n_channels = size(X, 2);
    cv = residual_variance * ((n_channels - 1) / max(1, n_channels - K - 1))^2;
    sil = silhouette_centroid_approx(X, labels, centres);
    props = zeros(1, K);
    mean_corr = zeros(1, K);
    for k = 1:K
        idx = labels == k;
        props(k) = mean(idx);
        if any(idx)
            mean_corr(k) = mean(abs(X(idx, :) * centres(k, :)'));
        else
            mean_corr(k) = NaN;
        end
    end
    metrics = empty_metrics_struct();
    metrics.K = K;
    metrics.n_maps = size(X, 1);
    metrics.gev = gev;
    metrics.wss = wss;
    metrics.residual_variance = residual_variance;
    metrics.cross_validation = cv;
    metrics.silhouette = sil;
    metrics.mean_abs_corr = mean(maxsim);
    metrics.min_state_proportion = min(props);
    metrics.max_state_proportion = max(props);
    metrics.empty_states = sum(props == 0);
end

function s = empty_metrics_struct()
    s = struct('K', NaN, 'n_maps', NaN, 'gev', NaN, 'wss', NaN, 'residual_variance', NaN, ...
        'cross_validation', NaN, 'silhouette', NaN, 'mean_abs_corr', NaN, ...
        'min_state_proportion', NaN, 'max_state_proportion', NaN, 'empty_states', NaN);
end

function sil = silhouette_centroid_approx(X, labels, centres)
    K = size(centres, 1);
    if K < 2 || size(X, 1) < K + 1
        sil = NaN;
        return;
    end
    sim = abs(X * centres');
    dist = 1 - sim;
    N = size(X, 1);
    a = nan(N, 1);
    b = nan(N, 1);
    for i = 1:N
        own = labels(i);
        a(i) = dist(i, own);
        other = dist(i, :);
        other(own) = Inf;
        b(i) = min(other);
    end
    vals = (b - a) ./ max(a, b);
    vals(~isfinite(vals)) = NaN;
    sil = mean(vals, 'omitnan');
end

function [K_est, best_idx] = select_K(K_candidates, criterion, model_comparison)
    criterion = lower(char(criterion));
    switch criterion
        case {'gev', 'global_explained_variance'}
            [~, best_idx] = max(model_comparison.gev);
        case {'silhouette', 'sil'}
            [~, best_idx] = max(model_comparison.silhouette);
        case {'cv', 'cross_validation'}
            [~, best_idx] = min(model_comparison.cross_validation);
        case {'wss'}
            [~, best_idx] = min(model_comparison.wss);
        case {'elbow'}
            best_idx = elbow_index(model_comparison.wss);
        case {'elbow_sil_combined', 'combined'}
            elbow_scores = local_elbow_scores(model_comparison.wss, K_candidates(:));
            sil_norm = normalise01(model_comparison.silhouette);
            gev_norm = normalise01(model_comparison.gev);
            score = 0.55 * elbow_scores(:) + 0.30 * sil_norm(:) + 0.15 * gev_norm(:);
            if all(~isfinite(score)) || range(score(isfinite(score))) <= eps
                [~, best_idx] = max(model_comparison.gev);
            else
                [~, best_idx] = max(score);
            end
        otherwise
            warning('Unknown criterion %s. Falling back to elbow_sil_combined.', criterion);
            [K_est, best_idx] = select_K(K_candidates, 'elbow_sil_combined', model_comparison);
            return;
    end
    K_est = K_candidates(best_idx);
end

function idx = elbow_index(y)
    y = double(y(:));
    n = numel(y);
    if n <= 2
        [~, idx] = min(y);
        return;
    end
    x = (1:n)';
    yn = normalise01(y);
    xn = normalise01(x);
    p1 = [xn(1), yn(1)];
    p2 = [xn(end), yn(end)];
    d = zeros(n, 1);
    for i = 1:n
        p = [xn(i), yn(i)];
        d(i) = abs((p2(2)-p1(2))*p(1) - (p2(1)-p1(1))*p(2) + p2(1)*p1(2) - p2(2)*p1(1)) / ...
            sqrt((p2(2)-p1(2))^2 + (p2(1)-p1(1))^2 + eps);
    end
    d(1) = -Inf;
    d(end) = -Inf;
    [~, idx] = max(d);
    if isempty(idx) || ~isfinite(d(idx))
        [~, idx] = min(y);
    end
end

function s = local_elbow_scores(wss, K)
    wss = double(wss(:));
    K = double(K(:));
    n = numel(wss);
    s = zeros(n, 1);
    if n <= 2
        s(elbow_index(wss)) = 1;
        return;
    end
    yn = normalise01(wss);
    xn = normalise01(K);
    p1 = [xn(1), yn(1)];
    p2 = [xn(end), yn(end)];
    for i = 1:n
        p = [xn(i), yn(i)];
        s(i) = abs((p2(2)-p1(2))*p(1) - (p2(1)-p1(1))*p(2) + p2(1)*p1(2) - p2(2)*p1(1)) / ...
            sqrt((p2(2)-p1(2))^2 + (p2(1)-p1(1))^2 + eps);
    end
    s(1) = 0;
    s(end) = 0;
    s = normalise01(s);
end

function y = normalise01(x)
    x = double(x(:));
    y = nan(size(x));
    finite = isfinite(x);
    if ~any(finite)
        return;
    end
    xmin = min(x(finite));
    xmax = max(x(finite));
    if xmax <= xmin + eps
        y(finite) = 0;
    else
        y(finite) = (x(finite) - xmin) ./ (xmax - xmin);
    end
    y(~finite) = 0;
end

function Xn = normalize_maps(X)
    X = double(X);
    X = X - mean(X, 2, 'omitnan');
    denom = sqrt(sum(X.^2, 2));
    denom(~isfinite(denom) | denom <= eps) = 1;
    Xn = X ./ denom;
end

function labels = assign_by_abs_correlation(maps, centres)
    X = normalize_maps(double(maps));
    C = normalize_maps(double(centres));
    [~, labels] = max(abs(X * C'), [], 2);
end

% -------------------------------------------------------------------------
% Template alignment with true subset/permutation search
% -------------------------------------------------------------------------
function Fit = attach_template_alignment(Fit, template_file, common_labels, common_chanlocs, common_pos, cfg)
    alignment = align_maps_to_template_file(Fit.centers, template_file, common_labels, common_chanlocs, common_pos, cfg);
    if alignment.ok
        Fit.centers = alignment.aligned_maps;
        Fit.template_alignment = alignment;
        Fit.labels = remap_labels_after_reorder(Fit.labels, alignment.old_order_to_new_order);
    else
        Fit.template_alignment = alignment;
    end
end

function labels_new = remap_labels_after_reorder(labels_old, old_order_to_new_order)
    labels_new = labels_old;
    for old = 1:numel(old_order_to_new_order)
        labels_new(labels_old == old) = old_order_to_new_order(old);
    end
end

function alignment = align_maps_to_template_file(estimated_maps, template_file, common_labels, common_chanlocs, common_pos, cfg)
    alignment = struct('ok', false, 'message', '', 'template_file', template_file);
    if isempty(template_file) || ~isfile(template_file)
        alignment.message = sprintf('Template file not found: %s', template_file);
        return;
    end
    try
        [template_maps, template_labels, template_channel_labels, template_chanlocs] = load_template_maps(template_file);
        template_common = remap_template_to_common(template_maps, template_channel_labels, template_chanlocs, common_labels, common_pos);
        template_common = normalize_maps(template_common);
        estimated_maps = normalize_maps(estimated_maps);
        R = estimated_maps * template_common';
        score = abs(R);
        assignment = optimal_rectangular_assignment(score);
        K = size(estimated_maps, 1);
        assigned_template = assignment(:);
        signs = ones(K, 1);
        assigned_corr = nan(K, 1);
        for k = 1:K
            r = R(k, assigned_template(k));
            signs(k) = sign_nonzero(r);
            assigned_corr(k) = abs(r);
        end
        assigned_labels = template_labels(assigned_template);
        [~, new_order] = sort(assigned_template, 'ascend');
        old_order_to_new_order = zeros(K, 1);
        for new = 1:K
            old_order_to_new_order(new_order(new)) = new;
        end
        aligned_maps = estimated_maps(new_order, :);
        for new = 1:K
            old = new_order(new);
            aligned_maps(new, :) = signs(old) .* aligned_maps(new, :);
        end
        alignment.ok = true;
        alignment.message = 'ok';
        alignment.correlation_matrix = R;
        alignment.abs_correlation_matrix = score;
        alignment.assigned_template_index_original_order = assigned_template;
        alignment.assigned_labels_original_order = assigned_labels(:);
        alignment.assigned_abs_corr_original_order = assigned_corr(:);
        alignment.new_order_old_indices = new_order(:);
        alignment.old_order_to_new_order = old_order_to_new_order(:);
        alignment.assigned_template_index = assigned_template(new_order);
        alignment.assigned_labels = assigned_labels(new_order);
        alignment.assigned_abs_corr = assigned_corr(new_order);
        alignment.aligned_maps = aligned_maps;
        alignment.mean_abs_corr = mean(assigned_corr, 'omitnan');
        alignment.n_strong_matches = sum(assigned_corr >= cfg.strong_template_corr);
        alignment.strong_threshold = cfg.strong_template_corr;
        alignment.template_labels = template_labels(:);
    catch ME
        alignment.ok = false;
        alignment.message = ME.message;
    end
end

function [maps, labels, channel_labels, chanlocs] = load_template_maps(template_file)
    [~, ~, ext] = fileparts(template_file);
    ext = lower(ext);
    if strcmp(ext, '.set')
        [maps, labels, channel_labels, chanlocs] = load_metamaps_templates(template_file, 'K', 7);
        return;
    elseif strcmp(ext, '.mat')
        S = load(template_file);
        if isfield(S, 'template_maps')
            maps = double(S.template_maps);
        elseif isfield(S, 'maps')
            maps = double(S.maps);
        elseif isfield(S, 'centers')
            maps = double(S.centers);
        else
            error('Template .mat must contain template_maps, maps, or centers.');
        end
        if isfield(S, 'channel_labels')
            channel_labels = cellstr(S.channel_labels);
        else
            channel_labels = arrayfun(@(i) sprintf('Ch%d', i), 1:size(maps, 2), 'UniformOutput', false)';
        end
        if isfield(S, 'template_labels')
            labels = cellstr(S.template_labels);
        else
            labels = arrayfun(@(i) sprintf('T%02d', i), 1:size(maps, 1), 'UniformOutput', false)';
        end
        chanlocs = [];
        return;
    else
        error('Unsupported template format: %s', ext);
    end
    labels = arrayfun(@(i) sprintf('T%02d', i), 1:size(maps, 1), 'UniformOutput', false)';
    if size(maps, 1) == 7
        labels = {'A'; 'B'; 'C'; 'D'; 'E'; 'F'; 'G'};
    elseif size(maps, 1) == 4
        labels = {'A'; 'B'; 'C'; 'D'};
    end
end

function template_common = remap_template_to_common(template_maps, template_channel_labels, template_chanlocs, common_labels, common_pos)
    template_can = canonical_channel_labels(template_channel_labels);
    common_can = canonical_channel_labels(common_labels);
    idx = nan(numel(common_can), 1);
    for i = 1:numel(common_can)
        hit = find(strcmp(template_can, common_can{i}), 1, 'first');
        if ~isempty(hit)
            idx(i) = hit;
        end
    end
    if all(isfinite(idx))
        template_common = template_maps(:, idx);
        return;
    end
    if ~isempty(template_chanlocs) && ~isempty(common_pos)
        source_pos = pos_from_chanlocs(template_chanlocs, numel(template_channel_labels));
        if size(source_pos, 1) == size(template_maps, 2) && all(any(isfinite(source_pos), 2))
            template_common = interpolate_template_maps(template_maps, source_pos, common_pos);
            return;
        end
    end
    missing = common_labels(~isfinite(idx));
    error('Template cannot be matched to common channels. Missing examples: %s', strjoin(missing(1:min(10,end)), ', '));
end

function out = interpolate_template_maps(template_maps, source_pos, target_pos)
    source_pos = double(source_pos);
    target_pos = double(target_pos);
    ok_source = all(isfinite(source_pos), 2);
    source_pos = source_pos(ok_source, :);
    template_maps = template_maps(:, ok_source);
    out = nan(size(template_maps, 1), size(target_pos, 1));
    for t = 1:size(target_pos, 1)
        d = sqrt(sum((source_pos - target_pos(t, :)).^2, 2));
        [ds, ord] = sort(d, 'ascend');
        m = min(6, numel(ord));
        ord = ord(1:m);
        ds = ds(1:m);
        if ds(1) <= eps
            w = zeros(m, 1); w(1) = 1;
        else
            w = 1 ./ (ds.^2 + eps);
            w = w ./ sum(w);
        end
        out(:, t) = template_maps(:, ord) * w;
    end
end

function assignment = optimal_rectangular_assignment(score)
    % score is K_estimated x K_template. K_template may be larger; choose the
    % best unique subset/permutation. This is the critical fix for K=4 vs a
    % 7-state canonical template: no first-K assumption is made.
    [K, T] = size(score);
    if K > T
        warning('More estimated maps than template maps; falling back to greedy assignment with replacement.');
        [~, assignment] = max(score, [], 2);
        return;
    end
    if T <= 10 && K <= 8
        combos = nchoosek(1:T, K);
        best_score = -Inf;
        assignment = combos(1, :)';
        for c = 1:size(combos, 1)
            perms_c = perms(combos(c, :));
            for p = 1:size(perms_c, 1)
                a = perms_c(p, :)';
                s = sum(score(sub2ind(size(score), (1:K)', a)));
                if s > best_score
                    best_score = s;
                    assignment = a;
                end
            end
        end
    else
        assignment = nan(K, 1);
        available = true(1, T);
        for k = 1:K
            row = score(k, :);
            row(~available) = -Inf;
            [~, a] = max(row);
            assignment(k) = a;
            available(a) = false;
        end
    end
end

function s = sign_nonzero(x)
    if x < 0
        s = -1;
    else
        s = 1;
    end
end

% -------------------------------------------------------------------------
% Pooled GFP layer
% -------------------------------------------------------------------------
function [pooled_maps, pooled_gfp, pooled_rows, population_filter] = pool_cached_gfp_maps(records, manifest, cfg)
    pooled_maps = [];
    pooled_gfp = [];
    pooled_rows = table();
    for i = 1:numel(records)
        rec = records{i};
        if isempty(rec) || ~isfield(rec, 'maps_norm')
            continue;
        end
        n = size(rec.maps_norm, 1);
        pooled_maps = [pooled_maps; double(rec.maps_norm)]; %#ok<AGROW>
        if isfield(rec, 'gfp_effective')
            pooled_gfp = [pooled_gfp; double(rec.gfp_effective(:))]; %#ok<AGROW>
        else
            pooled_gfp = [pooled_gfp; double(rec.gfp(:))]; %#ok<AGROW>
        end
        rows = table();
        rows.file_index = repmat(i, n, 1);
        rows.participant = repmat(manifest.participant(i), n, 1);
        rows.condition = repmat(manifest.condition(i), n, 1);
        rows.group = repmat(manifest.group(i), n, 1);
        rows.file_path = repmat(manifest.file_path(i), n, 1);
        rows.peak_sample = rec.peak_sample(:);
        rows.gfp = double(rec.gfp(:));
        if isfield(rec, 'gfp_effective')
            rows.gfp_effective = double(rec.gfp_effective(:));
        else
            rows.gfp_effective = double(rec.gfp(:));
        end
        rows.n_target_channels = repmat(double(rec.n_target_channels), n, 1);
        rows.n_observed_channels = repmat(double(rec.n_observed_channels), n, 1);
        rows.n_interpolated_channels = repmat(double(rec.n_interpolated_channels), n, 1);
        rows.interpolated_channel_fraction = repmat(double(rec.interpolated_fraction), n, 1);
        pooled_rows = [pooled_rows; rows]; %#ok<AGROW>
    end
    population_filter = struct();
    population_filter.n_before = size(pooled_maps, 1);
    if isempty(pooled_maps)
        error('No cached GFP maps available for pooled fit.');
    end
    keep = all(isfinite(pooled_maps), 2) & isfinite(pooled_gfp) & pooled_gfp > 0;
    if cfg.reject_gfp_peak_outliers_population
        keep2 = robust_upper_outlier_mask(pooled_gfp, cfg.gfp_outlier_mad_multiplier_population, max(100, 10 * max(cfg.pooled_K_candidates)));
        keep = keep & keep2;
    end
    pooled_maps = pooled_maps(keep, :);
    pooled_gfp = pooled_gfp(keep);
    pooled_rows = pooled_rows(keep, :);
    population_filter.n_after_outlier_filter = size(pooled_maps, 1);
    population_filter.keep_fraction = population_filter.n_after_outlier_filter / max(1, population_filter.n_before);

    if ~isempty(cfg.max_global_maps) && size(pooled_maps, 1) > cfg.max_global_maps
        idx = deterministic_subsample(size(pooled_maps, 1), cfg.max_global_maps);
        pooled_maps = pooled_maps(idx, :);
        pooled_gfp = pooled_gfp(idx);
        pooled_rows = pooled_rows(idx, :);
        population_filter.n_after_global_cap = size(pooled_maps, 1);
    else
        population_filter.n_after_global_cap = size(pooled_maps, 1);
    end
end

% -------------------------------------------------------------------------
% Output helpers
% -------------------------------------------------------------------------
function write_matrix_csv(file_path, X)
    T = array2table(double(X));
    names = cell(1, size(X, 2));
    for i = 1:size(X, 2)
        names{i} = sprintf('ch_%03d', i);
    end
    T.Properties.VariableNames = names;
    writetable(T, file_path);
end

function plot_center_grid(centres, chanlocs, out_file, fig_title)
    try
        K = size(centres, 1);
        fig = figure('Visible', 'off', 'Color', 'white');
        n_cols = min(K, 5);
        n_rows = ceil(K / n_cols);
        for k = 1:K
            subplot(n_rows, n_cols, k);
            vals = centres(k, :);
            if ~isempty(chanlocs) && exist('topoplot', 'file') == 2
                try
                    topoplot(vals, chanlocs, 'electrodes', 'off', 'numcontour', 6);
                catch
                    imagesc(vals); axis tight; colorbar;
                end
            else
                imagesc(vals); axis tight; colorbar;
            end
            title(sprintf('State %d', k));
        end
        sgtitle(fig_title, 'Interpreter', 'none');
        saveas(fig, out_file);
        close(fig);
    catch ME
        warning('Could not create plot %s: %s', out_file, ME.message);
    end
end
