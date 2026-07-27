function T = simulated_ms_retrieval_experiment(varargin)
% BAYESIAN_MS_COMPARISON_PIPELINE: Valid method-criterion combinations only.
%
% The simulation enumerates only supported method/criterion pairs. For the
% current retrieval comparison:
%   - kmeans_koenig: silhouette
%   - spm_vb: silhouette, free_energy, log_likelihood, bic, icl,
%             free_energy_elbow, elbow_sil_combined, gev/gfp, calinski_harabasz_score, covariance,
%             covariance_elbow, free_energy_covariance
%
% Requirements:
% spm on path (development version at spm/spm on github)
% matlab (version R2025a used for this - other versions not tested)
% Dr Rohan Kandasamy 8-11-2025
% 
% Experimental data run on 25-6-2026
%
% Using code from Thomas Koenig (MICROSTATELAB group)
% (the "microstates" repository at: ThomasKoenigBern/microstates on github)

    setup_microstate_paths();
    util = microstate_utilities();
    repo_cfg = util.load_config();
    sim_defaults = repo_cfg.simulation;
    path_defaults = repo_cfg.paths;

    if exist(path_defaults.koenig_code_dir, "dir")
        addpath(path_defaults.koenig_code_dir);
    end
    % This is a folder which should contain:
    %  - eeg_kMeans.m
    %  - L2NormDim.m
    %  - mywaitbar.m
    %  - popFitMSMaps.m
    % All from the "microstates" repository

    test = false;

    p = inputParser;
    if test
        addParameter(p, 'out_dir', char(sim_defaults.out_dir), @ischar);
        addParameter(p, 'reps', 1, @isnumeric);
        addParameter(p, 'rep_vals', [], @isnumeric);
        addParameter(p, 'K_true_vals', [4], @isnumeric);
        addParameter(p, 'SNR_dbs', [10], @isnumeric);
        addParameter(p, 'K_candidates', 4:5, @isnumeric);
    else
        addParameter(p, 'out_dir', char(sim_defaults.out_dir), @ischar);
        addParameter(p, 'reps', double(sim_defaults.reps), @isnumeric);
        addParameter(p, 'rep_vals', [], @isnumeric);
        addParameter(p, 'K_true_vals', double(sim_defaults.K_true_vals(:)'), @isnumeric);
        addParameter(p, 'SNR_dbs', double(sim_defaults.SNR_dbs(:)'), @isnumeric);
        addParameter(p, 'K_candidates', double(sim_defaults.K_candidates(:)'), @isnumeric);
    end;
    addParameter(p, 'duration_s', double(sim_defaults.duration_s), @isnumeric);
    addParameter(p, 'sfreq', double(sim_defaults.sfreq), @isnumeric);
    addParameter(p, 'set_file', char(path_defaults.template_file), @ischar);
    addParameter(p, 'n_workers', double(sim_defaults.n_workers), @isnumeric);
    addParameter(p, 'cleanup', true, @islogical);
    addParameter(p, 'spm_path', '', @ischar); % optional: point to SPM toolbox/mixture if not obvious
    addParameter(p, 'verbose', true, @islogical);
    addParameter(p, 'resume', true, @islogical);
    addParameter(p, 'save_json', true, @islogical);
    addParameter(p, 'save_k_candidate_metrics', true, @islogical);
    addParameter(p, 'montages', cellstr(string(sim_defaults.montages)), @iscell);  % montage robustness analysis
    addParameter(p, 'overlap_probs', double(sim_defaults.overlap_probs(:)'), @isnumeric); % run with and without overlap
    addParameter(p, 'overlap_ms_range', double(sim_defaults.overlap_ms_range(:)'), @isnumeric);
    addParameter(p, 'overlap_strength', double(sim_defaults.overlap_strength), @isnumeric);
    addParameter(p, 'compute_backfit_diagnostics', logical(util.get_field(sim_defaults, 'compute_backfit_diagnostics', false)), @islogical);
    addParameter(p, 'save_backfit_details', logical(util.get_field(sim_defaults, 'save_backfit_details', false)), @islogical);
    addParameter(p, 'backfit_downsample_factor', double(util.get_field(sim_defaults, 'backfit_downsample_factor', 5)), ...
        @(x) isnumeric(x) && isscalar(x) && x >= 1);
    addParameter(p, 'template_alignment_strong_threshold', double(util.get_field(sim_defaults, 'template_alignment_strong_threshold', 0.5)), @isnumeric);
    addParameter(p, 'ecological_profile', logical(util.get_field(sim_defaults, 'ecological_profile', true)), @islogical);
    addParameter(p, 'randomize_true_templates', logical(util.get_field(sim_defaults, 'randomize_true_templates', true)), @islogical);
    addParameter(p, 'true_template_pool_K', double(util.get_field(sim_defaults, 'true_template_pool_K', 7)), @isnumeric);
    addParameter(p, 'clean_sanity_profile', logical(util.get_field(sim_defaults, 'clean_sanity_profile', true)), @islogical);
    addParameter(p, 'clean_sanity_snr_db_threshold', double(util.get_field(sim_defaults, 'clean_sanity_snr_db_threshold', 40)), @isnumeric);
    addParameter(p, 'validate_simulation', logical(sim_defaults.validate_simulation), @islogical);
    addParameter(p, 'preprocessing', sim_defaults.preprocessing, @isstruct);
    addParameter(p, 'methods', {'spm_vb', 'kmeans_koenig'}, @(x) iscell(x) || isstring(x));
    addParameter(p, 'criteria', {'silhouette', 'free_energy', 'log_likelihood', 'bic', 'icl', 'free_energy_elbow', 'gev', 'calinski_harabasz_score', 'covariance', 'covariance_elbow', 'elbow_sil_combined', 'free_energy_covariance'}, @(x) iscell(x) || isstring(x));
    parse(p, varargin{:});
    
    CONFIG = p.Results;
    CONFIG.out_dir = util.resolve_path(CONFIG.out_dir, util.project_root());
    CONFIG.set_file = util.resolve_path(CONFIG.set_file, util.project_root());
    CONFIG.methods = cellstr(string(CONFIG.methods));
    CONFIG.criteria = cellstr(string(CONFIG.criteria));
    if isempty(CONFIG.rep_vals)
        rep_vals = 1:CONFIG.reps;
    else
        rep_vals = CONFIG.rep_vals(:)';
        CONFIG.reps = numel(rep_vals);
    end

    % Load channel information
    if CONFIG.verbose
        fprintf('Loading channel information from: %s\n', CONFIG.set_file);
    end
    [ch_labels, montage_pos] = load_eeg_montage(CONFIG.set_file);
    n_channels = length(ch_labels);
    if CONFIG.verbose
        fprintf('✓ Loaded %d channels\n', n_channels);
    end
    % Setup directories
    if ~exist(CONFIG.out_dir, 'dir'), mkdir(CONFIG.out_dir); end
    res_dir = fullfile(CONFIG.out_dir, 'results');
    if ~exist(res_dir, 'dir'), mkdir(res_dir); end
    checkpoint_dir = fullfile(res_dir, 'checkpoints');
    if ~exist(checkpoint_dir, 'dir'), mkdir(checkpoint_dir); end
    json_dir = fullfile(CONFIG.out_dir, 'microstates_json');
    if CONFIG.save_json && ~exist(json_dir, 'dir'), mkdir(json_dir); end
    backfit_dir = fullfile(res_dir, 'backfit_diagnostics');
    if CONFIG.compute_backfit_diagnostics && CONFIG.save_backfit_details && ~exist(backfit_dir, 'dir'), mkdir(backfit_dir); end
    confusion_dir = fullfile(res_dir, 'backfit_confusions');
    plots_dir = fullfile(CONFIG.out_dir, 'plots');
    if ~exist(plots_dir, 'dir'), mkdir(plots_dir); end

    if CONFIG.verbose
        fprintf('\n========================================\n');
        fprintf('Microstate Comparison Pipeline\n');
        fprintf('ALL METHOD × CRITERION COMBINATIONS\n');
        fprintf('========================================\n');
        fprintf('Output: %s\n', CONFIG.out_dir);
        fprintf('Reps: %s | K true: %s | SNR: %s\n', ...
            mat2str(rep_vals), mat2str(CONFIG.K_true_vals), mat2str(CONFIG.SNR_dbs));
        fprintf('Channels: %d | Workers: %d\n', n_channels, CONFIG.n_workers);
        fprintf('Montages: %s\n', strjoin(CONFIG.montages, ', '));
        fprintf('Overlap probs: %s (ms range %s, strength %.2f)\n\n', ...
            mat2str(CONFIG.overlap_probs), mat2str(CONFIG.overlap_ms_range), CONFIG.overlap_strength);
    end

    % ===== Valid method/criterion design =====
    [method_names, method_criteria_map, all_criteria, n_method_criteria_pairs] = ...
        build_method_criteria_design(CONFIG.methods, CONFIG.criteria);
    
    % Guard: if SPM methods requested but SPM not available, stop early with a clear message
    needs_spm = any(contains(method_names, 'spm'));
    [spm_ok, spm_info] = util.ensure_spm_mix(CONFIG.spm_path, path_defaults.spm_mixture_paths, CONFIG.verbose);
    if CONFIG.verbose
        fprintf('SPM root detected: %s\n', local_which_text('spm'));
        fprintf('SPM mixture detected: %s\n', local_which_text('spm_mix'));
    end
    if needs_spm && ~spm_ok
        error(['SPM mixture toolbox not found on MATLAB path. ', ...
               'Checked explicit spm_path, SPM_PATH/SPM_MIXTURE_PATH, configured paths, and the folder inferred from spm.m. ', ...
               'spm.m=''%s''; spm_mix=''%s''; attempted={%s}.'], ...
               local_which_text('spm'), local_which_text('spm_mix'), strjoin(spm_info.attempted, ', '));
    end
    
    % Montages to test
    n_montages = length(CONFIG.montages);
    overlap_probs = CONFIG.overlap_probs;
    n_overlap_conditions = length(overlap_probs);
    
    % Build list of EEG conditions
    eeg_conditions = [];
    for rep = rep_vals
        for K_true = CONFIG.K_true_vals
            for SNR_dB = CONFIG.SNR_dbs
                for ov = 1:n_overlap_conditions
                    overlap_prob = overlap_probs(ov);
                    eeg_conditions = [eeg_conditions; rep, K_true, SNR_dB, overlap_prob]; %#ok<AGROW>
                end
            end
        end
    end
    
    n_eeg_conditions = size(eeg_conditions, 1);
    n_methods = length(method_names);
    n_fits = n_eeg_conditions * n_montages * n_methods;
    n_total_results = n_eeg_conditions * n_montages * n_method_criteria_pairs;
    
    if CONFIG.verbose
        fprintf('PIPELINE STRUCTURE:\n');
        fprintf('  EEG conditions: %d (includes %d overlap settings)\n', n_eeg_conditions, n_overlap_conditions);
        fprintf('  Montages: %d\n', n_montages);
        fprintf('  Methods: %d\n', n_methods);
        fprintf('  Supported method-criterion pairs: %d\n', n_method_criteria_pairs);
        fprintf('  Total FITS: %d (one per montage+method+EEG)\n', n_fits);
        fprintf('  Total RESULTS: %d (fits × valid criteria)\n', n_total_results);
        for m_idx = 1:n_methods
            method_str = method_names{m_idx};
            criteria_this = method_criteria_map(method_str);
            fprintf('    %s -> %s\n', method_str, strjoin(criteria_this, ', '));
        end
        fprintf('\n');
    end

    % ===== UNIFIED PIPELINE: GENERATE → FIT → CRITERIA → SAVE (per EEG) =====
    if CONFIG.verbose
        fprintf('Pipeline: Generate EEG → Montage → Fit Methods → Apply Criteria → Save\n');
        fprintf('Processing %d EEG conditions with %d montages, %d methods and %d valid method-criterion pairs...\n', ...
            n_eeg_conditions, n_montages, n_methods, n_method_criteria_pairs);
    end
    
    rows = [];
    run_id = 0;
    n_failed = 0;
    n_successful_fits = 0;
    error_types = containers.Map();
    fit_id = 0;
    
    % Initialize parallel pool if needed (will be used within EEG loop for method fitting)
    try
        p_pool = gcp('nocreate');
        if isempty(p_pool)
            % Check available cores and set pool size
            max_workers = maxNumCompThreads();
            n_workers = min(CONFIG.n_workers, max_workers);
            fprintf('Starting parallel pool with %d workers (max available: %d)\n', n_workers, max_workers);
            parpool('local', n_workers);
        else
            fprintf('Using existing parallel pool with %d workers\n', p_pool.NumWorkers);
        end
    catch ME
        fprintf('Warning: Could not start parallel pool: %s\n', ME.message);
        fprintf('Falling back to serial execution\n');
    end
    
    % Prepare for parallel execution
    all_results = cell(n_eeg_conditions, 1);
    all_k_candidate_results = cell(n_eeg_conditions, 1);
    expected_rows_per_condition = n_montages * n_method_criteria_pairs;
    checkpoint_config = rmfield(CONFIG, {'verbose', 'resume', 'cleanup', 'save_json', ...
        'save_k_candidate_metrics', 'n_workers', 'spm_path'});
    checkpoint_files = cell(n_eeg_conditions, 1);
    for i = 1:n_eeg_conditions
        cond = eeg_conditions(i, :);
        checkpoint_files{i} = fullfile(checkpoint_dir, sprintf('condition_rep%d_K%d_SNR%s_ovl%s.mat', ...
            round(cond(1)), round(cond(2)), signed_token(cond(3)), signed_token(cond(4))));
    end
    if CONFIG.verbose && CONFIG.resume
        fprintf('Resume checkpoints available: %d / %d\n', nnz(cellfun(@isfile, checkpoint_files)), n_eeg_conditions);
    end

    % We cannot use the progress bar object inside parfor easily
    fprintf('Starting parallel processing of %d EEG conditions...\n', n_eeg_conditions);
    
    parfor eeg_idx = 1:n_eeg_conditions
        % Local variables for this iteration
        local_rows = [];
        local_k_candidate_rows = [];
        
        rep = eeg_conditions(eeg_idx, 1);
        K_true = eeg_conditions(eeg_idx, 2);
        SNR_dB = eeg_conditions(eeg_idx, 3);
        overlap_prob = eeg_conditions(eeg_idx, 4);
        checkpoint_file = checkpoint_files{eeg_idx};
        if CONFIG.resume && isfile(checkpoint_file)
            try
                checkpoint = load(checkpoint_file, 'local_rows', 'local_k_candidate_rows', 'checkpoint_config');
                if isfield(checkpoint, 'local_rows') && ...
                        isfield(checkpoint, 'checkpoint_config') && ...
                        isequaln(checkpoint.checkpoint_config, checkpoint_config) && ...
                        numel(checkpoint.local_rows) >= expected_rows_per_condition
                    all_results{eeg_idx} = checkpoint.local_rows;
                    if isfield(checkpoint, 'local_k_candidate_rows')
                        all_k_candidate_results{eeg_idx} = checkpoint.local_k_candidate_rows;
                    else
                        all_k_candidate_results{eeg_idx} = [];
                    end
                    if CONFIG.verbose
                        fprintf('Resumed EEG condition %d from %s\n', eeg_idx, checkpoint_file);
                    end
                    continue;
                end
            catch ME
                if CONFIG.verbose
                    fprintf('Warning: Could not load checkpoint %s: %s\n', checkpoint_file, ME.message);
                end
            end
        end

        % ===== STEP 1: GENERATE one EEG (full montage, 71 channels) =====
        try
            sim_seed = 42 + rep*1000 + K_true*100 + round((SNR_dB+10)*10);
            sim_opts = build_simulation_options(CONFIG, overlap_prob, SNR_dB, rep, K_true);
            [Sim, ~, ~] = generate_microstate_eeg(K_true, SNR_dB, CONFIG.duration_s, ...
                CONFIG.sfreq, sim_seed, sim_opts);
            if CONFIG.validate_simulation && exist('validate_simulated_eeg_representativeness', 'file') == 2
                Sim.sim_qc = validate_simulated_eeg_representativeness(Sim, 'verbose', false);
            elseif CONFIG.validate_simulation && CONFIG.verbose
                fprintf('\nSimulation QC requested but validate_simulated_eeg_representativeness.m is not on the path; skipping QC.\n');
            end
        catch ME
            if CONFIG.verbose
                fprintf('\n✗ EEG Generation Error (Rep %d, K=%d, SNR=%+.1f): %s\n', ...
                    rep, K_true, SNR_dB, ME.message);
            end
            continue;
        end
        
        % Store full Sim before montage reduction
        Sim_full = Sim;
        
        % ===== STEP 2: LOOP over MONTAGES =====
        for montage_idx = 1:n_montages
            montage_type = CONFIG.montages{montage_idx};
            
            % Apply montage selection
            if strcmp(montage_type, 'full')
                % Use full montage
                Sim = Sim_full;
                Sim.n_channels = size(Sim_full.X_noisy, 1);
            else
                % Reduce to specific montage
                [EEG_reduced, pos_reduced, chanlocs_reduced, labels_reduced, ~] = ...
                    select_montage_subset(Sim_full.X_noisy, Sim_full.pos, ...
                    Sim_full.chanlocs, ch_labels, montage_type);
                
                % Also reduce clean signal and true maps
                [X_clean_reduced, ~, ~, ~, ~] = ...
                    select_montage_subset(Sim_full.X_clean, Sim_full.pos, ...
                    Sim_full.chanlocs, ch_labels, montage_type);
                
                [maps_true_reduced, ~, ~, ~, ~] = ...
                    select_montage_subset(Sim_full.maps_true, Sim_full.pos, ...
                    Sim_full.chanlocs, ch_labels, montage_type);
                
                % Create reduced Sim structure
                Sim = Sim_full;
                Sim.X_noisy = EEG_reduced;
                Sim.X_clean = X_clean_reduced;
                Sim.pos = pos_reduced;
                Sim.chanlocs = chanlocs_reduced;
                Sim.channel_labels = labels_reduced;
                Sim.maps_true = maps_true_reduced;
                Sim.n_channels = length(labels_reduced);
            end
            
            % Add montage info to Sim
            Sim.montage_type = montage_type;
            Sim.n_leads = Sim.n_channels;
            Sim.preprocessing = CONFIG.preprocessing;
            if CONFIG.compute_backfit_diagnostics
                try
                    Sim.true_template_alignment = align_microstates_to_template(Sim.maps_true, CONFIG.set_file, ...
                        'estimated_channel_labels', Sim.channel_labels, ...
                        'strong_threshold', CONFIG.template_alignment_strong_threshold);
                catch
                    Sim.true_template_alignment = [];
                end
            end
            
            % ===== STEP 3: FIT all methods to this montage =====
            fit_results_for_montage = cell(n_methods, 1);
            
            for m_idx = 1:n_methods
                method_str = method_names{m_idx};
                
                try
                    % Call appropriate fitting function with a default criterion
                    % (Results will be reused with all criteria via select_K_by_criterion)
                    if strcmp(method_str, 'spm_vb')
                        Results = fit_microstate_spm_vb(Sim, CONFIG.K_candidates, 'elbow_sil_combined');
                    elseif strcmp(method_str, 'kmeans_koenig')
                        Results = fit_microstate_kmeans_koenig(Sim, CONFIG.K_candidates, 'silhouette');
                    elseif strcmp(method_str, 'spm_kmeans')
                        Results = fit_microstate_spm_kmeans(Sim, CONFIG.K_candidates, 'silhouette');
                    elseif strcmp(method_str, 'vb_kmeans')
                        Results = fit_microstate_vb_kmeans(Sim, CONFIG.K_candidates, 'free_energy');
                    elseif strcmp(method_str, 'dp_mixture')
                        Results = fit_microstate_dp_mixture(Sim, CONFIG.K_candidates, 'free_energy');
                    else
                        Results = [];
                    end
                    
                    if ~isempty(Results) && isfield(Results, 'valid_fit') && Results.valid_fit
                        nan_centers = isfield(Results, 'centers') && any(isnan(Results.centers(:)));
                        nan_scores = (isfield(Results, 'free_energy_vals') && any(isnan(Results.free_energy_vals(:)))) || ...
                                     (isfield(Results, 'silhouette_vals') && any(isnan(Results.silhouette_vals(:))));
                        if nan_centers || nan_scores
                            fprintf('Fit Warning (%s, montage %s): NaNs detected (centers=%d, scores=%d)\n', ...
                                method_str, montage_type, nan_centers, nan_scores);
                        end
                        % Store ONLY essential metadata, NOT full Results or Sim
                        % (Sim will be accessible in the save step, Results is available above)
                        fit_results_for_montage{m_idx} = struct(...
                            'eeg_idx', eeg_idx, ...
                            'rep', rep, ...
                            'K_true', K_true, ...
                            'SNR_dB', SNR_dB, ...
                            'method', method_str, ...
                            'montage_type', montage_type, ...
                            'n_leads', Sim.n_leads, ...
                            'Results', Results);
                    else
                        fit_results_for_montage{m_idx} = [];
                        fprintf('Fit Warning (%s, montage %s): invalid or failed fit (valid_fit missing/false)\n', method_str, montage_type);
                    end
                    
                catch ME
                    fit_results_for_montage{m_idx} = [];
                    fprintf('Fit Error (%s, montage %s): %s\n', method_str, montage_type, ME.message);
                end
            end
            
            % ===== STEP 4 & 5: APPLY CRITERIA and SAVE (for each fit) =====
            for m_idx = 1:n_methods
                if isempty(fit_results_for_montage{m_idx})
                    continue;
                end
                
                fit_result = fit_results_for_montage{m_idx};
                Results = fit_result.Results;
                method_str = char(string(fit_result.method));
                selected_solution_cache = struct();
                candidate_rows = build_k_candidate_metric_rows( ...
                    eeg_idx, m_idx, fit_result, Results, Sim, montage_type);
                local_k_candidate_rows = [local_k_candidate_rows; candidate_rows]; %#ok<AGROW>
                
                criteria_for_method = method_criteria_map(method_str);
                for c_idx = 1:numel(criteria_for_method)
                    criterion_str = char(string(criteria_for_method{c_idx}));
                    
                    % Extract K using this criterion
                    K_selected = select_K_by_criterion(Results, criterion_str);
                    
                    if isnan(K_selected)
                        continue;
                    end
                    
                    cache_key = sprintf('K_%d', K_selected);
                    if isfield(selected_solution_cache, cache_key)
                        SelectedResults = selected_solution_cache.(cache_key);
                    else
                        SelectedResults = extract_microstate_solution_for_k(Results, K_selected, ...
                            'Sim', Sim, ...
                            'criterion', criterion_str, ...
                            'template_file', CONFIG.set_file, ...
                            'estimated_channel_labels', Sim.channel_labels, ...
                            'strong_threshold', CONFIG.template_alignment_strong_threshold);
                        if CONFIG.compute_backfit_diagnostics
                            try
                                BackfitDiagnostics = compute_simulation_backfit_diagnostics(Sim, SelectedResults, CONFIG.set_file, ...
                                    'strong_threshold', CONFIG.template_alignment_strong_threshold, ...
                                    'downsample_factor', CONFIG.backfit_downsample_factor);
                                SelectedResults.backfit_diagnostics = BackfitDiagnostics;
                            catch
                            end
                        end
                        selected_solution_cache.(cache_key) = SelectedResults;
                    end
                    SelectedResults.criterion = criterion_str;
                    SelectedResults.best_criterion_value = criterion_specific_best_score(Results, K_selected, criterion_str);
                    rec_metrics = SelectedResults.recovery_metrics;
                    recovery_corr = util.padded_vector(rec_metrics.match_similarities, 10);
                    json_file = '';
                    backfit_diagnostic_file = '';
                    backfit_coverage_corr = NaN;
                    backfit_coverage_spearman = NaN;
                    backfit_coverage_mae = NaN;
                    backfit_coverage_rmse = NaN;
                    backfit_coverage_l1 = NaN;
                    backfit_overlap_fraction = NaN;
                    backfit_downsample_factor = CONFIG.backfit_downsample_factor;
                    backfit_n_samples = NaN;
                    backfit_n_samples_original = NaN;
                    backfit_hard_cluster_top1_accuracy = NaN;
                    backfit_hard_label_top1_accuracy = NaN;
                    backfit_hard_cluster_top1_accuracy_overlap = NaN;
                    backfit_hard_label_top1_accuracy_overlap = NaN;
                    backfit_hard_label_weight_mae = NaN;
                    backfit_hard_label_weight_mae_overlap = NaN;
                    backfit_mix_available = false;
                    backfit_mix_cluster_top1_accuracy = NaN;
                    backfit_mix_label_top1_accuracy = NaN;
                    backfit_mix_cluster_top1_accuracy_overlap = NaN;
                    backfit_mix_label_top1_accuracy_overlap = NaN;
                    backfit_mix_label_weight_mae = NaN;
                    backfit_mix_label_weight_mae_overlap = NaN;
                    cluster_identity_accuracy = NaN;
                    cluster_identity_accuracy_matched = NaN;
                    cluster_n_label_matches = NaN;
                    cluster_mean_matched_similarity = NaN;
                    backfit_hard_cluster_pair_accuracy_overlap = NaN;
                    backfit_hard_label_pair_accuracy_overlap = NaN;
                    backfit_mix_cluster_pair_accuracy_overlap = NaN;
                    backfit_mix_label_pair_accuracy_overlap = NaN;
                    selected_cov_trace_mean = NaN;
                    selected_cov_trace_median = NaN;
                    selected_cov_logdet_mean = NaN;
                    selected_spm_feature_dim = NaN;
                    if isfield(SelectedResults, 'selected_spm_mix_model') && ~isempty(SelectedResults.selected_spm_mix_model)
                        spm_model = SelectedResults.selected_spm_mix_model;
                        if isfield(spm_model, 'covariance_traces') && ~isempty(spm_model.covariance_traces)
                            selected_cov_trace_mean = mean(spm_model.covariance_traces, 'omitnan');
                            selected_cov_trace_median = median(spm_model.covariance_traces, 'omitnan');
                        end
                        if isfield(spm_model, 'covariance_logdets') && ~isempty(spm_model.covariance_logdets)
                            selected_cov_logdet_mean = mean(spm_model.covariance_logdets, 'omitnan');
                        end
                        if isfield(spm_model, 'feature_dim') && ~isempty(spm_model.feature_dim)
                            selected_spm_feature_dim = spm_model.feature_dim;
                        end
                    end
                    
                    % Generate a temporary subject name (IDs will be fixed later)
                    montage_name = char(string(montage_type));
                    overlap_tag = sprintf('ovl%02d', round(100 * Sim.overlap_prob));
                    subj_name = sprintf('fit_E%d_M%d_K%d_SNR%+d_%s_%s_%s_%s', ...
                        eeg_idx, m_idx, fit_result.K_true, round(fit_result.SNR_dB), ...
                        montage_name, overlap_tag, method_str, criterion_str);
                    if CONFIG.compute_backfit_diagnostics && isfield(SelectedResults, 'backfit_diagnostics')
                        BackfitDiagnostics = SelectedResults.backfit_diagnostics;
                        if isfield(BackfitDiagnostics, 'ok') && BackfitDiagnostics.ok
                            backfit_coverage_corr = BackfitDiagnostics.coverage_corr;
                            backfit_coverage_spearman = BackfitDiagnostics.coverage_spearman;
                            backfit_coverage_mae = BackfitDiagnostics.coverage_mae;
                            backfit_coverage_rmse = BackfitDiagnostics.coverage_rmse;
                            backfit_coverage_l1 = BackfitDiagnostics.coverage_l1;
                            if isfield(BackfitDiagnostics, 'overlap_sample_fraction')
                                backfit_overlap_fraction = BackfitDiagnostics.overlap_sample_fraction;
                            end
                            if isfield(BackfitDiagnostics, 'downsample_factor')
                                backfit_downsample_factor = BackfitDiagnostics.downsample_factor;
                            end
                            if isfield(BackfitDiagnostics, 'n_samples')
                                backfit_n_samples = BackfitDiagnostics.n_samples;
                            end
                            if isfield(BackfitDiagnostics, 'n_samples_original')
                                backfit_n_samples_original = BackfitDiagnostics.n_samples_original;
                            end
                            if isfield(BackfitDiagnostics, 'hard') && isstruct(BackfitDiagnostics.hard)
                                backfit_hard_cluster_top1_accuracy = BackfitDiagnostics.hard.cluster_top1_accuracy;
                                backfit_hard_label_top1_accuracy = BackfitDiagnostics.hard.label_top1_accuracy;
                                backfit_hard_cluster_top1_accuracy_overlap = BackfitDiagnostics.hard.cluster_top1_accuracy_overlap;
                                backfit_hard_label_top1_accuracy_overlap = BackfitDiagnostics.hard.label_top1_accuracy_overlap;
                                backfit_hard_label_weight_mae = BackfitDiagnostics.hard.label_weight_mae;
                                backfit_hard_label_weight_mae_overlap = BackfitDiagnostics.hard.label_weight_mae_overlap;
                                if isfield(BackfitDiagnostics.hard, 'cluster_pair_accuracy_overlap')
                                    backfit_hard_cluster_pair_accuracy_overlap = BackfitDiagnostics.hard.cluster_pair_accuracy_overlap;
                                    backfit_hard_label_pair_accuracy_overlap = BackfitDiagnostics.hard.label_pair_accuracy_overlap;
                                end
                            end
                            if isfield(BackfitDiagnostics, 'mixture') && isstruct(BackfitDiagnostics.mixture)
                                backfit_mix_available = isfield(BackfitDiagnostics.mixture, 'available') && BackfitDiagnostics.mixture.available;
                                backfit_mix_cluster_top1_accuracy = BackfitDiagnostics.mixture.cluster_top1_accuracy;
                                backfit_mix_label_top1_accuracy = BackfitDiagnostics.mixture.label_top1_accuracy;
                                backfit_mix_cluster_top1_accuracy_overlap = BackfitDiagnostics.mixture.cluster_top1_accuracy_overlap;
                                backfit_mix_label_top1_accuracy_overlap = BackfitDiagnostics.mixture.label_top1_accuracy_overlap;
                                backfit_mix_label_weight_mae = BackfitDiagnostics.mixture.label_weight_mae;
                                backfit_mix_label_weight_mae_overlap = BackfitDiagnostics.mixture.label_weight_mae_overlap;
                                if isfield(BackfitDiagnostics.mixture, 'cluster_pair_accuracy_overlap')
                                    backfit_mix_cluster_pair_accuracy_overlap = BackfitDiagnostics.mixture.cluster_pair_accuracy_overlap;
                                    backfit_mix_label_pair_accuracy_overlap = BackfitDiagnostics.mixture.label_pair_accuracy_overlap;
                                end
                            end
                            if isfield(BackfitDiagnostics, 'cluster') && isstruct(BackfitDiagnostics.cluster)
                                cluster_identity_accuracy = BackfitDiagnostics.cluster.identity_accuracy;
                                cluster_identity_accuracy_matched = BackfitDiagnostics.cluster.identity_accuracy_matched;
                                cluster_n_label_matches = BackfitDiagnostics.cluster.n_label_matches;
                                cluster_mean_matched_similarity = BackfitDiagnostics.cluster.mean_matched_similarity;
                            end
                            if CONFIG.save_backfit_details
                                backfit_diagnostic_file = fullfile(backfit_dir, [subj_name '_backfit.mat']);
                                backfit_payload = struct('BackfitDiagnostics', BackfitDiagnostics, ...
                                    'method_str', method_str, ...
                                    'criterion_str', criterion_str, ...
                                    'K_selected', K_selected, ...
                                    'subj_name', subj_name);
                                save(backfit_diagnostic_file, '-fromstruct', backfit_payload, '-v7.3');
                            end
                        end
                    end
                    sim_qc_status = 'NOT_RUN';
                    sim_qc_psd_slope = NaN;
                    sim_qc_mean_dwell_ms = NaN;
                    sim_qc_gfp_median_uv = NaN;
                    if isfield(Sim, 'sim_qc') && ~isempty(Sim.sim_qc)
                        sim_qc_status = Sim.sim_qc.overall_status;
                        sim_qc_psd_slope = Sim.sim_qc.spectrum.power_slope_2_40hz;
                        sim_qc_gfp_median_uv = Sim.sim_qc.amplitude.gfp_median_uv;
                        if isfield(Sim.sim_qc.microstate_dynamics, 'mean_dwell_ms')
                            sim_qc_mean_dwell_ms = Sim.sim_qc.microstate_dynamics.mean_dwell_ms;
                        end
                    end
                    true_template_labels_str = '';
                    true_template_indices_str = '';
                    if isfield(Sim, 'true_template_labels') && ~isempty(Sim.true_template_labels)
                        true_template_labels_str = strjoin(cellstr(string(Sim.true_template_labels)), '|');
                    end
                    if isfield(Sim, 'true_template_indices') && ~isempty(Sim.true_template_indices)
                        true_template_indices_str = mat2str(double(Sim.true_template_indices));
                    end

                    result_row = struct( ...
                        'fit_id', 0, ... % Placeholder
                        'subject', subj_name, ...
                        'rep', fit_result.rep, ...
                        'method', method_str, ...
                        'criterion', criterion_str, ...
                        'K_true', fit_result.K_true, ...
                        'SNR_dB', fit_result.SNR_dB, ...
                        'overlap_prob', Sim.overlap_prob, ...
                        'overlap_strength', Sim.overlap_strength, ...
                        'overlap_ms_min', Sim.overlap_ms_range(1), ...
                        'overlap_ms_max', Sim.overlap_ms_range(end), ...
                        'true_template_labels', true_template_labels_str, ...
                        'true_template_indices', true_template_indices_str, ...
                        'montage_type', montage_name, ...
                        'n_leads', fit_result.n_leads, ...
                        'sim_qc_status', sim_qc_status, ...
                        'sim_qc_psd_slope_2_40hz', sim_qc_psd_slope, ...
                        'sim_qc_gfp_median_uv', sim_qc_gfp_median_uv, ...
                        'sim_qc_mean_dwell_ms', sim_qc_mean_dwell_ms, ...
                        'K_estimated', K_selected, ...
                        'K_gap', fit_result.K_true - K_selected, ...
                        'K_correct', fit_result.K_true == K_selected, ...
                        'K_error', abs(fit_result.K_true - K_selected), ...
                        'n_maps', Results.n_maps, ...
                        'n_matched', rec_metrics.n_matched, ...
                        'mean_recovery_matched', rec_metrics.mean_recovery_matched, ...
                        'mean_recovery_padded', rec_metrics.mean_recovery_padded, ...
                        'sensitivity', rec_metrics.sensitivity, ...
                        'precision', rec_metrics.precision, ...
                        'f1_score', rec_metrics.f1_score, ...
                        'recovery_01', recovery_corr(1), ...
                        'recovery_02', recovery_corr(2), ...
                        'recovery_03', recovery_corr(3), ...
                        'cluster_identity_accuracy', cluster_identity_accuracy, ...
                        'cluster_identity_accuracy_matched', cluster_identity_accuracy_matched, ...
                        'cluster_n_label_matches', cluster_n_label_matches, ...
                        'cluster_mean_matched_similarity', cluster_mean_matched_similarity, ...
                        'selected_spm_cov_trace_mean', selected_cov_trace_mean, ...
                        'selected_spm_cov_trace_median', selected_cov_trace_median, ...
                        'selected_spm_cov_logdet_mean', selected_cov_logdet_mean, ...
                        'selected_spm_feature_dim', selected_spm_feature_dim, ...
                        'backfit_coverage_corr', backfit_coverage_corr, ...
                        'backfit_coverage_spearman', backfit_coverage_spearman, ...
                        'backfit_coverage_mae', backfit_coverage_mae, ...
                        'backfit_coverage_rmse', backfit_coverage_rmse, ...
                        'backfit_coverage_l1', backfit_coverage_l1, ...
                        'backfit_overlap_fraction', backfit_overlap_fraction, ...
                        'backfit_downsample_factor', backfit_downsample_factor, ...
                        'backfit_n_samples', backfit_n_samples, ...
                        'backfit_n_samples_original', backfit_n_samples_original, ...
                        'backfit_hard_cluster_top1_accuracy', backfit_hard_cluster_top1_accuracy, ...
                        'backfit_hard_label_top1_accuracy', backfit_hard_label_top1_accuracy, ...
                        'backfit_hard_cluster_top1_accuracy_overlap', backfit_hard_cluster_top1_accuracy_overlap, ...
                        'backfit_hard_label_top1_accuracy_overlap', backfit_hard_label_top1_accuracy_overlap, ...
                        'backfit_hard_label_weight_mae', backfit_hard_label_weight_mae, ...
                        'backfit_hard_label_weight_mae_overlap', backfit_hard_label_weight_mae_overlap, ...
                        'backfit_hard_cluster_pair_accuracy_overlap', backfit_hard_cluster_pair_accuracy_overlap, ...
                        'backfit_hard_label_pair_accuracy_overlap', backfit_hard_label_pair_accuracy_overlap, ...
                        'backfit_mix_available', backfit_mix_available, ...
                        'backfit_mix_cluster_top1_accuracy', backfit_mix_cluster_top1_accuracy, ...
                        'backfit_mix_label_top1_accuracy', backfit_mix_label_top1_accuracy, ...
                        'backfit_mix_cluster_top1_accuracy_overlap', backfit_mix_cluster_top1_accuracy_overlap, ...
                        'backfit_mix_label_top1_accuracy_overlap', backfit_mix_label_top1_accuracy_overlap, ...
                        'backfit_mix_label_weight_mae', backfit_mix_label_weight_mae, ...
                        'backfit_mix_label_weight_mae_overlap', backfit_mix_label_weight_mae_overlap, ...
                        'backfit_mix_cluster_pair_accuracy_overlap', backfit_mix_cluster_pair_accuracy_overlap, ...
                        'backfit_mix_label_pair_accuracy_overlap', backfit_mix_label_pair_accuracy_overlap, ...
                        'backfit_diagnostic_file', backfit_diagnostic_file, ...
                        'best_score', SelectedResults.best_criterion_value, ...
                        'json_file', json_file, ...
                        'runtime_s', SelectedResults.runtime);
                    
                    local_rows = [local_rows; result_row]; 
                    
                    % Save JSON with metadata for plotting & downstream use
                    if CONFIG.save_json
                        json_file = fullfile(json_dir, [subj_name '.json']);
                        try
                            META = struct();
                            META.subject = subj_name;
                            META.method = method_str;
                            META.criterion = criterion_str;
                            META.K_true = fit_result.K_true;
                            META.K_estimated = K_selected;
                            META.SNR_dB = fit_result.SNR_dB;
                            META.rep = fit_result.rep;
                            META.montage_type = montage_name;
                            META.n_leads = fit_result.n_leads;
                            META.overlap_prob = Sim.overlap_prob;
                            META.overlap_ms_range = Sim.overlap_ms_range;
                            META.overlap_strength = Sim.overlap_strength;
                            META.n_overlap_events = Sim.n_overlap_events;
                            if isfield(Sim, 'true_template_labels')
                                META.true_template_labels = Sim.true_template_labels;
                            end
                            if isfield(Sim, 'true_template_indices')
                                META.true_template_indices = Sim.true_template_indices;
                            end
                            META.sim_qc_status = sim_qc_status;
                            META.sim_qc_psd_slope_2_40hz = sim_qc_psd_slope;
                            META.sim_qc_gfp_median_uv = sim_qc_gfp_median_uv;
                            META.sim_qc_mean_dwell_ms = sim_qc_mean_dwell_ms;
                            META.runtime_s = SelectedResults.runtime;
                            META.channel_labels = Sim.channel_labels;  % Use Sim.channel_labels (montage-specific)
                            save_microstate_json(SelectedResults, Sim, json_file, META);
                            local_rows(end).json_file = json_file;
                        catch ME
                            if CONFIG.verbose
                                fprintf('⚠ Warning: Could not save JSON for %s: %s\n', subj_name, ME.message);
                            end
                        end
                    end
                end
            end
        end  % End montage loop

        % Store results for this EEG condition
        if numel(local_rows) >= expected_rows_per_condition
            try
                checkpoint_payload = struct( ...
                    'local_rows', local_rows, ...
                    'local_k_candidate_rows', local_k_candidate_rows, ...
                    'checkpoint_config', checkpoint_config);
                save(checkpoint_file, '-fromstruct', checkpoint_payload);
            catch ME
                if CONFIG.verbose
                    fprintf('Warning: Could not save checkpoint %s: %s\n', checkpoint_file, ME.message);
                end
            end
        end
        all_results{eeg_idx} = local_rows;
        all_k_candidate_results{eeg_idx} = local_k_candidate_rows;
        
    end  % End PARFOR loop
    
    % Combine all results
    rows = vertcat(all_results{:});
    k_candidate_rows = vertcat(all_k_candidate_results{:});
    
    % Fix IDs
    if ~isempty(rows)
        for i = 1:length(rows)
            rows(i).fit_id = i;
        end
    end
    
    if isempty(rows)
        error('✗✗✗ NO SUCCESSFUL RUNS! ✗✗✗\nCheck error summary above for details.');
    end
    
    % Convert to table
    T = struct2table(rows, 'AsArray', true);
    total_expected_fits = n_eeg_conditions * n_montages * n_methods;
    successful_fit_count = count_unique_method_fits(T);
    failed_fit_count = total_expected_fits - successful_fit_count;

    fprintf('\n========================================\n');
    fprintf('FIT SUMMARY\n');
    fprintf('========================================\n');
    fprintf('Successful method fits: %d / %d\n', successful_fit_count, total_expected_fits);
    fprintf('Failed or invalid method fits: %d\n', failed_fit_count);
    fprintf('Criterion result rows: %d\n', height(T));
    fprintf('========================================\n\n');
    
    % Save CSV
    summary_csv = fullfile(res_dir, 'comparison_results.csv');
    writetable(T, summary_csv);
    criterion_summary_csv = fullfile(res_dir, 'k_selection_summary_by_method_criterion.csv');
    Tcrit = k_selection_summary_by_method_criterion(T);
    writetable(Tcrit, criterion_summary_csv);
    if CONFIG.save_k_candidate_metrics && ~isempty(k_candidate_rows)
        Tk = struct2table(k_candidate_rows, 'AsArray', true);
        k_candidate_csv = fullfile(res_dir, 'k_candidate_metrics.csv');
        writetable(Tk, k_candidate_csv);
    end
    if CONFIG.compute_backfit_diagnostics
        if ~exist(confusion_dir, 'dir'), mkdir(confusion_dir); end
        write_simulation_backfit_reports(T, confusion_dir);
    end
    
    if CONFIG.verbose
        fprintf('✓ Saved: %s\n', summary_csv);
        fprintf('âœ“ Saved: %s\n', criterion_summary_csv);
        if CONFIG.save_k_candidate_metrics && ~isempty(k_candidate_rows)
            fprintf('✓ Saved: %s\n', fullfile(res_dir, 'k_candidate_metrics.csv'));
        end
        fprintf('  Successful: %d rows | Failed: %d\n', height(T), n_failed);
    end
    
    save_summary_info(CONFIG, ch_labels, montage_pos, res_dir);
    print_summary(T);
    
    fprintf('\n========================================\n');
    fprintf('Pipeline Complete!\n');
    fprintf('EEG Conditions: %d (overlap settings: %s)\n', n_eeg_conditions, mat2str(overlap_probs));
    fprintf('Montages: %d\n', n_montages);
    fprintf('Methods: %d\n', n_methods);
    fprintf('Valid method-criterion pairs: %d\n', n_method_criteria_pairs);
    fprintf('Total Fits: %d (EEG conditions × montages × methods)\n', n_eeg_conditions * n_montages * n_methods);
    fprintf('Total Results: %d (one per method×criterion per montage per EEG)\n', height(T));
    if CONFIG.save_json
        fprintf('JSON Microstates: %s\n', json_dir);
    else
        fprintf('JSON Microstates: disabled\n');
    end
    fprintf('Results CSV: %s\n', res_dir);
    fprintf('========================================\n');
end

% ======================== HELPERS ========================

function n = count_unique_method_fits(T)
    keys = strcat( ...
        string(T.rep), "|", ...
        string(T.K_true), "|", ...
        string(T.SNR_dB), "|", ...
        string(T.overlap_prob), "|", ...
        string(T.montage_type), "|", ...
        string(T.method));
    n = numel(unique(keys));
end

function Tcrit = k_selection_summary_by_method_criterion(T)
    Tcrit = groupsummary(T, {'method', 'criterion'}, {'mean', 'std'}, {'K_correct', 'K_error'});
    Tcrit.accuracy_pct = 100 * Tcrit.mean_K_correct;
end

function rows = build_k_candidate_metric_rows(eeg_idx, m_idx, fit_result, Results, Sim, montage_type)
    rows = [];
    if ~isfield(Results, 'K_candidates') || isempty(Results.K_candidates)
        return;
    end

    K_candidates = double(Results.K_candidates(:));
    nK = numel(K_candidates);
    method_str = char(string(fit_result.method));
    montage_name = char(string(montage_type));
    overlap_pct = round(100 * double(Sim.overlap_prob));
    run_group_id = sprintf('sim_E%d_rep%d_K%d_SNR%s_ovl%03d_%s', ...
        eeg_idx, fit_result.rep, fit_result.K_true, signed_token(fit_result.SNR_dB), ...
        overlap_pct, safe_token(montage_name));
    fit_group_id = sprintf('%s_%s', run_group_id, safe_token(method_str));

    free_energy = resize_numeric_metric(get_result_metric(Results, 'free_energy_vals'), nK, NaN);
    spm_pca_free_energy = resize_numeric_metric(get_result_metric(Results, 'spm_pca_free_energy_vals'), nK, NaN);
    log_likelihood = resize_numeric_metric(get_result_metric(Results, 'log_likelihood_vals'), nK, NaN);
    bic_vals = resize_numeric_metric(get_result_metric(Results, 'bic_vals'), nK, NaN);
    icl_vals = resize_numeric_metric(get_result_metric(Results, 'icl_vals'), nK, NaN);
    assignment_entropy = resize_numeric_metric(get_result_metric(Results, 'assignment_entropy_vals'), nK, NaN);
    n_parameters = resize_numeric_metric(get_result_metric(Results, 'n_parameters_vals'), nK, NaN);
    silhouette = resize_numeric_metric(get_result_metric(Results, 'silhouette_vals'), nK, NaN);
    gev_vals = resize_numeric_metric(get_result_metric(Results, 'gev_vals'), nK, NaN);
    calinski_harabasz = resize_numeric_metric(get_result_metric(Results, 'calinski_harabasz_vals'), nK, NaN);
    within_ss = resize_numeric_metric(get_result_metric(Results, 'within_ss'), nK, NaN);
    cov_trace_mean = resize_numeric_metric(get_result_metric(Results, 'covariance_trace_mean_vals'), nK, NaN);
    cov_trace_median = resize_numeric_metric(get_result_metric(Results, 'covariance_trace_median_vals'), nK, NaN);
    cov_logdet_mean = resize_numeric_metric(get_result_metric(Results, 'covariance_logdet_mean_vals'), nK, NaN);
    cov_logdet_median = resize_numeric_metric(get_result_metric(Results, 'covariance_logdet_median_vals'), nK, NaN);
    cov_logdet_per_dim = resize_numeric_metric(get_result_metric(Results, 'covariance_logdet_per_dim_mean_vals'), nK, NaN);
    [cov_primary, cov_primary_name] = primary_covariance_metric_for_export( ...
        cov_logdet_per_dim, cov_logdet_mean, cov_trace_mean, cov_trace_median);

    score_silhouette = criterion_score_vector_for_export(Results, 'silhouette');
    score_free_energy = criterion_score_vector_for_export(Results, 'free_energy');
    score_log_likelihood = criterion_score_vector_for_export(Results, 'log_likelihood');
    score_bic = criterion_score_vector_for_export(Results, 'bic');
    score_icl = criterion_score_vector_for_export(Results, 'icl');
    score_free_energy_elbow = criterion_score_vector_for_export(Results, 'free_energy_elbow');
    score_gev = criterion_score_vector_for_export(Results, 'gev');
    score_calinski_harabasz = criterion_score_vector_for_export(Results, 'calinski_harabasz_score');
    score_covariance = criterion_score_vector_for_export(Results, 'covariance');
    score_covariance_elbow = criterion_score_vector_for_export(Results, 'covariance_elbow');
    score_elbow_sil_combined = criterion_score_vector_for_export(Results, 'elbow_sil_combined');
    score_free_energy_covariance = criterion_score_vector_for_export(Results, 'free_energy_covariance');

    selected_silhouette = select_K_by_criterion(Results, 'silhouette');
    selected_free_energy = select_K_by_criterion(Results, 'free_energy');
    selected_log_likelihood = select_K_by_criterion(Results, 'log_likelihood');
    selected_bic = select_K_by_criterion(Results, 'bic');
    selected_icl = select_K_by_criterion(Results, 'icl');
    selected_free_energy_elbow = select_K_by_criterion(Results, 'free_energy_elbow');
    selected_gev = select_K_by_criterion(Results, 'gev');
    selected_calinski_harabasz = select_K_by_criterion(Results, 'calinski_harabasz_score');
    selected_covariance = select_K_by_criterion(Results, 'covariance');
    selected_covariance_elbow = select_K_by_criterion(Results, 'covariance_elbow');
    selected_elbow_sil_combined = select_K_by_criterion(Results, 'elbow_sil_combined');
    selected_free_energy_covariance = select_K_by_criterion(Results, 'free_energy_covariance');

    sim_qc_status = 'NOT_RUN';
    sim_qc_psd_slope = NaN;
    sim_qc_mean_dwell_ms = NaN;
    sim_qc_gfp_median_uv = NaN;
    if isfield(Sim, 'sim_qc') && ~isempty(Sim.sim_qc)
        sim_qc_status = Sim.sim_qc.overall_status;
        sim_qc_psd_slope = Sim.sim_qc.spectrum.power_slope_2_40hz;
        sim_qc_gfp_median_uv = Sim.sim_qc.amplitude.gfp_median_uv;
        if isfield(Sim.sim_qc.microstate_dynamics, 'mean_dwell_ms')
            sim_qc_mean_dwell_ms = Sim.sim_qc.microstate_dynamics.mean_dwell_ms;
        end
    end

    free_energy_norm = normalize_01_export(free_energy);
    log_likelihood_norm = normalize_01_export(log_likelihood);
    bic_norm = normalize_01_export(bic_vals);
    icl_norm = normalize_01_export(icl_vals);
    silhouette_norm = normalize_01_export(silhouette);
    gev_norm = normalize_01_export(gev_vals);
    calinski_harabasz_norm = normalize_01_export(calinski_harabasz);
    within_ss_norm = normalize_01_export(invert_metric_for_export(within_ss));
    covariance_primary_norm = normalize_01_export(invert_metric_for_export(cov_primary));
    free_energy_elbow_norm = normalize_01_export(score_free_energy_elbow);
    covariance_elbow_norm = normalize_01_export(score_covariance_elbow);

    row_cells = cell(nK, 1);
    for i = 1:nK
        K_candidate = K_candidates(i);
        row_cells{i} = struct( ...
            'fit_group_id', fit_group_id, ...
            'run_group_id', run_group_id, ...
            'k_row_id', sprintf('%s_K%02d', fit_group_id, round(K_candidate)), ...
            'eeg_idx', eeg_idx, ...
            'rep', fit_result.rep, ...
            'method', method_str, ...
            'Method', string(method_str), ...
            'K_true', fit_result.K_true, ...
            'K_candidate', K_candidate, ...
            'is_true_k', double(K_candidate == fit_result.K_true), ...
            'K', K_candidate, ...
            'K_correct', double(K_candidate == fit_result.K_true), ...
            'K_error', abs(K_candidate - fit_result.K_true), ...
            'K_gap', fit_result.K_true - K_candidate, ...
            'candidate_index', i, ...
            'n_k_candidates', nK, ...
            'edge_distance', min(i - 1, nK - i), ...
            'is_edge_candidate', double(i == 1 || i == nK), ...
            'SNR_dB', fit_result.SNR_dB, ...
            'SNR', fit_result.SNR_dB, ...
            'overlap_prob', Sim.overlap_prob, ...
            'overlap_strength', Sim.overlap_strength, ...
            'overlap_ms_min', Sim.overlap_ms_range(1), ...
            'overlap_ms_max', Sim.overlap_ms_range(end), ...
            'montage_type', montage_name, ...
            'n_leads', fit_result.n_leads, ...
            'N_channels', fit_result.n_leads, ...
            'sim_qc_status', sim_qc_status, ...
            'sim_qc_psd_slope_2_40hz', sim_qc_psd_slope, ...
            'sim_qc_gfp_median_uv', sim_qc_gfp_median_uv, ...
            'sim_qc_mean_dwell_ms', sim_qc_mean_dwell_ms, ...
            'free_energy', free_energy(i), ...
            'FreeEnergy', free_energy(i), ...
            'spm_pca_free_energy', spm_pca_free_energy(i), ...
            'log_likelihood', log_likelihood(i), ...
            'LL_K', log_likelihood(i), ...
            'bic', bic_vals(i), ...
            'BIC_K', bic_vals(i), ...
            'icl', icl_vals(i), ...
            'ICL_K', icl_vals(i), ...
            'assignment_entropy', assignment_entropy(i), ...
            'n_parameters', n_parameters(i), ...
            'silhouette', silhouette(i), ...
            'Silhouette', silhouette(i), ...
            'gev', gev_vals(i), ...
            'GEV', gev_vals(i), ...
            'calinski_harabasz', calinski_harabasz(i), ...
            'CalinskiHarabasz', calinski_harabasz(i), ...
            'GFP', sim_qc_gfp_median_uv, ...
            'GFP_median_uv', sim_qc_gfp_median_uv, ...
            'within_ss', within_ss(i), ...
            'covariance_primary', cov_primary(i), ...
            'Covariance', cov_primary(i), ...
            'covariance_primary_name', cov_primary_name, ...
            'covariance_trace_mean', cov_trace_mean(i), ...
            'covariance_trace_median', cov_trace_median(i), ...
            'covariance_logdet_mean', cov_logdet_mean(i), ...
            'covariance_logdet_median', cov_logdet_median(i), ...
            'covariance_logdet_per_dim_mean', cov_logdet_per_dim(i), ...
            'free_energy_norm', free_energy_norm(i), ...
            'log_likelihood_norm', log_likelihood_norm(i), ...
            'bic_norm', bic_norm(i), ...
            'icl_norm', icl_norm(i), ...
            'silhouette_norm', silhouette_norm(i), ...
            'gev_norm', gev_norm(i), ...
            'calinski_harabasz_norm', calinski_harabasz_norm(i), ...
            'within_ss_improvement_norm', within_ss_norm(i), ...
            'covariance_tightness_norm', covariance_primary_norm(i), ...
            'score_silhouette', score_silhouette(i), ...
            'score_free_energy', score_free_energy(i), ...
            'score_log_likelihood', score_log_likelihood(i), ...
            'score_bic', score_bic(i), ...
            'score_icl', score_icl(i), ...
            'score_free_energy_elbow', score_free_energy_elbow(i), ...
            'score_gev', score_gev(i), ...
            'score_calinski_harabasz', score_calinski_harabasz(i), ...
            'score_covariance', score_covariance(i), ...
            'score_covariance_elbow', score_covariance_elbow(i), ...
            'score_elbow_sil_combined', score_elbow_sil_combined(i), ...
            'score_free_energy_covariance', score_free_energy_covariance(i), ...
            'FreeEnergyElbow', score_free_energy_elbow(i), ...
            'CovarianceElbow', score_covariance_elbow(i), ...
            'free_energy_elbow_norm', free_energy_elbow_norm(i), ...
            'covariance_elbow_norm', covariance_elbow_norm(i), ...
            'selected_by_silhouette', double(K_candidate == selected_silhouette), ...
            'selected_by_free_energy', double(K_candidate == selected_free_energy), ...
            'selected_by_log_likelihood', double(K_candidate == selected_log_likelihood), ...
            'selected_by_bic', double(K_candidate == selected_bic), ...
            'selected_by_icl', double(K_candidate == selected_icl), ...
            'selected_by_free_energy_elbow', double(K_candidate == selected_free_energy_elbow), ...
            'selected_by_gev', double(K_candidate == selected_gev), ...
            'selected_by_calinski_harabasz', double(K_candidate == selected_calinski_harabasz), ...
            'selected_by_covariance', double(K_candidate == selected_covariance), ...
            'selected_by_covariance_elbow', double(K_candidate == selected_covariance_elbow), ...
            'selected_by_elbow_sil_combined', double(K_candidate == selected_elbow_sil_combined), ...
            'selected_by_free_energy_covariance', double(K_candidate == selected_free_energy_covariance));
    end
    rows = vertcat(row_cells{:});
end

function vec = criterion_score_vector_for_export(Results, criterion)
    if isfield(Results, 'method') && strcmpi(char(string(Results.method)), 'spm_vb')
        [~, ~, vec] = select_spm_vb_k_by_criterion(Results, criterion);
        vec = resize_numeric_metric(vec, numel(Results.K_candidates), NaN);
        return;
    end

    nK = numel(Results.K_candidates);
    vec = nan(nK, 1);
    switch lower(strtrim(char(string(criterion))))
        case 'silhouette'
            vec = resize_numeric_metric(get_result_metric(Results, 'silhouette_vals'), nK, NaN);
        case 'free_energy'
            vec = resize_numeric_metric(get_result_metric(Results, 'free_energy_vals'), nK, NaN);
        case {'gev', 'gfp', 'global_explained_variance'}
            vec = resize_numeric_metric(get_result_metric(Results, 'gev_vals'), nK, NaN);
        case {'calinski_harabasz', 'calinski_harabasz_score', 'ch'}
            vec = resize_numeric_metric(get_result_metric(Results, 'calinski_harabasz_vals'), nK, NaN);
        case {'free_energy_elbow', 'elbow'}
            fe_vals = resize_numeric_metric(get_result_metric(Results, 'free_energy_vals'), nK, NaN);
            valid_mask = isfinite(fe_vals) & fe_vals ~= 0;
            if any(valid_mask)
                k_valid = double(Results.K_candidates(valid_mask));
                [~, curvature_valid] = select_K_from_free_energy_elbow_helper(fe_vals(valid_mask), k_valid);
                vec(valid_mask) = curvature_valid;
            end
        otherwise
            vec = nan(nK, 1);
    end
end

function values = get_result_metric(Results, field_name)
    if isfield(Results, field_name) && ~isempty(Results.(field_name))
        values = double(Results.(field_name)(:));
    else
        values = [];
    end
end

function values = resize_numeric_metric(values, n, fill_value)
    if nargin < 3
        fill_value = NaN;
    end
    values = double(values(:));
    if isempty(values)
        values = repmat(fill_value, n, 1);
        return;
    end
    if numel(values) < n
        values(end + 1:n, 1) = fill_value;
    elseif numel(values) > n
        values = values(1:n);
    end
end

function [values, metric_name] = primary_covariance_metric_for_export(logdet_per_dim, logdet_mean, trace_mean, trace_median)
    candidates = { ...
        {'covariance_logdet_per_dim_mean', logdet_per_dim}, ...
        {'covariance_logdet_mean', logdet_mean}, ...
        {'covariance_trace_mean', trace_mean}, ...
        {'covariance_trace_median', trace_median}};
    values = nan(size(logdet_per_dim));
    metric_name = 'none';
    for i = 1:numel(candidates)
        metric_name_i = candidates{i}{1};
        values_i = candidates{i}{2};
        finite_vals = values_i(isfinite(values_i));
        if numel(finite_vals) >= 2 && range(finite_vals) > eps
            values = values_i;
            metric_name = metric_name_i;
            return;
        end
    end
end

function values_norm = normalize_01_export(values)
    values_norm = nan(size(values));
    finite_mask = isfinite(values);
    if ~any(finite_mask)
        return;
    end
    finite_vals = values(finite_mask);
    vmin = min(finite_vals);
    vmax = max(finite_vals);
    if ~isfinite(vmin) || ~isfinite(vmax) || abs(vmax - vmin) <= eps
        values_norm(finite_mask) = 0;
    else
        values_norm(finite_mask) = (finite_vals - vmin) ./ (vmax - vmin);
    end
end

function values_out = invert_metric_for_export(values_in)
    values_out = nan(size(values_in));
    finite_mask = isfinite(values_in);
    if ~any(finite_mask)
        return;
    end
    values_out(finite_mask) = max(values_in(finite_mask)) - values_in(finite_mask);
end

function K_selected = select_K_by_criterion(Results, criterion)
    if isfield(Results, 'method') && strcmpi(char(string(Results.method)), 'spm_vb')
        [K_selected, ~] = select_spm_vb_k_by_criterion(Results, criterion);
        return;
    end

    switch criterion
        case 'silhouette'
            if isfield(Results, 'silhouette_vals') && ~isempty(Results.silhouette_vals)
                sil = Results.silhouette_vals;
                if length(sil) > 4
                    [~, idx] = max(sil(2:(end-1)));
                    idx = idx + 1;
                else
                    [~, idx] = max(sil);
                end
                K_selected = Results.K_candidates(idx);
            else
                K_selected = NaN;
            end
            
        case {'free_energy_elbow', 'elbow'}
            if isfield(Results, 'free_energy_vals') && ~isempty(Results.free_energy_vals)
                fe = Results.free_energy_vals;
                valid_idx = ~isinf(fe) & fe ~= 0 & isfinite(fe);
                if any(valid_idx)
                    fe_v = fe(valid_idx);
                    k_cand_v = Results.K_candidates(valid_idx);
                    [K_selected, ~] = select_K_from_free_energy_elbow_helper(fe_v, k_cand_v);
                else
                    K_selected = NaN;
                end
            else
                K_selected = NaN;
            end
            
        case 'free_energy'
            if isfield(Results, 'free_energy_vals') && ~isempty(Results.free_energy_vals)
                fe = Results.free_energy_vals;
                valid_idx = ~isinf(fe) & fe ~= 0;
                if any(valid_idx)
                    [~, idx] = max(fe(valid_idx));
                    valid_k = find(valid_idx);
                    K_selected = Results.K_candidates(valid_k(idx));
                else
                    K_selected = NaN;
                end
            else
                K_selected = NaN;
            end

        case {'gev', 'gfp', 'global_explained_variance'}
            if isfield(Results, 'gev_vals') && ~isempty(Results.gev_vals)
                [~, idx] = max(Results.gev_vals);
                K_selected = Results.K_candidates(idx);
            else
                K_selected = NaN;
            end

        case {'calinski_harabasz', 'calinski_harabasz_score', 'ch'}
            if isfield(Results, 'calinski_harabasz_vals') && ~isempty(Results.calinski_harabasz_vals)
                ch_vals = Results.calinski_harabasz_vals(:);
                finite_idx = isfinite(ch_vals);
                if any(finite_idx)
                    [~, idx_local] = max(ch_vals(finite_idx));
                    valid_k = find(finite_idx);
                    K_selected = Results.K_candidates(valid_k(idx_local));
                else
                    K_selected = NaN;
                end
            else
                K_selected = NaN;
            end
            
        case {'elbow_sil_combined', 'elbow_only'}
            K_selected = Results.K_estimated;
            
        otherwise
            K_selected = NaN;
    end
end

function score = criterion_specific_best_score(Results, K_selected, criterion)
    score = NaN;
    if isfield(Results, 'method') && strcmpi(char(string(Results.method)), 'spm_vb')
        [~, score, score_by_k] = select_spm_vb_k_by_criterion(Results, criterion);
        idx_spm = find(double(Results.K_candidates(:)) == double(K_selected), 1, 'first');
        if ~isempty(idx_spm) && numel(score_by_k) >= idx_spm && isfinite(score_by_k(idx_spm))
            score = score_by_k(idx_spm);
        end
        return;
    end

    idx = find(double(Results.K_candidates(:)) == double(K_selected), 1, 'first');
    if isempty(idx)
        if isfield(Results, 'best_criterion_value')
            score = Results.best_criterion_value;
        end
        return;
    end
    switch criterion
        case 'silhouette'
            if isfield(Results, 'silhouette_vals') && numel(Results.silhouette_vals) >= idx
                score = Results.silhouette_vals(idx);
            end
        case 'free_energy'
            if isfield(Results, 'free_energy_vals') && numel(Results.free_energy_vals) >= idx
                score = Results.free_energy_vals(idx);
            end
        case {'gev', 'gfp', 'global_explained_variance'}
            if isfield(Results, 'gev_vals') && numel(Results.gev_vals) >= idx
                score = Results.gev_vals(idx);
            end
        case {'calinski_harabasz', 'calinski_harabasz_score', 'ch'}
            if isfield(Results, 'calinski_harabasz_vals') && numel(Results.calinski_harabasz_vals) >= idx
                score = Results.calinski_harabasz_vals(idx);
            end
        case {'free_energy_elbow', 'elbow'}
            if isfield(Results, 'free_energy_vals') && ~isempty(Results.free_energy_vals)
                valid_idx = ~isinf(Results.free_energy_vals(:)) & Results.free_energy_vals(:) ~= 0 & isfinite(Results.free_energy_vals(:));
                fe_v = Results.free_energy_vals(valid_idx);
                k_cand_v = Results.K_candidates(valid_idx);
                [~, curvature_by_k] = select_K_from_free_energy_elbow_helper(fe_v, k_cand_v);
                hit = find(double(k_cand_v(:)) == double(K_selected), 1, 'first');
                if ~isempty(hit) && numel(curvature_by_k) >= hit
                    score = curvature_by_k(hit);
                end
            end
        otherwise
            if isfield(Results, 'best_criterion_value')
                score = Results.best_criterion_value;
            end
    end
end

function [K_est, curvature_vals] = select_K_from_free_energy_elbow_helper(fe_vals, K_candidates)
    n = length(fe_vals);
    curvature_vals = zeros(n, 1);
    if n < 3
        [~, idx] = max(fe_vals);
        K_est = K_candidates(idx);
        curvature_vals(idx) = fe_vals(idx);
        return;
    end

    fe_norm = (fe_vals - min(fe_vals)) / (max(fe_vals) - min(fe_vals) + eps);
    k_norm = (K_candidates - min(K_candidates)) / (max(K_candidates) - min(K_candidates) + eps);
    p1 = [k_norm(1), fe_norm(1)];
    p2 = [k_norm(end), fe_norm(end)];
    for i = 2:(n - 1)
        p = [k_norm(i), fe_norm(i)];
        curvature_vals(i) = distance_from_line_local(p1, p2, p);
    end
    [~, idx] = max(curvature_vals);
    K_est = K_candidates(idx);
end

function [K_est, score] = select_K_from_elbow_helper(wss_vals, K_candidates)
    n = length(wss_vals);
    if n < 4
        [~, idx] = min(wss_vals);
        K_est = K_candidates(idx);
        score = wss_vals(idx);
        return;
    end
    
    wss_norm = (wss_vals - min(wss_vals)) / (max(wss_vals) - min(wss_vals) + eps);
    
    curvature = zeros(n, 1);
    for i = 2:(n-1)
        dy1 = wss_norm(i) - wss_norm(i-1);
        dy2 = wss_norm(i+1) - wss_norm(i);
        curvature(i) = abs(dy2 - dy1);
    end
    
    [~, idx] = max(curvature(2:(n-1)));
    idx = idx + 1;
    
    K_est = K_candidates(idx);
    score = curvature(idx);
end

function d = distance_from_line_local(p1, p2, p)
    d = abs((p2(2)-p1(2))*p(1) - (p2(1)-p1(1))*p(2) + p2(1)*p1(2) - p2(2)*p1(1)) / ...
        sqrt((p2(2)-p1(2))^2 + (p2(1)-p1(1))^2 + eps);
end

function [method_names, method_criteria_map, all_criteria, n_pairs] = build_method_criteria_design(methods_in, criteria_in)
    method_names = cellstr(string(methods_in(:)'));
    requested_criteria = cellstr(string(criteria_in(:)'));
    supported_map = containers.Map('KeyType', 'char', 'ValueType', 'any');
    supported_map('spm_vb') = {'silhouette', 'free_energy', 'log_likelihood', 'll', 'bic', 'icl', 'free_energy_elbow', 'gev', 'gfp', 'calinski_harabasz_score', 'covariance', 'covariance_elbow', 'elbow_sil_combined', 'free_energy_covariance'};
    supported_map('kmeans_koenig') = {'silhouette'};

    method_criteria_map = containers.Map('KeyType', 'char', 'ValueType', 'any');
    all_criteria = {};
    n_pairs = 0;
    for i = 1:numel(method_names)
        method_str = method_names{i};
        if ~isKey(supported_map, method_str)
            error('Unsupported simulation method: %s', method_str);
        end
        supported = supported_map(method_str);
        criteria_this = requested_criteria(ismember(requested_criteria, supported));
        if isempty(criteria_this)
            error('No requested criteria are supported for method %s.', method_str);
        end
        invalid_requested = requested_criteria(~ismember(requested_criteria, supported));
        if ~isempty(invalid_requested)
            fprintf('Skipping unsupported criteria for %s: %s\n', method_str, strjoin(invalid_requested, ', '));
        end
        method_criteria_map(method_str) = criteria_this;
        all_criteria = unique([all_criteria, criteria_this], 'stable'); %#ok<AGROW>
        n_pairs = n_pairs + numel(criteria_this);
    end
end

function save_summary_info(CONFIG, ch_labels, montage_pos, output_dir)
    info = struct();
    info.analysis_date = datestr(now, 'yyyy-mm-dd HH:MM:SS');
    info.user = getenv('USERNAME');
    info.config = CONFIG;
    info.channel_labels = ch_labels;
    info.n_channels = length(ch_labels);
    info.montage_positions = montage_pos;
    if isfield(CONFIG, 'rep_vals') && ~isempty(CONFIG.rep_vals)
        n_rep_conditions = numel(CONFIG.rep_vals);
    else
        n_rep_conditions = CONFIG.reps;
    end
    info.n_eeg_conditions = n_rep_conditions * numel(CONFIG.K_true_vals) * numel(CONFIG.SNR_dbs) * numel(CONFIG.overlap_probs);
    info.overlap_probs = CONFIG.overlap_probs;
    
    try
        json_info = jsonencode(info);
        fid = fopen(fullfile(output_dir, 'analysis_info.json'), 'w');
        fprintf(fid, '%s', json_info);
        fclose(fid);
    catch
    end
end

function print_summary(T)
    fprintf('\n========================================\n');
    fprintf('COMPARISON SUMMARY\n');
    fprintf('========================================\n');
    
    util = microstate_utilities();
    methods = unique(T.method);
    if istable(methods), methods = table2cell(methods); end
    
    fprintf('\nPerformance by Method (All Criteria Combined):\n');
    fprintf('%-20s %10s %10s %10s %10s\n', 'Method', 'Accuracy', 'F1 Score', 'Matched', 'Runtime');
    fprintf('%s\n', repmat('-', 1, 65));
    
    for m_idx = 1:length(methods)
        m = methods{m_idx};
        mask = strcmp(T.method, m);
        
        acc = 100 * mean(T.K_correct(mask));
        f1 = mean(T.f1_score(mask), 'omitnan');
        matched = mean(T.n_matched(mask), 'omitnan');
        rt = mean(T.runtime_s(mask), 'omitnan');
        
        % Format method name for display
        m_display = util.format_method_name(m);
        fprintf('%-20s %9.1f%% %10.3f %10.1f %9.1fs\n', m_display, acc, f1, matched, rt);
    end

    criteria = unique(T.criterion);
    if istable(criteria), criteria = table2cell(criteria); end

    fprintf('\nK Selection Accuracy by Method and Criterion:\n');
    fprintf('%-20s %-22s %10s %10s %8s\n', 'Method', 'Criterion', 'Accuracy', 'Mean |err|', 'N');
    fprintf('%s\n', repmat('-', 1, 76));

    for m_idx = 1:length(methods)
        m = methods{m_idx};
        for c_idx = 1:length(criteria)
            c = criteria{c_idx};
            mask = strcmp(T.method, m) & strcmp(T.criterion, c);
            if ~any(mask)
                continue;
            end
            acc = 100 * mean(T.K_correct(mask));
            kerr = mean(T.K_error(mask), 'omitnan');
            m_display = util.format_method_name(m);
            fprintf('%-20s %-22s %9.1f%% %10.3f %8d\n', ...
                m_display, c, acc, kerr, nnz(mask));
        end
    end
    
    fprintf('\n');
end

function sim_opts = build_simulation_options(CONFIG, overlap_prob, SNR_dB, rep, K_true)
    sim_opts = struct( ...
        'prob', overlap_prob, ...
        'ms_range', CONFIG.overlap_ms_range, ...
        'strength', CONFIG.overlap_strength, ...
        'ecological_profile', CONFIG.ecological_profile);

    if CONFIG.randomize_true_templates
        pool_K = round(double(CONFIG.true_template_pool_K));
        if K_true > pool_K
            error('K_true=%d exceeds true_template_pool_K=%d.', K_true, pool_K);
        end
        template_seed = 42 + rep * 1000 + K_true * 100;
        old_rng = rng;
        rng(template_seed, 'twister');
        sim_opts.template_pool_K = pool_K;
        sim_opts.template_indices = sort(randperm(pool_K, K_true));
        rng(old_rng);
    end

    if CONFIG.clean_sanity_profile && overlap_prob <= 0 && ...
            (isinf(SNR_dB) || SNR_dB >= CONFIG.clean_sanity_snr_db_threshold)
        sim_opts.clean_sanity_profile = true;
    end
end

function [ch_labels, montage_pos] = load_eeg_montage(set_file)
    if nargin > 0 && ~isempty(set_file) && ischar(set_file)
        if isfile(set_file)
            try
                if exist('pop_loadset', 'file')
                    EEG = pop_loadset(set_file);
                    if ~isempty(EEG.chanlocs)
                        ch_labels = {EEG.chanlocs.labels}';
                        
                        X = [EEG.chanlocs.X]';
                        Y = [EEG.chanlocs.Y]';
                        Z = [EEG.chanlocs.Z]';
                        
                        montage_pos = [X, Y, Z];
                        norms = sqrt(sum(montage_pos.^2, 2));
                        montage_pos = montage_pos ./ (norms + eps);
                        
                        fprintf('✓ Loaded %d channels from: %s\n', length(ch_labels), set_file);
                        return;
                    end
                end
                
                if endsWith(set_file, '.mat')
                    data = load(set_file);
                    if isfield(data, 'ch_labels') && isfield(data, 'montage_pos')
                        ch_labels = data.ch_labels;
                        montage_pos = data.montage_pos;
                        fprintf('✓ Loaded %d channels from: %s\n', length(ch_labels), set_file);
                        return;
                    end
                end
            catch ME
                fprintf('⚠ Could not load %s: %s\n', set_file, ME.message);
            end
        end
    end
    
    fprintf('Using standard 10-20 Extended montage (64 channels)\n');
    
    ch_labels = get_standard_10_20_labels();
    montage_pos = get_standard_10_20_positions();
end

function ch_labels = get_standard_10_20_labels()
    ch_labels = {
        'Cz', 'Fz', 'Pz', 'Oz', ...
        'Fp1', 'Fp2', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', ...
        'AF3', 'AF4', 'AF7', 'AF8', ...
        'C1', 'C2', 'C3', 'C4', 'C5', 'C6', ...
        'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FT7', 'FT8', ...
        'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', ...
        'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'TP7', 'TP8', ...
        'T3', 'T4', 'T5', 'T6', ...
        'O1', 'O2', ...
        'Iz', 'POz', 'FCz', 'CPz', ...
        'PO3', 'PO4', 'PO7', 'PO8'
    };
end

function montage_pos = get_standard_10_20_positions()
    montage_pos = [
        0.000,  0.000,  1.000;
        0.000,  0.809,  0.588;
        0.000, -0.809,  0.588;
        0.000,  0.000, -1.000;
        
        0.000,  1.000,  0.000;
        0.000,  1.000,  0.000;
        0.156,  0.951,  0.268;
        -0.156,  0.951,  0.268;
        0.309,  0.904,  0.293;
        -0.309,  0.904,  0.293;
        0.454,  0.809,  0.372;
        -0.454,  0.809,  0.372;
        0.588,  0.676,  0.447;
        -0.588,  0.676,  0.447;
        0.125,  0.968,  0.224;
        -0.125,  0.968,  0.224;
        0.274,  0.942,  0.195;
        -0.274,  0.942,  0.195;
        
        0.156,  0.000,  0.988;
        -0.156,  0.000,  0.988;
        0.309,  0.000,  0.951;
        -0.309,  0.000,  0.951;
        0.454,  0.000,  0.891;
        -0.454,  0.000,  0.891;
        0.250,  0.484,  0.835;
        -0.250,  0.484,  0.835;
        0.407,  0.809,  0.424;
        -0.407,  0.809,  0.424;
        0.588,  0.676,  0.447;
        -0.588,  0.676,  0.447;
        0.809,  0.484,  0.327;
        -0.809,  0.484,  0.327;
        
        0.156, -0.484,  0.859;
        -0.156, -0.484,  0.859;
        0.309, -0.809,  0.495;
        -0.309, -0.809,  0.495;
        0.454, -0.809,  0.372;
        -0.454, -0.809,  0.372;
        0.588, -0.676,  0.447;
        -0.588, -0.676,  0.447;
        0.250, -0.484,  0.835;
        -0.250, -0.484,  0.835;
        0.407, -0.809,  0.424;
        -0.407, -0.809,  0.424;
        0.588, -0.676,  0.447;
        -0.588, -0.676,  0.447;
        0.809, -0.484,  0.327;
        -0.809, -0.484,  0.327;
        
        0.951,  0.309,  0.000;
        -0.951,  0.309,  0.000;
        0.951, -0.309,  0.000;
        -0.951, -0.309,  0.000;
        
        0.156, -0.951,  0.268;
        -0.156, -0.951,  0.268;
        0.000, -1.000,  0.000;
        0.000, -0.951,  0.309;
        0.000,  0.485,  0.875;
        0.000, -0.485,  0.875;
        0.125, -0.951,  0.283;
        -0.125, -0.951,  0.283;
        0.274, -0.942,  0.195;
        -0.274, -0.942,  0.195;
    ];
end

function txt = local_which_text(name)
    txt = which(name);
    if isempty(txt)
        txt = '<not found>';
    end
end

function txt = signed_token(x)
    if x >= 0
        txt = sprintf('p%g', x);
    else
        txt = sprintf('m%g', abs(x));
    end
    txt = strrep(txt, '.', 'p');
end

function txt = safe_token(txt)
    txt = regexprep(char(string(txt)), '[^A-Za-z0-9]+', '_');
    txt = regexprep(txt, '_+', '_');
    txt = regexprep(txt, '^_|_$', '');
end
