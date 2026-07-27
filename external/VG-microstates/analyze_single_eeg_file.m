function [Results, json_file] = analyze_single_eeg_file(eeg_file, varargin)
% ANALYZE_SINGLE_EEG_FILE: Analyze a single real EEG file
%
% This function applies microstate analysis to a real EEG file without
% requiring ground truth maps. Optionally saves results as JSON for plotting.
%
% INPUTS:
%   eeg_file     - Path to EEG file (.set, .mat, or other supported format)
%   
% OPTIONAL NAME-VALUE PAIRS:
%   'method'         - Analysis method: 'kmeans_koenig', 'spm_kmeans', or 'spm_vb' (default: 'spm_vb')
%   'criterion'      - Selection criterion (default: auto-select based on method)
%   'K_candidates'   - Vector of K values to test (default: 2:10)
%   'true_maps'      - Optional ground truth maps for validation (default: [])
%   'sfreq'          - Sampling frequency (default: 250)
%   'verbose'        - Show progress messages (default: true)
%   'save_json'      - Save results to JSON file (default: true)
%   'json_dir'       - Directory for JSON output (default: config path)
%   'plot_dir'       - Directory for plots (default: config path)
%   'align_template' - Align results to template microstates (default: false)
%   'template_file'  - Template .set file for alignment (default: 'MetaMaps_2023_06.set')
%   'use_scalp_channels' - Drop non-scalp/auxiliary channels before fitting (default: true)
%   'apply_average_reference' - Apply common average reference before GFP extraction (default: true)
%   'spatial_filter' - Optional spatial filter before GFP extraction: 'none', 'smoothing', or 'laplacian' (default: 'none')
%   'spatial_filter_matrix' - Optional custom n_channels x n_channels spatial filter matrix (default: [])
%   'spatial_filter_neighbours' - Neighbour count for laplacian filter (default: 6)
%   'spatial_filter_strength' - Blend strength for smoothing/laplacian filter (default: 1)
%   'filter_band' - Temporal bandpass before GFP extraction (default: [2 20])
%   'reject_gfp_peak_outliers' - Reject extreme GFP peaks within the file (default: true)
%   'gfp_outlier_mad_multiplier' - MAD multiplier for GFP outlier rejection (default: 6)
%
% OUTPUTS:
%   Results   - Structure containing fitted microstates and metrics
%   json_file - Path to saved JSON file (empty if not saved)
%
% EXAMPLE:
%   % Analyze and save JSON with template alignment
%   [Results, json_file] = analyze_single_eeg_file('my_eeg.set', ...
%       'method', 'spm_vb', 'save_json', true, 'align_template', true);

    setup_microstate_paths();
    % Parse inputs
    util = microstate_utilities();
    repo_cfg = util.load_config();
    single_defaults = repo_cfg.single_file;
    pre_defaults = repo_cfg.preprocessing;
    path_defaults = repo_cfg.paths;

    p = inputParser;
    addRequired(p, 'eeg_file', @ischar);
    addParameter(p, 'method', char(single_defaults.method), @ischar);
    addParameter(p, 'criterion', char(single_defaults.criterion), @ischar);
    addParameter(p, 'K_candidates', double(single_defaults.K_candidates(:)'), @isnumeric);
    addParameter(p, 'true_maps', [], @isnumeric);
    addParameter(p, 'sfreq', 250, @isnumeric);
    addParameter(p, 'verbose', true, @islogical);
    addParameter(p, 'save_json', logical(single_defaults.save_json), @islogical);
    addParameter(p, 'json_dir', char(path_defaults.single_json_dir), @ischar);
    addParameter(p, 'plot_dir', char(path_defaults.single_plot_dir), @ischar);
    addParameter(p, 'align_template', logical(single_defaults.align_template), @islogical);
    addParameter(p, 'template_file', char(path_defaults.template_file), @ischar);
    addParameter(p, 'use_scalp_channels', logical(single_defaults.use_scalp_channels), @islogical);
    addParameter(p, 'apply_average_reference', logical(pre_defaults.apply_average_reference), @islogical);
    addParameter(p, 'spatial_filter', char(pre_defaults.spatial_filter), @(x) ischar(x) || isstring(x));
    addParameter(p, 'spatial_filter_matrix', [], @isnumeric);
    addParameter(p, 'spatial_filter_neighbours', double(pre_defaults.spatial_filter_neighbours), @(x) isnumeric(x) && isscalar(x) && x >= 1);
    addParameter(p, 'spatial_filter_strength', double(pre_defaults.spatial_filter_strength), @(x) isnumeric(x) && isscalar(x) && x >= 0);
    addParameter(p, 'filter_band', double(pre_defaults.filter_band(:)'), @(x) isempty(x) || (isnumeric(x) && numel(x) == 2));
    addParameter(p, 'reject_gfp_peak_outliers', logical(pre_defaults.reject_gfp_peak_outliers), @islogical);
    addParameter(p, 'gfp_outlier_mad_multiplier', double(pre_defaults.gfp_outlier_mad_multiplier), @(x) isnumeric(x) && isscalar(x) && x > 0);
    parse(p, eeg_file, varargin{:});
    
    CONFIG = p.Results;
    CONFIG.json_dir = util.resolve_path(CONFIG.json_dir, util.project_root());
    CONFIG.plot_dir = util.resolve_path(CONFIG.plot_dir, util.project_root());
    CONFIG.template_file = util.resolve_path(CONFIG.template_file, util.project_root());
    json_file = '';
    
    % Auto-select criterion if not specified
    if isempty(CONFIG.criterion)
        if strcmp(CONFIG.method, 'spm_vb')
            CONFIG.criterion = 'elbow_sil_combined';
        elseif strcmp(CONFIG.method, 'spm_kmeans')
            CONFIG.criterion = 'silhouette';
        else
            CONFIG.criterion = 'silhouette';
        end
    end
    
    % Validate method and criterion combination
    if strcmp(CONFIG.method, 'spm_kmeans') && (strcmp(CONFIG.criterion, 'free_energy') || strcmp(CONFIG.criterion, 'elbow_sil_combined'))
        error('Criterion %s is not supported for spm_kmeans method. Use ''silhouette'', ''gev'', or ''elbow'' instead.', CONFIG.criterion);
    end

    % SPM-based methods need the mixture toolbox subdirectory, which is not
    % guaranteed to be on the path just because spm.m is visible.
    if startsWith(string(CONFIG.method), "spm_")
        [spm_ok, spm_info] = util.ensure_spm_mix('', path_defaults.spm_mixture_paths, CONFIG.verbose);
        if ~spm_ok
            attempted = '';
            if isfield(spm_info, 'attempted') && ~isempty(spm_info.attempted)
                attempted = strjoin(spm_info.attempted, ', ');
            else
                attempted = '<none>';
            end
            error(['SPM spm_mix not found in MATLAB path after attempted initialisation. ', ...
                'Checked configured candidates: %s'], attempted);
        end
    end
    
    if CONFIG.verbose
        fprintf('\n========================================\n');
        fprintf('Single EEG File Analysis\n');
        fprintf('========================================\n');
        fprintf('File: %s\n', eeg_file);
        method_display = util.format_method_name(CONFIG.method);
        criterion_display = util.format_criterion_name(CONFIG.criterion);
        fprintf('Method: %s\n', method_display);
        fprintf('Criterion: %s\n', criterion_display);
        fprintf('K candidates: %s\n', mat2str(CONFIG.K_candidates));
        fprintf('Average reference: %s\n', iif(CONFIG.apply_average_reference, 'YES', 'NO'));
        fprintf('Spatial filter: %s\n', char(string(CONFIG.spatial_filter)));
        if CONFIG.reject_gfp_peak_outliers
            fprintf('GFP outlier rejection: YES (MAD x %.2f)\n', CONFIG.gfp_outlier_mad_multiplier);
        else
            fprintf('GFP outlier rejection: NO\n');
        end
        if CONFIG.align_template
            fprintf('Template alignment: YES\n');
        end
        if ~isempty(CONFIG.true_maps)
            fprintf('Ground truth: Provided (%d maps)\n', size(CONFIG.true_maps, 1));
        else
            fprintf('Ground truth: Not provided (real data mode)\n');
        end
        fprintf('========================================\n\n');
    end
    
    % Load EEG data
    if CONFIG.verbose
        fprintf('1. Loading EEG data...\n');
    end
    
    if ~exist(eeg_file, 'file')
        error('EEG file not found: %s', eeg_file);
    end
    
    % Load based on file extension
    [~, ~, ext] = fileparts(eeg_file);
    chanlocs = [];
    channel_labels = {};
    pos = [];
    
    if strcmp(ext, '.set')
        % EEGLAB .set file
        if ~exist('pop_loadset', 'file')
            error('EEGLAB not found. Please add EEGLAB to MATLAB path.');
        end
        EEG = pop_loadset(eeg_file);
        eeg_data = EEG.data;  % Channels × Time
        sfreq = EEG.srate;
        chanlocs = EEG.chanlocs;
        channel_labels = util.channel_labels_from_chanlocs(chanlocs, size(eeg_data, 1));
        pos = util.positions_from_chanlocs(chanlocs, size(eeg_data, 1));
        
    elseif strcmp(ext, '.mat')
        % MATLAB .mat file
        data = load(eeg_file);
        if isfield(data, 'eeg_data')
            eeg_data = data.eeg_data;
        elseif isfield(data, 'data')
            eeg_data = data.data;
        elseif isfield(data, 'EEG')
            eeg_data = data.EEG.data;
        else
            error('Could not find EEG data in .mat file. Expected fields: eeg_data, data, or EEG');
        end
        if isfield(data, 'sfreq')
            sfreq = data.sfreq;
        elseif isfield(data, 'srate')
            sfreq = data.srate;
        else
            sfreq = CONFIG.sfreq;
            if CONFIG.verbose
                fprintf('  ⚠ Sampling frequency not found, using default: %d Hz\n', sfreq);
            end
        end
        if isfield(data, 'chanlocs')
            chanlocs = data.chanlocs;
        elseif isfield(data, 'EEG') && isfield(data.EEG, 'chanlocs')
            chanlocs = data.EEG.chanlocs;
        end
        if isfield(data, 'channel_labels')
            channel_labels = cellstr(data.channel_labels);
        elseif isfield(data, 'ch_labels')
            channel_labels = cellstr(data.ch_labels);
        else
            channel_labels = util.channel_labels_from_chanlocs(chanlocs, size(eeg_data, 1));
        end
        if isfield(data, 'pos')
            pos = data.pos;
        else
            pos = util.positions_from_chanlocs(chanlocs, size(eeg_data, 1));
        end

    else
        error('Unsupported file format: %s. Use .set or .mat files.', ext);
    end
    
    original_channel_labels = channel_labels;
    original_chanlocs = chanlocs;
    original_pos = pos;

    if CONFIG.use_scalp_channels && ~isempty(chanlocs)
        scalp_mask = util.scalp_channel_mask(chanlocs, size(eeg_data, 1));
        if any(scalp_mask) && nnz(scalp_mask) < size(eeg_data, 1)
            excluded_labels = channel_labels(~scalp_mask);
            eeg_data = eeg_data(scalp_mask, :);
            chanlocs = chanlocs(scalp_mask);
            channel_labels = channel_labels(scalp_mask);
            if ~isempty(pos)
                pos = pos(scalp_mask, :);
            end
            if CONFIG.verbose
                fprintf('  Using %d scalp channels; excluded %d auxiliary/non-scalp channels: %s\n', ...
                    nnz(scalp_mask), numel(excluded_labels), strjoin(excluded_labels(:)', ', '));
            end
        elseif ~any(scalp_mask)
            warning('No scalp channels detected from chanlocs; fitting all channels.');
        end
    end

    [n_channels, n_samples] = size(eeg_data);
    duration_s = n_samples / sfreq;
    
    if CONFIG.verbose
        fprintf('  ✓ Loaded: %d channels, %d samples (%.1f s @ %d Hz)\n', ...
            n_channels, n_samples, duration_s, sfreq);
    end
    
    % Create simulation-like structure for compatibility with existing methods
    Sim = struct();
    Sim.X_noisy = eeg_data;  % Main EEG data field expected by preprocessing
    Sim.X_clean = eeg_data;  % For consistency
    Sim.sfreq = sfreq;
    Sim.duration_s = duration_s;
    Sim.n_channels = n_channels;
    Sim.n_samples = n_samples;
    Sim.chanlocs = chanlocs;
    Sim.channel_labels = channel_labels;
    Sim.pos = pos;
    Sim.original_chanlocs = original_chanlocs;
    Sim.original_channel_labels = original_channel_labels;
    Sim.original_pos = original_pos;
    Sim.preprocessing = build_preprocessing_config(CONFIG);
    
    % Add ground truth if provided
    if ~isempty(CONFIG.true_maps)
        if CONFIG.use_scalp_channels && exist('scalp_mask', 'var') && ...
                size(CONFIG.true_maps, 2) == numel(scalp_mask)
            Sim.maps_true = CONFIG.true_maps(:, scalp_mask);
        else
            Sim.maps_true = CONFIG.true_maps;
        end
        Sim.K_true = size(CONFIG.true_maps, 1);
        Sim.SNR_dB = NaN;  % Unknown for real data
    else
        % No ground truth - will skip recovery metrics
        Sim.maps_true = [];
        Sim.K_true = NaN;
        Sim.SNR_dB = NaN;
    end

    % Run analysis
    if CONFIG.verbose
        fprintf('\n2. Running microstate analysis...\n');
    end
    
    try
        if strcmp(CONFIG.method, 'spm_vb')
            Results = fit_microstate_spm_vb(Sim, CONFIG.K_candidates, CONFIG.criterion);
        elseif strcmp(CONFIG.method, 'kmeans_koenig')
            Results = fit_microstate_kmeans_koenig(Sim, CONFIG.K_candidates, CONFIG.criterion);
        elseif strcmp(CONFIG.method, 'spm_kmeans')
            Results = fit_microstate_spm_kmeans(Sim, CONFIG.K_candidates, CONFIG.criterion);
        else
            error('Unknown method: %s. Use ''kmeans_koenig'', ''spm_kmeans'', or ''spm_vb''.', CONFIG.method);
        end
    catch ME
        fprintf('\n✗ Analysis failed: %s\n', ME.message);
        fprintf('  Stack trace:\n%s\n', ME.getReport());
        rethrow(ME);
    end

    Results.preprocessing = Sim.preprocessing;

    if CONFIG.align_template
        if ~isfile(CONFIG.template_file)
            warning('Template file not found: %s. Skipping template alignment.', CONFIG.template_file);
        else
            try
                alignment = align_microstates_to_template(Results.centers, CONFIG.template_file, ...
                    'estimated_channel_labels', Sim.channel_labels);
                Results.template_alignment = alignment;
                Results.centers = alignment.aligned_maps;
                if CONFIG.verbose
                    fprintf('\nTemplate Alignment:\n');
                    fprintf('  Mean MetaMaps correlation: %.3f\n', alignment.mean_correlation);
                    fprintf('  Strong matches (r >= %.2f): %d/%d\n', ...
                        alignment.strong_threshold, alignment.n_strong_matches, Results.K_estimated);
                    fprintf('  Labels: %s\n', strjoin(alignment.labels(:)', ', '));
                end
            catch ME
                warning('Template alignment failed: %s', ME.message);
            end
        end
    end

    try
        Results.backfit_timecourse = backfit_microstate_timecourse(Sim, Results);
    catch ME
        if CONFIG.verbose
            fprintf('Backfit timecourse generation skipped: %s\n', ME.message);
        end
    end
    
    % Display results
    util = microstate_utilities();
    if CONFIG.verbose
        fprintf('\n========================================\n');
        fprintf('ANALYSIS RESULTS\n');
        fprintf('========================================\n');
        method_display = util.format_method_name(Results.method);
        criterion_display = util.format_criterion_name(Results.criterion);
        fprintf('Method: %s\n', method_display);
        fprintf('Criterion: %s\n', criterion_display);
        fprintf('K estimated: %d\n', Results.K_estimated);
        if ~isnan(Sim.K_true)
            fprintf('K true: %d (correct: %s)\n', Sim.K_true, ...
                iif(Results.K_estimated == Sim.K_true, 'YES', 'NO'));
        end
        fprintf('Runtime: %.2f s\n', Results.runtime);
        
        if isfield(Results, 'recovery_metrics') && ~isempty(Results.recovery_metrics)
            fprintf('\nRecovery Metrics:\n');
            fprintf('  Matched maps: %d\n', Results.recovery_metrics.n_matched);
            fprintf('  F1 score: %.4f\n', Results.recovery_metrics.f1_score);
            fprintf('  Sensitivity: %.4f\n', Results.recovery_metrics.sensitivity);
            fprintf('  Precision: %.4f\n', Results.recovery_metrics.precision);
        end
        fprintf('========================================\n\n');
    end
    
    % ===== SAVE JSON (optional) =====
    if CONFIG.save_json
        if ~exist(CONFIG.json_dir, 'dir')
            mkdir(CONFIG.json_dir);
        end
        
        [~, filename, ~] = fileparts(eeg_file);
        json_file = fullfile(CONFIG.json_dir, sprintf('%s_%s_%s.json', filename, CONFIG.method, CONFIG.criterion));
        
        try
            META = struct();
            META.subject = filename;
            META.method = CONFIG.method;
            META.criterion = CONFIG.criterion;
            META.K_estimated = Results.K_estimated;
            META.runtime_s = Results.runtime;
            META.align_template = CONFIG.align_template;
            
            if CONFIG.verbose
                fprintf('Saving JSON: %s\n', json_file);
            end
            
            save_microstate_json(Results, Sim, json_file, META);
            % Template alignment is already attached to Results above.
            if exist('never_use_legacy_template_json_path', 'var') && CONFIG.align_template && isfile(CONFIG.template_file)
                try
                    if exist('pop_loadset', 'file')
                        EEG_template = pop_loadset(CONFIG.template_file);
                        if ~isempty(EEG_template.chanlocs)
                            template_labels = {EEG_template.chanlocs.labels}';
                            % Assign state labels based on alignment
                            Sim_for_json = Sim;
                            Sim_for_json.channel_labels = template_labels;
                            save_microstate_json(Results, Sim_for_json, json_file, META);
                        else
                            save_microstate_json(Results, Sim, json_file, META);
                        end
                    else
                        save_microstate_json(Results, Sim, json_file, META);
                    end
                    if CONFIG.verbose
                        fprintf('✓ JSON saved with template alignment\n');
                    end
                catch ME
                    if CONFIG.verbose
                        fprintf('⚠ Template alignment failed (%s), saving without alignment\n', ME.message);
                    end
                    save_microstate_json(Results, Sim, json_file, META);
                end
            else
            end
            
        catch ME
            if CONFIG.verbose
                fprintf('⚠ Warning: Could not save JSON: %s\n', ME.message);
            end
            json_file = '';
        end
    end
    
    % ===== PLOT RESULTS (optional) =====
    if ~isempty(CONFIG.plot_dir) && CONFIG.save_json && ~isempty(json_file)
        if ~exist(CONFIG.plot_dir, 'dir')
            mkdir(CONFIG.plot_dir);
        end
        
        try
            if CONFIG.verbose
                fprintf('\n4. Generating plots...\n');
            end
            
            use_topographic_plotter = exist('plot_real_eeg_microstate_topographies', 'file') == 2;
            if use_topographic_plotter
                [~, filename, ~] = fileparts(json_file);
                plot_file = fullfile(CONFIG.plot_dir, sprintf('%s_microstates.png', filename));
                plot_real_eeg_microstate_topographies(json_file, eeg_file, plot_file);
                if CONFIG.verbose
                    fprintf('Topographic plot saved: %s\n', plot_file);
                end
            else
            
            % Load the JSON we just created and plot it
            json_data = jsondecode(fileread(json_file));
            
            % Create figure for microstate topography
            [~, filename, ~] = fileparts(json_file);
            fig = figure('Name', filename, 'NumberTitle', 'off', 'Color', 'white');
            
            K_est = Results.K_estimated;
            n_cols = min(5, K_est);
            n_rows = ceil(K_est / n_cols);
            
            % Get channel locations from Sim
            if isfield(Sim, 'pos') && ~isempty(Sim.pos)
                pos = Sim.pos;
            else
                pos = [];
            end
            
            % Plot estimated microstates
            for k = 1:K_est
                subplot(n_rows, n_cols, k);
                state_key = sprintf('state_%d', k);
                
                if isfield(json_data.estimated_microstates, state_key)
                    state_data = json_data.estimated_microstates.(state_key);
                    
                    % Extract channel values
                    ch_labels_sanitized = json_data.channel_info.labels_sanitized;
                    vals = zeros(1, length(ch_labels_sanitized));
                    
                    for c = 1:length(ch_labels_sanitized)
                        ch_key = ch_labels_sanitized{c};
                        if isfield(state_data, ch_key)
                            vals(c) = state_data.(ch_key);
                        end
                    end
                    
                    % Plot topoplot if available
                    if ~isempty(pos) && exist('topoplot', 'file')
                        try
                            topoplot(vals, pos, 'electrodes', 'off', 'numcontour', 6);
                        catch
                            imagesc(reshape(vals, 1, [])); colorbar;
                        end
                    else
                        imagesc(reshape(vals, 1, [])); colorbar;
                    end
                    
                    title(sprintf('State %d', k), 'FontSize', 10, 'FontWeight', 'bold');
                end
            end
            
            % Add overall title
            method_display = util.format_method_name(CONFIG.method);
            criterion_display = util.format_criterion_name(CONFIG.criterion);
            sgtitle(sprintf('%s | Method: %s | Criterion: %s | K: %d', filename, method_display, criterion_display, K_est), ...
                'FontSize', 12, 'FontWeight', 'bold', 'Interpreter', 'none');
            
            % Save figure
            plot_file = fullfile(CONFIG.plot_dir, sprintf('%s_microstates.png', filename));
            saveas(fig, plot_file);
            close(fig);
            
            if CONFIG.verbose
                fprintf('✓ Plot saved: %s\n', plot_file);
            end
            end
        catch ME
            if CONFIG.verbose
                fprintf('⚠ Warning: Could not generate plots: %s\n', ME.message);
                fprintf('%s\n', getReport(ME, 'extended', 'hyperlinks', 'off'));
            end
        end
    end
end

function out = iif(condition, true_val, false_val)
    % Inline if function
    if condition
        out = true_val;
    else
        out = false_val;
    end
end

function pre_cfg = build_preprocessing_config(CONFIG)
    pre_cfg = struct();
    pre_cfg.apply_average_reference = CONFIG.apply_average_reference;
    pre_cfg.filter_band = CONFIG.filter_band;
    pre_cfg.spatial_filter = char(string(CONFIG.spatial_filter));
    pre_cfg.spatial_filter_matrix = CONFIG.spatial_filter_matrix;
    pre_cfg.spatial_filter_neighbours = CONFIG.spatial_filter_neighbours;
    pre_cfg.spatial_filter_strength = CONFIG.spatial_filter_strength;
    pre_cfg.gfp_peak_min_distance = 3;
    pre_cfg.gfp_peak_threshold_schedule = [0.50, 0.60, 0.70, 0.80, 0.90];
    pre_cfg.reject_gfp_peak_outliers = CONFIG.reject_gfp_peak_outliers;
    pre_cfg.gfp_outlier_mad_multiplier = CONFIG.gfp_outlier_mad_multiplier;
    pre_cfg.gfp_outlier_upper_quantile = 0.995;
    pre_cfg.min_peak_count_after_gfp_rejection = 3;
end
