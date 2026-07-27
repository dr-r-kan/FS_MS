function utils = microstate_utilities()
% MICROSTATE_UTILITIES: Consolidated shared utilities
%
% Returns struct of function handles to avoid duplication

    if exist('pop_loadset', 'file') ~= 2
        eeglab_file = which('eeglab');
        if ~isempty(eeglab_file)
            addpath(genpath(fileparts(eeglab_file)));
        end
    end

    utils.preprocess_maps = @preprocess_maps_internal;
    utils.apply_spatial_filter = @apply_spatial_filter_public_internal;
    utils.normalize_maps = @normalize_maps_internal;
    utils.bandpass_filter = @bandpass_fft_zero_phase_internal;
    utils.extract_gfp_peaks = @gfp_peak_maps_internal;
    utils.robust_upper_outlier_mask = @robust_upper_outlier_mask_internal;
    utils.load_config = @load_config_internal;
    utils.project_root = @project_root_internal;
    utils.resolve_path = @resolve_path_internal;
    utils.ensure_dir = @ensure_dir_internal;
    utils.get_field = @get_field_internal;
    utils.channel_labels_from_chanlocs = @channel_labels_from_chanlocs_internal;
    utils.positions_from_chanlocs = @positions_from_chanlocs_internal;
    utils.prepare_chanlocs_for_topoplot = @prepare_chanlocs_for_topoplot_internal;
    utils.prepare_metamaps_chanlocs = @prepare_metamaps_chanlocs_internal;
    utils.scalp_channel_mask = @scalp_channel_mask_internal;
    utils.canonical_channel_labels = @canonical_channel_labels_internal;
    utils.sanitize_channel_labels = @sanitize_channel_labels_internal;
    utils.clean_complex_values = @clean_complex_values_internal;
    utils.padded_vector = @pad_vector_internal;
    utils.format_method_name = @format_method_name_internal;
    utils.format_criterion_name = @format_criterion_name_internal;
    utils.progress_bar = @progbar_internal;
    utils.duration_string = @dur_str_internal;
    utils.on_off_string = @onoff_internal;
    utils.fibonacci_sphere = @fibonacci_sphere_internal;
    utils.ensure_spm_mix = @ensure_spm_mix_internal;
end

% ======================== CONFIGURATION ========================

function cfg = load_config_internal(config_file)
%LOAD_CONFIG_INTERNAL Load repository defaults from JSON, with fallback.

    if nargin < 1 || isempty(config_file)
        config_file = fullfile(project_root_internal(), 'config', 'microstate_config.json');
    else
        config_file = resolve_path_internal(config_file, project_root_internal());
    end

    cfg = default_config_internal();
    if ~isfile(config_file)
        cfg = normalise_config_paths_internal(cfg);
        return;
    end

    try
        txt = fileread(config_file);
        user_cfg = jsondecode(txt);
        cfg = merge_structs_internal(cfg, user_cfg);
        cfg.CONFIG_FILE = config_file;
    catch ME
        warning('Could not read microstate config %s: %s', config_file, ME.message);
    end
    cfg = normalise_config_paths_internal(cfg);
end

function cfg = default_config_internal()
    root_dir = project_root_internal();
    cfg = struct();
    cfg.CONFIG_FILE = '';
    cfg.paths = struct( ...
        'template_file', 'MetaMaps_2023_06.set', ...
        'single_json_dir', fullfile('outputs', 'json'), ...
        'single_plot_dir', fullfile('outputs', 'plots'), ...
        'hierarchical_output_dir', fullfile('outputs', 'hierarchical_microstates'), ...
        'simulation_output_dir', fullfile('outputs', 'simulations'), ...
        'diagnostic_output_dir', fullfile('outputs', 'diagnostics'), ...
        'koenig_code_dir', fullfile('_internal', 'matlab', 'Koenig_code'), ...
        'spm_mixture_paths', {{ ...
            fullfile(getenv('HOME'), 'spm', 'toolbox', 'mixture'), ...
            fullfile(getenv('HOME'), 'Downloads', 'spm', 'toolbox', 'mixture')}});
    cfg.preprocessing = struct( ...
        'apply_average_reference', true, ...
        'filter_band', [2 20], ...
        'spatial_filter', 'none', ...
        'spatial_filter_neighbours', 6, ...
        'spatial_filter_strength', 1, ...
        'reject_gfp_peak_outliers', true, ...
        'gfp_outlier_mad_multiplier', 6);
    cfg.single_file = struct( ...
        'method', 'spm_vb', ...
        'criterion', '', ...
        'K_candidates', 2:10, ...
        'align_template', true, ...
        'use_scalp_channels', true, ...
        'save_json', true);
    cfg.hierarchical = struct( ...
        'K_candidates', 4:7, ...
        'criterion', 'elbow_sil_combined', ...
        'use_template_initialisation', true, ...
        'canonical_reporting_template_K', [], ...
        'canonical_prior_weight_global', 50, ...
        'prior_weight_group', 12.5, ...
        'prior_weight_condition', 17.5, ...
        'prior_weight_participant', 25, ...
        'prior_weight_file', 37.5, ...
        'spm_prior_weight', 1, ...
        'reject_template_misaligned_peaks', true, ...
        'template_peak_min_abs_corr', 0.65, ...
        'template_peak_template_K', 7, ...
        'min_peak_count_after_template_rejection', 50, ...
        'max_maps_per_file', 1500, ...
        'max_global_maps', 30000, ...
        'max_child_maps', 30000, ...
        'require_spm_initialisation', true);
    cfg.simulation = struct( ...
        'out_dir', fullfile('outputs', 'simulations'), ...
        'reps', 8, ...
        'K_true_vals', [4 5 6 7], ...
        'SNR_dbs', [-9 -3 1 0 1 3], ...
        'K_candidates', 2:10, ...
        'duration_s', 300, ...
        'sfreq', 250, ...
        'n_workers', 12, ...
        'montages', {{'full', '10-20-20', '10-20-12'}}, ...
        'overlap_probs', [0 0.5 1.0], ...
        'overlap_ms_range', [10 40], ...
        'overlap_strength', 0.5, ...
        'compute_backfit_diagnostics', true, ...
        'save_backfit_details', true, ...
        'backfit_downsample_factor', 5, ...
        'template_alignment_strong_threshold', 0.5, ...
        'ecological_profile', true, ...
        'clean_sanity_profile', true, ...
        'clean_sanity_snr_db_threshold', 40, ...
        'validate_simulation', false, ...
        'preprocessing', struct('apply_average_reference', false, 'spatial_filter', 'none', 'reject_gfp_peak_outliers', false));
    cfg.plotting = struct( ...
        'display_normalisation', 'zscore', ...
        'maplimits', 'global_percentile', ...
        'map_percentile', 75, ...
        'colormap_name', 'jet', ...
        'electrodes', 'off', ...
        'resolution', 300);
end

function cfg = normalise_config_paths_internal(cfg)
    root_dir = project_root_internal();
    if isfield(cfg, 'paths')
        path_fields = {'template_file', 'single_json_dir', 'single_plot_dir', ...
            'hierarchical_output_dir', 'simulation_output_dir', 'diagnostic_output_dir', ...
            'koenig_code_dir'};
        for i = 1:numel(path_fields)
            field_name = path_fields{i};
            if isfield(cfg.paths, field_name) && ~isempty(cfg.paths.(field_name))
                cfg.paths.(field_name) = resolve_path_internal(cfg.paths.(field_name), root_dir);
            end
        end
        if isfield(cfg.paths, 'spm_mixture_paths') && ~isempty(cfg.paths.spm_mixture_paths)
            cfg.paths.spm_mixture_paths = resolve_path_internal(cellstr(string(cfg.paths.spm_mixture_paths)), root_dir);
        end
    end
    if isfield(cfg, 'simulation') && isfield(cfg.simulation, 'out_dir')
        cfg.simulation.out_dir = resolve_path_internal(cfg.simulation.out_dir, root_dir);
    end
end

function root_dir = project_root_internal()
    root_dir = fileparts(mfilename('fullpath'));
    [parent_dir, folder_name] = fileparts(root_dir);
    if strcmp(folder_name, 'matlab')
        [grandparent_dir, parent_name] = fileparts(parent_dir);
        if strcmp(parent_name, '_internal')
            root_dir = grandparent_dir;
        end
    end
end

function pth = resolve_path_internal(pth, base_dir)
    if nargin < 2 || isempty(base_dir)
        base_dir = project_root_internal();
    end
    if isstring(pth), pth = char(pth); end
    if isempty(pth)
        return;
    end
    if iscell(pth)
        for i = 1:numel(pth)
            pth{i} = resolve_path_internal(pth{i}, base_dir);
        end
        return;
    end
    if isstruct(pth)
        return;
    end
    pth = expand_path_tokens_internal(pth);
    if is_absolute_path_internal(pth)
        return;
    end
    pth = fullfile(base_dir, pth);
end

function pth = expand_path_tokens_internal(pth)
    pth = char(pth);
    if startsWith(pth, '~/') || startsWith(pth, ['~' filesep]) || strcmp(pth, '~')
        home_dir = getenv('HOME');
        if ~isempty(home_dir)
            if strcmp(pth, '~')
                pth = home_dir;
            else
                pth = fullfile(home_dir, pth(3:end));
            end
        end
    end
    [tokens, matches] = regexp(pth, '\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?', 'tokens', 'match');
    for i = 1:numel(tokens)
        token = tokens{i}{1};
        value = getenv(token);
        if ~isempty(value)
            pth = strrep(pth, matches{i}, value);
        end
    end
end

function ensure_dir_internal(pth)
    if isstring(pth), pth = char(pth); end
    if isempty(pth)
        return;
    end
    if ~exist(pth, 'dir')
        mkdir(pth);
    end
end

function [ok, info] = ensure_spm_mix_internal(explicit_spm_path, configured_paths, verbose)
%ENSURE_SPM_MIX_INTERNAL Try to make spm_mix visible from common hints.

    if nargin < 1 || isempty(explicit_spm_path)
        explicit_spm_path = '';
    end
    if nargin < 2 || isempty(configured_paths)
        configured_paths = {};
    end
    if nargin < 3 || isempty(verbose)
        verbose = false;
    end

    info = struct();
    info.spm_before = which('spm');
    info.spm_mix_before = which('spm_mix');
    info.attempted = {};
    info.added = {};
    info.spmmix_path = info.spm_mix_before;

    if exist('spm_mix', 'file') == 2
        ok = true;
        return;
    end

    candidates = {};
    candidates = append_candidate_local(candidates, explicit_spm_path);
    candidates = append_candidate_local(candidates, getenv('SPM_MIXTURE_PATH'));
    candidates = append_candidate_local(candidates, getenv('SPM_PATH'));
    candidates = append_candidate_local(candidates, configured_paths);

    spm_file = which('spm');
    if ~isempty(spm_file)
        spm_root = fileparts(spm_file);
        candidates = append_candidate_local(candidates, spm_root);
        candidates = append_candidate_local(candidates, fullfile(spm_root, 'toolbox', 'mixture'));
    end

    if ~isempty(candidates)
        [~, ia] = unique(cellfun(@char, candidates, 'UniformOutput', false), 'stable');
        candidates = candidates(sort(ia));
    end

    for i = 1:numel(candidates)
        cand = char(candidates{i});
        if isempty(cand)
            continue;
        end
        info.attempted{end+1} = cand; %#ok<AGROW>
        added_now = add_spm_candidate_local(cand, verbose);
        if ~isempty(added_now)
            info.added = [info.added, added_now]; %#ok<AGROW>
        end
        if exist('spm_mix', 'file') == 2
            ok = true;
            info.spmmix_path = which('spm_mix');
            return;
        end
    end

    ok = exist('spm_mix', 'file') == 2;
    info.spmmix_path = which('spm_mix');
end

function candidates = append_candidate_local(candidates, value)
    if isempty(value)
        return;
    end
    if isstring(value)
        value = cellstr(value);
    elseif ischar(value)
        value = {value};
    elseif ~iscell(value)
        return;
    end
    for i = 1:numel(value)
        v = value{i};
        if isempty(v)
            continue;
        end
        v = resolve_path_internal(v, project_root_internal());
        candidates{end+1} = char(v); %#ok<AGROW>
    end
end

function added = add_spm_candidate_local(candidate_dir, verbose)
    added = {};
    if ~exist(candidate_dir, 'dir')
        return;
    end

    candidate_dir = char(candidate_dir);
    mix_dir = candidate_dir;

    if exist(fullfile(candidate_dir, 'spm_mix.m'), 'file') == 2
        addpath(candidate_dir);
        added{end+1} = candidate_dir; %#ok<AGROW>
    end

    if exist(fullfile(candidate_dir, 'spm.m'), 'file') == 2
        addpath(candidate_dir);
        added{end+1} = candidate_dir; %#ok<AGROW>
        mix_dir = fullfile(candidate_dir, 'toolbox', 'mixture');
    elseif exist(fullfile(candidate_dir, 'toolbox', 'mixture', 'spm_mix.m'), 'file') == 2
        mix_dir = fullfile(candidate_dir, 'toolbox', 'mixture');
    end

    if exist(fullfile(mix_dir, 'spm_mix.m'), 'file') == 2
        addpath(mix_dir);
        added{end+1} = mix_dir; %#ok<AGROW>
    end

    if verbose && ~isempty(added)
        for i = 1:numel(added)
            fprintf('Added SPM path candidate: %s\n', added{i});
        end
    end
end

function value = get_field_internal(S, field_name, default_value)
    value = default_value;
    if isstruct(S) && isfield(S, field_name) && ~isempty(S.(field_name))
        value = S.(field_name);
    end
end

function tf = is_absolute_path_internal(pth)
    pth = char(pth);
    tf = startsWith(pth, filesep) || ~isempty(regexp(pth, '^[A-Za-z]:[\\/]', 'once'));
end

function out = merge_structs_internal(base, override)
    out = base;
    if ~isstruct(override)
        out = override;
        return;
    end
    names = fieldnames(override);
    for i = 1:numel(names)
        name = names{i};
        if isfield(out, name) && isstruct(out.(name)) && isstruct(override.(name))
            out.(name) = merge_structs_internal(out.(name), override.(name));
        else
            out.(name) = override.(name);
        end
    end
end

% ======================== SIGNAL PROCESSING ========================

function [maps_norm, idx_peaks, gfp_vec, n_maps, C_dims, maps_original, preproc_info] = preprocess_maps_internal(Sim)
% Unified preprocessing: montage + optional spatial filter + bandpass +
% GFP peak extraction + normalization.

    cfg = preprocess_config_internal(Sim);
    X_proc = double(Sim.X_noisy);

    if cfg.apply_average_reference
        X_proc = apply_average_reference_internal(X_proc);
    end

    [X_proc, spatial_filter_info] = apply_spatial_filter_internal(X_proc, Sim, cfg);
    X_bp = bandpass_fft_zero_phase_internal(X_proc, Sim.sfreq, cfg.filter_band);
    [maps_nc, idx_peaks, gfp_vec, peak_info] = gfp_peak_maps_internal(X_bp, cfg.gfp_peak_min_distance, cfg);
    
    if isempty(maps_nc)
        warning('GFP peak extraction found no peaks; using uniform subsampling');
        idx_peaks = 1:2:size(X_bp, 2);
        maps_nc = X_bp(:, idx_peaks)';
        peak_info = struct();
        peak_info.idx_peaks_before_outlier_rejection = idx_peaks;
        peak_info.idx_peaks = idx_peaks;
        peak_info.peak_gfp_values_before_outlier_rejection = gfp_vec(idx_peaks);
        peak_info.peak_gfp_values = gfp_vec(idx_peaks);
        peak_info.n_candidate_peaks = numel(idx_peaks);
        peak_info.n_retained_peaks = numel(idx_peaks);
        peak_info.n_outlier_peaks_removed = 0;
        peak_info.outlier_threshold = NaN;
        peak_info.outlier_mode = 'fallback_subsample';
    end
    
    n_maps = size(maps_nc, 1);
    C_dims = size(maps_nc, 2);
    maps_original = maps_nc;
    maps_norm = normalize_maps_internal(maps_nc);
    preproc_info = peak_info;
    preproc_info.apply_average_reference = cfg.apply_average_reference;
    preproc_info.filter_band = cfg.filter_band;
    preproc_info.spatial_filter = cfg.spatial_filter;
    preproc_info.spatial_filter_info = spatial_filter_info;
    preproc_info.gfp_peak_min_distance = cfg.gfp_peak_min_distance;
    preproc_info.gfp_peak_threshold_schedule = cfg.gfp_peak_threshold_schedule;
    preproc_info.n_maps = n_maps;
    preproc_info.n_channels = C_dims;
end

function maps_norm = normalize_maps_internal(maps)
% Normalize maps to zero mean and unit norm
    
    if isempty(maps) || size(maps, 1) == 0
        maps_norm = maps;
        return;
    end
    
    maps_norm = maps - mean(maps, 2);
    norms = sqrt(sum(maps_norm.^2, 2));
    norms(norms < eps) = 1;
    maps_norm = maps_norm ./ norms;
end

function Xf = bandpass_fft_zero_phase_internal(X, sfreq, bp)
% Zero-phase bandpass filtering using FFT
    
    if isempty(bp) || numel(bp) ~= 2
        Xf = X;
        return;
    end
    
    T = size(X, 2);
    F = fft(X, [], 2);
    freqs = (0:T-1) * (sfreq / T);
    mask = (freqs >= bp(1) & freqs <= bp(2)) | (freqs >= sfreq - bp(2) & freqs <= sfreq - bp(1));
    F(:, ~mask) = 0;
    Xf = real(ifft(F, [], 2));
end

function [maps_nc, idx_peaks, gfp, peak_info] = gfp_peak_maps_internal(X, min_dist, cfg)
% Extract maps at Global Field Power (GFP) peaks

    if nargin < 2 || isempty(min_dist)
        min_dist = 3;
    end
    if nargin < 3 || isempty(cfg)
        cfg = preprocess_config_internal(struct());
        cfg.gfp_peak_min_distance = min_dist;
    end
    
    gfp = sqrt(mean((X - mean(X, 1)).^2, 1));

    peak_info = struct();
    idx_peaks = [];
    for pct = cfg.gfp_peak_threshold_schedule
        idx_peaks = find_local_peaks_internal(gfp, min_dist, pct);
        if ~isempty(idx_peaks) && length(idx_peaks) >= cfg.min_peak_count_after_gfp_rejection
            break;
        end
    end

    idx_before = idx_peaks;
    peak_gfp_before = gfp(idx_before);
    outlier_threshold = NaN;
    n_removed = 0;
    outlier_mode = 'none';

    if cfg.reject_gfp_peak_outliers && ~isempty(idx_peaks)
        [keep_mask, outlier_threshold, outlier_mode] = robust_upper_outlier_mask_internal( ...
            peak_gfp_before, cfg.gfp_outlier_mad_multiplier, cfg.gfp_outlier_upper_quantile, ...
            cfg.min_peak_count_after_gfp_rejection);
        if sum(keep_mask) >= cfg.min_peak_count_after_gfp_rejection
            idx_peaks = idx_peaks(keep_mask);
        else
            keep_mask = true(size(idx_peaks));
        end
        n_removed = sum(~keep_mask);
    end

    peak_info.idx_peaks_before_outlier_rejection = idx_before;
    peak_info.idx_peaks = idx_peaks;
    peak_info.peak_gfp_values_before_outlier_rejection = peak_gfp_before;
    peak_info.peak_gfp_values = gfp(idx_peaks);
    peak_info.n_candidate_peaks = numel(idx_before);
    peak_info.n_retained_peaks = numel(idx_peaks);
    peak_info.n_outlier_peaks_removed = n_removed;
    peak_info.outlier_threshold = outlier_threshold;
    peak_info.outlier_mode = outlier_mode;

    if isempty(idx_peaks)
        maps_nc = [];
    else
        maps_nc = X(:, idx_peaks)';
    end
end

function cfg = preprocess_config_internal(Sim)
    defaults = struct( ...
        'apply_average_reference', false, ...
        'filter_band', [2 20], ...
        'spatial_filter', 'none', ...
        'spatial_filter_matrix', [], ...
        'spatial_filter_neighbours', 6, ...
        'spatial_filter_strength', 1, ...
        'gfp_peak_min_distance', 3, ...
        'gfp_peak_threshold_schedule', [0.50, 0.60, 0.70, 0.80, 0.90], ...
        'reject_gfp_peak_outliers', false, ...
        'gfp_outlier_mad_multiplier', 6, ...
        'gfp_outlier_upper_quantile', 0.995, ...
        'min_peak_count_after_gfp_rejection', 3);

    cfg = defaults;
    if isfield(Sim, 'preprocessing') && isstruct(Sim.preprocessing)
        names = fieldnames(defaults);
        for i = 1:numel(names)
            if isfield(Sim.preprocessing, names{i}) && ~isempty(Sim.preprocessing.(names{i}))
                cfg.(names{i}) = Sim.preprocessing.(names{i});
            end
        end
    end
end

function Xr = apply_average_reference_internal(X)
    Xr = double(X) - mean(double(X), 1);
end

function [Xf, info] = apply_spatial_filter_internal(X, Sim, cfg)
    Xf = X;
    info = struct('mode', 'none', 'n_neighbours', NaN, 'strength', NaN);

    if ~isempty(cfg.spatial_filter_matrix)
        W = double(cfg.spatial_filter_matrix);
        if size(W, 1) ~= size(X, 1) || size(W, 2) ~= size(X, 1)
            error('spatial_filter_matrix must be n_channels x n_channels.');
        end
        Xf = W * X;
        info.mode = 'custom_matrix';
        return;
    end

    mode = lower(char(cfg.spatial_filter));
    switch mode
        case {'none', 'off'}
            info.mode = 'none';
        case {'laplacian', 'surface_laplacian', 'nearest_neighbour_laplacian'}
            pos = [];
            if isfield(Sim, 'pos') && ~isempty(Sim.pos)
                pos = double(Sim.pos);
            end
            if isempty(pos) || size(pos, 2) ~= 3 || nnz(sqrt(sum(pos.^2, 2)) > eps) < 4
                warning('Spatial filter ''%s'' requested but no channel positions were available. Skipping.', mode);
                info.mode = 'skipped_no_positions';
                return;
            end
            strength = max(0, double(cfg.spatial_filter_strength));
            W = nearest_neighbour_laplacian_internal(pos, cfg.spatial_filter_neighbours, strength);
            Xf = W * X;
            info.mode = 'nearest_neighbour_laplacian';
            info.n_neighbours = cfg.spatial_filter_neighbours;
            info.strength = strength;
        case {'smooth', 'smoothing', 'spatial_smoothing', 'nearest_neighbour_smoothing'}
            pos = [];
            if isfield(Sim, 'pos') && ~isempty(Sim.pos)
                pos = double(Sim.pos);
            end
            if isempty(pos) || size(pos, 2) ~= 3 || nnz(sqrt(sum(pos.^2, 2)) > eps) < 4
                warning('Spatial filter ''%s'' requested but no channel positions were available. Skipping.', mode);
                info.mode = 'skipped_no_positions';
                return;
            end
            N = nearest_neighbour_smoothing_matrix_internal(pos, cfg.spatial_filter_neighbours);
            strength = min(1, max(0, double(cfg.spatial_filter_strength)));
            W = (1 - strength) * eye(size(N)) + strength * N;
            Xf = W * X;
            info.mode = 'nearest_neighbour_smoothing';
            info.n_neighbours = cfg.spatial_filter_neighbours;
            info.strength = strength;
        case {'average_reference', 'car'}
            Xf = apply_average_reference_internal(X);
            info.mode = 'average_reference';
        otherwise
            error('Unknown spatial_filter mode: %s', cfg.spatial_filter);
    end
end

function [Xf, info] = apply_spatial_filter_public_internal(X, Sim)
    cfg = preprocess_config_internal(Sim);
    [Xf, info] = apply_spatial_filter_internal(X, Sim, cfg);
end

function W = nearest_neighbour_laplacian_internal(pos, n_neighbours, strength)
    if nargin < 3 || isempty(strength)
        strength = 1;
    end
    N = nearest_neighbour_smoothing_matrix_internal(pos, n_neighbours);
    W = eye(size(N)) - double(strength) * N;
end

function W = nearest_neighbour_smoothing_matrix_internal(pos, n_neighbours)
    pos = double(pos);
    n_channels = size(pos, 1);
    if size(pos, 2) ~= 3
        error('Channel positions must be n_channels x 3 for spatial filtering.');
    end
    norms = sqrt(sum(pos.^2, 2));
    valid_rows = isfinite(norms) & norms > eps;
    norms(norms < eps) = 1;
    pos = pos ./ norms;
    W = zeros(n_channels);
    n_neighbours = max(1, min(round(n_neighbours), max(1, n_channels - 1)));

    for i = 1:n_channels
        if ~valid_rows(i)
            W(i, i) = 1;
            continue;
        end
        d = sqrt(sum((pos - pos(i, :)).^2, 2));
        d(i) = Inf;
        valid = find(isfinite(d) & valid_rows);
        if isempty(valid)
            W(i, i) = 1;
            continue;
        end
        [~, ord] = sort(d(valid), 'ascend');
        keep = valid(ord(1:min(n_neighbours, numel(ord))));
        weights = 1 ./ max(d(keep), eps);
        weights = weights / sum(weights);
        W(i, keep) = weights;
    end
end

function [keep_mask, threshold, mode] = robust_upper_outlier_mask_internal(x, mad_multiplier, upper_quantile, min_keep)
    x = double(x(:));
    keep_mask = true(size(x));
    threshold = NaN;
    mode = 'none';

    if numel(x) < max(5, min_keep)
        return;
    end

    med_x = median(x);
    mad_x = median(abs(x - med_x));
    if isfinite(mad_x) && mad_x > eps
        threshold = med_x + mad_multiplier * 1.4826 * mad_x;
        mode = 'mad';
    else
        threshold = quantile(x, upper_quantile);
        mode = 'quantile_fallback';
    end

    if ~isfinite(threshold)
        keep_mask(:) = true;
        threshold = NaN;
        mode = 'none';
        return;
    end

    keep_mask = x <= threshold;
    if sum(keep_mask) < min_keep
        keep_mask(:) = true;
    end
end

function idx = find_local_peaks_internal(x, min_dist, pct)
% ✅ OPTIMIZED: Remove redundant loop
    
    x = x(:)';
    n = numel(x);
    if n < 3
        idx = 1:n;
        return;
    end
    
    % Single peak detection
    cand = find([false, x(2:end-1) > x(1:end-2) & x(2:end-1) >= x(3:end), false]);
    if isempty(cand)
        idx = [];
        return;
    end
    
    % Threshold (single computation)
    thr = quantile(x, pct);
    cand = cand(x(cand) >= thr);
    if isempty(cand)
        idx = [];
        return;
    end
    
    % Greedy assignment
    [~, ord] = sort(x(cand), 'descend');
    keep = false(1, n);
    for ii = ord
        c = cand(ii);
        lo = max(1, c - min_dist);
        hi = min(n, c + min_dist);
        if ~any(keep(lo:hi))
            keep(c) = true;
        end
    end
    
    idx = find(keep);
end

% ======================== VECTOR UTILITIES ========================

function v_pad = pad_vector_internal(v, n)
% Pad vector with NaN to length n
    
    v_pad = nan(1, n);
    if ~isempty(v) && ~all(isnan(v))
        v_pad(1:min(length(v), n)) = v(1:min(length(v), n));
    end
end

% ======================== PROGRESS & STRING UTILITIES ========================

function pb = progbar_internal(total, label)
% Simple progress bar
    
    if nargin < 2
        label = '';
    end
    
    c = struct('n', 0, 'N', total, 't0', tic, 'label', label);
    pb.update = @() local_update();
    pb.update_by = @(n) local_update_by(n);
    pb.done = @() fprintf('\n');
    
    function local_update()
        c.n = c.n + 1;
        if mod(c.n, max(1, floor(c.N/50))) == 0 || c.n == c.N
            dt = toc(c.t0);
            eta = dt * (c.N - c.n) / max(c.n, 1);
            fprintf('\r[%s] %d/%d (%.1f%%) ETA %s', c.label, c.n, c.N, 100*c.n/c.N, dur_str_internal(eta));
        end
    end
    
    function local_update_by(n)
        c.n = c.n + n;
        if mod(c.n, max(1, floor(c.N/50))) == 0 || c.n == c.N
            dt = toc(c.t0);
            eta = dt * (c.N - c.n) / max(c.n, 1);
            fprintf('\r[%s] %d/%d (%.1f%%) ETA %s', c.label, c.n, c.N, 100*c.n/c.N, dur_str_internal(eta));
        end
    end
end

function s = dur_str_internal(t)
% Format duration as string
    
    if t < 60
        s = sprintf('%.0fs', t);
    else
        m = floor(t / 60);
        ssec = mod(t, 60);
        s = sprintf('%dm%.0fs', m, ssec);
    end
end

function s = onoff_internal(bool_val)
% Convert boolean to ON/OFF string
    
    if bool_val
        s = 'ON';
    else
        s = 'OFF';
    end
end

% ======================== GEOMETRY ========================

function labels = channel_labels_from_chanlocs_internal(chanlocs, n_channels)
    labels = cell(1, n_channels);
    for i = 1:n_channels
        if ~isempty(chanlocs) && numel(chanlocs) >= i && ...
                isfield(chanlocs(i), 'labels') && ~isempty(chanlocs(i).labels)
            labels{i} = char(string(chanlocs(i).labels));
        else
            labels{i} = sprintf('Ch%03d', i);
        end
    end
end

function pos = positions_from_chanlocs_internal(chanlocs, n_channels)
    pos = nan(n_channels, 3);
    if isempty(chanlocs)
        return;
    end

    n = min(n_channels, numel(chanlocs));
    for i = 1:n
        if isfield(chanlocs(i), 'X') && isfield(chanlocs(i), 'Y') && isfield(chanlocs(i), 'Z') && ...
                ~isempty(chanlocs(i).X) && ~isempty(chanlocs(i).Y) && ~isempty(chanlocs(i).Z)
            xyz = [double_or_nan_internal(chanlocs(i).X), ...
                   double_or_nan_internal(chanlocs(i).Y), ...
                   double_or_nan_internal(chanlocs(i).Z)];
            if all(isfinite(xyz)) && norm(xyz) > eps
                pos(i, :) = xyz ./ norm(xyz);
                continue;
            end
        end

        if isfield(chanlocs(i), 'theta') && isfield(chanlocs(i), 'radius') && ...
                ~isempty(chanlocs(i).theta) && ~isempty(chanlocs(i).radius)
            theta = deg2rad(double_or_nan_internal(chanlocs(i).theta));
            radius = double_or_nan_internal(chanlocs(i).radius);
            if isfinite(theta) && isfinite(radius) && radius > 0
                x = radius * cos(theta);
                y = radius * sin(theta);
                z = sqrt(max(0, 1 - min(1, radius / 0.5) .^ 2));
                xyz = [x y z];
                if norm(xyz) > eps
                    pos(i, :) = xyz ./ norm(xyz);
                end
            end
        end
    end
end

function [chanlocs_out, keep_idx] = prepare_chanlocs_for_topoplot_internal(chanlocs, n_channels)
    chanlocs_out = chanlocs;
    keep_idx = [];
    if nargin < 2 || isempty(n_channels)
        n_channels = numel(chanlocs);
    end
    if isempty(chanlocs_out)
        return;
    end

    n = min(n_channels, numel(chanlocs_out));
    chanlocs_out = chanlocs_out(1:n);
    keep = false(1, n);
    for i = 1:n
        chanlocs_out = set_mne_topoplot_location_fields_internal(chanlocs_out, i);
        theta = double_or_nan_internal(get_field_or_empty_local(chanlocs_out(i), 'theta'));
        radius = double_or_nan_internal(get_field_or_empty_local(chanlocs_out(i), 'radius'));
        keep(i) = isfinite(theta) && isfinite(radius) && radius >= 0 && radius <= 1;
    end
    keep_idx = find(keep);
    chanlocs_out = chanlocs_out(keep);
end

function [chanlocs_out, keep_idx, pos_out] = prepare_metamaps_chanlocs_internal(chanlocs, n_channels)
%PREPARE_METAMAPS_CHANLOCS_INTERNAL Canonicalise MetaMaps geometry once.
%
% MetaMaps stores X/Y as anterior/left; rotate to the same right/anterior
% Cartesian frame used by FIF/MNE sidecars, then rebuild topoplot polar fields.

    if nargin < 2 || isempty(n_channels)
        n_channels = numel(chanlocs);
    end
    chanlocs_out = chanlocs;
    if isempty(chanlocs_out)
        keep_idx = [];
        pos_out = nan(0, 3);
        return;
    end

    n = min([n_channels, numel(chanlocs_out)]);
    chanlocs_out = chanlocs_out(1:n);
    keep = has_usable_topoplot_location_internal(chanlocs_out);
    keep_idx = find(keep);
    chanlocs_out = chanlocs_out(keep_idx);
    if isempty(chanlocs_out)
        pos_out = nan(0, 3);
        return;
    end

    chanlocs_out = rotate_chanlocs_xy_internal(chanlocs_out, 90);
    chanlocs_out = set_mne_topoplot_location_fields_internal(chanlocs_out);
    pos_out = positions_from_chanlocs_internal(chanlocs_out, numel(chanlocs_out));
end

function chanlocs_out = rotate_chanlocs_xy_internal(chanlocs_in, angle_deg)
    chanlocs_out = chanlocs_in;
    if isempty(chanlocs_in) || ~isfinite(angle_deg)
        return;
    end

    rot = [cosd(angle_deg) -sind(angle_deg); sind(angle_deg) cosd(angle_deg)];
    for i = 1:numel(chanlocs_out)
        xyz = [ ...
            double_or_nan_internal(get_field_or_empty_local(chanlocs_out(i), 'X')), ...
            double_or_nan_internal(get_field_or_empty_local(chanlocs_out(i), 'Y')), ...
            double_or_nan_internal(get_field_or_empty_local(chanlocs_out(i), 'Z'))];
        if all(isfinite(xyz))
            xy = rot * xyz(1:2)';
            chanlocs_out(i).X = xy(1);
            chanlocs_out(i).Y = xy(2);
            chanlocs_out(i).Z = xyz(3);
        end

        if isfield(chanlocs_out(i), 'theta'), chanlocs_out(i).theta = []; end
        if isfield(chanlocs_out(i), 'radius'), chanlocs_out(i).radius = []; end
    end
end

function chanlocs_out = set_mne_topoplot_location_fields_internal(chanlocs_in, idx)
    chanlocs_out = chanlocs_in;
    if nargin < 2
        idx = 1:numel(chanlocs_out);
    end
    for i = idx
        xyz = [ ...
            double_or_nan_internal(get_field_or_empty_local(chanlocs_out(i), 'X')), ...
            double_or_nan_internal(get_field_or_empty_local(chanlocs_out(i), 'Y')), ...
            double_or_nan_internal(get_field_or_empty_local(chanlocs_out(i), 'Z'))];
        if all(isfinite(xyz)) && norm(xyz) > eps
            [az, elev, sph_r] = cart2sph(xyz(1), xyz(2), xyz(3));
            theta = atan2d(xyz(1), xyz(2));
            radius = 0.5 - rad2deg(elev) / 180;
            chanlocs_out(i).theta = theta;
            chanlocs_out(i).radius = radius;
            chanlocs_out(i).sph_theta = rad2deg(az);
            chanlocs_out(i).sph_phi = rad2deg(elev);
            chanlocs_out(i).sph_radius = sph_r;
        end
    end
end

function valid = has_usable_topoplot_location_internal(chanlocs)
    valid = false(1, numel(chanlocs));
    for i = 1:numel(chanlocs)
        has_polar = isfield(chanlocs(i), 'theta') && ~isempty(chanlocs(i).theta) && ...
            isfield(chanlocs(i), 'radius') && ~isempty(chanlocs(i).radius) && ...
            isfinite(double(chanlocs(i).theta)) && isfinite(double(chanlocs(i).radius)) && ...
            double(chanlocs(i).radius) > 0 && double(chanlocs(i).radius) <= 0.5;
        has_xyz = isfield(chanlocs(i), 'X') && ~isempty(chanlocs(i).X) && ...
            isfield(chanlocs(i), 'Y') && ~isempty(chanlocs(i).Y) && ...
            isfield(chanlocs(i), 'Z') && ~isempty(chanlocs(i).Z) && ...
            all(isfinite(double([chanlocs(i).X chanlocs(i).Y chanlocs(i).Z]))) && ...
            norm(double([chanlocs(i).X chanlocs(i).Y chanlocs(i).Z])) > eps;
        valid(i) = has_polar || has_xyz;
    end
end

function angle_out = wrap_display_angle_internal(angle_in)
    angle_out = mod(angle_in + 180, 360) - 180;
end

function value = get_field_or_empty_local(S, field_name)
    if isfield(S, field_name)
        value = S.(field_name);
    else
        value = [];
    end
end

function mask = scalp_channel_mask_internal(chanlocs, n_channels, max_radius)
    if nargin < 3 || isempty(max_radius)
        max_radius = 0.5;
    end
    mask = false(n_channels, 1);
    if isempty(chanlocs)
        mask(:) = true;
        return;
    end

    aux_expr = '^(EXG|EOG|HEOG|VEOG|ECG|EKG|EMG|GSR|RESP|TRIG|STATUS|STI|MISC|AUX|REF)';
    exact_exclude = {'A1', 'A2', 'M1', 'M2', 'CB1', 'CB2', 'LM', 'RM'};
    pos = positions_from_chanlocs_internal(chanlocs, n_channels);
    has_any_geometry = any(all(isfinite(pos), 2));

    for i = 1:min(n_channels, numel(chanlocs))
        label = '';
        if isfield(chanlocs(i), 'labels') && ~isempty(chanlocs(i).labels)
            label = upper(strtrim(char(string(chanlocs(i).labels))));
        end
        if (~isempty(label) && ~isempty(regexp(label, aux_expr, 'once'))) || any(strcmp(label, exact_exclude))
            continue;
        end
        if has_any_geometry
            topo_radius = topoplot_radius_from_pos_internal(pos(i, :));
            mask(i) = all(isfinite(pos(i, :))) && isfinite(topo_radius) && topo_radius <= max_radius + eps;
        else
            mask(i) = true;
        end
    end
end

function radius = topoplot_radius_from_pos_internal(xyz)
    radius = NaN;
    if numel(xyz) < 3 || any(~isfinite(xyz)) || norm(xyz) <= eps
        return;
    end
    xyz = double(xyz(:));
    xyz = xyz ./ norm(xyz);
    [~, elev] = cart2sph(xyz(1), xyz(2), xyz(3));
    radius = 0.5 - rad2deg(elev) / 180;
end

function can = canonical_channel_labels_internal(labels)
    labels = cellstr(string(labels));
    can = cell(size(labels));
    for i = 1:numel(labels)
        s = lower(strtrim(labels{i}));
        s = regexprep(s, '\s+', '');
        s = regexprep(s, '^eeg', '');
        s = regexprep(s, '^channel', '');
        s = regexprep(s, '^chan', '');
        s = regexprep(s, '[^a-z0-9]', '');
        if ~isempty(regexp(s, '^\d+$', 'once'))
            s = regexprep(s, '^0+', '');
            if isempty(s)
                s = '0';
            end
        end
        can{i} = s;
    end
end

function sanitized = sanitize_channel_labels_internal(ch_labels)
    ch_labels = cellstr(string(ch_labels));
    sanitized = cell(size(ch_labels));
    for i = 1:numel(ch_labels)
        label = ch_labels{i};
        label = regexprep(label, '[-/\\\s\.\,\(\)\[\]\{\}]', '_');
        label = regexprep(label, '^_+|_+$', '');
        if isempty(label) || isempty(regexp(label(1), '[A-Za-z]', 'once'))
            label = ['Ch' label];
        end
        sanitized{i} = matlab.lang.makeValidName(label);
    end
end

function s = clean_complex_values_internal(s)
    if isstruct(s)
        if numel(s) > 1
            for idx = 1:numel(s)
                s(idx) = clean_complex_values_internal(s(idx));
            end
            return;
        end
        fields = fieldnames(s);
        for f = 1:length(fields)
            field_name = fields{f};
            field_val = s.(field_name);
            if isstruct(field_val)
                s.(field_name) = clean_complex_values_internal(field_val);
            elseif iscell(field_val)
                for c = 1:numel(field_val)
                    if isstruct(field_val{c})
                        field_val{c} = clean_complex_values_internal(field_val{c});
                    elseif isnumeric(field_val{c})
                        field_val{c} = double(real(field_val{c}));
                    end
                end
                s.(field_name) = field_val;
            elseif isnumeric(field_val)
                s.(field_name) = double(real(field_val));
            end
        end
    elseif iscell(s)
        for c = 1:numel(s)
            if isstruct(s{c})
                s{c} = clean_complex_values_internal(s{c});
            elseif isnumeric(s{c})
                s{c} = double(real(s{c}));
            end
        end
    end
end

function x = double_or_nan_internal(x)
    if isempty(x)
        x = NaN;
        return;
    end
    if isnumeric(x) || islogical(x)
        x = double(x(:));
        x = x(1);
        return;
    end
    if ischar(x) || (isstring(x) && isscalar(x))
        tmp = str2double(char(x));
        if isfinite(tmp)
            x = tmp;
        else
            x = NaN;
        end
        return;
    end
    try
        x = double(x);
        x = x(:);
        if isempty(x)
            x = NaN;
        else
            x = x(1);
        end
    catch
        x = NaN;
    end
end

function pos = fibonacci_sphere_internal(C)
% Generate uniformly distributed points on sphere using Fibonacci sequence
    
    ga = (sqrt(5) - 1) / 2;
    i = (0:C-1)' + 0.5;
    phi = 2 * pi * mod(i * ga, 1);
    z = 1 - 2 * i / C;
    r = sqrt(max(0, 1 - z.^2));
    pos = [r.*cos(phi), r.*sin(phi), z];
end

% ======================== DISPLAY FORMATTING ========================

function display_name = format_method_name_internal(method_code)
% FORMAT_METHOD_NAME_INTERNAL: Convert method code to display name
% Examples: 'kmeans_koenig' -> 'K-Means', 'spm_vb' -> 'VB-GMM'
    
    if ischar(method_code) || isstring(method_code)
        method_code = char(method_code);
    end
    
    switch method_code
        case 'kmeans_koenig'
            display_name = 'K-Means';
        case 'spm_vb'
            display_name = 'VB-GMM';
        case 'spm_kmeans'
            display_name = 'SPM K-Means';
        case 'vb_kmeans'
            display_name = 'VB K-Means';
        case 'dp_mixture'
            display_name = 'DP Mixture';
        otherwise
            display_name = method_code;  % Fallback to original
    end
end

function display_name = format_criterion_name_internal(criterion_code)
% FORMAT_CRITERION_NAME_INTERNAL: Convert criterion code to display name
% Examples: 'elbow_sil_combined' -> 'Elbow+Silhouette'
    
    if ischar(criterion_code) || isstring(criterion_code)
        criterion_code = char(criterion_code);
    end
    
    switch criterion_code
        case 'silhouette'
            display_name = 'Silhouette';
        case 'free_energy'
            display_name = 'Free Energy';
        case {'log_likelihood', 'll'}
            display_name = 'LL';
        case 'bic'
            display_name = 'BIC';
        case 'icl'
            display_name = 'ICL';
        case 'elbow'
            display_name = 'Elbow';
        case 'free_energy_elbow'
            display_name = 'Free Energy Elbow';
        case 'elbow_sil_combined'
            display_name = 'Elbow+Silhouette';
        case 'covariance'
            display_name = 'Covariance';
        case 'covariance_elbow'
            display_name = 'Covariance Elbow';
        case 'free_energy_covariance'
            display_name = 'Free Energy+Covariance';
        case 'gev'
            display_name = 'GEV';
        case {'calinski_harabasz_score', 'calinski_harabasz'}
            display_name = 'Calinski-Harabasz';
        otherwise
            display_name = criterion_code;  % Fallback to original
    end
end
