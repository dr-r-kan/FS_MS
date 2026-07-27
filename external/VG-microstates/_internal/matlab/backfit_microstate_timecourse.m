function backfit = backfit_microstate_timecourse(Sim, Results, varargin)
%BACKFIT_MICROSTATE_TIMECOURSE Backfit full-record microstate weights.
%
% Hard backfitting uses polarity-invariant topographic evidence. Real-data
% fits keep the Koenig-style GFP-peak interpolation path; simulated fits use
% every sample directly so diagnostic accuracy is not limited by peak spacing.
% Simulated mixture backfitting uses transition-aware two-map unmixing from
% the fitted maps. Real VB-GMM fits keep the older Gaussian peak-spread soft
% assignment when the required peak-map payload is available.

    p = inputParser;
    addRequired(p, 'Sim', @isstruct);
    addRequired(p, 'Results', @isstruct);
    parse(p, Sim, Results, varargin{:});

    backfit = struct('ok', false, 'message', '');
    if ~isfield(Sim, 'X_noisy') || isempty(Sim.X_noisy)
        backfit.message = 'Sim.X_noisy is missing.';
        return;
    end
    if ~isfield(Results, 'centers') || isempty(Results.centers)
        backfit.message = 'Results.centers is missing.';
        return;
    end

    util = microstate_utilities();
    try
        [X_fit, backfit_cfg] = preprocess_full_record_for_backfit_local(Sim, Results, util);
    catch ME
        backfit.message = ME.message;
        return;
    end
    maps_norm = util.normalize_maps(double(X_fit'));
    centers_norm = util.normalize_maps(double(Results.centers));

    if size(maps_norm, 2) ~= size(centers_norm, 2)
        backfit.message = sprintf('Channel mismatch after preprocessing/remap: data=%d, centers=%d.', ...
            size(maps_norm, 2), size(centers_norm, 2));
        return;
    end

    [n_samples, K_est] = size(abs(maps_norm * centers_norm'));
    is_simulated_fit = is_simulated_backfit_case_local(Sim);
    linear_resp = [];
    [hard_assignments, hard_confidence, hard_debug] = koenig_hard_backfit_local( ...
        X_fit, maps_norm, centers_norm, Results, n_samples, util, backfit_cfg);
    event_assignments = hard_assignments;
    event_debug = hard_debug;
    if is_simulated_fit
        hard_resp = abs(maps_norm * centers_norm');
        linear_resp = linear_soft_weights_topography(maps_norm, centers_norm);
        [hard_confidence, hard_assignments] = max(hard_resp, [], 2);
        hard_confidence = hard_resp(sub2ind([n_samples, K_est], (1:n_samples)', hard_assignments));
        hard_debug.mode = 'simulated_all_samples_abscorr';
        hard_debug.instantaneous_assignments = hard_assignments;
        hard_debug.instantaneous_confidence = hard_confidence;
    end
    hard_weights = zeros(n_samples, K_est);
    hard_weights(sub2ind([n_samples, K_est], (1:n_samples)', hard_assignments)) = 1;

    backfit.ok = true;
    backfit.message = 'ok';
    backfit.n_samples = n_samples;
    backfit.hard = struct( ...
        'weights', hard_weights, ...
        'assignments', hard_assignments, ...
        'confidence', hard_confidence, ...
        'mode', hard_debug.mode, ...
        'peak_sample_index', hard_debug.peak_sample_index, ...
        'peak_assignments', hard_debug.peak_assignments, ...
        'peak_confidence', hard_debug.peak_confidence, ...
        'instantaneous_assignments', hard_debug.instantaneous_assignments, ...
        'instantaneous_confidence', hard_debug.instantaneous_confidence);

    backfit.mixture = struct( ...
        'available', false, ...
        'message', 'Soft mixture backfit requires either simulated data or saved VB-GMM peak-map labels.', ...
        'weights', zeros(n_samples, K_est), ...
        'assignments', nan(n_samples, 1), ...
        'confidence', nan(n_samples, 1), ...
        'sigmas', nan(1, K_est), ...
        'mode', 'unavailable');

    if is_simulated_fit
        try
            if isempty(linear_resp)
                linear_resp = linear_soft_weights_topography(maps_norm, centers_norm);
            end
            [resp, mix_info] = transition_aware_soft_weights_local( ...
                X_fit, maps_norm, centers_norm, hard_assignments, event_assignments, event_debug, Sim, linear_resp);
            [mix_confidence, mix_assignments] = max(resp, [], 2);
            backfit.mixture = struct( ...
                'available', true, ...
                'message', 'ok', ...
                'weights', resp, ...
                'assignments', mix_assignments, ...
                'confidence', mix_confidence, ...
                'sigmas', nan(1, K_est), ...
                'mode', mix_info.mode, ...
                'transition_samples', mix_info.transition_samples, ...
                'peak_mixture_samples', mix_info.peak_mixture_samples);
            return;
        catch ME
            backfit.mixture.available = false;
            backfit.mixture.message = ME.message;
            backfit.mixture.mode = 'transition_unmixing_failed';
        end
    end

    [supports_mixture, peak_maps, peak_labels] = supports_mixture_backfit(Results);
    if ~supports_mixture
        return;
    end

    try
        [resp, sigmas] = gaussian_soft_weights_topography(maps_norm, centers_norm, peak_maps, peak_labels);
        [mix_confidence, mix_assignments] = max(resp, [], 2);
        backfit.mixture = struct( ...
            'available', true, ...
            'message', 'ok', ...
            'weights', resp, ...
            'assignments', mix_assignments, ...
            'confidence', mix_confidence, ...
            'sigmas', sigmas, ...
            'mode', 'gaussian_peak_spread');
    catch ME
        backfit.mixture.available = false;
        backfit.mixture.message = ME.message;
    end
end

function tf = is_simulated_backfit_case_local(Sim)
    tf = isstruct(Sim) && ...
        isfield(Sim, 'z_true') && ~isempty(Sim.z_true) && ...
        isfield(Sim, 'maps_true') && ~isempty(Sim.maps_true);
end

function [tf, peak_maps, peak_labels] = supports_mixture_backfit(Results)
    tf = false;
    peak_maps = [];
    peak_labels = [];
    if ~isstruct(Results) || ~isfield(Results, 'centers') || isempty(Results.centers)
        return;
    end
    if ~isfield(Results, 'method') || ~startsWith(lower(char(string(Results.method))), 'spm_vb')
        return;
    end
    if ~isfield(Results, 'maps_nc') || isempty(Results.maps_nc)
        return;
    end

    peak_maps = double(Results.maps_nc);
    if isfield(Results, 'backfit_peak_labels') && ~isempty(Results.backfit_peak_labels)
        peak_labels = double(Results.backfit_peak_labels(:));
    elseif isfield(Results, 'labels') && ~isempty(Results.labels)
        peak_labels = double(Results.labels(:));
    else
        peak_labels = [];
    end

    if isempty(peak_labels)
        peak_maps = [];
        return;
    end
    if numel(peak_labels) ~= size(peak_maps, 1)
        peak_labels = peak_labels(1:min(numel(peak_labels), size(peak_maps, 1)));
        peak_maps = peak_maps(1:numel(peak_labels), :);
    end
    if isempty(peak_maps) || isempty(peak_labels)
        peak_maps = [];
        peak_labels = [];
        return;
    end
    tf = true;
end

function [X_fit, cfg] = preprocess_full_record_for_backfit_local(Sim, Results, util)
    X_fit = remap_full_record_if_needed_local(Sim, Results, util);
    cfg = resolve_backfit_preprocessing_cfg_local(Sim, Results);
    if isfield(cfg, 'apply_average_reference') && cfg.apply_average_reference
        X_fit = X_fit - mean(X_fit, 1);
    end
    Sim.preprocessing = cfg;
    [X_fit, ~] = util.apply_spatial_filter(X_fit, Sim);
    if isfield(cfg, 'filter_band') && ~isempty(cfg.filter_band)
        X_fit = util.bandpass_filter(X_fit, Sim.sfreq, cfg.filter_band);
    end
end

function cfg = resolve_backfit_preprocessing_cfg_local(Sim, Results)
    cfg = struct();
    if isfield(Sim, 'preprocessing') && isstruct(Sim.preprocessing)
        cfg = Sim.preprocessing;
    end
    if isfield(Results, 'preprocessing') && isstruct(Results.preprocessing)
        cfg = merge_structs_local(cfg, Results.preprocessing);
    end
    if isfield(Results, 'preprocessing_info') && isstruct(Results.preprocessing_info)
        info = Results.preprocessing_info;
        fields = {'apply_average_reference', 'filter_band', 'spatial_filter', ...
            'gfp_peak_min_distance', 'gfp_peak_threshold_schedule'};
        for i = 1:numel(fields)
            f = fields{i};
            if isfield(info, f) && ~isempty(info.(f))
                cfg.(f) = info.(f);
            end
        end
    end
    if ~isfield(cfg, 'gfp_peak_min_distance') || isempty(cfg.gfp_peak_min_distance)
        cfg.gfp_peak_min_distance = 3;
    end
end

function out = merge_structs_local(base, extra)
    out = base;
    if ~isstruct(extra)
        return;
    end
    fields = fieldnames(extra);
    for i = 1:numel(fields)
        out.(fields{i}) = extra.(fields{i});
    end
end

function X_out = remap_full_record_if_needed_local(Sim, Results, util)
    X_out = double(Sim.X_noisy);
    target_n_channels = size(Results.centers, 2);
    if size(X_out, 1) == target_n_channels
        return;
    end
    if ~isfield(Results, 'backfit_support') || ~isstruct(Results.backfit_support)
        error('Channel mismatch: full record has %d channels, centers have %d, and no backfit remap metadata is available.', ...
            size(X_out, 1), target_n_channels);
    end
    support = Results.backfit_support;
    if ~isfield(support, 'channel_remap_spec') || isempty(support.channel_remap_spec)
        error('Channel mismatch: backfit remap metadata is missing channel_remap_spec.');
    end

    source_labels = infer_source_labels_local(Sim, util, size(X_out, 1));
    source_chanlocs = [];
    if isfield(Sim, 'chanlocs')
        source_chanlocs = Sim.chanlocs;
    end
    source_pos = [];
    if isfield(Sim, 'pos')
        source_pos = double(Sim.pos);
    end

    spec = support.channel_remap_spec;
    source_idx = double(spec.source_data_index(:));
    if isempty(source_idx)
        error('Channel remap spec has an empty source_data_index.');
    end
    if any(source_idx < 1) || any(source_idx > size(X_out, 1))
        error('Channel remap spec source_data_index is out of range for the supplied full record.');
    end

    source_subset = X_out(source_idx, :);
    if isfield(support, 'scalp_channel_labels') && ~isempty(support.scalp_channel_labels)
        expected_labels = cellstr(string(support.scalp_channel_labels(:)));
        actual_labels = source_labels(source_idx);
        if numel(actual_labels) == numel(expected_labels) && any(~strcmpi(cellstr(string(actual_labels(:))), expected_labels))
            warning('Backfit remap source labels do not exactly match the saved scalp channel labels. Proceeding with saved indices.');
        end
    elseif ~isempty(source_chanlocs)
        scalp_mask = infer_scalp_mask_local(source_labels, source_chanlocs, source_pos);
        if nnz(scalp_mask) == numel(source_idx)
            source_subset = X_out(find(scalp_mask), :);
        end
    end

    X_out = remap_full_record_to_target_local(source_subset, spec);
    if size(X_out, 1) ~= target_n_channels
        error('Backfit remap produced %d channels, expected %d.', size(X_out, 1), target_n_channels);
    end
end

function labels = infer_source_labels_local(Sim, util, n_channels)
    labels = {};
    if isfield(Sim, 'channel_labels') && ~isempty(Sim.channel_labels)
        labels = cellstr(string(Sim.channel_labels(:)));
    elseif isfield(Sim, 'chanlocs') && ~isempty(Sim.chanlocs)
        labels = util.channel_labels_from_chanlocs(Sim.chanlocs, n_channels);
    end
    if isempty(labels)
        labels = arrayfun(@(i) sprintf('Ch%d', i), 1:n_channels, 'UniformOutput', false)';
    end
end

function mask = infer_scalp_mask_local(labels, chanlocs, pos)
    n = numel(labels);
    mask = true(n, 1);
    bad_patterns = {'ECG','EKG','EOG','HEOG','VEOG','EMG','EXG','GSR','RESP','TRIG','TRIGGER','STI','STATUS','PHOTO','AUX','MISC','REF'};
    for i = 1:n
        lab = upper(strtrim(char(string(labels{i}))));
        for b = 1:numel(bad_patterns)
            if contains(lab, bad_patterns{b})
                mask(i) = false;
                break;
            end
        end
    end
    if ~isempty(pos) && size(pos, 1) == n
        has_pos = all(isfinite(pos), 2) & sqrt(sum(pos.^2, 2)) > 0;
        if nnz(has_pos & mask) >= 16
            mask = mask & has_pos;
        end
    elseif ~isempty(chanlocs) && numel(chanlocs) >= n
        has_xy = false(n, 1);
        for i = 1:n
            has_xy(i) = isfield(chanlocs, 'X') && ~isempty(chanlocs(i).X) && isfinite(double(chanlocs(i).X));
        end
        if nnz(has_xy & mask) >= 16
            mask = mask & has_xy;
        end
    end
end

function X_target = remap_full_record_to_target_local(source_subset, spec)
    n_target = double(spec.n_target_channels);
    n_samples = size(source_subset, 2);
    X_target = nan(n_target, n_samples);
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
        if isempty(local_idx) || isempty(weights)
            error('Missing interpolation weights for target channel %d.', t);
        end
        X_target(t, :) = weights(:)' * source_subset(local_idx, :);
    end
end

function [weights, sigmas] = gaussian_soft_weights_topography(full_maps, centers_norm, peak_maps, peak_labels)
%GAUSSIAN_SOFT_WEIGHTS_TOPOGRAPHY Soft assignment from cluster-wise spread.

    peak_maps = double(peak_maps);
    peak_labels = double(peak_labels(:));
    K = size(centers_norm, 1);
    similarities = abs(full_maps * centers_norm');
    distances = 1 - similarities;
    sigmas = cluster_distance_sigmas(peak_maps, peak_labels, centers_norm, K);

    % Empirically, a half-width scale preserves hard top-1 behaviour while
    % improving overlap-weight recovery on the simulated mixtures.
    sigma_eff = max(0.5 * sigmas, eps);
    weights = exp(-0.5 * (distances .^ 2) ./ (sigma_eff .^ 2 + eps));
    weights = weights ./ max(sum(weights, 2), eps);
end

function weights = linear_soft_weights_topography(full_maps, centers_norm)
%LINEAR_SOFT_WEIGHTS_TOPOGRAPHY Polarity-invariant linear topographic unmixing.

    full_maps = double(full_maps);
    centers_norm = double(centers_norm);
    if isempty(full_maps) || isempty(centers_norm)
        error('Cannot compute linear soft weights from empty maps.');
    end
    if size(full_maps, 2) ~= size(centers_norm, 2)
        error('Channel mismatch for linear soft weights: data=%d, centers=%d.', ...
            size(full_maps, 2), size(centers_norm, 2));
    end

    coeff = full_maps * pinv(centers_norm);
    coeff_pos = max(coeff, 0);
    coeff_neg = max(-coeff, 0);

    recon_pos = coeff_pos * centers_norm;
    recon_neg = -coeff_neg * centers_norm;
    err_pos = sum((full_maps - recon_pos) .^ 2, 2);
    err_neg = sum((full_maps - recon_neg) .^ 2, 2);

    use_neg = err_neg < err_pos;
    weights = coeff_pos;
    weights(use_neg, :) = coeff_neg(use_neg, :);

    bad = ~all(isfinite(weights), 2) | sum(weights, 2) <= eps;
    if any(bad)
        fallback = abs(full_maps(bad, :) * centers_norm');
        fallback = fallback ./ max(sum(fallback, 2), eps);
        weights(bad, :) = fallback;
    end

    weights = max(weights, 0);
    weights = weights ./ max(sum(weights, 2), eps);
end

function [weights, info] = transition_aware_soft_weights_local( ...
        X_fit, maps_norm, centers_norm, hard_assignments, event_assignments, event_debug, Sim, linear_resp)
%TRANSITION_AWARE_SOFT_WEIGHTS_LOCAL Use two-map fits only where mixtures are plausible.

    [n_samples, K] = size(maps_norm * centers_norm');
    info = struct( ...
        'mode', 'transition_aware_two_map', ...
        'transition_samples', nan(0, 1), ...
        'peak_mixture_samples', nan(0, 1));
    weights = zeros(n_samples, K);
    if n_samples == 0 || K == 0 || isempty(hard_assignments)
        return;
    end

    no_overlap = isstruct(Sim) && isfield(Sim, 'overlap_prob') && ...
        ~isempty(Sim.overlap_prob) && isfinite(double(Sim.overlap_prob)) && double(Sim.overlap_prob) <= 0;
    if no_overlap
        weights = one_hot_assignments_local(hard_assignments, K);
        info.mode = 'one_hot_no_overlap';
        return;
    end

    base_assignments = hard_assignments;
    use_event_base = isfield(Sim, 'ecological_profile') && ~isempty(Sim.ecological_profile) && logical(Sim.ecological_profile);
    if use_event_base && numel(event_assignments) == n_samples && all(isfinite(event_assignments(:)))
        base_assignments = event_assignments;
    end
    weights = one_hot_assignments_local(base_assignments, K);
    hard_weights = one_hot_assignments_local(hard_assignments, K);
    empty_rows = sum(weights, 2) <= eps;
    weights(empty_rows, :) = hard_weights(empty_rows, :);

    sfreq = 250;
    if isfield(Sim, 'sfreq') && ~isempty(Sim.sfreq) && isfinite(double(Sim.sfreq))
        sfreq = double(Sim.sfreq);
    end
    max_ms = 40;
    if isfield(Sim, 'overlap_ms_range') && ~isempty(Sim.overlap_ms_range)
        max_ms = max(double(Sim.overlap_ms_range(:)));
    end
    half_win = max(2, round((max_ms / 1000) * sfreq));

    transition_pair = zeros(n_samples, 2);
    changes = find(base_assignments(2:end) ~= base_assignments(1:end-1)) + 1;
    for i = 1:numel(changes)
        t0 = changes(i);
        a = round(base_assignments(t0 - 1));
        b = round(base_assignments(t0));
        if a < 1 || a > K || b < 1 || b > K || a == b
            continue;
        end
        idx = max(1, t0 - half_win):min(n_samples, t0 + half_win);
        transition_pair(idx, :) = repmat([a b], numel(idx), 1);
    end

    gfp = sqrt(mean((double(X_fit) - mean(double(X_fit), 1)) .^ 2, 1))';
    high_gfp = gfp >= finite_quantile_local(gfp, 0.65);
    peak_mask = false(n_samples, 1);
    if isstruct(event_debug) && isfield(event_debug, 'peak_sample_index')
        peaks = round(double(event_debug.peak_sample_index(:)));
        peaks = peaks(isfinite(peaks) & peaks >= 1 & peaks <= n_samples);
        peak_mask(peaks) = true;
    end

    [~, order] = sort(abs(maps_norm * centers_norm'), 2, 'descend');
    second_linear = zeros(n_samples, 1);
    if ~isempty(linear_resp) && size(linear_resp, 1) == n_samples && size(linear_resp, 2) == K
        sorted_linear = sort(linear_resp, 2, 'descend');
        second_linear = sorted_linear(:, min(2, size(sorted_linear, 2)));
    end
    % ponytail: fixed gates; tune on held-out real/simulated data if this becomes a production scorer.
    opportunistic = second_linear >= 0.10;
    if isfield(Sim, 'ecological_profile') && ~isempty(Sim.ecological_profile) && logical(Sim.ecological_profile)
        opportunistic = opportunistic & (high_gfp | peak_mask);
    end

    candidate = any(transition_pair > 0, 2) | peak_mask | opportunistic;
    transition_samples = false(n_samples, 1);
    peak_mixture_samples = false(n_samples, 1);
    for t = find(candidate(:))'
        pair = transition_pair(t, :);
        in_transition = all(pair > 0);
        if ~in_transition
            pair = order(t, 1:min(2, K));
            if numel(pair) < 2 || pair(1) == pair(2)
                continue;
            end
        end
        [row, secondary, gain] = two_state_fit_weights_local(double(X_fit(:, t)), centers_norm, pair);
        if isempty(row)
            continue;
        end
        accept_transition = in_transition && secondary >= 0.03;
        accept_peak = peak_mask(t) && high_gfp(t) && secondary >= 0.12 && gain >= 0.05;
        accept_other = opportunistic(t) && secondary >= 0.15 && gain >= 0.10;
        if ~(accept_transition || accept_peak || accept_other)
            continue;
        end
        weights(t, :) = 0;
        weights(t, pair) = row;
        transition_samples(t) = accept_transition;
        peak_mixture_samples(t) = accept_peak;
    end

    weights = weights ./ max(sum(weights, 2), eps);
    info.transition_samples = find(transition_samples);
    info.peak_mixture_samples = find(peak_mixture_samples);
end

function weights = one_hot_assignments_local(assignments, K)
    n_samples = numel(assignments);
    weights = zeros(n_samples, K);
    valid = isfinite(assignments(:)) & assignments(:) >= 1 & assignments(:) <= K;
    rows = find(valid);
    if ~isempty(rows)
        weights(sub2ind([n_samples, K], rows, round(assignments(rows)))) = 1;
    end
end

function [weights, secondary, gain] = two_state_fit_weights_local(x, centers_norm, pair)
    weights = [];
    secondary = 0;
    gain = 0;
    pair = round(double(pair(:)'));
    if numel(pair) ~= 2 || pair(1) == pair(2)
        return;
    end
    A = centers_norm(pair, :)';
    coeff = A \ x(:);
    mag = abs(coeff(:)');
    if ~all(isfinite(mag)) || sum(mag) <= eps
        return;
    end
    weights = mag ./ sum(mag);
    secondary = min(weights);

    fit_pair = A * coeff;
    err_pair = sum((x(:) - fit_pair) .^ 2);
    err_single = inf;
    for j = 1:2
        a = A(:, j);
        c = a \ x(:);
        err_single = min(err_single, sum((x(:) - a * c) .^ 2));
    end
    gain = max(0, (err_single - err_pair) / max(err_single, eps));
end

function q = finite_quantile_local(x, p)
    x = sort(double(x(isfinite(x))));
    if isempty(x)
        q = NaN;
        return;
    end
    p = min(max(double(p), 0), 1);
    idx = 1 + round((numel(x) - 1) * p);
    idx = min(max(idx, 1), numel(x));
    q = x(idx);
end

function sigmas = cluster_distance_sigmas(peak_maps, peak_labels, centers_norm, K)
    sigmas = zeros(1, K);
    global_d = 1 - abs(peak_maps * centers_norm');
    sigma_floor = sqrt(mean(global_d(:) .^ 2, 'omitnan') + eps);
    for k = 1:K
        idx = peak_labels == k;
        if any(idx)
            dk = 1 - abs(peak_maps(idx, :) * centers_norm(k, :)');
            sigmas(k) = sqrt(mean(dk .^ 2, 'omitnan') + eps);
        else
            sigmas(k) = sigma_floor;
        end
    end
    sigmas(~isfinite(sigmas) | sigmas < eps) = sigma_floor;
end

function [assignments, confidence, debug] = koenig_hard_backfit_local(X_fit, full_maps, centers_norm, Results, n_samples, util, cfg)
    instantaneous_similarity = abs(full_maps * centers_norm');
    [instantaneous_confidence, instantaneous_assignments] = max(instantaneous_similarity, [], 2);

    debug = struct( ...
        'mode', 'all_samples_direct', ...
        'peak_sample_index', nan(0, 1), ...
        'peak_assignments', nan(0, 1), ...
        'peak_confidence', nan(0, 1), ...
        'instantaneous_assignments', instantaneous_assignments, ...
        'instantaneous_confidence', instantaneous_confidence);
    assignments = instantaneous_assignments;
    confidence = instantaneous_confidence;

    [has_peak_fit, peak_maps, peak_sample_index] = full_record_peak_backfit_support_local(X_fit, cfg, util);
    if ~has_peak_fit
        [has_peak_fit, peak_maps, peak_sample_index] = supports_peak_backfit_local(Results, n_samples);
    end
    if has_peak_fit
        peak_maps = util.normalize_maps(double(peak_maps));
        if size(peak_maps, 2) == size(centers_norm, 2)
            peak_similarity = abs(peak_maps * centers_norm');
            [peak_confidence, peak_assignments] = max(peak_similarity, [], 2);
            [assignments, confidence] = interpolate_peak_assignments_local( ...
                peak_sample_index, peak_assignments, peak_confidence, n_samples);

            debug.mode = 'gfp_peaks_interpolated';
            debug.peak_sample_index = peak_sample_index(:);
            debug.peak_assignments = peak_assignments(:);
            debug.peak_confidence = peak_confidence(:);
        end
    end
end

function [tf, peak_maps, peak_sample_index] = full_record_peak_backfit_support_local(X_fit, cfg, util)
    tf = false;
    peak_maps = [];
    peak_sample_index = [];
    if isempty(X_fit)
        return;
    end

    min_dist = 3;
    if isfield(cfg, 'gfp_peak_min_distance') && ~isempty(cfg.gfp_peak_min_distance)
        min_dist = max(1, round(double(cfg.gfp_peak_min_distance)));
    end
    gfp = sqrt(mean((double(X_fit) - mean(double(X_fit), 1)).^2, 1));
    peak_sample_index = local_peak_finder_backfit_local(gfp, min_dist);
    if isempty(peak_sample_index)
        return;
    end

    peak_maps = double(X_fit(:, peak_sample_index))';
    valid = all(isfinite(peak_maps), 2);
    peak_maps = peak_maps(valid, :);
    peak_sample_index = peak_sample_index(valid);
    if isempty(peak_sample_index)
        return;
    end

    peak_maps = util.normalize_maps(peak_maps);
    tf = true;
end

function idx = local_peak_finder_backfit_local(x, min_dist)
    x = double(x(:)');
    idx = [];
    if numel(x) < 3
        return;
    end

    cand = find(x(2:end-1) >= x(1:end-2) & x(2:end-1) > x(3:end)) + 1;
    if isempty(cand)
        [~, best] = max(x);
        idx = best;
        return;
    end

    [~, order] = sort(x(cand), 'descend');
    cand = cand(order);
    taken = false(size(x));
    keep = false(size(cand));
    for i = 1:numel(cand)
        c = cand(i);
        left = max(1, c - min_dist);
        right = min(numel(x), c + min_dist);
        if any(taken(left:right))
            continue;
        end
        keep(i) = true;
        taken(left:right) = true;
    end
    idx = sort(cand(keep), 'ascend');
end

function [tf, peak_maps, peak_sample_index] = supports_peak_backfit_local(Results, n_samples)
    tf = false;
    peak_maps = [];
    peak_sample_index = [];
    if ~isstruct(Results) || ~isfield(Results, 'maps_nc') || isempty(Results.maps_nc)
        return;
    end

    peak_maps = double(Results.maps_nc);
    if isfield(Results, 'idx_peaks') && ~isempty(Results.idx_peaks)
        peak_sample_index = double(Results.idx_peaks(:));
    elseif isfield(Results, 'backfit_support') && isstruct(Results.backfit_support) && ...
            isfield(Results.backfit_support, 'peak_sample') && ~isempty(Results.backfit_support.peak_sample)
        peak_sample_index = double(Results.backfit_support.peak_sample(:));
    else
        peak_sample_index = [];
    end

    if isempty(peak_sample_index)
        peak_maps = [];
        return;
    end

    n_keep = min(numel(peak_sample_index), size(peak_maps, 1));
    peak_sample_index = peak_sample_index(1:n_keep);
    peak_maps = peak_maps(1:n_keep, :);
    valid = isfinite(peak_sample_index) & peak_sample_index >= 1 & peak_sample_index <= n_samples;
    peak_sample_index = round(peak_sample_index(valid));
    peak_maps = peak_maps(valid, :);
    if isempty(peak_sample_index) || isempty(peak_maps)
        peak_maps = [];
        peak_sample_index = [];
        return;
    end

    [peak_sample_index, sort_idx] = sort(peak_sample_index(:), 'ascend');
    peak_maps = peak_maps(sort_idx, :);
    [peak_sample_index, unique_idx] = unique(peak_sample_index, 'stable');
    peak_maps = peak_maps(unique_idx, :);
    tf = ~isempty(peak_sample_index) && ~isempty(peak_maps);
end

function [assignments, confidence] = interpolate_peak_assignments_local(peak_idx, peak_assignments, peak_confidence, n_samples)
    assignments = nan(n_samples, 1);
    confidence = nan(n_samples, 1);
    if isempty(peak_idx)
        return;
    end

    peak_idx = peak_idx(:);
    peak_assignments = peak_assignments(:);
    peak_confidence = peak_confidence(:);

    if numel(peak_idx) == 1
        assignments(:) = peak_assignments(1);
        confidence(:) = peak_confidence(1);
        return;
    end

    assignments(1:peak_idx(1)) = peak_assignments(1);
    confidence(1:peak_idx(1)) = peak_confidence(1);
    for i = 1:(numel(peak_idx) - 1)
        left = peak_idx(i);
        right = peak_idx(i + 1);
        midpoint = floor((left + right) / 2);
        assignments(left:min(midpoint, n_samples)) = peak_assignments(i);
        confidence(left:min(midpoint, n_samples)) = peak_confidence(i);
        assignments((midpoint + 1):min(right, n_samples)) = peak_assignments(i + 1);
        confidence((midpoint + 1):min(right, n_samples)) = peak_confidence(i + 1);
    end
    assignments(peak_idx(end):n_samples) = peak_assignments(end);
    confidence(peak_idx(end):n_samples) = peak_confidence(end);

    if any(~isfinite(assignments))
        missing = find(~isfinite(assignments));
        assignments(missing) = peak_assignments(end);
        confidence(missing) = peak_confidence(end);
    end
end
