function diagnostics = compute_simulation_backfit_diagnostics(Sim, Results, template_file, varargin)
%COMPUTE_SIMULATION_BACKFIT_DIAGNOSTICS Compare backfit outputs to truth.
%
% Reports sample-wise dominant-state accuracy, canonical label accuracy,
% overlap-specific accuracy, and label-weight agreement for both:
% 1. Hard winner-take-all backfitting
% 2. Gaussian-mixture backfitting for VB-GMM fits

    p = inputParser;
    addRequired(p, 'Sim', @isstruct);
    addRequired(p, 'Results', @isstruct);
    addRequired(p, 'template_file', @(x) ischar(x) || isstring(x));
    addParameter(p, 'template_K', 7, @(x) isnumeric(x) && isscalar(x) && x >= 1);
    addParameter(p, 'strong_threshold', 0.5, @(x) isnumeric(x) && isscalar(x) && x >= 0 && x <= 1);
    addParameter(p, 'downsample_factor', 1, @(x) isnumeric(x) && isscalar(x) && x >= 1);
    parse(p, Sim, Results, template_file, varargin{:});

    diagnostics = struct('ok', false, 'message', '');
    if ~isfield(Sim, 'z_true') || isempty(Sim.z_true)
        diagnostics.message = 'Sim.z_true is missing.';
        return;
    end
    if ~isfield(Results, 'centers') || isempty(Results.centers)
        diagnostics.message = 'Results.centers is missing.';
        return;
    end

    util = microstate_utilities();
    template_file = char(template_file);
    estimated_channel_labels = {};
    if isfield(Sim, 'channel_labels') && ~isempty(Sim.channel_labels)
        estimated_channel_labels = cellstr(Sim.channel_labels);
    end

    if isfield(Sim, 'true_template_alignment') && ~isempty(Sim.true_template_alignment)
        true_alignment = Sim.true_template_alignment;
    else
        true_alignment = align_microstates_to_template(Sim.maps_true, template_file, ...
            'estimated_channel_labels', estimated_channel_labels, ...
            'template_K', p.Results.template_K, ...
            'strong_threshold', p.Results.strong_threshold);
    end
    if isfield(Results, 'template_alignment') && ~isempty(Results.template_alignment)
        estimated_alignment = Results.template_alignment;
    else
        estimated_alignment = align_microstates_to_template(Results.centers, template_file, ...
            'estimated_channel_labels', estimated_channel_labels, ...
            'template_K', p.Results.template_K, ...
            'strong_threshold', p.Results.strong_threshold);
    end

    if isfield(Results, 'backfit_timecourse') && isstruct(Results.backfit_timecourse) && ...
            isfield(Results.backfit_timecourse, 'ok') && Results.backfit_timecourse.ok
        backfit = Results.backfit_timecourse;
    else
        backfit = backfit_microstate_timecourse(Sim, Results);
    end
    if ~isfield(backfit, 'ok') || ~backfit.ok
        diagnostics.message = 'Backfit timecourse generation failed.';
        if isfield(backfit, 'message')
            diagnostics.message = backfit.message;
        end
        return;
    end

    true_state_weights = true_state_weight_matrix(Sim);
    n_samples_original = size(true_state_weights, 1);
    downsample_factor = max(1, round(double(p.Results.downsample_factor)));
    mixture_available = isfield(backfit, 'mixture') && isfield(backfit.mixture, 'available') && backfit.mixture.available;
    n_samples_aligned = min(n_samples_original, size(backfit.hard.weights, 1));
    if mixture_available
        n_samples_aligned = min(n_samples_aligned, size(backfit.mixture.weights, 1));
    end
    sample_idx = 1:downsample_factor:n_samples_aligned;
    true_state_weights = true_state_weights(sample_idx, :);
    backfit.hard.weights = backfit.hard.weights(sample_idx, :);
    if mixture_available
        backfit.mixture.weights = backfit.mixture.weights(sample_idx, :);
    end
    true_top_idx = dominant_state_index(true_state_weights);
    overlap_mask = sum(true_state_weights > 1e-6, 2) > 1;

    true_state_labels = true_labels_from_sim_or_alignment(Sim, true_alignment, size(Sim.maps_true, 1));
    est_state_labels = alignment_labels_or_fallback(estimated_alignment, size(Results.centers, 1), 'E');
    template_order = template_order_union(true_alignment, estimated_alignment, true_state_labels, est_state_labels);

    est_to_true_idx = estimated_to_true_state_map(Sim.maps_true, Results.centers, util);
    true_label_weights = aggregate_weights_by_label(true_state_weights, true_state_labels, template_order);
    true_top_label_idx = dominant_state_index(true_label_weights);
    cluster = summarise_cluster_identity(Sim, Results, true_state_labels, est_state_labels, template_order);

    hard = summarise_backfit_mode(backfit.hard.weights, est_to_true_idx, est_state_labels, ...
        true_state_weights, true_top_idx, true_label_weights, true_top_label_idx, template_order, overlap_mask);

    if mixture_available
        mixture = summarise_backfit_mode(backfit.mixture.weights, est_to_true_idx, est_state_labels, ...
            true_state_weights, true_top_idx, true_label_weights, true_top_label_idx, template_order, overlap_mask);
        mixture.available = true;
        mixture.message = 'ok';
    else
        mixture = empty_mode_summary(numel(template_order), size(true_state_weights, 2));
        mixture.available = false;
        if isfield(backfit, 'mixture') && isfield(backfit.mixture, 'message')
            mixture.message = backfit.mixture.message;
        else
            mixture.message = 'Gaussian-mixture backfit unavailable.';
        end
    end

    diagnostics.ok = true;
    diagnostics.message = 'ok';
    diagnostics.n_samples = size(true_state_weights, 1);
    diagnostics.n_samples_original = n_samples_original;
    diagnostics.n_samples_aligned = n_samples_aligned;
    diagnostics.downsample_factor = downsample_factor;
    diagnostics.n_overlap_samples = sum(overlap_mask);
    diagnostics.overlap_sample_fraction = mean(overlap_mask);
    diagnostics.template_labels = template_order(:);
    diagnostics.true_state_template_labels = true_state_labels(:);
    diagnostics.estimated_state_template_labels = est_state_labels(:);
    diagnostics.estimated_to_true_state_index = est_to_true_idx(:);
    diagnostics.true_alignment = true_alignment;
    diagnostics.estimated_alignment = estimated_alignment;
    diagnostics.cluster = cluster;
    diagnostics.hard = hard;
    diagnostics.mixture = mixture;
    diagnostics.cluster_confusion_counts = cluster.label_confusion_counts;
    diagnostics.cluster_confusion_row_normalized = cluster.label_confusion_row_normalized;

    % Preserve legacy top-level fields for existing consumers.
    diagnostics.coverage_true = hard.coverage_true(:);
    diagnostics.coverage_estimated = hard.coverage_estimated(:);
    diagnostics.coverage_difference = hard.coverage_difference(:);
    diagnostics.coverage_corr = hard.coverage_corr;
    diagnostics.coverage_spearman = hard.coverage_spearman;
    diagnostics.coverage_mae = hard.coverage_mae;
    diagnostics.coverage_rmse = hard.coverage_rmse;
    diagnostics.coverage_l1 = hard.coverage_l1;
    diagnostics.coverage_linf = hard.coverage_linf;
    diagnostics.confusion_counts = hard.label_confusion_counts;
    diagnostics.confusion_row_normalized = hard.label_confusion_row_normalized;
end

function summary = summarise_backfit_mode(est_cluster_weights, est_to_true_idx, est_state_labels, ...
        true_state_weights, true_top_idx, true_label_weights, true_top_label_idx, template_order, overlap_mask)

    pred_true_weights = project_estimated_weights_to_true_states(est_cluster_weights, est_to_true_idx, size(true_state_weights, 2));
    pred_label_weights = aggregate_weights_by_label(est_cluster_weights, est_state_labels, template_order);
    pred_top_true_idx = dominant_state_index(pred_true_weights);
    pred_top_label_idx = dominant_state_index(pred_label_weights);

    coverage_true = mean(true_label_weights, 1)';
    coverage_est = mean(pred_label_weights, 1)';
    active = coverage_true > 0 | coverage_est > 0;
    if ~any(active)
        active(:) = true;
    end
    diff_vec = coverage_est(active) - coverage_true(active);
    if numel(diff_vec) >= 2 && numel(unique(coverage_true(active))) > 1 && numel(unique(coverage_est(active))) > 1
        coverage_corr = corr(coverage_true(active), coverage_est(active), 'Type', 'Pearson', 'Rows', 'complete');
        coverage_spearman = corr(coverage_true(active), coverage_est(active), 'Type', 'Spearman', 'Rows', 'complete');
    else
        coverage_corr = NaN;
        coverage_spearman = NaN;
    end

    label_confusion_counts = confusion_from_indices(true_top_label_idx, pred_top_label_idx, numel(template_order));
    row_sums = sum(label_confusion_counts, 2);
    label_confusion_row_normalized = label_confusion_counts ./ max(row_sums, 1);
    cluster_confusion_counts = confusion_from_indices(true_top_idx, pred_top_true_idx, size(true_state_weights, 2));

    summary = struct();
    summary.cluster_top1_accuracy = mean(pred_top_true_idx == true_top_idx);
    summary.label_top1_accuracy = mean(pred_top_label_idx == true_top_label_idx);
    summary.cluster_top1_accuracy_overlap = masked_accuracy(pred_top_true_idx, true_top_idx, overlap_mask);
    summary.label_top1_accuracy_overlap = masked_accuracy(pred_top_label_idx, true_top_label_idx, overlap_mask);
    summary.cluster_weight_mae = mean(abs(pred_true_weights - true_state_weights), 'all');
    summary.label_weight_mae = mean(abs(pred_label_weights - true_label_weights), 'all');
    summary.cluster_weight_mae_overlap = masked_weight_mae(pred_true_weights, true_state_weights, overlap_mask);
    summary.label_weight_mae_overlap = masked_weight_mae(pred_label_weights, true_label_weights, overlap_mask);
    summary.cluster_pair_accuracy_overlap = masked_pair_accuracy(pred_true_weights, true_state_weights, overlap_mask);
    summary.label_pair_accuracy_overlap = masked_pair_accuracy(pred_label_weights, true_label_weights, overlap_mask);
    summary.coverage_true = coverage_true(:);
    summary.coverage_estimated = coverage_est(:);
    summary.coverage_difference = coverage_est(:) - coverage_true(:);
    summary.coverage_corr = coverage_corr;
    summary.coverage_spearman = coverage_spearman;
    summary.coverage_mae = mean(abs(diff_vec), 'omitnan');
    summary.coverage_rmse = sqrt(mean(diff_vec .^ 2, 'omitnan'));
    summary.coverage_l1 = sum(abs(diff_vec), 'omitnan');
    summary.coverage_linf = max(abs(diff_vec));
    summary.cluster_confusion_counts = cluster_confusion_counts;
    summary.label_confusion_counts = label_confusion_counts;
    summary.label_confusion_row_normalized = label_confusion_row_normalized;
    summary.true_top_cluster_index = true_top_idx(:);
    summary.pred_top_cluster_index = pred_top_true_idx(:);
    summary.true_top_label_index = true_top_label_idx(:);
    summary.pred_top_label_index = pred_top_label_idx(:);
end

function summary = empty_mode_summary(n_labels, K_true)
    summary = struct( ...
        'cluster_top1_accuracy', NaN, ...
        'label_top1_accuracy', NaN, ...
        'cluster_top1_accuracy_overlap', NaN, ...
        'label_top1_accuracy_overlap', NaN, ...
        'cluster_weight_mae', NaN, ...
        'label_weight_mae', NaN, ...
        'cluster_weight_mae_overlap', NaN, ...
        'label_weight_mae_overlap', NaN, ...
        'cluster_pair_accuracy_overlap', NaN, ...
        'label_pair_accuracy_overlap', NaN, ...
        'coverage_true', nan(n_labels, 1), ...
        'coverage_estimated', nan(n_labels, 1), ...
        'coverage_difference', nan(n_labels, 1), ...
        'coverage_corr', NaN, ...
        'coverage_spearman', NaN, ...
        'coverage_mae', NaN, ...
        'coverage_rmse', NaN, ...
        'coverage_l1', NaN, ...
        'coverage_linf', NaN, ...
        'cluster_confusion_counts', nan(K_true, K_true), ...
        'label_confusion_counts', nan(n_labels, n_labels), ...
        'label_confusion_row_normalized', nan(n_labels, n_labels), ...
        'true_top_cluster_index', nan(0, 1), ...
        'pred_top_cluster_index', nan(0, 1), ...
        'true_top_label_index', nan(0, 1), ...
        'pred_top_label_index', nan(0, 1));
end

function weights = true_state_weight_matrix(Sim)
    if isfield(Sim, 'state_weights_true') && ~isempty(Sim.state_weights_true)
        weights = double(Sim.state_weights_true');
    else
        K_true = size(Sim.maps_true, 1);
        idx = Sim.z_true(:);
        weights = zeros(numel(idx), K_true);
        weights(sub2ind(size(weights), (1:numel(idx))', idx)) = 1;
    end
    weights = weights ./ max(sum(weights, 2), eps);
end

function idx = dominant_state_index(weights)
    [~, idx] = max(weights, [], 2);
end

function est_to_true_idx = estimated_to_true_state_map(true_maps, estimated_maps, util)
    true_maps = util.normalize_maps(double(true_maps));
    estimated_maps = util.normalize_maps(double(estimated_maps));
    sim = abs(estimated_maps * true_maps');
    [~, est_to_true_idx] = max(sim, [], 2);
end

function weights_true = project_estimated_weights_to_true_states(weights_est, est_to_true_idx, K_true)
    weights_true = zeros(size(weights_est, 1), K_true);
    for k = 1:numel(est_to_true_idx)
        weights_true(:, est_to_true_idx(k)) = weights_true(:, est_to_true_idx(k)) + weights_est(:, k);
    end
    weights_true = weights_true ./ max(sum(weights_true, 2), eps);
end

function label_weights = aggregate_weights_by_label(state_weights, state_labels, label_order)
    label_weights = zeros(size(state_weights, 1), numel(label_order));
    for k = 1:numel(state_labels)
        label_idx = find(strcmp(label_order, state_labels{k}), 1, 'first');
        if isempty(label_idx)
            continue;
        end
        label_weights(:, label_idx) = label_weights(:, label_idx) + state_weights(:, k);
    end
    label_weights = label_weights ./ max(sum(label_weights, 2), eps);
end

function labels = alignment_labels_or_fallback(alignment, n_states, prefix)
    labels = cell(n_states, 1);
    if isstruct(alignment) && isfield(alignment, 'labels') && numel(alignment.labels) >= n_states
        raw = cellstr(alignment.labels);
        for i = 1:n_states
            labels{i} = char(string(raw{i}));
        end
        return;
    end
    for i = 1:n_states
        labels{i} = sprintf('%s%d', prefix, i);
    end
end

function labels = true_labels_from_sim_or_alignment(Sim, alignment, n_states)
    if isfield(Sim, 'true_template_labels') && numel(Sim.true_template_labels) >= n_states
        raw = cellstr(string(Sim.true_template_labels));
        labels = raw(1:n_states);
        labels = labels(:);
        return;
    end
    labels = alignment_labels_or_fallback(alignment, n_states, 'T');
end

function summary = summarise_cluster_identity(Sim, Results, true_labels, est_labels, label_order)
    K_true = numel(true_labels);
    K_est = numel(est_labels);
    metrics = [];
    if isfield(Results, 'recovery_metrics') && isstruct(Results.recovery_metrics)
        metrics = Results.recovery_metrics;
    elseif isfield(Sim, 'maps_true') && isfield(Results, 'centers') && ~isempty(Sim.maps_true) && ~isempty(Results.centers)
        metrics = microstate_partial_alignment(Sim.maps_true, Results.centers, ...
            'distance_type', 'cosine', 'threshold', 0.0, 'polarity', true);
    end

    pairs = zeros(0, 2);
    if isstruct(metrics) && isfield(metrics, 'match_assignment') && ~isempty(metrics.match_assignment)
        pairs = double(metrics.match_assignment(:, 1:2));
    end

    counts = zeros(numel(label_order), numel(label_order));
    is_label_match = false(size(pairs, 1), 1);
    similarities = nan(size(pairs, 1), 1);
    for i = 1:size(pairs, 1)
        est_idx = pairs(i, 1);
        true_idx = pairs(i, 2);
        if true_idx < 1 || true_idx > K_true || est_idx < 1 || est_idx > K_est
            continue;
        end
        row = find(strcmp(label_order, true_labels{true_idx}), 1, 'first');
        col = find(strcmp(label_order, est_labels{est_idx}), 1, 'first');
        if ~isempty(row) && ~isempty(col)
            counts(row, col) = counts(row, col) + 1;
        end
        is_label_match(i) = strcmp(true_labels{true_idx}, est_labels{est_idx});
        if isstruct(metrics) && isfield(metrics, 'similarity_matrix') && ...
                size(metrics.similarity_matrix, 1) >= est_idx && size(metrics.similarity_matrix, 2) >= true_idx
            similarities(i) = metrics.similarity_matrix(est_idx, true_idx);
        end
    end

    row_sums = sum(counts, 2);
    denom = max([K_true, K_est, 1]);
    summary = struct();
    summary.identity_accuracy = sum(is_label_match) / denom;
    summary.identity_accuracy_matched = mean(double(is_label_match), 'omitnan');
    summary.n_label_matches = sum(is_label_match);
    summary.n_matched = size(pairs, 1);
    summary.n_unmatched_true = K_true - numel(unique(pairs(:, 2)));
    summary.n_unmatched_estimated = K_est - numel(unique(pairs(:, 1)));
    summary.mean_matched_similarity = mean(similarities, 'omitnan');
    summary.label_confusion_counts = counts;
    summary.label_confusion_row_normalized = counts ./ max(row_sums, 1);
    summary.matched_pairs = pairs;
    summary.match_similarities = similarities;
end

function order = template_order_union(true_alignment, est_alignment, true_labels, est_labels)
    order = {};
    if isstruct(true_alignment) && isfield(true_alignment, 'template_labels') && ~isempty(true_alignment.template_labels)
        order = [order; cellstr(true_alignment.template_labels(:))]; %#ok<AGROW>
    elseif isstruct(est_alignment) && isfield(est_alignment, 'template_labels') && ~isempty(est_alignment.template_labels)
        order = [order; cellstr(est_alignment.template_labels(:))]; %#ok<AGROW>
    end
    order = [order; true_labels(:); est_labels(:)]; %#ok<AGROW>
    order = unique(order, 'stable');
end

function counts = confusion_from_indices(true_idx, pred_idx, n_states)
    counts = zeros(n_states, n_states);
    valid = isfinite(true_idx) & isfinite(pred_idx) & true_idx >= 1 & pred_idx >= 1;
    true_idx = true_idx(valid);
    pred_idx = pred_idx(valid);
    for i = 1:numel(true_idx)
        counts(true_idx(i), pred_idx(i)) = counts(true_idx(i), pred_idx(i)) + 1;
    end
end

function acc = masked_accuracy(pred_idx, true_idx, mask)
    if ~any(mask)
        acc = NaN;
        return;
    end
    acc = mean(pred_idx(mask) == true_idx(mask));
end

function val = masked_weight_mae(pred_weights, true_weights, mask)
    if ~any(mask)
        val = NaN;
        return;
    end
    val = mean(abs(pred_weights(mask, :) - true_weights(mask, :)), 'all');
end

function acc = masked_pair_accuracy(pred_weights, true_weights, mask)
    rows = find(mask(:));
    if isempty(rows)
        acc = NaN;
        return;
    end
    correct = false(numel(rows), 1);
    usable = false(numel(rows), 1);
    for i = 1:numel(rows)
        r = rows(i);
        true_active = find(true_weights(r, :) > 1e-6);
        n_active = numel(true_active);
        if n_active < 2
            continue;
        end
        [pred_vals, pred_order] = sort(pred_weights(r, :), 'descend');
        if numel(pred_order) < n_active || pred_vals(n_active) <= 1e-6
            usable(i) = true;
            continue;
        end
        usable(i) = true;
        correct(i) = isequal(sort(true_active), sort(pred_order(1:n_active)));
    end
    if ~any(usable)
        acc = NaN;
    else
        acc = mean(correct(usable));
    end
end
