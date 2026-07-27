function alignment = align_microstates_to_template(estimated_maps, template_file, varargin)
% ALIGN_MICROSTATES_TO_TEMPLATE Match estimated microstates to MetaMaps templates.
%
% Matching is polarity-invariant and channel-label aware when labels are
% provided.  The returned maps keep the estimated-state order, with polarity
% flipped where needed to match the assigned template.

    p = inputParser;
    addRequired(p, 'estimated_maps', @isnumeric);
    addRequired(p, 'template_file', @(x) ischar(x) || isstring(x));
    addParameter(p, 'estimated_channel_labels', {}, @(x) iscell(x) || isstring(x));
    addParameter(p, 'estimated_chanlocs', [], @(x) isempty(x) || isstruct(x));
    addParameter(p, 'template_K', 7, @isnumeric);
    addParameter(p, 'label_override', {}, @(x) iscell(x) || isstring(x));
    addParameter(p, 'strong_threshold', 0.5, @isnumeric);
    parse(p, estimated_maps, template_file, varargin{:});

    util = microstate_utilities();
    estimated_maps = util.normalize_maps(double(estimated_maps));
    [template_maps, template_labels, template_channel_labels, template_chanlocs] = ...
        load_metamaps_templates(template_file, 'K', p.Results.template_K);

    est_labels = cellstr(p.Results.estimated_channel_labels);
    tmpl_labels = cellstr(template_channel_labels);
    n_est_channels = size(estimated_maps, 2);
    n_template_channels = size(template_maps, 2);
    min_label_matches = max(16, ceil(0.5 * min(n_est_channels, n_template_channels)));

    if ~isempty(est_labels) && numel(est_labels) >= n_est_channels && ~isempty(tmpl_labels)
        [common_labels, idx_est, idx_template] = intersect( ...
            lower(strtrim(est_labels(1:n_est_channels))), ...
            lower(strtrim(tmpl_labels(1:n_template_channels))), 'stable');
        if numel(common_labels) >= min_label_matches
            estimated_common = estimated_maps(:, idx_est);
            template_common = template_maps(:, idx_template);
            channel_match_mode = 'labels';
        else
            [estimated_common, template_common, channel_match_mode, common_labels, idx_est, idx_template] = ...
                coordinate_or_first_common(estimated_maps, template_maps, p.Results.estimated_chanlocs, ...
                template_chanlocs, tmpl_labels, util);
        end
    else
        [estimated_common, template_common, channel_match_mode, common_labels, idx_est, idx_template] = ...
            coordinate_or_first_common(estimated_maps, template_maps, p.Results.estimated_chanlocs, ...
            template_chanlocs, tmpl_labels, util);
    end

    estimated_common = util.normalize_maps(estimated_common);
    template_common = util.normalize_maps(template_common);
    signed_corr = estimated_common * template_common';
    corr_matrix = abs(signed_corr);

    n_est = size(estimated_common, 1);
    n_template = size(template_common, 1);
    label_override = cellstr(string(p.Results.label_override(:)));
    if isempty(label_override)
        template_idx = optimal_template_assignment(corr_matrix);
    else
        template_idx = assignment_from_label_override(label_override, template_labels, n_est);
    end
    template_corr = zeros(n_est, 1);
    polarity = ones(n_est, 1);
    for e = 1:n_est
        t = template_idx(e);
        if ~isnan(t)
            template_corr(e) = corr_matrix(e, t);
            if signed_corr(e, t) < 0
                polarity(e) = -1;
            end
        end
    end

    labels = cell(n_est, 1);
    for e = 1:n_est
        if ~isnan(template_idx(e))
            labels{e} = template_labels{template_idx(e)};
        else
            labels{e} = sprintf('X%d', e);
        end
    end

    aligned_maps = estimated_maps;
    for e = 1:n_est
        aligned_maps(e, :) = polarity(e) * aligned_maps(e, :);
    end

    strong = template_corr >= p.Results.strong_threshold;
    alignment = struct( ...
        'labels', {labels}, ...
        'correlations', template_corr, ...
        'template_indices', template_idx, ...
        'polarity', polarity, ...
        'aligned_maps', aligned_maps, ...
        'corr_matrix', corr_matrix, ...
        'estimated_common_maps', estimated_common, ...
        'template_common_maps', template_common, ...
        'template_labels', {template_labels}, ...
        'label_override', {label_override}, ...
        'channel_match_mode', channel_match_mode, ...
        'matched_channel_labels', {common_labels}, ...
        'min_label_matches', min_label_matches, ...
        'estimated_channel_indices', idx_est, ...
        'template_channel_indices', idx_template, ...
        'n_common_channels', size(estimated_common, 2), ...
        'mean_correlation', mean(template_corr(template_corr > 0)), ...
        'median_correlation', median(template_corr(template_corr > 0)), ...
        'min_correlation', min(template_corr(template_corr > 0)), ...
        'n_strong_matches', sum(strong), ...
        'strong_threshold', p.Results.strong_threshold);
end

function template_idx = assignment_from_label_override(labels, template_labels, n_est)
    if numel(labels) ~= n_est
        error('label_override has %d labels, but the fitted solution has %d states.', numel(labels), n_est);
    end
    if numel(unique(lower(strtrim(string(labels(:)))))) ~= n_est
        error('label_override labels must be unique.');
    end
    template_idx = nan(n_est, 1);
    keys = lower(strtrim(string(template_labels(:))));
    for e = 1:n_est
        hit = find(keys == lower(strtrim(string(labels{e}))), 1, 'first');
        if isempty(hit)
            error('label_override label "%s" was not found in the selected template set.', labels{e});
        end
        template_idx(e) = hit;
    end
end

function [estimated_common, template_common, mode, common_labels, idx_est, idx_template] = coordinate_or_first_common(estimated_maps, template_maps, estimated_chanlocs, template_chanlocs, template_labels, util)
    [estimated_common, template_common, mode, common_labels, idx_est, idx_template] = ...
        use_coordinate_interpolation(estimated_maps, template_maps, estimated_chanlocs, template_chanlocs, template_labels, util);
    if ~isempty(estimated_common)
        return;
    end

    [estimated_common, template_common, mode, idx_est, idx_template] = ...
        use_first_common_channels(estimated_maps, template_maps);
    common_labels = arrayfun(@(i) sprintf('Ch%03d', i), idx_est, 'UniformOutput', false);
end

function [estimated_common, template_common, mode, common_labels, idx_est, idx_template] = use_coordinate_interpolation(estimated_maps, template_maps, estimated_chanlocs, template_chanlocs, template_labels, util)
    estimated_common = [];
    template_common = [];
    mode = '';
    common_labels = {};
    idx_est = [];
    idx_template = [];
    if isempty(estimated_chanlocs) || isempty(template_chanlocs)
        return;
    end

    est_pos = util.positions_from_chanlocs(estimated_chanlocs, size(estimated_maps, 2));
    tmpl_pos = util.positions_from_chanlocs(template_chanlocs, size(template_maps, 2));
    est_ok = all(isfinite(est_pos), 2);
    tmpl_ok = all(isfinite(tmpl_pos), 2);
    if sum(est_ok) < 4 || sum(tmpl_ok) < 4
        return;
    end

    est_pos = normalise_rows(est_pos(est_ok, :));
    tmpl_pos = normalise_rows(tmpl_pos(tmpl_ok, :));
    idx_est = find(est_ok);
    idx_template = find(tmpl_ok);
    estimated_common = idw_interpolate_maps(estimated_maps(:, idx_est), est_pos, tmpl_pos, 6);
    template_common = template_maps(:, idx_template);
    common_labels = template_labels(idx_template);
    mode = 'coordinates_idw_to_template';
end

function pos = normalise_rows(pos)
    norms = sqrt(sum(pos.^2, 2));
    norms(norms < eps) = 1;
    pos = pos ./ norms;
end

function maps_out = idw_interpolate_maps(maps_in, source_pos, target_pos, n_neighbours)
    n_neighbours = min(max(1, n_neighbours), size(source_pos, 1));
    maps_out = zeros(size(maps_in, 1), size(target_pos, 1));
    for t = 1:size(target_pos, 1)
        d = sqrt(sum((source_pos - target_pos(t, :)).^2, 2));
        [ds, ord] = sort(d, 'ascend');
        ord = ord(1:n_neighbours);
        ds = ds(1:n_neighbours);
        if ds(1) <= eps
            w = zeros(numel(ord), 1);
            w(1) = 1;
        else
            w = 1 ./ (ds.^2 + eps);
            w = w ./ sum(w);
        end
        maps_out(:, t) = maps_in(:, ord) * w;
    end
end

function [estimated_common, template_common, mode, idx_est, idx_template] = use_first_common_channels(estimated_maps, template_maps)
    n_common = min(size(estimated_maps, 2), size(template_maps, 2));
    idx_est = 1:n_common;
    idx_template = 1:n_common;
    estimated_common = estimated_maps(:, idx_est);
    template_common = template_maps(:, idx_template);
    mode = 'first_common_channels';
end

function template_idx = optimal_template_assignment(corr_matrix)
%OPTIMAL_TEMPLATE_ASSIGNMENT Maximise total absolute correlation.
% K is small for MetaMaps, so exact permutation search is simpler and more
% reliable than relying on toolbox-specific Hungarian implementations.

    [n_est, n_template] = size(corr_matrix);
    template_idx = nan(n_est, 1);

    if n_est <= n_template
        combos = nchoosek(1:n_template, n_est);
        best_score = -Inf;
        best_assignment = [];
        for ci = 1:size(combos, 1)
            perms_this = perms(combos(ci, :));
            for pi = 1:size(perms_this, 1)
                assignment = perms_this(pi, :);
                score = sum(corr_matrix(sub2ind(size(corr_matrix), 1:n_est, assignment)));
                if score > best_score
                    best_score = score;
                    best_assignment = assignment;
                end
            end
        end
        template_idx(:) = best_assignment(:);
        return;
    end

    combos = nchoosek(1:n_est, n_template);
    best_score = -Inf;
    best_est = [];
    best_assignment = [];
    for ci = 1:size(combos, 1)
        est_idx = combos(ci, :);
        perms_this = perms(1:n_template);
        for pi = 1:size(perms_this, 1)
            assignment = perms_this(pi, :);
            score = sum(corr_matrix(sub2ind(size(corr_matrix), est_idx, assignment)));
            if score > best_score
                best_score = score;
                best_est = est_idx;
                best_assignment = assignment;
            end
        end
    end
    template_idx(best_est) = best_assignment;
end
