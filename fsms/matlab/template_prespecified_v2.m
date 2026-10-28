function result = template_prespecified_v2(centers, common_labels, common_chanlocs, varargin)
%TEMPLATE_PRESPECIFIED_V2 Pre-specified, deterministic NAMING of the global K=6 microstate solution.
%
%   RULE (registered; constants live in template_prespecified_v2_constants.m)
%     1. Spatial correlation: |r| between each of the six fitted global maps and every MetaMaps_2023_06
%        K=7 template map (polarity invariant). Channels are matched by label when >= 16 labels match,
%        otherwise by interpolating the fitted maps to the template electrode positions (inverse-distance
%        weighting on unit-sphere coordinates). The silent 'first_common_channels' fallback of
%        align_microstates_to_template is NOT accepted.
%     2. Class set: the six classes {A,B,C,E,F,G}. Class D is excluded a priori.
%     3. Assignment: the exact optimal one-to-one assignment of fitted states to the six classes that
%        maximises the SUM of assigned |r| (all 720 permutations are evaluated). Ties (total |r| within
%        1e-9) are broken deterministically (lexicographically smallest assignment vector, where entry k is
%        the index A,B,C,E,F,G = 1..6 of the class given to fitted state k) and are reported; an exact tie
%        is a TECHNICAL failure (identical fitted maps: the naming is ill-defined).
%     4. Quality flag (descriptive, never a stop): labelling_quality = 'ok' when every assigned |r| >= 0.5,
%        otherwise 'below_threshold' with the list of states below 0.5. State names are nominal; primary and
%        secondary outcomes do not depend on them; below_threshold means descriptive per-state names are
%        unreliable.
%     5. Technical failures (result.accepted = false; the caller writes LABELLING_REJECTED.json and stops):
%        channel-matching mode not in {labels, coordinates_idw_to_template} (e.g. first_common_channels);
%        fewer than 8 fitted-map channels in the comparison; non-finite maps or correlations; labels not
%        unique; tied optimal assignments. (Fewer fitted states than classes is checked by the caller.)
%
%   result = template_prespecified_v2(centers, common_labels, common_chanlocs, 'template_file', F)
%   centers is 6 x C (fitted state x channel), common_labels is a C-element cellstr, and
%   common_chanlocs is the EEGLAB-style struct array of those channels (positions in MNE RAS head
%   coordinates, as written by the FIF->MAT converter) or [].
%
%   This function is pure: it reads the template and returns the evidence; it writes nothing.

    p = inputParser;
    addParameter(p, 'template_file', '', @(x) ischar(x) || isstring(x));
    parse(p, varargin{:});
    const = template_prespecified_v2_constants();
    template_file = char(string(p.Results.template_file));
    if isempty(template_file)
        env = fsms_paths();
        template_file = fullfile(env.vg_root, 'MetaMaps_2023_06.set');
    end

    centers = double(centers);
    if ~ismatrix(centers) || size(centers, 1) ~= const.K
        error('FSMS:LabellingBadInput', 'Pre-specified labelling needs exactly K=%d fitted maps; got %d.', ...
            const.K, size(centers, 1));
    end
    common_labels = cellstr(string(common_labels(:)));
    if numel(common_labels) ~= size(centers, 2)
        error('FSMS:LabellingBadInput', 'common_labels has %d entries but the maps have %d channels.', ...
            numel(common_labels), size(centers, 2));
    end
    reasons = {};
    if any(~isfinite(centers(:)))
        error('FSMS:LabellingBadInput', 'Fitted global maps contain non-finite values.');
    end

    alignment = align_microstates_to_template(centers, template_file, ...
        'estimated_channel_labels', common_labels, ...
        'estimated_chanlocs', common_chanlocs, ...
        'template_K', const.template_K, ...
        'strong_threshold', const.strong_threshold);
    template_order = cellstr(string(alignment.template_labels(:)));
    full_abs = alignment.corr_matrix;                        % 6 x 7, |r|, template classes alphabetical
    class_columns = zeros(1, numel(const.classes));
    for c = 1:numel(const.classes)
        hit = find(strcmp(template_order, const.classes{c}), 1);
        if isempty(hit)
            error('FSMS:LabellingTemplate', 'Template %s lacks class %s.', template_file, const.classes{c});
        end
        class_columns(c) = hit;
    end
    corr_abs = full_abs(:, class_columns);                   % 6 x 6, columns A,B,C,E,F,G
    signed_full = alignment.estimated_common_maps * alignment.template_common_maps';
    corr_signed = signed_full(:, class_columns);

    mode = char(string(alignment.channel_match_mode));
    n_used = numel(alignment.estimated_channel_indices);
    if ~any(strcmp(mode, const.allowed_modes))
        reasons{end + 1} = sprintf('CHANNEL_MATCH_MODE_NOT_ALLOWED:%s', mode); %#ok<AGROW>
    end
    if n_used < const.min_estimated_channels
        reasons{end + 1} = sprintf('TOO_FEW_COMMON_CHANNELS:n=%d,minimum=%d', n_used, const.min_estimated_channels); %#ok<AGROW>
    end
    if any(~isfinite(corr_abs(:)))
        reasons{end + 1} = 'NONFINITE_CORRELATION'; %#ok<AGROW>
    end

    % Exact optimal one-to-one assignment (720 permutations, lexicographic order, deterministic ties).
    perm_table = sortrows(perms(1:const.K));                 % ascending lexicographic
    score = zeros(size(perm_table, 1), 1);
    for r = 1:size(perm_table, 1)
        score(r) = sum(corr_abs(sub2ind(size(corr_abs), 1:const.K, perm_table(r, :))));
    end
    safe_score = score;
    safe_score(~isfinite(safe_score)) = -Inf;
    best_total = max(safe_score);
    best_row = find(safe_score >= best_total - const.tie_tolerance, 1, 'first');
    n_ties = sum(safe_score >= best_total - const.tie_tolerance);
    if n_ties > 1
        reasons{end + 1} = sprintf('AMBIGUOUS_ASSIGNMENT:%d_tied_optimal_assignments', n_ties); %#ok<AGROW>
    end
    assignment = perm_table(best_row, :)';                   % state k -> class index
    lower_scores = safe_score(safe_score < best_total - const.tie_tolerance);
    if isempty(lower_scores)
        runner_up = NaN;
    else
        runner_up = max(lower_scores);
    end
    labels = const.classes(assignment)';
    assigned_abs = zeros(const.K, 1);
    assigned_signed = zeros(const.K, 1);
    next_best_class = cell(const.K, 1);
    next_best_abs = zeros(const.K, 1);
    for k = 1:const.K
        assigned_abs(k) = corr_abs(k, assignment(k));
        assigned_signed(k) = corr_signed(k, assignment(k));
        others = corr_abs(k, :);
        others(assignment(k)) = -Inf;
        [next_best_abs(k), j] = max(others);
        next_best_class{k} = const.classes{j};
    end
    if numel(unique(labels)) ~= const.K
        reasons{end + 1} = 'LABELS_NOT_UNIQUE'; %#ok<AGROW>
    end
    polarity = ones(const.K, 1);
    polarity(assigned_signed < 0) = -1;

    below = find(~(assigned_abs >= const.strong_threshold));
    if isempty(below)
        quality = 'ok';
    else
        quality = 'below_threshold';
    end
    states_below = struct('state_index', num2cell(below(:))', 'label', labels(below)', ...
        'assigned_abs_r', num2cell(assigned_abs(below))');
    if isempty(below)
        states_below = struct('state_index', {}, 'label', {}, 'assigned_abs_r', {});
    end

    result = struct();
    result.rule_id = const.rule_id;
    result.accepted = isempty(reasons);                      % false only for TECHNICAL failures
    result.reasons = reasons(:);
    result.labelling_quality = quality;
    result.states_below_threshold = states_below;
    result.nominal_label_semantics = const.nominal_label_semantics;
    result.classes = const.classes(:);
    result.excluded_classes = const.excluded_classes(:);
    result.labels = labels;
    result.state_index = (1:const.K)';
    result.assignment_class_index = assignment;
    result.assigned_abs_r = assigned_abs;
    result.assigned_signed_r = assigned_signed;
    result.polarity = polarity;
    result.next_best_class = next_best_class;
    result.next_best_abs_r = next_best_abs;
    result.corr_abs = corr_abs;
    result.corr_abs_all_template_classes = full_abs;
    result.template_class_order = template_order;
    result.total_abs_r = best_total;
    result.runner_up_total_abs_r = runner_up;
    result.margin_total_abs_r = best_total - runner_up;
    result.n_tied_optimal_assignments = n_ties;
    result.strong_threshold = const.strong_threshold;
    result.tie_tolerance = const.tie_tolerance;
    result.channel_match_mode = mode;
    result.n_common_channels = double(alignment.n_common_channels);
    result.n_estimated_channels_used = n_used;
    result.min_label_matches = double(alignment.min_label_matches);
    result.matched_channel_labels = cellstr(string(alignment.matched_channel_labels(:)));
    result.template_file = template_file;
    result.template_K = const.template_K;
end
