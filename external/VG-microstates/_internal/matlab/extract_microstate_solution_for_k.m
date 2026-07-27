function Selected = extract_microstate_solution_for_k(Results, K_selected, varargin)
% EXTRACT_MICROSTATE_SOLUTION_FOR_K Build a criterion-specific result view.
%
% This helper materialises the map/label solution associated with a
% particular K from a fit result that stores per-K candidate solutions.

    p = inputParser;
    addRequired(p, 'Results', @isstruct);
    addRequired(p, 'K_selected', @(x) isnumeric(x) && isscalar(x) && isfinite(x));
    addParameter(p, 'Sim', struct(), @isstruct);
    addParameter(p, 'criterion', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'template_file', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'estimated_channel_labels', {}, @(x) iscell(x) || isstring(x));
    addParameter(p, 'template_K', 7, @(x) isnumeric(x) && isscalar(x) && x >= 1);
    addParameter(p, 'strong_threshold', 0.5, @(x) isnumeric(x) && isscalar(x) && x >= 0 && x <= 1);
    addParameter(p, 'compute_backfit_timecourse', false, @islogical);
    parse(p, Results, K_selected, varargin{:});

    Selected = Results;
    Selected.K_estimated = K_selected;
    if isfield(Selected, 'K_model_selected')
        Selected.K_model_selected = K_selected;
    end
    if ~isempty(p.Results.criterion)
        Selected.criterion = char(p.Results.criterion);
    end

    idx = find_result_k_index(Results, K_selected);
    if isnan(idx)
        if isfield(Results, 'K_estimated') && isequal(double(Results.K_estimated), double(K_selected))
            idx = NaN;
        else
            error('No stored solution was found for K=%g.', K_selected);
        end
    end

    if ~isnan(idx)
        Selected.selected_solution_index = idx;
        if isfield(Results, 'centers_by_K') && numel(Results.centers_by_K) >= idx && ~isempty(Results.centers_by_K{idx})
            Selected.centers = Results.centers_by_K{idx};
        end
        if isfield(Results, 'labels_by_K') && numel(Results.labels_by_K) >= idx && ~isempty(Results.labels_by_K{idx})
            Selected.labels = Results.labels_by_K{idx};
        end
        if isfield(Results, 'spm_mix_model_summaries') && numel(Results.spm_mix_model_summaries) >= idx
            Selected.selected_spm_mix_model = Results.spm_mix_model_summaries(idx);
            if isfield(Selected.selected_spm_mix_model, 'covariances')
                Selected.selected_spm_covariances = Selected.selected_spm_mix_model.covariances;
            end
            if isfield(Selected.selected_spm_mix_model, 'means')
                Selected.selected_spm_means = Selected.selected_spm_mix_model.means;
            end
            if isfield(Selected.selected_spm_mix_model, 'priors')
                Selected.selected_spm_priors = Selected.selected_spm_mix_model.priors;
            end
        end
        if isfield(Results, 'feature_backfit_models') && numel(Results.feature_backfit_models) >= idx
            Selected.selected_feature_backfit_model = Results.feature_backfit_models{idx};
        end
        Selected.best_criterion_value = lookup_selected_score(Results, idx, Selected.criterion);
    end

    if isfield(Selected, 'labels') && ~isempty(Selected.labels) && isfield(Selected, 'centers') && ~isempty(Selected.centers)
        K_eff = size(Selected.centers, 1);
        cluster_weights = zeros(1, K_eff);
        for k = 1:K_eff
            cluster_weights(k) = mean(Selected.labels == k);
        end
        Selected.cluster_weights = cluster_weights / (sum(cluster_weights) + eps);
    end

    if isfield(p.Results.Sim, 'maps_true') && ~isempty(p.Results.Sim.maps_true) && isfield(Selected, 'centers') && ~isempty(Selected.centers)
        util = microstate_utilities();
        true_maps_norm = util.normalize_maps(p.Results.Sim.maps_true);
        Selected.maps_true = true_maps_norm;
        Selected.recovery_metrics = microstate_partial_alignment(true_maps_norm, Selected.centers, ...
            'distance_type', 'cosine', 'threshold', 0.0, 'polarity', true);
        Selected.mean_recovery = Selected.recovery_metrics.mean_recovery_matched;
        Selected.recovery_corr = Selected.recovery_metrics.match_similarities;
        Selected.avg_recovery_per_state = Selected.recovery_metrics.mean_recovery_padded;
    end

    template_file = char(p.Results.template_file);
    if ~isempty(template_file) && isfile(template_file) && isfield(Selected, 'centers') && ~isempty(Selected.centers)
        try
            Selected.template_alignment = align_microstates_to_template(Selected.centers, template_file, ...
                'estimated_channel_labels', p.Results.estimated_channel_labels, ...
                'template_K', p.Results.template_K, ...
                'strong_threshold', p.Results.strong_threshold);
        catch
        end
    end

    if p.Results.compute_backfit_timecourse && ...
            isfield(p.Results.Sim, 'X_noisy') && ~isempty(p.Results.Sim.X_noisy) && ...
            isfield(Selected, 'centers') && ~isempty(Selected.centers)
        try
            Selected.backfit_timecourse = backfit_microstate_timecourse(p.Results.Sim, Selected);
        catch
        end
    end
end

function idx = find_result_k_index(Results, K_selected)
    idx = NaN;
    if ~isfield(Results, 'K_candidates') || isempty(Results.K_candidates)
        return;
    end
    hit = find(double(Results.K_candidates(:)) == double(K_selected), 1, 'first');
    if ~isempty(hit)
        idx = hit;
    end
end

function score = lookup_selected_score(Results, idx, criterion)
    score = NaN;
    criterion = lower(strtrim(char(string(criterion))));
    if isfield(Results, 'method') && strcmpi(char(string(Results.method)), 'spm_vb')
        [~, score, score_by_k] = select_spm_vb_k_by_criterion(Results, criterion);
        if numel(score_by_k) >= idx && isfinite(score_by_k(idx))
            score = score_by_k(idx);
        end
        return;
    end

    switch criterion
        case 'free_energy'
            if isfield(Results, 'free_energy_vals') && numel(Results.free_energy_vals) >= idx
                score = Results.free_energy_vals(idx);
            end
        case 'silhouette'
            if isfield(Results, 'silhouette_vals') && numel(Results.silhouette_vals) >= idx
                score = Results.silhouette_vals(idx);
            end
        case 'gev'
            if isfield(Results, 'gev_vals') && numel(Results.gev_vals) >= idx
                score = Results.gev_vals(idx);
            end
        case 'elbow'
            if isfield(Results, 'within_ss') && numel(Results.within_ss) >= idx
                score = Results.within_ss(idx);
            end
        otherwise
            if isfield(Results, 'best_criterion_value')
                score = Results.best_criterion_value;
            end
    end
end
