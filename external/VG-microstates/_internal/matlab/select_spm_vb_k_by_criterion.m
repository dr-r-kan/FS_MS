function [K_selected, best_score, score_by_k, details] = select_spm_vb_k_by_criterion(Results, criterion)
%SELECT_SPM_VB_K_BY_CRITERION Recompute VB-GMM model selection from Results.
%
% Supports both the legacy criteria and the covariance-aware additions:
%   - gev / gfp
%   - log_likelihood / ll
%   - bic
%   - icl
%   - calinski_harabasz_score
%   - covariance
%   - covariance_elbow
%   - free_energy_covariance

    if nargin < 2 || isempty(criterion)
        criterion = 'elbow_sil_combined';
    end
    criterion = lower(strtrim(char(string(criterion))));

    metrics = extract_selection_metrics_local(Results);
    K_all = metrics.K_candidates(:);
    score_by_k = nan(size(K_all));
    details = struct('criterion', criterion, 'metrics', metrics);
    K_selected = NaN;
    best_score = NaN;

    if isempty(K_all)
        return;
    end

    valid_mask = isfinite(metrics.free_energy) & metrics.free_energy ~= 0;
    if ~any(valid_mask)
        return;
    end

    K_valid = K_all(valid_mask);
    fe_valid = metrics.free_energy(valid_mask);
    ll_valid = metrics.log_likelihood(valid_mask);
    bic_valid = metrics.bic(valid_mask);
    icl_valid = metrics.icl(valid_mask);
    sil_valid = metrics.silhouette(valid_mask);
    gev_valid = metrics.gev(valid_mask);
    ch_valid = metrics.calinski_harabasz(valid_mask);
    cov_valid = select_primary_covariance_metric_local(metrics, valid_mask);

    switch criterion
        case {'gev', 'gfp', 'global_explained_variance'}
            [best_score, best_local] = max(gev_valid);
            K_selected = K_valid(best_local);
            score_by_k(valid_mask) = gev_valid;

        case {'calinski_harabasz', 'calinski_harabasz_score', 'ch'}
            if ~any(isfinite(ch_valid))
                return;
            end
            local_scores = ch_valid;
            local_scores(~isfinite(local_scores)) = -Inf;
            [best_score, best_local] = max(local_scores);
            K_selected = K_valid(best_local);
            score_by_k(valid_mask) = ch_valid;

        case 'silhouette'
            local_scores = sil_valid;
            if numel(local_scores) > 4
                search_idx = 2:(numel(local_scores) - 1);
                [best_score, best_local] = max(local_scores(search_idx));
                best_local = search_idx(best_local);
            else
                [best_score, best_local] = max(local_scores);
            end
            K_selected = K_valid(best_local);
            score_by_k(valid_mask) = local_scores;

        case 'free_energy'
            [best_score, best_local] = max(fe_valid);
            K_selected = K_valid(best_local);
            score_by_k(valid_mask) = fe_valid;

        case {'log_likelihood', 'll'}
            if ~any(isfinite(ll_valid))
                return;
            end
            local_scores = ll_valid;
            local_scores(~isfinite(local_scores)) = -Inf;
            [best_score, best_local] = max(local_scores);
            K_selected = K_valid(best_local);
            score_by_k(valid_mask) = ll_valid;

        case 'bic'
            if ~any(isfinite(bic_valid))
                return;
            end
            local_scores = bic_valid;
            local_scores(~isfinite(local_scores)) = -Inf;
            [best_score, best_local] = max(local_scores);
            K_selected = K_valid(best_local);
            score_by_k(valid_mask) = bic_valid;

        case 'icl'
            if ~any(isfinite(icl_valid))
                return;
            end
            local_scores = icl_valid;
            local_scores(~isfinite(local_scores)) = -Inf;
            [best_score, best_local] = max(local_scores);
            K_selected = K_valid(best_local);
            score_by_k(valid_mask) = icl_valid;

        case {'free_energy_elbow', 'elbow'}
            [K_selected, local_scores, elbow_info] = elbow_select_local(fe_valid, K_valid, 'increasing');
            score_by_k(valid_mask) = local_scores;
            details.free_energy_elbow = elbow_info;
            best_score = score_at_k_local(K_selected, K_all, score_by_k);

        case {'elbow_sil_combined', 'elbow_only'}
            local_scores = elbow_silhouette_score_local(fe_valid, K_valid, sil_valid);
            [best_score, best_local] = max(local_scores);
            K_selected = K_valid(best_local);
            score_by_k(valid_mask) = local_scores;

        case {'covariance', 'covariance_raw', 'covariance_min'}
            if ~any(isfinite(cov_valid))
                return;
            end
            local_scores = max(cov_valid) - cov_valid;
            [best_score, best_local] = max(local_scores);
            K_selected = K_valid(best_local);
            score_by_k(valid_mask) = local_scores;

        case 'covariance_elbow'
            if ~any(isfinite(cov_valid))
                return;
            end
            [K_selected, local_scores, cov_info] = elbow_select_local(cov_valid, K_valid, 'decreasing');
            score_by_k(valid_mask) = local_scores;
            details.covariance_elbow = cov_info;
            best_score = score_at_k_local(K_selected, K_all, score_by_k);

        case {'free_energy_covariance', 'covariance_free_energy', 'free_energy_covariance_hybrid'}
            if ~any(isfinite(cov_valid))
                [best_score, best_local] = max(fe_valid);
                K_selected = K_valid(best_local);
                score_by_k(valid_mask) = fe_valid;
                return;
            end
            [~, fe_elbow, fe_info] = elbow_select_local(fe_valid, K_valid, 'increasing');
            [~, cov_elbow, cov_info] = elbow_select_local(cov_valid, K_valid, 'decreasing');
            fe_elbow_n = normalize_01_local(fe_elbow);
            cov_elbow_n = normalize_01_local(cov_elbow);
            cov_tight_n = normalize_01_local(max(cov_valid) - cov_valid);
            fe_raw_n = normalize_01_local(fe_valid);

            % Main signal is the agreement between the free-energy elbow and
            % the covariance elbow. Smaller covariance is kept as a weak
            % tie-breaker rather than the dominant term.
            local_scores = 0.45 * fe_elbow_n + 0.35 * cov_elbow_n + ...
                0.15 * cov_tight_n + 0.05 * fe_raw_n;
            [best_score, best_local] = max(local_scores);
            K_selected = K_valid(best_local);
            score_by_k(valid_mask) = local_scores;
            details.free_energy_elbow = fe_info;
            details.covariance_elbow = cov_info;
            details.components = struct( ...
                'free_energy_elbow', fe_elbow, ...
                'covariance_elbow', cov_elbow, ...
                'covariance_tightness', cov_tight_n, ...
                'free_energy_raw', fe_raw_n);

        otherwise
            return;
    end
end

function metrics = extract_selection_metrics_local(Results)
    metrics = struct();
    metrics.K_candidates = coerce_numeric_vector_local(field_or_local(Results, 'K_candidates', []));
    metrics.free_energy = coerce_numeric_vector_local(field_or_local(Results, 'free_energy_vals', []));
    if isempty(metrics.free_energy)
        metrics.free_energy = coerce_numeric_vector_local(field_or_local(Results, 'free_energy', []));
    end
    metrics.log_likelihood = coerce_numeric_vector_local(field_or_local(Results, 'log_likelihood_vals', []));
    metrics.bic = coerce_numeric_vector_local(field_or_local(Results, 'bic_vals', []));
    if isempty(metrics.bic)
        metrics.bic = metrics.free_energy;
    end
    metrics.icl = coerce_numeric_vector_local(field_or_local(Results, 'icl_vals', []));
    metrics.silhouette = coerce_numeric_vector_local(field_or_local(Results, 'silhouette_vals', []));
    metrics.gev = coerce_numeric_vector_local(field_or_local(Results, 'gev_vals', []));
    metrics.calinski_harabasz = coerce_numeric_vector_local(field_or_local(Results, 'calinski_harabasz_vals', []));
    metrics.covariance_trace_mean = coerce_numeric_vector_local(field_or_local(Results, 'covariance_trace_mean_vals', []));
    metrics.covariance_trace_median = coerce_numeric_vector_local(field_or_local(Results, 'covariance_trace_median_vals', []));
    metrics.covariance_logdet_mean = coerce_numeric_vector_local(field_or_local(Results, 'covariance_logdet_mean_vals', []));
    metrics.covariance_logdet_median = coerce_numeric_vector_local(field_or_local(Results, 'covariance_logdet_median_vals', []));
    metrics.covariance_logdet_per_dim_mean = coerce_numeric_vector_local(field_or_local(Results, 'covariance_logdet_per_dim_mean_vals', []));

    if isempty(metrics.covariance_trace_mean) || isempty(metrics.covariance_logdet_mean)
        [trace_mean, trace_median, logdet_mean, logdet_median, logdet_per_dim] = ...
            covariance_arrays_from_summaries_local(field_or_local(Results, 'spm_mix_model_summaries', []));
        if isempty(metrics.covariance_trace_mean), metrics.covariance_trace_mean = trace_mean; end
        if isempty(metrics.covariance_trace_median), metrics.covariance_trace_median = trace_median; end
        if isempty(metrics.covariance_logdet_mean), metrics.covariance_logdet_mean = logdet_mean; end
        if isempty(metrics.covariance_logdet_median), metrics.covariance_logdet_median = logdet_median; end
        if isempty(metrics.covariance_logdet_per_dim_mean), metrics.covariance_logdet_per_dim_mean = logdet_per_dim; end
    end

    nK = numel(metrics.K_candidates);
    metrics.free_energy = resize_vector_local(metrics.free_energy, nK, -Inf);
    metrics.log_likelihood = resize_vector_local(metrics.log_likelihood, nK, NaN);
    metrics.bic = resize_vector_local(metrics.bic, nK, NaN);
    metrics.icl = resize_vector_local(metrics.icl, nK, NaN);
    metrics.silhouette = resize_vector_local(metrics.silhouette, nK, NaN);
    metrics.gev = resize_vector_local(metrics.gev, nK, NaN);
    metrics.calinski_harabasz = resize_vector_local(metrics.calinski_harabasz, nK, NaN);
    metrics.covariance_trace_mean = resize_vector_local(metrics.covariance_trace_mean, nK, NaN);
    metrics.covariance_trace_median = resize_vector_local(metrics.covariance_trace_median, nK, NaN);
    metrics.covariance_logdet_mean = resize_vector_local(metrics.covariance_logdet_mean, nK, NaN);
    metrics.covariance_logdet_median = resize_vector_local(metrics.covariance_logdet_median, nK, NaN);
    metrics.covariance_logdet_per_dim_mean = resize_vector_local(metrics.covariance_logdet_per_dim_mean, nK, NaN);
end

function cov_metric = select_primary_covariance_metric_local(metrics, valid_mask)
    candidates = { ...
        metrics.covariance_logdet_per_dim_mean, ...
        metrics.covariance_logdet_mean, ...
        metrics.covariance_trace_mean, ...
        metrics.covariance_trace_median};

    cov_metric = nan(sum(valid_mask), 1);
    for i = 1:numel(candidates)
        values = candidates{i};
        if isempty(values)
            continue;
        end
        values = values(valid_mask);
        finite_vals = values(isfinite(values));
        if numel(finite_vals) >= 2 && range(finite_vals) > eps
            cov_metric = values;
            return;
        end
    end
end

function [K_est, score_by_k, info] = elbow_select_local(values, K_candidates, trend)
    values = values(:);
    K_candidates = K_candidates(:);
    n = numel(values);
    score_by_k = zeros(n, 1);
    info = struct('trend', trend, 'curve', values);

    finite_mask = isfinite(values);
    if ~any(finite_mask)
        K_est = NaN;
        score_by_k(:) = NaN;
        return;
    end

    if n < 3
        if strcmpi(trend, 'decreasing')
            [~, idx] = min(values);
        else
            [~, idx] = max(values);
        end
        K_est = K_candidates(idx);
        score_by_k(idx) = values(idx);
        return;
    end

    y = values;
    if strcmpi(trend, 'decreasing')
        y = max(y) - y;
    end
    y_norm = normalize_01_local(y);
    k_norm = normalize_01_local(K_candidates);

    p1 = [k_norm(1), y_norm(1)];
    p2 = [k_norm(end), y_norm(end)];
    for i = 2:(n - 1)
        p = [k_norm(i), y_norm(i)];
        score_by_k(i) = distance_from_line_local(p1, p2, p);
    end

    if all(score_by_k <= eps)
        if strcmpi(trend, 'decreasing')
            [~, idx] = min(values);
        else
            [~, idx] = max(values);
        end
    else
        [~, idx] = max(score_by_k);
    end
    K_est = K_candidates(idx);
    info.curvature = score_by_k;
    info.normalized_curve = y_norm;
end

function scores = elbow_silhouette_score_local(fe_valid, K_valid, sil_valid)
    fe_norm = normalize_01_local(fe_valid);
    k_norm = normalize_01_local(K_valid);
    elbow_scores = zeros(size(fe_norm));
    p1 = [k_norm(1), fe_norm(1)];
    p2 = [k_norm(end), fe_norm(end)];
    for i = 2:(numel(fe_norm) - 1)
        p = [k_norm(i), fe_norm(i)];
        elbow_scores(i) = distance_from_line_local(p1, p2, p);
    end

    [~, elbow_idx] = max(elbow_scores);
    K_elbow = K_valid(elbow_idx);

    scores = zeros(size(K_valid));
    for i = 1:numel(K_valid)
        elbow_penalty = exp(-abs(K_valid(i) - K_elbow));
        sil_bonus = (sil_valid(i) + 1) / 2;
        scores(i) = 0.6 * elbow_penalty + 0.4 * sil_bonus;
    end
end

function [trace_mean, trace_median, logdet_mean, logdet_median, logdet_per_dim] = covariance_arrays_from_summaries_local(summaries)
    n = numel(summaries);
    trace_mean = nan(n, 1);
    trace_median = nan(n, 1);
    logdet_mean = nan(n, 1);
    logdet_median = nan(n, 1);
    logdet_per_dim = nan(n, 1);
    for i = 1:n
        summary = summaries(i);
        if isfield(summary, 'covariance_traces') && ~isempty(summary.covariance_traces)
            trace_mean(i) = mean(summary.covariance_traces, 'omitnan');
            trace_median(i) = median(summary.covariance_traces, 'omitnan');
        end
        if isfield(summary, 'covariance_logdets') && ~isempty(summary.covariance_logdets)
            logdet_mean(i) = mean(summary.covariance_logdets, 'omitnan');
            logdet_median(i) = median(summary.covariance_logdets, 'omitnan');
            if isfield(summary, 'feature_dim') && isfinite(summary.feature_dim) && summary.feature_dim > 0
                logdet_per_dim(i) = logdet_mean(i) / summary.feature_dim;
            end
        end
    end
end

function values = coerce_numeric_vector_local(values)
    if isempty(values)
        values = [];
    elseif isnumeric(values) || islogical(values)
        values = double(values(:));
    else
        values = str2double(string(values(:)));
    end
end

function values = resize_vector_local(values, n_target, fill_value)
    if nargin < 3
        fill_value = NaN;
    end
    if isempty(values)
        values = repmat(fill_value, n_target, 1);
        return;
    end
    values = values(:);
    if numel(values) < n_target
        values(end + 1:n_target, 1) = fill_value; %#ok<AGROW>
    elseif numel(values) > n_target
        values = values(1:n_target);
    end
end

function value = field_or_local(S, field_name, default_value)
    if isstruct(S) && isfield(S, field_name)
        value = S.(field_name);
    else
        value = default_value;
    end
end

function x = normalize_01_local(x)
    x = double(x(:));
    finite_mask = isfinite(x);
    if ~any(finite_mask)
        return;
    end
    xmin = min(x(finite_mask));
    xmax = max(x(finite_mask));
    if xmax - xmin <= eps
        x(finite_mask) = 0;
    else
        x(finite_mask) = (x(finite_mask) - xmin) / (xmax - xmin);
    end
    x(~finite_mask) = 0;
end

function d = distance_from_line_local(p1, p2, p)
    d = abs((p2(2) - p1(2)) * p(1) - (p2(1) - p1(1)) * p(2) + ...
        p2(1) * p1(2) - p2(2) * p1(1)) / ...
        sqrt((p2(2) - p1(2))^2 + (p2(1) - p1(1))^2 + eps);
end

function score = score_at_k_local(K_selected, K_all, score_by_k)
    score = NaN;
    idx = find(double(K_all(:)) == double(K_selected), 1, 'first');
    if ~isempty(idx) && numel(score_by_k) >= idx
        score = score_by_k(idx);
    end
end
