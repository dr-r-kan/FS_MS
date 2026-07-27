function Results = fit_microstate_spm_vb(Sim, K_candidates, criterion)
% FIT_MICROSTATE_SPM_VB: VB microstate fitting using SPM
%
% INPUTS:
%   Sim          - Simulation structure
%   K_candidates - Vector of K values to test
%   criterion    - 'silhouette', 'free_energy', 'log_likelihood', 'bic',
%                  'icl', 'free_energy_elbow',
%                  'elbow_sil_combined', 'gev'/'gfp',
%                  'calinski_harabasz_score', 'covariance',
%                  'covariance_elbow', or 'free_energy_covariance'
%
% OUTPUTS:
%   Results      - Complete results with recovery metrics

    if nargin < 2
        K_candidates = 2:10;
    end
    if nargin < 3
        criterion = 'silhouette';
    end
    
    t_start = tic;

    fprintf('\n========================================\n');
    fprintf('Microstate VB Fitting (SPM)\n');
    fprintf('========================================\n');
    fprintf('Criterion: %s\n', criterion);
    if isfield(Sim, 'K_true') && ~isnan(Sim.K_true)
        fprintf('True K: %d, SNR: %+.1f dB\n', Sim.K_true, Sim.SNR_dB);
    else
        fprintf('Real data mode (no ground truth)\n');
    end

    if ~exist('spm_mix', 'file')
        error('SPM spm_mix not found in MATLAB path');
    end

    % Load shared utilities
    util = microstate_utilities();

    % Preprocessing (using shared utility)
    fprintf('1. Preprocessing...\n');
    [maps_norm, idx_peaks, gfp_vec, n_maps, C_dims, maps_original, preprocessing_info] = util.preprocess_maps(Sim);
    fprintf('   Extracted %d GFP peak maps (%d channels)\n', n_maps, C_dims);

    % SPM's mixture code is fragile in the full sensor space for real EEG:
    % microstate maps are average-reference normalized, so the channel-space
    % covariance is rank-deficient.  Fit the mixture in a finite PCA subspace,
    % then recover topographies from labels in the original normalized maps.
    [linear_features, pca_info] = pca_features_for_spm(maps_norm);
    [features, polarity_feature_info] = polarity_invariant_projective_features(linear_features);
    [matrix_features, matrix_feature_info] = polarity_invariant_matrix_features(maps_norm);
    fprintf(['2. Using polarity-invariant PCA features: %d PCA dims -> %d projective dims ', ...
        '(%.3f%% variance, rank %d of %d channels)\n'], ...
        size(linear_features, 2), size(features, 2), ...
        pca_info.variance_explained_pct, pca_info.rank_est, C_dims);
    fprintf('   Matrix-space evidence features: %d dims (intrinsic dim %d)\n', ...
        size(matrix_features, 2), matrix_feature_info.intrinsic_dim);

    % Fit VB GMM
    fprintf('3. Fitting VB GMM models using SPM...\n');
    nK = numel(K_candidates);
    free_energy = zeros(nK, 1);
    spm_pca_free_energy = zeros(nK, 1);
    log_likelihood_vals = nan(nK, 1);
    bic_vals = nan(nK, 1);
    icl_vals = nan(nK, 1);
    assignment_entropy_vals = nan(nK, 1);
    n_parameters_vals = nan(nK, 1);
    silhouette_score = zeros(nK, 1);
    within_ss = zeros(nK, 1);
    gev_vals = zeros(nK, 1);
    calinski_harabasz_vals = nan(nK, 1);
    vbmix = cell(nK, 1);
    spm_mix_model_summaries = repmat(empty_spm_mix_summary(size(features, 2)), nK, 1);
    centers_all = cell(nK, 1);
    labels_all = cell(nK, 1);
    K_effective = K_candidates(:);
    polarity_duplicate_info = cell(nK, 1);
    feature_backfit_models = cell(nK, 1);
    matrix_evidence_models = cell(nK, 1);

    for iK = 1:nK
        K = K_candidates(iK);
        fprintf('   K=%d... ', K);
        
        try
            % Use isotropic covariance to avoid singularity; evalc prevents
            % SPM internals from dumping large NaN matrices to the console.
            evalc('result = spm_mix(features, K, 0);');
            
            if ~is_valid_spm_mix_result(result)
                fprintf('Invalid\n');
                free_energy(iK) = -Inf;
                spm_pca_free_energy(iK) = -Inf;
                silhouette_score(iK) = -1;
                within_ss(iK) = Inf;
                gev_vals(iK) = 0;
                calinski_harabasz_vals(iK) = NaN;
            else
                spm_pca_free_energy(iK) = result.fm;
                vbmix{iK} = result;
                spm_mix_model_summaries(iK) = summarise_spm_mix_result(result, K, size(features, 2), result.fm);
                
                labels = assign_samples(features, result);
                
                % ✅ Recover centers to compute GEV
                temp_centers = recover_centers_from_labels(maps_norm, labels, K);
                [labels, temp_centers] = polarity_refine_to_target_k(maps_norm, labels, temp_centers, K, 25);
                K_eff = K;
                labels_all{iK} = labels;
                centers_all{iK} = temp_centers;
                K_effective(iK) = K_eff;
                polarity_duplicate_info{iK} = detect_polarity_duplicate_centers(temp_centers, 0.90);
                feature_backfit_models{iK} = build_feature_backfit_model(features, labels, K_eff);
                evidence = compute_matrix_space_evidence(matrix_features, labels, K_eff, matrix_feature_info.intrinsic_dim);
                free_energy(iK) = evidence.free_energy;
                log_likelihood_vals(iK) = evidence.log_likelihood;
                bic_vals(iK) = evidence.bic;
                icl_vals(iK) = evidence.icl;
                assignment_entropy_vals(iK) = evidence.assignment_entropy;
                n_parameters_vals(iK) = evidence.n_parameters;
                matrix_evidence_models{iK} = evidence.model;
                sim = abs(maps_original * temp_centers');
                [max_sim, ~] = max(sim, [], 2);
                gfp_squared = sum(maps_original.^2, 2);
                gev_vals(iK) = sum(max_sim.^2) / (sum(gfp_squared) + eps);

                % Compute silhouette using cosine distance in map space
                % (matching Koenig's polarity-invariant topography metric).
                sil = silhouette_cosine(maps_norm, labels, temp_centers);
                silhouette_score(iK) = sil;
                
                % Compute within-cluster sum of squares for elbow method
                wss = compute_within_ss(maps_norm, labels, K_eff);
                within_ss(iK) = wss;
                calinski_harabasz_vals(iK) = compute_calinski_harabasz_score(features, labels, K_eff);
                
                dup_count = size(polarity_duplicate_info{iK}.pairs, 1);
                if dup_count > 0
                    fprintf('F=%.1f, Sil=%.3f, WSS=%.1f, polarity-duplicate pairs=%d\n', ...
                        free_energy(iK), sil, wss, dup_count);
                else
                    fprintf('F=%.1f, Sil=%.3f, WSS=%.1f\n', free_energy(iK), sil, wss);
                end
            end
        catch ME
            fprintf('ERROR: %s\n', ME.message);
            free_energy(iK) = -Inf;
            spm_pca_free_energy(iK) = -Inf;
            silhouette_score(iK) = -1;
            within_ss(iK) = Inf;
            gev_vals(iK) = 0;
            calinski_harabasz_vals(iK) = NaN;
            within_ss(iK) = Inf;
            spm_mix_model_summaries(iK) = empty_spm_mix_summary(size(features, 2), K);
        end
    end

    % Model selection
    valid_idx = isfinite(free_energy) & free_energy ~= 0;
    
    if ~any(valid_idx)
        warning('All fits failed!');
        Results = create_empty_results(Sim, K_candidates, n_maps, C_dims, criterion);
        return;
    end
    
    fe_valid = free_energy(valid_idx);
    K_valid = K_candidates(valid_idx);
    sil_valid = silhouette_score(valid_idx);
    wss_valid = within_ss(valid_idx);
    [cov_trace_mean_vals, cov_trace_median_vals, cov_logdet_mean_vals, ...
        cov_logdet_median_vals, cov_logdet_per_dim_mean_vals] = ...
        summarise_covariance_arrays(spm_mix_model_summaries);
    
    fprintf('\nModel selection using: %s\n', criterion);

    selection_payload = struct( ...
        'K_candidates', K_candidates(:), ...
        'free_energy_vals', free_energy(:), ...
        'spm_pca_free_energy_vals', spm_pca_free_energy(:), ...
        'log_likelihood_vals', log_likelihood_vals(:), ...
        'bic_vals', bic_vals(:), ...
        'icl_vals', icl_vals(:), ...
        'silhouette_vals', silhouette_score(:), ...
        'calinski_harabasz_vals', calinski_harabasz_vals(:), ...
        'covariance_trace_mean_vals', cov_trace_mean_vals(:), ...
        'covariance_trace_median_vals', cov_trace_median_vals(:), ...
        'covariance_logdet_mean_vals', cov_logdet_mean_vals(:), ...
        'covariance_logdet_median_vals', cov_logdet_median_vals(:), ...
        'covariance_logdet_per_dim_mean_vals', cov_logdet_per_dim_mean_vals(:), ...
        'spm_mix_model_summaries', spm_mix_model_summaries);
    [K_model_selected, best_score, selection_score_by_k, selection_details] = ...
        select_spm_vb_k_by_criterion(selection_payload, criterion);
    if ~isfinite(K_model_selected)
        [best_score, idx] = max(fe_valid);
        K_model_selected = K_valid(idx);
    end
    best_idx = find(double(K_candidates(:)) == double(K_model_selected), 1, 'first');
    if isempty(best_idx)
        [~, idx] = max(fe_valid);
        best_idx = find(valid_idx);
        best_idx = best_idx(idx);
        K_model_selected = K_candidates(best_idx);
    end

    switch lower(strtrim(char(string(criterion))))
        case 'silhouette'
            fprintf('  Silhouette (Koenig): selected K=%d (score=%.4f)\n', K_model_selected, best_score);
        case {'gev', 'gfp', 'global_explained_variance'}
            fprintf('  GEV/GFP: selected K=%d (score=%.4f)\n', K_model_selected, best_score);
        case {'calinski_harabasz', 'calinski_harabasz_score', 'ch'}
            fprintf('  Calinski-Harabasz: selected K=%d (score=%.4f)\n', K_model_selected, best_score);
        case 'free_energy'
            fprintf('  Free Energy: selected K=%d (FE=%.1f)\n', K_model_selected, best_score);
        case {'log_likelihood', 'll'}
            fprintf('  Matrix log likelihood: selected K=%d (LL=%.1f)\n', K_model_selected, best_score);
        case 'bic'
            fprintf('  Matrix BIC evidence: selected K=%d (BIC=%.1f)\n', K_model_selected, best_score);
        case 'icl'
            fprintf('  Matrix ICL evidence: selected K=%d (ICL=%.1f)\n', K_model_selected, best_score);
        case {'free_energy_elbow', 'elbow'}
            fprintf('  Free-energy elbow: selected K=%d (score=%.4f)\n', K_model_selected, best_score);
        case 'elbow_sil_combined'
            fprintf('  Combined (Elbow+Silhouette): selected K=%d (score=%.4f)\n', K_model_selected, best_score);
        case {'covariance', 'covariance_raw', 'covariance_min'}
            fprintf('  Covariance minimum: selected K=%d (score=%.4f)\n', K_model_selected, best_score);
        case 'covariance_elbow'
            fprintf('  Covariance elbow: selected K=%d (score=%.4f)\n', K_model_selected, best_score);
        case {'free_energy_covariance', 'covariance_free_energy', 'free_energy_covariance_hybrid'}
            fprintf('  Free-energy+covariance: selected K=%d (score=%.4f)\n', K_model_selected, best_score);
        otherwise
            fprintf('  %s: selected K=%d (score=%.4f)\n', criterion, K_model_selected, best_score);
    end
            
    K_estimated = K_effective(best_idx);
    fprintf('Best model K: %d; effective polarity-invariant K: %d\n', ...
        K_model_selected, K_estimated);
    if isfield(Sim, 'K_true') && ~isnan(Sim.K_true)
        fprintf('True K: %d\n\n', Sim.K_true);
    else
        fprintf('\n');
    end

    % Recover centers
    fprintf('4. Recovering microstate topographies...\n');
    labels = labels_all{best_idx};
    centers = centers_all{best_idx};
    fprintf('   Recovered %d microstate centers\n', K_estimated);

    cluster_weights = zeros(1, K_estimated);
    for k = 1:K_estimated
        cluster_weights(k) = mean(labels == k);
    end
    cluster_weights = cluster_weights / (sum(cluster_weights) + eps);

    % Map recovery (only if ground truth is available)
    fprintf('5. Computing map recovery...\n');
    if isfield(Sim, 'maps_true') && ~isempty(Sim.maps_true)
        true_maps_norm = util.normalize_maps(Sim.maps_true);
        recovery_metrics = microstate_partial_alignment(true_maps_norm, centers, ...
            'distance_type', 'cosine', 'threshold', 0.0, 'polarity', true);
        
        fprintf('Map Recovery Analysis:\n');
        fprintf('  K true: %d, K estimated: %d\n', recovery_metrics.K_true, recovery_metrics.K_estimated);
        fprintf('  Matched: %d, F1: %.4f\n', recovery_metrics.n_matched, recovery_metrics.f1_score);
        fprintf('  Sensitivity: %.4f, Precision: %.4f\n\n', ...
            recovery_metrics.sensitivity, recovery_metrics.precision);
    else
        % No ground truth - create empty recovery metrics
        recovery_metrics = struct(...
            'K_true', NaN, ...
            'K_estimated', K_estimated, ...
            'n_matched', 0, ...
            'mean_recovery_matched', NaN, ...
            'mean_recovery_padded', NaN, ...
            'sensitivity', NaN, ...
            'precision', NaN, ...
            'f1_score', NaN, ...
            'match_similarities', []);
        true_maps_norm = [];
        fprintf('No ground truth available - skipping recovery metrics\n\n');
    end
    
    runtime = toc(t_start);

    % Return results
    if isfield(Sim, 'K_true')
        K_true_val = Sim.K_true;
    else
        K_true_val = NaN;
    end
    if isfield(Sim, 'SNR_dB')
        SNR_dB_val = Sim.SNR_dB;
    else
        SNR_dB_val = NaN;
    end
    if isfield(Sim, 'duration_s')
        duration_s_val = Sim.duration_s;
    else
        duration_s_val = NaN;
    end
    
    Results = struct( ...
    'method', 'spm_vb', ...
    'criterion', criterion, ...
    'K_true', K_true_val, ...
    'K_estimated', K_estimated, ...
    'K_model_selected', K_model_selected, ...
    'K_candidates', K_candidates, ...
    'K_effective_vals', K_effective, ...
    'SNR_dB', SNR_dB_val, ...
    'duration_s', duration_s_val, ...
    'n_maps', n_maps, ...
    'centers', centers, ...
    'centers_by_K', {centers_all}, ...
    'cluster_weights', cluster_weights, ...  % ✅ Added confidence measure
    'maps_true', true_maps_norm, ...
    'labels', labels, ...
    'labels_by_K', {labels_all}, ...
    'free_energy', free_energy, ...
    'matrix_free_energy_vals', free_energy, ...
    'spm_pca_free_energy_vals', spm_pca_free_energy, ...
    'log_likelihood_vals', log_likelihood_vals, ...
    'bic_vals', bic_vals, ...
    'icl_vals', icl_vals, ...
    'assignment_entropy_vals', assignment_entropy_vals, ...
    'n_parameters_vals', n_parameters_vals, ...
    'silhouette_vals', silhouette_score, ...  
    'free_energy_vals', free_energy, ...      
    'within_ss', within_ss, ...            
    'gev_vals', gev_vals, ...
    'calinski_harabasz_vals', calinski_harabasz_vals, ...
    'covariance_trace_mean_vals', cov_trace_mean_vals, ...
    'covariance_trace_median_vals', cov_trace_median_vals, ...
    'covariance_logdet_mean_vals', cov_logdet_mean_vals, ...
    'covariance_logdet_median_vals', cov_logdet_median_vals, ...
    'covariance_logdet_per_dim_mean_vals', cov_logdet_per_dim_mean_vals, ...
    'selection_score_by_k', selection_score_by_k, ...
    'selection_details', selection_details, ...
    'spm_mix_model_summaries', spm_mix_model_summaries, ...
    'selected_spm_mix_model', spm_mix_model_summaries(best_idx), ...
    'selected_spm_covariances', spm_mix_model_summaries(best_idx).covariances, ...
    'selected_spm_means', spm_mix_model_summaries(best_idx).means, ...
    'selected_spm_priors', spm_mix_model_summaries(best_idx).priors, ...
    'feature_backfit_models', {feature_backfit_models}, ...
    'selected_feature_backfit_model', feature_backfit_models{best_idx}, ...
    'matrix_evidence_models', {matrix_evidence_models}, ...
    'selected_matrix_evidence_model', matrix_evidence_models{best_idx}, ...
    'best_criterion_value', best_score, ...
    'maps_nc', maps_norm, ...
    'idx_peaks', idx_peaks, ...
    'gfp_vec', gfp_vec, ...
    'preprocessing_info', preprocessing_info, ...
    'pca_info', pca_info, ...
    'polarity_feature_info', polarity_feature_info, ...
    'matrix_feature_info', matrix_feature_info, ...
    'polarity_duplicate_info', {polarity_duplicate_info}, ...
    'recovery_metrics', recovery_metrics, ...
    'mean_recovery', recovery_metrics.mean_recovery_matched, ...
    'recovery_corr', recovery_metrics.match_similarities, ...
    'avg_recovery_per_state', recovery_metrics.mean_recovery_padded, ...
    'valid_fit', true, ...
    'runtime', runtime);
end

function summary = summarise_spm_mix_result(result, K_candidate, feature_dim, free_energy_val)
    if nargin < 4
        free_energy_val = NaN;
    end
    summary = empty_spm_mix_summary(feature_dim, K_candidate);
    summary.free_energy = double(real(free_energy_val));
    if ~isstruct(result) || ~isfield(result, 'state') || isempty(result.state)
        return;
    end

    n_components = numel(result.state);
    summary.n_components = n_components;
    summary.means = nan(n_components, feature_dim);
    summary.priors = nan(n_components, 1);
    summary.covariances = nan(feature_dim, feature_dim, n_components);
    summary.covariance_traces = nan(n_components, 1);
    summary.covariance_logdets = nan(n_components, 1);

    for k = 1:n_components
        if isfield(result.state(k), 'm') && ~isempty(result.state(k).m)
            m = double(real(result.state(k).m(:)));
            summary.means(k, 1:min(feature_dim, numel(m))) = m(1:min(feature_dim, numel(m)));
        end
        if isfield(result.state(k), 'C') && ~isempty(result.state(k).C)
            C = coerce_covariance_matrix(result.state(k).C, feature_dim);
            summary.covariances(:, :, k) = C;
            summary.covariance_traces(k) = trace(C);
            summary.covariance_logdets(k) = safe_logdet(C);
        end
        summary.priors(k) = extract_state_prior(result, k);
    end
end

function summary = empty_spm_mix_summary(feature_dim, K_candidate)
    if nargin < 2
        K_candidate = NaN;
    end
    summary = struct( ...
        'K_candidate', double(real(K_candidate)), ...
        'feature_dim', double(real(feature_dim)), ...
        'n_components', 0, ...
        'free_energy', NaN, ...
        'means', nan(0, feature_dim), ...
        'priors', nan(0, 1), ...
        'covariances', nan(feature_dim, feature_dim, 0), ...
        'covariance_traces', nan(0, 1), ...
        'covariance_logdets', nan(0, 1));
end

function C = coerce_covariance_matrix(C_in, feature_dim)
    C = double(real(C_in));
    if isscalar(C)
        C = C * eye(feature_dim);
    elseif isvector(C)
        C = diag(C(1:min(feature_dim, numel(C))));
    end
    if size(C, 1) ~= feature_dim || size(C, 2) ~= feature_dim
        n = min([size(C, 1), size(C, 2), feature_dim]);
        C_fixed = nan(feature_dim, feature_dim);
        C_fixed(1:n, 1:n) = C(1:n, 1:n);
        C = C_fixed;
    end
end

function prior = extract_state_prior(result, k)
    prior = NaN;
    if isfield(result.state(k), 'prior') && ~isempty(result.state(k).prior)
        prior = double(real(result.state(k).prior));
    elseif isfield(result, 'priors') && numel(result.priors) >= k && ~isempty(result.priors(k))
        prior = double(real(result.priors(k)));
    end
end

function val = safe_logdet(C)
    val = NaN;
    if isempty(C) || any(~isfinite(C(:)))
        return;
    end
    C = (C + C') / 2;
    try
        rc = rcond(C);
        if ~isfinite(rc) || rc <= eps
            return;
        end
        u = chol(C);
        val = 2 * sum(log(diag(u)));
    catch
        try
            e = eig(C);
            if any(~isfinite(e) | e <= eps)
                return;
            end
            val = sum(log(e));
        catch
            val = NaN;
        end
    end
end

function [trace_mean, trace_median, logdet_mean, logdet_median, logdet_per_dim] = summarise_covariance_arrays(summaries)
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
        end
        if isfield(summary, 'feature_dim') && isfinite(summary.feature_dim) && summary.feature_dim > 0 && isfinite(logdet_mean(i))
            logdet_per_dim(i) = logdet_mean(i) / summary.feature_dim;
        end
    end
end

% ======================== K SELECTION HELPERS ========================

function [K_est, score] = select_K_from_elbow_spm(wss_vals, K_candidates)
    % Elbow detection for within-cluster sum of squares
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
    
    fprintf('  Elbow: K=%d (WSS=%.1f, curvature=%.4f)\n', K_est, wss_vals(idx), score);
end

function [K_est, score] = select_K_from_free_energy_curve(fe_vals, K_candidates)
    n = length(fe_vals);
    if n < 3
        [score, idx] = max(fe_vals);
        K_est = K_candidates(idx);
        return;
    end

    fe_norm = (fe_vals - min(fe_vals)) / (max(fe_vals) - min(fe_vals) + eps);
    k_norm = (K_candidates - min(K_candidates)) / (max(K_candidates) - min(K_candidates) + eps);
    p1 = [k_norm(1), fe_norm(1)];
    p2 = [k_norm(end), fe_norm(end)];
    elbow_scores = zeros(size(fe_norm));
    for i = 2:(n - 1)
        p = [k_norm(i), fe_norm(i)];
        elbow_scores(i) = abs((p2(2)-p1(2))*p(1) - (p2(1)-p1(1))*p(2) + ...
            p2(1)*p1(2) - p2(2)*p1(1)) / ...
            sqrt((p2(2)-p1(2))^2 + (p2(1)-p1(1))^2 + eps);
    end
    [score, idx] = max(elbow_scores);
    K_est = K_candidates(idx);
    fprintf('  Free-energy elbow: K=%d (FE=%.1f, curvature=%.4f)\n', K_est, fe_vals(idx), score);
end

function [K_est, score] = select_K_combined_elbow_silhouette(fe_valid, K_valid, sil_valid)
    % Original combined elbow + silhouette heuristic (backwards compatibility)
    
    fe_norm = (fe_valid - min(fe_valid)) / (max(fe_valid) - min(fe_valid) + eps);
    k_norm = (K_valid - min(K_valid)) / (max(K_valid) - min(K_valid) + eps);
    
    elbow_scores = zeros(size(fe_norm));
    for i = 2:length(fe_norm)-1
        p1 = [k_norm(1), fe_norm(1)];
        p2 = [k_norm(end), fe_norm(end)];
        p = [k_norm(i), fe_norm(i)];
        
        elbow_scores(i) = abs((p2(2)-p1(2))*p(1) - (p2(1)-p1(1))*p(2) + ...
            p2(1)*p1(2) - p2(2)*p1(1)) / ...
            sqrt((p2(2)-p1(2))^2 + (p2(1)-p1(1))^2 + eps);
    end
    
    [~, elbow_idx] = max(elbow_scores);
    K_elbow = K_valid(elbow_idx);
    
    % Combined score
    combined_score = zeros(length(K_valid), 1);
    for i = 1:length(K_valid)
        elbow_penalty = exp(-abs(K_valid(i) - K_elbow));
        sil_bonus = (sil_valid(i) + 1) / 2;
        combined_score(i) = 0.6 * elbow_penalty + 0.4 * sil_bonus;
    end
    
    [score, best_idx] = max(combined_score);
    K_est = K_valid(best_idx);
    
    fprintf('  Combined (Elbow+Silhouette): selected K=%d (score=%.4f, elbow K=%d)\n', K_est, score, K_elbow);
end

% ======================== HELPERS (LOCAL ONLY) ========================

function labels = assign_samples(X, vbmix)
% Assign samples to mixture components
    
    [N, D] = size(X);
    K = vbmix.m;
    log_prob = zeros(N, K);
    
    for k = 1:K
        m = vbmix.state(k).m(:)';
        C = vbmix.state(k).C;
        
        if length(m) ~= D
            m = m(1:min(D, length(m)));
            if size(C, 1) ~= D
                C = C(1:min(D, size(C, 1)), 1:min(D, size(C, 2)));
            end
        end
        
        if isfield(vbmix.state(k), 'prior')
            prior = vbmix.state(k).prior;
        elseif isfield(vbmix, 'priors') && length(vbmix.priors) >= k
            prior = vbmix.priors(k);
        else
            prior = 1 / K;
        end
        
        log_prob(:, k) = log(prior + eps) + log_mvnpdf(X, m, C);
    end
    
    [~, labels] = max(log_prob, [], 2);
end

function [features, info] = pca_features_for_spm(maps_norm)
%PCA_FEATURES_FOR_SPM Return a numerically stable feature space for spm_mix.

    [N, D] = size(maps_norm);
    [coeff, score, latent, ~, ~, mu] = pca(maps_norm, 'Centered', true);
    latent = real(latent(:));
    total_var = sum(latent);
    if total_var <= eps
        features = maps_norm;
        info = struct('rank_est', size(features, 2), ...
            'n_dims', size(features, 2), ...
            'variance_explained_pct', 100);
        return;
    end

    tol = max(size(maps_norm)) * eps(max(latent));
    rank_est = sum(latent > tol);
    var_explained = cumsum(latent) / total_var;
    n_dims = find(var_explained >= 0.999, 1, 'first');
    if isempty(n_dims)
        n_dims = rank_est;
    end
    % The polarity-invariant outer-product embedding grows as D*(D+1)/2.
    % Keep the linear PCA subspace deliberately compact so SPM's VB mixture
    % remains well-conditioned in real EEG, where rank can otherwise expand
    % to a numerically awkward projective space.
    n_dims = min([n_dims, rank_est, N - 1, D - 1, 8]);
    n_dims = max(1, n_dims);

    features = score(:, 1:n_dims);
    scale = std(features, 0, 1);
    scale(scale < eps) = 1;
    features = features ./ scale;

    info = struct('rank_est', rank_est, ...
        'n_dims', n_dims, ...
        'variance_explained_pct', 100 * var_explained(n_dims), ...
        'coeff', coeff(:, 1:n_dims), ...
        'latent', latent(1:n_dims), ...
        'mean', mu(:)', ...
        'scale', scale(1:n_dims));
end

function [features_projective, info] = polarity_invariant_projective_features(features_linear)
%POLARITY_INVARIANT_PROJECTIVE_FEATURES Make x and -x identical to SPM.
%
% For unit vectors, ||xx' - yy'|| is a monotonic function of
% 1 - (x'y)^2, which is the standard polarity-invariant microstate
% distance.  The sqrt(2) scaling on off-diagonal terms preserves the
% Frobenius inner product after vectorising the upper triangle.

    Y = normalize_rows(features_linear);
    D = size(Y, 2);
    n_features = D * (D + 1) / 2;
    features_projective = zeros(size(Y, 1), n_features);
    col = 0;
    for a = 1:D
        for b = a:D
            col = col + 1;
            if a == b
                scale = 1;
            else
                scale = sqrt(2);
            end
            features_projective(:, col) = scale * Y(:, a) .* Y(:, b);
        end
    end

    projective_mean = mean(features_projective, 1);
    features_projective = features_projective - projective_mean;
    sd = std(features_projective, 0, 1);
    keep = sd > (10 * eps);
    features_projective = features_projective(:, keep);
    sd = sd(keep);
    features_projective = features_projective ./ sd;

    info = struct('mode', 'projective_outer_product', ...
        'linear_dims', D, ...
        'projective_dims', size(features_projective, 2), ...
        'dropped_constant_dims', n_features - size(features_projective, 2), ...
        'keep_mask', keep, ...
        'mean', projective_mean(keep), ...
        'scale', sd);
end

function [features_matrix, info] = polarity_invariant_matrix_features(maps_norm)
%POLARITY_INVARIANT_MATRIX_FEATURES Full sensor-space xx' embedding.

    [features_matrix, info] = polarity_invariant_projective_features(maps_norm);
    rank_est = estimate_matrix_rank(maps_norm);
    info.mode = 'sensor_outer_product';
    info.channel_dims = size(maps_norm, 2);
    % ponytail: matrix features live in a high-dimensional embedding; BIC uses
    % the intrinsic scalp-map dimension so K is not penalised for redundant xx' entries.
    info.intrinsic_dim = max(1, rank_est - 1);
end

function rank_est = estimate_matrix_rank(X)
    s = svd(double(X), 'econ');
    if isempty(s)
        rank_est = 1;
        return;
    end
    tol = max(size(X)) * eps(max(s));
    rank_est = max(1, sum(s > tol));
end

function model = build_feature_backfit_model(features, labels, K)
%BUILD_FEATURE_BACKFIT_MODEL Regularized Gaussian model aligned to final labels.

    if nargin < 3
        K = max(labels);
    end
    [N, D] = size(features);
    means = zeros(K, D);
    priors = zeros(K, 1);
    covariances = zeros(D, D, K);
    global_var = var(features, 0, 1);
    global_var(~isfinite(global_var) | global_var < 1e-6) = 1e-6;
    global_cov = diag(global_var);
    ridge = max(1e-6, mean(global_var) * 1e-3);

    for k = 1:K
        idx = labels == k;
        priors(k) = mean(idx);
        if ~any(idx)
            means(k, :) = mean(features, 1);
            covariances(:, :, k) = global_cov + ridge * eye(D);
            continue;
        end
        Xk = features(idx, :);
        means(k, :) = mean(Xk, 1);
        if size(Xk, 1) >= 2
            Ck = cov(Xk, 1);
        else
            Ck = global_cov;
        end
        if ~all(isfinite(Ck(:))) || isempty(Ck)
            Ck = global_cov;
        end
        Ck = (Ck + Ck') / 2;
        covariances(:, :, k) = Ck + ridge * eye(D);
    end

    priors = priors / max(sum(priors), eps);
    model = struct( ...
        'means', means, ...
        'covariances', covariances, ...
        'priors', priors, ...
        'feature_dim', D, ...
        'n_components', K, ...
        'ridge', ridge);
end

function evidence = compute_matrix_space_evidence(features, labels, K, intrinsic_dim)
%COMPUTE_MATRIX_SPACE_EVIDENCE LL/BIC/ICL for the final solution.

    if nargin < 4 || isempty(intrinsic_dim) || ~isfinite(intrinsic_dim)
        intrinsic_dim = size(features, 2);
    end
    model = build_diagonal_feature_model(features, labels, K);
    [log_joint, log_likelihood_by_row] = matrix_model_log_joint(features, model);
    log_likelihood = sum(log_likelihood_by_row);
    resp = exp(log_joint - log_likelihood_by_row);
    resp(~isfinite(resp)) = 0;
    assignment_entropy = -sum(resp(:) .* log(resp(:) + eps));
    n_parameters = (K - 1) + 2 * K * max(1, intrinsic_dim);
    % Matrix-space free energy is the BIC-form log-evidence approximation.
    bic = log_likelihood - 0.5 * n_parameters * log(size(features, 1));
    icl = bic - assignment_entropy;
    evidence = struct( ...
        'free_energy', bic, ...
        'log_likelihood', log_likelihood, ...
        'bic', bic, ...
        'icl', icl, ...
        'assignment_entropy', assignment_entropy, ...
        'n_parameters', n_parameters, ...
        'model', model);
end

function model = build_diagonal_feature_model(features, labels, K)
    [~, D] = size(features);
    labels = labels(:);
    means = zeros(K, D);
    variances = zeros(K, D);
    priors = zeros(K, 1);
    global_var = var(features, 0, 1);
    global_var(~isfinite(global_var) | global_var < 1e-8) = 1e-8;
    ridge = max(1e-8, median(global_var) * 1e-4);

    for k = 1:K
        idx = labels == k;
        priors(k) = mean(idx);
        if ~any(idx)
            means(k, :) = mean(features, 1);
            variances(k, :) = global_var + ridge;
            continue;
        end
        Xk = features(idx, :);
        means(k, :) = mean(Xk, 1);
        if size(Xk, 1) >= 2
            vk = var(Xk, 0, 1);
        else
            vk = global_var;
        end
        vk(~isfinite(vk) | vk < 1e-8) = 1e-8;
        variances(k, :) = vk + ridge;
    end

    priors = priors / max(sum(priors), eps);
    model = struct('means', means, 'variances', variances, 'priors', priors, ...
        'feature_dim', D, 'n_components', K, 'ridge', ridge);
end

function [log_joint, log_likelihood_by_row] = matrix_model_log_joint(features, model)
    [N, D] = size(features);
    K = model.n_components;
    log_joint = -inf(N, K);
    for k = 1:K
        v = model.variances(k, :);
        xc = features - model.means(k, :);
        log_pdf = -0.5 * (D * log(2 * pi) + sum(log(v)) + sum((xc .^ 2) ./ v, 2));
        log_joint(:, k) = log(model.priors(k) + eps) + log_pdf;
    end
    log_likelihood_by_row = logsumexp_rows(log_joint);
end

function y = logsumexp_rows(X)
    m = max(X, [], 2);
    y = m + log(sum(exp(X - m), 2) + eps);
end

function info = detect_polarity_duplicate_centers(centers, threshold)
    if nargin < 2
        threshold = 0.85;
    end
    centers = normalize_rows(centers);
    signed_corr = centers * centers';
    pairs = [];
    for i = 1:(size(centers, 1) - 1)
        for j = (i + 1):size(centers, 1)
            if abs(signed_corr(i, j)) >= threshold
                pairs(end+1, :) = [i, j, signed_corr(i, j), abs(signed_corr(i, j))]; %#ok<AGROW>
            end
        end
    end
    info = struct('threshold', threshold, 'pairs', pairs, 'signed_corr', signed_corr);
end

function [labels, centers] = polarity_refine_to_target_k(maps_norm, labels, centers, K, max_iter)
%POLARITY_REFINE_TO_TARGET_K Koenig-style polarity-invariant refinement.

    if nargin < 5
        max_iter = 25;
    end
    duplicate_threshold = 0.90;
    centers = normalize_rows(centers);
    centers = ensure_target_k_centers(maps_norm, centers, K);
    labels = assign_by_abs_correlation(maps_norm, centers);
    [labels, centers] = repair_empty_clusters(maps_norm, labels, centers, K);
    [labels, centers] = repair_polarity_duplicate_centers(maps_norm, labels, centers, K, duplicate_threshold);

    for iter = 1:max_iter
        old_labels = labels;
        old_centers = centers;
        centers = recover_centers_from_labels(maps_norm, labels, K);
        labels = assign_by_abs_correlation(maps_norm, centers);
        [labels, centers] = repair_empty_clusters(maps_norm, labels, centers, K);
        [labels, centers] = repair_polarity_duplicate_centers(maps_norm, labels, centers, K, duplicate_threshold);
        if isequal(labels, old_labels) && max(abs(centers(:) - old_centers(:))) < 1e-8
            break;
        end
    end
end

function centers = ensure_target_k_centers(maps_norm, centers, K)
    if isempty(centers)
        centers = maps_norm(randperm(size(maps_norm, 1), min(K, size(maps_norm, 1))), :);
    end
    centers = normalize_rows(centers);
    while size(centers, 1) < K
        sims = abs(maps_norm * centers');
        novelty = 1 - max(sims, [], 2);
        [~, idx] = max(novelty);
        centers(end+1, :) = maps_norm(idx, :); %#ok<AGROW>
    end
    if size(centers, 1) > K
        centers = centers(1:K, :);
    end
    centers = normalize_rows(centers);
end

function [labels, centers] = repair_empty_clusters(maps_norm, labels, centers, K)
    counts = accumarray(labels(:), 1, [K, 1], @sum, 0);
    empty = find(counts == 0)';
    for k = empty
        sims = abs(maps_norm * centers');
        assigned_sim = sims(sub2ind(size(sims), (1:size(maps_norm, 1))', labels));
        non_singleton = counts(labels) > 1;
        candidate_score = (1 - assigned_sim) .* non_singleton;
        [~, idx] = max(candidate_score);
        if candidate_score(idx) <= 0
            [~, idx] = min(max(sims, [], 2));
        end
        counts(labels(idx)) = counts(labels(idx)) - 1;
        labels(idx) = k;
        counts(k) = 1;
        centers(k, :) = maps_norm(idx, :);
    end
    centers = recover_centers_from_labels(maps_norm, labels, K);
end

function [labels, centers] = repair_polarity_duplicate_centers(maps_norm, labels, centers, K, threshold)
    for attempt = 1:K
        info = detect_polarity_duplicate_centers(centers, threshold);
        if isempty(info.pairs)
            return;
        end
        pair = info.pairs(1, 1:2);
        counts = accumarray(labels(:), 1, [K, 1], @sum, 0);
        if counts(pair(1)) <= counts(pair(2))
            replace_idx = pair(1);
        else
            replace_idx = pair(2);
        end
        keep_idx = setdiff(1:K, replace_idx);
        sims = abs(maps_norm * centers(keep_idx, :)');
        novelty = 1 - max(sims, [], 2);
        [~, seed_idx] = max(novelty);
        centers(replace_idx, :) = maps_norm(seed_idx, :);
        labels = assign_by_abs_correlation(maps_norm, centers);
        [labels, centers] = repair_empty_clusters(maps_norm, labels, centers, K);
    end
end

function [labels, centers, info] = collapse_polarity_duplicate_states(maps_norm, centers_in, threshold)
% Collapse centers that are the same topography up to polarity.

    if nargin < 3 || isempty(threshold)
        threshold = 0.85;
    end
    centers_in = normalize_rows(centers_in);
    K0 = size(centers_in, 1);
    if K0 <= 1
        centers = centers_in;
        labels = ones(size(maps_norm, 1), 1);
        info = struct('original_K', K0, 'effective_K', K0, 'threshold', threshold, ...
            'merged_pairs', [], 'abs_corr', 1);
        return;
    end

    signed_corr = centers_in * centers_in';
    abs_corr = abs(signed_corr);
    parent = 1:K0;
    merged_pairs = [];
    for i = 1:(K0 - 1)
        for j = (i + 1):K0
            if abs_corr(i, j) >= threshold
                parent = union_roots(parent, i, j);
                merged_pairs(end+1, :) = [i, j, signed_corr(i, j), abs_corr(i, j)]; %#ok<AGROW>
            end
        end
    end

    group_id = zeros(1, K0);
    roots = zeros(1, K0);
    n_groups = 0;
    for i = 1:K0
        r = find_root(parent, i);
        hit = find(roots(1:n_groups) == r, 1);
        if isempty(hit)
            n_groups = n_groups + 1;
            roots(n_groups) = r;
            group_id(i) = n_groups;
        else
            group_id(i) = hit;
        end
    end

    centers = zeros(n_groups, size(centers_in, 2));
    for g = 1:n_groups
        members = find(group_id == g);
        ref = centers_in(members(1), :);
        aligned = centers_in(members, :);
        flips = aligned * ref' < 0;
        aligned(flips, :) = -aligned(flips, :);
        centers(g, :) = mean(aligned, 1);
    end
    centers = normalize_rows(centers);

    labels = assign_by_abs_correlation(maps_norm, centers);
    for iter = 1:5
        [labels, centers] = remove_empty_states(labels, centers);
        centers_new = recover_centers_from_labels(maps_norm, labels, size(centers, 1));
        labels_new = assign_by_abs_correlation(maps_norm, centers_new);
        if isequal(labels_new, labels)
            centers = centers_new;
            break;
        end
        labels = labels_new;
        centers = centers_new;
    end
    [labels, centers] = remove_empty_states(labels, centers);
    centers = normalize_rows(centers);

    info = struct('original_K', K0, ...
        'effective_K', size(centers, 1), ...
        'threshold', threshold, ...
        'merged_pairs', merged_pairs, ...
        'abs_corr', abs_corr);
end

function labels = assign_by_abs_correlation(maps_norm, centers)
    sims = abs(maps_norm * centers');
    [~, labels] = max(sims, [], 2);
end

function [labels_out, centers_out] = remove_empty_states(labels, centers)
    used = unique(labels(:))';
    used = used(used >= 1 & used <= size(centers, 1));
    centers_out = centers(used, :);
    labels_out = zeros(size(labels));
    for k = 1:numel(used)
        labels_out(labels == used(k)) = k;
    end
end

function Xn = normalize_rows(X)
    Xn = X - mean(X, 2);
    norms = sqrt(sum(Xn.^2, 2));
    norms(norms < eps) = 1;
    Xn = Xn ./ norms;
end

function parent = union_roots(parent, a, b)
    ra = find_root(parent, a);
    rb = find_root(parent, b);
    if ra ~= rb
        parent(rb) = ra;
    end
end

function r = find_root(parent, i)
    r = i;
    while parent(r) ~= r
        r = parent(r);
    end
end

function ok = is_valid_spm_mix_result(result)
%IS_VALID_SPM_MIX_RESULT Guard against SPM returning NaN internals.

    ok = isstruct(result) && isfield(result, 'fm') && isfinite(result.fm) && result.fm ~= 0 && ...
        isfield(result, 'state') && ~isempty(result.state);
    if ~ok
        return;
    end
    for k = 1:numel(result.state)
        if ~isfield(result.state(k), 'm') || any(~isfinite(result.state(k).m(:)))
            ok = false;
            return;
        end
        if isfield(result.state(k), 'C') && any(~isfinite(result.state(k).C(:)))
            ok = false;
            return;
        end
    end
end

function log_p = log_mvnpdf(X, mu, Sigma)
% Log multivariate normal PDF
    
    [N, D] = size(X);
    X_centered = X - repmat(mu, N, 1);
    
    [U, S, ~] = svd(Sigma);
    s = diag(S);
    s(s < 1e-10) = 1e-10;
    Sigma_reg = U * diag(s) * U';
    
    try
        L = chol(Sigma_reg, 'lower');
        z = L \ X_centered';
        maha = sum(z.^2, 1)';
    catch
        Sigma_inv = pinv(Sigma_reg);
        maha = sum((X_centered * Sigma_inv) .* X_centered, 2);
    end
    
    log_det = sum(log(s));
    log_p = -0.5 * (D*log(2*pi) + log_det + maha);
end

function sil = silhouette_cosine(X, labels, centers)
    % Silhouette score using polarity-insensitive cosine distance (matching Koenig)
    % Distance = 1 - |cosine_similarity|
    
    K = size(centers, 1);
    N = size(X, 1);
    
    if K < 2 || N < 2
        sil = 0;
        return;
    end
    
    sil_vals = zeros(N, 1);
    
    for i = 1:N
        k = labels(i);
        
        % Intra-cluster: mean distance to other points in same cluster
        same_cluster = find(labels == k);
        if numel(same_cluster) > 1
            % Distance = 1 - |cosine_similarity| (polarity-insensitive)
            similarities = abs(X(same_cluster, :) * X(i, :)');
            a = mean(1 - similarities(similarities < 1));  % Exclude self-similarity (=1)
            if isnan(a) || a < 0, a = 0; end
        else
            a = 0;
        end
        
        % Inter-cluster: mean distance to nearest other cluster
        b = Inf;
        for kk = 1:K
            if kk == k, continue; end
            other_cluster = find(labels == kk);
            if ~isempty(other_cluster)
                similarities = abs(X(other_cluster, :) * X(i, :)');
                b_kk = mean(1 - similarities);
                b = min(b, b_kk);
            end
        end
        
        if isinf(b) || isnan(b)
            b = a;
        end
        
        if a == 0 && b == 0
            sil_vals(i) = 0;
        else
            sil_vals(i) = (b - a) / (max(a, b) + eps);
        end
    end
    
    sil = mean(sil_vals(~isnan(sil_vals) & ~isinf(sil_vals)));
    if isnan(sil) || isinf(sil)
        sil = 0;
    end
end

function wss = compute_within_ss(X, labels, K)
% Compute within-cluster sum of squares using cosine distance (matching Koenig)
    
    wss = 0;
    centers = recover_centers_from_labels(X, labels, K);
    for k = 1:K
        cluster_idx = find(labels == k);
        if ~isempty(cluster_idx)
            cluster_data = X(cluster_idx, :);
            % Use cosine distance: 1 - |correlation|
            dists = 1 - abs(cluster_data * centers(k, :)');
            wss = wss + sum(dists.^2);
        end
    end
end

function ch = compute_calinski_harabasz_score(X, labels, K)
% Compute the Calinski-Harabasz score in the fitted feature space.

    X = double(X);
    labels = double(labels(:));
    [N, D] = size(X);
    if K <= 1 || N <= K || D == 0
        ch = NaN;
        return;
    end

    grand_mean = mean(X, 1);
    W = 0;
    B = 0;
    for k = 1:K
        cluster_idx = (labels == k);
        nk = nnz(cluster_idx);
        if nk == 0
            continue;
        end
        Xk = X(cluster_idx, :);
        mu_k = mean(Xk, 1);
        diffs = Xk - mu_k;
        W = W + sum(diffs(:).^2);
        mean_diff = mu_k - grand_mean;
        B = B + nk * sum(mean_diff.^2);
    end

    if W <= eps
        ch = Inf;
        return;
    end

    ch = (B / max(K - 1, 1)) / (W / max(N - K, 1));
end

function centers = recover_centers_from_labels(maps, labels, K)
% Recover cluster centers with Koenig-style polarity-invariant update.
    
    [N, D] = size(maps);
    centers = zeros(K, D);
    
    for k = 1:K
        idx = find(labels == k);
        
        if isempty(idx)
            centers(k, :) = maps(randi(N), :);
            continue;
        end
        
        cluster_maps = maps(idx, :);
        cvm = cluster_maps' * cluster_maps;
        try
            [v, ~] = eigs(double(cvm), 1);
            centers(k, :) = v(:, 1)';
        catch
            [v, d] = eig(double(cvm));
            [~, best] = max(diag(d));
            centers(k, :) = v(:, best)';
        end
        centers(k, :) = centers(k, :) - mean(centers(k, :));
        centers(k, :) = centers(k, :) / (norm(centers(k, :)) + eps);
    end
end

function Results = create_empty_results(Sim, K_candidates, n_maps, C_dims, criterion)
% Create empty results structure for failed fits
    if isfield(Sim, 'K_true') && isfinite(Sim.K_true) && Sim.K_true > 0
        maps_true = nan(Sim.K_true, C_dims);
        K_true_val = Sim.K_true;
    else
        maps_true = [];
        K_true_val = NaN;
    end
    if isfield(Sim, 'SNR_dB'), SNR_dB_val = Sim.SNR_dB; else, SNR_dB_val = NaN; end
    if isfield(Sim, 'duration_s'), duration_s_val = Sim.duration_s; else, duration_s_val = NaN; end
    
    Results = struct(...
        'method', 'spm_vb', ...
        'criterion', criterion, ...
        'K_true', K_true_val, ...
        'K_estimated', NaN, ...
        'K_model_selected', NaN, ...
        'K_candidates', K_candidates, ...
        'K_effective_vals', nan(numel(K_candidates), 1), ...
        'SNR_dB', SNR_dB_val, ...
        'duration_s', duration_s_val, ...
        'n_maps', n_maps, ...
        'centers', nan(1, C_dims), ...
        'centers_by_K', {cell(numel(K_candidates), 1)}, ...
        'maps_true', maps_true, ...
        'labels', ones(n_maps, 1), ...
        'labels_by_K', {cell(numel(K_candidates), 1)}, ...
        'free_energy', -Inf(numel(K_candidates), 1), ...
        'matrix_free_energy_vals', -Inf(numel(K_candidates), 1), ...
        'spm_pca_free_energy_vals', -Inf(numel(K_candidates), 1), ...
        'free_energy_vals', -Inf(numel(K_candidates), 1), ...
        'log_likelihood_vals', nan(numel(K_candidates), 1), ...
        'bic_vals', nan(numel(K_candidates), 1), ...
        'icl_vals', nan(numel(K_candidates), 1), ...
        'assignment_entropy_vals', nan(numel(K_candidates), 1), ...
        'n_parameters_vals', nan(numel(K_candidates), 1), ...
        'silhouette_vals', -ones(numel(K_candidates), 1), ...
        'within_ss', Inf(numel(K_candidates), 1), ...
        'gev_vals', zeros(numel(K_candidates), 1), ...
        'calinski_harabasz_vals', nan(numel(K_candidates), 1), ...
        'covariance_trace_mean_vals', nan(numel(K_candidates), 1), ...
        'covariance_trace_median_vals', nan(numel(K_candidates), 1), ...
        'covariance_logdet_mean_vals', nan(numel(K_candidates), 1), ...
        'covariance_logdet_median_vals', nan(numel(K_candidates), 1), ...
        'covariance_logdet_per_dim_mean_vals', nan(numel(K_candidates), 1), ...
        'selection_score_by_k', nan(numel(K_candidates), 1), ...
        'selection_details', struct(), ...
        'spm_mix_model_summaries', repmat(empty_spm_mix_summary(C_dims), numel(K_candidates), 1), ...
        'selected_spm_mix_model', empty_spm_mix_summary(C_dims), ...
        'selected_spm_covariances', nan(C_dims, C_dims, 0), ...
        'selected_spm_means', nan(0, C_dims), ...
        'selected_spm_priors', nan(0, 1), ...
        'feature_backfit_models', {cell(numel(K_candidates), 1)}, ...
        'selected_feature_backfit_model', struct(), ...
        'matrix_evidence_models', {cell(numel(K_candidates), 1)}, ...
        'selected_matrix_evidence_model', struct(), ...
        'pca_info', struct(), ...
        'polarity_feature_info', struct(), ...
        'matrix_feature_info', struct(), ...
        'polarity_duplicate_info', {cell(numel(K_candidates), 1)}, ...
        'recovery_metrics', struct('n_matched', 0, 'mean_recovery_matched', NaN, ...
            'mean_recovery_padded', NaN, 'f1_score', NaN, 'sensitivity', NaN, ...
            'precision', NaN, 'match_similarities', []), ...
        'mean_recovery', NaN, ...
        'recovery_corr', [], ...
        'avg_recovery_per_state', NaN, ...
        'valid_fit', false, ...
        'runtime', 0);
end
