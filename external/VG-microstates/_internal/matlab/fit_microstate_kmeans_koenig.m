function Results = fit_microstate_kmeans_koenig(Sim, K_candidates, criterion)
% FIT_MICROSTATE_KMEANS_KOENIG: K-Means with Thomas Koenig's algorithm
%
% Implements the k-means clustering method from MICROSTATELAB by Thomas Koenig
% with polarity invariance, GEV calculation, and robust model selection.
% 
% Uses imported functions from the microstates repository (eeg_kMeans).
%
% INPUTS:
%   Sim          - Simulation structure
%   K_candidates - Vector of K values to test
%   criterion    - 'silhouette', 'gev', or 'elbow'
%
% OUTPUTS:
%   Results      - Complete results with recovery metrics

    if nargin < 2, K_candidates = 2:10; end
    if nargin < 3, criterion = 'silhouette'; end
    
    t_start = tic;

    fprintf('\n========================================\n');
    fprintf('K-Means Microstate Fitting (Koenig Method)\n');
    fprintf('========================================\n');
    fprintf('Computing ALL criteria (will apply %s)\n', criterion);
    if isfield(Sim, 'K_true') && ~isnan(Sim.K_true)
        fprintf('True K: %d, SNR: %+.1f dB\n', Sim.K_true, Sim.SNR_dB);
    else
        fprintf('Real data mode (no ground truth)\n');
    end

    util = microstate_utilities();
    [maps_norm, idx_peaks, gfp_vec, n_maps, C_dims, maps_original, preprocessing_info] = util.preprocess_maps(Sim);
    features = maps_norm;
    fprintf('Fitting K-Means for K = %s...\n', mat2str(K_candidates));
    nK = numel(K_candidates);
    R = 20; % Number of restarts (can be parameterized)
    
    % Compute ALL criteria
    gev_vals = zeros(nK, 1);
    silhouette_vals = zeros(nK, 1);
    within_ss = zeros(nK, 1);
    centers_all = cell(nK, 1);
    labels_all = cell(nK, 1);
    exp_var_all = zeros(nK, 1);
    
    for iK = 1:nK
        K = K_candidates(iK);
        fprintf('  K=%d: ', K);
        
        try
            % ✅ IMPORT FROM MICROSTATES REPOSITORY
            % Call eeg_kMeans from the Koenig microstates repository
            % Parameters: eeg_kMeans(eeg, n_mod, reruns, max_n, flags, chanloc)
            
            % Build flags for polarity-insensitive clustering (standard for resting-state EEG)
            flags = '';  % No polarity mode (p flag) = polarity-insensitive (abs correlation)
            % flags = 'p';  % Use this for polarity-sensitive clustering if needed
            
            % Call the K-Means implementation
            % eeg_kMeans returns: [b_model, b_ind, b_loading, exp_var]
            % where:
            %   b_model = cluster centers (K x N_channels)
            %   b_ind = cluster assignments (N_samples x 1)
            %   b_loading = amplitude/loading for each sample (N_samples x 1)
            %   exp_var = explained variance per cluster (K x 1) - NOT scalar!
            [centers, labels, loading, exp_var_per_cluster] = eeg_kMeans(maps_norm, K, 20, n_maps, flags);
            
            centers_all{iK} = centers;
            labels_all{iK} = labels;
            
            % exp_var_per_cluster is a vector of length K
            % Store mean explained variance
            exp_var_all(iK) = mean(exp_var_per_cluster);
            
            % Compute ALL scores
            
            % 1. GEV (KOENIG METHOD: uses original amplitudes)
            sim = abs(maps_original * centers');
            [max_sim, ~] = max(sim, [], 2);
            gfp_squared = sum(maps_original.^2, 2);
            gev_vals(iK) = sum(max_sim.^2) / (sum(gfp_squared) + eps);
            
            % 2. Silhouette (KOENIG METHOD - polarity-insensitive cosine-based)
            silhouette_vals(iK) = silhouette_microstatelab(maps_norm, labels, centers);
            
            % 3. Within-cluster sum of squares (Elbow)
            wss = 0;
            for k = 1:K
                cluster_maps = maps_norm(labels == k, :);
                if ~isempty(cluster_maps)
                    dists = 1 - abs(cluster_maps * centers(k, :)');
                    wss = wss + sum(dists.^2);
                end
            end
            within_ss(iK) = wss;
            
            fprintf('GEV=%.3f, Sil=%.3f, WSS=%.1f, ExpVar=%.3f\n', gev_vals(iK), silhouette_vals(iK), within_ss(iK), exp_var_all(iK));
        catch ME
            fprintf('ERROR: %s\n', ME.message);
            gev_vals(iK) = 0;
            silhouette_vals(iK) = -1;
            within_ss(iK) = Inf;
            exp_var_all(iK) = 0;
        end
    end
    
    % Select K using specified criterion
    fprintf('\nModel selection using: %s\n', criterion);
    
    switch criterion
        case 'silhouette'
            % KOENIG METHOD: Simple max
            [K_est, best_score] = select_K_from_silhouette_koenig(silhouette_vals, K_candidates);
            
        case 'gev'
            [best_score, idx] = max(gev_vals);
            K_est = K_candidates(idx);
            
        case 'elbow'
            [K_est, best_score] = select_K_from_elbow(within_ss, K_candidates);
            
        otherwise
            error('Unknown criterion: %s', criterion);
    end
    
    best_idx = K_est == K_candidates;
    best_idx = find(best_idx, 1);
    
    fprintf('Selected: K=%d (score=%.4f', K_est, best_score);
    if isfield(Sim, 'K_true') && ~isnan(Sim.K_true)
        fprintf(', true K=%d)\n\n', Sim.K_true);
    else
        fprintf(')\n\n');
    end
    
    centers = centers_all{best_idx};
    labels = labels_all{best_idx};
    
    % Recovery (only if ground truth is available)
    if isfield(Sim, 'maps_true') && ~isempty(Sim.maps_true)
        true_maps_norm = util.normalize_maps(Sim.maps_true);
        recovery_metrics = microstate_partial_alignment(true_maps_norm, centers, ...
            'distance_type', 'cosine', 'threshold', 0.0, 'polarity', true);
        
        fprintf('Map Recovery: Matched=%d, F1=%.4f, Sens=%.4f, Prec=%.4f\n\n', ...
            recovery_metrics.n_matched, recovery_metrics.f1_score, ...
            recovery_metrics.sensitivity, recovery_metrics.precision);
    else
        % No ground truth - create empty recovery metrics
        recovery_metrics = struct(...
            'K_true', NaN, ...
            'K_estimated', K_est, ...
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
    
    % Return ALL computed scores
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
        'method', 'kmeans_koenig', ...
        'criterion', criterion, ...
        'K_true', K_true_val, ...
        'K_estimated', K_est, ...
        'K_candidates', K_candidates, ...
        'SNR_dB', SNR_dB_val, ...
        'duration_s', duration_s_val, ...
        'n_maps', n_maps, ...
        'centers', centers, ...
        'centers_by_K', {centers_all}, ...
        'maps_true', true_maps_norm, ...
        'labels', labels, ...
        'labels_by_K', {labels_all}, ...
        'gev_vals', gev_vals, ...
        'silhouette_vals', silhouette_vals, ...
        'free_energy_vals', zeros(nK, 1), ...
        'within_ss', within_ss, ...
        'explained_variance', exp_var_all, ...
        'best_criterion_value', best_score, ...
        'maps_nc', maps_norm, ...
        'idx_peaks', idx_peaks, ...
        'gfp_vec', gfp_vec, ...
        'preprocessing_info', preprocessing_info, ...
        'recovery_metrics', recovery_metrics, ...
        'mean_recovery', recovery_metrics.mean_recovery_matched, ...
        'recovery_corr', recovery_metrics.match_similarities, ...
        'avg_recovery_per_state', recovery_metrics.mean_recovery_padded, ...
        'valid_fit', true, ...
        'runtime', runtime);
end

% ======================== K SELECTION HELPERS ========================

function [K_est, score] = select_K_from_silhouette_koenig(sil_vals, K_candidates)
    % KOENIG METHOD: Simple maximum without 2nd derivatives
    % Find K with highest silhouette score
    
    [best_score, idx] = max(sil_vals);
    K_est = K_candidates(idx);
    score = best_score;
    
    fprintf('  Silhouette (Koenig): selected K=%d (score=%.4f)\n', K_est, score);
end

function [K_est, score] = select_K_from_elbow(wss_vals, K_candidates)
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

% ======================== IMPORTED SILHOUETTE METHOD ========================

function sil = silhouette_microstatelab(X, labels, centers)
    % SILHOUETTE_MICROSTATELAB: Silhouette score using Koenig's method
    % Computes polarity-insensitive (absolute) cosine-based silhouette coefficient
    %
    % INPUTS:
    %   X       - Data matrix (N_samples x N_channels)
    %   labels  - Cluster assignments (N_samples x 1)
    %   centers - Cluster centers (K_clusters x N_channels)
    %
    % OUTPUT:
    %   sil     - Average silhouette coefficient
    
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
