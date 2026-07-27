function [recovery_metrics] = microstate_partial_alignment(true_maps, estimated_maps, varargin)
% MICROSTATE_PARTIAL_ALIGNMENT: Compute recovery metrics with partial alignment
%
% Handles the case where K_estimated ≠ K_true by finding the best partial
% matching between estimated and true microstates using rectangular Hungarian.
%
% INPUTS:
%   true_maps        - True microstate maps (K_true x C)
%   estimated_maps   - Estimated microstate maps (K_est x C)
%   Optional Name-Value Pairs:
%     'distance_type' - 'cosine' (default) or 'euclidean'
%     'threshold'     - Minimum correlation for valid match (default 0.0)
%     'polarity'      - true (default) to use abs correlation
%
% OUTPUTS:
%   recovery_metrics - Comprehensive structure with all metrics

    p = inputParser;
    addParameter(p, 'distance_type', 'cosine', @(x) ismember(x, {'cosine', 'euclidean'}));
    addParameter(p, 'threshold', 0.0, @isnumeric);
    addParameter(p, 'polarity', true, @islogical);
    parse(p, varargin{:});
    
    distance_type = p.Results.distance_type;
    threshold = p.Results.threshold;
    use_polarity = p.Results.polarity;
    
    % Validate inputs
    if isempty(true_maps) || isempty(estimated_maps)
        error('true_maps and estimated_maps must be non-empty');
    end
    
    % Normalize both sets
    true_maps = normalize_maps_local(true_maps);
    estimated_maps = normalize_maps_local(estimated_maps);
    
    K_true = size(true_maps, 1);
    K_est = size(estimated_maps, 1);
    
    % Compute similarity matrix
    if strcmp(distance_type, 'cosine')
        if use_polarity
            similarity = abs(estimated_maps * true_maps');  % K_est x K_true
        else
            similarity = estimated_maps * true_maps';
            similarity = max(similarity, 0);
        end
    elseif strcmp(distance_type, 'euclidean')
        dist_mat = pdist2(estimated_maps, true_maps);
        similarity = 1 - (dist_mat / sqrt(2));
        similarity = max(similarity, 0);
    end
    
    % Find best rectangular partial matching
    [match_assignment] = rectangular_hungarian(similarity, threshold);
    
    % Extract match information
    if isempty(match_assignment)
        n_matched = 0;
        recovery_matched = [];
        mean_recovery_matched = 0;
        matched_pairs = [];
        unmatched_true = 1:K_true;
        unmatched_estimated = 1:K_est;
    else
        matched_pairs = match_assignment(:, 1:2);
        n_matched = size(matched_pairs, 1);
        
        recovery_matched = zeros(n_matched, 1);
        for i = 1:n_matched
            est_idx = matched_pairs(i, 1);
            true_idx = matched_pairs(i, 2);
            recovery_matched(i) = similarity(est_idx, true_idx);
        end
        
        mean_recovery_matched = mean(recovery_matched);
        unmatched_true = setdiff(1:K_true, matched_pairs(:, 2));
        unmatched_estimated = setdiff(1:K_est, matched_pairs(:, 1));
    end
    
    % Compute padded recovery
    recovery_padded = zeros(max(K_est, K_true), 1);
    if n_matched > 0
        recovery_padded(1:n_matched) = recovery_matched;
    end
    mean_recovery_padded = mean(recovery_padded);
    
    % Compute performance metrics
    n_true_pos = n_matched;
    n_false_pos = K_est - n_matched;
    n_false_neg = K_true - n_matched;
    
    sensitivity = n_true_pos / max(1, n_true_pos + n_false_neg);
    precision = n_true_pos / max(1, n_true_pos + n_false_pos);
    
    if (precision + sensitivity) > 0
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity);
    else
        f1_score = 0;
    end
    
    % Return comprehensive structure
    recovery_metrics = struct(...
        'K_true', K_true, ...
        'K_estimated', K_est, ...
        'n_matched', n_matched, ...
        'recovery_matched', recovery_matched, ...
        'mean_recovery_matched', mean_recovery_matched, ...
        'mean_recovery_padded', mean_recovery_padded, ...
        'true_pos', n_true_pos, ...
        'false_pos', n_false_pos, ...
        'false_neg', n_false_neg, ...
        'sensitivity', sensitivity, ...
        'precision', precision, ...
        'f1_score', f1_score, ...
        'match_assignment', matched_pairs, ...
        'match_similarities', recovery_matched, ...
        'unmatched_true', unmatched_true, ...
        'unmatched_estimated', unmatched_estimated, ...
        'similarity_matrix', similarity);
end

function [match_assignment] = rectangular_hungarian(similarity_matrix, threshold)
    % RECTANGULAR_HUNGARIAN: Optimal partial assignment for rectangular matrices
    
    if nargin < 2
        threshold = 0;
    end
    
    [M, N] = size(similarity_matrix);
    
    % Convert to cost matrix
    cost = 1 - similarity_matrix;
    cost(cost < 0) = 0;
    
    % Make square matrix by padding
    n = max(M, N);
    max_cost = max(cost(:)) + 1;
    cost_sq = ones(n, n) * max_cost;
    cost_sq(1:M, 1:N) = cost;
    
    % Apply Hungarian algorithm
    try
        [assignment] = hungarian_munkres(cost_sq);
    catch
        match_assignment = [];
        return;
    end
    
    % Extract valid assignments
    if isempty(assignment)
        match_assignment = [];
        return;
    end
    
    valid_idx = assignment(:, 1) <= M & assignment(:, 2) <= N;
    assignment = assignment(valid_idx, :);
    
    % Filter by threshold
    if ~isempty(assignment)
        match_sims = zeros(size(assignment, 1), 1);
        for i = 1:size(assignment, 1)
            match_sims(i) = similarity_matrix(assignment(i, 1), assignment(i, 2));
        end
        
        threshold_idx = match_sims >= threshold;
        assignment = assignment(threshold_idx, :);
    end
    
    match_assignment = assignment;
end

function [assignment] = hungarian_munkres(cost_matrix)
    % HUNGARIAN_MUNKRES: Complete Hungarian algorithm implementation
    
    n = size(cost_matrix, 1);
    cost = cost_matrix;
    
    % Step 1: Subtract row minima
    for i = 1:n
        row_min = min(cost(i, :));
        if isfinite(row_min)
            cost(i, :) = cost(i, :) - row_min;
        end
    end
    
    % Step 2: Subtract column minima
    for j = 1:n
        col_min = min(cost(:, j));
        if isfinite(col_min)
            cost(:, j) = cost(:, j) - col_min;
        end
    end
    
    % Repeat Steps 3-5 until matching is complete
    max_iterations = 2 * n;
    for iteration = 1:max_iterations
        [row_covered, col_covered] = cover_zeros_greedy(cost);
        
        n_covered = sum(row_covered) + sum(col_covered);
        if n_covered >= n
            break;
        end
        
        uncovered_min = inf;
        for i = 1:n
            for j = 1:n
                if ~row_covered(i) && ~col_covered(j)
                    if cost(i, j) < uncovered_min
                        uncovered_min = cost(i, j);
                    end
                end
            end
        end
        
        if ~isfinite(uncovered_min)
            break;
        end
        
        for i = 1:n
            for j = 1:n
                if row_covered(i) && col_covered(j)
                    cost(i, j) = cost(i, j) + uncovered_min;
                elseif ~row_covered(i) && ~col_covered(j)
                    cost(i, j) = cost(i, j) - uncovered_min;
                end
            end
        end
    end
    
    % Step 5: Extract assignment
    assignment = extract_assignment(cost);
    
    if size(assignment, 1) < n
        assigned_rows = assignment(:, 1);
        assigned_cols = assignment(:, 2);
        unassigned_rows = setdiff(1:n, assigned_rows);
        unassigned_cols = setdiff(1:n, assigned_cols);
        
        for i = 1:length(unassigned_rows)
            if i <= length(unassigned_cols)
                assignment = [assignment; unassigned_rows(i), unassigned_cols(i)];
            end
        end
    end
end

function [row_covered, col_covered] = cover_zeros_greedy(cost_matrix)
    n = size(cost_matrix, 1);
    row_covered = false(n, 1);
    col_covered = false(n, 1);
    
    for iteration = 1:n
        row_zeros = zeros(n, 1);
        col_zeros = zeros(n, 1);
        
        for i = 1:n
            if ~row_covered(i)
                row_zeros(i) = sum((cost_matrix(i, :) == 0) & ~col_covered');
            end
        end
        
        for j = 1:n
            if ~col_covered(j)
                col_zeros(j) = sum((cost_matrix(:, j) == 0) & ~row_covered);
            end
        end
        
        [max_row_zeros, max_row_idx] = max(row_zeros);
        [max_col_zeros, max_col_idx] = max(col_zeros);
        
        if max_row_zeros >= max_col_zeros && max_row_zeros > 0
            row_covered(max_row_idx) = true;
        elseif max_col_zeros > 0
            col_covered(max_col_idx) = true;
        else
            break;
        end
    end
end

function [assignment] = extract_assignment(cost_matrix)
    n = size(cost_matrix, 1);
    assignment = [];
    row_assigned = false(n, 1);
    col_assigned = false(n, 1);
    
    % First pass: rows/cols with unique zeros
    for i = 1:n
        zero_cols = find(cost_matrix(i, :) == 0 & ~col_assigned');
        if numel(zero_cols) == 1 && ~row_assigned(i)
            assignment = [assignment; i, zero_cols];
            row_assigned(i) = true;
            col_assigned(zero_cols) = true;
        end
    end
    
    % Second pass: greedy for remaining
    for i = 1:n
        if ~row_assigned(i)
            zero_cols = find(cost_matrix(i, :) == 0 & ~col_assigned');
            if ~isempty(zero_cols)
                assignment = [assignment; i, zero_cols(1)];
                row_assigned(i) = true;
                col_assigned(zero_cols(1)) = true;
            end
        end
    end
end

function maps_norm = normalize_maps_local(maps)
    if isempty(maps)
        maps_norm = maps;
        return;
    end
    
    if size(maps, 1) == 0
        maps_norm = maps;
        return;
    end
    
    maps_norm = maps - mean(maps, 2);
    norms = sqrt(sum(maps_norm.^2, 2));
    norms(norms < eps) = 1;
    maps_norm = maps_norm ./ norms;
end