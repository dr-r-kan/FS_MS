function analyze_montage_robustness(varargin)
% ANALYZE_MONTAGE_ROBUSTNESS: Comprehensive analysis of montage effects
%
% Analyzes how reduced EEG montages (71 → 20 → 12 leads) affect:
%   - K estimation accuracy
%   - Microstate center identification precision
%   - Recovery metrics stability
%
% USAGE:
%   analyze_montage_robustness()
%   analyze_montage_robustness(results_csv)
%   analyze_montage_robustness(results_csv, 'output_dir', out_dir)
%   analyze_montage_robustness(..., 'K_true', [4 5], 'SNR_dB', [-10 10])
%
% INPUTS:
%   results_csv - Path to comparison_results.csv (optional)
%   'output_dir' - Directory for plots and summary files
%   'K_true' - Filter by true K values
%   'SNR_dB' - Filter by SNR values
%   'method' - Filter by method name
%   'criterion' - Filter by criterion name
%
% OUTPUTS:
%   Generates plots:
%     - K estimation accuracy vs lead count
%     - Mean absolute K error vs lead count
%     - Center correlation vs lead count
%     - Center precision vs lead count
%     - Heatmaps (method × montage performance)
%     - SNR interaction plots
%   Exports summary_montage_robustness.csv

    setup_microstate_paths();
    util = microstate_utilities();
    cfg = util.load_config();

    % Parse inputs
    p = inputParser;
    default_csv = fullfile(char(cfg.paths.simulation_output_dir), 'results', 'comparison_results.csv');
    default_out_dir = fullfile(char(cfg.paths.simulation_output_dir), 'analysis_plots', 'montage_robustness');
    
    addOptional(p, 'results_csv', default_csv, @(x) ischar(x) || isstring(x));
    addParameter(p, 'output_dir', default_out_dir, @(x) ischar(x) || isstring(x));
    addParameter(p, 'K_true', [], @isnumeric);
    addParameter(p, 'SNR_dB', [], @isnumeric);
    addParameter(p, 'method', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'criterion', '', @(x) ischar(x) || isstring(x));
    parse(p, varargin{:});
    
    results_csv = util.resolve_path(char(p.Results.results_csv), util.project_root());
    output_dir = util.resolve_path(char(p.Results.output_dir), util.project_root());
    filter_K = p.Results.K_true;
    filter_SNR = p.Results.SNR_dB;
    filter_method = char(p.Results.method);
    filter_criterion = char(p.Results.criterion);
    
    % Create output directory
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    
    fprintf('\n========================================\n');
    fprintf('Montage Robustness Analysis\n');
    fprintf('========================================\n');
    fprintf('Results CSV: %s\n', results_csv);
    fprintf('Output directory: %s\n\n', output_dir);
    
    % Load data
    if ~exist(results_csv, 'file')
        error('Results file not found: %s', results_csv);
    end
    
    T = readtable(results_csv);
    fprintf('✓ Loaded %d observations\n', height(T));
    
    % Check for montage columns
    if ~ismember('montage_type', T.Properties.VariableNames)
        error('Results file does not contain montage_type column. Please run pipeline with montage analysis enabled.');
    end

    if ~ismember('method', T.Properties.VariableNames)
        error('Results file does not contain method column.');
    end

    method_raw = cellstr(string(T.method));
    method_canon = cellfun(@canonicalize_method_local, method_raw, 'UniformOutput', false);
    supported_mask = ismember(method_canon, {'koenig kmeans', 'spm vb'});
    if ~any(supported_mask)
        error(['No supported methods found in montage results. ', ...
            'Expected K-Means and/or VB-GMM rows.']);
    end
    if any(~supported_mask)
        dropped_methods = unique(method_canon(~supported_mask), 'stable');
        fprintf('  Filtering out unsupported methods: %s\n', strjoin(dropped_methods, ', '));
        T = T(supported_mask, :);
        method_canon = method_canon(supported_mask);
    end
    T.method = string(method_canon(:));
    
    % Apply filters
    if ~isempty(filter_K)
        T = T(ismember(T.K_true, filter_K), :);
        fprintf('  Filtered to K_true = %s: %d obs\n', mat2str(filter_K), height(T));
    end
    
    if ~isempty(filter_SNR)
        T = T(ismember(T.SNR_dB, filter_SNR), :);
        fprintf('  Filtered to SNR_dB = %s: %d obs\n', mat2str(filter_SNR), height(T));
    end
    
    if ~isempty(filter_method)
        filter_method = canonicalize_method_local(filter_method);
        T = T(strcmp(T.method, filter_method), :);
        fprintf('  Filtered to method = %s: %d obs\n', display_method_label_local(filter_method), height(T));
    end
    
    if ~isempty(filter_criterion)
        T = T(strcmp(T.criterion, filter_criterion), :);
        fprintf('  Filtered to criterion = %s: %d obs\n', filter_criterion, height(T));
    end

    if height(T) == 0
        error('No rows remain after applying montage robustness filters.');
    end
    
    fprintf('\n');

    % Derived columns for robust downstream summaries/plots
    if ~ismember('K_abs_error', T.Properties.VariableNames)
        T.K_abs_error = abs(T.K_error);
    end

    if ~ismember('criterion_clean', T.Properties.VariableNames)
        T.criterion_clean = strrep(string(T.criterion), '_', ' ');
    else
        T.criterion_clean = string(T.criterion_clean);
    end
    T.method = string(T.method);
    T.montage_type = string(T.montage_type);
    T.criterion = string(T.criterion);
    
    % Get unique montages and methods
    montages = cellstr(unique(T.montage_type));
    methods = cellstr(unique(T.method));
    criteria = cellstr(unique(T.criterion));
    
    fprintf('Montages found: %s\n', strjoin(montages, ', '));
    fprintf('Methods: %s\n', strjoin(methods, ', '));
    fprintf('Criteria: %s\n', strjoin(criteria, ', '));
    fprintf('\n');

    design = describe_method_criterion_design(T);
    T_method = T;
    method_context = struct( ...
        'suffix', '', ...
        'title_suffix', '', ...
        'summary_label', 'all_rows');

    if numel(methods) > 1 && isempty(filter_criterion)
        if isempty(design.shared_criteria)
            warning(['No shared criteria across methods. Method-by-montage comparisons ' ...
                     'cannot be made fairly without a criterion filter.']);
            T_method = T([]);
        else
            shared_mask = ismember(T.criterion, design.shared_criteria);
            T_method = T(shared_mask, :);
            method_context.suffix = '_shared_criteria';
            method_context.title_suffix = sprintf(' (shared criteria: %s)', ...
                strjoin(design.shared_criteria, ', '));
            method_context.summary_label = 'shared_criteria_only';

            fprintf(['Method-comparison plots will use only criteria shared across all methods: %s\n'], ...
                strjoin(design.shared_criteria, ', '));
            fprintf('  Retained %d / %d observations for fair method comparisons.\n\n', ...
                height(T_method), height(T));
        end
    end
    
    % ===== ANALYSIS 1: K ESTIMATION ACCURACY VS LEAD COUNT =====
    fprintf('Generating K estimation accuracy plots...\n');
    if ~isempty(T_method)
        plot_k_accuracy_by_montage(T_method, output_dir, method_context);
    end
    
    % ===== ANALYSIS 2: RECOVERY METRICS VS LEAD COUNT =====
    fprintf('Generating recovery metrics plots...\n');
    if ~isempty(T_method)
        plot_recovery_by_montage(T_method, output_dir, method_context);
    end
    
    % ===== ANALYSIS 3: HEATMAPS (METHOD × MONTAGE) =====
    fprintf('Generating heatmaps...\n');
    if ~isempty(T_method)
        plot_heatmaps(T_method, output_dir, method_context);
    end
    
    % ===== ANALYSIS 4: SNR INTERACTION PLOTS =====
    fprintf('Generating SNR interaction plots...\n');
    if ~isempty(T_method)
        plot_snr_interactions(T_method, output_dir, method_context);
    end
    
    % ===== ANALYSIS 5: SUMMARY STATISTICS =====
    fprintf('Computing summary statistics...\n');
    summary_stats = compute_summary_statistics(T, 'by_criterion', true, ...
        'summary_scope', "all_rows");
    
    % Export summary to CSV
    summary_csv = fullfile(output_dir, 'summary_montage_robustness_all_rows.csv');
    writetable(summary_stats, summary_csv);
    fprintf('✓ Saved summary statistics: %s\n', summary_csv);

    if ~isempty(T_method)
        summary_method = compute_summary_statistics(T_method, 'by_criterion', false, ...
            'summary_scope', string(method_context.summary_label));
        summary_method_csv = fullfile(output_dir, 'summary_montage_robustness_method_comparable.csv');
        writetable(summary_method, summary_method_csv);
        fprintf('✓ Saved method-comparable summary: %s\n', summary_method_csv);
    end
    
    fprintf('\n========================================\n');
    fprintf('Analysis Complete!\n');
    fprintf('Plots saved to: %s\n', output_dir);
    fprintf('========================================\n\n');
end

% ======================== PLOTTING FUNCTIONS ========================

function design = describe_method_criterion_design(T)
    methods = cellstr(unique(T.method));
    criteria_per_method = cell(numel(methods), 1);
    for i = 1:numel(methods)
        criteria_per_method{i} = cellstr(unique(T.criterion(strcmp(T.method, methods{i}))));
    end

    shared_criteria = criteria_per_method{1};
    for i = 2:numel(criteria_per_method)
        shared_criteria = intersect(shared_criteria, criteria_per_method{i}, 'stable');
    end

    fprintf('Observed method-criterion design:\n');
    for i = 1:numel(methods)
        fprintf('  %s: %s\n', methods{i}, strjoin(criteria_per_method{i}, ', '));
    end
    if isempty(shared_criteria)
        fprintf('Shared criteria across all methods: none\n\n');
    else
        fprintf('Shared criteria across all methods: %s\n\n', ...
            strjoin(shared_criteria, ', '));
    end

    design = struct();
    design.methods = methods;
    design.criteria_per_method = {criteria_per_method};
    design.shared_criteria = shared_criteria;
end

function plot_k_accuracy_by_montage(T, output_dir, context)
    % K estimation accuracy vs lead count by method
    
    montages = cellstr(unique(T.montage_type));
    methods = cellstr(unique(T.method));
    
    % Get lead counts for each montage
    lead_counts = zeros(size(montages));
    for i = 1:length(montages)
        idx = find(strcmp(T.montage_type, montages{i}), 1);
        if ~isempty(idx)
            lead_counts(i) = T.n_leads(idx);
        end
    end
    
    % Sort by lead count
    [lead_counts, sort_idx] = sort(lead_counts);
    montages = montages(sort_idx);
    
    figure('Position', [100 100 1200 500]);
    
    % Plot 1: Accuracy by method
    subplot(1, 2, 1);
    hold on;
    colors = lines(length(methods));
    
    for m = 1:length(methods)
        method = methods{m};
        acc = zeros(size(montages));
        err = zeros(size(montages));
        
        for i = 1:length(montages)
            idx = strcmp(T.montage_type, montages{i}) & strcmp(T.method, method);
            if any(idx)
                acc(i) = mean(T.K_correct(idx));
                err(i) = std(T.K_correct(idx)) / sqrt(sum(idx));
            end
        end
        
        errorbar(lead_counts, acc * 100, err * 100, 'o-', ...
            'LineWidth', 2, 'MarkerSize', 8, 'Color', colors(m, :), ...
            'DisplayName', display_method_label_local(method));
    end
    
    xlabel('Number of Leads');
    ylabel('K Estimation Accuracy (%)');
    title(['K Estimation Accuracy vs Lead Count' context.title_suffix]);
    legend('Location', 'best');
    grid on;
    set(gca, 'FontSize', 12);
    
    % Plot 2: Mean absolute error
    subplot(1, 2, 2);
    hold on;
    
    for m = 1:length(methods)
        method = methods{m};
        mae = zeros(size(montages));
        err = zeros(size(montages));
        
        for i = 1:length(montages)
            idx = strcmp(T.montage_type, montages{i}) & strcmp(T.method, method);
            if any(idx)
                mae(i) = mean(T.K_abs_error(idx));
                err(i) = std(T.K_abs_error(idx)) / sqrt(sum(idx));
            end
        end
        
        errorbar(lead_counts, mae, err, 'o-', ...
            'LineWidth', 2, 'MarkerSize', 8, 'Color', colors(m, :), ...
            'DisplayName', display_method_label_local(method));
    end
    
    xlabel('Number of Leads');
    ylabel('Mean Absolute K Error');
    title(['K Error vs Lead Count' context.title_suffix]);
    legend('Location', 'best');
    grid on;
    set(gca, 'FontSize', 12);
    
    saveas(gcf, fullfile(output_dir, ['k_accuracy_vs_leads' context.suffix '.png']));
    close(gcf);
end

function plot_recovery_by_montage(T, output_dir, context)
    % Recovery metrics vs lead count
    
    montages = cellstr(unique(T.montage_type));
    methods = cellstr(unique(T.method));
    
    % Get lead counts
    lead_counts = zeros(size(montages));
    for i = 1:length(montages)
        idx = find(strcmp(T.montage_type, montages{i}), 1);
        if ~isempty(idx)
            lead_counts(i) = T.n_leads(idx);
        end
    end
    [lead_counts, sort_idx] = sort(lead_counts);
    montages = montages(sort_idx);
    
    figure('Position', [100 100 1200 900]);
    colors = lines(length(methods));
    
    % Metrics to plot
    metrics = {'mean_recovery_matched', 'mean_recovery_padded', 'sensitivity', 'precision'};
    metric_names = {'Mean Recovery (Matched)', 'Mean Recovery (Padded)', 'Sensitivity', 'Precision'};
    
    for p = 1:4
        subplot(2, 2, p);
        hold on;
        
        for m = 1:length(methods)
            method = methods{m};
            vals = zeros(size(montages));
            errs = zeros(size(montages));
            
            for i = 1:length(montages)
                idx = strcmp(T.montage_type, montages{i}) & strcmp(T.method, method);
                if any(idx) && ismember(metrics{p}, T.Properties.VariableNames)
                    vals(i) = mean(T.(metrics{p})(idx));
                    errs(i) = std(T.(metrics{p})(idx)) / sqrt(sum(idx));
                end
            end
            
            errorbar(lead_counts, vals, errs, 'o-', ...
                'LineWidth', 2, 'MarkerSize', 8, 'Color', colors(m, :), ...
                'DisplayName', display_method_label_local(method));
        end
        
        xlabel('Number of Leads');
        ylabel(metric_names{p});
        title([metric_names{p} ' vs Lead Count' context.title_suffix]);
        legend('Location', 'best');
        grid on;
        set(gca, 'FontSize', 11);
    end
    
    saveas(gcf, fullfile(output_dir, ['recovery_metrics_vs_leads' context.suffix '.png']));
    close(gcf);
end

function plot_heatmaps(T, output_dir, context)
    % Heatmaps showing method × montage performance
    
    montages = cellstr(unique(T.montage_type));
    methods = cellstr(unique(T.method));
    
    % Get lead counts for labels
    lead_counts = zeros(size(montages));
    for i = 1:length(montages)
        idx = find(strcmp(T.montage_type, montages{i}), 1);
        if ~isempty(idx)
            lead_counts(i) = T.n_leads(idx);
        end
    end
    [lead_counts, sort_idx] = sort(lead_counts);
    montages = montages(sort_idx);
    
    % Create montage labels with lead counts
    montage_labels = cell(size(montages));
    for i = 1:length(montages)
        montage_labels{i} = sprintf('%s (%d)', montages{i}, lead_counts(i));
    end
    
    figure('Position', [100 100 1400 500]);
    
    % Heatmap 1: K Accuracy
    subplot(1, 3, 1);
    acc_matrix = zeros(length(methods), length(montages));
    for m = 1:length(methods)
        for i = 1:length(montages)
            idx = strcmp(T.montage_type, montages{i}) & strcmp(T.method, methods{m});
            if any(idx)
                acc_matrix(m, i) = mean(T.K_correct(idx)) * 100;
            end
        end
    end
    imagesc(acc_matrix);
    colorbar;
    colormap('hot');
    xlabel('Montage');
    ylabel('Method');
    title(['K Estimation Accuracy (%)' context.title_suffix]);
    xticks(1:length(montages));
    xticklabels(montage_labels);
    xtickangle(45);
    yticks(1:length(methods));
    yticklabels(cellfun(@display_method_label_local, methods, 'UniformOutput', false));
    set(gca, 'FontSize', 10);
    
    % Add text annotations
    for m = 1:length(methods)
        for i = 1:length(montages)
            text(i, m, sprintf('%.1f', acc_matrix(m, i)), ...
                'HorizontalAlignment', 'center', 'Color', 'white', 'FontSize', 9);
        end
    end
    
    % Heatmap 2: Mean Recovery
    subplot(1, 3, 2);
    rec_matrix = zeros(length(methods), length(montages));
    for m = 1:length(methods)
        for i = 1:length(montages)
            idx = strcmp(T.montage_type, montages{i}) & strcmp(T.method, methods{m});
            if any(idx)
                rec_matrix(m, i) = mean(T.mean_recovery_matched(idx));
            end
        end
    end
    imagesc(rec_matrix);
    colorbar;
    colormap('hot');
    xlabel('Montage');
    ylabel('Method');
    title(['Mean Recovery (Matched)' context.title_suffix]);
    xticks(1:length(montages));
    xticklabels(montage_labels);
    xtickangle(45);
    yticks(1:length(methods));
    yticklabels(cellfun(@display_method_label_local, methods, 'UniformOutput', false));
    set(gca, 'FontSize', 10);
    
    % Add text annotations
    for m = 1:length(methods)
        for i = 1:length(montages)
            text(i, m, sprintf('%.2f', rec_matrix(m, i)), ...
                'HorizontalAlignment', 'center', 'Color', 'white', 'FontSize', 9);
        end
    end
    
    % Heatmap 3: Mean K Error
    subplot(1, 3, 3);
    err_matrix = zeros(length(methods), length(montages));
    for m = 1:length(methods)
        for i = 1:length(montages)
            idx = strcmp(T.montage_type, montages{i}) & strcmp(T.method, methods{m});
            if any(idx)
                err_matrix(m, i) = mean(T.K_abs_error(idx));
            end
        end
    end
    imagesc(err_matrix);
    colorbar;
    colormap('hot');
    xlabel('Montage');
    ylabel('Method');
    title(['Mean Absolute K Error' context.title_suffix]);
    xticks(1:length(montages));
    xticklabels(montage_labels);
    xtickangle(45);
    yticks(1:length(methods));
    yticklabels(cellfun(@display_method_label_local, methods, 'UniformOutput', false));
    set(gca, 'FontSize', 10);
    
    % Add text annotations
    for m = 1:length(methods)
        for i = 1:length(montages)
            text(i, m, sprintf('%.2f', err_matrix(m, i)), ...
                'HorizontalAlignment', 'center', 'Color', 'white', 'FontSize', 9);
        end
    end
    
    saveas(gcf, fullfile(output_dir, ['heatmaps_method_montage' context.suffix '.png']));
    close(gcf);
end

function plot_snr_interactions(T, output_dir, context)
    % SNR × Montage interaction plots with method-specific lines
    
    montages = cellstr(unique(T.montage_type));
    methods = cellstr(unique(T.method));
    snr_vals = unique(T.SNR_dB);
    
    % Get lead counts
    lead_counts = zeros(size(montages));
    for i = 1:length(montages)
        idx = find(strcmp(T.montage_type, montages{i}), 1);
        if ~isempty(idx)
            lead_counts(i) = T.n_leads(idx);
        end
    end
    [lead_counts, sort_idx] = sort(lead_counts);
    montages = montages(sort_idx);
    
    figure('Position', [100 100 1400 max(350 * length(montages), 500)]);
    colors = lines(length(methods));

    for i = 1:length(montages)
        montage = montages{i};

        subplot(length(montages), 2, 2 * (i - 1) + 1);
        hold on;
        for m = 1:length(methods)
            method = methods{m};
            acc = nan(size(snr_vals));
            for s = 1:length(snr_vals)
                idx = strcmp(T.montage_type, montage) & strcmp(T.method, method) & (T.SNR_dB == snr_vals(s));
                if any(idx)
                    acc(s) = mean(T.K_correct(idx)) * 100;
                end
            end
            plot(snr_vals, acc, 'o-', 'LineWidth', 2, 'MarkerSize', 8, ...
                'Color', colors(m, :), 'DisplayName', display_method_label_local(method));
        end
        xlabel('SNR (dB)');
        ylabel('K Estimation Accuracy (%)');
        title(sprintf('%s (%d leads): K Accuracy%s', montage, lead_counts(i), context.title_suffix));
        legend('Location', 'best');
        grid on;
        set(gca, 'FontSize', 11);

        subplot(length(montages), 2, 2 * (i - 1) + 2);
        hold on;
        for m = 1:length(methods)
            method = methods{m};
            rec = nan(size(snr_vals));
            for s = 1:length(snr_vals)
                idx = strcmp(T.montage_type, montage) & strcmp(T.method, method) & (T.SNR_dB == snr_vals(s));
                if any(idx)
                    rec(s) = mean(T.mean_recovery_matched(idx));
                end
            end
            plot(snr_vals, rec, 'o-', 'LineWidth', 2, 'MarkerSize', 8, ...
                'Color', colors(m, :), 'DisplayName', display_method_label_local(method));
        end
        xlabel('SNR (dB)');
        ylabel('Mean Recovery (Matched)');
        title(sprintf('%s (%d leads): Recovery%s', montage, lead_counts(i), context.title_suffix));
        legend('Location', 'best');
        grid on;
        set(gca, 'FontSize', 11);
    end

    saveas(gcf, fullfile(output_dir, ['snr_method_montage_interactions' context.suffix '.png']));
    close(gcf);
end

function summary_stats = compute_summary_statistics(T, varargin)
    % Compute summary statistics by montage, method, and optionally criterion

    p = inputParser;
    addParameter(p, 'by_criterion', true, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'summary_scope', "all_rows", @(x) isstring(x) || ischar(x));
    parse(p, varargin{:});

    by_criterion = logical(p.Results.by_criterion);
    summary_scope = string(p.Results.summary_scope);
    
    montages = cellstr(unique(T.montage_type));
    methods = cellstr(unique(T.method));
    if by_criterion
        criteria = cellstr(unique(T.criterion));
    else
        criteria = {''};
    end
    
    rows = [];
    
    for i = 1:length(montages)
        montage = montages{i};
        
        % Get lead count
        idx = find(strcmp(T.montage_type, montage), 1);
        n_leads = T.n_leads(idx);
        
        for m = 1:length(methods)
            method = methods{m};

            for c = 1:length(criteria)
                idx = strcmp(T.montage_type, montage) & strcmp(T.method, method);
                if by_criterion
                    idx = idx & strcmp(T.criterion, criteria{c});
                end

                if any(idx)
                    row = struct();
                    row.summary_scope = char(summary_scope);
                    row.montage_type = char(montage);
                    row.n_leads = n_leads;
                    row.method = char(method);
                    if by_criterion
                        row.criterion = criteria{c};
                    else
                        row.criterion = 'all_shared_rows';
                    end
                    row.n_obs = sum(idx);
                    row.k_accuracy_mean = mean(T.K_correct(idx));
                    row.k_accuracy_std = std(T.K_correct(idx));
                    row.k_error_mean = mean(T.K_error(idx));
                    row.k_error_std = std(T.K_error(idx));
                    row.k_abs_error_mean = mean(T.K_abs_error(idx));
                    row.k_abs_error_std = std(T.K_abs_error(idx));
                    row.recovery_matched_mean = mean(T.mean_recovery_matched(idx));
                    row.recovery_matched_std = std(T.mean_recovery_matched(idx));
                    row.sensitivity_mean = mean(T.sensitivity(idx));
                    row.sensitivity_std = std(T.sensitivity(idx));
                    row.precision_mean = mean(T.precision(idx));
                    row.precision_std = std(T.precision(idx));

                    rows = [rows; row]; %#ok<AGROW>
                end
            end
        end
    end
    
    summary_stats = struct2table(rows);
end

function m = canonicalize_method_local(m_in)
    if isempty(m_in), m = ''; return; end
    m = char(string(m_in));
    m = lower(m);
    m = strrep(m, '_', ' ');
    m = regexprep(m, '\s+', ' ');
    m = strtrim(m);
    if contains(m, 'spm vb')
        m = 'spm vb';
    elseif contains(m, 'spm kmeans')
        m = 'spm kmeans';
    elseif contains(m, 'kmeans koenig') || contains(m, 'koenig kmeans')
        m = 'koenig kmeans';
    elseif contains(m, 'kmeans standard') || contains(m, 'standard kmeans')
        m = 'standard kmeans';
    end
end

function label = display_method_label_local(m_in)
    switch canonicalize_method_local(m_in)
        case 'spm vb'
            label = 'VB-GMM';
        case 'koenig kmeans'
            label = 'K-Means';
        case 'standard kmeans'
            label = 'Standard K-Means';
        otherwise
            label = canonicalize_method_local(m_in);
    end
end
