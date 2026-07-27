function output_csv = summarize_first_line_spm_vb_metrics(results_source, varargin)
%SUMMARIZE_FIRST_LINE_SPM_VB_METRICS Summarise raw single-metric K selectors.
%
% Produces a compact CSV focused on the single component criteria that are
% meant to be analysed on the training simulation batch before any combined
% model is trained.

    p = inputParser;
    addRequired(p, 'results_source');
    addParameter(p, 'method', 'spm_vb', @(x) ischar(x) || isstring(x));
    addParameter(p, 'criteria', {'silhouette', 'free_energy', 'log_likelihood', 'bic', 'icl', 'free_energy_elbow', 'covariance', 'calinski_harabasz_score'}, ...
        @(x) iscell(x) || isstring(x));
    addParameter(p, 'output_csv', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'verbose', true, @islogical);
    parse(p, results_source, varargin{:});
    cfg = p.Results;

    [results_csv, default_output_csv] = resolve_results_paths_local(results_source);
    output_csv = char(string(cfg.output_csv));
    if isempty(output_csv)
        output_csv = default_output_csv;
    end

    if ~isfile(results_csv)
        error('Comparison results CSV not found: %s', results_csv);
    end

    T = readtable(results_csv, 'TextType', 'string');
    if ~all(ismember({'method', 'criterion'}, T.Properties.VariableNames))
        error('Expected method and criterion columns in %s', results_csv);
    end

    T.method_clean = lower(strtrim(string(T.method)));
    T.criterion_clean = strings(height(T), 1);
    for i = 1:height(T)
        T.criterion_clean(i) = canonicalize_criterion_local(T.criterion(i));
    end

    wanted_method = lower(strtrim(string(cfg.method)));
    wanted_criteria = cellfun(@canonicalize_criterion_local, cellstr(string(cfg.criteria)), 'UniformOutput', false);
    mask = (T.method_clean == wanted_method) & ismember(cellstr(T.criterion_clean), wanted_criteria);
    Tsel = T(mask, :);
    if height(Tsel) == 0
        warning('No rows matched method=%s and requested criteria in %s', wanted_method, results_csv);
        output_csv = '';
        return;
    end

    metric_vars = intersect({'K_correct', 'K_error', 'f1_score', 'sensitivity', 'precision', 'runtime', 'best_criterion_value'}, ...
        Tsel.Properties.VariableNames, 'stable');
    Tsummary = groupsummary(Tsel, 'criterion_clean', {'mean', 'std'}, metric_vars);
    Tsummary.Properties.VariableNames{1} = 'criterion';
    if ismember('mean_K_correct', Tsummary.Properties.VariableNames)
        Tsummary.accuracy_pct = 100 * Tsummary.mean_K_correct;
    end
    if ismember('mean_K_correct', Tsummary.Properties.VariableNames)
        sort_vars = {'mean_K_correct'};
        sort_dirs = {'descend'};
        if ismember('mean_K_error', Tsummary.Properties.VariableNames)
            sort_vars{end + 1} = 'mean_K_error'; %#ok<AGROW>
            sort_dirs{end + 1} = 'ascend'; %#ok<AGROW>
        end
        Tsummary = sortrows(Tsummary, sort_vars, sort_dirs);
    end

    writetable(Tsummary, output_csv);
    if cfg.verbose
        fprintf('Saved first-line single-metric summary: %s\n', output_csv);
    end
end

function [results_csv, output_csv] = resolve_results_paths_local(results_source)
    source = char(string(results_source));
    if isfolder(source)
        results_csv = fullfile(source, 'comparison_results.csv');
        output_csv = fullfile(source, 'first_line_spm_vb_metric_summary.csv');
    else
        results_csv = source;
        [folder, ~, ~] = fileparts(results_csv);
        output_csv = fullfile(folder, 'first_line_spm_vb_metric_summary.csv');
    end
end

function s = canonicalize_criterion_local(value_in)
    s = lower(strtrim(char(string(value_in))));
    s = strrep(s, '_', ' ');
    s = regexprep(s, '\s+', ' ');
    if strcmp(s, 'elbow')
        s = 'free energy elbow';
    elseif any(strcmp(s, {'gfp', 'global field power', 'global explained variance'}))
        s = 'gev';
    elseif any(strcmp(s, {'covariance raw', 'covariance min'}))
        s = 'covariance';
    elseif any(strcmp(s, {'calinski harabasz', 'ch'}))
        s = 'calinski harabasz score';
    end
end
