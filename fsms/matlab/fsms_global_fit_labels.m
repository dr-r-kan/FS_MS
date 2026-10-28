function fit_labels = fsms_global_fit_labels(hierarchy_dir)
%FSMS_GLOBAL_FIT_LABELS Fitted-state labels (state 1..K order) of a hierarchy run, cross-checked.
%   Reads Fit.template_alignment.labels from global/global_solution.mat and, when present, checks them
%   against global/global_manual_labels.csv (the file the pre-specified rule writes and the Python side
%   reads). Errors on any disagreement or on labels that are not six unique A-G letters.
    hierarchy_dir = char(string(hierarchy_dir));
    solution_file = fullfile(hierarchy_dir, 'global', 'global_solution.mat');
    if ~isfile(solution_file)
        error('FSMS:GlobalSolutionMissing', 'Global solution not found: %s', solution_file);
    end
    S = load(solution_file, 'Fit');
    if ~isfield(S.Fit, 'template_alignment') || ~isfield(S.Fit.template_alignment, 'labels')
        error('FSMS:GlobalLabelsMissing', ['The global solution in %s carries no fitted-state labels; ' ...
            'run the labelling step and the second hierarchy pass first.'], hierarchy_dir);
    end
    fit_labels = string(S.Fit.template_alignment.labels(:));
    if numel(fit_labels) ~= 6 || numel(unique(fit_labels)) ~= 6 || any(~ismember(fit_labels, ["A","B","C","D","E","F","G"]))
        error('FSMS:GlobalLabelsInvalid', 'Global labels must be six unique A-G letters, got: %s', strjoin(fit_labels, ','));
    end
    csv_file = fullfile(hierarchy_dir, 'global', 'global_manual_labels.csv');
    if isfile(csv_file)
        T = readtable(csv_file, 'TextType', 'string', 'Delimiter', ',');
        T = sortrows(T, 'state_index');
        if ~isequal(string(T.label(:)), fit_labels(:))
            error('FSMS:GlobalLabelsDisagree', ['Fitted-state labels in global_solution.mat (%s) disagree with ' ...
                'global_manual_labels.csv (%s).'], strjoin(fit_labels, ','), strjoin(T.label, ','));
        end
    end
end
