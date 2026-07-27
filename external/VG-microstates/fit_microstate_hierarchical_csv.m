function [HResults, results_mat] = fit_microstate_hierarchical_csv(manifest_csv, varargin)
%FIT_MICROSTATE_HIERARCHICAL_CSV General EEG CSV entrypoint.
%
% The CSV must contain participant, condition, and file path columns.
% group is optional and defaults to all. condition may be named Dreaming.
%
% Example:
%   H = fit_microstate_hierarchical_csv('my_eeg_manifest.csv');

    if nargin < 1 || isempty(manifest_csv)
        error('Provide a CSV manifest with participant, condition, and file path columns.');
    end

    assert_general_manifest(manifest_csv);
    if ~has_name_value(varargin, 'infer_localizer_dir')
        varargin = [{'infer_localizer_dir', false}, varargin];
    end
    [HResults, results_mat] = fit_microstate_hierarchical_dataset(manifest_csv, varargin{:});
end

function assert_general_manifest(manifest_csv)
    manifest_csv = char(string(manifest_csv));
    if ~isfile(manifest_csv)
        error('Manifest not found: %s', manifest_csv);
    end

    opts = detectImportOptions(manifest_csv, 'FileType', 'text', 'TextType', 'string', ...
        'Delimiter', manifest_delimiter(manifest_csv));
    opts.VariableNamingRule = 'preserve';
    T = readtable(manifest_csv, opts);
    names = lower(regexprep(string(T.Properties.VariableNames), '[^a-zA-Z0-9]', ''));
    has_participant = any(ismember(names, ["participant", "subject", "sub", "id", "subjectid", "participantid"]));
    has_condition = any(ismember(names, ["condition", "dreaming", "state", "task", "eyes", "session"]));
    has_file = any(ismember(names, ["filepath", "file", "path", "filename"]));
    missing = strings(0, 1);
    if ~has_participant, missing(end+1) = "participant"; end
    if ~has_condition, missing(end+1) = "condition"; end
    if ~has_file, missing(end+1) = "file path"; end
    if ~isempty(missing)
        error('Manifest is missing required column(s): %s', strjoin(missing, ', '));
    end
end

function delimiter = manifest_delimiter(input_path)
    [~, ~, ext] = fileparts(char(input_path));
    if strcmpi(ext, '.tsv')
        delimiter = '\t';
    else
        delimiter = ',';
    end
end

function tf = has_name_value(args, name)
    tf = false;
    for i = 1:2:numel(args)
        if ischar(args{i}) || (isstring(args{i}) && isscalar(args{i}))
            if strcmpi(char(args{i}), name)
                tf = true;
                return;
            end
        end
    end
end
