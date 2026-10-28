function [T, info] = fsms_read_manifest(manifest_csv, varargin)
%FSMS_READ_MANIFEST Read and validate a study manifest; resolve EEG paths exactly as the hierarchy does.
%
%   [T, info] = fsms_read_manifest(manifest_csv)
%
%   Required columns: participant, condition, group, file_path. Extra columns are preserved.
%   Relative file_path values are resolved against the manifest folder (with the same basename
%   fallback the vendored hierarchy uses). The table is returned sorted by file_path (the order in
%   which the vendored hierarchy numbers its caches) with a manifest_row column holding the original
%   1-based row. Errors (coded) on: missing columns, empty/duplicate file paths, missing EEG files
%   (unless 'check_files' is false), or participant/condition values that collide after the
%   file-name cleaning applied to cache and solution folders.
    p = inputParser;
    addParameter(p, 'check_files', true, @islogical);
    parse(p, varargin{:});
    manifest_csv = char(string(manifest_csv));
    fsms_assert_unrestricted(manifest_csv, 'reading the manifest');
    if ~isfile(manifest_csv)
        error('FSMS:ManifestMissing', 'Manifest not found: %s', manifest_csv);
    end
    T = readtable(manifest_csv, 'TextType', 'string', 'VariableNamingRule', 'preserve', 'Delimiter', ',');
    required = ["participant", "condition", "group", "file_path"];
    missing = required(~ismember(required, string(T.Properties.VariableNames)));
    if ~isempty(missing)
        error('FSMS:ManifestColumns', 'Manifest %s lacks required column(s): %s', manifest_csv, strjoin(missing, ', '));
    end
    if height(T) < 1
        error('FSMS:ManifestEmpty', 'Manifest %s has no rows.', manifest_csv);
    end
    for name = required
        T.(name) = string(T.(name));
    end
    T.participant = strtrim(T.participant);
    T.condition = strtrim(T.condition);
    T.group = strtrim(T.group);
    T.group(ismissing(T.group) | strlength(T.group) == 0) = "all";
    if any(ismissing(T.participant) | strlength(T.participant) == 0)
        error('FSMS:ManifestParticipant', 'Manifest %s has rows with an empty participant.', manifest_csv);
    end
    if any(ismissing(T.condition) | strlength(T.condition) == 0)
        error('FSMS:ManifestCondition', 'Manifest %s has rows with an empty condition.', manifest_csv);
    end
    if any(ismissing(T.file_path) | strlength(strtrim(T.file_path)) == 0)
        error('FSMS:ManifestFilePath', 'Manifest %s has rows with an empty file_path.', manifest_csv);
    end
    if ~ismember('manifest_row', T.Properties.VariableNames)
        T.manifest_row = (1:height(T))';
    end
    base_dir = fileparts(manifest_csv);
    if isempty(base_dir), base_dir = pwd; end
    for i = 1:height(T)
        value = char(strtrim(T.file_path(i)));
        if ~local_is_absolute(value)
            value = fullfile(base_dir, value);
        end
        fsms_assert_unrestricted(value, 'reading an EEG file');
        if ~isfile(value)
            local_candidate = fullfile(base_dir, local_basename(value));
            if isfile(local_candidate), value = local_candidate; end
        end
        T.file_path(i) = string(value);
    end
    keys = arrayfun(@(i) string(fsms_path_key(T.file_path(i))), (1:height(T))');
    [~, first] = unique(keys, 'stable');
    if numel(first) ~= height(T)
        dup = keys(setdiff(1:height(T), first));
        error('FSMS:ManifestDuplicatePath', 'Duplicate file_path entries in %s, e.g. %s', manifest_csv, dup(1));
    end
    if p.Results.check_files
        absent = find(~arrayfun(@(i) isfile(T.file_path(i)), (1:height(T))'));
        if ~isempty(absent)
            error('FSMS:MissingEEGFile', '%d manifest EEG file(s) not found, first: %s', numel(absent), T.file_path(absent(1)));
        end
    end
    local_assert_clean_unique(T.participant, 'participant');
    local_assert_clean_unique(T.condition, 'condition');
    [~, order] = sort(T.file_path);
    T = T(order, :);
    info = struct('manifest_csv', manifest_csv, 'manifest_sha256', fsms_sha256_file(manifest_csv), ...
        'n_rows', height(T), 'n_participants', numel(unique(T.participant)));
end

function local_assert_clean_unique(values, label)
    unique_values = unique(values);
    cleaned = arrayfun(@(i) string(fsms_clean_key(unique_values(i))), (1:numel(unique_values))');
    [~, first] = unique(cleaned, 'stable');
    if numel(first) ~= numel(unique_values)
        clash = unique_values(setdiff(1:numel(unique_values), first));
        error('FSMS:ManifestKeyCollision', ['Two different %s values map to the same cache folder name ' ...
            'after cleaning (e.g. %s). Rename them in the manifest.'], label, clash(1));
    end
end

function tf = local_is_absolute(value)
    tf = startsWith(value, filesep) || ~isempty(regexp(value, '^[A-Za-z]:[\\/]', 'once'));
end

function name = local_basename(value)
    parts = regexp(char(value), '[\\/]', 'split');
    name = parts{end};
end
