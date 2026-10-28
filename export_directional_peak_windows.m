function summary = export_directional_peak_windows(results_dir, manifest_csv, output_csv, varargin)
%EXPORT_DIRECTIONAL_PEAK_WINDOWS Cheap, label-aware GFP-peak backfit windows.
%
%   summary = export_directional_peak_windows(hierarchy_dir, manifest_csv, output_csv)
%
% This does not refit microstates. It applies each participant's K=6 solution (from the pre-specified
% global labelling) to the cached GFP-peak maps of the hierarchy run and exports non-overlapping
% 60-second state/transition counts for preictal and genuine interictal records, plus the first
% postictal minute for equal-duration recovery.
%
% Columns n_<label> and t_<from>_<to> use the canonical presentation order ("F","E","G","B","A","C" by
% default; override with 'presentation_order'). Fitted states are mapped to that order through the
% labels derived by the labelling rule (never through a hard-coded fitted-state order). The mapping is
% cross-checked against the global labels and every participant solution.
%
% Every manifest record must have a hierarchy cache whose stored file_path matches the manifest (a
% stale or mismatched cache is an error, not a skipped record). The CSV is written atomically.

    p = inputParser;
    addRequired(p, 'results_dir', @(x) ischar(x) || isstring(x));
    addRequired(p, 'manifest_csv', @(x) ischar(x) || isstring(x));
    addRequired(p, 'output_csv', @(x) ischar(x) || isstring(x));
    addParameter(p, 'presentation_order', ["F", "E", "G", "B", "A", "C"], @(x) isstring(x) || iscellstr(x));
    parse(p, results_dir, manifest_csv, output_csv, varargin{:});
    fsms_paths();
    results_dir = char(string(results_dir));
    output_csv = char(string(output_csv));
    fsms_assert_unrestricted(results_dir, 'reading hierarchy results');
    fsms_assert_unrestricted(output_csv, 'writing the windows CSV');
    labels = string(p.Results.presentation_order(:))';

    fit_labels = fsms_global_fit_labels(results_dir);
    if numel(labels) ~= numel(fit_labels) || ~all(ismember(labels, fit_labels))
        error('FSMS:PresentationOrder', 'Presentation order %s is not a permutation of the fitted labels %s.', ...
            strjoin(labels, ','), strjoin(fit_labels, ','));
    end
    n_states = numel(labels);

    manifest = fsms_read_manifest(manifest_csv);
    for column = ["seizure_type", "hypo_hyper"]
        if ~ismember(column, string(manifest.Properties.VariableNames))
            manifest.(column) = repmat("", height(manifest), 1);
        end
    end
    for column = ["crop_start_seconds", "crop_end_seconds"]
        if ~ismember(column, string(manifest.Properties.VariableNames))
            error('FSMS:ManifestColumns', 'Manifest lacks required column %s for window export.', column);
        end
    end
    condition = lower(strtrim(manifest.condition));
    keep_condition = condition == "ii" | condition == "preict" | condition == "postict";
    manifest = manifest(keep_condition, :);
    manifest.condition_key = condition(keep_condition);

    cache_index = hierarchy_cache_index(results_dir);

    variables = ["participant", "file_path", "condition", "group", "seizure_type", ...
        "hypo_hyper", "window_index", "window_start_s", "window_end_s", ...
        "window_end_to_ictus_s", "n_gfp_peaks", "n_state_changes"];
    for label = labels
        variables(end + 1) = "n_" + label; %#ok<AGROW>
    end
    for from = 1:n_states
        for to = 1:n_states
            variables(end + 1) = sprintf("t_%d_%d", from, to); %#ok<AGROW>
        end
    end

    rows = cell(0, numel(variables));
    participants = unique(manifest.participant, "stable");
    records_done = 0;
    for pidx = 1:numel(participants)
        participant = participants(pidx);
        solution_file = fullfile(results_dir, "participants", fsms_clean_key(participant), "participant_solution.mat");
        if ~isfile(solution_file)
            error('FSMS:ParticipantSolutionMissing', 'Missing participant solution: %s', solution_file);
        end
        solution = load(solution_file, "Fit", "common_labels");
        [centres, canonical_index] = aligned_centres(solution.Fit, labels, fit_labels, participant);
        records = manifest(manifest.participant == participant, :);
        for r = 1:height(records)
            record = records(r, :);
            key = fsms_path_key(record.file_path);
            if ~isKey(cache_index, key)
                error('FSMS:PeakCacheMissing', 'No hierarchy GFP-peak cache for manifest record: %s', record.file_path);
            end
            cached = load(cache_index(key), "rec");
            rec = cached.rec;
            if ~strcmp(fsms_path_key(rec.file_path), key)
                error('FSMS:PeakCacheStale', 'Peak cache %s belongs to %s, not to manifest record %s.', ...
                    cache_index(key), rec.file_path, record.file_path);
            end
            if numel(rec.common_labels) ~= numel(solution.common_labels) || ...
                    any(~strcmpi(string(rec.common_labels(:)), string(solution.common_labels(:))))
                error('FSMS:ChannelMismatch', 'Channel mismatch in %s', cache_index(key));
            end
            maps = double(rec.maps_norm);
            maps = maps ./ max(vecnorm(maps, 2, 2), eps);
            [~, local_state] = max(abs(maps * centres'), [], 2);
            state = canonical_index(local_state);
            peak_time = (double(rec.peak_sample(:)) - 1) / double(rec.sfreq);
            duration = double(record.crop_end_seconds - record.crop_start_seconds);
            if ~isfinite(duration) || duration <= 0
                error('FSMS:BadCropDuration', 'Record %s has a non-positive or missing crop duration.', record.file_path);
            end
            [widx, wstart, wstop, wend] = fsms_window_specs(record.condition_key, duration);
            for w = 1:numel(widx)
                rows(end + 1, :) = make_row(record, widx(w), wstart(w), wstop(w), wend(w), ...
                    peak_time, state, n_states); %#ok<AGROW>
            end
            records_done = records_done + 1;
            if mod(records_done, 50) == 0
                fprintf("Directional peak export: %d records, %d windows\n", records_done, size(rows, 1));
            end
        end
    end

    if isempty(rows)
        output = cell2table(cell(0, numel(variables)), VariableNames=cellstr(variables));
    else
        output = cell2table(rows, VariableNames=cellstr(variables));
        numeric = variables(7:end);
        for name = numeric
            if iscell(output.(name))
                output.(name) = cell2mat(output.(name));
            end
        end
    end
    fsms_atomic_writetable(output, output_csv);
    summary = struct("records", records_done, "windows", height(output), ...
        "participants", numel(unique(output.participant)), "output_csv", output_csv, ...
        "fit_labels", {cellstr(fit_labels(:))}, "presentation_order", {cellstr(labels(:))});
    fprintf("Saved %d non-overlapping peak-backfit windows to %s\n", height(output), output_csv);
end


function [centres, canonical_index] = aligned_centres(Fit, labels, fit_labels, participant)
if ~isfield(Fit, "centers") || size(Fit.centers, 1) ~= numel(labels)
    error('FSMS:NotK6', "Participant solution %s is not K=%d", participant, numel(labels));
end
if ~isfield(Fit, "template_alignment") || ~isfield(Fit.template_alignment, "labels")
    error('FSMS:ParticipantLabels', "Participant solution %s lacks aligned labels", participant);
end
local_labels = string(Fit.template_alignment.labels(:));
if ~isequal(local_labels(:), fit_labels(:))
    error('FSMS:ParticipantLabelOrder', ['Participant %s carries fitted-state labels %s, which differ from ' ...
        'the global labels %s.'], participant, strjoin(local_labels, ','), strjoin(fit_labels, ','));
end
centres = double(Fit.centers);
centres = centres ./ max(vecnorm(centres, 2, 2), eps);
canonical_index = zeros(numel(labels), 1);
for k = 1:numel(labels)
    canonical_index(k) = find(labels == local_labels(k), 1);
end
end


function index = hierarchy_cache_index(results_dir)
% Map path_key -> GFP-peak cache file, from the hierarchy's own normalised manifest (never by scanning).
normalised = fullfile(results_dir, 'normalised_input_manifest.csv');
if ~isfile(normalised)
    error('FSMS:HierarchyManifestMissing', 'Hierarchy manifest not found: %s', normalised);
end
N = readtable(normalised, 'TextType', 'string', 'VariableNamingRule', 'preserve', 'Delimiter', ',');
index = containers.Map('KeyType', 'char', 'ValueType', 'char');
for i = 1:height(N)
    cache_file = fullfile(results_dir, 'gfp_peaks', sprintf('%03d_%s_%s_gfp_peaks.mat', i, ...
        fsms_clean_key(N.participant(i)), fsms_clean_key(N.condition(i))));
    if isfile(cache_file)
        index(fsms_path_key(N.file_path(i))) = cache_file;
    end
end
end


function row = make_row(record, window_index, start_s, stop_s, endpoint_s, peak_time, state, n_states)
keep = peak_time >= start_s & peak_time < stop_s;
sequence = state(keep);
if isempty(sequence)
    collapsed = sequence;
else
    collapsed = sequence([true; diff(sequence) ~= 0]);
end
counts = zeros(n_states, n_states);
for k = 1:max(0, numel(collapsed) - 1)
    counts(collapsed(k), collapsed(k + 1)) = counts(collapsed(k), collapsed(k + 1)) + 1;
end
row = {safe_char(record.participant), safe_char(record.file_path), safe_char(record.condition_key), ...
    safe_char(record.group), safe_char(record.seizure_type), safe_char(record.hypo_hyper), ...
    window_index, start_s, stop_s, endpoint_s, numel(sequence), max(0, numel(collapsed) - 1)};
for k = 1:n_states
    row{end + 1} = sum(sequence == k); %#ok<AGROW>
end
for from = 1:n_states
    for to = 1:n_states
        row{end + 1} = counts(from, to); %#ok<AGROW>
    end
end
assert(numel(row) == 12 + n_states + n_states^2);
end


function value = safe_char(value)
value = string(value);
if ismissing(value)
    value = "";
end
value = char(value);
end
