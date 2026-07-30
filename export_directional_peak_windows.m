function summary = export_directional_peak_windows(results_dir, manifest_csv, output_csv)
%EXPORT_DIRECTIONAL_PEAK_WINDOWS Cheap, label-aware GFP-peak backfit windows.
%
% This does not refit microstates. It applies each participant's manually
% aligned K=6 solution to cached GFP-peak maps and exports non-overlapping
% 60-second state/transition counts for preictal and genuine interictal
% records, plus the first postictal minute for equal-duration recovery.

arguments
    results_dir (1,1) string
    manifest_csv (1,1) string
    output_csv (1,1) string
end

labels = ["F", "E", "G", "B", "A", "C"];
manifest = readtable(manifest_csv, TextType="string");
manifest.manifest_index = (1:height(manifest))';
condition = lower(strtrim(manifest.condition));
keep_condition = condition == "ii" | condition == "preict" | condition == "postict";
manifest = manifest(keep_condition, :);
manifest.condition_key = condition(keep_condition);

variables = ["participant", "file_path", "condition", "group", "seizure_type", ...
    "hypo_hyper", "window_index", "window_start_s", "window_end_s", ...
    "window_end_to_ictus_s", "n_gfp_peaks", "n_state_changes"];
for label = labels
    variables(end + 1) = "n_" + label; %#ok<AGROW>
end
for from = 1:6
    for to = 1:6
        variables(end + 1) = sprintf("t_%d_%d", from, to); %#ok<AGROW>
    end
end

rows = cell(0, numel(variables));
participants = unique(manifest.participant, "stable");
records_done = 0;
for p = 1:numel(participants)
    participant = participants(p);
    solution_file = fullfile(results_dir, "participants", participant, "participant_solution.mat");
    if ~isfile(solution_file)
        warning("Missing participant solution: %s", solution_file);
        continue
    end
    solution = load(solution_file, "Fit", "common_labels");
    [centres, canonical_index] = aligned_centres(solution.Fit, labels);
    records = manifest(manifest.participant == participant, :);
    cache_map = participant_cache_map(results_dir, participant);
    for r = 1:height(records)
        record = records(r, :);
        file_key = lower(replace(record.file_path, "/", "\"));
        if ~isKey(cache_map, char(file_key))
            warning("Missing GFP cache for manifest record: %s", record.file_path);
            continue
        end
        cache_file = cache_map(char(file_key));
        cached = load(cache_file, "rec");
        rec = cached.rec;
        if numel(rec.common_labels) ~= numel(solution.common_labels) || ...
                any(~strcmpi(string(rec.common_labels(:)), string(solution.common_labels(:))))
            error("Channel mismatch in %s", cache_file);
        end
        maps = double(rec.maps_norm);
        maps = maps ./ max(vecnorm(maps, 2, 2), eps);
        [~, local_state] = max(abs(maps * centres'), [], 2);
        state = canonical_index(local_state);
        peak_time = (double(rec.peak_sample(:)) - 1) / double(rec.sfreq);
        duration = double(record.crop_end_seconds - record.crop_start_seconds);
        if record.condition_key == "preict"
            endpoints = 0:60:min(1140, floor((duration - 60) / 60) * 60);
            for w = 1:numel(endpoints)
                endpoint = endpoints(w);
                stop = duration - endpoint;
                start = stop - 60;
                rows(end + 1, :) = make_row(record, w, start, stop, endpoint, ...
                    peak_time, state, labels); %#ok<AGROW>
            end
        elseif record.condition_key == "ii"
            starts = 0:60:(floor(duration / 60) * 60 - 60);
            for w = 1:numel(starts)
                start = starts(w);
                stop = start + 60;
                rows(end + 1, :) = make_row(record, w, start, stop, NaN, ...
                    peak_time, state, labels); %#ok<AGROW>
            end
        elseif duration >= 60
            rows(end + 1, :) = make_row(record, 1, 0, 60, NaN, ...
                peak_time, state, labels); %#ok<AGROW>
        end
        records_done = records_done + 1;
        if mod(records_done, 50) == 0
            fprintf("Directional peak export: %d records, %d windows\n", records_done, size(rows, 1));
        end
    end
end

output = cell2table(rows, VariableNames=cellstr(variables));
numeric = variables(7:end);
for name = numeric
    if iscell(output.(name))
        output.(name) = cell2mat(output.(name));
    end
end
out_dir = fileparts(output_csv);
if strlength(out_dir) > 0 && ~isfolder(out_dir)
    mkdir(out_dir);
end
writetable(output, output_csv);
summary = struct("records", records_done, "windows", height(output), ...
    "participants", numel(unique(output.participant)), "output_csv", output_csv);
fprintf("Saved %d non-overlapping peak-backfit windows to %s\n", height(output), output_csv);
end


function [centres, canonical_index] = aligned_centres(Fit, labels)
if ~isfield(Fit, "centers") || size(Fit.centers, 1) ~= 6
    error("Participant solution is not K=6");
end
if ~isfield(Fit, "template_alignment") || ~isfield(Fit.template_alignment, "labels")
    error("Participant solution lacks manual-label alignment");
end
local_labels = string(Fit.template_alignment.labels(:));
if numel(local_labels) ~= 6 || numel(unique(local_labels)) ~= 6 || ~all(ismember(labels, local_labels))
    error("Participant labels are not a one-to-one F/E/G/B/A/C mapping");
end
centres = double(Fit.centers);
centres = centres ./ max(vecnorm(centres, 2, 2), eps);
canonical_index = zeros(6, 1);
for k = 1:6
    canonical_index(k) = find(labels == local_labels(k), 1);
end
end


function cache_map = participant_cache_map(results_dir, participant)
files = dir(fullfile(results_dir, "gfp_peaks", "*_" + participant + "_*_gfp_peaks.mat"));
cache_map = containers.Map("KeyType", "char", "ValueType", "char");
for k = 1:numel(files)
    cache_file = fullfile(files(k).folder, files(k).name);
    cached = load(cache_file, "rec");
    key = lower(replace(string(cached.rec.file_path), "/", "\"));
    cache_map(char(key)) = cache_file;
end
end


function row = make_row(record, window_index, start_s, stop_s, endpoint_s, peak_time, state, labels)
keep = peak_time >= start_s & peak_time < stop_s;
sequence = state(keep);
if isempty(sequence)
    collapsed = sequence;
else
    collapsed = sequence([true; diff(sequence) ~= 0]);
end
counts = zeros(6, 6);
for k = 1:max(0, numel(collapsed) - 1)
    counts(collapsed(k), collapsed(k + 1)) = counts(collapsed(k), collapsed(k + 1)) + 1;
end
row = {safe_char(record.participant), safe_char(record.file_path), safe_char(record.condition_key), ...
    safe_char(record.group), safe_char(record.seizure_type), safe_char(record.hypo_hyper), ...
    window_index, start_s, stop_s, endpoint_s, numel(sequence), max(0, numel(collapsed) - 1)};
for k = 1:6
    row{end + 1} = sum(sequence == k); %#ok<AGROW>
end
for from = 1:6
    for to = 1:6
        row{end + 1} = counts(from, to); %#ok<AGROW>
    end
end
assert(numel(row) == 12 + numel(labels) + 36);
end


function value = safe_char(value)
value = string(value);
if ismissing(value)
    value = "";
end
value = char(value);
end
