function audit = recover_timestamped_transition_backfits(manifest_csv, hierarchy_dir, output_dir, varargin)
%RECOVER_TIMESTAMPED_TRANSITION_BACKFITS Export participant-aligned exact transitions.
%
%   audit = recover_timestamped_transition_backfits(manifest_csv, hierarchy_dir, output_dir, ...)
%
% Reuses the K=6 hierarchy of the SAME run (hierarchy_dir: global/, participants/, gfp_peaks/) and
% backfits every full signal with the participant's solution, storing every genuine state-change sample
% in transition_cache/*.mat, plus transition_index.csv and recovery_audit.json in output_dir.
%
% The fitted-state label order is DERIVED from the hierarchy (global labelling by the pre-specified rule,
% cross-checked against every participant solution); it is never assumed. State indices stored in the
% caches are fitted-state indices; recovery_audit.json carries fit_labels to map them to classes.
%
% Name-value options
%   conditions               {'II','preict'}  conditions exported
%   validation_windows_csv   ''  CSV freshly written by export_directional_peak_windows for THIS run
%   validation_records       'all' | 'first_per_participant'   records re-extracted for the window gate
%   full_backfit_validation  'first_per_participant' | 'first' | 'all'   records whose compact export is
%                            compared with the full hard backfit
%   max_records              Inf   (debugging only; recorded in the audit)
%   overwrite                false (recompute every cache)
%   provenance               struct()  extra provenance merged into the audit (stage parameters etc.)
%
% THE TWO GATES (both computed here, never asserted)
%  compact_export_validated_against_full_backfit
%     The compact transition list (GFP-peak assignments joined at midpoint samples) is compared sample
%     for sample with the full hard backfit (backfit_microstate_timecourse) for the selected records.
%  focused_window_fit_recovery_validation.transition_matrix_exact_match
%     For the selected records the GFP peaks are RE-EXTRACTED FROM THE RAW EEG FILE (independent of the
%     hierarchy peak cache), assigned to the participant's fitted states, mapped to the presentation
%     order through the labels derived from the global labelling (independent of the label mapping the
%     export used), and the per-window peak/state/transition counts are recomputed by a separate counting
%     loop and compared with EVERY window of those records in the freshly exported CSV. It also requires
%     the re-extracted peaks to equal the hierarchy cache. It therefore proves (i) the hierarchy cache is
%     the deterministic product of the manifest file and the frozen preprocessing (no stale or mismatched
%     cache), and (ii) the label mapping used by the window export equals the one used here. It does not
%     validate the labelling itself (see the labelling audit) or the VB fit.

    p = inputParser;
    addRequired(p, 'manifest_csv', @(x) ischar(x) || isstring(x));
    addRequired(p, 'hierarchy_dir', @(x) ischar(x) || isstring(x));
    addRequired(p, 'output_dir', @(x) ischar(x) || isstring(x));
    addParameter(p, 'conditions', {'II', 'preict'}, @(x) iscell(x) || isstring(x));
    addParameter(p, 'validation_windows_csv', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'validation_records', 'all', @(x) any(strcmp(char(x), {'all', 'first_per_participant'})));
    addParameter(p, 'full_backfit_validation', 'first_per_participant', ...
        @(x) any(strcmp(char(x), {'first_per_participant', 'first', 'all'})));
    addParameter(p, 'max_records', Inf, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'overwrite', false, @islogical);
    addParameter(p, 'provenance', struct(), @isstruct);
    parse(p, manifest_csv, hierarchy_dir, output_dir, varargin{:});
    cfg_run = p.Results;

    env = fsms_paths();
    util = microstate_utilities();
    hierarchy_dir = char(string(hierarchy_dir));
    output_dir = char(string(output_dir));
    fsms_assert_unrestricted(hierarchy_dir, 'reading hierarchy results');
    fsms_assert_unrestricted(output_dir, 'writing transition caches');
    if ~isempty(char(string(cfg_run.validation_windows_csv)))
        fsms_assert_unrestricted(cfg_run.validation_windows_csv, 'reading the windows CSV');
    end
    cache_dir = fullfile(output_dir, 'transition_cache');
    ensure_dir(output_dir); ensure_dir(cache_dir);

    solution_file = fullfile(hierarchy_dir, 'global', 'global_solution.mat');
    must_exist(manifest_csv); must_exist(solution_file);
    must_exist(fullfile(hierarchy_dir, 'normalised_input_manifest.csv'));
    S = load(solution_file, 'Fit', 'cfg', 'common_labels', 'common_chanlocs');
    cfg = S.cfg;
    common_labels = S.common_labels;
    common_can = util.canonical_channel_labels(common_labels);
    common_pos = util.positions_from_chanlocs(S.common_chanlocs, numel(common_labels));
    fit_labels = fsms_global_fit_labels(hierarchy_dir);
    labelling = read_labelling_audit(hierarchy_dir, fit_labels);

    [paper, manifest_info] = fsms_read_manifest(manifest_csv);
    hierarchy_manifest = readtable(fullfile(hierarchy_dir, 'normalised_input_manifest.csv'), ...
        'TextType', 'string', 'VariableNamingRule', 'preserve', 'Delimiter', ',');
    paper_keys = arrayfun(@(i) string(fsms_path_key(paper.file_path(i))), (1:height(paper))');
    hier_keys = arrayfun(@(i) string(fsms_path_key(hierarchy_manifest.file_path(i))), (1:height(hierarchy_manifest))');
    if ~isequal(paper_keys, hier_keys)
        error('FSMS:ManifestHierarchyMismatch', ['The manifest %s does not list exactly the records (in ' ...
            'file_path order) of the hierarchy run in %s. Pass the analysis manifest that run used.'], ...
            manifest_csv, hierarchy_dir);
    end

    % Hierarchy GFP-peak caches: one per manifest row, verified against the manifest.
    peak_files = strings(height(paper), 1);
    peak_counts = zeros(height(paper), 1);
    for i = 1:height(paper)
        peak_files(i) = string(fullfile(hierarchy_dir, 'gfp_peaks', sprintf('%03d_%s_%s_gfp_peaks.mat', i, ...
            fsms_clean_key(paper.participant(i)), fsms_clean_key(paper.condition(i)))));
        must_exist(peak_files(i));
        C = load(peak_files(i), 'rec');
        if ~strcmp(fsms_path_key(C.rec.file_path), paper_keys(i))
            error('FSMS:PeakCacheStale', 'Peak cache %s belongs to %s, not to manifest record %s.', ...
                peak_files(i), C.rec.file_path, paper.file_path(i));
        end
        peak_counts(i) = size(C.rec.maps_norm, 1);
    end
    total_peaks = sum(peak_counts);
    Fit = S.Fit;

    % Participant solutions: one per participant, labels must equal the derived global labels.
    participants = unique(paper.participant, 'stable');
    participant_fit = containers.Map('KeyType', 'char', 'ValueType', 'any');
    for k = 1:numel(participants)
        participant_file = fullfile(hierarchy_dir, 'participants', fsms_clean_key(participants(k)), 'participant_solution.mat');
        must_exist(participant_file);
        participant_solution = load(participant_file, 'Fit');
        pfit = participant_solution.Fit;
        if ~isfield(pfit, 'template_alignment') || ~isfield(pfit.template_alignment, 'labels') || ...
                ~isequal(string(pfit.template_alignment.labels(:)), fit_labels(:))
            error('FSMS:ParticipantLabelOrder', ['Participant %s does not retain the derived fitted-state label ' ...
                'order (%s).'], participants(k), strjoin(fit_labels, ','));
        end
        participant_fit(char(participants(k))) = pfit;
    end

    wanted = ismember(lower(strtrim(paper.condition)), lower(string(cfg_run.conditions(:))));
    selected = find(wanted);
    selected = selected(1:min(numel(selected), floor(cfg_run.max_records)));
    if isempty(selected)
        error('FSMS:NoRecordsSelected', 'No manifest record has a condition in {%s}.', strjoin(string(cfg_run.conditions), ','));
    end

    % Gate 2: window export vs independent re-extraction (before the expensive transition export).
    validation = struct('performed', false, 'transition_matrix_exact_match', false);
    validation_timer = tic;
    validation_csv = char(string(cfg_run.validation_windows_csv));
    if ~isempty(validation_csv)
        must_exist(validation_csv);
        validation = validate_window_export(validation_csv, paper, paper_keys, peak_files, cfg, ...
            common_can, common_pos, fit_labels, participant_fit, char(cfg_run.validation_records));
    end

    if ~isempty(char(string(cfg_run.validation_windows_csv)))
        fprintf('[timing] window-export validation (%d records re-extracted): %.1f s\n', ...
            validation_records_checked(validation), toc(validation_timer));
    end

    % Transition export.
    must_validate = false(height(paper), 1);
    switch char(cfg_run.full_backfit_validation)
        case 'all'
            must_validate(selected) = true;
        case 'first'
            must_validate(selected(1)) = true;
        otherwise
            seen = containers.Map('KeyType', 'char', 'ValueType', 'logical');
            for j = 1:numel(selected)
                key = char(paper.participant(selected(j)));
                if ~isKey(seen, key)
                    seen(key) = true;
                    must_validate(selected(j)) = true;
                end
            end
    end
    preproc_sha = preprocessing_sha(cfg, common_can);
    index_parts = cell(numel(selected), 1);
    n_validated = 0;
    n_rebuilt = 0;
    for j = 1:numel(selected)
        i = selected(j);
        record_fit = participant_fit(char(paper.participant(i)));
        cache_file = fullfile(cache_dir, sprintf('%04d_%s_%s_transitions.mat', i, ...
            fsms_clean_key(paper.participant(i)), fsms_clean_key(paper.condition(i))));
        transition = [];
        if ~cfg_run.overwrite && isfile(cache_file)
            transition = load_valid_cache(cache_file, paper.file_path(i), record_fit, fit_labels, ...
                preproc_sha, must_validate(i));
        end
        if isempty(transition)
            request = struct('command', 'timestamped_transitions', ...
                'file_path', char(paper.file_path(i)), 'common_can', {common_can}, ...
                'common_pos', common_pos, 'cfg', cfg, 'fit', record_fit, ...
                'validate_full_backfit', must_validate(i));
            [transition, ~] = fit_microstate_hierarchical_dataset(request);
            transition.fit_centers = single(record_fit.centers);
            transition.fit_labels = cellstr(fit_labels(:));
            transition.preproc_sha256 = preproc_sha;
            transition.manifest_index = double(i);
            fsms_atomic_save(cache_file, struct('transition', transition));
            n_rebuilt = n_rebuilt + 1;
        end
        n_validated = n_validated + double(logical(transition.full_backfit_validation));
        index_parts{j} = table(double(i), paper.participant(i), paper.condition(i), paper.group(i), ...
            paper.file_path(i), string(cache_file), double(transition.sfreq), double(transition.n_samples), ...
            double(numel(transition.peak_sample)), double(numel(transition.transition_sample)), ...
            'VariableNames', {'manifest_index','participant','condition','group','file_path','cache_file', ...
            'sfreq','n_samples','n_gfp_peaks','n_state_changes'});
        if mod(j, 10) == 0 || j == numel(selected)
            fprintf('[%d/%d] exact transition records exported (%d rebuilt)\n', j, numel(selected), n_rebuilt);
        end
    end
    index = vertcat(index_parts{:});
    expected_validated = sum(must_validate(selected));
    full_validation = n_validated >= expected_validated && expected_validated > 0;
    if ~full_validation
        error('FSMS:FullBackfitValidationIncomplete', ...
            'Only %d of %d records requested for full-backfit validation were validated.', n_validated, expected_validated);
    end
    fsms_atomic_writetable(index, fullfile(output_dir, 'transition_index.csv'));

    fingerprint = fsms_script_fingerprint(env.repo_root);
    audit = struct();
    audit.schema_version = 2;
    audit.stage = 'C_vg_microstates_transition_backfit';
    % manifest_* identify the STUDY manifest (supplied by run_study_hierarchy via provenance); the manifest this
    % function consumed (the retained-records analysis manifest) is recorded separately.
    audit.manifest_csv = char(manifest_csv);
    audit.manifest_sha256 = manifest_info.manifest_sha256;
    if isfield(cfg_run.provenance, 'study_manifest_sha256')
        audit.manifest_csv = cfg_run.provenance.study_manifest_csv;
        audit.manifest_sha256 = cfg_run.provenance.study_manifest_sha256;
    end
    audit.analysis_manifest_csv = char(manifest_csv);
    audit.analysis_manifest_sha256 = manifest_info.manifest_sha256;
    audit.hierarchy_dir = hierarchy_dir;
    audit.legacy_results_dir = hierarchy_dir;
    audit.output_dir = output_dir;
    audit.n_manifest_records = height(paper);
    audit.n_recovered_peak_records = double(validation_records_checked(validation));
    audit.n_global_peaks = total_peaks;
    audit.n_global_fit_peaks = Fit.n_maps;
    audit.fit_labels = cellstr(fit_labels(:));
    audit.manual_labels_by_fitted_state = cellstr(fit_labels(:));
    audit.labelling_rule = labelling.rule_id;
    audit.labelling_correlations = labelling.correlations;
    audit.labelling_quality = labelling.quality;
    audit.labelling_states_below_threshold = labelling.states_below;
    audit.labelling_semantics = labelling.semantics;
    audit.labelling_audit_sha256 = labelling.audit_sha256;
    audit.n_timestamped_records = height(index);
    audit.n_transitions = sum(index.n_state_changes);
    audit.n_full_backfit_validated_records = n_validated;
    audit.full_backfit_validation_scope = char(cfg_run.full_backfit_validation);
    audit.compact_export_validated_against_full_backfit = logical(full_validation);
    audit.focused_window_fit_recovery_validation = validation;
    audit.transition_time_resolution = 'one EEG sample; 1/sfreq seconds';
    audit.transition_ordering = 'exact hard backfit ordering; GFP-peak labels joined at midpoint sample boundaries';
    audit.backfit_fit_level = 'participant-specific K=6 solution aligned to the global fit; labels from the pre-specified rule';
    audit.state_index_convention = 'transition caches store 1-based fitted-state indices; fit_labels[i-1] is the class of state i';
    audit.conditions = cellstr(string(cfg_run.conditions(:)));
    audit.max_records = cfg_run.max_records;
    audit.matlab_version = env.matlab_version;
    audit.matlab_release = env.matlab_release;
    audit.spm = env.spm;
    audit.script_sha256 = fingerprint.files;
    audit.script_fingerprint = fingerprint.combined;
    audit.created_utc = char(datetime('now', 'TimeZone', 'UTC', 'Format', 'yyyy-MM-dd''T''HH:mm:ssXXX'));
    if ~isempty(fieldnames(cfg_run.provenance))
        audit.provenance = cfg_run.provenance;
    end
    fsms_write_json(fullfile(output_dir, 'recovery_audit.json'), audit);
end

function must_exist(path_value)
    if ~isfile(path_value), error('FSMS:RequiredFileMissing', 'Required file not found: %s', path_value); end
end

function ensure_dir(path_value)
    if ~isfolder(path_value), mkdir(path_value); end
end

function n = validation_records_checked(validation)
    n = 0;
    if isfield(validation, 'n_records_reextracted'), n = validation.n_records_reextracted; end
end

function sha = preprocessing_sha(cfg, common_can)
    keys = {'apply_average_reference', 'spatial_filter', 'spatial_filter_neighbours', ...
        'spatial_filter_strength', 'filter_band', 'gfp_peak_min_distance', 'use_scalp_channels', ...
        'scalp_channel_max_radius', 'exclude_channels', 'interpolate_missing_channels'};
    payload = struct();
    for i = 1:numel(keys)
        if isfield(cfg, keys{i}), payload.(keys{i}) = cfg.(keys{i}); end
    end
    payload.common_can = common_can(:)';
    sha = fsms_sha256_text(jsonencode(payload));
end

function transition = load_valid_cache(cache_file, file_path, record_fit, fit_labels, preproc_sha, need_validation)
% Return the cached transition struct if it is complete and matches this record, fit and settings;
% otherwise [] so the caller rebuilds (heals) it.
    transition = [];
    try
        T = load(cache_file, 'transition');
    catch
        return;
    end
    if ~isfield(T, 'transition'), return; end
    t = T.transition;
    required = {'fit_centers', 'full_backfit_validation', 'file_path', 'fit_labels', 'preproc_sha256', ...
        'sfreq', 'n_samples', 'peak_sample', 'transition_sample', 'from_state', 'to_state'};
    if ~all(isfield(t, required)), return; end
    if ~strcmp(fsms_path_key(t.file_path), fsms_path_key(file_path)), return; end
    if ~isequal(string(t.fit_labels(:)), fit_labels(:)), return; end
    if ~strcmp(char(t.preproc_sha256), preproc_sha), return; end
    if ~isequal(size(t.fit_centers), size(record_fit.centers)), return; end
    if max(abs(double(t.fit_centers(:)) - double(record_fit.centers(:)))) >= 1e-7, return; end
    if need_validation && ~logical(t.full_backfit_validation), return; end
    transition = t;
end

function labelling = read_labelling_audit(hierarchy_dir, fit_labels)
    audit_file = fullfile(hierarchy_dir, 'global', 'labelling_audit.json');
    must_exist(audit_file);
    A = fsms_read_json(audit_file);
    if ~isequal(string(A.labels_by_fitted_state(:)), fit_labels(:))
        error('FSMS:LabellingAuditDisagrees', 'labelling_audit.json labels (%s) differ from the global fit labels (%s).', ...
            strjoin(string(A.labels_by_fitted_state), ','), strjoin(fit_labels, ','));
    end
    if ~islogical(A.accepted) || ~A.accepted
        error('FSMS:LabellingNotAccepted', 'labelling_audit.json does not record a technically sound labelling.');
    end
    labelling = struct('rule_id', A.rule_id, 'audit_sha256', fsms_sha256_file(audit_file));
    labelling.quality = A.labelling_quality;
    labelling.states_below = num2cell(A.states_below_threshold(:))';
    labelling.semantics = A.nominal_label_semantics;
    labelling.correlations = struct( ...
        'matrix_abs', A.correlation_matrix_abs, ...
        'columns', {A.correlation_matrix_abs_columns}, ...
        'assigned_abs_r', A.assigned_abs_r, ...
        'next_best_abs_r', A.next_best_abs_r, ...
        'strong_threshold', A.strong_threshold, ...
        'channel_match_mode', A.channel_match_mode, ...
        'template_sha256', A.template_sha256);
end

function validation = validate_window_export(validation_csv, paper, paper_keys, peak_files, cfg, ...
        common_can, common_pos, fit_labels, participant_fit, scope)
% Gate 2: recompute the exported windows from independently re-extracted GFP peaks.
    expected = readtable(validation_csv, 'TextType', 'string', 'VariableNamingRule', 'preserve', 'Delimiter', ',');
    if height(expected) < 1
        error('FSMS:ValidationWindowsEmpty', 'Validation windows CSV %s has no windows.', validation_csv);
    end
    names = string(expected.Properties.VariableNames);
    count_columns = names(startsWith(names, "n_") & strlength(names) == 3);
    presentation = extractAfter(count_columns, "n_");
    n_states = numel(fit_labels);
    if numel(presentation) ~= n_states || ~all(ismember(presentation, fit_labels))
        error('FSMS:ValidationWindowsColumns', 'Validation CSV state-count columns (%s) do not match the fitted labels (%s).', ...
            strjoin(count_columns, ','), strjoin(fit_labels, ','));
    end
    fitted_to_presentation = zeros(n_states, 1);
    for k = 1:n_states
        fitted_to_presentation(k) = find(presentation == fit_labels(k), 1);
    end
    csv_keys = arrayfun(@(i) string(fsms_path_key(expected.file_path(i))), (1:height(expected))');
    record_keys = unique(csv_keys, 'stable');
    record_rows = zeros(numel(record_keys), 1);
    for r = 1:numel(record_keys)
        hit = find(paper_keys == record_keys(r), 1);
        if isempty(hit)
            error('FSMS:ValidationRecordAbsent', 'Validation record %s is absent from the manifest.', record_keys(r));
        end
        record_rows(r) = hit;
    end
    [record_rows, order] = sort(record_rows);
    record_keys = record_keys(order);
    if strcmp(scope, 'first_per_participant')
        keep = false(numel(record_rows), 1);
        seen = containers.Map('KeyType', 'char', 'ValueType', 'logical');
        for r = 1:numel(record_rows)
            participant = char(paper.participant(record_rows(r)));
            if ~isKey(seen, participant)
                seen(participant) = true;
                keep(r) = true;
            end
        end
        record_rows = record_rows(keep);
        record_keys = record_keys(keep);
    end

    n_windows = 0;
    mismatches = strings(0, 1);
    max_map_difference = 0;
    cache_reproduced = true;
    for r = 1:numel(record_rows)
        i = record_rows(r);
        request = struct('command', 'extract_peak_record', 'file_path', char(paper.file_path(i)), ...
            'common_can', {common_can}, 'common_pos', common_pos, 'cfg', cfg);
        [rec2, ~] = fit_microstate_hierarchical_dataset(request);
        cached = load(peak_files(i), 'rec');
        same_samples = isequal(double(rec2.peak_sample(:)), double(cached.rec.peak_sample(:)));
        if same_samples
            difference = max(abs(single(rec2.maps_norm(:)) - single(cached.rec.maps_norm(:))));
            max_map_difference = max(max_map_difference, double(difference));
        end
        if ~same_samples || max_map_difference > 1e-6
            cache_reproduced = false;
            error('FSMS:PeakCacheNotReproducible', ['GFP peaks re-extracted from %s differ from the hierarchy ' ...
                'cache %s (identical peak samples: %d, max map difference %.3g). The cache is stale or the ' ...
                'extraction is not deterministic.'], paper.file_path(i), peak_files(i), same_samples, max_map_difference);
        end
        centres = double(participant_fit(char(paper.participant(i))).centers);
        centres = centres ./ max(vecnorm(centres, 2, 2), eps);
        maps = double(rec2.maps_norm);
        maps = maps ./ max(vecnorm(maps, 2, 2), eps);
        [~, fitted] = max(abs(maps * centres'), [], 2);
        state = fitted_to_presentation(fitted(:));
        peak_time = (double(rec2.peak_sample(:)) - 1) / double(rec2.sfreq);
        rows = find(csv_keys == record_keys(r));
        for w = rows(:)'
            [peaks, changes, state_counts, transitions] = count_window(peak_time, state, ...
                double(expected.window_start_s(w)), double(expected.window_end_s(w)), n_states);
            target_transitions = zeros(n_states, n_states);
            for source = 1:n_states
                for destination = 1:n_states
                    target_transitions(source, destination) = double(expected.(sprintf('t_%d_%d', source, destination))(w));
                end
            end
            target_counts = arrayfun(@(k) double(expected.(char(count_columns(k)))(w)), 1:n_states);
            ok = peaks == double(expected.n_gfp_peaks(w)) && changes == double(expected.n_state_changes(w)) && ...
                isequal(state_counts(:)', target_counts(:)') && isequal(transitions, target_transitions);
            if ~ok
                mismatches(end + 1, 1) = sprintf('%s window %d [%g,%g): peaks %d/%d, transition abs diff %d', ...
                    paper.file_path(i), expected.window_index(w), expected.window_start_s(w), ...
                    expected.window_end_s(w), peaks, expected.n_gfp_peaks(w), ...
                    sum(abs(transitions - target_transitions), 'all')); %#ok<AGROW>
            end
            n_windows = n_windows + 1;
        end
    end
    if n_windows < 1
        error('FSMS:ValidationVacuous', 'The validation covered zero windows; the gate would be vacuous.');
    end
    if ~isempty(mismatches)
        error('FSMS:WindowExportMismatch', ['Window export disagrees with the independent recomputation in %d of ' ...
            '%d windows, first: %s'], numel(mismatches), n_windows, mismatches(1));
    end
    validation = struct('performed', true, ...
        'validation_windows_csv', validation_csv, ...
        'validation_windows_sha256', fsms_sha256_file(validation_csv), ...
        'records_scope', scope, ...
        'n_records_reextracted', numel(record_rows), ...
        'n_windows_checked', n_windows, ...
        'n_windows_mismatched', 0, ...
        'peak_cache_reproduced_from_raw', logical(cache_reproduced), ...
        'max_abs_map_difference', max_map_difference, ...
        'label_mapping_cross_checked', true, ...
        'transition_matrix_exact_match', true, ...
        'method', ['GFP peaks re-extracted from the raw EEG file and compared with the hierarchy cache; per-window ' ...
        'peak, state and transition counts recomputed independently using the global-label-derived state mapping ' ...
        'and compared with every exported window of the checked records']);
end

function [peaks, changes, state_counts, transitions] = count_window(peak_time, state, start_s, stop_s, n_states)
% Straightforward loop implementation (deliberately different from the exporter's vectorised one).
    state_counts = zeros(1, n_states);
    transitions = zeros(n_states, n_states);
    peaks = 0;
    changes = 0;
    previous = 0;
    for n = 1:numel(peak_time)
        if peak_time(n) >= start_s && peak_time(n) < stop_s
            peaks = peaks + 1;
            state_counts(state(n)) = state_counts(state(n)) + 1;
            if previous ~= 0 && state(n) ~= previous
                transitions(previous, state(n)) = transitions(previous, state(n)) + 1;
                changes = changes + 1;
            end
            previous = state(n);
        end
    end
end
