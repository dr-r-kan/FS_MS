function result = run_study_hierarchy(manifest_csv, out_dir, varargin)
%RUN_STUDY_HIERARCHY Stage C of the FS/ES microstate study, headless, deterministic and idempotent.
%
%   result = run_study_hierarchy(manifest_csv, out_dir, ...)
%
%   manifest_csv  study manifest (participant, condition, group, file_path, ... crop_start/end_seconds)
%   out_dir       a FRESH, versioned output folder (the orchestrator supplies it). Layout:
%       STAGE_C_INPUTS.json           inputs fingerprint (manifest SHA-256 + parameters + script hashes)
%       STAGE_C_DONE.json             done-sentinel with output hashes (written last; its presence + matching
%                                     fingerprint makes a re-run skip the stage)
%       STAGE_C_FAILED.json           written if the stage errors (cleared on success)
%       manifest_full.csv, analysis_manifest.csv, exclusions.csv, screening.json, screening_peak_counts.csv
%       screening_peaks/              peak extraction used for the pre-specified screening
%       hierarchy/                    global/, groups/, conditions/, participants/, gfp_peaks/ ...
%                                     global/global_manual_labels.csv|mat + labelling_audit.json
%       validation/directional_peak_backfit_windows.csv   freshly exported from THIS run
%       timestamped_backfits/         transition_index.csv, recovery_audit.json, transition_cache/*.mat
%
%   PIPELINE
%     1 hash manifest/parameters/sources; refuse a folder built from a different manifest, parameters or
%       sources (FSMS:StaleOutputDir / FSMS:StageInputsChanged / FSMS:UnstampedOutputDir)
%     2 screening: GFP peaks are extracted once (small global fit, discarded); records/participants below the
%       pre-specified peak minima are EXCLUDED with a reason code into exclusions.csv; the study continues
%     3 global K=6 fit -> pre-specified labels (template_prespecified_v2: names are nominal, low template
%       similarity is flagged as labelling_quality='below_threshold'; only TECHNICAL failures stop the stage with
%       global/LABELLING_REJECTED.json) -> group/condition/participant hierarchy (no TANOVA, topoplots,
%       cardiac or participant-condition fits)
%     4 export_directional_peak_windows -> validation/directional_peak_backfit_windows.csv
%     5 recover_timestamped_transition_backfits using THAT CSV for the window gate
%
%   Name-value options (defaults are the pre-specified study settings)
%     validation_windows_csv   destination of the freshly exported windows CSV
%                              (default <out_dir>/validation/directional_peak_backfit_windows.csv)
%     conditions               {'II','preict'}
%     min_peaks_per_record     20    GFP-peak minimum for a record (the vendored min_gfp_peaks_per_file)
%     min_peaks_per_participant 60   10 x K; below it the participant's K=6 fit is not estimable
%     validation_records       'all' ('first_per_participant' to save time; recorded in the audit)
%     full_backfit_validation  'first_per_participant' ('first' | 'all')
%     max_records              Inf  (debugging only)
%     real_study_mode          false  lifts the restricted-data guard (fsms_assert_unrestricted) for this call only;
%                              the orchestrator sets it after fsms.lock has verified the registration record
%     hierarchy_args           {}   extra name-value pairs for fit_microstate_hierarchical_dataset
%                              (diagnostics/tests, e.g. {'max_global_maps', 20000}; part of the fingerprint
%                              and the audit; do NOT use for the registered run)

    p = inputParser;
    addRequired(p, 'manifest_csv', @(x) ischar(x) || isstring(x));
    addRequired(p, 'out_dir', @(x) ischar(x) || isstring(x));
    addParameter(p, 'validation_windows_csv', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'conditions', {'II', 'preict'}, @(x) iscell(x) || isstring(x));
    addParameter(p, 'min_peaks_per_record', 20, @(x) isnumeric(x) && isscalar(x) && x >= 1);
    addParameter(p, 'min_peaks_per_participant', 60, @(x) isnumeric(x) && isscalar(x) && x >= 7);
    addParameter(p, 'validation_records', 'all', @(x) any(strcmp(char(x), {'all', 'first_per_participant'})));
    addParameter(p, 'full_backfit_validation', 'first_per_participant', ...
        @(x) any(strcmp(char(x), {'first_per_participant', 'first', 'all'})));
    addParameter(p, 'max_records', Inf, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'hierarchy_args', {}, @(x) iscell(x) && mod(numel(x), 2) == 0);
    addParameter(p, 'real_study_mode', false, @islogical);
    parse(p, manifest_csv, out_dir, varargin{:});
    o = p.Results;
    t_start = tic;

    env = fsms_paths();
    manifest_csv = absolute_path(manifest_csv);
    out_dir = absolute_path(out_dir);
    if o.real_study_mode
        setappdata(0, 'FSMS_REAL_STUDY_MODE', true);
        guard_cleanup = onCleanup(@() local_clear_real_mode()); %#ok<NASGU>
    end
    fsms_assert_unrestricted(manifest_csv, 'reading the manifest');
    fsms_assert_unrestricted(out_dir, 'writing stage-C output');
    const = template_prespecified_v2_constants();
    [T_full, minfo] = fsms_read_manifest(manifest_csv);

    stage_params = struct( ...
        'K', const.K, 'labelling_rule', const.rule_id, 'labelling_classes', {const.classes}, ...
        'labelling_strong_threshold', const.strong_threshold, ...
        'labelling_policy', 'low similarity flagged (labelling_quality); technical failures stop', ...
        'conditions', {cellstr(string(o.conditions(:)))}, ...
        'min_peaks_per_record', o.min_peaks_per_record, ...
        'min_peaks_per_participant', o.min_peaks_per_participant, ...
        'validation_records', char(o.validation_records), ...
        'full_backfit_validation', char(o.full_backfit_validation), ...
        'max_records', o.max_records, ...
        'hierarchy_args', {o.hierarchy_args}, ...
        'screening_max_global_maps', 3000, ...
        'analyses_off', 'tanova, topoplots, cardiac phase, participant-condition fits, backfit metric tables');
    fp = fsms_script_fingerprint(env.repo_root);
    environment = struct('matlab_release', env.matlab_release, 'spm_version', env.spm.version, ...
        'spm_git_commit', env.spm.git_commit, 'spm_mix_sha256', env.spm.spm_mix_sha256, ...
        'spm_mix_prior_sha256', env.spm.spm_mix_prior_sha256);
    inputs = struct('manifest_sha256', minfo.manifest_sha256, 'params', stage_params, ...
        'script_fingerprint', fp.combined, 'environment', environment);
    inputs_fp = fsms_sha256_text(jsonencode(inputs));

    stamp_file = fullfile(out_dir, 'STAGE_C_INPUTS.json');
    done_file = fullfile(out_dir, 'STAGE_C_DONE.json');
    failed_file = fullfile(out_dir, 'STAGE_C_FAILED.json');
    preflight(out_dir, stamp_file, inputs, inputs_fp);
    heal_mode = false;
    if isfile(done_file)
        [status, result, problems] = verify_done(done_file, out_dir, inputs_fp);
        if strcmp(status, 'complete')
            fprintf('Stage C already complete for these inputs; skipping (%s).\n', done_file);
            return;
        end
        fprintf('Stage C sentinel present but derived outputs are missing or changed (%s); healing.\n', ...
            strjoin(problems(1:min(numel(problems), 5)), '; '));
        heal_mode = true;
    end
    if ~isfolder(out_dir), mkdir(out_dir); end
    if ~isfile(stamp_file)
        stamp = struct('schema_version', 1, 'manifest_sha256', minfo.manifest_sha256, ...
            'manifest_csv', manifest_csv, 'inputs_fingerprint', inputs_fp, 'inputs', inputs, ...
            'created_utc', now_utc());
        fsms_write_json(stamp_file, stamp);
    end

    try
        result = run_body(env, T_full, minfo, manifest_csv, out_dir, o, stage_params, inputs_fp, ...
            fp, environment, t_start, heal_mode);
    catch ME
        info = struct('identifier', ME.identifier, 'message', ME.message, 'created_utc', now_utc());
        if ~isempty(ME.stack)
            info.top_of_stack = sprintf('%s:%d', ME.stack(1).name, ME.stack(1).line);
        end
        try, fsms_write_json(failed_file, info); catch, end
        rethrow(ME);
    end
    if isfile(failed_file), delete(failed_file); end
end

function result = run_body(env, T_full, minfo, manifest_csv, out_dir, o, stage_params, inputs_fp, fp, environment, t_start, heal_mode)
    hierarchy_dir = fullfile(out_dir, 'hierarchy');
    screen_dir = fullfile(out_dir, 'screening_peaks');
    backfit_dir = fullfile(out_dir, 'timestamped_backfits');
    windows_csv = char(string(o.validation_windows_csv));
    if isempty(windows_csv)
        windows_csv = fullfile(out_dir, 'validation', 'directional_peak_backfit_windows.csv');
    end
    fsms_assert_unrestricted(windows_csv, 'writing the windows CSV');
    full_csv = fullfile(out_dir, 'manifest_full.csv');
    analysis_csv = fullfile(out_dir, 'analysis_manifest.csv');
    exclusions_csv = fullfile(out_dir, 'exclusions.csv');
    counts_csv = fullfile(out_dir, 'screening_peak_counts.csv');
    screening_json = fullfile(out_dir, 'screening.json');

    phase = tic;
    % ---- 2. screening -------------------------------------------------------------------------
    fsms_atomic_writetable(T_full, full_csv, 'QuoteStrings', true);
    if isfile(screening_json)
        decision = fsms_read_json(screening_json);
        if ~strcmp(decision.analysis_manifest_sha256, fsms_sha256_file(analysis_csv))
            error('FSMS:AnalysisManifestChanged', 'analysis_manifest.csv no longer matches screening.json in %s.', out_dir);
        end
        fprintf('Screening already decided (%d excluded record(s)).\n', decision.n_excluded_records);
    else
        fprintf('Screening: extracting GFP peaks for %d record(s)...\n', height(T_full));
        screen_args = [o.hierarchy_args, {'max_global_maps', 3000, 'template_file', ''}];
        run_vg_microstates(full_csv, screen_dir, 'labelling_mode', 'template_auto', ...
            'require_template_labels', false, 'global_only', true, 'extra_hierarchy_args', screen_args);
        [retained, exclusions, peak_counts] = fsms_screen_peak_counts(T_full, screen_dir, ...
            o.min_peaks_per_record, o.min_peaks_per_participant);
        T_keep = T_full(retained, :);
        fsms_atomic_writetable(T_keep, analysis_csv, 'QuoteStrings', true);
        fsms_atomic_writetable(exclusions, exclusions_csv, 'QuoteStrings', true);
        fsms_atomic_writetable(table(T_full.manifest_row, T_full.file_path, peak_counts, ~retained, ...
            'VariableNames', {'manifest_row', 'file_path', 'n_gfp_peaks', 'excluded'}), counts_csv, ...
            'QuoteStrings', true);
        decision = struct('schema_version', 1, 'n_input_records', height(T_full), ...
            'n_retained_records', height(T_keep), 'n_excluded_records', height(exclusions), ...
            'n_excluded_participants', numel(unique(exclusions.participant(exclusions.reason_code == "PARTICIPANT_TOO_FEW_GFP_PEAKS"))), ...
            'min_peaks_per_record', o.min_peaks_per_record, ...
            'min_peaks_per_participant', o.min_peaks_per_participant, ...
            'reason_codes', {{'RECORD_TOO_FEW_GFP_PEAKS', 'PARTICIPANT_TOO_FEW_GFP_PEAKS'}}, ...
            'analysis_manifest_sha256', fsms_sha256_file(analysis_csv), ...
            'exclusions_sha256', fsms_sha256_file(exclusions_csv), ...
            'decided_utc', now_utc());
        fsms_write_json(screening_json, decision);
        fprintf('Screening: %d retained, %d excluded.\n', height(T_keep), height(exclusions));
    end
    [T_analysis, analysis_info] = fsms_read_manifest(analysis_csv);
    fprintf('[timing] screening: %.1f s\n', toc(phase)); phase = tic;

    % ---- 3. hierarchy ---------------------------------------------------------------------------
    % In heal mode STAGE_C_DONE.json has just verified every frozen file (hierarchy solutions, labels,
    % manifests, screening), so the hierarchy is not re-run (the vendored code would re-save its files).
    if heal_mode
        fprintf('Heal mode: hierarchy verified intact; only the window export and transition stage are re-run.\n');
    else
    if decision.n_excluded_records == 0
        % Peak extraction does not depend on the global fit: reuse the screening caches (atomic copies).
        reuse_peak_caches(screen_dir, hierarchy_dir, height(T_analysis));
    end
    fprintf('Hierarchy: global fit, pre-specified labelling, group/condition/participant fits...\n');
    run_vg_microstates(analysis_csv, hierarchy_dir, 'labelling_mode', 'prespecified', ...
        'extra_hierarchy_args', o.hierarchy_args);
    % Retained records must still satisfy the minima on the caches the final run used.
    [~, post_exclusions] = fsms_screen_peak_counts(T_analysis, hierarchy_dir, ...
        o.min_peaks_per_record, o.min_peaks_per_participant);
    if ~isempty(post_exclusions)
        error('FSMS:RetainedBelowThreshold', ['After the final extraction %d retained record(s) fall below the ' ...
            'pre-specified GFP-peak minima (first: %s). Re-run in a fresh output folder.'], ...
            height(post_exclusions), post_exclusions.file_path(1));
    end
    end

    fprintf('[timing] hierarchy (global fit, labelling, group/condition/participant fits): %.1f s\n', toc(phase)); phase = tic;
    % ---- 4. window export (validation windows from THIS run) -----------------------------------
    fprintf('Exporting directional peak windows...\n');
    export_directional_peak_windows(hierarchy_dir, analysis_csv, windows_csv);

    fprintf('[timing] window export: %.1f s\n', toc(phase)); phase = tic;
    % ---- 5. exact transitions + audit ------------------------------------------------------------
    provenance = struct('stage_params', stage_params, 'inputs_fingerprint', inputs_fp, ...
        'study_manifest_csv', manifest_csv, 'study_manifest_sha256', minfo.manifest_sha256, ...
        'analysis_manifest_sha256', analysis_info.manifest_sha256, ...
        'n_input_records', decision.n_input_records, 'n_excluded_records', decision.n_excluded_records, ...
        'exclusions_csv', exclusions_csv, 'exclusions_sha256', decision.exclusions_sha256, ...
        'validation_windows_csv', windows_csv, ...
        'global_labels_csv', fullfile(hierarchy_dir, 'global', 'global_manual_labels.csv'), ...
        'labelling_audit_json', fullfile(hierarchy_dir, 'global', 'labelling_audit.json'));
    audit = recover_timestamped_transition_backfits(analysis_csv, hierarchy_dir, backfit_dir, ...
        'conditions', o.conditions, 'validation_windows_csv', windows_csv, ...
        'validation_records', char(o.validation_records), ...
        'full_backfit_validation', char(o.full_backfit_validation), ...
        'max_records', o.max_records, 'provenance', provenance);

    fprintf('[timing] window validation + transition export + audit: %.1f s\n', toc(phase));
    % ---- sentinel --------------------------------------------------------------------------------
    outputs = struct('path', {}, 'sha256', {}, 'bytes', {});
    key_files = {fullfile(out_dir, 'STAGE_C_INPUTS.json'), screening_json, full_csv, analysis_csv, exclusions_csv, ...
        counts_csv, windows_csv, fullfile(backfit_dir, 'transition_index.csv'), ...
        fullfile(backfit_dir, 'recovery_audit.json'), ...
        fullfile(hierarchy_dir, 'global', 'global_manual_labels.csv'), ...
        fullfile(hierarchy_dir, 'global', 'global_manual_labels.mat'), ...
        fullfile(hierarchy_dir, 'global', 'labelling_audit.json')};
    for i = 1:numel(key_files)
        info = dir(key_files{i});
        outputs(end + 1) = struct('path', relative_to(key_files{i}, out_dir), ... %#ok<AGROW>
            'sha256', fsms_sha256_file(key_files{i}), 'bytes', info.bytes);
    end
    solutions = struct('path', {}, 'sha256', {}, 'bytes', {});
    solution_files = [{fullfile(hierarchy_dir, 'global', 'global_solution.mat')}; ...
        arrayfun(@(k) fullfile(hierarchy_dir, 'participants', fsms_clean_key(k), 'participant_solution.mat'), ...
        unique(T_analysis.participant, 'stable'), 'UniformOutput', false)];
    for i = 1:numel(solution_files)
        info = dir(solution_files{i});
        solutions(end + 1) = struct('path', relative_to(solution_files{i}, out_dir), ... %#ok<AGROW>
            'sha256', fsms_sha256_file(solution_files{i}), 'bytes', info.bytes);
    end
    caches = dir(fullfile(backfit_dir, 'transition_cache', '*_transitions.mat'));
    cache_list = struct('name', {}, 'bytes', {});
    for i = 1:numel(caches)
        cache_list(end + 1) = struct('name', caches(i).name, 'bytes', caches(i).bytes); %#ok<AGROW>
    end
    done = struct('schema_version', 1, 'stage', 'C_vg_microstates_transition_backfit', ...
        'inputs_fingerprint', inputs_fp, 'manifest_sha256', minfo.manifest_sha256, ...
        'params', stage_params, 'script_fingerprint', fp.combined, 'environment', environment, ...
        'fit_labels', {audit.fit_labels}, 'labelling_rule', audit.labelling_rule, 'labelling_quality', audit.labelling_quality, ...
        'n_records_analysed', audit.n_timestamped_records, 'n_excluded_records', decision.n_excluded_records, ...
        'n_transitions', audit.n_transitions, ...
        'compact_export_validated_against_full_backfit', audit.compact_export_validated_against_full_backfit, ...
        'transition_matrix_exact_match', audit.focused_window_fit_recovery_validation.transition_matrix_exact_match, ...
        'runtime_s', toc(t_start), 'finished_utc', now_utc(), ...
        'outputs', outputs, 'hierarchy_solutions', solutions, 'transition_cache', cache_list);
    fsms_write_json(fullfile(out_dir, 'STAGE_C_DONE.json'), done);
    result = result_from_done(done, out_dir, false);
    fprintf('Stage C complete in %.1f s: labels %s (labelling_quality=%s), %d records, %d transitions.\n', toc(t_start), ...
        strjoin(string(audit.fit_labels), ''), audit.labelling_quality, audit.n_timestamped_records, audit.n_transitions);
end

function preflight(out_dir, stamp_file, inputs, inputs_fp)
% Refuse to adopt a folder that was not built from exactly these inputs.
    if ~isfolder(out_dir), return; end
    listing = dir(out_dir);
    listing = listing(~ismember({listing.name}, {'.', '..'}));
    if isempty(listing), return; end
    if ~isfile(stamp_file)
        error('FSMS:UnstampedOutputDir', ['%s is not empty and holds no STAGE_C_INPUTS.json; refusing to adopt ' ...
            'a folder of unknown provenance. Use a fresh, versioned output folder.'], out_dir);
    end
    stamp = fsms_read_json(stamp_file);
    if ~strcmp(stamp.manifest_sha256, inputs.manifest_sha256)
        error('FSMS:StaleOutputDir', ['%s was built from a different manifest (stamped SHA-256 %s, current ' ...
            '%s). Caches from another manifest must not be reused; use a fresh, versioned output folder.'], ...
            out_dir, stamp.manifest_sha256, inputs.manifest_sha256);
    end
    if ~strcmp(stamp.inputs_fingerprint, inputs_fp)
        changed = {};
        if ~strcmp(stamp.inputs.script_fingerprint, inputs.script_fingerprint), changed{end + 1} = 'MATLAB sources/template'; end
        if ~strcmp(fsms_sha256_text(jsonencode(stamp.inputs.params)), fsms_sha256_text(jsonencode(jsondecode(jsonencode(inputs.params))))), changed{end + 1} = 'parameters'; end
        if ~strcmp(fsms_sha256_text(jsonencode(stamp.inputs.environment)), fsms_sha256_text(jsonencode(jsondecode(jsonencode(inputs.environment))))), changed{end + 1} = 'MATLAB/SPM environment'; end
        if isempty(changed), changed{1} = 'inputs'; end
        error('FSMS:StageInputsChanged', ['%s was built with different %s (same manifest). Existing caches ' ...
            'must not be mixed with new settings; use a fresh, versioned output folder.'], out_dir, strjoin(changed, ' and '));
    end
end

function [status, result, problems] = verify_done(done_file, out_dir, inputs_fp)
% 'complete': every recorded output is intact -> skip. 'heal': only DERIVED outputs (transition caches, index,
% audit, windows CSV) are missing or changed -> the body is re-run in resume mode, which revalidates every
% cache and rebuilds the bad ones. Any change to a FROZEN input (manifests, screening, exclusions, labels,
% hierarchy solutions) is an error: those must never be regenerated in place.
    done = fsms_read_json(done_file);
    if ~strcmp(done.inputs_fingerprint, inputs_fp)
        error('FSMS:StageInputsChanged', 'STAGE_C_DONE.json in %s was written for different inputs.', out_dir);
    end
    derived = {'validation/directional_peak_backfit_windows.csv', 'timestamped_backfits/transition_index.csv', ...
        'timestamped_backfits/recovery_audit.json'};
    problems = {};
    entries = [num2cell(done.outputs(:)); num2cell(done.hierarchy_solutions(:))];
    for i = 1:numel(entries)
        entry = entries{i};
        file = fullfile(out_dir, strrep(entry.path, '/', filesep));
        intact = isfile(file) && strcmp(fsms_sha256_file(file), entry.sha256);
        if intact, continue; end
        if any(strcmp(entry.path, derived))
            problems{end + 1} = ['derived output changed or missing: ' entry.path]; %#ok<AGROW>
        else
            error('FSMS:DoneSentinelInconsistent', ['Frozen stage-C file %s was modified or removed after the ' ...
                'stage completed. Do not re-run in place; investigate or use a fresh folder.'], entry.path);
        end
    end
    caches = done.transition_cache;
    for i = 1:numel(caches)
        info = dir(fullfile(out_dir, 'timestamped_backfits', 'transition_cache', caches(i).name));
        if isempty(info) || info.bytes ~= caches(i).bytes
            problems{end + 1} = ['transition cache changed or missing: ' caches(i).name]; %#ok<AGROW>
        end
    end
    if isempty(problems), status = 'complete'; else, status = 'heal'; end
    result = result_from_done(done, out_dir, strcmp(status, 'complete'));
end

function result = result_from_done(done, out_dir, skipped)
    result = struct('skipped', skipped, 'out_dir', out_dir, ...
        'hierarchy_dir', fullfile(out_dir, 'hierarchy'), ...
        'backfit_dir', fullfile(out_dir, 'timestamped_backfits'), ...
        'transition_index', fullfile(out_dir, 'timestamped_backfits', 'transition_index.csv'), ...
        'recovery_audit', fullfile(out_dir, 'timestamped_backfits', 'recovery_audit.json'), ...
        'fit_labels', {cellstr(string(done.fit_labels(:)))}, 'labelling_quality', done.labelling_quality, ...
        'n_records_analysed', done.n_records_analysed, 'n_excluded_records', done.n_excluded_records, ...
        'inputs_fingerprint', done.inputs_fingerprint, 'done_file', fullfile(out_dir, 'STAGE_C_DONE.json'));
end

function reuse_peak_caches(screen_dir, hierarchy_dir, n_records)
    source = fullfile(screen_dir, 'gfp_peaks');
    target = fullfile(hierarchy_dir, 'gfp_peaks');
    if ~isfolder(target), mkdir(target); end
    files = dir(fullfile(source, '*_gfp_peaks.mat'));
    if numel(files) ~= n_records
        error('FSMS:PeakCacheCount', 'Screening produced %d peak caches for %d records.', numel(files), n_records);
    end
    for i = 1:numel(files)
        destination = fullfile(target, files(i).name);
        if ~isfile(destination)
            fsms_copy_atomic(fullfile(source, files(i).name), destination);
        end
    end
    meta_target = fullfile(hierarchy_dir, 'common_channel_metadata.mat');
    if ~isfile(meta_target)
        fsms_copy_atomic(fullfile(screen_dir, 'common_channel_metadata.mat'), meta_target);
    end
end

function local_clear_real_mode()
    if isappdata(0, 'FSMS_REAL_STUDY_MODE'), rmappdata(0, 'FSMS_REAL_STUDY_MODE'); end
end

function out = absolute_path(path_value)
    out = char(java.io.File(char(string(path_value))).getAbsolutePath());
end

function out = relative_to(path_value, root)
    out = strrep(path_value(numel(root) + 2:end), '\', '/');
end

function s = now_utc()
    s = char(datetime('now', 'TimeZone', 'UTC', 'Format', 'yyyy-MM-dd''T''HH:mm:ssXXX'));
end
