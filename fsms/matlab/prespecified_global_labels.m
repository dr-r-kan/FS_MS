function result = prespecified_global_labels(hierarchy_dir, varargin)
%PRESPECIFIED_GLOBAL_LABELS Apply the pre-specified naming rule to a fitted global K=6 solution.
%
%   result = prespecified_global_labels(hierarchy_dir)
%
%   Reads  <hierarchy_dir>/global/global_solution.mat (Fit, common_labels, common_chanlocs), applies
%   template_prespecified_v2 (see that file for the rule) and writes, in the exact format
%   fit_microstate_hierarchical_dataset/resolve_global_manual_labels and
%   microstate_labels.load_global_labels already read:
%       global/global_manual_labels.csv   state_index,label,selected_utc,source
%       global/global_manual_labels.mat   labels, reference_centers, selected_utc, label_source
%   plus global/labelling_audit.json (rule id, template and script SHA-256, correlation matrices,
%   assignment, labelling_quality, states below threshold, channel-matching mode, MATLAB/SPM versions).
%
%   State names are NOMINAL: primary and secondary outcomes do not depend on them. Low template similarity
%   (any assigned |r| < 0.5) therefore does NOT stop the stage; it sets labelling_quality = 'below_threshold'
%   (descriptive per-state names unreliable) in the audit. Only TECHNICAL failures that make the template
%   comparison meaningless (channel mode not labels/coordinates_idw_to_template, too few channels,
%   non-finite maps, fewer or more fitted states than classes, non-unique or tied assignment) write
%   global/LABELLING_REJECTED.json, write NO label files and error with FSMS:LabellingRejected.
%
%   Idempotent: a repeat call with an unchanged global solution leaves existing label files untouched.
%   Optional 'template_file' overrides the MetaMaps path (tests only; the template SHA-256 is recorded).

    p = inputParser;
    addParameter(p, 'template_file', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'provenance', struct(), @isstruct);
    parse(p, varargin{:});
    env = fsms_paths();
    const = template_prespecified_v2_constants();
    hierarchy_dir = char(string(hierarchy_dir));
    fsms_assert_unrestricted(hierarchy_dir, 'reading/writing global labels');
    global_dir = fullfile(hierarchy_dir, 'global');
    solution_file = fullfile(global_dir, 'global_solution.mat');
    if ~isfile(solution_file)
        error('FSMS:GlobalSolutionMissing', 'Global solution not found: %s', solution_file);
    end
    template_file = char(string(p.Results.template_file));
    if isempty(template_file)
        template_file = fullfile(env.vg_root, 'MetaMaps_2023_06.set');
    end
    rejection_file = fullfile(global_dir, 'LABELLING_REJECTED.json');
    audit_file = fullfile(global_dir, 'labelling_audit.json');
    csv_file = fullfile(global_dir, 'global_manual_labels.csv');
    mat_file = fullfile(global_dir, 'global_manual_labels.mat');

    S = load(solution_file, 'Fit', 'common_labels', 'common_chanlocs');
    centers = double(S.Fit.centers);

    % Technical pre-checks that make the template comparison impossible.
    pre = {};
    if size(centers, 1) < const.K
        pre{end + 1} = sprintf('FEWER_STATES_THAN_CLASSES:states=%d,classes=%d', size(centers, 1), const.K);
    elseif size(centers, 1) > const.K
        pre{end + 1} = sprintf('MORE_STATES_THAN_CLASSES:states=%d,classes=%d', size(centers, 1), const.K);
    end
    if any(~isfinite(centers(:)))
        pre{end + 1} = 'NONFINITE_MAPS';
    end
    if ~isempty(pre)
        evidence = struct('rule_id', const.rule_id, 'accepted', false, 'rejection_reasons', {pre(:)}, ...
            'n_fitted_states', size(centers, 1), 'global_solution_sha256', fsms_sha256_file(solution_file), ...
            'nominal_label_semantics', const.nominal_label_semantics, ...
            'action_required', 'Technical failure: the template comparison is meaningless. The stage has stopped.', ...
            'created_utc', char(datetime('now', 'TimeZone', 'UTC', 'Format', 'yyyy-MM-dd''T''HH:mm:ssXXX')));
        fsms_write_json(rejection_file, evidence);
        error('FSMS:LabellingRejected', 'Pre-specified labelling rejected (%s). Evidence written to %s', ...
            strjoin(pre, '; '), rejection_file);
    end

    rule = template_prespecified_v2(centers, S.common_labels, S.common_chanlocs, 'template_file', template_file);
    audit = build_audit(rule, env, solution_file, template_file, centers, p.Results.provenance);

    if ~rule.accepted
        audit.accepted = false;
        audit.rejection_reasons = rule.reasons;
        audit.action_required = ['Technical failure: the template comparison is meaningless. The stage has ' ...
            'stopped. Investigate the evidence below; do not substitute hand-chosen labels.'];
        fsms_write_json(rejection_file, audit);
        error('FSMS:LabellingRejected', ...
            'Pre-specified labelling rejected (%s). Evidence written to %s', ...
            strjoin(rule.reasons, '; '), rejection_file);
    end
    if isfile(rejection_file)
        error('FSMS:LabellingRejectionMarkerPresent', ...
            ['A LABELLING_REJECTED.json marker exists in %s but the rule now accepts the solution. ' ...
             'The directory is inconsistent; use a fresh output directory.'], global_dir);
    end

    labels = rule.labels(:);
    existing = existing_labels(mat_file, csv_file);
    if ~isempty(existing)
        if ~strcmp(existing.source, const.rule_id)
            error('FSMS:ForeignLabelFile', ['global_manual_labels was written by source "%s", not by the ' ...
                'pre-specified rule. Refusing to mix labelling sources in %s.'], existing.source, global_dir);
        end
        if ~isequal(existing.labels(:), labels(:))
            error('FSMS:LabelsChanged', ['Existing pre-specified labels (%s) differ from the labels now ' ...
                'derived (%s). The global solution changed; use a fresh output directory.'], ...
                strjoin(existing.labels, ','), strjoin(labels, ','));
        end
        result = rule;
        result.audit_file = audit_file;
        result.labels_csv = csv_file;
        result.reused_existing = true;
        if ~isfile(audit_file)
            fsms_write_json(audit_file, audit);
        end
        return;
    end

    selected_utc = char(datetime('now', 'TimeZone', 'UTC', 'Format', 'yyyy-MM-dd''T''HH:mm:ssXXX'));
    audit.selected_utc = selected_utc;
    reference_centers = centers; %#ok<NASGU>
    payload = struct('labels', {labels}, 'reference_centers', reference_centers, ...
        'selected_utc', string(selected_utc), 'label_source', const.rule_id);
    fsms_atomic_save(mat_file, payload, '-v7');
    T = table((1:const.K)', string(labels), repmat(string(selected_utc), const.K, 1), ...
        repmat(string(const.rule_id), const.K, 1), ...
        'VariableNames', {'state_index', 'label', 'selected_utc', 'source'});
    fsms_atomic_writetable(T, csv_file);
    fsms_write_json(audit_file, audit);
    result = rule;
    result.audit_file = audit_file;
    result.labels_csv = csv_file;
    result.reused_existing = false;
    fprintf('Pre-specified labels (%s): %s | labelling_quality=%s (assigned |r|: %s)\n', const.rule_id, ...
        strjoin(compose('State %d=%s', (1:const.K)', string(labels))', ', '), rule.labelling_quality, ...
        strjoin(compose('%.3f', rule.assigned_abs_r(:))', ' '));
    if strcmp(rule.labelling_quality, 'below_threshold')
        fprintf(['NOTE: %d state(s) have assigned |r| < %.2f; state names are nominal and do not affect the ' ...
            'registered outcomes, but descriptive per-state names are unreliable.\n'], ...
            numel(rule.states_below_threshold), const.strong_threshold);
    end
end

function audit = build_audit(rule, env, solution_file, template_file, centers, provenance)
    here = fileparts(mfilename('fullpath'));
    vg = env.vg_root;
    scripts = {fullfile(here, 'prespecified_global_labels.m'), ...
        fullfile(here, 'template_prespecified_v2.m'), ...
        fullfile(here, 'template_prespecified_v2_constants.m'), ...
        fullfile(vg, '_internal', 'matlab', 'align_microstates_to_template.m'), ...
        fullfile(vg, '_internal', 'matlab', 'load_metamaps_templates.m'), ...
        fullfile(vg, '_internal', 'matlab', 'microstate_utilities.m')};
    script_sha = struct();
    for i = 1:numel(scripts)
        [~, name] = fileparts(scripts{i});
        script_sha.(name) = fsms_sha256_file(scripts{i});
    end
    audit = struct();
    audit.rule_id = rule.rule_id;
    audit.accepted = rule.accepted;
    audit.description = ['Exact optimal one-to-one assignment of the six global fitted states to classes ' ...
        'A,B,C,E,F,G (D excluded a priori) maximising the summed polarity-invariant |spatial correlation| ' ...
        'with MetaMaps_2023_06 K=7. accepted=true means the comparison was technically sound; low similarity ' ...
        '(assigned |r| < strong_threshold) is reported in labelling_quality, not enforced.'];
    audit.nominal_label_semantics = rule.nominal_label_semantics;
    audit.labelling_quality = rule.labelling_quality;
    audit.states_below_threshold = num2cell(rule.states_below_threshold(:))';   % always a JSON array
    audit.labels_by_fitted_state = rule.labels(:);
    audit.state_index = rule.state_index(:);
    audit.assigned_abs_r = rule.assigned_abs_r(:);
    audit.assigned_signed_r = rule.assigned_signed_r(:);
    audit.polarity = rule.polarity(:);
    audit.next_best_class = rule.next_best_class(:);
    audit.next_best_abs_r = rule.next_best_abs_r(:);
    audit.classes_allowed = rule.classes(:);
    audit.classes_excluded_a_priori = rule.excluded_classes(:);
    audit.correlation_matrix_abs = rule.corr_abs;
    audit.correlation_matrix_abs_columns = rule.classes(:);
    audit.correlation_matrix_abs_all_template_classes = rule.corr_abs_all_template_classes;
    audit.correlation_matrix_abs_all_template_classes_columns = rule.template_class_order(:);
    audit.assignment_class_index = rule.assignment_class_index(:);
    audit.total_abs_r = rule.total_abs_r;
    audit.runner_up_total_abs_r = rule.runner_up_total_abs_r;
    audit.margin_total_abs_r = rule.margin_total_abs_r;
    audit.n_tied_optimal_assignments = rule.n_tied_optimal_assignments;
    audit.strong_threshold = rule.strong_threshold;
    audit.tie_rule = ['lexicographically smallest assignment vector (entry k = index of the class ' ...
        'A,B,C,E,F,G given to fitted state k); total |r| tolerance 1e-9; exact ties are a technical failure'];
    audit.channel_match_mode = rule.channel_match_mode;
    audit.n_common_channels = rule.n_common_channels;
    audit.n_estimated_channels_used = rule.n_estimated_channels_used;
    audit.min_label_matches = rule.min_label_matches;
    audit.matched_channel_labels = rule.matched_channel_labels(:);
    audit.template_file = relative_to(template_file, env.repo_root);
    audit.template_sha256 = fsms_sha256_file(template_file);
    audit.template_K = rule.template_K;
    audit.global_solution_file = relative_to(solution_file, env.repo_root);
    audit.global_solution_sha256 = fsms_sha256_file(solution_file);
    audit.reference_centers_sha256 = fsms_sha256_array(centers);
    audit.script_sha256 = script_sha;
    audit.matlab_version = env.matlab_version;
    audit.matlab_release = env.matlab_release;
    audit.spm = env.spm;
    audit.created_utc = char(datetime('now', 'TimeZone', 'UTC', 'Format', 'yyyy-MM-dd''T''HH:mm:ssXXX'));
    if ~isempty(fieldnames(provenance))
        audit.provenance = provenance;
    end
end

function out = relative_to(path_value, root)
    path_value = char(string(path_value));
    root = char(string(root));
    if startsWith(lower(path_value), lower(root))
        out = strrep(path_value(numel(root) + 2:end), '\', '/');
    else
        out = strrep(path_value, '\', '/');
    end
end

function existing = existing_labels(mat_file, csv_file)
    existing = [];
    if isfile(mat_file)
        cached = load(mat_file);
        source = '';
        if isfield(cached, 'label_source'), source = char(string(cached.label_source)); end
        existing = struct('labels', {cellstr(string(cached.labels(:)))}, 'source', source);
    elseif isfile(csv_file)
        T = readtable(csv_file, 'TextType', 'string', 'Delimiter', ',');
        T = sortrows(T, 'state_index');
        existing = struct('labels', {cellstr(T.label)}, 'source', char(T.source(1)));
    end
end
