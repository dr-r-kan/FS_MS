function [TANOVA, results_csv] = run_microstate_hierarchical_tanova(results_mat, varargin)
% RUN_MICROSTATE_HIERARCHICAL_TANOVA
%
% Non-parametric topographic ANOVA/TANOVA for the output of
% fit_microstate_hierarchical_dataset.m.
%
% The script tests whether participant-level microstate template maps differ
% topographically between groups, between conditions, and for the
% group-by-condition interaction when those factors are present in HResults.
%
% Best-practice choices used here:
%   1. The unit of inference is the participant-level template, not the
%      already-averaged group/condition template.
%   2. Each scalp map is common-average centred, sign-aligned to the global
%      template for the corresponding microstate, and GFP/RMS-normalised.
%      This removes strength-only effects and tests topographic shape.
%   3. The statistic is dGFP/RMS of factor-level mean-map differences.
%   4. Condition effects use within-participant label randomisation.
%   5. Group effects use participant-level group-label randomisation.
%   6. Mixed interactions use within-participant condition-label
%      randomisation while preserving group labels.
%   7. Per-effect max-statistic correction is applied across microstate
%      classes, with additional BH-FDR columns written for convenience.
%
% Usage:
%   TANOVA = run_microstate_hierarchical_tanova();
%   TANOVA = run_microstate_hierarchical_tanova('outputs/hierarchical_microstates/hierarchical_microstate_results.mat');
%   TANOVA = run_microstate_hierarchical_tanova(results_mat, 'n_permutations', 10000);
%
% Name-value options:
%   'config_file'                      JSON config; used only for defaults.
%                                      Default: 'microstate_config.json'.
%   'output_dir'                       Output directory. Default: config
%                                      diagnostic dir / microstate_tanova, or
%                                      sibling tanova folder beside results.
%   'n_permutations'                   Randomisation count. Default: 5000.
%   'alpha'                            Alpha for plot threshold. Default: 0.05.
%   'random_seed'                      RNG seed. Default: 1.
%   'exclude_inherited_nodes'          Exclude inherited participant nodes.
%                                      Default: true.
%   'require_complete_within_subject'  For condition/interactions, keep only
%                                      units with all condition levels.
%                                      Default: true.
%   'renormalise_factor_means'         GFP-normalise factor-level mean maps
%                                      before computing dGFP. Default: true.
%   'run_posthoc'                      Pairwise and simple-effect tests.
%                                      Default: true.
%   'run_bayesian'                     Bayesian bootstrap TANOVA summaries.
%                                      Default: true.
%   'n_posterior'                      Bayesian bootstrap draws. Default:
%                                      same as n_permutations.
%   'bayesian_rope'                    Region of practical equivalence for
%                                      dGFP/RMS map-difference statistics.
%                                      Default: 0.05.
%   'save_plots'                       Save heatmap and topographic summaries.
%                                      Default: true.
%   'verbose'                          Print progress. Default: true.
%
% Outputs written:
%   tanova_results.csv
%   bayesian_tanova_results.csv
%   tanova_participant_template_manifest.csv
%   tanova_summary.mat
%   tanova_methods_notes.txt
%   plots/tanova_pmax_heatmap.png
%   plots/mean_maps_*.png, if EEGLAB topoplot and chanlocs are available/on path.

    if nargin < 1 || isempty(results_mat)
        results_mat = default_results_mat();
    end

    p = inputParser;
    addRequired(p, 'results_mat', @(x) ischar(x) || isstring(x));
    addParameter(p, 'config_file', 'microstate_config.json', @(x) ischar(x) || isstring(x));
    addParameter(p, 'output_dir', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'semiology_manifest_csv', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'n_permutations', 5000, @(x) isnumeric(x) && isscalar(x) && x >= 1);
    addParameter(p, 'alpha', 0.05, @(x) isnumeric(x) && isscalar(x) && x > 0 && x < 1);
    addParameter(p, 'random_seed', 1, @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'exclude_inherited_nodes', true, @(x) islogical(x) && isscalar(x));
    addParameter(p, 'require_complete_within_subject', true, @(x) islogical(x) && isscalar(x));
    addParameter(p, 'renormalise_factor_means', true, @(x) islogical(x) && isscalar(x));
    addParameter(p, 'run_posthoc', true, @(x) islogical(x) && isscalar(x));
    addParameter(p, 'run_bayesian', true, @(x) islogical(x) && isscalar(x));
    addParameter(p, 'n_posterior', NaN, @(x) isnumeric(x) && isscalar(x) && (isnan(x) || x >= 1));
    addParameter(p, 'bayesian_rope', 0.05, @(x) isnumeric(x) && isscalar(x) && x >= 0);
    addParameter(p, 'bayesian_results_csv', 'bayesian_tanova_results.csv', @(x) ischar(x) || isstring(x));
    addParameter(p, 'save_plots', true, @(x) islogical(x) && isscalar(x));
    addParameter(p, 'verbose', true, @(x) islogical(x) && isscalar(x));
    parse(p, results_mat, varargin{:});
    cfg = p.Results;
    cfg.results_mat = char(cfg.results_mat);
    cfg.config_file = char(cfg.config_file);
    cfg.n_permutations = round(cfg.n_permutations);
    if isnan(cfg.n_posterior)
        cfg.n_posterior = cfg.n_permutations;
    end
    cfg.n_posterior = round(cfg.n_posterior);
    cfg.bayesian_results_csv = char(cfg.bayesian_results_csv);

    if ~isfile(cfg.results_mat)
        error('Results MAT file not found: %s', cfg.results_mat);
    end
    if isempty(cfg.output_dir)
        cfg.output_dir = default_output_dir(cfg.results_mat, cfg.config_file);
    else
        cfg.output_dir = char(cfg.output_dir);
    end
    if ~exist(cfg.output_dir, 'dir')
        mkdir(cfg.output_dir);
    end
    plot_dir = fullfile(cfg.output_dir, 'plots');
    if cfg.save_plots && ~exist(plot_dir, 'dir')
        mkdir(plot_dir);
    end

    rng(cfg.random_seed, 'twister');

    if cfg.verbose
        fprintf('\n========================================\n');
        fprintf('Microstate hierarchical TANOVA\n');
        fprintf('========================================\n');
        fprintf('Input:        %s\n', cfg.results_mat);
        fprintf('Output:       %s\n', cfg.output_dir);
        fprintf('Permutations: %d\n', cfg.n_permutations);
        fprintf('Inherited participant nodes excluded: %s\n', tf(cfg.exclude_inherited_nodes));
        fprintf('========================================\n\n');
    end

    S = load(cfg.results_mat);
    if isfield(S, 'HResults')
        H = S.HResults;
    elseif isfield(S, 'MResults') && isstruct(S.MResults) && isfield(S.MResults, 'HResults')
        H = S.MResults.HResults;
    elseif isfield(S, 'MResults') && isstruct(S.MResults) && isfield(S.MResults, 'manifest') && isfield(S.MResults, 'file_fits') && isfield(S.MResults, 'meta_fit')
        H = build_hresults_from_mresults(S.MResults);
    else
        error('The MAT file does not contain HResults.');
    end

    D = extract_participant_template_data(H, cfg, "participant_condition");
    D_event = extract_participant_template_data(H, cfg, "event_type");
    D_semiology = extract_participant_template_data(H, cfg, "semiology");
    if isempty(D.X)
        error('No participant-level template maps were available for TANOVA.');
    end

    if cfg.verbose
        fprintf('Participant-template rows retained: %d\n', size(D.X, 1));
        fprintf('Participants/units: %d\n', numel(unique(D.unit)));
        fprintf('Microstates: %d | channels: %d\n', D.K, D.n_channels);
        fprintf('Detected factors: group=%s (%d levels), condition=%s (%d levels)\n\n', ...
            tf(D.has_group), numel(unique(D.group(D.group ~= ""))), ...
            tf(D.has_condition), numel(unique(D.condition(D.condition ~= ""))));
    end

    manifest_csv = fullfile(cfg.output_dir, 'tanova_participant_template_manifest.csv');
    writetable(D.manifest, manifest_csv);
    writetable(D_event.manifest, fullfile(cfg.output_dir, 'tanova_event_type_manifest.csv'));
    writetable(D_semiology.manifest, fullfile(cfg.output_dir, 'tanova_semiology_manifest.csv'));

    [all_tables, notes] = run_limited_planned_tests(D, D_event, D_semiology, cfg, false);

    results_table = vertcat_nonempty_tables(all_tables);
    if ~isempty(results_table)
        results_table = add_fdr_columns(results_table);
    end

    results_csv = fullfile(cfg.output_dir, 'tanova_results.csv');
    writetable(results_table, results_csv);

    bayesian_table = table();
    bayesian_csv = '';
    if cfg.run_bayesian
        bayesian_tables = run_limited_planned_tests(D, D_event, D_semiology, cfg, true);
        bayesian_table = vertcat_nonempty_tables(bayesian_tables);
        bayesian_csv = output_path(cfg.output_dir, cfg.bayesian_results_csv);
        writetable(bayesian_table, bayesian_csv);
    end

    TANOVA = struct();
    TANOVA.config = cfg;
    TANOVA.results = results_table;
    TANOVA.bayesian_results = bayesian_table;
    TANOVA.participant_template_manifest = D.manifest;
    TANOVA.data = rmfield_if_present(D, {'X'});
    TANOVA.notes = notes(:);
    TANOVA.results_csv = results_csv;
    TANOVA.bayesian_results_csv = bayesian_csv;
    TANOVA.manifest_csv = manifest_csv;
    TANOVA.created = datestr(now, 30);

    save(fullfile(cfg.output_dir, 'tanova_summary.mat'), 'TANOVA', '-v7.3');
    write_methods_notes(fullfile(cfg.output_dir, 'tanova_methods_notes.txt'), TANOVA, H);

    if cfg.save_plots
        plot_pmax_heatmap(results_table, plot_dir, cfg);
        plot_descriptive_mean_maps(D, H, plot_dir, cfg);
    end

    if cfg.verbose
        fprintf('\nDone.\n');
        fprintf('Results:  %s\n', results_csv);
        if ~isempty(bayesian_csv)
            fprintf('Bayesian: %s\n', bayesian_csv);
        end
        fprintf('Manifest: %s\n', manifest_csv);
    end
end

% ======================================================================
% Data extraction
% ======================================================================

function D = extract_participant_template_data(H, cfg, grouping)
    nodes = [];
    if isfield(H,'peak_records') && ~isempty(H.peak_records) && isfield(H,'manifest') && ~isempty(H.manifest)
        P = extract_fixed_reference_peak_distributions(H, cfg.semiology_manifest_csv, grouping);
        nodes = repmat(struct('participant',"",'group',"",'condition',"",'name',"", ...
            'event_type',"",'semiology',"",'level',"fixed_global_backfit",'centers',[],'n_maps',0,'inherited',false),numel(P.participant),1);
        for i = 1:numel(nodes)
            nodes(i).participant=P.participant(i); nodes(i).group=P.group(i);
            nodes(i).condition=P.condition(i); nodes(i).name=P.participant(i)+"_"+P.condition(i);
            nodes(i).event_type=P.event_type(i);
            nodes(i).semiology=P.semiology(i);
            nodes(i).centers=squeeze(P.mu(i,:,:)); nodes(i).n_maps=sum(P.n_maps(i,:));
        end
    elseif isfield(H, 'participant_conditions') && ~isempty(H.participant_conditions) && nodes_have_nonempty_field(H.participant_conditions, 'condition')
        nodes = H.participant_conditions;
    elseif isfield(H, 'participants') && ~isempty(H.participants)
        nodes = H.participants;
    elseif isfield(H, 'participant_conditions') && ~isempty(H.participant_conditions)
        nodes = H.participant_conditions;
    else
        error('HResults does not contain participants or participant_conditions.');
    end

    if isfield(H, 'selected_K') && ~isempty(H.selected_K)
        K = double(H.selected_K);
    elseif isfield(H, 'global') && isfield(H.global, 'centers')
        K = size(H.global.centers, 1);
    else
        K = size(nodes(1).centers, 1);
    end

    if isfield(H, 'global') && isfield(H.global, 'centers') && ~isempty(H.global.centers)
        global_ref = double(H.global.centers);
    else
        global_ref = [];
    end
    if isempty(global_ref)
        global_ref = nodes(1).centers;
    end
    global_ref = normalise_maps_2d(global_ref);

    rows = 0;
    X = [];
    participant = strings(0, 1);
    group = strings(0, 1);
    condition = strings(0, 1);
    semiology = strings(0, 1);
    event_type = strings(0, 1);
    node_name = strings(0, 1);
    node_level = strings(0, 1);
    inherited = false(0, 1);
    n_maps = nan(0, 1);

    for i = 1:numel(nodes)
        node = nodes(i);
        if ~isfield(node, 'centers') || isempty(node.centers)
            continue;
        end
        is_inherited = isfield(node, 'inherited') && logical(node.inherited);
        if cfg.exclude_inherited_nodes && is_inherited
            continue;
        end
        centers = double(node.centers);
        if size(centers, 1) < K
            warning('Skipping node %d because it has fewer template rows than selected K.', i);
            continue;
        end
        centers = centers(1:K, :);
        centers = sign_align_and_normalise_centers(centers, global_ref);

        rows = rows + 1;
        if isempty(X)
            X = nan(0, K, size(centers, 2));
        end
        X(rows, :, :) = centers;
        participant(rows, 1) = string(get_field_or(node, 'participant', ''));
        group(rows, 1) = string(get_field_or(node, 'group', ''));
        condition(rows, 1) = string(get_field_or(node, 'condition', ''));
        semiology(rows, 1) = string(get_field_or(node, 'semiology', ''));
        event_type(rows, 1) = string(get_field_or(node, 'event_type', ''));
        node_name(rows, 1) = string(get_field_or(node, 'name', sprintf('node_%d', i)));
        node_level(rows, 1) = string(get_field_or(node, 'level', 'participant'));
        inherited(rows, 1) = is_inherited;
        n_maps(rows, 1) = double(get_field_or(node, 'n_maps', NaN));
    end

    if isempty(X)
        D = struct('X', [], 'K', K, 'n_channels', NaN);
        return;
    end

    participant(participant == "") = "unknown_participant";
    has_group = valid_factor_present(group);
    has_condition = valid_factor_present(condition);

    if ~has_group
        group(:) = "";
    end
    if ~has_condition
        condition(:) = "";
    end

    unit = participant;
    if has_group && participant_factor_varies(participant, group)
        warning(['At least one participant appears in more than one group. ', ...
                 'Group is therefore treated as participant-by-group unit for randomisation. ', ...
                 'This is only defensible if those rows are genuinely independent.']);
        unit = participant + "__" + group;
    end

    stratum = strings(size(participant));
    if grouping == "event_type", stratum = event_type; end
    if grouping == "semiology", stratum = event_type + "||" + semiology; end
    event_type_before_merge = event_type; semiology_before_merge = semiology;
    [X, participant, group, condition, unit, node_name, node_level, inherited, n_maps, representative] = ...
        merge_duplicate_design_rows(X, participant, group, condition, unit, node_name, node_level, inherited, n_maps, global_ref, stratum);
    event_type = event_type_before_merge(representative);
    semiology = semiology_before_merge(representative);

    manifest = table(cellstr(participant), cellstr(unit), cellstr(group), cellstr(condition), cellstr(event_type), cellstr(semiology), ...
        cellstr(node_name), cellstr(node_level), inherited, n_maps, ...
        'VariableNames', {'participant', 'unit', 'group', 'condition', 'event_type', 'semiology', 'node_name', 'node_level', 'inherited', 'n_maps'});

    D = struct();
    D.X = X;
    D.participant = participant;
    D.unit = unit;
    D.group = group;
    D.condition = condition;
    D.semiology = semiology;
    D.event_type = event_type;
    D.K = size(X, 2);
    D.n_channels = size(X, 3);
    D.global_ref = global_ref;
    D.has_group = has_group && numel(unique(group(group ~= ""))) >= 2;
    D.has_condition = has_condition && numel(unique(condition(condition ~= ""))) >= 2;
    D.manifest = manifest;
end

function H = build_hresults_from_mresults(MResults)
    if isfield(MResults, 'HResults') && isstruct(MResults.HResults)
        H = MResults.HResults;
        return;
    end
    if ~isfield(MResults, 'manifest') || ~isfield(MResults, 'file_fits') || ~isfield(MResults, 'meta_fit')
        error('MResults does not contain enough information to build HResults.');
    end

    manifest = MResults.manifest;
    global_ref = MResults.meta_fit.centers;
    K = size(global_ref, 1);

    H = struct();
    H.source = 'metamicrostate_dataset_pipeline';
    H.created = datestr(now, 30);
    H.manifest = manifest;
    H.global = struct('name', 'global', 'level', 'global', 'centers', global_ref, 'n_maps', NaN, 'inherited', false);

    n_files = numel(MResults.file_fits);
    nodes = repmat(struct('participant', '', 'group', '', 'condition', '', 'name', '', 'level', 'participant_condition', 'centers', [], 'n_maps', NaN, 'inherited', false), n_files, 1);
    for i = 1:n_files
        fit = MResults.file_fits{i};
        if isempty(fit) || ~isstruct(fit) || ~isfield(fit, 'centers')
            centers = global_ref;
            n_maps = NaN;
        else
            centers = coerce_centers_to_k(fit.centers, K, global_ref);
            centers = sign_align_and_normalise_centers(centers, global_ref);
            n_maps = get_struct_field_or_default(fit, 'n_maps', NaN);
        end
        nodes(i).participant = char(string(manifest.participant{i}));
        nodes(i).group = char(string(manifest.group{i}));
        nodes(i).condition = char(string(manifest.condition{i}));
        nodes(i).name = sprintf('%s__%s__%s', nodes(i).participant, nodes(i).group, nodes(i).condition);
        nodes(i).centers = centers;
        nodes(i).n_maps = n_maps;
    end

    H.files = nodes;
    H.participants = nodes;
    H.participant_conditions = nodes;
    H.group_conditions = build_group_condition_nodes(nodes, manifest, global_ref);
    H.common_channel_labels = get_struct_field_or_default(MResults, 'common_labels', {});
    H.common_chanlocs = get_struct_field_or_default(MResults, 'common_chanlocs', []);
    H.common_pos = get_struct_field_or_default(MResults, 'common_pos', []);
end

function centers = coerce_centers_to_k(centers, K, fallback)
    centers = double(centers);
    if isempty(centers)
        centers = fallback;
        return;
    end
    if size(centers, 1) >= K
        centers = centers(1:K, :);
    else
        pad = fallback(size(centers, 1)+1:K, :);
        centers = [centers; pad];
    end
end

function nodes = build_group_condition_nodes(file_nodes, manifest, global_ref)
    nodes = repmat(struct('participant', '', 'group', '', 'condition', '', 'name', '', 'level', 'group_condition', 'centers', [], 'n_maps', NaN, 'inherited', false), 0, 1);
    group_levels = unique_nonempty_strings(manifest.group);
    condition_levels = unique_nonempty_strings(manifest.condition);
    idx = 0;
    for g = 1:numel(group_levels)
        for c = 1:numel(condition_levels)
            mask = string(manifest.group) == group_levels(g) & string(manifest.condition) == condition_levels(c);
            if ~any(mask)
                continue;
            end
            idx = idx + 1;
            centers = cat(3, file_nodes(mask).centers);
            centers = mean(centers, 3, 'omitnan');
            centers = coerce_centers_to_k(centers, size(global_ref, 1), global_ref);
            centers = sign_align_and_normalise_centers(centers, global_ref);
            nodes(idx, 1) = struct( ...
                'participant', '', ...
                'group', char(group_levels(g)), ...
                'condition', char(condition_levels(c)), ...
                'name', sprintf('%s__%s', char(group_levels(g)), char(condition_levels(c))), ...
                'level', 'group_condition', ...
                'centers', centers, ...
                'n_maps', sum([file_nodes(mask).n_maps]), ...
                'inherited', false);
        end
    end
end

function value = get_struct_field_or_default(S, field_name, default_value)
    if isstruct(S) && isfield(S, field_name)
        value = S.(field_name);
    else
        value = default_value;
    end
end

function vals = unique_nonempty_strings(x)
    vals = unique(strtrim(string(x)));
    vals = vals(strlength(vals) > 0 & ~ismissing(vals));
    vals = vals(:)';
end

function centers = sign_align_and_normalise_centers(centers, ref)
    centers = normalise_maps_2d(centers);
    ref = normalise_maps_2d(ref);
    K = min(size(centers, 1), size(ref, 1));
    for k = 1:K
        if all(isfinite(centers(k, :))) && all(isfinite(ref(k, :)))
            if dot(centers(k, :), ref(k, :)) < 0
                centers(k, :) = -centers(k, :);
            end
        end
    end
end

function X = normalise_maps_2d(X)
    X = double(X);
    if isempty(X)
        return;
    end
    X = X - mean(X, 2, 'omitnan');
    denom = sqrt(mean(X .^ 2, 2, 'omitnan'));
    denom(~isfinite(denom) | denom <= eps) = 1;
    X = X ./ denom;
end

function X = normalise_x3(X)
    for i = 1:size(X, 1)
        for k = 1:size(X, 2)
            v = squeeze(X(i, k, :))';
            v = normalise_maps_2d(v);
            X(i, k, :) = v;
        end
    end
end

function [X2, participant2, group2, condition2, unit2, node_name2, node_level2, inherited2, n_maps2, representative] = ...
    merge_duplicate_design_rows(X, participant, group, condition, unit, node_name, node_level, inherited, n_maps, ref, stratum)

    keys = unit + "||" + group + "||" + condition + "||" + stratum;
    [ukeys, ia] = unique(keys, 'stable');
    representative = ia;
    n = numel(ukeys);
    X2 = nan(n, size(X, 2), size(X, 3));
    participant2 = participant(ia);
    group2 = group(ia);
    condition2 = condition(ia);
    unit2 = unit(ia);
    node_name2 = node_name(ia);
    node_level2 = node_level(ia);
    inherited2 = false(n, 1);
    n_maps2 = nan(n, 1);

    for i = 1:n
        idx = find(keys == ukeys(i));
        M = squeeze(mean(X(idx, :, :), 1, 'omitnan'));
        M = sign_align_and_normalise_centers(M, ref);
        X2(i, :, :) = M;
        inherited2(i) = any(inherited(idx));
        n_maps2(i) = sum(n_maps(idx), 'omitnan');
        if numel(idx) > 1
            node_name2(i) = node_name2(i) + "__merged_" + string(numel(idx));
        end
    end
end

% ======================================================================
% Omnibus and post-hoc tests
% ======================================================================

function R = run_group_test(D, cfg, effect_name, mask, test_kind)
    note = '';
    D2 = subset_D(D, mask);
    if D2.has_condition && cfg.require_complete_within_subject
        complete_units = units_with_all_conditions(D2.unit, D2.condition);
        keep = ismember(D2.unit, complete_units);
        D2 = subset_D(D2, keep);
        note = sprintf('Group test uses condition-complete units only: %d units retained.', numel(unique(D2.unit)));
    end

    if isempty(D2.X) || numel(unique(D2.group(D2.group ~= ""))) < 2
        R = empty_result(effect_name, test_kind, D.K, 'Insufficient group levels.');
        return;
    end

    [Xu, group_u, unit_u] = collapse_to_unit(D2.X, D2.unit, D2.group, D2.global_ref);
    if size(Xu, 1) < 3 || numel(unique(group_u)) < 2
        R = empty_result(effect_name, test_kind, D.K, 'Insufficient independent units after collapse.');
        return;
    end

    stat_fun = @(labels) stat_main_effect(Xu, labels, cfg.renormalise_factor_means);
    obs = stat_fun(group_u);
    perm = nan(cfg.n_permutations, D.K);
    for b = 1:cfg.n_permutations
        perm_labels = group_u(randperm(numel(group_u)));
        perm(b, :) = stat_fun(perm_labels);
    end
    R = finish_result(effect_name, test_kind, 'between_subject_group_label_permutation', obs, perm, ...
        unique(group_u), numel(unit_u), size(Xu, 1), note);
end

function R = run_condition_test(D, cfg, effect_name, mask, test_kind)
    D2 = subset_D(D, mask);
    if isempty(D2.X) || numel(unique(D2.condition(D2.condition ~= ""))) < 2
        R = empty_result(effect_name, test_kind, D.K, 'Insufficient condition levels.');
        return;
    end

    if cfg.require_complete_within_subject
        complete_units = units_with_all_conditions(D2.unit, D2.condition);
        keep = ismember(D2.unit, complete_units);
        D2 = subset_D(D2, keep);
        note = sprintf('Condition test uses condition-complete units only: %d units retained.', numel(unique(D2.unit)));
    else
        keep_units = units_with_at_least_n_conditions(D2.unit, D2.condition, 2);
        keep = ismember(D2.unit, keep_units);
        D2 = subset_D(D2, keep);
        note = sprintf('Condition test uses units with at least two conditions: %d units retained.', numel(unique(D2.unit)));
    end

    if isempty(D2.X) || numel(unique(D2.condition(D2.condition ~= ""))) < 2
        R = empty_result(effect_name, test_kind, D.K, 'Insufficient within-unit condition data after filtering.');
        return;
    end

    stat_fun = @(labels) stat_main_effect(D2.X, labels, cfg.renormalise_factor_means);
    obs = stat_fun(D2.condition);
    perm = nan(cfg.n_permutations, D.K);
    for b = 1:cfg.n_permutations
        perm_labels = permute_labels_within_units(D2.condition, D2.unit);
        perm(b, :) = stat_fun(perm_labels);
    end
    R = finish_result(effect_name, test_kind, 'within_subject_condition_label_permutation', obs, perm, ...
        unique(D2.condition), numel(unique(D2.unit)), size(D2.X, 1), note);
end

function R = run_interaction_test(D, cfg, effect_name, mask, test_kind)
    D2 = subset_D(D, mask);
    if isempty(D2.X) || numel(unique(D2.group(D2.group ~= ""))) < 2 || numel(unique(D2.condition(D2.condition ~= ""))) < 2
        R = empty_result(effect_name, test_kind, D.K, 'Insufficient group and/or condition levels.');
        return;
    end

    if cfg.require_complete_within_subject
        complete_units = units_with_all_conditions(D2.unit, D2.condition);
        keep = ismember(D2.unit, complete_units);
        D2 = subset_D(D2, keep);
        note = sprintf('Interaction test uses condition-complete units only: %d units retained.', numel(unique(D2.unit)));
    else
        note = 'Interaction test allows incomplete within-unit condition data; interpret cautiously.';
    end

    if isempty(D2.X) || ~all_cells_present(D2.group, D2.condition)
        R = empty_result(effect_name, test_kind, D.K, 'At least one group-by-condition cell is empty after filtering.');
        return;
    end

    stat_fun = @(cond_labels) stat_interaction(D2.X, D2.group, cond_labels, cfg.renormalise_factor_means);
    obs = stat_fun(D2.condition);
    perm = nan(cfg.n_permutations, D.K);
    for b = 1:cfg.n_permutations
        perm_cond = permute_labels_within_units(D2.condition, D2.unit);
        perm(b, :) = stat_fun(perm_cond);
    end
    R = finish_result(effect_name, test_kind, 'within_subject_condition_label_permutation_group_fixed', obs, perm, ...
        unique(D2.group + " x " + D2.condition), numel(unique(D2.unit)), size(D2.X, 1), note);
end

function R = run_mixed_event_test(D, cfg, effect_name, test_kind)
    if isempty(D.X) || numel(unique(D.condition)) < 2
        R = empty_result(effect_name,test_kind,D.K,'Insufficient event-type levels.'); return;
    end
    stat_fun=@(labels)stat_main_effect(D.X,labels,cfg.renormalise_factor_means);
    obs=stat_fun(D.condition); perm=nan(cfg.n_permutations,D.K);
    for b=1:cfg.n_permutations, perm(b,:)=stat_fun(hybrid_permute_labels(D.condition,D.unit)); end
    R=finish_result(effect_name,test_kind,'paired_swaps_plus_unpaired_participant_label_permutation', ...
        obs,perm,unique(D.condition),numel(unique(D.unit)),size(D.X,1), ...
        'DD participants contribute paired event types; single-diagnosis participants contribute one event type.');
end

function [tables, notes] = run_limited_planned_tests(D, D_event, D_semiology, cfg, bayesian)
    [sets, names, kinds] = planned_comparison_sets(D, D_event, D_semiology);
    tables = cell(numel(sets), 1); notes = cell(numel(sets), 1);
    for i = 1:numel(sets)
        S = sets{i}; mask = true(size(S.unit)); test_kind = sprintf('planned_%02d', i);
        if bayesian
            if kinds(i) == "group"
                tables{i} = bayesian_group_test(S, cfg, names(i), mask, "bayesian_" + test_kind);
            elseif kinds(i) == "mixed_event"
                tables{i} = bayesian_mixed_event_test(S, cfg, names(i), "bayesian_" + test_kind);
            else
                tables{i} = bayesian_condition_test(S, cfg, names(i), mask, "bayesian_" + test_kind);
            end
            notes{i} = '';
        else
            if kinds(i) == "group"
                R = run_group_test(S, cfg, names(i), mask, test_kind);
            elseif kinds(i) == "mixed_event"
                R = run_mixed_event_test(S, cfg, names(i), test_kind);
            else
                R = run_condition_test(S, cfg, names(i), mask, test_kind);
            end
            tables{i} = result_to_table(R); notes{i} = R.note;
        end
    end
end

function [sets, names, kinds] = planned_comparison_sets(D, E, S)
    preict = @(x) ismember(lower(x), ["preict", "preictal"]);
    postict = @(x) ismember(lower(x), ["postict", "postictal"]);
    baseline = @(x) lower(x) == "ii";
    sets = cell(10,1); kinds = ["group";"group";"group";"mixed_event";repmat("condition",6,1)];
    names = ["People: ES only vs FS only"; "People: DD vs FS only"; "People: DD vs ES only"; ...
        "Events: Preict functional seizure vs preict epileptic seizure"; ...
        "Functional events (FS or DD): II vs Preict"; "Epileptic events (ES or DD): II vs Preict"; ...
        "Within DD: Preict epileptic vs preict functional event"; ...
        "Epileptic events (ES or DD): Preict vs Postict"; ...
        "Functional events (FS or DD): Preict vs Postict"; ...
        "Functional events (FS or DD): Preict Hypomotor vs Preict Hypermotor"];

    sets{1} = subset_D(D, ismember(D.group,["ES","FS"]));
    sets{2} = subset_D(D, ismember(D.group,["DD","FS"]));
    sets{3} = subset_D(D, ismember(D.group,["DD","ES"]));
    for i=1:3, sets{i}.condition(:)=""; sets{i}.has_condition=false; end
    sets{4} = subset_D(E, preict(E.condition) & ismember(E.event_type,["NEAD","ES"]));
    sets{4}.condition = sets{4}.event_type;
    sets{5} = subset_D(E, ismember(E.group,["FS","DD"]) & (baseline(E.condition) | (preict(E.condition) & E.event_type=="NEAD")));
    sets{6} = subset_D(E, ismember(E.group,["ES","DD"]) & (baseline(E.condition) | (preict(E.condition) & E.event_type=="ES")));
    sets{7} = subset_D(E, E.group == "DD" & preict(E.condition) & ismember(E.event_type,["ES","NEAD"]));
    sets{7}.condition = sets{7}.event_type;
    sets{8} = subset_D(E, ismember(E.group,["ES","DD"]) & ismember(E.event_type,["ES",""]) & (preict(E.condition) | postict(E.condition)));
    sets{8} = subset_D(sets{8}, sets{8}.event_type=="ES");
    sets{9} = subset_D(E, ismember(E.group,["FS","DD"]) & (preict(E.condition) | postict(E.condition)) & E.event_type=="NEAD");
    sets{10} = subset_D(S, ismember(S.group,["FS","DD"]) & preict(S.condition) & S.event_type=="NEAD" & ismember(S.semiology,["hypo","hyper"]));
    sets{10}.condition(sets{10}.semiology == "hypo") = "Hypomotor";
    sets{10}.condition(sets{10}.semiology == "hyper") = "Hypermotor";
end

function posthoc_tables = run_posthoc_tests(D, cfg)
    posthoc_tables = {};

    if D.has_group
        gl = unique(D.group(D.group ~= ""));
        for i = 1:numel(gl)
            for j = i+1:numel(gl)
                mask = D.group == gl(i) | D.group == gl(j);
                name = sprintf('group_pair_%s_vs_%s', safe_label(gl(i)), safe_label(gl(j)));
                R = run_group_test(D, cfg, name, mask, 'posthoc_group_pairwise');
                posthoc_tables{end+1} = result_to_table(R);
            end
        end
    end

    if D.has_condition
        cl = unique(D.condition(D.condition ~= ""));
        for i = 1:numel(cl)
            for j = i+1:numel(cl)
                mask = D.condition == cl(i) | D.condition == cl(j);
                name = sprintf('condition_pair_%s_vs_%s', safe_label(cl(i)), safe_label(cl(j)));
                R = run_condition_test(D, cfg, name, mask, 'posthoc_condition_pairwise');
                posthoc_tables{end+1} = result_to_table(R);
            end
        end
    end

    if D.has_group && D.has_condition
        gl = unique(D.group(D.group ~= ""));
        cl = unique(D.condition(D.condition ~= ""));
        for i = 1:numel(gl)
            mask = D.group == gl(i);
            name = sprintf('condition_simple_effect_within_%s', safe_label(gl(i)));
            R = run_condition_test(D, cfg, name, mask, 'posthoc_simple_condition_within_group');
            posthoc_tables{end+1} = result_to_table(R);
        end
        for i = 1:numel(cl)
            mask = D.condition == cl(i);
            name = sprintf('group_simple_effect_at_%s', safe_label(cl(i)));
            R = run_group_test(D, cfg, name, mask, 'posthoc_simple_group_within_condition');
            posthoc_tables{end+1} = result_to_table(R);
        end

        pairs = planned_condition_pairs(D.condition);
        for g = 1:numel(gl)
            for p = 1:size(pairs, 1)
                pair_mask = ismember(D.condition, pairs(p, :));
                name = sprintf('planned_%s_vs_%s_within_%s', safe_label(pairs(p,1)), ...
                    safe_label(pairs(p,2)), safe_label(gl(g)));
                R = run_condition_test(D, cfg, name, D.group == gl(g) & pair_mask, 'planned_condition_within_event_type');
                posthoc_tables{end+1} = result_to_table(R);
                semiology_levels = unique(D.semiology(D.group == gl(g) & D.semiology ~= ""));
                for s = 1:numel(semiology_levels)
                    sem_name = sprintf('%s_%s', name, safe_label(semiology_levels(s)));
                    mask = D.group == gl(g) & D.semiology == semiology_levels(s) & pair_mask;
                    R = run_condition_test(D, cfg, sem_name, mask, 'planned_condition_within_event_type_and_semiology');
                    posthoc_tables{end+1} = result_to_table(R);
                end
            end
        end
    end
end

function pairs = planned_condition_pairs(conditions)
    levels = unique(string(conditions(conditions ~= "")), 'stable');
    ii = find_condition_level(levels, ["ii"]);
    preictal = find_condition_level(levels, ["preict", "preictal"]);
    postictal = find_condition_level(levels, ["postict", "postictal"]);
    pairs = strings(0, 2);
    if ii ~= "" && preictal ~= "", pairs(end+1, :) = [ii preictal]; end
    if preictal ~= "" && postictal ~= "", pairs(end+1, :) = [preictal postictal]; end
end

function level = find_condition_level(levels, aliases)
    hit = find(ismember(lower(levels), lower(string(aliases))), 1);
    if isempty(hit), level = ""; else, level = levels(hit); end
end

% ======================================================================
% Bayesian bootstrap TANOVA
% ======================================================================

function T = run_bayesian_tanova_tests(D, cfg)
    tables = {};

    if D.has_group
        tables{end+1} = bayesian_group_test(D, cfg, 'group', true(size(D.unit)), 'bayesian_omnibus_group');
    end
    if D.has_condition
        tables{end+1} = bayesian_condition_test(D, cfg, 'condition', true(size(D.unit)), 'bayesian_omnibus_condition');
    end
    if D.has_group && D.has_condition
        tables{end+1} = bayesian_interaction_test(D, cfg, 'group_x_condition', true(size(D.unit)), 'bayesian_omnibus_interaction');
    end

    if cfg.run_posthoc
        if D.has_group
            gl = unique(D.group(D.group ~= ""));
            for i = 1:numel(gl)
                for j = i+1:numel(gl)
                    mask = D.group == gl(i) | D.group == gl(j);
                    name = sprintf('group_pair_%s_vs_%s', safe_label(gl(i)), safe_label(gl(j)));
                    tables{end+1} = bayesian_group_test(D, cfg, name, mask, 'bayesian_posthoc_group_pairwise');
                end
            end
        end
        if D.has_condition
            cl = unique(D.condition(D.condition ~= ""));
            for i = 1:numel(cl)
                for j = i+1:numel(cl)
                    mask = D.condition == cl(i) | D.condition == cl(j);
                    name = sprintf('condition_pair_%s_vs_%s', safe_label(cl(i)), safe_label(cl(j)));
                    tables{end+1} = bayesian_condition_test(D, cfg, name, mask, 'bayesian_posthoc_condition_pairwise');
                end
            end
        end
        if D.has_group && D.has_condition
            gl = unique(D.group(D.group ~= ""));
            cl = unique(D.condition(D.condition ~= ""));
            for i = 1:numel(gl)
                mask = D.group == gl(i);
                name = sprintf('condition_simple_effect_within_%s', safe_label(gl(i)));
                tables{end+1} = bayesian_condition_test(D, cfg, name, mask, 'bayesian_simple_condition_within_group');
            end
            for i = 1:numel(cl)
                mask = D.condition == cl(i);
                name = sprintf('group_simple_effect_at_%s', safe_label(cl(i)));
                tables{end+1} = bayesian_group_test(D, cfg, name, mask, 'bayesian_simple_group_within_condition');
            end
            pairs = planned_condition_pairs(D.condition);
            for g = 1:numel(gl)
                for p = 1:size(pairs, 1)
                    pair_mask = ismember(D.condition, pairs(p, :));
                    name = sprintf('planned_%s_vs_%s_within_%s', safe_label(pairs(p,1)), ...
                        safe_label(pairs(p,2)), safe_label(gl(g)));
                    mask = D.group == gl(g) & pair_mask;
                    tables{end+1} = bayesian_condition_test(D, cfg, name, mask, 'bayesian_planned_condition_within_event_type');
                    semiology_levels = unique(D.semiology(D.group == gl(g) & D.semiology ~= ""));
                    for s = 1:numel(semiology_levels)
                        sem_name = sprintf('%s_%s', name, safe_label(semiology_levels(s)));
                        mask = D.group == gl(g) & D.semiology == semiology_levels(s) & pair_mask;
                        tables{end+1} = bayesian_condition_test(D, cfg, sem_name, mask, ...
                            'bayesian_planned_condition_within_event_type_and_semiology');
                    end
                end
            end
        end
    end

    T = vertcat_nonempty_tables(tables);
    if isempty(T)
        T = empty_bayesian_table();
    end
end

function T = bayesian_group_test(D, cfg, effect_name, mask, test_kind)
    note = '';
    D2 = subset_D(D, mask);
    if D2.has_condition && cfg.require_complete_within_subject
        complete_units = units_with_all_conditions(D2.unit, D2.condition);
        D2 = subset_D(D2, ismember(D2.unit, complete_units));
        note = sprintf('Bayesian group test uses condition-complete units only: %d units retained.', numel(unique(D2.unit)));
    end
    if isempty(D2.X) || numel(unique(D2.group(D2.group ~= ""))) < 2
        T = empty_bayesian_result(effect_name, test_kind, D.K, 'Insufficient group levels.', cfg);
        return;
    end

    [Xu, group_u, unit_u] = collapse_to_unit(D2.X, D2.unit, D2.group, D2.global_ref);
    if size(Xu, 1) < 3 || numel(unique(group_u)) < 2
        T = empty_bayesian_result(effect_name, test_kind, D.K, 'Insufficient independent units after collapse.', cfg);
        return;
    end

    obs = stat_main_effect(Xu, group_u, cfg.renormalise_factor_means);
    draws = nan(cfg.n_posterior, D.K);
    for b = 1:cfg.n_posterior
        row_weights = bayesian_group_row_weights(group_u);
        draws(b, :) = stat_main_effect_weighted(Xu, group_u, row_weights, cfg.renormalise_factor_means);
    end
    T = bayesian_result_to_table(effect_name, test_kind, 'bayesian_bootstrap_between_group', obs, draws, ...
        unique(group_u), numel(unit_u), size(Xu, 1), cfg, note);
end

function T = bayesian_mixed_event_test(D,cfg,effect_name,test_kind)
    if isempty(D.X) || numel(unique(D.condition)) < 2
        T=empty_bayesian_result(effect_name,test_kind,D.K,'Insufficient event-type levels.',cfg); return;
    end
    obs=stat_main_effect(D.X,D.condition,cfg.renormalise_factor_means);
    draws=nan(cfg.n_posterior,D.K); units=unique(D.unit,'stable');
    for b=1:cfg.n_posterior
        weights=equal_unit_row_weights(D.unit,units,draw_dirichlet(numel(units)));
        draws(b,:)=stat_main_effect_weighted(D.X,D.condition,weights,cfg.renormalise_factor_means);
    end
    T=bayesian_result_to_table(effect_name,test_kind,'bayesian_bootstrap_participant_clustered_mixed_event', ...
        obs,draws,unique(D.condition),numel(units),size(D.X,1),cfg, ...
        'Each participant has total weight one, split across their event types.');
end

function T = bayesian_condition_test(D, cfg, effect_name, mask, test_kind)
    D2 = subset_D(D, mask);
    if cfg.require_complete_within_subject
        complete_units = units_with_all_conditions(D2.unit, D2.condition);
        D2 = subset_D(D2, ismember(D2.unit, complete_units));
        note = sprintf('Bayesian condition test uses condition-complete units only: %d units retained.', numel(unique(D2.unit)));
    else
        keep_units = units_with_at_least_n_conditions(D2.unit, D2.condition, 2);
        D2 = subset_D(D2, ismember(D2.unit, keep_units));
        note = sprintf('Bayesian condition test uses units with at least two conditions: %d units retained.', numel(unique(D2.unit)));
    end
    if isempty(D2.X) || numel(unique(D2.condition(D2.condition ~= ""))) < 2
        T = empty_bayesian_result(effect_name, test_kind, D.K, 'Insufficient within-unit condition data after filtering.', cfg);
        return;
    end

    obs = stat_main_effect(D2.X, D2.condition, cfg.renormalise_factor_means);
    draws = nan(cfg.n_posterior, D.K);
    units = unique(D2.unit, 'stable');
    for b = 1:cfg.n_posterior
        row_weights = row_weights_from_unit_weights(D2.unit, units, draw_dirichlet(numel(units)));
        draws(b, :) = stat_main_effect_weighted(D2.X, D2.condition, row_weights, cfg.renormalise_factor_means);
    end
    T = bayesian_result_to_table(effect_name, test_kind, 'bayesian_bootstrap_within_unit_condition', obs, draws, ...
        unique(D2.condition), numel(units), size(D2.X, 1), cfg, note);
end

function T = bayesian_interaction_test(D, cfg, effect_name, mask, test_kind)
    D2 = subset_D(D, mask);
    if cfg.require_complete_within_subject
        complete_units = units_with_all_conditions(D2.unit, D2.condition);
        D2 = subset_D(D2, ismember(D2.unit, complete_units));
        note = sprintf('Bayesian interaction test uses condition-complete units only: %d units retained.', numel(unique(D2.unit)));
    else
        note = 'Bayesian interaction test allows incomplete within-unit condition data; interpret cautiously.';
    end
    if isempty(D2.X) || numel(unique(D2.group(D2.group ~= ""))) < 2 || numel(unique(D2.condition(D2.condition ~= ""))) < 2 || ~all_cells_present(D2.group, D2.condition)
        T = empty_bayesian_result(effect_name, test_kind, D.K, 'Insufficient group-by-condition cells.', cfg);
        return;
    end

    obs = stat_interaction(D2.X, D2.group, D2.condition, cfg.renormalise_factor_means);
    draws = nan(cfg.n_posterior, D.K);
    for b = 1:cfg.n_posterior
        row_weights = bayesian_grouped_unit_row_weights(D2.unit, D2.group);
        draws(b, :) = stat_interaction_weighted(D2.X, D2.group, D2.condition, row_weights, cfg.renormalise_factor_means);
    end
    T = bayesian_result_to_table(effect_name, test_kind, 'bayesian_bootstrap_grouped_units', obs, draws, ...
        unique(D2.group + " x " + D2.condition), numel(unique(D2.unit)), size(D2.X, 1), cfg, note);
end

% ======================================================================
% Test statistics and permutation utilities
% ======================================================================

function stat = stat_main_effect(X, labels, renorm_level_means)
    labels = string(labels(:));
    levels = unique(labels(labels ~= ""), 'stable');
    K = size(X, 2);
    C = size(X, 3);
    M = nan(numel(levels), K, C);
    for l = 1:numel(levels)
        idx = labels == levels(l);
        M(l, :, :) = squeeze(mean(X(idx, :, :), 1, 'omitnan'));
    end
    if renorm_level_means
        M = normalise_level_maps(M);
    end
    G = squeeze(mean(M, 1, 'omitnan'));
    stat = nan(1, K);
    for k = 1:K
        A = squeeze(M(:, k, :));
        g = G(k, :);
        D = A - repmat(g, size(A, 1), 1);
        stat(k) = sqrt(mean(D(:) .^ 2, 'omitnan'));
    end
end

function stat = stat_interaction(X, group_labels, condition_labels, renorm_level_means)
    group_labels = string(group_labels(:));
    condition_labels = string(condition_labels(:));
    groups = unique(group_labels(group_labels ~= ""), 'stable');
    conds = unique(condition_labels(condition_labels ~= ""), 'stable');
    K = size(X, 2);
    C = size(X, 3);
    Cell = nan(numel(groups), numel(conds), K, C);

    for g = 1:numel(groups)
        for c = 1:numel(conds)
            idx = group_labels == groups(g) & condition_labels == conds(c);
            if any(idx)
                Cell(g, c, :, :) = squeeze(mean(X(idx, :, :), 1, 'omitnan'));
            end
        end
    end
    if renorm_level_means
        for g = 1:numel(groups)
            for c = 1:numel(conds)
                M = squeeze(Cell(g, c, :, :));
                Cell(g, c, :, :) = normalise_maps_2d(M);
            end
        end
    end

    row_mean = squeeze(mean(Cell, 2, 'omitnan'));
    col_mean = squeeze(mean(Cell, 1, 'omitnan'));
    grand = squeeze(mean(mean(Cell, 1, 'omitnan'), 2, 'omitnan'));

    stat = nan(1, K);
    for k = 1:K
        residuals = [];
        for g = 1:numel(groups)
            for c = 1:numel(conds)
                cell_map = squeeze(Cell(g, c, k, :))';
                if any(~isfinite(cell_map))
                    continue;
                end
                r = cell_map - squeeze(row_mean(g, k, :))' - squeeze(col_mean(c, k, :))' + squeeze(grand(k, :));
                residuals = [residuals, r];
            end
        end
        stat(k) = sqrt(mean(residuals .^ 2, 'omitnan'));
    end
end

function stat = stat_main_effect_weighted(X, labels, row_weights, renorm_level_means)
    labels = string(labels(:));
    row_weights = double(row_weights(:));
    levels = unique(labels(labels ~= ""), 'stable');
    K = size(X, 2);
    C = size(X, 3);
    M = nan(numel(levels), K, C);
    for l = 1:numel(levels)
        idx = labels == levels(l);
        M(l, :, :) = weighted_mean_maps(X(idx, :, :), row_weights(idx));
    end
    if renorm_level_means
        M = normalise_level_maps(M);
    end
    G = squeeze(mean(M, 1, 'omitnan'));
    stat = nan(1, K);
    for k = 1:K
        A = squeeze(M(:, k, :));
        g = G(k, :);
        D = A - repmat(g, size(A, 1), 1);
        stat(k) = sqrt(mean(D(:) .^ 2, 'omitnan'));
    end
end

function stat = stat_interaction_weighted(X, group_labels, condition_labels, row_weights, renorm_level_means)
    group_labels = string(group_labels(:));
    condition_labels = string(condition_labels(:));
    row_weights = double(row_weights(:));
    groups = unique(group_labels(group_labels ~= ""), 'stable');
    conds = unique(condition_labels(condition_labels ~= ""), 'stable');
    K = size(X, 2);
    C = size(X, 3);
    Cell = nan(numel(groups), numel(conds), K, C);

    for g = 1:numel(groups)
        for c = 1:numel(conds)
            idx = group_labels == groups(g) & condition_labels == conds(c);
            if any(idx)
                Cell(g, c, :, :) = weighted_mean_maps(X(idx, :, :), row_weights(idx));
            end
        end
    end
    if renorm_level_means
        for g = 1:numel(groups)
            for c = 1:numel(conds)
                M = squeeze(Cell(g, c, :, :));
                Cell(g, c, :, :) = normalise_maps_2d(M);
            end
        end
    end

    row_mean = squeeze(mean(Cell, 2, 'omitnan'));
    col_mean = squeeze(mean(Cell, 1, 'omitnan'));
    grand = squeeze(mean(mean(Cell, 1, 'omitnan'), 2, 'omitnan'));

    stat = nan(1, K);
    for k = 1:K
        residuals = [];
        for g = 1:numel(groups)
            for c = 1:numel(conds)
                cell_map = squeeze(Cell(g, c, k, :))';
                if any(~isfinite(cell_map))
                    continue;
                end
                r = cell_map - squeeze(row_mean(g, k, :))' - squeeze(col_mean(c, k, :))' + squeeze(grand(k, :));
                residuals = [residuals, r];
            end
        end
        stat(k) = sqrt(mean(residuals .^ 2, 'omitnan'));
    end
end

function M = weighted_mean_maps(X, weights)
    if isempty(X)
        M = nan(size(X, 2), size(X, 3));
        return;
    end
    weights = double(weights(:));
    weights(~isfinite(weights) | weights < 0) = 0;
    if sum(weights) <= eps
        weights = ones(size(weights));
    end
    weights = weights ./ sum(weights);
    W = reshape(weights, [], 1, 1);
    valid = isfinite(X);
    X0 = X;
    X0(~valid) = 0;
    num = squeeze(sum(bsxfun(@times, X0, W), 1));
    den = squeeze(sum(bsxfun(@times, double(valid), W), 1));
    den(den <= eps) = NaN;
    M = num ./ den;
    if isvector(M)
        M = reshape(M, size(X, 2), size(X, 3));
    end
end

function M = normalise_level_maps(M)
    for a = 1:size(M, 1)
        for b = 1:size(M, 2)
            v = squeeze(M(a, b, :))';
            v = normalise_maps_2d(v);
            M(a, b, :) = v;
        end
    end
end

function perm_labels = permute_labels_within_units(labels, units)
    labels = string(labels(:));
    units = string(units(:));
    perm_labels = labels;
    ul = unique(units, 'stable');
    for i = 1:numel(ul)
        idx = find(units == ul(i));
        if numel(idx) > 1
            perm_labels(idx) = labels(idx(randperm(numel(idx))));
        end
    end
end

function permuted = hybrid_permute_labels(labels,units)
    labels=string(labels(:)); units=string(units(:)); permuted=labels; unpaired=[];
    levels=unique(labels,'stable'); ul=unique(units,'stable');
    for i=1:numel(ul)
        idx=find(units==ul(i)); present=unique(labels(idx));
        if numel(present)==numel(levels) && numel(idx)==numel(levels)
            if rand<.5, permuted(idx)=flipud(labels(idx)); end
        else
            unpaired=[unpaired;idx(:)]; %#ok<AGROW>
        end
    end
    if numel(unpaired)>1, permuted(unpaired)=labels(unpaired(randperm(numel(unpaired)))); end
end

function [Xu, label_u, unit_u] = collapse_to_unit(X, units, labels, ref)
    units = string(units(:));
    labels = string(labels(:));
    unit_u = unique(units, 'stable');
    Xu = nan(numel(unit_u), size(X, 2), size(X, 3));
    label_u = strings(numel(unit_u), 1);
    for i = 1:numel(unit_u)
        idx = find(units == unit_u(i));
        M = squeeze(mean(X(idx, :, :), 1, 'omitnan'));
        M = sign_align_and_normalise_centers(M, ref);
        Xu(i, :, :) = M;
        labs = unique(labels(idx), 'stable');
        labs = labs(labs ~= "");
        if isempty(labs)
            label_u(i) = "";
        elseif numel(labs) == 1
            label_u(i) = labs(1);
        else
            error('collapse_to_unit found multiple labels for one unit. This should not occur for a between-subject factor.');
        end
    end
end

function complete_units = units_with_all_conditions(units, conditions)
    units = string(units(:));
    conditions = string(conditions(:));
    all_conds = unique(conditions(conditions ~= ""), 'stable');
    ul = unique(units, 'stable');
    keep = false(numel(ul), 1);
    for i = 1:numel(ul)
        c = unique(conditions(units == ul(i) & conditions ~= ""));
        keep(i) = numel(c) == numel(all_conds) && all(ismember(all_conds, c));
    end
    complete_units = ul(keep);
end

function keep_units = units_with_at_least_n_conditions(units, conditions, n)
    units = string(units(:));
    conditions = string(conditions(:));
    ul = unique(units, 'stable');
    keep = false(numel(ul), 1);
    for i = 1:numel(ul)
        c = unique(conditions(units == ul(i) & conditions ~= ""));
        keep(i) = numel(c) >= n;
    end
    keep_units = ul(keep);
end

function ok = all_cells_present(groups, conditions)
    groups = string(groups(:));
    conditions = string(conditions(:));
    gl = unique(groups(groups ~= ""), 'stable');
    cl = unique(conditions(conditions ~= ""), 'stable');
    ok = true;
    for g = 1:numel(gl)
        for c = 1:numel(cl)
            ok = ok && any(groups == gl(g) & conditions == cl(c));
        end
    end
end

% ======================================================================
% Results tables
% ======================================================================

function R = finish_result(effect_name, test_kind, permutation_scheme, obs, perm, levels, n_units, n_observations, note)
    obs = obs(:)';
    p_unc = (1 + sum(perm >= obs, 1)) ./ (size(perm, 1) + 1);
    max_perm = max(perm, [], 2);
    p_max = nan(size(obs));
    for k = 1:numel(obs)
        p_max(k) = (1 + sum(max_perm >= obs(k))) ./ (size(perm, 1) + 1);
    end
    R = struct();
    R.effect = char(effect_name);
    R.test_kind = char(test_kind);
    R.permutation_scheme = char(permutation_scheme);
    R.statistic = obs;
    R.p_uncorrected = p_unc;
    R.p_max_state = p_max;
    R.levels = string(levels(:));
    R.n_units = n_units;
    R.n_observations = n_observations;
    R.n_permutations = size(perm, 1);
    R.note = char(note);
end

function R = empty_result(effect_name, test_kind, K, note)
    R = struct();
    R.effect = char(effect_name);
    R.test_kind = char(test_kind);
    R.permutation_scheme = 'not_run';
    R.statistic = nan(1, K);
    R.p_uncorrected = nan(1, K);
    R.p_max_state = nan(1, K);
    R.levels = strings(0, 1);
    R.n_units = 0;
    R.n_observations = 0;
    R.n_permutations = 0;
    R.note = char(note);
end

function T = result_to_table(R)
    K = numel(R.statistic);
    effect = repmat({R.effect}, K, 1);
    test_kind = repmat({R.test_kind}, K, 1);
    state = (1:K)';
    statistic = R.statistic(:);
    p_uncorrected = R.p_uncorrected(:);
    p_max_state = R.p_max_state(:);
    permutation_scheme = repmat({R.permutation_scheme}, K, 1);
    levels = repmat({strjoin(cellstr(R.levels), '|')}, K, 1);
    n_units = repmat(R.n_units, K, 1);
    n_observations = repmat(R.n_observations, K, 1);
    n_permutations = repmat(R.n_permutations, K, 1);
    note = repmat({R.note}, K, 1);
    T = table(effect, test_kind, state, statistic, p_uncorrected, p_max_state, ...
        permutation_scheme, levels, n_units, n_observations, n_permutations, note);
end

function T = bayesian_result_to_table(effect_name, test_kind, bootstrap_scheme, obs, draws, levels, n_units, n_observations, cfg, note)
    K = numel(obs);
    rope = double(cfg.bayesian_rope);
    effect = repmat({char(effect_name)}, K, 1);
    test_kind_col = repmat({char(test_kind)}, K, 1);
    state = (1:K)';
    statistic = obs(:);
    posterior_mean = mean(draws, 1, 'omitnan')';
    posterior_sd = std(draws, 0, 1, 'omitnan')';
    posterior_median = median(draws, 1, 'omitnan')';
    posterior_ci_lower = percentile_cols(draws, 2.5)';
    posterior_ci_upper = percentile_cols(draws, 97.5)';
    bayesian_rope = repmat(rope, K, 1);
    posterior_prob_gt_rope = mean(draws > rope, 1, 'omitnan')';
    posterior_prob_le_rope = mean(draws <= rope, 1, 'omitnan')';
    evidence_ratio_gt_rope = safe_ratio(posterior_prob_gt_rope, posterior_prob_le_rope);
    bootstrap_scheme_col = repmat({char(bootstrap_scheme)}, K, 1);
    levels_col = repmat({strjoin(cellstr(string(levels(:))), '|')}, K, 1);
    n_units_col = repmat(n_units, K, 1);
    n_observations_col = repmat(n_observations, K, 1);
    n_posterior = repmat(size(draws, 1), K, 1);
    note_col = repmat({char(note)}, K, 1);

    T = table(effect, test_kind_col, state, statistic, posterior_mean, posterior_sd, ...
        posterior_median, posterior_ci_lower, posterior_ci_upper, bayesian_rope, ...
        posterior_prob_gt_rope, posterior_prob_le_rope, evidence_ratio_gt_rope, ...
        bootstrap_scheme_col, levels_col, n_units_col, n_observations_col, n_posterior, note_col, ...
        'VariableNames', {'effect', 'test_kind', 'state', 'statistic', 'posterior_mean', 'posterior_sd', ...
        'posterior_median', 'posterior_ci_lower', 'posterior_ci_upper', 'bayesian_rope', ...
        'posterior_prob_gt_rope', 'posterior_prob_le_rope', 'evidence_ratio_gt_rope', ...
        'bootstrap_scheme', 'levels', 'n_units', 'n_observations', 'n_posterior', 'note'});
end

function T = empty_bayesian_result(effect_name, test_kind, K, note, cfg)
    draws = nan(cfg.n_posterior, K);
    obs = nan(1, K);
    T = bayesian_result_to_table(effect_name, test_kind, 'not_run', obs, draws, strings(0, 1), 0, 0, cfg, note);
end

function T = empty_bayesian_table()
    T = table(cell(0,1), cell(0,1), zeros(0,1), zeros(0,1), zeros(0,1), zeros(0,1), ...
        zeros(0,1), zeros(0,1), zeros(0,1), zeros(0,1), zeros(0,1), zeros(0,1), zeros(0,1), ...
        cell(0,1), cell(0,1), zeros(0,1), zeros(0,1), zeros(0,1), cell(0,1), ...
        'VariableNames', {'effect', 'test_kind', 'state', 'statistic', 'posterior_mean', 'posterior_sd', ...
        'posterior_median', 'posterior_ci_lower', 'posterior_ci_upper', 'bayesian_rope', ...
        'posterior_prob_gt_rope', 'posterior_prob_le_rope', 'evidence_ratio_gt_rope', ...
        'bootstrap_scheme', 'levels', 'n_units', 'n_observations', 'n_posterior', 'note'});
end

function p = percentile_cols(X, pct)
    p = nan(1, size(X, 2));
    for k = 1:size(X, 2)
        x = X(:, k);
        x = sort(x(isfinite(x)));
        if isempty(x)
            continue;
        end
        pos = 1 + (numel(x) - 1) * pct / 100;
        lo = floor(pos);
        hi = ceil(pos);
        if lo == hi
            p(k) = x(lo);
        else
            p(k) = x(lo) * (hi - pos) + x(hi) * (pos - lo);
        end
    end
end

function r = safe_ratio(num, den)
    r = nan(size(num));
    for i = 1:numel(num)
        if ~isfinite(num(i)) || ~isfinite(den(i))
            continue;
        end
        r(i) = (num(i) + eps) ./ (den(i) + eps);
    end
end

function T = vertcat_nonempty_tables(tables)
    keep = false(numel(tables), 1);
    for i = 1:numel(tables)
        keep(i) = istable(tables{i}) && height(tables{i}) > 0;
    end
    if ~any(keep)
        T = table();
        return;
    end
    tables = tables(keep);
    T = tables{1};
    for i = 2:numel(tables)
        T = [T; tables{i}];
    end
end

function T = add_fdr_columns(T)
    if isempty(T) || ~ismember('p_uncorrected', T.Properties.VariableNames)
        return;
    end
    T.q_bh_from_uncorrected_all = bh_fdr(T.p_uncorrected);
    T.q_bh_from_pmax_all = bh_fdr(T.p_max_state);

    q_effect = nan(height(T), 1);
    effects = unique(string(T.effect), 'stable');
    for i = 1:numel(effects)
        idx = string(T.effect) == effects(i);
        q_effect(idx) = bh_fdr(T.p_uncorrected(idx));
    end
    T.q_bh_from_uncorrected_within_effect = q_effect;
end

function q = bh_fdr(p)
    p = p(:);
    q = nan(size(p));
    valid = isfinite(p);
    if ~any(valid)
        return;
    end
    pv = p(valid);
    m = numel(pv);
    [ps, idx] = sort(pv);
    qs = ps .* m ./ (1:m)';
    qs = flipud(cummin(flipud(qs)));
    qs(qs > 1) = 1;
    tmp = nan(size(pv));
    tmp(idx) = qs;
    q(valid) = tmp;
end

% ======================================================================
% Plotting and notes
% ======================================================================

function plot_pmax_heatmap(T, plot_dir, cfg)
    if isempty(T) || height(T) == 0 || ~all(ismember({'effect','state','p_max_state'}, T.Properties.VariableNames))
        return;
    end
    effects = unique(string(T.effect), 'stable');
    states = unique(T.state, 'stable');
    Z = nan(numel(states), numel(effects));
    for e = 1:numel(effects)
        for s = 1:numel(states)
            idx = string(T.effect) == effects(e) & T.state == states(s);
            if any(idx)
                p = T.p_max_state(find(idx, 1, 'first'));
                Z(s, e) = -log10(max(p, realmin));
            end
        end
    end

    fig = figure('Color', 'white', 'Visible', 'off', 'Position', [100 100 max(800, 40*numel(effects)) 350]);
    imagesc(Z);
    cb = colorbar;
    ylabel(cb, '-log10 p_{max-state}');
    yticklabels(compose('State %d', states));
    xticks(1:numel(effects));
    xticklabels(cellstr(effects));
    set(gca, 'TickLabelInterpreter', 'none');
    xtickangle(45);
    yticks(1:numel(states));
    hold on;
    threshold = -log10(cfg.alpha);
    title(sprintf('Microstate TANOVA, max-state corrected p-values; dashed threshold p=%.3f', cfg.alpha), 'Interpreter', 'none');
    for e = 1:numel(effects)
        for s = 1:numel(states)
            if isfinite(Z(s,e)) && Z(s,e) >= threshold
                text(e, s, '*', 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 14);
            end
        end
    end
    axis tight;
    out = fullfile(plot_dir, 'tanova_pmax_heatmap.png');
    exportgraphics(fig, out, 'Resolution', 200);
    close(fig);
end

function plot_descriptive_mean_maps(D, H, plot_dir, cfg)
    if exist('topoplot', 'file') ~= 2
        return;
    end
    if ~isfield(H, 'common_chanlocs') || isempty(H.common_chanlocs)
        return;
    end
    chanlocs = H.common_chanlocs;
    if numel(chanlocs) ~= D.n_channels
        return;
    end
    utils = microstate_utilities();
    [chanlocs, keep_idx] = utils.prepare_chanlocs_for_topoplot(chanlocs, D.n_channels);
    if numel(keep_idx) ~= D.n_channels, return; end
    if D.has_group
        [Xu, gu] = collapse_to_unit(D.X, D.unit, D.group, D.global_ref);
        plot_mean_maps_by_factor(Xu, gu, chanlocs, fullfile(plot_dir, 'mean_maps_by_group.png'), 'Group mean microstate maps', cfg);
    end
    if D.has_condition
        complete_units = units_with_all_conditions(D.unit, D.condition);
        keep = ismember(D.unit, complete_units);
        if any(keep)
            plot_mean_maps_by_factor(D.X(keep,:,:), D.condition(keep), chanlocs, fullfile(plot_dir, 'mean_maps_by_condition.png'), 'Condition mean microstate maps', cfg);
        end
    end
    if D.has_group && D.has_condition
        complete_units = units_with_all_conditions(D.unit, D.condition);
        keep = ismember(D.unit, complete_units);
        if any(keep)
            cell_labels = D.group(keep) + " x " + D.condition(keep);
            plot_mean_maps_by_factor(D.X(keep,:,:), cell_labels, chanlocs, fullfile(plot_dir, 'mean_maps_by_group_condition.png'), 'Group x condition mean microstate maps', cfg);
        end
    end
end

function plot_mean_maps_by_factor(X, labels, chanlocs, out_file, title_text, cfg)
    labels = string(labels(:));
    levels = unique(labels(labels ~= ""), 'stable');
    if isempty(levels)
        return;
    end
    K = size(X, 2);
    n_rows = numel(levels);
    n_cols = K;
    fig = figure('Color', 'white', 'Visible', 'off', 'Position', [100 100 max(900, 180*n_cols) max(300, 170*n_rows)]);
    clim = 1;
    for l = 1:numel(levels)
        idx = labels == levels(l);
        M = squeeze(mean(X(idx, :, :), 1, 'omitnan'));
        M = normalise_maps_2d(M);
        clim = max(clim, max(abs(M(:))));
    end
    for l = 1:numel(levels)
        idx = labels == levels(l);
        M = squeeze(mean(X(idx, :, :), 1, 'omitnan'));
        M = normalise_maps_2d(M);
        for k = 1:K
            subplot(n_rows, n_cols, (l-1)*n_cols + k);
            topoplot(M(k, :), chanlocs, 'electrodes', 'off', 'numcontour', 6, 'maplimits', [-clim clim]);
            title(sprintf('%s | state %d', char(levels(l)), k), 'Interpreter', 'none', 'FontSize', 8);
        end
    end
    sgtitle(title_text, 'Interpreter', 'none', 'FontWeight', 'bold');
    exportgraphics(fig, out_file, 'Resolution', 200);
    close(fig);
end

function write_methods_notes(file_path, TANOVA, H)
    fid = fopen(file_path, 'w');
    if fid < 0
        warning('Could not write notes file: %s', file_path);
        return;
    end
    c = onCleanup(@() fclose(fid));
    cfg = TANOVA.config;
    fprintf(fid, 'Microstate hierarchical TANOVA notes\n');
    fprintf(fid, '====================================\n\n');
    fprintf(fid, 'Input MAT: %s\n', cfg.results_mat);
    fprintf(fid, 'Created: %s\n', TANOVA.created);
    if isfield(H, 'selected_K')
        fprintf(fid, 'Selected K: %d\n', H.selected_K);
    end
    if isfield(H, 'hierarchy') && isfield(H.hierarchy, 'description')
        fprintf(fid, 'Hierarchy: %s\n', H.hierarchy.description);
    end
    fprintf(fid, 'Permutations: %d\n', cfg.n_permutations);
    fprintf(fid, 'Bayesian bootstrap draws: %d\n', cfg.n_posterior);
    fprintf(fid, 'Bayesian ROPE: %.6f dGFP/RMS units\n', cfg.bayesian_rope);
    fprintf(fid, 'Alpha: %.4f\n', cfg.alpha);
    fprintf(fid, 'Exclude inherited participant nodes: %s\n', tf(cfg.exclude_inherited_nodes));
    fprintf(fid, 'Require complete within-subject condition data: %s\n', tf(cfg.require_complete_within_subject));
    fprintf(fid, 'Renormalise factor-level mean maps: %s\n\n', tf(cfg.renormalise_factor_means));
    fprintf(fid, 'Interpretation:\n');
    fprintf(fid, '- Significant TANOVA effects indicate differences in scalp-field topography, not amplitude.\n');
    fprintf(fid, '- Because the analysed objects are microstate templates, polarity was aligned to the global template before testing.\n');
    fprintf(fid, '- p_max_state controls each effect across the selected K microstate classes using a max-statistic.\n');
    fprintf(fid, '- q_bh_* columns are provided as secondary summaries and should not replace the permutation max-statistic for the primary claim.\n\n');
    fprintf(fid, 'Bayesian bootstrap interpretation:\n');
    fprintf(fid, '- bayesian_tanova_results.csv uses the same aligned dGFP/RMS statistics as TANOVA.\n');
    fprintf(fid, '- Posterior uncertainty is estimated by Bayesian bootstrap weights over independent units.\n');
    fprintf(fid, '- Condition effects share one bootstrap weight across each participant/unit, preserving pairing.\n');
    fprintf(fid, '- Group effects bootstrap units within groups; interactions bootstrap units within groups and preserve condition pairing.\n');
    fprintf(fid, '- posterior_prob_gt_rope is the posterior probability that the topographic effect exceeds the configured ROPE.\n');
    fprintf(fid, '- evidence_ratio_gt_rope is posterior_prob_gt_rope / posterior_prob_le_rope, an evidence ratio against practical equivalence rather than a marginal-likelihood Bayes factor.\n\n');
    fprintf(fid, 'Runtime notes:\n');
    for i = 1:numel(TANOVA.notes)
        fprintf(fid, '- %s\n', TANOVA.notes{i});
    end
end

% ======================================================================
% Generic helpers
% ======================================================================

function D2 = subset_D(D, mask)
    mask = logical(mask(:));
    D2 = D;
    if isempty(D.X)
        return;
    end
    D2.X = D.X(mask, :, :);
    D2.participant = D.participant(mask);
    D2.unit = D.unit(mask);
    D2.group = D.group(mask);
    D2.condition = D.condition(mask);
    D2.semiology = D.semiology(mask);
    D2.event_type = D.event_type(mask);
    D2.manifest = D.manifest(mask, :);
    D2.has_group = valid_factor_present(D2.group) && numel(unique(D2.group(D2.group ~= ""))) >= 2;
    D2.has_condition = valid_factor_present(D2.condition) && numel(unique(D2.condition(D2.condition ~= ""))) >= 2;
end

function weights = bayesian_group_row_weights(labels)
    labels = string(labels(:));
    weights = zeros(numel(labels), 1);
    levels = unique(labels(labels ~= ""), 'stable');
    for i = 1:numel(levels)
        idx = find(labels == levels(i));
        weights(idx) = draw_dirichlet(numel(idx));
    end
end

function weights = bayesian_grouped_unit_row_weights(units, groups)
    units = string(units(:));
    groups = string(groups(:));
    weights = zeros(numel(units), 1);
    group_levels = unique(groups(groups ~= ""), 'stable');
    for g = 1:numel(group_levels)
        units_g = unique(units(groups == group_levels(g)), 'stable');
        unit_weights = draw_dirichlet(numel(units_g));
        weights = weights + row_weights_from_unit_weights(units, units_g, unit_weights);
    end
end

function weights = row_weights_from_unit_weights(units, unit_levels, unit_weights)
    units = string(units(:));
    unit_levels = string(unit_levels(:));
    unit_weights = double(unit_weights(:));
    weights = zeros(numel(units), 1);
    for i = 1:numel(unit_levels)
        idx = find(units == unit_levels(i));
        if isempty(idx)
            continue;
        end
        weights(idx) = unit_weights(i);
    end
end

function weights = equal_unit_row_weights(units,unit_levels,unit_weights)
    weights=zeros(numel(units),1); units=string(units(:));
    for i=1:numel(unit_levels)
        idx=find(units==unit_levels(i));
        if ~isempty(idx), weights(idx)=unit_weights(i)/numel(idx); end
    end
end

function w = draw_dirichlet(n)
    if n <= 0
        w = zeros(0, 1);
        return;
    end
    x = -log(max(rand(n, 1), realmin));
    s = sum(x);
    if s <= eps
        w = ones(n, 1) ./ n;
    else
        w = x ./ s;
    end
end

function present = valid_factor_present(x)
    x = string(x(:));
    x = x(strlength(x) > 0 & x ~= "" & ~ismissing(x));
    present = numel(unique(x)) >= 1;
end

function present = nodes_have_nonempty_field(nodes, field_name)
    present = false;
    if ~isstruct(nodes) || ~isfield(nodes, field_name)
        return;
    end
    for i = 1:numel(nodes)
        val = string(get_field_or(nodes(i), field_name, ''));
        if strlength(strtrim(val)) > 0 && ~ismissing(val)
            present = true;
            return;
        end
    end
end

function varies = participant_factor_varies(participant, factor)
    participant = string(participant(:));
    factor = string(factor(:));
    ul = unique(participant, 'stable');
    varies = false;
    for i = 1:numel(ul)
        f = unique(factor(participant == ul(i) & factor ~= ""));
        if numel(f) > 1
            varies = true;
            return;
        end
    end
end

function val = get_field_or(S, field, default_val)
    if isstruct(S) && isfield(S, field) && ~isempty(S.(field))
        val = S.(field);
    else
        val = default_val;
    end
end

function s = tf(x)
    if logical(x)
        s = 'true';
    else
        s = 'false';
    end
end

function s = safe_label(x)
    s = char(x);
    s = regexprep(s, '[^A-Za-z0-9_\-]+', '_');
    s = regexprep(s, '_+', '_');
    s = regexprep(s, '^_|_$', '');
    if isempty(s)
        s = 'empty';
    end
end

function S = rmfield_if_present(S, fields)
    for i = 1:numel(fields)
        if isfield(S, fields{i})
            S = rmfield(S, fields{i});
        end
    end
end

function results_mat = default_results_mat()
    cfg_file = 'microstate_config.json';
    if isfile(cfg_file)
        cfg = jsondecode(fileread(cfg_file));
        if isfield(cfg, 'paths') && isfield(cfg.paths, 'hierarchical_output_dir')
            results_mat = fullfile(char(cfg.paths.hierarchical_output_dir), 'hierarchical_microstate_results.mat');
            return;
        end
    end
    results_mat = fullfile('outputs', 'hierarchical_microstates', 'hierarchical_microstate_results.mat');
end

function out_dir = default_output_dir(results_mat, config_file)
    out_dir = '';
    if isfile(config_file)
        try
            cfg = jsondecode(fileread(config_file));
            if isfield(cfg, 'paths') && isfield(cfg.paths, 'diagnostic_output_dir')
                out_dir = fullfile(char(cfg.paths.diagnostic_output_dir), 'microstate_tanova');
            elseif isfield(cfg, 'paths') && isfield(cfg.paths, 'hierarchical_output_dir')
                out_dir = fullfile(char(cfg.paths.hierarchical_output_dir), 'tanova');
            end
        catch
            out_dir = '';
        end
    end
    if isempty(out_dir)
        out_dir = fullfile(fileparts(results_mat), 'tanova');
    end
end

function path_out = output_path(output_dir, file_name)
    file_name = char(string(file_name));
    if startsWith(file_name, filesep) || ~isempty(regexp(file_name, '^[A-Za-z]:[\\/]', 'once'))
        path_out = file_name;
    else
        path_out = fullfile(output_dir, file_name);
    end
end
