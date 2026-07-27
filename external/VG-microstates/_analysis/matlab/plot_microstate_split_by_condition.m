function out_files = plot_microstate_split_by_condition(results_mat, varargin)
% PLOT_MICROSTATE_SPLIT_BY_CONDITION
%
% Plot hierarchical microstate templates split by condition.
%
% Usage:
%   plot_microstate_split_by_condition();
%   plot_microstate_split_by_condition(results_mat);
%   plot_microstate_split_by_condition(results_mat, 'source', 'participant_mean');
%
% source:
%   'auto'                 Prefer HResults.conditions if present, otherwise
%                          participant-level marginal means, otherwise
%                          averaged group-condition nodes.
%   'hierarchy'            Use fitted condition-level nodes only. This requires
%                          HResults.conditions from the adaptive fitter.
%   'participant_mean'     Average participant-level templates within condition.
%   'group_condition_mean' Average fitted group x condition nodes within condition.
%
% The default is intentionally marginal over group: each row is one condition,
% not one group-by-condition cell.

    if nargin < 1 || isempty(results_mat)
        results_mat = default_results_mat();
    end

    p = inputParser;
    addRequired(p, 'results_mat', @(x) ischar(x) || isstring(x));
    addParameter(p, 'config_file', 'microstate_config.json', @(x) ischar(x) || isstring(x));
    addParameter(p, 'output_dir', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'file_prefix', 'microstate_templates_by_condition', @(x) ischar(x) || isstring(x));
    addParameter(p, 'format', 'png', @(x) ischar(x) || isstring(x));
    addParameter(p, 'resolution', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x) && x > 0));
    addParameter(p, 'display_normalisation', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'maplimits', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'map_percentile', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x) && x > 0 && x <= 100));
    addParameter(p, 'colormap_name', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'electrodes', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'visible', false, @(x) islogical(x) && isscalar(x));
    addParameter(p, 'source', 'auto', @(x) ischar(x) || isstring(x));
    addParameter(p, 'exclude_inherited_nodes', true, @(x) islogical(x) && isscalar(x));
    parse(p, results_mat, varargin{:});
    cfg = p.Results;

    cfg.results_mat = char(cfg.results_mat);
    cfg.config_file = char(cfg.config_file);
    cfg.file_prefix = char(cfg.file_prefix);
    cfg.format = lower(char(cfg.format));
    cfg.source = lower(char(cfg.source));
    cfg = apply_config_defaults(cfg, 'by_condition');

    if exist('topoplot', 'file') ~= 2
        error('EEGLAB topoplot is not on the MATLAB path. Start EEGLAB or add it to the path first.');
    end
    if ~isfile(cfg.results_mat)
        error('Results MAT file not found: %s', cfg.results_mat);
    end
    if ~exist(cfg.output_dir, 'dir')
        mkdir(cfg.output_dir);
    end

    H = load_hresults(cfg.results_mat);
    [maps3, level_names, manifest, actual_source] = get_condition_maps(H, cfg);
    if isempty(maps3)
        error('No condition-split maps could be extracted. The fit probably did not contain a condition factor.');
    end

    K = size(maps3, 2);
    n_channels = size(maps3, 3);
    [chanlocs, keep_idx] = get_usable_chanlocs(H, n_channels);
    maps3 = maps3(:, :, keep_idx);

    maps2 = reshape(maps3, [], numel(keep_idx));
    maps2 = normalise_for_display(maps2, cfg.display_normalisation);
    maps3_plot = reshape(maps2, size(maps3, 1), K, numel(keep_idx));
    state_names = state_labels(H, K);
    clim_abs = compute_clim(maps2, cfg);

    n_rows = size(maps3_plot, 1);
    n_cols = K;
    fig_visible = ternary_char(cfg.visible, 'on', 'off');
    fig = figure('Name', 'Microstate templates by condition', ...
        'NumberTitle', 'off', 'Color', 'white', 'Visible', fig_visible, ...
        'Position', [100, 100, max(950, 190*n_cols), max(300, 185*n_rows + 120)]);

    for r = 1:n_rows
        for k = 1:K
            subplot(n_rows, n_cols, (r-1)*n_cols + k);
            plot_one_topomap(squeeze(maps3_plot(r, k, :))', chanlocs, cfg, clim_abs);
            title(sprintf('%s | %s', char(level_names(r)), state_names{k}), ...
                'FontSize', 8, 'FontWeight', 'bold', 'Interpreter', 'none');
        end
    end

    title_text = sprintf('Microstate templates split by condition | source=%s | K=%d', actual_source, K);
    sgtitle(title_text, 'FontSize', 12, 'FontWeight', 'bold', 'Interpreter', 'none');

    out_fig = fullfile(cfg.output_dir, [cfg.file_prefix '.' cfg.format]);
    export_figure(fig, out_fig, cfg);
    if ~cfg.visible
        close(fig);
    end

    out_mat = fullfile(cfg.output_dir, [cfg.file_prefix '.mat']);
    out_csv = fullfile(cfg.output_dir, [cfg.file_prefix '_manifest.csv']);
    plotted_maps = maps3_plot; %#ok<NASGU>
    condition_names = level_names; %#ok<NASGU>
    plotted_channel_labels = channel_labels_from_chanlocs(chanlocs); %#ok<NASGU>
    save(out_mat, 'plotted_maps', 'condition_names', 'state_names', 'plotted_channel_labels', 'manifest', 'cfg', 'actual_source', '-v7.3');
    writetable(manifest, out_csv);

    out_files = struct('figure', out_fig, 'mat', out_mat, 'manifest_csv', out_csv);
    fprintf('Saved condition-split microstate plot: %s\n', out_fig);
end

% ======================================================================
% Condition extraction
% ======================================================================

function [maps3, level_names, manifest, actual_source] = get_condition_maps(H, cfg)
    global_ref = get_global_ref(H);
    maps3 = [];
    level_names = strings(0, 1);
    manifest = table();
    actual_source = '';

    if strcmp(cfg.source, 'auto') || strcmp(cfg.source, 'hierarchy')
        [maps3, level_names, manifest] = get_condition_maps_from_hierarchy(H, global_ref);
        if ~isempty(maps3)
            actual_source = 'hierarchy';
            return;
        elseif strcmp(cfg.source, 'hierarchy')
            error('source="hierarchy" was requested, but HResults.conditions did not contain usable condition maps.');
        end
    end

    if strcmp(cfg.source, 'auto') || strcmp(cfg.source, 'participant_mean')
        [maps3, level_names, manifest] = get_condition_maps_from_participants(H, cfg, global_ref);
        if ~isempty(maps3)
            actual_source = 'participant_mean';
            return;
        elseif strcmp(cfg.source, 'participant_mean')
            error('source="participant_mean" was requested, but participant-level condition maps were unavailable.');
        end
    end

    if strcmp(cfg.source, 'auto') || strcmp(cfg.source, 'group_condition_mean')
        [maps3, level_names, manifest] = get_condition_maps_from_group_condition_nodes(H, global_ref);
        if ~isempty(maps3)
            actual_source = 'group_condition_mean';
            return;
        elseif strcmp(cfg.source, 'group_condition_mean')
            error('source="group_condition_mean" was requested, but group-condition nodes were unavailable.');
        end
    end

    if isempty(maps3)
        error('Unknown source or no condition maps available: %s', cfg.source);
    end
end

function [maps3, level_names, manifest] = get_condition_maps_from_hierarchy(H, global_ref)
    maps3 = [];
    level_names = strings(0, 1);
    rows = {};
    if ~isfield(H, 'conditions') || isempty(H.conditions)
        manifest = table();
        return;
    end
    nodes = H.conditions;
    for i = 1:numel(nodes)
        node = nodes(i);
        if ~isfield(node, 'centers') || isempty(node.centers)
            continue;
        end
        c = string(get_field_or(node, 'condition', ''));
        if strlength(c) == 0
            c = string(strip_condition_prefix(get_field_or(node, 'name', sprintf('condition_%d', i))));
        end
        if strlength(c) == 0 || ismissing(c)
            continue;
        end
        M = sign_align_and_normalise_centers(double(node.centers), global_ref);
        if isempty(maps3)
            maps3 = nan(0, size(M, 1), size(M, 2));
        end
        maps3(end+1, :, :) = M; %#ok<AGROW>
        level_names(end+1, 1) = c; %#ok<AGROW>
        rows(end+1, :) = {char(c), char(get_field_or(node, 'name', '')), char(get_field_or(node, 'level', 'condition')), double(get_field_or(node, 'n_maps', NaN)), logical(get_field_or(node, 'inherited', false))}; %#ok<AGROW>
    end
    if isempty(rows)
        manifest = table();
    else
        manifest = cell2table(rows, 'VariableNames', {'condition', 'node_name', 'node_level', 'n_maps', 'inherited'});
    end
end

function [maps3, level_names, manifest] = get_condition_maps_from_participants(H, cfg, global_ref)
    D = extract_participant_template_data(H, cfg, global_ref);
    maps3 = [];
    level_names = strings(0, 1);
    manifest = table();
    if isempty(D.X) || ~valid_factor_present(D.condition)
        return;
    end
    levels = unique(D.condition(D.condition ~= ""), 'stable');
    rows = cell(numel(levels), 6);
    for li = 1:numel(levels)
        idx = D.condition == levels(li);
        M = squeeze(mean(D.X(idx, :, :), 1, 'omitnan'));
        M = sign_align_and_normalise_centers(M, global_ref);
        if isempty(maps3)
            maps3 = nan(0, size(M, 1), size(M, 2));
        end
        maps3(end+1, :, :) = M; %#ok<AGROW>
        level_names(end+1, 1) = levels(li); %#ok<AGROW>
        rows(li, :) = {char(levels(li)), 'participant_mean', 'participant', sum(idx), sum(D.n_maps(idx), 'omitnan'), any(D.inherited(idx))};
    end
    manifest = cell2table(rows, 'VariableNames', {'condition', 'node_name', 'node_level', 'n_units', 'n_maps', 'any_inherited'});
end

function [maps3, level_names, manifest] = get_condition_maps_from_group_condition_nodes(H, global_ref)
    maps3 = [];
    level_names = strings(0, 1);
    manifest = table();
    if ~isfield(H, 'group_conditions') || isempty(H.group_conditions)
        return;
    end
    nodes = H.group_conditions;
    condition = strings(numel(nodes), 1);
    good = false(numel(nodes), 1);
    for i = 1:numel(nodes)
        if isfield(nodes(i), 'centers') && ~isempty(nodes(i).centers)
            condition(i) = string(get_field_or(nodes(i), 'condition', ''));
            good(i) = strlength(condition(i)) > 0 && ~ismissing(condition(i));
        end
    end
    levels = unique(condition(good), 'stable');
    if isempty(levels)
        return;
    end
    rows = cell(numel(levels), 5);
    for li = 1:numel(levels)
        idx = find(good & condition == levels(li));
        stack = [];
        total_maps = 0;
        any_inherited = false;
        for j = 1:numel(idx)
            node = nodes(idx(j));
            M = sign_align_and_normalise_centers(double(node.centers), global_ref);
            if isempty(stack)
                stack = nan(0, size(M, 1), size(M, 2));
            end
            stack(end+1, :, :) = M; %#ok<AGROW>
            total_maps = total_maps + double(get_field_or(node, 'n_maps', 0));
            any_inherited = any_inherited || logical(get_field_or(node, 'inherited', false));
        end
        Mmean = squeeze(mean(stack, 1, 'omitnan'));
        Mmean = sign_align_and_normalise_centers(Mmean, global_ref);
        if isempty(maps3)
            maps3 = nan(0, size(Mmean, 1), size(Mmean, 2));
        end
        maps3(end+1, :, :) = Mmean; %#ok<AGROW>
        level_names(end+1, 1) = levels(li); %#ok<AGROW>
        rows(li, :) = {char(levels(li)), 'group_condition_mean', numel(idx), total_maps, any_inherited};
    end
    manifest = cell2table(rows, 'VariableNames', {'condition', 'node_name', 'n_group_condition_nodes', 'n_maps', 'any_inherited'});
end

function D = extract_participant_template_data(H, cfg, global_ref)
    nodes = [];
    if isfield(H, 'participants') && ~isempty(H.participants)
        nodes = H.participants;
    elseif isfield(H, 'participant_conditions') && ~isempty(H.participant_conditions)
        nodes = H.participant_conditions;
    elseif isfield(H, 'files') && ~isempty(H.files)
        nodes = H.files;
    else
        D = empty_participant_data();
        return;
    end
    K = size(global_ref, 1);
    X = [];
    participant = strings(0, 1);
    group = strings(0, 1);
    condition = strings(0, 1);
    inherited = false(0, 1);
    n_maps = nan(0, 1);

    for i = 1:numel(nodes)
        node = nodes(i);
        if ~isfield(node, 'centers') || isempty(node.centers)
            continue;
        end
        is_inherited = logical(get_field_or(node, 'inherited', false));
        if cfg.exclude_inherited_nodes && is_inherited
            continue;
        end
        M = double(node.centers);
        if size(M, 1) < K
            continue;
        end
        M = M(1:K, :);
        M = sign_align_and_normalise_centers(M, global_ref);
        if isempty(X)
            X = nan(0, K, size(M, 2));
        end
        X(end+1, :, :) = M; %#ok<AGROW>
        participant(end+1, 1) = string(get_field_or(node, 'participant', '')); %#ok<AGROW>
        group(end+1, 1) = string(get_field_or(node, 'group', '')); %#ok<AGROW>
        condition(end+1, 1) = string(get_field_or(node, 'condition', '')); %#ok<AGROW>
        inherited(end+1, 1) = is_inherited; %#ok<AGROW>
        n_maps(end+1, 1) = double(get_field_or(node, 'n_maps', NaN)); %#ok<AGROW>
    end

    D = struct('X', X, 'participant', participant, 'group', group, 'condition', condition, 'inherited', inherited, 'n_maps', n_maps);
end

function D = empty_participant_data()
    D = struct('X', [], 'participant', strings(0, 1), 'group', strings(0, 1), ...
        'condition', strings(0, 1), 'inherited', false(0, 1), 'n_maps', nan(0, 1));
end

% ======================================================================
% Generic helpers
% ======================================================================

function H = load_hresults(results_mat)
    S = load(results_mat);
    if isfield(S, 'HResults')
        H = S.HResults;
    else
        error('The MAT file does not contain HResults.');
    end
end

function results_mat = default_results_mat()
    cfg_file = 'microstate_config.json';
    if isfile(cfg_file)
        try
            C = jsondecode(fileread(cfg_file));
            if isfield(C, 'paths') && isfield(C.paths, 'hierarchical_output_dir')
                results_mat = fullfile(char(C.paths.hierarchical_output_dir), 'hierarchical_microstate_results.mat');
                return;
            end
        catch
        end
    end
    results_mat = fullfile('outputs', 'hierarchical_microstates', 'hierarchical_microstate_results.mat');
end

function cfg = apply_config_defaults(cfg, subdir_name)
    C = struct();
    if isfile(cfg.config_file)
        try
            C = jsondecode(fileread(cfg.config_file));
        catch ME
            warning('Could not read config file %s: %s', cfg.config_file, ME.message);
        end
    end
    if isempty(cfg.output_dir)
        cfg.output_dir = default_plot_output_dir(C, cfg.results_mat, subdir_name);
    else
        cfg.output_dir = char(cfg.output_dir);
    end
    if isempty(cfg.resolution)
        cfg.resolution = get_nested_or(C, {'plotting', 'resolution'}, 300);
    end
    if strlength(string(cfg.display_normalisation)) == 0
        cfg.display_normalisation = char(get_nested_or(C, {'plotting', 'display_normalisation'}, 'zscore'));
    else
        cfg.display_normalisation = char(cfg.display_normalisation);
    end
    if strlength(string(cfg.maplimits)) == 0
        cfg.maplimits = char(get_nested_or(C, {'plotting', 'maplimits'}, 'global_percentile'));
    else
        cfg.maplimits = char(cfg.maplimits);
    end
    if isempty(cfg.map_percentile)
        cfg.map_percentile = double(get_nested_or(C, {'plotting', 'map_percentile'}, 75));
    end
    if strlength(string(cfg.colormap_name)) == 0
        cfg.colormap_name = char(get_nested_or(C, {'plotting', 'colormap_name'}, 'jet'));
    else
        cfg.colormap_name = char(cfg.colormap_name);
    end
    if strlength(string(cfg.electrodes)) == 0
        cfg.electrodes = char(get_nested_or(C, {'plotting', 'electrodes'}, 'off'));
    else
        cfg.electrodes = char(cfg.electrodes);
    end
end

function out_dir = default_plot_output_dir(C, results_mat, subdir_name)
    out_dir = '';
    if isstruct(C) && isfield(C, 'paths')
        if isfield(C.paths, 'single_plot_dir')
            out_dir = fullfile(char(C.paths.single_plot_dir), 'hierarchical', subdir_name);
        elseif isfield(C.paths, 'diagnostic_output_dir')
            out_dir = fullfile(char(C.paths.diagnostic_output_dir), 'hierarchical_microstate_plots', subdir_name);
        elseif isfield(C.paths, 'hierarchical_output_dir')
            out_dir = fullfile(char(C.paths.hierarchical_output_dir), 'plots', subdir_name);
        end
    end
    if isempty(out_dir)
        out_dir = fullfile(fileparts(results_mat), 'plots', subdir_name);
    end
end

function val = get_nested_or(S, fields, default_val)
    val = default_val;
    try
        cur = S;
        for i = 1:numel(fields)
            if ~isstruct(cur) || ~isfield(cur, fields{i})
                return;
            end
            cur = cur.(fields{i});
        end
        val = cur;
    catch
        val = default_val;
    end
end

function global_ref = get_global_ref(H)
    if ~isfield(H, 'global') || ~isfield(H.global, 'centers') || isempty(H.global.centers)
        error('HResults.global.centers is missing.');
    end
    global_ref = normalise_maps_2d(double(H.global.centers));
    if isfield(H, 'selected_K') && ~isempty(H.selected_K)
        global_ref = global_ref(1:min(size(global_ref, 1), double(H.selected_K)), :);
    end
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
    if isempty(X), return; end
    X = X - mean(X, 2, 'omitnan');
    denom = sqrt(mean(X .^ 2, 2, 'omitnan'));
    denom(~isfinite(denom) | denom <= eps) = 1;
    X = X ./ denom;
end

function present = valid_factor_present(x)
    x = string(x(:));
    x = x(strlength(x) > 0 & x ~= "" & ~ismissing(x));
    present = ~isempty(unique(x));
end

function [chanlocs, keep_idx] = get_usable_chanlocs(H, n_channels)
    if ~isfield(H, 'common_chanlocs') || isempty(H.common_chanlocs)
        error('HResults.common_chanlocs is missing. Cannot use topoplot.');
    end
    chanlocs_all = H.common_chanlocs;
    if numel(chanlocs_all) ~= n_channels
        error('Number of channel locations (%d) does not match map width (%d).', numel(chanlocs_all), n_channels);
    end
    utils = microstate_utilities();
    [chanlocs, keep_idx] = utils.prepare_chanlocs_for_topoplot(chanlocs_all, n_channels);
    if numel(chanlocs) < 4
        error('Fewer than four usable scalp channel locations are available.');
    end
end

function labels = state_labels(H, K)
    labels = cell(K, 1);
    base = [];
    if isfield(H, 'global') && isfield(H.global, 'canonical_state_labels') && ~isempty(H.global.canonical_state_labels)
        base = H.global.canonical_state_labels;
    elseif isfield(H, 'canonical_template_alignment') && isfield(H.canonical_template_alignment, 'labels')
        base = H.canonical_template_alignment.labels;
    end
    for k = 1:K
        if ~isempty(base) && numel(base) >= k
            labels{k} = sprintf('State %d (%s)', k, char(string(base(k))));
        else
            labels{k} = sprintf('State %d', k);
        end
    end
end

function X = normalise_for_display(X, mode)
    mode = lower(char(mode));
    switch mode
        case {'zscore', 'z'}
            X = X - mean(X, 2, 'omitnan');
            sd = std(X, 0, 2, 'omitnan');
            sd(~isfinite(sd) | sd <= eps) = 1;
            X = X ./ sd;
        case {'rms', 'gfp'}
            X = normalise_maps_2d(X);
        case {'none', 'raw'}
        otherwise
            error('Unknown display_normalisation: %s', mode);
    end
end

function clim_abs = compute_clim(X, cfg)
    vals = abs(X(:));
    vals = vals(isfinite(vals));
    if isempty(vals)
        clim_abs = 1;
        return;
    end
    switch lower(char(cfg.maplimits))
        case 'global_percentile'
            clim_abs = prctile(vals, cfg.map_percentile);
        case {'global_absmax', 'absmax'}
            clim_abs = max(vals);
        otherwise
            clim_abs = max(vals);
    end
    if ~isfinite(clim_abs) || clim_abs <= eps
        clim_abs = max(vals);
    end
    if ~isfinite(clim_abs) || clim_abs <= eps
        clim_abs = 1;
    end
end

function plot_one_topomap(v, chanlocs, cfg, clim_abs)
    if strcmpi(cfg.maplimits, 'state_absmax')
        local = max(abs(v(isfinite(v))));
        if ~isfinite(local) || local <= eps, local = clim_abs; end
        lims = [-local local];
    else
        lims = [-clim_abs clim_abs];
    end
    topoplot(v, chanlocs, 'electrodes', char(cfg.electrodes), 'numcontour', 6, 'maplimits', lims);
    colormap(gca, char(cfg.colormap_name));
    colorbar;
end

function export_figure(fig, out_file, cfg)
    [out_dir, ~, ~] = fileparts(out_file);
    if ~isempty(out_dir) && ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end
    switch lower(cfg.format)
        case {'png', 'pdf', 'tif', 'tiff', 'jpg', 'jpeg'}
            exportgraphics(fig, out_file, 'Resolution', cfg.resolution);
        case 'fig'
            savefig(fig, out_file);
        otherwise
            error('Unsupported output format: %s', cfg.format);
    end
end

function labels = channel_labels_from_chanlocs(chanlocs)
    labels = cell(numel(chanlocs), 1);
    for i = 1:numel(chanlocs)
        if isfield(chanlocs(i), 'labels') && ~isempty(chanlocs(i).labels)
            labels{i} = char(chanlocs(i).labels);
        else
            labels{i} = sprintf('Ch%d', i);
        end
    end
end

function s = strip_condition_prefix(s)
    s = char(s);
    s = regexprep(s, '^condition:', '');
    s = regexprep(s, '^group_condition:[^:]+:', '');
end

function val = get_field_or(S, field, default_val)
    if isstruct(S) && isfield(S, field) && ~isempty(S.(field))
        val = S.(field);
    else
        val = default_val;
    end
end

function y = ternary_char(tf, a, b)
    if tf, y = a; else, y = b; end
end
