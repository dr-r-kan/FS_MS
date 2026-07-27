function out = plot_simulated_microstate_method_comparison(varargin)
% PLOT_SIMULATED_MICROSTATE_METHOD_COMPARISON
% Plot one randomly selected simulated EEG as three aligned microstate rows:
% true, K-Means, and VB-GMM.
%
% The helper works from the current simulation output folder by using the
% manifest written by simulated_ms_retrieval_experiment.m and the per-result
% JSON exports in <output_dir>/microstates_json.
%
% Example
%   plot_simulated_microstate_method_comparison('seed', 7);
%
%   out = plot_simulated_microstate_method_comparison( ...
%       'montage_type', '10-20-20', ...
%       'spm_criterion', 'elbow_sil_combined', ...
%       'kmeans_criterion', 'silhouette');

    setup_microstate_paths();
    util = microstate_utilities();
    repo_cfg = util.load_config();

    p = inputParser;
    addParameter(p, 'output_dir', char(repo_cfg.simulation.out_dir), @(x) ischar(x) || isstring(x));
    addParameter(p, 'results_csv', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'json_dir', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'template_file', char(repo_cfg.paths.template_file), @(x) ischar(x) || isstring(x));
    addParameter(p, 'seed', 42, @(x) isnumeric(x) && isscalar(x) && isfinite(x));
    addParameter(p, 'montage_type', 'full', @(x) ischar(x) || isstring(x));
    addParameter(p, 'K_true', 7, @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'spm_method', 'spm_vb', @(x) ischar(x) || isstring(x));
    addParameter(p, 'kmeans_method', 'kmeans_koenig', @(x) ischar(x) || isstring(x));
    addParameter(p, 'spm_criterion', 'elbow_sil_combined', @(x) ischar(x) || isstring(x));
    addParameter(p, 'kmeans_criterion', 'silhouette', @(x) ischar(x) || isstring(x));
    addParameter(p, 'require_exact_k_estimated', true, @islogical);
    addParameter(p, 'allow_spm_criterion_fallback', true, @islogical);
    addParameter(p, 'allow_montage_fallback', true, @islogical);
    addParameter(p, 'template_K', 7, @(x) isnumeric(x) && isscalar(x) && x >= 1);
    addParameter(p, 'topoplot_rotation_deg', -90, @(x) isnumeric(x) && isscalar(x) && isfinite(x));
    addParameter(p, 'strong_threshold', 0.5, @(x) isnumeric(x) && isscalar(x) && x >= 0 && x <= 1);
    addParameter(p, 'output_file', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'visible', false, @islogical);
    addParameter(p, 'resolution', double(util.get_field(repo_cfg.plotting, 'resolution', 300)), ...
        @(x) isnumeric(x) && isscalar(x) && x > 0);
    parse(p, varargin{:});

    cfg = p.Results;
    cfg.output_dir = util.resolve_path(char(cfg.output_dir), util.project_root());
    cfg.template_file = util.resolve_path(char(cfg.template_file), util.project_root());

    if isempty(cfg.results_csv)
        cfg.results_csv = fullfile(cfg.output_dir, 'results', 'comparison_results.csv');
    else
        cfg.results_csv = util.resolve_path(char(cfg.results_csv), util.project_root());
    end
    if isempty(cfg.json_dir)
        cfg.json_dir = fullfile(cfg.output_dir, 'microstates_json');
    else
        cfg.json_dir = util.resolve_path(char(cfg.json_dir), util.project_root());
    end

    if ~exist('pop_loadset', 'file')
        error('EEGLAB pop_loadset is not on the MATLAB path.');
    end
    if ~exist('topoplot', 'file')
        error('EEGLAB topoplot is not on the MATLAB path.');
    end
    if ~isfile(cfg.results_csv)
        error('Results manifest not found: %s', cfg.results_csv);
    end
    if ~isfile(cfg.template_file)
        error('Template file not found: %s', cfg.template_file);
    end

    T = readtable(cfg.results_csv, 'TextType', 'string');
    required_vars = {'rep', 'K_true', 'SNR_dB', 'overlap_prob', 'montage_type', ...
        'method', 'criterion', 'json_file', 'K_estimated'};
    missing_vars = required_vars(~ismember(required_vars, T.Properties.VariableNames));
    if ~isempty(missing_vars)
        error('Results manifest is missing required columns: %s', strjoin(missing_vars, ', '));
    end

    json_mask = has_usable_json_file(T.json_file, cfg.output_dir, cfg.json_dir);
    key_vars = intersect({'rep', 'K_true', 'SNR_dB', 'overlap_prob', 'montage_type', 'n_leads'}, ...
        T.Properties.VariableNames, 'stable');
    keep_vars = [key_vars, {'json_file', 'K_estimated'}];
    [T_join, cfg] = select_joined_recovery_examples(T, json_mask, key_vars, keep_vars, cfg);

    rng(double(cfg.seed), 'twister');
    selected_idx = randi(height(T_join));
    selection = T_join(selected_idx, :);

    spm_json_file = resolve_json_path(selection.spm_json_file(1), cfg.output_dir, cfg.json_dir);
    km_json_file = resolve_json_path(selection.kmeans_json_file(1), cfg.output_dir, cfg.json_dir);
    spm_data = jsondecode(fileread(spm_json_file));
    km_data = jsondecode(fileread(km_json_file));

    [spm_maps, spm_labels_raw] = extract_state_maps(spm_data, 'estimated_microstates');
    [km_maps, km_labels_raw] = extract_state_maps(km_data, 'estimated_microstates');
    [true_maps, true_labels_raw] = extract_state_maps(spm_data, 'true_microstates');
    common_channel_labels = intersect_labels_stable(spm_labels_raw, km_labels_raw);
    common_channel_labels = intersect_labels_stable(common_channel_labels, true_labels_raw);
    [display_channel_labels, chanlocs] = resolve_display_chanlocs(common_channel_labels, cfg.template_file, cfg.template_K);
    chanlocs = rotate_topoplot_chanlocs(chanlocs, cfg.topoplot_rotation_deg);
    if numel(display_channel_labels) < 4
        error('Only %d display channels could be matched to scalp locations.', numel(display_channel_labels));
    end

    true_alignment = align_microstates_to_template(true_maps, cfg.template_file, ...
        'estimated_channel_labels', true_labels_raw, ...
        'template_K', cfg.template_K, ...
        'strong_threshold', cfg.strong_threshold);
    spm_alignment = align_microstates_to_template(spm_maps, cfg.template_file, ...
        'estimated_channel_labels', spm_labels_raw, ...
        'template_K', cfg.template_K, ...
        'strong_threshold', cfg.strong_threshold);
    km_alignment = align_microstates_to_template(km_maps, cfg.template_file, ...
        'estimated_channel_labels', km_labels_raw, ...
        'template_K', cfg.template_K, ...
        'strong_threshold', cfg.strong_threshold);

    template_labels = pick_template_labels(true_alignment, spm_alignment, km_alignment);
    row_true = alignment_to_display_row(true_alignment, true_labels_raw, display_channel_labels, template_labels);
    row_km = alignment_to_display_row(km_alignment, km_labels_raw, display_channel_labels, template_labels);
    row_spm = alignment_to_display_row(spm_alignment, spm_labels_raw, display_channel_labels, template_labels);

    column_labels = template_labels(:)';
    if isempty(column_labels)
        error('No aligned template labels were available for plotting.');
    end
    row_true = project_row_to_columns(row_true, column_labels);
    row_km = project_row_to_columns(row_km, column_labels);
    row_spm = project_row_to_columns(row_spm, column_labels);

    all_maps = cat(1, row_true.maps(~isnan(row_true.maps(:, 1)), :), ...
        row_km.maps(~isnan(row_km.maps(:, 1)), :), ...
        row_spm.maps(~isnan(row_spm.maps(:, 1)), :));
    if isempty(all_maps)
        error('No finite maps were available after alignment and channel matching.');
    end
    clim = max(abs(all_maps(:)));
    if ~isfinite(clim) || clim <= eps
        clim = 1;
    end

    if isempty(cfg.output_file)
        plot_dir = fullfile(cfg.output_dir, 'analysis_plots');
        util.ensure_dir(plot_dir);
        cfg.output_file = fullfile(plot_dir, default_plot_filename(selection, cfg.seed, cfg.montage_type));
    else
        cfg.output_file = util.resolve_path(char(cfg.output_file), util.project_root());
        out_dir = fileparts(cfg.output_file);
        if ~isempty(out_dir)
            util.ensure_dir(out_dir);
        end
    end

    n_cols = numel(column_labels);
    fig = figure('Name', 'Simulated microstate method comparison', ...
        'Color', 'white', ...
        'Visible', onoff(cfg.visible), ...
        'NumberTitle', 'off', ...
        'Position', [80, 80, max(320 * n_cols, 1100), 900]);
    if ~cfg.visible
        cleaner = onCleanup(@() close_if_valid(fig)); %#ok<NASGU>
    end

    tl = tiledlayout(fig, 3, n_cols, 'TileSpacing', 'compact', 'Padding', 'compact');
    colormap(fig, 'jet');

    row_defs = { ...
        struct('title', 'True', 'data', row_true), ...
        struct('title', sprintf('K-Means (%s)', prettify_token(cfg.kmeans_criterion)), 'data', row_km), ...
        struct('title', sprintf('VB-GMM (%s)', prettify_token(cfg.spm_criterion)), 'data', row_spm)};

    last_plot_ax = [];
    for r = 1:3
        row_data = row_defs{r}.data;
        for c = 1:n_cols
            ax = nexttile(tl, (r - 1) * n_cols + c);
            if all(~isfinite(row_data.maps(c, :)))
                axis(ax, 'off');
                text(ax, 0.5, 0.5, '-', 'Units', 'normalized', ...
                    'HorizontalAlignment', 'center', 'FontSize', 16, 'Color', [0.55 0.55 0.55]);
            else
                topoplot(row_data.maps(c, :), chanlocs, ...
                    'electrodes', 'off', ...
                    'numcontour', 6, ...
                    'maplimits', [-clim clim]);
                axis(ax, 'off');
                last_plot_ax = ax;
                if isfinite(row_data.correlations(c))
                    text(ax, 0.5, -0.08, sprintf('r=%.2f', row_data.correlations(c)), ...
                        'Units', 'normalized', 'HorizontalAlignment', 'center', ...
                        'FontSize', 9, 'Color', [0.25 0.25 0.25]);
                end
            end

            if r == 1
                title(ax, column_labels{c}, ...
                    'FontWeight', 'bold', 'FontSize', 14, 'Interpreter', 'none');
            end
            if c == 1
                text(ax, -0.24, 0.5, row_defs{r}.title, 'Units', 'normalized', ...
                    'Rotation', 90, 'HorizontalAlignment', 'center', ...
                    'FontWeight', 'bold', 'FontSize', 12, 'Interpreter', 'none');
            end
        end
    end

    if ~isempty(last_plot_ax)
        cb = colorbar(last_plot_ax, 'eastoutside');
        cb.Label.String = 'Normalised map value';
    end

    sgtitle(tl, selection_title(selection, selected_idx, height(T_join), cfg), ...
        'FontWeight', 'bold', 'FontSize', 14, 'Interpreter', 'none');
    exportgraphics(fig, cfg.output_file, 'Resolution', cfg.resolution);

    out = struct();
    out.plot_file = cfg.output_file;
    out.seed = double(cfg.seed);
    out.selection_index = selected_idx;
    out.n_candidates = height(T_join);
    out.selection = selection;
    out.column_labels = column_labels;
    out.spm_json_file = spm_json_file;
    out.kmeans_json_file = km_json_file;
end

function [T_join, cfg] = select_joined_recovery_examples(T, json_mask, key_vars, keep_vars, cfg)
    montage_options = string(cfg.montage_type);
    if cfg.allow_montage_fallback
        montage_options = unique([montage_options; string(T.montage_type)], 'stable');
    end

    spm_criterion_options = string(cfg.spm_criterion);
    if cfg.allow_spm_criterion_fallback
        spm_rows = strtrim(lower(T.method)) == strtrim(lower(string(cfg.spm_method)));
        spm_criterion_options = unique([spm_criterion_options; string(T.criterion(spm_rows))], 'stable');
    end

    for m = 1:numel(montage_options)
        montage = montage_options(m);
        base_mask = json_mask & strtrim(lower(T.montage_type)) == strtrim(lower(montage));
        if isfinite(cfg.K_true) && ismember('K_true', T.Properties.VariableNames)
            base_mask = base_mask & double(T.K_true) == double(cfg.K_true);
        end
        for c = 1:numel(spm_criterion_options)
            spm_criterion = spm_criterion_options(c);
            candidate = join_requested_method_rows(T, base_mask, key_vars, keep_vars, cfg, spm_criterion);
            if isempty(candidate)
                continue;
            end
            if cfg.require_exact_k_estimated
                exact = double(candidate.K_estimated_spm) == double(candidate.K_true) & ...
                    double(candidate.K_estimated_kmeans) == double(candidate.K_true);
                candidate = candidate(exact, :);
            end
            if ~isempty(candidate)
                cfg.montage_type = char(montage);
                cfg.spm_criterion = char(spm_criterion);
                T_join = candidate;
                fprintf('Using simulation recovery example pool: montage=%s | K_true=%g | K-Means K_est=K_true | VB-GMM=%s K_est=K_true | n=%d\n', ...
                    cfg.montage_type, double(cfg.K_true), cfg.spm_criterion, height(T_join));
                return;
            end
        end
    end

    error(['No simulation rows satisfy the recovery-plot requirements. ', ...
        'Need K_true=%g with K_estimated=K_true for both K-Means/%s and VB-GMM.'], ...
        double(cfg.K_true), cfg.kmeans_criterion);
end

function T_join = join_requested_method_rows(T, base_mask, key_vars, keep_vars, cfg, spm_criterion)
    spm_mask = base_mask & ...
        strtrim(lower(T.method)) == strtrim(lower(string(cfg.spm_method))) & ...
        strtrim(lower(T.criterion)) == strtrim(lower(spm_criterion));
    km_mask = base_mask & ...
        strtrim(lower(T.method)) == strtrim(lower(string(cfg.kmeans_method))) & ...
        strtrim(lower(T.criterion)) == strtrim(lower(string(cfg.kmeans_criterion)));

    if ~any(spm_mask) || ~any(km_mask)
        T_join = table();
        return;
    end

    T_spm = T(spm_mask, keep_vars);
    T_km = T(km_mask, keep_vars);
    T_spm = renamevars(T_spm, {'json_file', 'K_estimated'}, {'spm_json_file', 'K_estimated_spm'});
    T_km = renamevars(T_km, {'json_file', 'K_estimated'}, {'kmeans_json_file', 'K_estimated_kmeans'});
    T_join = innerjoin(T_spm, T_km, 'Keys', key_vars);
end

function tf = has_usable_json_file(paths, output_dir, json_dir)
    tf = false(size(paths));
    for i = 1:numel(paths)
        pth = resolve_json_path(paths(i), output_dir, json_dir);
        tf(i) = isfile(pth);
    end
end

function json_file = resolve_json_path(raw_path, output_dir, json_dir)
    raw_path = char(string(raw_path));
    if isempty(strtrim(raw_path))
        json_file = '';
        return;
    end
    if isfile(raw_path)
        json_file = raw_path;
        return;
    end
    candidate = fullfile(output_dir, raw_path);
    if isfile(candidate)
        json_file = candidate;
        return;
    end
    candidate = fullfile(json_dir, raw_path);
    if isfile(candidate)
        json_file = candidate;
        return;
    end
    [~, base, ext] = fileparts(raw_path);
    candidate = fullfile(json_dir, [base ext]);
    if isfile(candidate)
        json_file = candidate;
    else
        json_file = raw_path;
    end
end

function [display_labels, chanlocs] = resolve_display_chanlocs(source_labels, template_file, template_K)
    [~, ~, template_channel_labels, chanlocs_all] = load_metamaps_templates(template_file, 'K', template_K);
    [template_idx, ~] = match_source_channels_to_template(source_labels, template_channel_labels);
    display_labels = template_channel_labels(template_idx);
    chanlocs = chanlocs_all(template_idx);
end

function [template_idx, source_idx] = match_source_channels_to_template(source_labels, template_labels)
    source_labels_l = cellfun(@(s) lower(strtrim(char(s))), cellstr(source_labels(:)), 'UniformOutput', false);
    template_labels_l = cellfun(@(s) lower(strtrim(char(s))), cellstr(template_labels(:)), 'UniformOutput', false);
    template_idx = [];
    source_idx = [];
    for t = 1:numel(template_labels_l)
        hit = find(strcmp(template_labels_l{t}, source_labels_l), 1, 'first');
        if ~isempty(hit)
            template_idx(end+1) = t; %#ok<AGROW>
            source_idx(end+1) = hit; %#ok<AGROW>
        end
    end
end

function valid = has_usable_topoplot_location(chanlocs)
    valid = false(1, numel(chanlocs));
    for i = 1:numel(chanlocs)
        has_polar = isfield(chanlocs(i), 'theta') && ~isempty(chanlocs(i).theta) && ...
            isfield(chanlocs(i), 'radius') && ~isempty(chanlocs(i).radius) && ...
            isfinite(chanlocs(i).theta) && isfinite(chanlocs(i).radius) && ...
            chanlocs(i).radius > 0 && chanlocs(i).radius <= 0.5;
        has_xyz = isfield(chanlocs(i), 'X') && ~isempty(chanlocs(i).X) && ...
            isfield(chanlocs(i), 'Y') && ~isempty(chanlocs(i).Y) && ...
            isfield(chanlocs(i), 'Z') && ~isempty(chanlocs(i).Z) && ...
            all(isfinite([chanlocs(i).X chanlocs(i).Y chanlocs(i).Z])) && ...
            norm([chanlocs(i).X chanlocs(i).Y chanlocs(i).Z]) > eps;
        valid(i) = has_polar || (~isfield(chanlocs(i), 'radius') && has_xyz);
    end
end

function chanlocs = rotate_topoplot_chanlocs(chanlocs, angle_deg)
    if isempty(chanlocs) || ~isfinite(angle_deg) || abs(angle_deg) <= eps
        return;
    end
    for i = 1:numel(chanlocs)
        if isfield(chanlocs(i), 'theta') && ~isempty(chanlocs(i).theta) && isfinite(chanlocs(i).theta)
            chanlocs(i).theta = double(chanlocs(i).theta) + double(angle_deg);
        elseif all(isfield(chanlocs, {'X', 'Y'})) && ~isempty(chanlocs(i).X) && ~isempty(chanlocs(i).Y)
            xy = rotate_xy([double(chanlocs(i).X); double(chanlocs(i).Y)], angle_deg);
            chanlocs(i).X = xy(1);
            chanlocs(i).Y = xy(2);
        end
        if isfield(chanlocs(i), 'sph_theta') && ~isempty(chanlocs(i).sph_theta) && isfinite(chanlocs(i).sph_theta)
            chanlocs(i).sph_theta = double(chanlocs(i).sph_theta) + double(angle_deg);
        end
    end
end

function xy_out = rotate_xy(xy, angle_deg)
    theta = deg2rad(double(angle_deg));
    R = [cos(theta), -sin(theta); sin(theta), cos(theta)];
    xy_out = R * xy;
end

function [maps, channel_labels] = extract_state_maps(data, field_name)
    if ~isfield(data, field_name) || isempty(data.(field_name))
        error('JSON payload is missing %s.', field_name);
    end
    state_struct = data.(field_name);
    state_names = fieldnames(state_struct);
    state_order = zeros(numel(state_names), 1);
    for i = 1:numel(state_names)
        tok = regexp(state_names{i}, '^state_(\d+)$', 'tokens', 'once');
        if isempty(tok)
            state_order(i) = i;
        else
            state_order(i) = str2double(tok{1});
        end
    end
    [~, sort_idx] = sort(state_order);
    state_names = state_names(sort_idx);

    channel_labels = cellstr(string(data.channel_info.labels));
    if isfield(data.channel_info, 'labels_sanitized')
        channel_keys = cellstr(string(data.channel_info.labels_sanitized));
    else
        channel_keys = sanitize_channel_labels_plot(channel_labels);
    end

    n_states = numel(state_names);
    n_channels = numel(channel_labels);
    maps = nan(n_states, n_channels);
    for s = 1:n_states
        state_data = state_struct.(state_names{s});
        for c = 1:n_channels
            if isfield(state_data, channel_keys{c})
                maps(s, c) = double(state_data.(channel_keys{c}));
            end
        end
    end
end

function template_labels = pick_template_labels(varargin)
    template_labels = {};
    for i = 1:nargin
        A = varargin{i};
        if isfield(A, 'template_labels') && ~isempty(A.template_labels)
            template_labels = cellstr(A.template_labels);
            return;
        end
    end
end

function row = alignment_to_display_row(alignment, source_channel_labels, display_labels, template_labels)
    n_states = size(alignment.aligned_maps, 1);
    label_order = inf(n_states, 1);
    for i = 1:n_states
        hit = find(strcmp(alignment.labels{i}, template_labels), 1, 'first');
        if ~isempty(hit)
            label_order(i) = hit;
        end
    end
    [~, order] = sort(label_order);

    maps_ordered = alignment.aligned_maps(order, :);
    labels_ordered = alignment.labels(order);
    corr_ordered = alignment.correlations(order);

    maps_display = nan(size(maps_ordered, 1), numel(display_labels));
    source_labels_l = cellfun(@(s) lower(strtrim(char(s))), cellstr(source_channel_labels(:)), 'UniformOutput', false);
    display_labels_l = cellfun(@(s) lower(strtrim(char(s))), cellstr(display_labels(:)), 'UniformOutput', false);
    [lia, locb] = ismember(display_labels_l, source_labels_l);
    for c = 1:numel(display_labels)
        if lia(c)
            maps_display(:, c) = maps_ordered(:, locb(c));
        end
    end

    keep = all(isfinite(maps_display), 2);
    row = struct();
    row.labels = labels_ordered(keep);
    row.correlations = corr_ordered(keep);
    row.maps = maps_display(keep, :);
end

function labels_out = intersect_labels_stable(labels_a, labels_b)
    a = cellstr(labels_a(:));
    b = cellstr(labels_b(:));
    a_l = cellfun(@(s) lower(strtrim(char(s))), a, 'UniformOutput', false);
    b_l = cellfun(@(s) lower(strtrim(char(s))), b, 'UniformOutput', false);
    keep = ismember(a_l, b_l);
    labels_out = a(keep);
end

function column_labels = used_template_columns(template_labels, varargin)
    present = {};
    for i = 1:numel(varargin)
        row = varargin{i};
        present = [present; row.labels(:)]; %#ok<AGROW>
    end
    present = unique(present, 'stable');
    column_labels = {};
    for i = 1:numel(template_labels)
        if any(strcmp(template_labels{i}, present))
            column_labels{end+1} = template_labels{i}; %#ok<AGROW>
        end
    end
    for i = 1:numel(present)
        if ~any(strcmp(present{i}, column_labels))
            column_labels{end+1} = present{i}; %#ok<AGROW>
        end
    end
end

function row_out = project_row_to_columns(row_in, column_labels)
    n_cols = numel(column_labels);
    row_out = struct();
    row_out.labels = column_labels(:)';
    row_out.maps = nan(n_cols, size(row_in.maps, 2));
    row_out.correlations = nan(n_cols, 1);
    for i = 1:numel(row_in.labels)
        hit = find(strcmp(row_in.labels{i}, column_labels), 1, 'first');
        if ~isempty(hit)
            row_out.maps(hit, :) = row_in.maps(i, :);
            row_out.correlations(hit) = row_in.correlations(i);
        end
    end
end

function name = default_plot_filename(selection, seed, montage_type)
    rep = double(selection.rep(1));
    K_true = double(selection.K_true(1));
    SNR_dB = double(selection.SNR_dB(1));
    overlap_prob = double(selection.overlap_prob(1));
    name = sprintf('simulated_microstate_comparison_seed%d_rep%d_K%d_SNR%s_ovl%03d_%s.png', ...
        round(seed), rep, K_true, signed_token(SNR_dB), round(100 * overlap_prob), safe_token(montage_type));
end

function txt = selection_title(selection, selected_idx, n_candidates, cfg)
    txt = sprintf(['Simulated EEG comparison | selection %d/%d via seed %d | rep=%d | ', ...
        'K_{true}=%d | SNR=%+g dB | overlap=%.2f | montage=%s | ', ...
        'K-Means=%s/%s (K=%d) | VB-GMM=%s/%s (K=%d)'], ...
        selected_idx, n_candidates, round(cfg.seed), ...
        double(selection.rep(1)), double(selection.K_true(1)), double(selection.SNR_dB(1)), ...
        double(selection.overlap_prob(1)), char(selection.montage_type(1)), ...
        cfg.kmeans_method, cfg.kmeans_criterion, double(selection.K_estimated_kmeans(1)), ...
        cfg.spm_method, cfg.spm_criterion, double(selection.K_estimated_spm(1)));
end

function txt = prettify_token(txt)
    txt = strrep(char(string(txt)), '_', '-');
end

function txt = signed_token(x)
    if x >= 0
        txt = sprintf('p%g', x);
    else
        txt = sprintf('m%g', abs(x));
    end
    txt = strrep(txt, '.', 'p');
end

function txt = safe_token(txt)
    txt = regexprep(char(string(txt)), '[^A-Za-z0-9]+', '_');
    txt = regexprep(txt, '_+', '_');
    txt = regexprep(txt, '^_|_$', '');
end

function value = onoff(tf)
    if tf
        value = 'on';
    else
        value = 'off';
    end
end

function close_if_valid(fig)
    if ~isempty(fig) && isgraphics(fig)
        close(fig);
    end
end

function sanitized = sanitize_channel_labels_plot(ch_labels)
    sanitized = cell(size(ch_labels));
    for i = 1:length(ch_labels)
        label = char(ch_labels{i});
        label = regexprep(label, '[-/\\\s\.\,\(\)\[\]\{\}]', '_');
        label = regexprep(label, '^_+|_+$', '');
        if isempty(label) || isempty(regexp(label(1), '[A-Za-z]', 'once'))
            label = ['Ch' label];
        end
        sanitized{i} = matlab.lang.makeValidName(label);
    end
end
