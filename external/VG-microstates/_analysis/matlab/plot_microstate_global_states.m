function out_files = plot_microstate_global_states(results_mat, varargin)
% PLOT_MICROSTATE_GLOBAL_STATES
%
% Plot the global microstate templates from fit_microstate_hierarchical_dataset.m.
%
% Usage:
%   plot_microstate_global_states();
%   plot_microstate_global_states('outputs/hierarchical_microstates/hierarchical_microstate_results.mat');
%   plot_microstate_global_states(results_mat, 'output_dir', 'outputs/plots/global');
%
% The script uses HResults.common_chanlocs and EEGLAB topoplot. It therefore
% requires EEGLAB, but it does not need the original EEG file.

    if nargin < 1 || isempty(results_mat)
        results_mat = default_results_mat();
    end

    p = inputParser;
    addRequired(p, 'results_mat', @(x) ischar(x) || isstring(x));
    addParameter(p, 'config_file', 'microstate_config.json', @(x) ischar(x) || isstring(x));
    addParameter(p, 'output_dir', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'file_prefix', 'global_microstate_templates', @(x) ischar(x) || isstring(x));
    addParameter(p, 'format', 'png', @(x) ischar(x) || isstring(x));
    addParameter(p, 'resolution', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x) && x > 0));
    addParameter(p, 'display_normalisation', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'maplimits', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'map_percentile', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x) && x > 0 && x <= 100));
    addParameter(p, 'colormap_name', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'electrodes', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'visible', false, @(x) islogical(x) && isscalar(x));
    parse(p, results_mat, varargin{:});
    cfg = p.Results;

    cfg.results_mat = char(cfg.results_mat);
    cfg.config_file = char(cfg.config_file);
    cfg.file_prefix = char(cfg.file_prefix);
    cfg.format = lower(char(cfg.format));
    cfg = apply_config_defaults(cfg, 'global');

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
    if ~isfield(H, 'global') || ~isfield(H.global, 'centers') || isempty(H.global.centers)
        error('HResults.global.centers is missing or empty.');
    end

    maps = double(H.global.centers);
    if isfield(H, 'selected_K') && ~isempty(H.selected_K)
        maps = maps(1:min(size(maps, 1), double(H.selected_K)), :);
    end
    [chanlocs, keep_idx] = get_usable_chanlocs(H, size(maps, 2));
    maps = maps(:, keep_idx);
    maps = normalise_maps_2d(maps);
    maps_plot = normalise_for_display(maps, cfg.display_normalisation);

    labels = state_labels(H, size(maps_plot, 1));
    clim_abs = compute_clim(maps_plot, cfg);

    n_states = size(maps_plot, 1);
    n_cols = min(4, n_states);
    n_rows = ceil(n_states / n_cols);
    fig_visible = ternary_char(cfg.visible, 'on', 'off');
    fig = figure('Name', 'Global microstate templates', ...
        'NumberTitle', 'off', 'Color', 'white', 'Visible', fig_visible, ...
        'Position', [100, 100, max(900, 260*n_cols), 260*n_rows + 110]);

    for k = 1:n_states
        subplot(n_rows, n_cols, k);
        plot_one_topomap(maps_plot(k, :), chanlocs, cfg, clim_abs);
        title(labels{k}, 'FontSize', 10, 'FontWeight', 'bold', 'Interpreter', 'none');
    end

    title_text = sprintf('Global microstate templates | K=%d | %s', n_states, short_path(cfg.results_mat));
    sgtitle(title_text, 'FontSize', 12, 'FontWeight', 'bold', 'Interpreter', 'none');

    out_fig = fullfile(cfg.output_dir, [cfg.file_prefix '.' cfg.format]);
    export_figure(fig, out_fig, cfg);
    if ~cfg.visible
        close(fig);
    end

    out_mat = fullfile(cfg.output_dir, [cfg.file_prefix '.mat']);
    out_csv = fullfile(cfg.output_dir, [cfg.file_prefix '.csv']);
    plotted_maps = maps_plot; %#ok<NASGU>
    raw_normalised_maps = maps; %#ok<NASGU>
    plotted_channel_labels = channel_labels_from_chanlocs(chanlocs); %#ok<NASGU>
    save(out_mat, 'plotted_maps', 'raw_normalised_maps', 'plotted_channel_labels', 'labels', 'cfg', '-v7.3');
    write_map_csv(out_csv, maps_plot, labels, plotted_channel_labels);

    out_files = struct('figure', out_fig, 'mat', out_mat, 'csv', out_csv);
    fprintf('Saved global microstate plot: %s\n', out_fig);
end

% ======================================================================
% Helpers
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

function X = normalise_maps_2d(X)
    X = double(X);
    if isempty(X), return; end
    X = X - mean(X, 2, 'omitnan');
    denom = sqrt(mean(X .^ 2, 2, 'omitnan'));
    denom(~isfinite(denom) | denom <= eps) = 1;
    X = X ./ denom;
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

function write_map_csv(out_csv, maps, state_names, channel_labels)
    state = strings(size(maps, 1), 1);
    for i = 1:size(maps, 1)
        state(i) = string(state_names{i});
    end
    T = table(state);
    for c = 1:numel(channel_labels)
        T.(matlab.lang.makeValidName(channel_labels{c})) = maps(:, c);
    end
    writetable(T, out_csv);
end

function s = short_path(p)
    [parent, name, ext] = fileparts(p);
    [~, parent_name] = fileparts(parent);
    s = fullfile(parent_name, [name ext]);
end

function y = ternary_char(tf, a, b)
    if tf, y = a; else, y = b; end
end
