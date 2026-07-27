function out_files = plot_metamaps_template_microstates(template_file, varargin)
% PLOT_METAMAPS_TEMPLATE_MICROSTATES Plot all MetaMaps template microstates.
%
% This uses the same topography rendering pattern as the other plotting
% helpers in this repository: tiled topoplots, shared color scaling, and
% exportgraphics output.
%
% Usage:
%   plot_metamaps_template_microstates();
%   plot_metamaps_template_microstates('MetaMaps_2023_06.set');
%   plot_metamaps_template_microstates('MetaMaps_2023_06.set', 'K_values', 4:7);

    setup_microstate_paths();
    if nargin < 1 || isempty(template_file)
        template_file = default_template_file();
    end

    p = inputParser;
    addRequired(p, 'template_file', @(x) ischar(x) || isstring(x));
    addParameter(p, 'config_file', 'microstate_config.json', @(x) ischar(x) || isstring(x));
    addParameter(p, 'output_dir', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'file_prefix', 'metamaps_template_microstates', @(x) ischar(x) || isstring(x));
    addParameter(p, 'format', 'png', @(x) ischar(x) || isstring(x));
    addParameter(p, 'resolution', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x) && x > 0));
    addParameter(p, 'display_normalisation', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'maplimits', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'map_percentile', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x) && x > 0 && x <= 100));
    addParameter(p, 'colormap_name', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'electrodes', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'visible', false, @(x) islogical(x) && isscalar(x));
    addParameter(p, 'K_values', 4:7, @(x) isnumeric(x) && ~isempty(x));
    parse(p, template_file, varargin{:});
    cfg = p.Results;

    cfg.template_file = char(cfg.template_file);
    cfg.config_file = char(cfg.config_file);
    cfg.file_prefix = char(cfg.file_prefix);
    cfg.format = lower(char(cfg.format));
    cfg = apply_config_defaults(cfg);

    if ~isfile(cfg.template_file)
        error('Template file not found: %s', cfg.template_file);
    end
    if exist('topoplot', 'file') ~= 2
        error('EEGLAB topoplot is not on the MATLAB path. Start EEGLAB or add it to the path first.');
    end
    if ~exist(cfg.output_dir, 'dir')
        mkdir(cfg.output_dir);
    end

    K_values = unique(double(cfg.K_values(:))', 'stable');
    out_files = cell(numel(K_values), 1);

    for i = 1:numel(K_values)
        K = K_values(i);
        [maps, labels, ~, chanlocs] = load_metamaps_templates(cfg.template_file, 'K', K);
        [chanlocs, keep_idx] = get_usable_chanlocs(chanlocs, size(maps, 2));
        maps = maps(:, keep_idx);
        maps = normalise_maps_2d(maps);
        maps_plot = normalise_for_display(maps, cfg.display_normalisation);
        clim_abs = compute_clim(maps_plot, cfg);

        n_states = size(maps_plot, 1);
        n_cols = min(4, n_states);
        n_rows = ceil(n_states / n_cols);
        fig_visible = ternary_char(cfg.visible, 'on', 'off');
        fig = figure('Name', sprintf('MetaMaps template microstates K=%d', K), ...
            'NumberTitle', 'off', 'Color', 'white', 'Visible', fig_visible, ...
            'Position', [100, 100, max(900, 260*n_cols), 260*n_rows + 110]);

        for k = 1:n_states
            subplot(n_rows, n_cols, k);
            plot_one_topomap(maps_plot(k, :), chanlocs, cfg, clim_abs);
            title(labels{k}, 'FontSize', 10, 'FontWeight', 'bold', 'Interpreter', 'none');
        end

        title_text = sprintf('MetaMaps template microstates | K=%d | %s', K, short_path(cfg.template_file));
        sgtitle(title_text, 'FontSize', 12, 'FontWeight', 'bold', 'Interpreter', 'none');

        out_fig = fullfile(cfg.output_dir, sprintf('%s_K%d.%s', cfg.file_prefix, K, cfg.format));
        export_figure(fig, out_fig, cfg);
        if ~cfg.visible
            close(fig);
        end

        out_files{i} = out_fig; %#ok<AGROW>
        fprintf('Saved template microstate plot: %s\n', out_fig);
    end
end

% ======================================================================
% Helpers
% ======================================================================

function cfg = apply_config_defaults(cfg)
    util = microstate_utilities();
    C = util.load_config(cfg.config_file);

    if isempty(cfg.output_dir)
        if isfield(C, 'paths') && isfield(C.paths, 'single_plot_dir')
            cfg.output_dir = fullfile(char(C.paths.single_plot_dir), 'template_microstates');
        elseif isfield(C, 'paths') && isfield(C.paths, 'diagnostic_output_dir')
            cfg.output_dir = fullfile(char(C.paths.diagnostic_output_dir), 'template_microstates');
        else
            cfg.output_dir = fullfile('outputs', 'plots', 'template_microstates');
        end
    else
        cfg.output_dir = char(cfg.output_dir);
    end

    if isempty(cfg.template_file) && isfield(C, 'paths') && isfield(C.paths, 'template_file')
        cfg.template_file = char(C.paths.template_file);
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
    if isempty(cfg.resolution)
        cfg.resolution = double(get_nested_or(C, {'plotting', 'resolution'}, 300));
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

function results = default_template_file()
    util = microstate_utilities();
    cfg = util.load_config();
    if isfield(cfg, 'paths') && isfield(cfg.paths, 'template_file') && ~isempty(cfg.paths.template_file)
        results = char(cfg.paths.template_file);
    else
        results = 'MetaMaps_2023_06.set';
    end
end

function [chanlocs, keep_idx] = get_usable_chanlocs(chanlocs_all, n_channels)
    if numel(chanlocs_all) ~= n_channels
        error('Number of channel locations (%d) does not match map width (%d).', numel(chanlocs_all), n_channels);
    end
    keep = has_usable_topoplot_location(chanlocs_all);
    keep_idx = find(keep);
    chanlocs = chanlocs_all(keep_idx);
    if numel(chanlocs) < 4
        error('Fewer than four usable scalp channel locations are available.');
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
        if ~isfinite(local) || local <= eps
            local = clim_abs;
        end
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

function s = short_path(p)
    [parent, name, ext] = fileparts(p);
    [~, parent_name] = fileparts(parent);
    s = fullfile(parent_name, [name ext]);
end

function y = ternary_char(tf, a, b)
    if tf
        y = a;
    else
        y = b;
    end
end
