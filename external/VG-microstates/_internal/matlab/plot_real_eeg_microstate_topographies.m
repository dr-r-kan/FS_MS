function plot_file = plot_real_eeg_microstate_topographies(json_file, eeg_file, plot_file)
% PLOT_REAL_EEG_MICROSTATE_TOPOGRAPHIES Plot estimated microstates as scalp maps.
%
% Uses montage/channel-location information from the real EEG .set file,
% rather than treating microstate vectors as images.

    if nargin < 3 || isempty(plot_file)
        [json_dir, json_name, ~] = fileparts(json_file);
        plot_file = fullfile(json_dir, [json_name '_microstates.png']);
    end

    if ~exist('pop_loadset', 'file')
        error('EEGLAB pop_loadset is not on the MATLAB path.');
    end
    if ~exist('topoplot', 'file')
        error('EEGLAB topoplot is not on the MATLAB path.');
    end
    if ~exist(json_file, 'file')
        error('JSON file not found: %s', json_file);
    end
    if ~exist(eeg_file, 'file')
        error('EEG file not found: %s', eeg_file);
    end

    data = jsondecode(fileread(json_file));
    EEG = pop_loadset('filename', eeg_file);
    chanlocs_all = EEG.chanlocs;

    json_labels = cellstr(data.channel_info.labels);
    if isfield(data.channel_info, 'labels_sanitized')
        json_labels_sanitized = cellstr(data.channel_info.labels_sanitized);
    else
        json_labels_sanitized = sanitize_channel_labels_plot(json_labels);
    end

    [json_idx, set_idx] = match_json_channels_to_set(json_labels, chanlocs_all);
    [chanlocs, keep_idx] = prepare_real_topoplot_chanlocs(chanlocs_all(set_idx));
    json_idx = json_idx(keep_idx);
    set_idx = set_idx(keep_idx); %#ok<NASGU>
    if numel(json_idx) < 4
        error('Only %d channels could be matched to usable scalp locations.', numel(json_idx));
    end

    est_states = fieldnames(data.estimated_microstates);
    est_states = sort_estimated_state_keys(data.estimated_microstates, est_states);
    K = numel(est_states);
    maps = zeros(K, numel(json_idx));

    for k = 1:K
        state_data = data.estimated_microstates.(est_states{k});
        for c = 1:numel(json_idx)
            ch_key = json_labels_sanitized{json_idx(c)};
            if isfield(state_data, ch_key)
                maps(k, c) = state_data.(ch_key);
            else
                maps(k, c) = NaN;
            end
        end
    end

    keep_ch = all(isfinite(maps), 1);
    maps = maps(:, keep_ch);
    chanlocs = chanlocs(keep_ch);
    if numel(chanlocs) < 4
        error('Fewer than four finite mapped channels remain after JSON extraction.');
    end

    clim_abs = max(abs(maps(:)));
    if clim_abs <= eps || ~isfinite(clim_abs)
        clim_abs = 1;
    end

    n_cols = min(4, K);
    n_rows = ceil(K / n_cols);
    fig = figure('Name', 'Microstate Topographies', ...
        'NumberTitle', 'off', ...
        'Color', 'white', ...
        'Visible', 'off', ...
        'Position', [100, 100, 280*n_cols, 260*n_rows + 80]);

    for k = 1:K
        subplot(n_rows, n_cols, k);
        topoplot(maps(k, :), chanlocs, ...
            'electrodes', 'off', ...
            'numcontour', 6, ...
            'maplimits', [-clim_abs clim_abs]);
        colormap(gca, 'jet');
        colorbar;
        title(state_title(data.estimated_microstates.(est_states{k}), k), ...
            'FontSize', 10, 'FontWeight', 'bold', 'Interpreter', 'none');
    end

    title_text = json_title(data, json_file);
    sgtitle(title_text, 'FontSize', 12, 'FontWeight', 'bold', 'Interpreter', 'none');

    out_dir = fileparts(plot_file);
    if ~isempty(out_dir) && ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end
    exportgraphics(fig, plot_file, 'Resolution', 200);
    close(fig);
end

function [json_idx, set_idx] = match_json_channels_to_set(json_labels, chanlocs)
    set_labels = cell(numel(chanlocs), 1);
    for i = 1:numel(chanlocs)
        set_labels{i} = lower(strtrim(char(chanlocs(i).labels)));
    end
    json_idx = [];
    set_idx = [];
    for j = 1:numel(json_labels)
        label = lower(strtrim(char(json_labels{j})));
        hit = find(strcmp(label, set_labels), 1, 'first');
        if ~isempty(hit)
            json_idx(end+1) = j; %#ok<AGROW>
            set_idx(end+1) = hit; %#ok<AGROW>
        end
    end
end

function [chanlocs_out, keep_idx] = prepare_real_topoplot_chanlocs(chanlocs_in)
    n = numel(chanlocs_in);
    chanlocs_out = repmat(minimal_chanloc_template(), 1, 0);
    keep_idx = [];
    if n == 0
        return;
    end

    keep = false(1, n);
    out = repmat(minimal_chanloc_template(), 1, n);

    for i = 1:n
        out(i) = sanitize_chanloc_struct(chanlocs_in(i), i);

        xyz = [ ...
            local_scalar_numeric(getfield_if_present(out(i), 'X')), ...
            local_scalar_numeric(getfield_if_present(out(i), 'Y')), ...
            local_scalar_numeric(getfield_if_present(out(i), 'Z'))];
        if all(isfinite(xyz)) && norm(xyz) > eps
            xyz = xyz ./ norm(xyz);
            keep(i) = true;
            out(i) = fill_chanloc_geometry(out(i), xyz);
            continue;
        end

        theta = local_scalar_numeric(getfield_if_present(out(i), 'theta'));
        radius = local_scalar_numeric(getfield_if_present(out(i), 'radius'));
        if isfinite(theta) && isfinite(radius) && radius > 0 && radius <= 0.5
            x = radius * cosd(theta);
            y = radius * sind(theta);
            z = sqrt(max(0, 1 - min(1, radius / 0.5) .^ 2));
            xyz = [x y z];
            if all(isfinite(xyz)) && norm(xyz) > eps
                xyz = xyz ./ norm(xyz);
                keep(i) = true;
                out(i) = fill_chanloc_geometry(out(i), xyz, theta, radius);
            end
        end
    end

    keep_idx = find(keep);
    chanlocs_out = out(keep_idx);
    if numel(chanlocs_out) < 4
        error('Fewer than four matched channels have usable topoplot geometry.');
    end
end

function out = sanitize_chanloc_struct(in, idx)
    out = minimal_chanloc_template();
    if ~isfield(in, 'labels') || isempty(in.labels)
        out.labels = sprintf('Ch%d', idx);
    else
        out.labels = char(string(in.labels));
    end

    numeric_fields = {'X', 'Y', 'Z', 'theta', 'radius', 'sph_theta', 'sph_phi', 'sph_radius'};
    for f = 1:numel(numeric_fields)
        name = numeric_fields{f};
        if isfield(in, name)
            val = local_scalar_numeric(in.(name));
        else
            val = NaN;
        end
        if isfinite(val)
            out.(name) = val;
        else
            out.(name) = [];
        end
    end
end

function out = fill_chanloc_geometry(out, xyz, theta_in, radius_in)
    if nargin < 3
        theta_in = NaN;
    end
    if nargin < 4
        radius_in = NaN;
    end
    xyz = double(xyz(:)');
    xyz = xyz ./ max(norm(xyz), eps);
    out.X = xyz(1);
    out.Y = xyz(2);
    out.Z = xyz(3);

    if ~(isfinite(theta_in) && isfinite(radius_in) && radius_in > 0 && radius_in <= 0.5)
        [az, el] = cart2sph(xyz(1), xyz(2), xyz(3));
        [~, theta_in, radius_in] = sph2topo([1 rad2deg(el) rad2deg(az)], 1, 2);
    end
    out.theta = theta_in;
    out.radius = radius_in;
    out.sph_theta = [];
    out.sph_phi = [];
    out.sph_radius = [];
end

function s = minimal_chanloc_template()
    s = struct( ...
        'labels', '', ...
        'theta', [], ...
        'radius', [], ...
        'X', [], ...
        'Y', [], ...
        'Z', [], ...
        'sph_theta', [], ...
        'sph_phi', [], ...
        'sph_radius', []);
end

function v = getfield_if_present(S, name)
    if isfield(S, name)
        v = S.(name);
    else
        v = [];
    end
end

function x = local_scalar_numeric(v)
    if isempty(v)
        x = NaN;
        return;
    end
    if isnumeric(v) || islogical(v)
        v = double(v(:));
        x = v(1);
        return;
    end
    if ischar(v) || (isstring(v) && isscalar(v))
        x = str2double(char(v));
        if ~isfinite(x)
            x = NaN;
        end
        return;
    end
    try
        v = double(v);
        v = v(:);
        if isempty(v)
            x = NaN;
        else
            x = v(1);
        end
    catch
        x = NaN;
    end
end

function title_text = state_title(state_data, k)
    title_text = sprintf('State %d', k);
    if isfield(state_data, 'template_label')
        title_text = sprintf('%s', char(string(state_data.template_label)));
        detail = sprintf('state %d', k);
        if isfield(state_data, 'template_correlation')
            detail = sprintf('%s, r=%.2f', detail, state_data.template_correlation);
        end
        title_text = sprintf('%s (%s)', title_text, detail);
    end
end

function ordered = sort_estimated_state_keys(estimated_microstates, est_states)
    labels = cell(numel(est_states), 1);
    for i = 1:numel(est_states)
        state_data = estimated_microstates.(est_states{i});
        if isfield(state_data, 'template_label') && ~isempty(state_data.template_label)
            labels{i} = upper(strtrim(char(string(state_data.template_label))));
        else
            labels{i} = sprintf('ZZZ_%04d', i);
        end
    end
    [~, ord] = sort(labels);
    ordered = est_states(ord);
end

function title_text = json_title(data, json_file)
    [~, json_name, ~] = fileparts(json_file);
    title_text = json_name;
    if isfield(data, 'metadata')
        meta = data.metadata;
        parts = {json_name};
        if isfield(meta, 'method')
            parts{end+1} = sprintf('Method: %s', meta.method); %#ok<AGROW>
        end
        if isfield(meta, 'criterion')
            parts{end+1} = sprintf('Criterion: %s', meta.criterion); %#ok<AGROW>
        end
        if isfield(meta, 'K_estimated')
            parts{end+1} = sprintf('K: %d', meta.K_estimated); %#ok<AGROW>
        end
        title_text = strjoin(parts, ' | ');
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
