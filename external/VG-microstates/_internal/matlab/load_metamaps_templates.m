function [template_maps, template_labels, channel_labels, chanlocs] = load_metamaps_templates(template_file, varargin)
% LOAD_METAMAPS_TEMPLATES Load MetaMaps microstate templates from an EEGLAB .set file.
%
% Returns maps as K x channels, zero-mean/unit-norm per map.
%
% The MetaMaps_2023_06.set file stores labelled solutions in
% EEG.msinfo.MSMaps(K).  Older/fallback readers may also see concatenated
% solutions in EEG.data:
%   indices  1:4   -> K=4
%   indices  5:9   -> K=5
%   indices 10:15  -> K=6
%   indices 16:22  -> K=7
%
% Returned labels/maps are reordered into alphabetical canonical order for
% consistent plotting.

    p = inputParser;
    addRequired(p, 'template_file', @(x) ischar(x) || isstring(x));
    addParameter(p, 'K', 7, @isnumeric);
    parse(p, template_file, varargin{:});
    K = p.Results.K;

    if ~exist(char(template_file), 'file')
        error('Template file not found: %s', char(template_file));
    end
    EEG = load_template_set(char(template_file));
    [template_maps, template_labels] = maps_from_msinfo(EEG, K);
    if isempty(template_maps)
        [template_maps, template_labels] = maps_from_raw_data(EEG, K);
    end

    chanlocs = [];
    util = microstate_utilities();
    if isfield(EEG, 'chanlocs') && ~isempty(EEG.chanlocs)
        chanlocs = EEG.chanlocs;
        n_channels = min(numel(chanlocs), size(template_maps, 2));
        chanlocs = chanlocs(1:n_channels);
        channel_labels = cell(1, n_channels);
        for c = 1:n_channels
            if isfield(chanlocs(c), 'labels') && ~isempty(chanlocs(c).labels)
                channel_labels{c} = char(chanlocs(c).labels);
            else
                channel_labels{c} = sprintf('Ch%03d', c);
            end
        end
        template_maps = template_maps(:, 1:n_channels);
        [chanlocs, keep_idx] = util.prepare_metamaps_chanlocs(chanlocs, n_channels);
        if numel(chanlocs) < n_channels
            template_maps = template_maps(:, keep_idx);
            channel_labels = channel_labels(keep_idx);
        end
    else
        channel_labels = arrayfun(@(c) sprintf('Ch%03d', c), 1:size(template_maps, 2), 'UniformOutput', false);
    end

    [template_labels, sort_idx] = sort_microstate_labels(template_labels);
    template_maps = template_maps(sort_idx, :);
    template_maps = util.normalize_maps(template_maps);
end

function EEG = load_template_set(template_file)
    if exist('pop_loadset', 'file') == 2
        EEG = pop_loadset('filename', template_file);
        return;
    end

    EEG = load(template_file, '-mat');
    if isfield(EEG, 'EEG')
        EEG = EEG.EEG;
    end
end

function [template_maps, template_labels] = maps_from_msinfo(EEG, K)
    template_maps = [];
    template_labels = {};
    if ~isfield(EEG, 'msinfo') || ~isfield(EEG.msinfo, 'MSMaps') || numel(EEG.msinfo.MSMaps) < K
        return;
    end

    rec = EEG.msinfo.MSMaps(K);
    if ~isfield(rec, 'Maps') || isempty(rec.Maps)
        return;
    end

    template_maps = double(rec.Maps);
    if size(template_maps, 1) ~= K && size(template_maps, 2) == K
        template_maps = template_maps';
    end
    if size(template_maps, 1) ~= K
        error('MetaMaps msinfo entry %d has %d maps, expected %d.', K, size(template_maps, 1), K);
    end

    if isfield(rec, 'Labels') && numel(rec.Labels) >= K
        template_labels = cellstr(string(rec.Labels));
        template_labels = template_labels(1:K);
    else
        template_labels = arrayfun(@(i) char('A' + i - 1), 1:K, 'UniformOutput', false);
    end
end

function [template_maps, template_labels] = maps_from_raw_data(EEG, K)
    data = double(squeeze(EEG.data));
    if ~ismatrix(data)
        error('Template data must be a 2-D channels x maps or maps x channels matrix.');
    end

    if isfield(EEG, 'nbchan') && size(data, 1) == EEG.nbchan
        all_maps = data';
    elseif isfield(EEG, 'nbchan') && size(data, 2) == EEG.nbchan
        all_maps = data;
    elseif size(data, 1) > size(data, 2)
        all_maps = data';
    else
        all_maps = data;
    end

    n_maps_total = size(all_maps, 1);
    if n_maps_total >= 22 && K >= 4 && K <= 7
        idx_start = 1 + sum(4:(K - 1));
        idx = idx_start:(idx_start + K - 1);
        template_labels = raw_data_labels(K);
    elseif K <= n_maps_total
        idx = (n_maps_total - K + 1):n_maps_total;
        template_labels = arrayfun(@(i) char('A' + i - 1), 1:K, 'UniformOutput', false);
    else
        error('Requested K=%d templates, but only %d maps were found.', K, n_maps_total);
    end

    template_maps = all_maps(idx, :);
end

function labels = raw_data_labels(K)
    switch K
        case 4
            labels = {'B', 'C', 'A', 'D'};
        case 5
            labels = {'D', 'C', 'E', 'B', 'A'};
        case 6
            labels = {'E', 'C', 'A', 'G', 'D', 'B'};
        case 7
            labels = {'D', 'A', 'C', 'F', 'B', 'G', 'E'};
        otherwise
            labels = arrayfun(@(i) char('A' + i - 1), 1:K, 'UniformOutput', false);
    end
end

function [labels_out, sort_idx] = sort_microstate_labels(labels_in)
    labels = cellstr(string(labels_in(:)));
    keys = lower(strtrim(labels));
    [~, sort_idx] = sort(keys);
    labels_out = labels(sort_idx);
end
