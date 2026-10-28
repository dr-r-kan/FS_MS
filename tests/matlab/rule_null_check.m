function summary = rule_null_check(n_sets, seed, out_json)
%RULE_NULL_CHECK Monte Carlo: how often does template_prespecified_v2 report labelling_quality 'ok' for six maps that are not microstates?
%
%   summary = rule_null_check(n_sets, seed, out_json)
%
%   Diagnostic for the quality flag (labelling_quality is 'ok' when |r| >= 0.5 for all six assigned states). On the
%   standard 19-channel montage, with the pipeline's own smoothing, the template classes are low-dimensional
%   smooth topographies, so unstructured maps can still clear 0.5 by chance. This script quantifies that, and
%   reports the margin statistics (assigned minus next-best |r|) that distinguish true from null maps.
%   Conditions: iid Gaussian maps; the same maps after the pipeline's default spatial smoothing; and true
%   template maps plus noise (with smoothing) as the reference.
    if nargin < 1, n_sets = 100; end
    if nargin < 2, seed = 11; end
    here = fileparts(mfilename('fullpath'));
    addpath(fullfile(fileparts(fileparts(here)), 'fsms', 'matlab'));
    env = fsms_paths();
    util = microstate_utilities();
    [tmaps, tlabels, tchannels, tchanlocs] = load_metamaps_templates(fullfile(env.vg_root, 'MetaMaps_2023_06.set'), 'K', 7);
    ch19 = {'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2'};
    alias = containers.Map({'T3', 'T4', 'T5', 'T6'}, {'T7/T3', 'T8/T4', 'P7/T5', 'P8/T6'});
    idx = cellfun(@(c) find(strcmp(tchannels, local_alias(alias, c)), 1), ch19);
    chanlocs = tchanlocs(idx);
    for i = 1:numel(ch19), chanlocs(i).labels = ch19{i}; end
    pos = util.positions_from_chanlocs(chanlocs, numel(ch19));
    Sim = struct('pos', pos, 'preprocessing', struct('spatial_filter', 'smoothing', ...
        'spatial_filter_neighbours', 12, 'spatial_filter_strength', 0.75));
    classes = {'A', 'B', 'C', 'E', 'F', 'G'};
    true_maps = util.normalize_maps(tmaps(cellfun(@(c) find(strcmp(tlabels, c)), classes), idx));
    rng(seed);
    conditions = {'iid_gaussian', 'iid_gaussian_pipeline_smoothed', 'template_plus_noise_0.3_smoothed', 'template_plus_noise_1.0_smoothed'};
    summary = struct('n_sets', n_sets, 'seed', seed, 'conditions', {conditions}, 'accept_fraction', zeros(1, 4), ...
        'median_min_assigned_abs_r', zeros(1, 4), 'median_min_gap', zeros(1, 4), 'p05_min_gap', zeros(1, 4));
    for c = 1:4
        accepted = false(n_sets, 1); min_r = zeros(n_sets, 1); min_gap = zeros(n_sets, 1);
        for s = 1:n_sets
            switch c
                case 1
                    maps = util.normalize_maps(randn(6, numel(ch19)));
                case 2
                    maps = smooth(util.normalize_maps(randn(6, numel(ch19))), Sim, util);
                case 3
                    maps = smooth(util.normalize_maps(true_maps + 0.3 * randn(6, numel(ch19)) / sqrt(numel(ch19))), Sim, util);
                case 4
                    maps = smooth(util.normalize_maps(true_maps + 1.0 * randn(6, numel(ch19)) / sqrt(numel(ch19))), Sim, util);
            end
            maps = maps(randperm(6), :);
            r = template_prespecified_v2(maps, ch19, chanlocs);
            accepted(s) = strcmp(r.labelling_quality, 'ok');
            min_r(s) = min(r.assigned_abs_r);
            min_gap(s) = min(r.assigned_abs_r - r.next_best_abs_r);
        end
        summary.accept_fraction(c) = mean(accepted);
        summary.median_min_assigned_abs_r(c) = median(min_r);
        summary.median_min_gap(c) = median(min_gap);
        summary.p05_min_gap(c) = prctile(min_gap, 5);
        fprintf('%-36s accept %.2f  median min|r| %.2f  median min gap %.2f  p05 gap %.2f\n', conditions{c}, ...
            summary.accept_fraction(c), summary.median_min_assigned_abs_r(c), summary.median_min_gap(c), summary.p05_min_gap(c));
    end
    if nargin >= 3 && ~isempty(out_json)
        fsms_write_json(out_json, summary);
    end
end

function key = local_alias(alias, name)
    key = name;
    if isKey(alias, name), key = alias(name); end
end

function maps = smooth(maps, Sim, util)
    filtered = util.apply_spatial_filter(maps', Sim);
    filtered = filtered - mean(filtered, 1);
    maps = util.normalize_maps(filtered');
end
