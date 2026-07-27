function Demo = demonstrate_topographic_recovery(varargin)
% DEMONSTRATE_TOPOGRAPHIC_RECOVERY Simulate, fit, align, and plot K=7 microstates.

    root_dir = fileparts(mfilename('fullpath'));
    setup_microstate_paths(root_dir);
    ensure_eeglab_path();

    p = inputParser;
    addParameter(p, 'duration_s', 600, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'sfreq', 250, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'snr_db', 6, @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'seed', 42, @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'output_dir', fullfile(root_dir, 'outputs', 'topographic_recovery_demo'), @(x) ischar(x) || isstring(x));
    parse(p, varargin{:});
    cfg = p.Results;
    cfg.output_dir = char(cfg.output_dir);
    if ~isfolder(cfg.output_dir), mkdir(cfg.output_dir); end

    if exist('topoplot', 'file') ~= 2
        error('EEGLAB topoplot is not on the MATLAB path.');
    end

    util = microstate_utilities();
    repo_cfg = util.load_config();
    [spm_ok, spm_info] = util.ensure_spm_mix('', repo_cfg.paths.spm_mixture_paths, true);
    if ~spm_ok
        error('SPM spm_mix not found. Checked: %s', strjoin(spm_info.attempted, ', '));
    end

    K = 7;
    sim_opts = struct( ...
        'prob', 0, ...
        'clean_sanity_profile', true, ...
        'ecological_profile', false, ...
        'inject_artifacts', false, ...
        'map_jitter_fraction', 0.05);

    [Sim, ~, ~] = generate_microstate_eeg(K, cfg.snr_db, cfg.duration_s, cfg.sfreq, cfg.seed, sim_opts);
    Sim.preprocessing = struct( ...
        'apply_average_reference', true, ...
        'filter_band', [2 20], ...
        'spatial_filter', 'none', ...
        'reject_gfp_peak_outliers', false);

    KMeans = fit_microstate_kmeans_koenig(Sim, K, 'gev');
    VBGMM = fit_microstate_spm_vb(Sim, K, 'free_energy');

    [true_maps, labels] = ordered_true_maps(Sim, util);
    KMeansPlot = align_to_truth(KMeans.centers, true_maps, util);
    VBGMMPlot = align_to_truth(VBGMM.centers, true_maps, util);

    [plot_chanlocs, keep_idx] = util.prepare_metamaps_chanlocs(Sim.chanlocs, size(true_maps, 2));
    plot_chanlocs = rotate_topoplot_chanlocs(plot_chanlocs, -90);
    true_plot = true_maps(:, keep_idx);
    kmeans_plot = KMeansPlot.maps(:, keep_idx);
    vbgmm_plot = VBGMMPlot.maps(:, keep_idx);

    figure_png = fullfile(cfg.output_dir, 'topographic_recovery_k7.png');
    plot_recovery_topoplots(true_plot, kmeans_plot, vbgmm_plot, labels, plot_chanlocs, figure_png);

    Demo = struct( ...
        'Sim', Sim, ...
        'KMeans', KMeans, ...
        'VBGMM', VBGMM, ...
        'labels', {labels}, ...
        'kmeans_alignment', KMeansPlot, ...
        'vbgmm_alignment', VBGMMPlot, ...
        'figure_png', figure_png);

    fprintf('Topographic recovery figure: %s\n', figure_png);
    if nargout == 0
        clear Demo;
    end
end

function ensure_eeglab_path()
    if exist('topoplot', 'file') == 2 && exist('pop_loadset', 'file') == 2
        return;
    end
    home_dir = getenv('HOME');
    candidates = { ...
        fullfile(home_dir, 'Documents', 'MATLAB', 'eeglab'), ...
        fullfile(home_dir, 'MATLAB', 'eeglab'), ...
        fullfile(home_dir, 'Downloads', 'eeglab'), ...
        fullfile(home_dir, 'eeglab')};
    for i = 1:numel(candidates)
        if isfile(fullfile(candidates{i}, 'eeglab.m'))
            addpath(genpath(candidates{i}));
            return;
        end
    end
end

function [maps, labels] = ordered_true_maps(Sim, util)
    maps = util.normalize_maps(double(Sim.maps_true));
    if isfield(Sim, 'true_template_labels') && ~isempty(Sim.true_template_labels)
        labels = cellstr(string(Sim.true_template_labels(:)));
    else
        labels = arrayfun(@(k) char('A' + k - 1), 1:size(maps, 1), 'UniformOutput', false)';
    end
    [labels, order] = sort(labels);
    maps = maps(order, :);
end

function Aligned = align_to_truth(centers, true_maps, util)
    true_maps = util.normalize_maps(double(true_maps));
    centers = util.normalize_maps(double(centers));
    metrics = microstate_partial_alignment(true_maps, centers, ...
        'distance_type', 'cosine', 'threshold', 0, 'polarity', true);

    aligned = nan(size(true_maps));
    estimated_index = nan(size(true_maps, 1), 1);
    correlation = nan(size(true_maps, 1), 1);
    for i = 1:size(metrics.match_assignment, 1)
        est_idx = metrics.match_assignment(i, 1);
        true_idx = metrics.match_assignment(i, 2);
        signed_corr = centers(est_idx, :) * true_maps(true_idx, :)';
        aligned(true_idx, :) = sign_or_one(signed_corr) * centers(est_idx, :);
        estimated_index(true_idx) = est_idx;
        correlation(true_idx) = abs(signed_corr);
    end

    Aligned = struct( ...
        'maps', aligned, ...
        'estimated_index_by_true_state', estimated_index, ...
        'abs_correlation_by_true_state', correlation, ...
        'recovery_metrics', metrics);
end

function s = sign_or_one(x)
    if x < 0
        s = -1;
    else
        s = 1;
    end
end

function plot_recovery_topoplots(true_maps, kmeans_maps, vbgmm_maps, labels, chanlocs, out_png)
    all_maps = [true_maps; kmeans_maps; vbgmm_maps];
    cmax = max(abs(all_maps(:)), [], 'omitnan');
    maplimits = [-cmax cmax];
    row_names = {'True', 'K-means', 'VB-GMM'};
    rows = {true_maps, kmeans_maps, vbgmm_maps};
    K = numel(labels);

    fig_w = max(980, 145 * K);
    fig_h = 430;
    fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 fig_w fig_h]);
    left = 0.10; right = 0.02; top = 0.15; bottom = 0.06;
    slot_w = (1 - left - right) / K;
    slot_h = (1 - top - bottom) / numel(rows);
    tile_w = 0.92 * min(slot_w, slot_h * fig_h / fig_w);
    tile_h = tile_w * fig_w / fig_h;
    for r = 1:numel(rows)
        maps = rows{r};
        y = bottom + (numel(rows) - r) * slot_h + (slot_h - tile_h) / 2;
        annotation(fig, 'textbox', [0.005 y + tile_h / 2 - 0.035 0.085 0.07], ...
            'String', row_names{r}, 'EdgeColor', 'none', ...
            'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle', ...
            'FontWeight', 'bold', 'Color', [0 0 0], 'FontSize', 11);
        for k = 1:K
            x = left + (k - 1) * slot_w + (slot_w - tile_w) / 2;
            axes('Position', [x y tile_w tile_h]); %#ok<LAXES>
            if all(isfinite(maps(k, :)))
                topoplot(maps(k, :), chanlocs, 'electrodes', 'off', 'numcontour', 6, 'maplimits', maplimits);
            else
                axis off;
            end
            if r == 1
                title(labels{k}, 'FontWeight', 'bold', 'Color', [0 0 0], 'FontSize', 12);
            end
        end
    end
    if exist('sgtitle', 'file') == 2
        sgtitle('K=7 topographic recovery: A-G order, polarity aligned', ...
            'Color', [0 0 0], 'FontSize', 14, 'FontWeight', 'bold');
    end
    exportgraphics(fig, out_png, 'Resolution', 300);
    close(fig);
end

function chanlocs = rotate_topoplot_chanlocs(chanlocs, angle_deg)
    rot = [cosd(angle_deg) -sind(angle_deg); sind(angle_deg) cosd(angle_deg)];
    for i = 1:numel(chanlocs)
        if ~(isfield(chanlocs(i), 'X') && isfield(chanlocs(i), 'Y') && isfield(chanlocs(i), 'Z'))
            continue;
        end
        xyz = double([chanlocs(i).X chanlocs(i).Y chanlocs(i).Z]);
        if any(~isfinite(xyz))
            continue;
        end
        xy = rot * xyz(1:2)';
        chanlocs(i).X = xy(1);
        chanlocs(i).Y = xy(2);
        [az, elev, sph_r] = cart2sph(chanlocs(i).X, chanlocs(i).Y, chanlocs(i).Z);
        chanlocs(i).theta = atan2d(chanlocs(i).X, chanlocs(i).Y);
        chanlocs(i).radius = 0.5 - rad2deg(elev) / 180;
        chanlocs(i).sph_theta = rad2deg(az);
        chanlocs(i).sph_phi = rad2deg(elev);
        chanlocs(i).sph_radius = sph_r;
    end
end
