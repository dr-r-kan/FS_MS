function outputs = analyse_hierarchical_outputs(results_mat, varargin)
% ANALYSE_HIERARCHICAL_OUTPUTS Omnibus topographic/statistical analysis for hierarchy fits.

    setup_microstate_paths();
    if nargin < 1 || isempty(results_mat)
        results_mat = fullfile('outputs', 'hierarchical_microstates', 'hierarchical_microstate_results.mat');
    end

    p = inputParser;
    addRequired(p, 'results_mat', @(x) ischar(x) || isstring(x));
    addParameter(p, 'output_dir', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'semiology_manifest_csv', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'run_topoplots', true, @islogical);
    addParameter(p, 'run_tanova', true, @islogical);
    addParameter(p, 'tanova_permutations', 5000, @isnumeric);
    parse(p, results_mat, varargin{:});
    cfg = p.Results;

    cfg.results_mat = char(cfg.results_mat);
    if isempty(char(cfg.output_dir))
        cfg.output_dir = fullfile(fileparts(cfg.results_mat), 'omnibus_analysis');
    else
        cfg.output_dir = char(cfg.output_dir);
    end
    if ~exist(cfg.output_dir, 'dir')
        mkdir(cfg.output_dir);
    end

    outputs = struct();
    outputs.output_dir = cfg.output_dir;

    if cfg.run_topoplots
        outputs.global_topoplots = plot_microstate_global_states( ...
            cfg.results_mat, 'output_dir', fullfile(cfg.output_dir, 'global_topoplots'));
        outputs.group_topoplots = plot_microstate_split_by_group( ...
            cfg.results_mat, 'output_dir', fullfile(cfg.output_dir, 'group_topoplots'));
        outputs.condition_topoplots = plot_microstate_split_by_condition( ...
            cfg.results_mat, 'output_dir', fullfile(cfg.output_dir, 'condition_topoplots'));
        outputs.combined_topoplots = combine_topoplot_figures( ...
            outputs.global_topoplots.figure, outputs.group_topoplots.figure, ...
            outputs.condition_topoplots.figure, cfg.output_dir);
    end

    if cfg.run_tanova
        [outputs.tanova, outputs.tanova_csv] = run_microstate_hierarchical_tanova( ...
            cfg.results_mat, ...
            'output_dir', fullfile(cfg.output_dir, 'tanova'), ...
            'semiology_manifest_csv', char(cfg.semiology_manifest_csv), ...
            'n_permutations', double(cfg.tanova_permutations));
        [outputs.bhattacharyya_tanova, outputs.bhattacharyya_tanova_csv] = ...
            run_microstate_bhattacharyya_tanova(cfg.results_mat, ...
                'output_dir', fullfile(cfg.output_dir, 'bhattacharyya_tanova'), ...
                'semiology_manifest_csv', char(cfg.semiology_manifest_csv), ...
                'n_permutations', double(cfg.tanova_permutations));
    end

    fprintf('Hierarchical omnibus outputs: %s\n', cfg.output_dir);
end

function out_file = combine_topoplot_figures(global_file, group_file, condition_file, output_dir)
% Keep the detailed topoplot layouts, while also providing one review image.
    files = {global_file, group_file, condition_file};
    section_labels = {'Global microstates', 'Microstates by group', 'Microstates by condition'};
    images = cell(size(files));
    max_width = 0;
    total_height = 0;
    for i = 1:numel(files)
        images{i} = imread(files{i});
        max_width = max(max_width, size(images{i}, 2));
        total_height = total_height + size(images{i}, 1);
    end

    scale = min([1, 1800 / max(max_width, 1), 2600 / max(total_height, 1)]);
    fig = figure('Visible', 'off', 'Color', 'white', 'Name', 'Hierarchical microstate topoplots', ...
        'Position', [50 50 max(900, round(max_width * scale)) max(900, round(total_height * scale))]);
    layout = tiledlayout(fig, 3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
    for i = 1:numel(images)
        ax = nexttile(layout);
        image(ax, images{i});
        axis(ax, 'image');
        axis(ax, 'off');
        set(ax, 'Color', 'white');
        title(ax, section_labels{i}, 'Interpreter', 'none', 'FontWeight', 'bold', 'Color', 'black');
    end
    title(layout, 'Global, group, and condition microstate clusters', ...
        'Interpreter', 'none', 'FontWeight', 'bold', 'Color', 'black');

    out_file = fullfile(output_dir, 'hierarchical_microstate_topoplots.png');
    exportgraphics(fig, out_file, 'Resolution', 300, 'BackgroundColor', 'white');
    close(fig);
    fprintf('Saved combined hierarchical topoplot image: %s\n', out_file);
end
