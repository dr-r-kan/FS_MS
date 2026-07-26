function output_file = export_global_manual_preview(results_dir, output_file)
% Export an unlabelled State 1..K preview from the cached global solution.

    repo_dir = fileparts(mfilename('fullpath'));
    vg_dir = fullfile(repo_dir, 'external', 'VG-microstates');
    addpath(vg_dir);
    setup_microstate_paths(vg_dir);

    if nargin < 2 || isempty(output_file)
        output_file = fullfile(results_dir, 'global', 'global_centers_for_manual_labelling.png');
    end
    source = fullfile(results_dir, 'global', 'global_solution.mat');
    if ~isfile(source), error('Missing cached global solution: %s', source); end
    S = load(source, 'Fit', 'common_chanlocs');
    maps = double(S.Fit.centers);
    utils = microstate_utilities();
    [plot_chanlocs, keep_idx] = utils.prepare_chanlocs_for_topoplot(S.common_chanlocs, size(maps, 2));
    maps_plot = maps(:, keep_idx);
    K = size(maps, 1);
    fig = figure('Visible', 'off', 'Color', 'white', ...
        'Position', [100 100 max(1000, 230 * K) 330]);
    for k = 1:K
        subplot(1, K, k);
        if exist('topoplot', 'file') == 2 && numel(plot_chanlocs) >= 4
            topoplot(maps_plot(k, :), plot_chanlocs, 'electrodes', 'on', 'numcontour', 8);
        else
            imagesc(maps(k, :)); axis tight; colorbar;
        end
        title(sprintf('State %d', k), 'FontWeight', 'bold');
    end
    sgtitle('Global microstate templates for manual labelling (K=6)', 'FontWeight', 'bold');
    exportgraphics(fig, output_file, 'Resolution', 300, 'BackgroundColor', 'white');
    close(fig);
    fprintf('Saved manual-label preview: %s\n', output_file);
end
