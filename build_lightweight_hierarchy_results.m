function output_file = build_lightweight_hierarchy_results(results_dir, output_file)
% Build a plot-only, non-HDF5 hierarchy MAT from the small fit caches.

    if nargin < 2 || isempty(output_file)
        output_file = fullfile(results_dir, 'hierarchical_topoplot_results_manual_labels.mat');
    end
    global_file = fullfile(results_dir, 'global', 'global_solution.mat');
    if ~isfile(global_file)
        error('Missing global solution: %s', global_file);
    end
    G = load(global_file, 'Fit', 'rows', 'common_labels', 'common_chanlocs');
    HResults = struct();
    HResults.source = 'lightweight_cached_solutions';
    HResults.common_labels = G.common_labels;
    HResults.common_channel_labels = G.common_labels;
    HResults.common_chanlocs = G.common_chanlocs;
    HResults.selected_K = G.Fit.K_estimated;
    HResults.global = make_node('global', '', 'all', '', 'global', G.Fit, row_count(G), global_file);
    HResults.groups = load_nodes(fullfile(results_dir, 'groups'), 'group', 'group');
    HResults.conditions = load_nodes(fullfile(results_dir, 'conditions'), 'condition', 'condition');
    save(output_file, 'HResults', '-v7');
    fprintf('Saved lightweight hierarchy results: %s\n', output_file);
end

function nodes = load_nodes(root_dir, prefix, level)
    entries = dir(root_dir);
    entries = entries([entries.isdir] & ~ismember({entries.name}, {'.', '..'}));
    nodes = struct([]);
    for i = 1:numel(entries)
        fit_file = fullfile(entries(i).folder, entries(i).name, [prefix '_solution.mat']);
        if ~isfile(fit_file), continue; end
        S = load(fit_file, 'Fit', 'rows');
        if strcmp(level, 'group')
            node = make_node(level, '', entries(i).name, '', ['group_' entries(i).name], ...
                S.Fit, row_count(S), fit_file);
        else
            node = make_node(level, '', '', entries(i).name, ['condition_' entries(i).name], ...
                S.Fit, row_count(S), fit_file);
        end
        if isempty(nodes)
            nodes = node;
        else
            nodes(end + 1) = node; %#ok<AGROW>
        end
    end
end

function n = row_count(S)
    if isfield(S, 'rows') && istable(S.rows)
        n = height(S.rows);
    else
        n = NaN;
    end
end

function node = make_node(level, participant, group, condition, name, Fit, n_maps, fit_path)
    node = struct('level', string(level), 'participant', string(participant), ...
        'group', string(group), 'condition', string(condition), 'name', string(name), ...
        'centers', Fit.centers, 'K_estimated', Fit.K_estimated, 'n_maps', n_maps, ...
        'canonical_state_labels', {{}}, 'fit_path', string(fit_path));
    if isfield(Fit, 'template_alignment') && isfield(Fit.template_alignment, 'labels')
        node.canonical_state_labels = Fit.template_alignment.labels;
    end
end
