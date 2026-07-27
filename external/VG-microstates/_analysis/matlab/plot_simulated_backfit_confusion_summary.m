function out = plot_simulated_backfit_confusion_summary(varargin)
%PLOT_SIMULATED_BACKFIT_CONFUSION_SUMMARY Summarise simulated method and backfit comparisons.
%
% Produces a 2x2 figure:
%   Top-left: K-selection accuracy comparing K-Means and VB-GMM
%             using silhouette for both methods.
%   Top-right: Backfit accuracy comparing K-Means using
%              silhouette against VB-GMM using its best K-selection
%              criterion.
%   Bottom-left: Row-normalized confusion grid for K-Means.
%   Bottom-right: Row-normalized confusion grid for VB-GMM.

    setup_microstate_paths();
    util = microstate_utilities();
    repo_cfg = util.load_config();

    p = inputParser;
    addParameter(p, 'output_dir', char(repo_cfg.simulation.out_dir), @(x) ischar(x) || isstring(x));
    addParameter(p, 'results_csv', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'output_file', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'template_file', char(repo_cfg.paths.template_file), @(x) ischar(x) || isstring(x));
    addParameter(p, 'methods', {'kmeans_koenig', 'spm_vb'}, @(x) iscell(x) || isstring(x));
    addParameter(p, 'criteria', {}, @(x) iscell(x) || isstring(x));
    addParameter(p, 'kmeans_method', 'kmeans_koenig', @(x) ischar(x) || isstring(x));
    addParameter(p, 'spm_method', 'spm_vb', @(x) ischar(x) || isstring(x));
    addParameter(p, 'kmeans_criterion', 'silhouette', @(x) ischar(x) || isstring(x));
    addParameter(p, 'spm_comparison_criterion', 'silhouette', @(x) ischar(x) || isstring(x));
    addParameter(p, 'spm_best_criterion', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'selection_metric', 'K_correct', @(x) ischar(x) || isstring(x));
    addParameter(p, 'k_gaps', 0:3, @(x) isnumeric(x) && isvector(x));
    addParameter(p, 'visible', false, @islogical);
    addParameter(p, 'resolution', double(util.get_field(repo_cfg.plotting, 'resolution', 300)), ...
        @(x) isnumeric(x) && isscalar(x) && x > 0);
    parse(p, varargin{:});

    cfg = p.Results;
    cfg.output_dir = util.resolve_path(char(cfg.output_dir), util.project_root());
    cfg.template_file = util.resolve_path(char(cfg.template_file), util.project_root());
    cfg.methods = cellstr(string(cfg.methods));
    cfg.criteria = cellstr(string(cfg.criteria));
    cfg.k_gaps = unique(double(cfg.k_gaps(:))', 'stable');

    if isempty(cfg.results_csv)
        cfg.results_csv = fullfile(cfg.output_dir, 'results', 'comparison_results.csv');
    else
        cfg.results_csv = util.resolve_path(char(cfg.results_csv), util.project_root());
    end
    if isempty(cfg.output_file)
        plot_dir = fullfile(cfg.output_dir, 'analysis_plots');
        util.ensure_dir(plot_dir);
        cfg.output_file = fullfile(plot_dir, 'backfit_confusion_summary.png');
    else
        cfg.output_file = util.resolve_path(char(cfg.output_file), util.project_root());
        out_dir = fileparts(cfg.output_file);
        if ~isempty(out_dir)
            util.ensure_dir(out_dir);
        end
    end

    if ~isfile(cfg.results_csv)
        error('Results manifest not found: %s', cfg.results_csv);
    end

    T = readtable(cfg.results_csv, 'TextType', 'string');
    required_vars = {'method', 'criterion', 'K_correct', 'backfit_diagnostic_file'};
    missing_vars = required_vars(~ismember(required_vars, T.Properties.VariableNames));
    if ~isempty(missing_vars)
        error('Results manifest is missing required columns: %s', strjoin(missing_vars, ', '));
    end

    method_vals = cellfun(@canonicalize_method_token, cellstr(string(T.method)), 'UniformOutput', false);
    criterion_vals = cellfun(@canonicalize_criterion_token, cellstr(string(T.criterion)), 'UniformOutput', false);
    selection_metric = char(string(cfg.selection_metric));
    if ~ismember(selection_metric, T.Properties.VariableNames)
        selection_metric = 'K_correct';
    end

    kmeans_method = canonicalize_method_token(cfg.kmeans_method);
    spm_method = canonicalize_method_token(cfg.spm_method);
    kmeans_criterion = canonicalize_criterion_token(cfg.kmeans_criterion);
    spm_comparison_criterion = canonicalize_criterion_token(cfg.spm_comparison_criterion);

    if isempty(cfg.criteria)
        criterion_candidate_mask = true(height(T), 1);
    else
        criteria_requested = cellfun(@canonicalize_criterion_token, cfg.criteria, 'UniformOutput', false);
        criterion_candidate_mask = ismember(criterion_vals, criteria_requested);
    end

    if isempty(strtrim(char(string(cfg.spm_best_criterion))))
        spm_best_criterion = select_best_spm_criterion( ...
            T, method_vals, criterion_vals, spm_method, criterion_candidate_mask, selection_metric);
    else
        spm_best_criterion = canonicalize_criterion_token(cfg.spm_best_criterion);
    end

    file_vals = cellstr(string(T.backfit_diagnostic_file));
    valid_file = ~cellfun(@isempty, file_vals) & cellfun(@isfile, file_vals);
    gap_vals = compute_k_gap(T);
    gap_mask = ismember(gap_vals, cfg.k_gaps);

    mask_kacc_kmeans = strcmp(method_vals, kmeans_method) & strcmp(criterion_vals, kmeans_criterion);
    mask_kacc_spm = strcmp(method_vals, spm_method) & strcmp(criterion_vals, spm_comparison_criterion);
    mask_backfit_kmeans = mask_kacc_kmeans & valid_file & gap_mask;
    mask_backfit_spm = strcmp(method_vals, spm_method) & strcmp(criterion_vals, spm_best_criterion) & valid_file & gap_mask;

    if ~any(mask_kacc_kmeans)
        error('No rows found for K-Means + %s.', kmeans_criterion);
    end
    if ~any(mask_kacc_spm)
        error('No rows found for VB-GMM + %s.', spm_comparison_criterion);
    end
    if ~any(mask_backfit_kmeans)
        error('No valid backfit diagnostics found for K-Means + %s.', kmeans_criterion);
    end
    if ~any(mask_backfit_spm)
        error('No valid backfit diagnostics found for VB-GMM + %s.', spm_best_criterion);
    end

    kacc_summary = [ ...
        mean(to_numeric_vector(T.K_correct(mask_kacc_kmeans)), 'omitnan'), ...
        mean(to_numeric_vector(T.K_correct(mask_kacc_spm)), 'omitnan')];
    kacc_counts = [nnz(mask_kacc_kmeans), nnz(mask_kacc_spm)];

    diag_kmeans = collect_backfit_diagnostics(file_vals(mask_backfit_kmeans));
    diag_spm = collect_backfit_diagnostics(file_vals(mask_backfit_spm));

    if isempty(diag_kmeans) || isempty(diag_spm)
        error('Backfit diagnostics were present in the table but could not be loaded.');
    end

    label_order = merge_label_order({}, extract_all_labels(diag_kmeans));
    label_order = merge_label_order(label_order, extract_all_labels(diag_spm));

    confusion_kmeans = mean_confusion_matrix(diag_kmeans, label_order);
    confusion_spm = mean_confusion_matrix(diag_spm, label_order);

    backfit_summary = [ ...
        mean([diag_kmeans.accuracy], 'omitnan'), ...
        mean([diag_spm.accuracy], 'omitnan')];
    backfit_counts = [numel(diag_kmeans), numel(diag_spm)];

    fig = figure('Name', 'Simulated backfit confusion summary', ...
        'Color', 'white', 'Visible', char(util.on_off_string(cfg.visible)), 'NumberTitle', 'off', ...
        'Position', [70, 70, 1500, 1100]);
    if ~cfg.visible
        cleaner = onCleanup(@() close_if_valid(fig)); %#ok<NASGU>
    end

    tl = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    sgtitle(tl, 'Simulated method comparison and backfit confusion summary', ...
        'FontWeight', 'bold', 'FontSize', 15);

    ax_kacc = nexttile(tl, 1);
    render_method_bar_panel(ax_kacc, kacc_summary, kacc_counts, ...
        {'K-Means', 'VB-GMM'}, ...
        sprintf('K selection accuracy\n(silhouette for both methods)'));

    ax_backfit = nexttile(tl, 2);
    render_method_bar_panel(ax_backfit, backfit_summary, backfit_counts, ...
        {'K-Means', 'VB-GMM'}, ...
        sprintf(['Backfit accuracy\n(K-Means: %s; VB-GMM: %s)'], ...
        prettify_criterion(kmeans_criterion), prettify_criterion(spm_best_criterion)));

    ax_conf_km = nexttile(tl, 3);
    render_confusion_panel(ax_conf_km, confusion_kmeans, label_order, ...
        sprintf('Confusion grid: K-Means\n(%s; n=%d runs)', ...
        prettify_criterion(kmeans_criterion), numel(diag_kmeans)));

    ax_conf_spm = nexttile(tl, 4);
    render_confusion_panel(ax_conf_spm, confusion_spm, label_order, ...
        sprintf('Confusion grid: VB-GMM\n(%s; n=%d runs)', ...
        prettify_criterion(spm_best_criterion), numel(diag_spm)));

    exportgraphics(fig, cfg.output_file, 'Resolution', cfg.resolution);

    out = struct();
    out.plot_file = cfg.output_file;
    out.results_csv = cfg.results_csv;
    out.kmeans_criterion = kmeans_criterion;
    out.spm_comparison_criterion = spm_comparison_criterion;
    out.spm_best_criterion = spm_best_criterion;
    out.label_order = label_order(:);
    out.kacc_summary = kacc_summary;
    out.kacc_counts = kacc_counts;
    out.backfit_summary = backfit_summary;
    out.backfit_counts = backfit_counts;
    out.confusion_kmeans = confusion_kmeans;
    out.confusion_spm = confusion_spm;
end

function spm_best_criterion = select_best_spm_criterion(T, method_vals, criterion_vals, spm_method, criterion_candidate_mask, metric_name)
    mask_spm = strcmp(method_vals, spm_method) & criterion_candidate_mask;
    if ~any(mask_spm)
        error('No VB-GMM rows were available to choose the best criterion.');
    end

    criteria = unique(criterion_vals(mask_spm), 'stable');
    metric_col = to_numeric_vector(T.(metric_name));
    best_score = -inf;
    best_n = -inf;
    spm_best_criterion = '';
    for i = 1:numel(criteria)
        crit_i = criteria{i};
        mask_i = mask_spm & strcmp(criterion_vals, crit_i);
        vals_i = metric_col(mask_i);
        vals_i = vals_i(isfinite(vals_i));
        if isempty(vals_i)
            continue;
        end
        score_i = mean(vals_i, 'omitnan');
        n_i = numel(vals_i);
        if score_i > best_score || (abs(score_i - best_score) <= eps && n_i > best_n)
            best_score = score_i;
            best_n = n_i;
            spm_best_criterion = crit_i;
        end
    end

    if isempty(spm_best_criterion)
        error('Could not identify a VB-GMM criterion with finite %s values.', metric_name);
    end
end

function gap_vals = compute_k_gap(T)
    if ismember('K_gap', T.Properties.VariableNames)
        gap_vals = to_numeric_vector(T.K_gap);
        return;
    end
    if ismember('K_true', T.Properties.VariableNames) && ismember('K_estimated', T.Properties.VariableNames)
        gap_vals = to_numeric_vector(T.K_true) - to_numeric_vector(T.K_estimated);
        return;
    end
    gap_vals = zeros(height(T), 1);
end

function diagnostics = collect_backfit_diagnostics(files)
    diagnostics = struct('labels', {}, 'counts', {}, 'accuracy', {});
    file_list = cellstr(string(files));
    for i = 1:numel(file_list)
        file_i = char(string(file_list{i}));
        if isempty(file_i) || ~isfile(file_i)
            continue;
        end
        S = load(file_i, 'BackfitDiagnostics');
        if ~isfield(S, 'BackfitDiagnostics') || ~isfield(S.BackfitDiagnostics, 'ok') || ~S.BackfitDiagnostics.ok
            continue;
        end
        diag_i = S.BackfitDiagnostics;
        labels_i = cellstr(string(diag_i.template_labels(:)));
        counts_i = double(diag_i.confusion_counts);
        if isempty(labels_i) || isempty(counts_i)
            continue;
        end
        diagnostics(end + 1) = struct( ... %#ok<AGROW>
            'labels', {labels_i}, ...
            'counts', counts_i, ...
            'accuracy', local_accuracy(diag_i));
    end
end

function labels = extract_all_labels(diagnostics)
    labels = {};
    for i = 1:numel(diagnostics)
        labels = merge_label_order(labels, diagnostics(i).labels);
    end
end

function values = mean_confusion_matrix(diagnostics, label_order)
    values = nan(numel(label_order), numel(label_order));
    if isempty(diagnostics)
        return;
    end
    mats = cell(1, numel(diagnostics));
    keep = false(1, numel(diagnostics));
    for i = 1:numel(diagnostics)
        mats{i} = normalize_confusion_rows(reorder_confusion_matrix( ...
            diagnostics(i).counts, diagnostics(i).labels, label_order));
        keep(i) = ~isempty(mats{i}) && any(isfinite(mats{i}), 'all');
    end
    mats = mats(keep);
    if isempty(mats)
        return;
    end
    values = mean(cat(3, mats{:}), 3, 'omitnan');
end

function accuracy = local_accuracy(diag_i)
    accuracy = NaN;
    if isfield(diag_i, 'hard') && isstruct(diag_i.hard) && isfield(diag_i.hard, 'label_top1_accuracy')
        accuracy = double(diag_i.hard.label_top1_accuracy);
    end
end

function labels_out = merge_label_order(labels_a, labels_b)
    labels_out = labels_a(:);
    for i = 1:numel(labels_b)
        if ~any(strcmp(labels_out, labels_b{i}))
            labels_out{end + 1, 1} = labels_b{i}; %#ok<AGROW>
        end
    end
end

function matrix_out = reorder_confusion_matrix(matrix_in, labels_in, label_order)
    n = numel(label_order);
    matrix_out = nan(n, n);
    if isempty(matrix_in) || isempty(labels_in)
        return;
    end

    labels_in = cellstr(string(labels_in(:)));
    for r = 1:n
        src_r = find(strcmp(labels_in, label_order{r}), 1, 'first');
        if isempty(src_r)
            continue;
        end
        for c = 1:n
            src_c = find(strcmp(labels_in, label_order{c}), 1, 'first');
            if isempty(src_c)
                continue;
            end
            matrix_out(r, c) = matrix_in(src_r, src_c);
        end
    end
end

function rownorm = normalize_confusion_rows(values)
    rownorm = values;
    if isempty(values)
        return;
    end
    for r = 1:size(values, 1)
        row = values(r, :);
        finite_mask = isfinite(row);
        if ~any(finite_mask)
            continue;
        end
        row_sum = sum(row(finite_mask));
        if row_sum > 0
            rownorm(r, finite_mask) = row(finite_mask) ./ row_sum;
        else
            rownorm(r, finite_mask) = NaN;
        end
    end
end

function render_method_bar_panel(ax, values, counts, labels, title_str)
    axes(ax);
    cla(ax);

    x = 1:numel(values);
    b = bar(ax, x, values, 0.6, 'FaceColor', 'flat', 'EdgeColor', 'black', 'LineWidth', 1.2);
    b.CData = [0.20 0.46 0.70; 0.87 0.48 0.16];
    ylim(ax, [0 1]);
    xlim(ax, [0.4, numel(values) + 0.6]);
    grid(ax, 'on');
    ax.GridAlpha = 0.25;
    ax.LineWidth = 1;
    ax.Color = 'white';
    ax.XTick = x;
    ax.XTickLabel = labels;
    ax.XTickLabelRotation = 15;
    ylabel(ax, 'Mean accuracy', 'FontWeight', 'bold');
    title(ax, title_str, 'FontWeight', 'bold', 'FontSize', 12, 'Interpreter', 'none');

    for i = 1:numel(values)
        if ~isfinite(values(i))
            continue;
        end
        text(ax, x(i), values(i) + 0.03, sprintf('%.2f\n(n=%d)', values(i), counts(i)), ...
            'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
    end
end

function render_confusion_panel(ax, values, label_order, title_str)
    axes(ax);
    cla(ax);
    if isempty(values) || all(~isfinite(values), 'all')
        text(0.5, 0.5, 'No data', 'Units', 'normalized', ...
            'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 14);
        axis(ax, 'off');
        return;
    end

    imagesc(ax, values, [0 1]);
    axis(ax, 'square');
    colormap(ax, parula(256));
    cb = colorbar(ax);
    cb.Label.String = 'Confusion rate';
    cb.Label.FontWeight = 'bold';
    set(ax, 'XTick', 1:numel(label_order), 'XTickLabel', label_order, ...
        'YTick', 1:numel(label_order), 'YTickLabel', label_order, ...
        'XTickLabelRotation', 45, 'Color', 'white', 'XColor', 'black', 'YColor', 'black', ...
        'FontSize', 10);
    xlabel(ax, 'Estimated template label', 'FontWeight', 'bold');
    ylabel(ax, 'True template label', 'FontWeight', 'bold');
    title(ax, title_str, 'FontWeight', 'bold', 'FontSize', 12, 'Interpreter', 'none');

    for r = 1:size(values, 1)
        for c = 1:size(values, 2)
            v = values(r, c);
            if ~isfinite(v)
                continue;
            end
            text_color = [0 0 0];
            if v >= 0.55
                text_color = [1 1 1];
            end
            text(ax, c, r, sprintf('%.2f', v), 'HorizontalAlignment', 'center', ...
                'Color', text_color, 'FontWeight', 'bold', 'FontSize', 9);
        end
    end
end

function vals = to_numeric_vector(col)
    if isnumeric(col) || islogical(col)
        vals = double(col);
    elseif iscell(col)
        vals = nan(numel(col), 1);
        for i = 1:numel(col)
            v = col{i};
            if isnumeric(v) || islogical(v)
                vals(i) = double(v);
            elseif isstring(v) || ischar(v)
                vals(i) = str2double(char(string(v)));
            end
        end
    else
        vals = str2double(string(col));
    end
    vals = vals(:);
end

function token = canonicalize_method_token(token_in)
    token = lower(strtrim(char(string(token_in))));
    token = strrep(token, '_', ' ');
    token = strrep(token, '-', ' ');
    token = regexprep(token, '\s+', ' ');
    if contains(token, 'spm vb')
        token = 'spm_vb';
    elseif contains(token, 'traditional kmeans') || contains(token, 'traditional k means') || ...
            contains(token, 'kmeans koenig') || contains(token, 'koenig kmeans')
        token = 'kmeans_koenig';
    end
end

function token = canonicalize_criterion_token(token_in)
    token = lower(strtrim(char(string(token_in))));
    token = strrep(token, '_', ' ');
    token = regexprep(token, '\s+', ' ');
    if strcmp(token, 'elbow')
        token = 'free energy elbow';
    elseif strcmp(token, 'silhouette only')
        token = 'silhouette';
    elseif strcmp(token, 'free energy elbow only')
        token = 'free energy elbow';
    elseif contains(token, 'elbow sil') || contains(token, 'free energy elbow sil')
        token = 'elbow sil combined';
    end
end

function txt = prettify_criterion(token)
    txt = strrep(char(string(token)), '_', ' ');
end

function close_if_valid(fig)
    if ~isempty(fig) && isgraphics(fig)
        close(fig);
    end
end
