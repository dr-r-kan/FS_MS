function manifest = write_simulation_backfit_reports(T, output_dir)
% WRITE_SIMULATION_BACKFIT_REPORTS Aggregate coverage and confusion outputs.

    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    manifest = table();
    if ~istable(T) || height(T) == 0 || ...
            ~ismember('backfit_diagnostic_file', T.Properties.VariableNames)
        return;
    end

    valid_file = ~cellfun(@isempty, cellstr(string(T.backfit_diagnostic_file))) & ...
        cellfun(@isfile, cellstr(string(T.backfit_diagnostic_file)));
    if ~any(valid_file)
        return;
    end

    T_valid = T(valid_file, :);
    coverage_vars = intersect({'backfit_coverage_corr', 'backfit_coverage_spearman', ...
        'backfit_coverage_mae', 'backfit_coverage_rmse', 'backfit_coverage_l1', ...
        'cluster_identity_accuracy', 'cluster_identity_accuracy_matched', ...
        'cluster_n_label_matches', 'cluster_mean_matched_similarity', ...
        'backfit_overlap_fraction', ...
        'backfit_hard_cluster_top1_accuracy', 'backfit_hard_label_top1_accuracy', ...
        'backfit_hard_cluster_top1_accuracy_overlap', 'backfit_hard_label_top1_accuracy_overlap', ...
        'backfit_hard_label_weight_mae', 'backfit_hard_label_weight_mae_overlap', ...
        'backfit_hard_cluster_pair_accuracy_overlap', 'backfit_hard_label_pair_accuracy_overlap', ...
        'backfit_mix_cluster_top1_accuracy', 'backfit_mix_label_top1_accuracy', ...
        'backfit_mix_cluster_top1_accuracy_overlap', 'backfit_mix_label_top1_accuracy_overlap', ...
        'backfit_mix_label_weight_mae', 'backfit_mix_label_weight_mae_overlap', ...
        'backfit_mix_cluster_pair_accuracy_overlap', 'backfit_mix_label_pair_accuracy_overlap'}, ...
        T_valid.Properties.VariableNames, 'stable');
    if ~isempty(coverage_vars)
        coverage_summary = groupsummary(T_valid, {'method', 'criterion', 'K_true', 'K_estimated'}, 'mean', coverage_vars);
        writetable(coverage_summary, fullfile(output_dir, 'backfit_coverage_summary.csv'));
    end

    if ~ismember('K_gap', T_valid.Properties.VariableNames)
        T_valid.K_gap = double(T_valid.K_true) - double(T_valid.K_estimated);
    end

    keep_gap_mask = isfinite(T_valid.K_gap) & T_valid.K_gap >= 0 & T_valid.K_gap <= 3;
    T_gap = T_valid(keep_gap_mask, :);
    if height(T_gap) == 0
        return;
    end
    write_named_confusion_reports(T_gap, output_dir, 'cluster', 'cluster_confusion', 'Cluster identity confusion');
    T_equal = T_gap(double(T_gap.K_true) == double(T_gap.K_estimated), :);
    write_named_confusion_reports(T_equal, output_dir, 'hard', 'backfit_hard_confusion', 'Hard backfit confusion');
    write_named_confusion_reports(T_equal, output_dir, 'mixture', 'backfit_mixture_confusion', 'VB mixture backfit confusion');

    group_keys = strcat( ...
        string(T_gap.method), "|", ...
        string(T_gap.criterion), "|", ...
        string(T_gap.K_true), "|", ...
        string(T_gap.K_estimated));
    [unique_keys, ~, key_idx] = unique(group_keys, 'stable');

    rows = cell(numel(unique_keys), 1);
    for g = 1:numel(unique_keys)
        group_rows = T_gap(key_idx == g, :);
        [counts, labels] = aggregate_confusions(group_rows.backfit_diagnostic_file);
        if isempty(counts)
            continue;
        end
        row_norm = counts ./ max(sum(counts, 2), 1);

        method = char(string(group_rows.method(1)));
        criterion = char(string(group_rows.criterion(1)));
        K_true = double(group_rows.K_true(1));
        K_estimated = double(group_rows.K_estimated(1));
        file_stub = sprintf('%s__%s__Ktrue%d__Kest%d__delta%d', ...
            sanitize_stub(method), sanitize_stub(criterion), K_true, K_estimated, K_true - K_estimated);

        counts_csv = fullfile(output_dir, [file_stub '_counts.csv']);
        rownorm_csv = fullfile(output_dir, [file_stub '_row_normalized.csv']);
        counts_png = fullfile(output_dir, [file_stub '_counts.png']);
        rownorm_png = fullfile(output_dir, [file_stub '_row_normalized.png']);

        writetable(array2table(counts, 'VariableNames', matlab.lang.makeValidName(labels), 'RowNames', labels), counts_csv, 'WriteRowNames', true);
        writetable(array2table(row_norm, 'VariableNames', matlab.lang.makeValidName(labels), 'RowNames', labels), rownorm_csv, 'WriteRowNames', true);
        plot_confusion_heatmap(counts, labels, counts_png, sprintf('%s | %s | K_{true}=%d, K_{est}=%d', method, criterion, K_true, K_estimated), 'Counts');
        plot_confusion_heatmap(row_norm, labels, rownorm_png, sprintf('%s | %s | K_{true}=%d, K_{est}=%d', method, criterion, K_true, K_estimated), 'Row-normalized');

        rows{g} = table( ...
            string(method), string(criterion), K_true, K_estimated, K_true - K_estimated, ...
            height(group_rows), string(counts_csv), string(rownorm_csv), string(counts_png), string(rownorm_png), ...
            'VariableNames', {'method', 'criterion', 'K_true', 'K_estimated', 'K_gap', 'n_runs', ...
            'counts_csv', 'row_normalized_csv', 'counts_png', 'row_normalized_png'});
    end

    rows = rows(~cellfun(@isempty, rows));
    if isempty(rows)
        return;
    end
    manifest = vertcat(rows{:});
    writetable(manifest, fullfile(output_dir, 'backfit_confusion_manifest.csv'));

    gap_manifest = aggregate_gap_confusion_reports(T_gap, output_dir);
    if height(gap_manifest) > 0
        writetable(gap_manifest, fullfile(output_dir, 'backfit_confusion_gap_manifest.csv'));
    end
end

function manifest = write_named_confusion_reports(T_groupable, output_dir, mode, prefix, title_prefix)
    manifest = table();
    if height(T_groupable) == 0
        return;
    end

    group_keys = strcat( ...
        string(T_groupable.method), "|", ...
        string(T_groupable.criterion), "|", ...
        string(T_groupable.K_true), "|", ...
        string(T_groupable.K_estimated));
    [unique_keys, ~, key_idx] = unique(group_keys, 'stable');

    rows = cell(numel(unique_keys), 1);
    for g = 1:numel(unique_keys)
        group_rows = T_groupable(key_idx == g, :);
        [counts, labels] = aggregate_confusions(group_rows.backfit_diagnostic_file, mode);
        if isempty(counts)
            continue;
        end
        row_norm = counts ./ max(sum(counts, 2), 1);

        method = char(string(group_rows.method(1)));
        criterion = char(string(group_rows.criterion(1)));
        K_true = double(group_rows.K_true(1));
        K_estimated = double(group_rows.K_estimated(1));
        file_stub = sprintf('%s__%s__%s__Ktrue%d__Kest%d', ...
            sanitize_stub(prefix), sanitize_stub(method), sanitize_stub(criterion), K_true, K_estimated);

        counts_csv = fullfile(output_dir, [file_stub '_counts.csv']);
        rownorm_csv = fullfile(output_dir, [file_stub '_row_normalized.csv']);
        counts_png = fullfile(output_dir, [file_stub '_counts.png']);
        rownorm_png = fullfile(output_dir, [file_stub '_row_normalized.png']);

        writetable(array2table(counts, 'VariableNames', matlab.lang.makeValidName(labels), 'RowNames', labels), counts_csv, 'WriteRowNames', true);
        writetable(array2table(row_norm, 'VariableNames', matlab.lang.makeValidName(labels), 'RowNames', labels), rownorm_csv, 'WriteRowNames', true);
        plot_confusion_heatmap(counts, labels, counts_png, sprintf('%s | %s | %s | K_{true}=%d, K_{est}=%d', title_prefix, method, criterion, K_true, K_estimated), 'Counts');
        plot_confusion_heatmap(row_norm, labels, rownorm_png, sprintf('%s | %s | %s | K_{true}=%d, K_{est}=%d', title_prefix, method, criterion, K_true, K_estimated), 'Row-normalized');

        rows{g} = table( ...
            string(mode), string(method), string(criterion), K_true, K_estimated, K_true - K_estimated, ...
            height(group_rows), string(counts_csv), string(rownorm_csv), string(counts_png), string(rownorm_png), ...
            'VariableNames', {'analysis', 'method', 'criterion', 'K_true', 'K_estimated', 'K_gap', 'n_runs', ...
            'counts_csv', 'row_normalized_csv', 'counts_png', 'row_normalized_png'});
    end

    rows = rows(~cellfun(@isempty, rows));
    if isempty(rows)
        return;
    end
    manifest = vertcat(rows{:});
    writetable(manifest, fullfile(output_dir, [sanitize_stub(prefix) '_manifest.csv']));
end

function manifest = aggregate_gap_confusion_reports(T_gap, output_dir)
    rows = {};
    row_idx = 0;
    group_keys = strcat(string(T_gap.method), "|", string(T_gap.criterion), "|", string(T_gap.K_gap));
    [unique_keys, ~, key_idx] = unique(group_keys, 'stable');
    for g = 1:numel(unique_keys)
        group_rows = T_gap(key_idx == g, :);
        [counts, labels] = aggregate_confusions(group_rows.backfit_diagnostic_file);
        if isempty(counts)
            continue;
        end
        row_norm = counts ./ max(sum(counts, 2), 1);
        method = char(string(group_rows.method(1)));
        criterion = char(string(group_rows.criterion(1)));
        K_gap = double(group_rows.K_gap(1));
        file_stub = sprintf('%s__%s__kgap%d', ...
            sanitize_stub(method), sanitize_stub(criterion), K_gap);
        counts_csv = fullfile(output_dir, [file_stub '_counts.csv']);
        rownorm_csv = fullfile(output_dir, [file_stub '_row_normalized.csv']);
        counts_png = fullfile(output_dir, [file_stub '_counts.png']);
        rownorm_png = fullfile(output_dir, [file_stub '_row_normalized.png']);
        writetable(array2table(counts, 'VariableNames', matlab.lang.makeValidName(labels), 'RowNames', labels), counts_csv, 'WriteRowNames', true);
        writetable(array2table(row_norm, 'VariableNames', matlab.lang.makeValidName(labels), 'RowNames', labels), rownorm_csv, 'WriteRowNames', true);
        plot_confusion_heatmap(counts, labels, counts_png, sprintf('%s | %s | K_{true}-K_{est}=%d', method, criterion, K_gap), 'Counts');
        plot_confusion_heatmap(row_norm, labels, rownorm_png, sprintf('%s | %s | K_{true}-K_{est}=%d', method, criterion, K_gap), 'Row-normalized');
        row_idx = row_idx + 1;
        rows{row_idx, 1} = table( ...
            string(method), string(criterion), K_gap, height(group_rows), ...
            string(counts_csv), string(rownorm_csv), string(counts_png), string(rownorm_png), ...
            'VariableNames', {'method', 'criterion', 'K_gap', 'n_runs', ...
            'counts_csv', 'row_normalized_csv', 'counts_png', 'row_normalized_png'});
    end
    if isempty(rows)
        manifest = table();
    else
        manifest = vertcat(rows{:});
    end
end

function [counts_total, labels] = aggregate_confusions(files, mode)
    if nargin < 2
        mode = 'legacy';
    end
    counts_total = [];
    labels = {};
    file_list = cellstr(string(files));
    for i = 1:numel(file_list)
        file_i = char(file_list{i});
        if isempty(file_i) || ~isfile(file_i)
            continue;
        end
        S = load(file_i, 'BackfitDiagnostics');
        if ~isfield(S, 'BackfitDiagnostics') || ~isfield(S.BackfitDiagnostics, 'ok') || ~S.BackfitDiagnostics.ok
            continue;
        end
        diag_i = S.BackfitDiagnostics;
        [counts_i, labels_i] = extract_confusion_counts(diag_i, mode);
        if isempty(counts_i)
            continue;
        end
        if isempty(counts_total)
            labels = labels_i;
            counts_total = counts_i;
        else
            [labels, counts_total] = merge_confusion_counts(labels, counts_total, labels_i, counts_i);
        end
    end
end

function [counts, labels] = extract_confusion_counts(diag_i, mode)
    counts = [];
    labels = {};
    if isfield(diag_i, 'template_labels') && ~isempty(diag_i.template_labels)
        labels = cellstr(string(diag_i.template_labels(:)));
    end

    switch char(string(mode))
        case 'cluster'
            if isfield(diag_i, 'cluster') && isfield(diag_i.cluster, 'label_confusion_counts')
                counts = double(diag_i.cluster.label_confusion_counts);
            elseif isfield(diag_i, 'cluster_confusion_counts')
                counts = double(diag_i.cluster_confusion_counts);
            end
        case 'hard'
            if isfield(diag_i, 'hard') && isfield(diag_i.hard, 'label_confusion_counts')
                counts = double(diag_i.hard.label_confusion_counts);
            end
        case 'mixture'
            if isfield(diag_i, 'mixture') && isfield(diag_i.mixture, 'available') && ...
                    diag_i.mixture.available && isfield(diag_i.mixture, 'label_confusion_counts')
                counts = double(diag_i.mixture.label_confusion_counts);
            end
        otherwise
            if isfield(diag_i, 'confusion_counts')
                counts = double(diag_i.confusion_counts);
            end
    end

    if isempty(counts) || isempty(labels)
        counts = [];
        labels = {};
    elseif numel(labels) ~= size(counts, 1)
        labels = labels(1:min(numel(labels), size(counts, 1)));
        counts = counts(1:numel(labels), 1:numel(labels));
    end
end

function [labels_out, counts_out] = merge_confusion_counts(labels_a, counts_a, labels_b, counts_b)
    labels_out = unique([labels_a(:); labels_b(:)], 'stable');
    n = numel(labels_out);
    counts_out = zeros(n, n);
    idx_a = map_label_indices(labels_out, labels_a);
    idx_b = map_label_indices(labels_out, labels_b);
    counts_out(idx_a, idx_a) = counts_out(idx_a, idx_a) + counts_a;
    counts_out(idx_b, idx_b) = counts_out(idx_b, idx_b) + counts_b;
end

function idx = map_label_indices(master, subset)
    idx = zeros(numel(subset), 1);
    for i = 1:numel(subset)
        idx(i) = find(strcmp(master, subset{i}), 1, 'first');
    end
end

function plot_confusion_heatmap(values, labels, output_file, title_str, colorbar_label)
    fig = figure('Visible', 'off', 'Position', [100 100 900 760], 'Color', 'white');
    imagesc(values);
    axis square;
    colormap(parula(256));
    cb = colorbar;
    ylabel(cb, colorbar_label, 'FontSize', 10);
    set(gca, 'XTick', 1:numel(labels), 'XTickLabel', labels, ...
        'YTick', 1:numel(labels), 'YTickLabel', labels, ...
        'XTickLabelRotation', 45, 'Color', 'white', 'XColor', 'black', 'YColor', 'black');
    xlabel('Estimated template label', 'FontSize', 11, 'FontWeight', 'bold');
    ylabel('True template label', 'FontSize', 11, 'FontWeight', 'bold');
    title(title_str, 'Interpreter', 'none', 'FontSize', 12, 'FontWeight', 'bold');
    for r = 1:size(values, 1)
        for c = 1:size(values, 2)
            text(c, r, format_heatmap_value(values(r, c)), ...
                'HorizontalAlignment', 'center', 'Color', 'black', 'FontSize', 9, 'FontWeight', 'bold');
        end
    end
    saveas(fig, output_file);
    close(fig);
end

function s = format_heatmap_value(v)
    if abs(v - round(v)) < 1e-9
        s = sprintf('%d', round(v));
    else
        s = sprintf('%.2f', v);
    end
end

function s = sanitize_stub(s)
    s = lower(char(string(s)));
    s = regexprep(s, '[^a-z0-9]+', '_');
    s = regexprep(s, '^_+|_+$', '');
end
