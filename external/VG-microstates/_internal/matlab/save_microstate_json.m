function save_microstate_json(Results, Sim, output_file, META)
% SAVE_MICROSTATE_JSON Save microstate analysis to a plotting-friendly JSON file.

    if nargin < 4
        META = struct();
    end
    util = microstate_utilities();

    C = infer_channel_count(Results, Sim);
    ch_labels_raw = infer_channel_labels(Sim, META, C, util);
    ch_labels = util.sanitize_channel_labels(ch_labels_raw(1:C));

    json_data = struct();

    % ===== TRUE MICROSTATES =====
    json_data.true_microstates = struct();
    if isfield(Results, 'maps_true') && ~isempty(Results.maps_true) && ...
            isfield(Results, 'K_true') && isfinite(Results.K_true) && Results.K_true > 0
        for k = 1:Results.K_true
            state_key = sprintf('state_%d', k);
            state_data = struct();
            for c = 1:C
                state_data.(ch_labels{c}) = double(real(Results.maps_true(k, c)));
            end
            json_data.true_microstates.(state_key) = state_data;
        end
    end

    % ===== ESTIMATED MICROSTATES =====
    json_data.estimated_microstates = struct();
    for k = 1:Results.K_estimated
        state_key = sprintf('state_%d', k);
        state_data = struct();
        for c = 1:C
            state_data.(ch_labels{c}) = double(real(Results.centers(k, c)));
        end

        if isfield(Results, 'cluster_weights') && numel(Results.cluster_weights) >= k
            state_data.confidence = double(real(Results.cluster_weights(k)));
        end
        if isfield(Results, 'template_alignment') && ~isempty(Results.template_alignment)
            state_data = add_template_state_fields(state_data, Results.template_alignment, k);
        end

        json_data.estimated_microstates.(state_key) = state_data;
    end

    % ===== METADATA =====
    json_data.metadata = struct();
    if isfield(Results, 'method'), json_data.metadata.method = Results.method; end
    if isfield(Results, 'criterion'), json_data.metadata.criterion = Results.criterion; end
    if isfield(Results, 'K_true'), json_data.metadata.K_true = Results.K_true; end
    if isfield(Results, 'K_estimated'), json_data.metadata.K_estimated = Results.K_estimated; end
    if isfield(Results, 'K_model_selected'), json_data.metadata.K_model_selected = Results.K_model_selected; end
    if isfield(Results, 'K_candidates'), json_data.metadata.K_candidates = Results.K_candidates; end
    if isfield(Results, 'K_effective_vals'), json_data.metadata.K_effective_vals = Results.K_effective_vals; end
    selected_idx = find_selected_model_index(Results);
    if ~isnan(selected_idx)
        json_data.metadata.selected_model_index = double(selected_idx);
    end
    if isfield(Results, 'polarity_feature_info')
        json_data.metadata.polarity_feature_info = Results.polarity_feature_info;
    end
    if isfield(Results, 'polarity_duplicate_info') && isfield(Results, 'K_candidates')
        dup_counts = zeros(size(Results.K_candidates));
        for i = 1:numel(Results.K_candidates)
            if numel(Results.polarity_duplicate_info) >= i && ...
                    ~isempty(Results.polarity_duplicate_info{i}) && ...
                    isfield(Results.polarity_duplicate_info{i}, 'pairs')
                dup_counts(i) = size(Results.polarity_duplicate_info{i}.pairs, 1);
            end
        end
        json_data.metadata.polarity_duplicate_pair_counts = dup_counts;
    end
    if isfield(Results, 'SNR_dB'), json_data.metadata.SNR_dB = Results.SNR_dB; end
    if isfield(Results, 'duration_s'), json_data.metadata.duration_s = Results.duration_s; end
    if isfield(Sim, 'sfreq'), json_data.metadata.sfreq = Sim.sfreq; end
    json_data.metadata.n_channels = C;
    if isfield(Results, 'n_maps'), json_data.metadata.n_samples_analyzed = Results.n_maps; end
    if isfield(Results, 'runtime'), json_data.metadata.runtime_s = Results.runtime; end

    if isfield(Results, 'silhouette_vals') && ~isempty(Results.silhouette_vals)
        json_data.metadata.silhouette_vals = double(real(Results.silhouette_vals(:)'));
    end
    if isfield(Results, 'free_energy_vals') && ~isempty(Results.free_energy_vals)
        json_data.metadata.free_energy_vals = double(real(Results.free_energy_vals(:)'));
    elseif isfield(Results, 'free_energy') && ~isempty(Results.free_energy)
        json_data.metadata.free_energy_vals = double(real(Results.free_energy(:)'));
    end
    if isfield(Results, 'within_ss') && ~isempty(Results.within_ss)
        json_data.metadata.within_ss = double(real(Results.within_ss(:)'));
    end
    if isfield(Results, 'gev_vals') && ~isempty(Results.gev_vals)
        json_data.metadata.gev_vals = double(real(Results.gev_vals(:)'));
    end
    if ~isnan(selected_idx)
        if isfield(Results, 'free_energy_vals') && numel(Results.free_energy_vals) >= selected_idx
            json_data.metadata.selected_model_free_energy = double(real(Results.free_energy_vals(selected_idx)));
            json_data.metadata.model_evidence_free_energy = double(real(Results.free_energy_vals(selected_idx)));
        elseif isfield(Results, 'free_energy') && numel(Results.free_energy) >= selected_idx
            json_data.metadata.selected_model_free_energy = double(real(Results.free_energy(selected_idx)));
            json_data.metadata.model_evidence_free_energy = double(real(Results.free_energy(selected_idx)));
        end
        if isfield(Results, 'silhouette_vals') && numel(Results.silhouette_vals) >= selected_idx
            json_data.metadata.selected_model_silhouette = double(real(Results.silhouette_vals(selected_idx)));
        end
        if isfield(Results, 'within_ss') && numel(Results.within_ss) >= selected_idx
            json_data.metadata.selected_model_within_ss = double(real(Results.within_ss(selected_idx)));
        end
    end
    if isfield(Results, 'template_alignment') && ~isempty(Results.template_alignment)
        json_data.metadata.template_alignment = template_alignment_summary(Results.template_alignment);
    end
    if isfield(Results, 'backfit_diagnostics') && ~isempty(Results.backfit_diagnostics)
        json_data.metadata.backfit = backfit_summary(Results.backfit_diagnostics);
    end

    if isfield(Results, 'spm_mix_model_summaries') && ~isempty(Results.spm_mix_model_summaries)
        spm_mix_payload = struct();
        spm_mix_payload.per_k_models = Results.spm_mix_model_summaries;
        if isfield(Results, 'selected_spm_mix_model') && ~isempty(Results.selected_spm_mix_model)
            spm_mix_payload.selected_model = Results.selected_spm_mix_model;
        end
        json_data.spm_mix = spm_mix_payload;
    end

    if ~isempty(META) && isstruct(META)
        mfields = fieldnames(META);
        for mf = 1:length(mfields)
            try
                json_data.metadata.(mfields{mf}) = META.(mfields{mf});
            catch
            end
        end
    end

    % ===== RECOVERY METRICS =====
    if isfield(Results, 'recovery_metrics') && ~isempty(Results.recovery_metrics)
        metrics = Results.recovery_metrics;
    else
        metrics = empty_recovery_metrics();
    end
    json_data.recovery = struct();
    json_data.recovery.n_matched = double(metrics.n_matched);
    json_data.recovery.sensitivity = double(metrics.sensitivity);
    json_data.recovery.precision = double(metrics.precision);
    json_data.recovery.f1_score = double(metrics.f1_score);
    json_data.recovery.mean_recovery_matched = double(metrics.mean_recovery_matched);
    json_data.recovery.mean_recovery_padded = double(metrics.mean_recovery_padded);

    % ===== CHANNEL INFORMATION =====
    json_data.channel_info = struct();
    json_data.channel_info.labels = ch_labels_raw(1:C);
    json_data.channel_info.labels_sanitized = ch_labels;
    json_data.channel_info.n_channels = C;

    % ===== CHANNEL POSITIONS =====
    json_data.electrode_positions = struct();
    if isfield(Sim, 'pos') && ~isempty(Sim.pos) && size(Sim.pos, 1) >= C
        for c = 1:C
            json_data.electrode_positions.(ch_labels{c}) = struct( ...
                'x', double(real(Sim.pos(c, 1))), ...
                'y', double(real(Sim.pos(c, 2))), ...
                'z', double(real(Sim.pos(c, 3))));
        end
    end

    % ===== MATCHING INFORMATION =====
    if isfield(metrics, 'match_assignment') && ~isempty(metrics.match_assignment)
        json_data.matches = struct();
        for i = 1:size(metrics.match_assignment, 1)
            est_idx = metrics.match_assignment(i, 1);
            true_idx = metrics.match_assignment(i, 2);
            similarity = metrics.match_similarities(i);
            match_key = sprintf('match_%d', i);
            json_data.matches.(match_key) = struct( ...
                'estimated_state', double(est_idx), ...
                'true_state', double(true_idx), ...
                'similarity', double(real(similarity)));
        end
    else
        json_data.matches = struct();
    end

    json_data = util.clean_complex_values(json_data);

    try
        json_str = jsonencode(json_data);
    catch ME
        fprintf('ERROR: JSON encoding failed: %s\n', ME.message);
        fid = fopen(output_file, 'w');
        fprintf(fid, '{\n');
        fprintf(fid, '  "error": "JSON encoding failed",\n');
        fprintf(fid, '  "reason": "%s"\n', ME.message);
        fprintf(fid, '}\n');
        fclose(fid);
        return;
    end

    fid = fopen(output_file, 'w');
    if fid < 0
        error('Could not open output JSON file: %s', output_file);
    end
    cleaner = onCleanup(@() fclose(fid));
    fprintf(fid, '%s', json_str);
end

function C = infer_channel_count(Results, Sim)
    if isfield(Results, 'centers') && ~isempty(Results.centers)
        C = size(Results.centers, 2);
    elseif isfield(Results, 'maps_true') && ~isempty(Results.maps_true)
        C = size(Results.maps_true, 2);
    elseif isfield(Sim, 'X_noisy') && ~isempty(Sim.X_noisy)
        C = size(Sim.X_noisy, 1);
    else
        C = 0;
    end
end

function ch_labels_raw = infer_channel_labels(Sim, META, C, util)
    ch_labels_raw = {};
    if isfield(Sim, 'channel_labels') && ~isempty(Sim.channel_labels)
        ch_labels_raw = cellstr(Sim.channel_labels);
    elseif isfield(Sim, 'chanlocs') && ~isempty(Sim.chanlocs)
        ch_labels_raw = util.channel_labels_from_chanlocs(Sim.chanlocs, numel(Sim.chanlocs));
    elseif isfield(META, 'channel_labels') && ~isempty(META.channel_labels)
        ch_labels_raw = cellstr(META.channel_labels);
    end
    if isempty(ch_labels_raw) || length(ch_labels_raw) < C
        ch_labels_raw = arrayfun(@(i) sprintf('Ch%03d', i), 1:C, 'UniformOutput', false);
    end
end

function state_data = add_template_state_fields(state_data, ta, k)
    if isfield(ta, 'labels') && numel(ta.labels) >= k
        state_data.template_label = ta.labels{k};
    end
    if isfield(ta, 'correlations') && numel(ta.correlations) >= k
        state_data.template_correlation = double(real(ta.correlations(k)));
    end
    if isfield(ta, 'polarity') && numel(ta.polarity) >= k
        state_data.template_polarity = double(real(ta.polarity(k)));
    end
end

function s = template_alignment_summary(ta)
    s = struct();
    fields = {'mean_correlation', 'median_correlation', 'min_correlation', ...
        'n_strong_matches', 'strong_threshold', 'n_common_channels'};
    for i = 1:numel(fields)
        f = fields{i};
        if isfield(ta, f)
            s.(f) = double(real(ta.(f)));
        end
    end
    if isfield(ta, 'channel_match_mode')
        s.channel_match_mode = ta.channel_match_mode;
    end
    if isfield(ta, 'labels')
        s.labels = ta.labels;
    end
    if isfield(ta, 'correlations')
        s.correlations = double(real(ta.correlations(:)'));
    end
    if isfield(ta, 'template_labels')
        s.template_labels = ta.template_labels;
    end
end

function metrics = empty_recovery_metrics()
    metrics = struct( ...
        'n_matched', 0, ...
        'sensitivity', NaN, ...
        'precision', NaN, ...
        'f1_score', NaN, ...
        'mean_recovery_matched', NaN, ...
        'mean_recovery_padded', NaN, ...
        'match_similarities', []);
end

function s = backfit_summary(diag)
    s = struct();
    fields = {'ok', 'message', 'n_samples', 'template_labels', ...
        'true_state_template_labels', 'estimated_state_template_labels', ...
        'coverage_true', 'coverage_estimated', 'coverage_difference', ...
        'coverage_corr', 'coverage_spearman', 'coverage_mae', ...
        'coverage_rmse', 'coverage_l1', 'coverage_linf', ...
        'confusion_counts', 'confusion_row_normalized', ...
        'n_overlap_samples', 'overlap_sample_fraction', ...
        'estimated_to_true_state_index', 'hard', 'mixture'};
    for i = 1:numel(fields)
        f = fields{i};
        if isfield(diag, f)
            s.(f) = diag.(f);
        end
    end
end

function idx = find_selected_model_index(Results)
    idx = NaN;
    if ~isfield(Results, 'K_candidates') || isempty(Results.K_candidates) || ...
            ~isfield(Results, 'K_model_selected') || isempty(Results.K_model_selected)
        return;
    end
    hit = find(double(Results.K_candidates(:)) == double(Results.K_model_selected), 1, 'first');
    if ~isempty(hit)
        idx = hit;
    end
end
