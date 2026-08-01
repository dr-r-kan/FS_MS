function audit = recover_timestamped_transition_backfits(manifest_csv, legacy_results_dir, output_dir, varargin)
%RECOVER_TIMESTAMPED_TRANSITION_BACKFITS Export participant-aligned exact transitions.
% Reuses the deterministic K=6 hierarchy and per-record GFP caches, then
% backfits full signals and stores every genuine state-change sample.

    p = inputParser;
    addRequired(p, 'manifest_csv', @(x) ischar(x) || isstring(x));
    addRequired(p, 'legacy_results_dir', @(x) ischar(x) || isstring(x));
    addRequired(p, 'output_dir', @(x) ischar(x) || isstring(x));
    addParameter(p, 'conditions', {'II','preict'}, @(x) iscell(x) || isstring(x));
    addParameter(p, 'validation_windows_csv', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'max_records', Inf, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'overwrite', false, @islogical);
    parse(p, manifest_csv, legacy_results_dir, output_dir, varargin{:});
    cfg_run = p.Results;

    root_dir = fileparts(mfilename('fullpath'));
    addpath(root_dir);
    vg_root = fullfile(root_dir, 'external', 'VG-microstates');
    addpath(vg_root);
    setup_microstate_paths(vg_root);
    util = microstate_utilities();

    output_dir = char(output_dir);
    cache_dir = fullfile(output_dir, 'transition_cache');
    recovered_peak_dir = fullfile(output_dir, 'recovered_peak_cache');
    ensure_dir(output_dir); ensure_dir(cache_dir); ensure_dir(recovered_peak_dir);

    legacy_solution = fullfile(char(legacy_results_dir), 'global', 'global_solution.mat');
    legacy_manifest_csv = fullfile(char(legacy_results_dir), 'normalised_input_manifest.csv');
    legacy_peak_dir = fullfile(char(legacy_results_dir), 'gfp_peaks');
    must_exist(manifest_csv); must_exist(legacy_solution); must_exist(legacy_manifest_csv);
    S = load(legacy_solution, 'Fit', 'cfg', 'common_labels', 'common_chanlocs');
    cfg = S.cfg;
    cfg.K_candidates = 6;
    cfg.spm_prior_weight = 1;
    cfg.template_file = fullfile(vg_root, 'MetaMaps_2023_06.set');
    cfg.template_reference_K = 7;
    cfg.max_global_maps = 50000;
    common_labels = S.common_labels;
    common_can = util.canonical_channel_labels(common_labels);
    common_pos = util.positions_from_chanlocs(S.common_chanlocs, numel(common_labels));

    paper = readtable(manifest_csv, 'TextType', 'string', 'VariableNamingRule', 'preserve', 'Delimiter', ',');
    paper = sortrows(paper, 'file_path');
    legacy = readtable(legacy_manifest_csv, 'TextType', 'string', 'VariableNamingRule', 'preserve', 'Delimiter', ',');
    legacy = sortrows(legacy, 'file_path');
    cache_by_path = containers.Map('KeyType', 'char', 'ValueType', 'char');
    for i = 1:height(legacy)
        cache_file = fullfile(legacy_peak_dir, sprintf('%03d_%s_%s_gfp_peaks.mat', i, ...
            clean_key(legacy.participant(i)), clean_key(legacy.condition(i))));
        if isfile(cache_file)
            cache_by_path(path_key(legacy.file_path(i))) = cache_file;
        end
    end

    rec_paths = strings(height(paper), 1);
    recovered_count = 0;
    peak_counts = zeros(height(paper), 1);
    for i = 1:height(paper)
        key = path_key(paper.file_path(i));
        if isKey(cache_by_path, key)
            rec_paths(i) = string(cache_by_path(key));
        else
            recovered_file = fullfile(recovered_peak_dir, sprintf('%04d_%s_%s_gfp_peaks.mat', i, ...
                clean_key(paper.participant(i)), clean_key(paper.condition(i))));
            if cfg_run.overwrite || ~isfile(recovered_file)
                request = struct('command', 'extract_peak_record', ...
                    'file_path', char(paper.file_path(i)), 'common_can', {common_can}, ...
                    'common_pos', common_pos, 'cfg', cfg);
                [rec, ~] = fit_microstate_hierarchical_dataset(request);
                save(recovered_file, 'rec', 'cfg', '-v7.3');
            end
            rec_paths(i) = string(recovered_file);
            recovered_count = recovered_count + 1;
        end
        C = load(rec_paths(i), 'rec');
        if ~strcmpi(path_key(C.rec.file_path), key)
            error('Peak cache points to the wrong record: %s', rec_paths(i));
        end
        peak_counts(i) = size(C.rec.maps_norm, 1);
    end

    fit_file = fullfile(output_dir, 'recovered_global_solution.mat');
    if cfg_run.overwrite || ~isfile(fit_file)
        total_peaks = sum(peak_counts);
        exact_manifest = height(legacy) == height(paper) && ...
            all(strcmpi(cellstr(legacy.file_path), cellstr(paper.file_path)));
        if exact_manifest && recovered_count == 0
            Fit = S.Fit;
            expected_labels = string({'A','F','C','G','B','E'});
            if ~isfield(Fit, 'template_alignment') || ...
                    ~isequal(string(Fit.template_alignment.labels(:))', expected_labels)
                error('Exact cached global fit does not carry the authoritative fitted-state labels.');
            end
            fit_idx = nan(0, 1);
        else
            fit_idx = deterministic_subsample(total_peaks, cfg.max_global_maps);
            maps = nan(numel(fit_idx), numel(common_labels), 'single');
            gfp = nan(numel(fit_idx), 1);
            offset = 0;
            written = 0;
            for i = 1:height(paper)
                C = load(rec_paths(i), 'rec');
                local = fit_idx(fit_idx > offset & fit_idx <= offset + peak_counts(i)) - offset;
                if ~isempty(local)
                    target = written + (1:numel(local));
                    maps(target, :) = C.rec.maps_norm(local, :);
                    gfp(target) = double(C.rec.gfp_peak(local));
                    written = written + numel(local);
                end
                offset = offset + peak_counts(i);
            end
            assert(written == numel(fit_idx) && all(isfinite(gfp)), 'Global peak-bank recovery is incomplete.');
            request = struct('command', 'fit_global_maps', 'maps', maps, 'gfp', gfp, ...
                'cfg', cfg, 'common_labels', {common_labels}, 'common_chanlocs', S.common_chanlocs, ...
                'manual_labels', {{'A','F','C','G','B','E'}});
            [Fit, ~] = fit_microstate_hierarchical_dataset(request);
        end
        save(fit_file, 'Fit', 'cfg', 'common_labels', 'common_can', 'common_pos', ...
            'fit_idx', 'total_peaks', 'rec_paths', '-v7.3');
    else
        F = load(fit_file, 'Fit', 'total_peaks');
        Fit = F.Fit;
        total_peaks = F.total_peaks;
    end

    validation = struct('performed', false);
    if strlength(string(cfg_run.validation_windows_csv)) > 0
        must_exist(cfg_run.validation_windows_csv);
        validation = validate_focused_window(cfg_run.validation_windows_csv, paper, rec_paths, legacy_results_dir);
    end

    wanted = ismember(lower(strtrim(paper.condition)), lower(string(cfg_run.conditions(:))));
    selected = find(wanted);
    selected = selected(1:min(numel(selected), floor(cfg_run.max_records)));
    index_parts = cell(numel(selected), 1);
    full_validation = false;
    current_participant = "";
    record_fit = struct();
    for j = 1:numel(selected)
        i = selected(j);
        if current_participant ~= paper.participant(i)
            current_participant = paper.participant(i);
            participant_file = fullfile(char(legacy_results_dir), 'participants', ...
                char(current_participant), 'participant_solution.mat');
            must_exist(participant_file);
            participant_solution = load(participant_file, 'Fit');
            record_fit = participant_solution.Fit;
            if ~isfield(record_fit, 'template_alignment') || ...
                    ~isequal(string(record_fit.template_alignment.labels(:))', string({'A','F','C','G','B','E'}))
                error('Participant %s does not retain the authoritative fitted-state label order.', current_participant);
            end
        end
        cache_file = fullfile(cache_dir, sprintf('%04d_%s_%s_transitions.mat', i, ...
            clean_key(paper.participant(i)), clean_key(paper.condition(i))));
        use_cache = ~cfg_run.overwrite && isfile(cache_file);
        if use_cache
            T = load(cache_file, 'transition');
            use_cache = isfield(T, 'transition') && isfield(T.transition, 'fit_centers') && ...
                isfield(T.transition, 'full_backfit_validation') && ...
                isequal(size(T.transition.fit_centers), size(record_fit.centers)) && ...
                max(abs(double(T.transition.fit_centers(:)) - double(record_fit.centers(:)))) < 1e-7;
        end
        if ~use_cache
            request = struct('command', 'timestamped_transitions', ...
                'file_path', char(paper.file_path(i)), 'common_can', {common_can}, ...
                'common_pos', common_pos, 'cfg', cfg, 'fit', record_fit, ...
                'validate_full_backfit', j == 1);
            [transition, ~] = fit_microstate_hierarchical_dataset(request);
            transition.fit_centers = single(record_fit.centers);
            save(cache_file, 'transition');
        else
            transition = T.transition;
        end
        full_validation = full_validation || logical(transition.full_backfit_validation);
        index_parts{j} = table(double(i), paper.participant(i), paper.condition(i), paper.group(i), ...
            paper.file_path(i), string(cache_file), double(transition.sfreq), double(transition.n_samples), ...
            double(numel(transition.peak_sample)), double(numel(transition.transition_sample)), ...
            'VariableNames', {'manifest_index','participant','condition','group','file_path','cache_file', ...
            'sfreq','n_samples','n_gfp_peaks','n_state_changes'});
        if mod(j, 10) == 0 || j == numel(selected)
            writetable(vertcat(index_parts{1:j}), fullfile(output_dir, 'transition_index.csv'));
            fprintf('[%d/%d] exact transition records exported\n', j, numel(selected));
        end
    end
    index = vertcat(index_parts{:});
    writetable(index, fullfile(output_dir, 'transition_index.csv'));

    audit = struct('manifest_csv', char(manifest_csv), 'legacy_results_dir', char(legacy_results_dir), ...
        'output_dir', output_dir, 'n_manifest_records', height(paper), 'n_recovered_peak_records', recovered_count, ...
        'n_global_peaks', total_peaks, 'n_global_fit_peaks', Fit.n_maps, ...
        'manual_labels_by_fitted_state', {{'A','F','C','G','B','E'}}, ...
        'n_timestamped_records', height(index), 'n_transitions', sum(index.n_state_changes), ...
        'compact_export_validated_against_full_backfit', full_validation, ...
        'focused_window_fit_recovery_validation', validation, ...
        'transition_time_resolution', 'one EEG sample; 1/sfreq seconds', ...
        'transition_ordering', 'exact hard backfit ordering; GFP-peak labels joined at midpoint sample boundaries');
    audit.backfit_fit_level = 'participant-specific manually aligned K=6 solution';
    fid = fopen(fullfile(output_dir, 'recovery_audit.json'), 'w');
    cleaner = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid, '%s', jsonencode(audit, 'PrettyPrint', true));
end

function must_exist(path_value)
    if ~isfile(path_value), error('Required file not found: %s', path_value); end
end

function ensure_dir(path_value)
    if ~isfolder(path_value), mkdir(path_value); end
end

function key = clean_key(value)
    key = regexprep(char(string(value)), '[^A-Za-z0-9_-]', '_');
    if isempty(key), key = 'empty'; end
end

function key = path_key(value)
    key = lower(strrep(char(string(value)), '/', '\'));
end

function idx = deterministic_subsample(N, maxN)
    if isempty(maxN) || ~isfinite(maxN) || N <= maxN
        idx = (1:N)';
    else
        idx = unique(round(linspace(1, N, round(maxN))))';
        idx = idx(1:min(numel(idx), maxN));
    end
end

function validation = validate_focused_window(validation_csv, paper, rec_paths, results_dir)
    expected = readtable(validation_csv, 'TextType', 'string', 'VariableNamingRule', 'preserve', 'Delimiter', ',');
    row = expected(1, :);
    paper_index = find(strcmpi(cellstr(paper.file_path), cellstr(row.file_path)), 1, 'first');
    if isempty(paper_index), error('Validation record is absent from the paper manifest.'); end
    participant_file = fullfile(char(results_dir), 'participants', char(row.participant), 'participant_solution.mat');
    must_exist(participant_file);
    participant_solution = load(participant_file, 'Fit');
    Fit = participant_solution.Fit;
    C = load(rec_paths(paper_index), 'rec');
    maps = double(C.rec.maps_norm);
    maps = maps ./ max(vecnorm(maps, 2, 2), eps);
    centers = double(Fit.centers);
    centers = centers ./ max(vecnorm(centers, 2, 2), eps);
    [~, fitted] = max(abs(maps * centers'), [], 2);
    fitted_labels = string(Fit.template_alignment.labels(:));
    canonical_labels = string({'F','E','G','B','A','C'});
    fitted_to_canonical = zeros(1, 6);
    for i = 1:6, fitted_to_canonical(i) = find(canonical_labels == fitted_labels(i), 1); end
    state = fitted_to_canonical(fitted(:));
    peak_time = (double(C.rec.peak_sample(:)) - 1) / double(C.rec.sfreq);
    keep = peak_time >= double(row.window_start_s) & peak_time < double(row.window_end_s);
    sequence = state(keep);
    sequence = sequence(:);
    collapsed = sequence([true; diff(sequence) ~= 0]);
    observed = zeros(6, 6);
    for i = 1:max(0, numel(collapsed) - 1)
        observed(collapsed(i), collapsed(i + 1)) = observed(collapsed(i), collapsed(i + 1)) + 1;
    end
    target = zeros(6, 6);
    for source = 1:6
        for destination = 1:6
            target(source, destination) = double(row.(sprintf('t_%d_%d', source, destination)));
        end
    end
    if numel(sequence) ~= double(row.n_gfp_peaks) || ~isequal(observed, target)
        error(['Recovered global fit failed the preserved focused-window gate: peaks %d/%d, ' ...
            'transition absolute difference %d.'], numel(sequence), double(row.n_gfp_peaks), sum(abs(observed - target), 'all'));
    end
    validation = struct('performed', true, 'file_path', char(row.file_path), ...
        'window_start_s', double(row.window_start_s), 'window_end_s', double(row.window_end_s), ...
        'n_gfp_peaks', numel(sequence), 'n_state_changes', sum(observed, 'all'), ...
        'transition_matrix_exact_match', true);
end
