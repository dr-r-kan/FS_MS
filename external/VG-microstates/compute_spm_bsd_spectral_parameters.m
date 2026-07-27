function [Results, output_files] = compute_spm_bsd_spectral_parameters(input_path, varargin)
%COMPUTE_SPM_BSD_SPECTRAL_PARAMETERS Fit BSD spectra by participant/condition.
%
%   R = compute_spm_bsd_spectral_parameters(folder_or_manifest)
%
% Input may be a folder containing .set/.mat EEG files or a CSV manifest with
% file_path plus optional participant, condition, and group columns. The main
% output is one aperiodic row per participant-condition and one periodic row
% per participant-condition-band.
%
% Stretch microstate fits:
%   pass 'microstate_results_mat', 'outputs/.../hierarchical_microstate_results.mat'
%   to also try per-state spectra from hard backfit activations. Very short
%   state runs are skipped rather than concatenated across discontinuities.

    setup_microstate_paths();
    if nargin < 1 || isempty(input_path)
        input_path = fullfile(getenv('HOME'), 'EEG', 'LEMON');
    elseif (ischar(input_path) || isstring(input_path)) && strcmpi(strtrim(char(input_path)), 'LEMON')
        input_path = fullfile(getenv('HOME'), 'EEG', 'LEMON');
    end

    util = microstate_utilities();
    repo_cfg = util.load_config();

    p = inputParser;
    addRequired(p, 'input_path', @(x) ischar(x) || isstring(x));
    addParameter(p, 'output_dir', fullfile('outputs', 'spm_bsd_spectra'), @(x) ischar(x) || isstring(x));
    addParameter(p, 'freqs', 1:45, @(x) isnumeric(x) && isvector(x));
    addParameter(p, 'peak_bands', [1 4; 4 8; 8 12; 12 30; 30 45], @(x) isnumeric(x) && size(x, 2) == 2);
    addParameter(p, 'peak_band_names', {'delta', 'theta', 'alpha', 'beta', 'low_gamma'}, @(x) iscell(x) || isstring(x));
    addParameter(p, 'apply_average_reference', logical(repo_cfg.preprocessing.apply_average_reference), @islogical);
    addParameter(p, 'filter_band', [], @(x) isempty(x) || (isnumeric(x) && numel(x) == 2));
    addParameter(p, 'use_scalp_channels', true, @islogical);
    addParameter(p, 'exclude_channels', {'PO9', 'PO10'}, @(x) iscell(x) || isstring(x));
    addParameter(p, 'welch_window_s', 2, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'welch_overlap', 0.5, @(x) isnumeric(x) && isscalar(x) && x >= 0 && x < 1);
    addParameter(p, 'spm_path', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'bsd_path', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'fitlog', true, @islogical);
    addParameter(p, 'separatenull', true, @islogical);
    addParameter(p, 'powerline', [49 51], @(x) isempty(x) || (isnumeric(x) && numel(x) == 2));
    addParameter(p, 'save_bsd_models', false, @islogical);
    addParameter(p, 'microstate_results_mat', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'min_state_segment_cycles', 3, @(x) isnumeric(x) && isscalar(x) && x >= 0);
    addParameter(p, 'min_state_segment_s', 0, @(x) isnumeric(x) && isscalar(x) && x >= 0);
    addParameter(p, 'verbose', true, @islogical);
    parse(p, input_path, varargin{:});
    cfg = p.Results;

    cfg.input_path = char(cfg.input_path);
    cfg.output_dir = util.resolve_path(cfg.output_dir, util.project_root());
    cfg.freqs = unique(double(cfg.freqs(:)'));
    cfg.freqs = cfg.freqs(isfinite(cfg.freqs) & cfg.freqs > 0);
    cfg.peak_bands = double(cfg.peak_bands);
    cfg.peak_band_names = cellstr(string(cfg.peak_band_names(:)));
    cfg.exclude_channels = cellstr(string(cfg.exclude_channels(:)));
    cfg.microstate_results_mat = char(cfg.microstate_results_mat);
    if ~isempty(cfg.microstate_results_mat)
        cfg.microstate_results_mat = util.resolve_path(cfg.microstate_results_mat, util.project_root());
    end
    if isempty(cfg.freqs)
        error('freqs must contain at least one positive frequency.');
    end
    if numel(cfg.peak_band_names) ~= size(cfg.peak_bands, 1)
        cfg.peak_band_names = arrayfun(@(i) sprintf('band_%02d', i), 1:size(cfg.peak_bands, 1), 'UniformOutput', false)';
    end

    util.ensure_dir(cfg.output_dir);
    ensure_spm_bsd(cfg);

    manifest = read_or_build_manifest_local(cfg.input_path);
    writetable(manifest, fullfile(cfg.output_dir, 'normalised_input_manifest.csv'));

    if cfg.verbose
        fprintf('\nBSD spectral parameters\nInput rows: %d\nOutput: %s\n', height(manifest), cfg.output_dir);
    end

    [spectra, spectrum_summary] = participant_condition_spectra(manifest, cfg, util);
    [aperiodic_table, periodic_table, fit_summary, bsd_models] = fit_spectra_table(spectra, cfg);

    state_aperiodic_table = table();
    state_periodic_table = table();
    state_fit_summary = table();
    state_spectrum_summary = table();
    state_bsd_models = {};
    if ~isempty(cfg.microstate_results_mat)
        [state_spectra, state_spectrum_summary] = microstate_activation_spectra(manifest, cfg, util);
        if ~isempty(state_spectra)
            [state_aperiodic_table, state_periodic_table, state_fit_summary, state_bsd_models] = fit_spectra_table(state_spectra, cfg);
        end
    end

    output_files = struct();
    output_files.manifest_csv = fullfile(cfg.output_dir, 'normalised_input_manifest.csv');
    output_files.aperiodic_csv = fullfile(cfg.output_dir, 'participant_condition_aperiodic_parameters.csv');
    output_files.periodic_csv = fullfile(cfg.output_dir, 'participant_condition_periodic_parameters.csv');
    output_files.spectrum_summary_csv = fullfile(cfg.output_dir, 'participant_condition_spectrum_summary.csv');
    output_files.fit_summary_csv = fullfile(cfg.output_dir, 'participant_condition_bsd_fit_summary.csv');
    output_files.results_mat = fullfile(cfg.output_dir, 'spm_bsd_spectral_parameters.mat');
    writetable(aperiodic_table, output_files.aperiodic_csv);
    writetable(periodic_table, output_files.periodic_csv);
    writetable(spectrum_summary, output_files.spectrum_summary_csv);
    writetable(fit_summary, output_files.fit_summary_csv);

    if ~isempty(cfg.microstate_results_mat)
        output_files.state_aperiodic_csv = fullfile(cfg.output_dir, 'participant_condition_microstate_aperiodic_parameters.csv');
        output_files.state_periodic_csv = fullfile(cfg.output_dir, 'participant_condition_microstate_periodic_parameters.csv');
        output_files.state_spectrum_summary_csv = fullfile(cfg.output_dir, 'participant_condition_microstate_spectrum_summary.csv');
        output_files.state_fit_summary_csv = fullfile(cfg.output_dir, 'participant_condition_microstate_bsd_fit_summary.csv');
        writetable(state_aperiodic_table, output_files.state_aperiodic_csv);
        writetable(state_periodic_table, output_files.state_periodic_csv);
        writetable(state_spectrum_summary, output_files.state_spectrum_summary_csv);
        writetable(state_fit_summary, output_files.state_fit_summary_csv);
    end

    Results = struct();
    Results.source = 'compute_spm_bsd_spectral_parameters';
    Results.created = char(datetime('now', 'Format', 'yyyyMMdd''T''HHmmss'));
    Results.cfg = cfg;
    Results.manifest = manifest;
    Results.spectrum_summary = spectrum_summary;
    Results.aperiodic = aperiodic_table;
    Results.periodic = periodic_table;
    Results.fit_summary = fit_summary;
    Results.state_spectrum_summary = state_spectrum_summary;
    Results.state_aperiodic = state_aperiodic_table;
    Results.state_periodic = state_periodic_table;
    Results.state_fit_summary = state_fit_summary;
    Results.output_files = output_files;
    if cfg.save_bsd_models
        Results.bsd_models = bsd_models;
        Results.state_bsd_models = state_bsd_models;
    end
    save(output_files.results_mat, 'Results', '-v7.3');

    if cfg.verbose
        fprintf('Aperiodic CSV: %s\nPeriodic CSV: %s\n', output_files.aperiodic_csv, output_files.periodic_csv);
        if ~isempty(cfg.microstate_results_mat)
            fprintf('Microstate-state rows fitted: %d\n', height(state_aperiodic_table));
        end
    end
end

function ensure_spm_bsd(cfg)
    candidates = {};
    candidates = append_candidate(candidates, cfg.bsd_path);
    candidates = append_candidate(candidates, getenv('SPM_BSD_PATH'));
    candidates = append_candidate(candidates, fullfile(getenv('SPM_PATH'), 'toolbox', 'BSD'));
    candidates = append_candidate(candidates, fullfile(getenv('HOME'), 'spm', 'toolbox', 'BSD'));
    candidates = append_candidate(candidates, fullfile(getenv('HOME'), 'Downloads', 'spm', 'toolbox', 'BSD'));

    spm_roots = {};
    spm_roots = append_candidate(spm_roots, cfg.spm_path);
    spm_roots = append_candidate(spm_roots, getenv('SPM_PATH'));
    spm_roots = append_candidate(spm_roots, fullfile(getenv('HOME'), 'spm'));
    spm_roots = append_candidate(spm_roots, fullfile(getenv('HOME'), 'Downloads', 'spm'));

    for i = 1:numel(spm_roots)
        if isfolder(spm_roots{i})
            addpath(spm_roots{i});
            dcm_meeg = fullfile(spm_roots{i}, 'toolbox', 'dcm_meeg');
            if isfolder(dcm_meeg)
                addpath(dcm_meeg);
            end
        end
    end
    for i = 1:numel(candidates)
        if isfolder(candidates{i})
            addpath(candidates{i});
        end
    end
    add_spm_bsd_compat_if_needed();

    if exist('spm_bsd', 'file') ~= 2
        error('spm_bsd not found. Pass ''bsd_path'' or set SPM_BSD_PATH.');
    end
    if exist('spm', 'file') ~= 2
        error('SPM not found. Pass ''spm_path'' or set SPM_PATH.');
    end
end

function add_spm_bsd_compat_if_needed()
    try
        n = nargin('spm_logdet');
    catch
        n = [];
    end
    if isempty(n) || n ~= 1
        return;
    end
    compat_dir = fullfile(fileparts(mfilename('fullpath')), '_internal', 'matlab', 'compat', 'spm_bsd');
    if isfolder(compat_dir)
        addpath(compat_dir, '-begin');
    end
end

function candidates = append_candidate(candidates, pth)
    if nargin < 2 || isempty(pth)
        return;
    end
    pth = char(string(pth));
    if isempty(strtrim(pth))
        return;
    end
    candidates{end+1} = pth;
end

function [spectra, summary] = participant_condition_spectra(manifest, cfg, util)
    spectra = {};
    rows = {};
    key = string(manifest.participant) + "|" + string(manifest.condition);
    levels = unique(key, 'stable');

    for i = 1:numel(levels)
        parts = split(levels(i), "|");
        participant = parts(1);
        condition = parts(2);
        mask = string(manifest.participant) == participant & string(manifest.condition) == condition;
        [power, n_files, n_samples, sfreqs, status, message] = average_file_spectra(manifest(mask, :), cfg, util);
        group_value = first_string(manifest.group(mask));
        spectra{end+1, 1} = make_spectrum_record(participant, condition, group_value, "", NaN, "", ...
            "participant_condition", cfg.freqs, power, n_files, n_samples, sfreqs, status, message); %#ok<AGROW>
        rows{end+1, 1} = spectrum_summary_row(spectra{end}); %#ok<AGROW>
    end

    summary = vertcat_nonempty(rows);
end

function [power, n_files, n_samples_total, sfreqs, status, message] = average_file_spectra(T, cfg, util)
    power_acc = zeros(numel(cfg.freqs), 1);
    weight_acc = 0;
    n_files = 0;
    n_samples_total = 0;
    sfreqs = [];
    status = "ok";
    message = "";

    for i = 1:height(T)
        try
            [data, sfreq, chanlocs, labels, pos] = load_eeg_file_local(T.file_path(i), util);
            data = select_spectral_channels(data, labels, chanlocs, pos, cfg, util);
            data = preprocess_spectrum_matrix(data, sfreq, cfg, util);
            pxx = welch_mean_power(data, sfreq, cfg.freqs, cfg.welch_window_s, cfg.welch_overlap);
            weight = size(data, 2);
            power_acc = power_acc + pxx(:) * weight;
            weight_acc = weight_acc + weight;
            n_files = n_files + 1;
            n_samples_total = n_samples_total + size(data, 2);
            sfreqs(end+1) = sfreq; %#ok<AGROW>
        catch ME
            status = "partial";
            message = append_message(message, sprintf('%s: %s', char(T.file_path(i)), ME.message));
        end
    end

    if weight_acc <= 0
        power = nan(numel(cfg.freqs), 1);
        status = "failed";
        if strlength(message) == 0
            message = "No spectra could be estimated.";
        end
    else
        power = power_acc ./ weight_acc;
    end
end

function data = select_spectral_channels(data, labels, chanlocs, ~, cfg, util)
    keep = true(size(data, 1), 1);
    if cfg.use_scalp_channels
        keep = util.scalp_channel_mask(chanlocs, size(data, 1));
    end
    can = util.canonical_channel_labels(labels);
    exclude_can = util.canonical_channel_labels(cfg.exclude_channels);
    keep = keep & ~ismember(can(:), exclude_can(:));
    if nnz(keep) < 1
        error('No channels remain after scalp/exclusion filtering.');
    end
    data = double(data(keep, :));
    good_t = all(isfinite(data), 1);
    data = data(:, good_t);
    if size(data, 2) < 8
        error('Too few valid samples for spectral estimation.');
    end
end

function X = preprocess_spectrum_matrix(X, sfreq, cfg, util)
    X = double(X);
    X = X - mean(X, 2, 'omitnan');
    if cfg.apply_average_reference && size(X, 1) > 1
        X = X - mean(X, 1, 'omitnan');
    end
    if ~isempty(cfg.filter_band)
        X = util.bandpass_filter(X, sfreq, cfg.filter_band);
    end
end

function pxx = welch_mean_power(X, sfreq, freqs, window_s, overlap)
    if max(freqs) >= sfreq / 2
        error('Requested max frequency %.3g Hz exceeds Nyquist %.3g Hz.', max(freqs), sfreq / 2);
    end
    [n_channels, n_samples] = size(X);
    win_len = min(n_samples, max(8, round(window_s * sfreq)));
    step = max(1, round(win_len * (1 - overlap)));
    starts = 1:step:(n_samples - win_len + 1);
    if isempty(starts)
        starts = 1;
        win_len = n_samples;
    end
    nfft = 2 ^ nextpow2(win_len);
    win = 0.5 - 0.5 * cos(2 * pi * (0:(win_len - 1)) / max(win_len - 1, 1));
    win_power = sum(win .^ 2);
    f = (0:(nfft / 2))' * sfreq / nfft;
    acc = zeros(numel(f), 1);
    n = 0;
    for s = starts
        seg = X(:, s:(s + win_len - 1));
        seg = seg - mean(seg, 2, 'omitnan');
        Y = fft(seg .* win, nfft, 2);
        P = abs(Y(:, 1:(nfft / 2 + 1))) .^ 2 ./ max(sfreq * win_power, eps);
        if size(P, 2) > 2
            P(:, 2:end-1) = 2 * P(:, 2:end-1);
        end
        acc = acc + mean(P, 1, 'omitnan')';
        n = n + 1;
    end
    pxx = interp1(f, acc ./ max(n, 1), freqs(:), 'linear', NaN);
    pxx = max(real(pxx), eps);
    if any(~isfinite(pxx))
        error('PSD interpolation produced non-finite values.');
    end
    pxx = pxx(:);
    if n_channels < 1
        error('No channels supplied for spectral estimation.');
    end
end

function rec = make_spectrum_record(participant, condition, group, state_label, state_index, file_path, scope, freqs, power, n_files, n_samples, sfreqs, status, message)
    rec = struct();
    rec.participant = string(participant);
    rec.condition = string(condition);
    rec.group = string(group);
    rec.state_label = string(state_label);
    rec.state_index = double(state_index);
    rec.file_path = string(file_path);
    rec.scope = string(scope);
    rec.freqs = double(freqs(:));
    rec.power = double(power(:));
    rec.n_files = double(n_files);
    rec.n_samples = double(n_samples);
    rec.sfreq_mean = mean_or_nan(sfreqs);
    rec.status = string(status);
    rec.message = string(message);
end

function row = spectrum_summary_row(rec)
    row = table(rec.participant, rec.condition, rec.group, rec.scope, rec.state_index, rec.state_label, rec.file_path, ...
        rec.n_files, rec.n_samples, rec.sfreq_mean, rec.status, rec.message, ...
        'VariableNames', {'participant', 'condition', 'group', 'scope', 'state_index', 'state_label', 'file_path', ...
        'n_files', 'n_samples', 'sfreq_mean', 'status', 'message'});
end

function [aperiodic_table, periodic_table, fit_summary, bsd_models] = fit_spectra_table(spectra, cfg)
    a_rows = {};
    p_rows = {};
    f_rows = {};
    bsd_models = {};
    for i = 1:numel(spectra)
        rec = spectra{i};
        model_name = clean_key(sprintf('%s_%s_%s_%s', char(rec.scope), char(rec.participant), char(rec.condition), char(rec.state_label)));
        [fit, BSD] = fit_one_bsd_spectrum(rec.freqs, rec.power, cfg, model_name);
        a_rows{end+1, 1} = aperiodic_row(rec, fit); %#ok<AGROW>
        p_rows{end+1, 1} = periodic_rows(rec, fit, cfg); %#ok<AGROW>
        f_rows{end+1, 1} = fit_summary_row(rec, fit); %#ok<AGROW>
        if cfg.save_bsd_models
            bsd_models{end+1, 1} = BSD; %#ok<AGROW>
            save(fullfile(cfg.output_dir, [model_name '_BSD.mat']), 'BSD', '-v7.3');
        end
    end
    aperiodic_table = vertcat_nonempty(a_rows);
    periodic_table = vertcat_nonempty(p_rows);
    fit_summary = vertcat_nonempty(f_rows);
end

function [fit, BSD] = fit_one_bsd_spectrum(freqs, power, cfg, model_name)
    BSD = [];
    fit = empty_fit();
    if any(~isfinite(power)) || all(power <= 0)
        fit.status = "failed";
        fit.message = "Spectrum is empty or non-finite.";
        return;
    end
    try
        BSD = struct();
        BSD.name = fullfile(cfg.output_dir, [model_name '_BSD.mat']);
        BSD.xY.Hz = freqs(:)';
        BSD.xY.y = {max(power(:), eps)};
        BSD.xY.dt = 1;
        BSD.xU.X = sparse(1, 0);
        BSD.fqs = matrix_rows_to_cells(cfg.peak_bands);
        BSD.options.DATA = 0;
        BSD.options.SAVE = 0;
        BSD.options.RESULTS = 0;
        BSD.options.spatial = 'chan';
        BSD.options.Nmodes = 1;
        BSD.options.Fdcm = freqs(:)';
        BSD.options.fitlog = cfg.fitlog;
        BSD.options.separatenull = cfg.separatenull;
        BSD.options.powerline = cfg.powerline;
        BSD.options.noprint = ~cfg.verbose;
        BSD.options.nograph = true;
        BSD = spm_bsd(BSD);

        spec = spm_bsd_param2spec(BSD.Ep, BSD.M);
        noise_spec = spec;
        if cfg.separatenull && isfield(BSD, 'null') && isfield(BSD.null, 'Ep')
            noise_spec = spm_bsd_param2spec(BSD.null.Ep, BSD.null.M);
        end
        fit.status = "ok";
        fit.message = "";
        fit.F = scalar_field_or_nan(BSD, 'F');
        fit.peak_frequency_hz = vector_or_nan(spec, 'freq', size(cfg.peak_bands, 1));
        fit.peak_fwhm_hz = vector_or_nan(spec, 'fwhm', size(cfg.peak_bands, 1));
        fit.peak_amplitude = vector_or_nan(spec, 'ampl', size(cfg.peak_bands, 1));
        fit.peak_probability = posterior_peak_probability(BSD, size(cfg.peak_bands, 1));
        noise = vector_or_nan(noise_spec, 'noise', 3);
        fit.aperiodic_intercept = noise(1);
        fit.aperiodic_exponent = noise(2);
        fit.aperiodic_knee_parameter = noise(3);
        fit.aperiodic_knee_frequency_hz = noise(3) .^ (1 ./ max(noise(2), eps));
    catch ME
        fit.status = "failed";
        fit.message = string(ME.message);
    end
end

function fit = empty_fit()
    fit = struct();
    fit.status = "failed";
    fit.message = "";
    fit.F = NaN;
    fit.aperiodic_intercept = NaN;
    fit.aperiodic_exponent = NaN;
    fit.aperiodic_knee_parameter = NaN;
    fit.aperiodic_knee_frequency_hz = NaN;
    fit.peak_frequency_hz = [];
    fit.peak_fwhm_hz = [];
    fit.peak_amplitude = [];
    fit.peak_probability = [];
end

function row = aperiodic_row(rec, fit)
    row = table(rec.participant, rec.condition, rec.group, rec.scope, rec.state_index, rec.state_label, rec.file_path, ...
        rec.n_files, rec.n_samples, rec.sfreq_mean, fit.status, string(fit.message), double(fit.F), ...
        double(fit.aperiodic_intercept), double(fit.aperiodic_exponent), ...
        double(fit.aperiodic_knee_parameter), double(fit.aperiodic_knee_frequency_hz), ...
        'VariableNames', {'participant', 'condition', 'group', 'scope', 'state_index', 'state_label', 'file_path', ...
        'n_files', 'n_samples', 'sfreq_mean', 'fit_status', 'fit_message', 'log_evidence', ...
        'aperiodic_intercept', 'aperiodic_exponent', 'aperiodic_knee_parameter', 'aperiodic_knee_frequency_hz'});
end

function T = periodic_rows(rec, fit, cfg)
    n_bands = size(cfg.peak_bands, 1);
    rows = cell(n_bands, 1);
    freq = pad_vector(fit.peak_frequency_hz, n_bands);
    fwhm = pad_vector(fit.peak_fwhm_hz, n_bands);
    ampl = pad_vector(fit.peak_amplitude, n_bands);
    prob = pad_vector(fit.peak_probability, n_bands);
    for i = 1:n_bands
        rows{i, 1} = table(rec.participant, rec.condition, rec.group, rec.scope, rec.state_index, rec.state_label, rec.file_path, ...
            string(cfg.peak_band_names{i}), double(cfg.peak_bands(i, 1)), double(cfg.peak_bands(i, 2)), ...
            fit.status, string(fit.message), double(fit.F), double(freq(i)), double(fwhm(i)), double(ampl(i)), double(prob(i)), ...
            'VariableNames', {'participant', 'condition', 'group', 'scope', 'state_index', 'state_label', 'file_path', ...
            'band_name', 'band_low_hz', 'band_high_hz', 'fit_status', 'fit_message', 'log_evidence', ...
            'peak_frequency_hz', 'peak_fwhm_hz', 'peak_amplitude', 'peak_probability'});
    end
    T = vertcat_nonempty(rows);
end

function row = fit_summary_row(rec, fit)
    row = table(rec.participant, rec.condition, rec.group, rec.scope, rec.state_index, rec.state_label, rec.file_path, ...
        rec.n_files, rec.n_samples, rec.sfreq_mean, fit.status, string(fit.message), double(fit.F), ...
        'VariableNames', {'participant', 'condition', 'group', 'scope', 'state_index', 'state_label', 'file_path', ...
        'n_files', 'n_samples', 'sfreq_mean', 'fit_status', 'fit_message', 'log_evidence'});
end

function cells = matrix_rows_to_cells(x)
    cells = cell(size(x, 1), 1);
    for i = 1:size(x, 1)
        cells{i} = x(i, :);
    end
end

function v = posterior_peak_probability(BSD, n)
    v = nan(n, 1);
    if isfield(BSD, 'Pp') && isfield(BSD.Pp, 'a')
        a = squeeze(BSD.Pp.a);
        v(1:min(n, numel(a))) = a(1:min(n, numel(a)));
    end
end

function [state_spectra, state_summary] = microstate_activation_spectra(manifest, cfg, util)
    S = load(cfg.microstate_results_mat);
    if isfield(S, 'HResults')
        H = S.HResults;
    elseif isfield(S, 'Results')
        H = S.Results;
    else
        error('No HResults or Results variable found in %s.', cfg.microstate_results_mat);
    end
    if ~isfield(H, 'participant_conditions') || ~isfield(H, 'common_labels')
        error('Microstate results must contain participant_conditions and common_labels.');
    end

    common_labels = cellstr(string(H.common_labels(:)));
    common_pos = [];
    if isfield(H, 'common_pos')
        common_pos = H.common_pos;
    end
    hcfg = struct();
    if isfield(H, 'cfg') && isstruct(H.cfg)
        hcfg = H.cfg;
    end

    state_spectra = {};
    rows = {};
    grouped = containers.Map('KeyType', 'char', 'ValueType', 'any');

    for i = 1:height(manifest)
        participant = manifest.participant(i);
        condition = manifest.condition(i);
        node = find_microstate_node(H.participant_conditions, participant, condition);
        if isempty(node) && isfield(H, 'participants')
            node = find_microstate_node(H.participants, participant, "");
        end
        if isempty(node)
            rows{end+1, 1} = skipped_state_summary_row(participant, condition, manifest.group(i), manifest.file_path(i), ...
                NaN, "", "missing_microstate_node"); %#ok<AGROW>
            continue;
        end

        try
            [raw_common, sfreq, labels] = load_common_label_data(manifest.file_path(i), common_labels, common_pos, hcfg, util);
            Sim = struct();
            Sim.X_noisy = raw_common;
            Sim.sfreq = sfreq;
            Sim.channel_labels = labels;
            Sim.pos = common_pos;
            Sim.preprocessing = backfit_preprocessing_from_cfg(hcfg);
            backfit = backfit_microstate_timecourse(Sim, node.fit);
            if ~isstruct(backfit) || ~isfield(backfit, 'ok') || ~backfit.ok
                error('Backfit failed: %s', backfit.message);
            end
            Xpsd = preprocess_spectrum_matrix(raw_common, sfreq, cfg, util);
            assignments = double(backfit.hard.assignments(:));
            K = size(node.fit.centers, 1);
            state_labels = state_labels_from_node(node.fit, K);
            for k = 1:K
                [power, n_active, n_segments, msg] = state_run_spectrum(Xpsd, sfreq, assignments == k, cfg);
                key = sprintf('%s|%s|%d|%s', char(participant), char(condition), k, state_labels{k});
                spec = get_or_empty_group(grouped, key, cfg.freqs);
                if n_segments > 0
                    spec.power_acc = spec.power_acc + power(:) * n_active;
                    spec.weight_acc = spec.weight_acc + n_active;
                end
                spec.n_files = spec.n_files + double(n_segments > 0);
                spec.n_samples = spec.n_samples + n_active;
                spec.sfreqs(end+1) = sfreq;
                spec.participant = string(participant);
                spec.condition = string(condition);
                spec.group = string(manifest.group(i));
                spec.state_index = k;
                spec.state_label = string(state_labels{k});
                spec.status = merge_status(spec.status, n_segments > 0);
                spec.message = append_message(spec.message, msg);
                grouped(key) = spec;
            end
        catch ME
            rows{end+1, 1} = skipped_state_summary_row(participant, condition, manifest.group(i), manifest.file_path(i), ...
                NaN, "", ME.message); %#ok<AGROW>
        end
    end

    keys = grouped.keys;
    for i = 1:numel(keys)
        spec = grouped(keys{i});
        if spec.weight_acc > 0
            power = spec.power_acc ./ spec.weight_acc;
        else
            power = nan(numel(cfg.freqs), 1);
            spec.status = "failed";
        end
        rec = make_spectrum_record(spec.participant, spec.condition, spec.group, spec.state_label, spec.state_index, "", ...
            "microstate_activation", cfg.freqs, power, spec.n_files, spec.n_samples, spec.sfreqs, spec.status, spec.message);
        state_spectra{end+1, 1} = rec; %#ok<AGROW>
        rows{end+1, 1} = spectrum_summary_row(rec); %#ok<AGROW>
    end
    state_summary = vertcat_nonempty(rows);
end

function [power, n_active, n_segments, message] = state_run_spectrum(X, sfreq, active, cfg)
    active = logical(active(:));
    n_active = sum(active);
    n_segments = 0;
    message = "";
    power_acc = zeros(numel(cfg.freqs), 1);
    weight_acc = 0;
    min_len = max(round(cfg.min_state_segment_s * sfreq), ceil(cfg.min_state_segment_cycles * sfreq / min(cfg.freqs)));
    min_len = max(min_len, 8);
    runs = active_run_bounds(active);
    for r = 1:size(runs, 1)
        len = runs(r, 2) - runs(r, 1) + 1;
        if len < min_len
            continue;
        end
        pxx = welch_mean_power(X(:, runs(r, 1):runs(r, 2)), sfreq, cfg.freqs, min(cfg.welch_window_s, len / sfreq), cfg.welch_overlap);
        power_acc = power_acc + pxx(:) * len;
        weight_acc = weight_acc + len;
        n_segments = n_segments + 1;
    end
    if weight_acc > 0
        power = power_acc ./ weight_acc;
    else
        power = nan(numel(cfg.freqs), 1);
        message = sprintf('No active run reached %d samples (%.3g s); lower min_state_segment_cycles or fit higher frequencies only.', min_len, min_len / sfreq);
    end
end

function spec = get_or_empty_group(map, key, freqs)
    if isKey(map, key)
        spec = map(key);
        return;
    end
    spec = struct();
    spec.power_acc = zeros(numel(freqs), 1);
    spec.weight_acc = 0;
    spec.n_files = 0;
    spec.n_samples = 0;
    spec.sfreqs = [];
    spec.participant = "";
    spec.condition = "";
    spec.group = "";
    spec.state_index = NaN;
    spec.state_label = "";
    spec.status = "failed";
    spec.message = "";
end

function status = merge_status(old_status, ok)
    if ok
        if old_status == "failed"
            status = "ok";
        else
            status = old_status;
        end
    elseif old_status == "ok"
        status = "partial";
    else
        status = old_status;
    end
end

function runs = active_run_bounds(active)
    d = diff([false; active(:); false]);
    starts = find(d == 1);
    stops = find(d == -1) - 1;
    runs = [starts stops];
end

function row = skipped_state_summary_row(participant, condition, group, file_path, state_index, state_label, message)
    row = table(string(participant), string(condition), string(group), "microstate_activation", double(state_index), string(state_label), string(file_path), ...
        0, 0, NaN, "failed", string(message), ...
        'VariableNames', {'participant', 'condition', 'group', 'scope', 'state_index', 'state_label', 'file_path', ...
        'n_files', 'n_samples', 'sfreq_mean', 'status', 'message'});
end

function labels = state_labels_from_node(Fit, K)
    labels = arrayfun(@(k) sprintf('state_%02d', k), 1:K, 'UniformOutput', false);
    if isfield(Fit, 'template_alignment') && isstruct(Fit.template_alignment) && ...
            isfield(Fit.template_alignment, 'labels') && numel(Fit.template_alignment.labels) >= K
        labels = cellstr(string(Fit.template_alignment.labels(:)));
    end
end

function node = find_microstate_node(nodes, participant, condition)
    node = [];
    for i = 1:numel(nodes)
        p_ok = strlength(participant) == 0 || string(nodes(i).participant) == string(participant);
        c_ok = strlength(condition) == 0 || string(nodes(i).condition) == string(condition);
        if p_ok && c_ok
            node = nodes(i);
            return;
        end
    end
end

function [data, sfreq, labels] = load_common_label_data(file_path, common_labels, common_pos, cfg, util)
    [data0, sfreq, ~, labels0, pos0] = load_eeg_file_local(file_path, util);
    can0 = util.canonical_channel_labels(labels0);
    common_can = util.canonical_channel_labels(common_labels);
    data = nan(numel(common_can), size(data0, 2));
    idx = nan(numel(common_can), 1);
    for c = 1:numel(common_can)
        local = find(strcmp(can0, common_can{c}), 1, 'first');
        if ~isempty(local)
            idx(c) = local;
            data(c, :) = double(data0(local, :));
        end
    end
    missing = find(~isfinite(idx));
    interpolate = isfield(cfg, 'interpolate_missing_channels') && cfg.interpolate_missing_channels;
    if ~isempty(missing)
        if ~interpolate
            error('Missing channel %s in %s.', common_labels{missing(1)}, file_path);
        end
        data = fill_missing_channels_local(data, missing, pos0, idx, common_pos, file_path);
    end
    good_t = all(isfinite(data), 1);
    data = data(:, good_t);
    labels = common_labels(:)';
end

function data = fill_missing_channels_local(data, missing, source_pos, direct_idx, target_pos, file_path)
    if isempty(source_pos) || isempty(target_pos)
        error('Cannot interpolate missing channels in %s without channel positions.', file_path);
    end
    available = find(isfinite(direct_idx));
    source_xyz = source_pos(direct_idx(available), :);
    for i = 1:numel(missing)
        t = missing(i);
        d = sqrt(sum((source_xyz - target_pos(t, :)).^2, 2));
        [ds, ord] = sort(d, 'ascend');
        m = min(6, numel(ord));
        ord = ord(1:m);
        ds = ds(1:m);
        w = 1 ./ (ds .^ 2 + eps);
        w = w ./ sum(w);
        data(t, :) = w' * data(available(ord), :);
    end
end

function preprocessing = backfit_preprocessing_from_cfg(cfg)
    preprocessing = struct();
    fields = {'apply_average_reference', 'filter_band', 'gfp_peak_min_distance', 'gfp_peak_threshold_schedule'};
    for i = 1:numel(fields)
        if isfield(cfg, fields{i})
            preprocessing.(fields{i}) = cfg.(fields{i});
        end
    end
end

function manifest = read_or_build_manifest_local(input_path)
    input_path = char(input_path);
    if isfolder(input_path)
        files = [dir(fullfile(input_path, '**', '*.set')); dir(fullfile(input_path, '**', '*.mat'))];
        paths = string(fullfile({files.folder}', {files.name}'));
        skip = contains(lower(paths), [filesep 'outputs' filesep]) | contains(lower(paths), [filesep 'old' filesep]);
        paths = paths(~skip);
        if isempty(paths)
            error('No .set or .mat files found under %s.', input_path);
        end
        [paths, ~] = sort(paths);
        participant = strings(numel(paths), 1);
        condition = strings(numel(paths), 1);
        group = repmat("all", numel(paths), 1);
        for i = 1:numel(paths)
            participant(i) = infer_participant(paths(i), i);
            condition(i) = infer_condition(paths(i));
        end
        file_path = paths(:);
    elseif isfile(input_path)
        [~, ~, manifest_ext] = fileparts(input_path);
        if strcmpi(manifest_ext, '.csv')
            opts = detectImportOptions(input_path, 'FileType', 'text', 'TextType', 'string', 'Delimiter', ',');
        else
            opts = detectImportOptions(input_path, 'FileType', 'text', 'TextType', 'string');
        end
        opts.VariableNamingRule = 'preserve';
        T = readtable(input_path, opts);
        names = normalise_names(T.Properties.VariableNames);
        file_col = find_first(names, ["filepath", "file_path", "file", "path", "filename"]);
        if isempty(file_col)
            error('Manifest must contain a file_path column.');
        end
        file_path = string(T{:, file_col});
        base_dir = fileparts(input_path);
        for i = 1:numel(file_path)
            if ~is_absolute_path(file_path(i))
                file_path(i) = string(fullfile(base_dir, file_path(i)));
            end
        end
        p_col = find_first(names, ["participant", "subject", "sub", "id", "subjectid", "participantid"]);
        c_col = find_first(names, ["condition", "state", "task", "eyes", "session"]);
        g_col = find_first(names, ["group", "diagnosis", "cohort", "class"]);
        participant = column_or_infer(T, p_col, file_path, "participant");
        condition = column_or_default(T, c_col, "condition", numel(file_path));
        group = column_or_default(T, g_col, "all", numel(file_path));
        [file_path, ord] = sort(file_path);
        participant = participant(ord);
        condition = condition(ord);
        group = group(ord);
    else
        error('Input path not found: %s', input_path);
    end
    manifest = table(participant(:), condition(:), group(:), file_path(:), ...
        'VariableNames', {'participant', 'condition', 'group', 'file_path'});
end

function [data, sfreq, chanlocs, labels, pos] = load_eeg_file_local(file_path, util)
    file_path = char(file_path);
    if ~isfile(file_path)
        error('EEG file not found: %s', file_path);
    end
    [~, ~, ext] = fileparts(file_path);
    ext = lower(ext);
    chanlocs = [];
    sfreq = 250;
    switch ext
        case '.set'
            if exist('pop_loadset', 'file') ~= 2
                error('EEGLAB pop_loadset not found on MATLAB path.');
            end
            EEG = pop_loadset(file_path);
            data = double(EEG.data);
            sfreq = double(EEG.srate);
            chanlocs = EEG.chanlocs;
        case '.mat'
            S = load(file_path);
            if isfield(S, 'EEG')
                data = double(S.EEG.data);
                if isfield(S.EEG, 'srate'), sfreq = double(S.EEG.srate); end
                if isfield(S.EEG, 'chanlocs'), chanlocs = S.EEG.chanlocs; end
            elseif isfield(S, 'eeg_data')
                data = double(S.eeg_data);
            elseif isfield(S, 'data')
                data = double(S.data);
            else
                error('No EEG data field found in %s.', file_path);
            end
            if isfield(S, 'sfreq'), sfreq = double(S.sfreq); end
            if isfield(S, 'srate'), sfreq = double(S.srate); end
            if isfield(S, 'chanlocs'), chanlocs = S.chanlocs; end
        otherwise
            error('Unsupported EEG extension: %s', ext);
    end
    data = squeeze(data);
    if ~ismatrix(data)
        data = reshape(data, size(data, 1), []);
    end
    if size(data, 1) > size(data, 2) && size(data, 2) < 128
        data = data';
    end
    labels = util.channel_labels_from_chanlocs(chanlocs, size(data, 1));
    pos = util.positions_from_chanlocs(chanlocs, size(data, 1));
end

function participant = column_or_infer(T, col, file_path, fallback)
    if isempty(col)
        participant = strings(numel(file_path), 1);
        for i = 1:numel(file_path)
            participant(i) = infer_participant(file_path(i), i);
        end
        return;
    end
    participant = string(T{:, col});
    bad = ismissing(participant) | strlength(strtrim(participant)) == 0;
    for i = find(bad(:))'
        participant(i) = infer_participant(file_path(i), i);
    end
    participant(strlength(strtrim(participant)) == 0) = fallback;
end

function values = column_or_default(T, col, default_value, n)
    if isempty(col)
        values = repmat(string(default_value), n, 1);
    else
        values = string(T{:, col});
        values(ismissing(values) | strlength(strtrim(values)) == 0) = string(default_value);
    end
end

function names = normalise_names(names_in)
    names = lower(regexprep(string(names_in), '[^a-zA-Z0-9]', ''));
end

function idx = find_first(names, choices)
    idx = [];
    choices = lower(regexprep(string(choices), '[^a-zA-Z0-9]', ''));
    for i = 1:numel(choices)
        hit = find(names == choices(i), 1, 'first');
        if ~isempty(hit)
            idx = hit;
            return;
        end
    end
end

function tf = is_absolute_path(pth)
    pth = char(pth);
    tf = startsWith(pth, filesep) || ~isempty(regexp(pth, '^[A-Za-z]:[\\/]', 'once'));
end

function pid = infer_participant(file_path, row_index)
    [parent, stem] = fileparts(char(file_path));
    hit = regexp(stem, '(sub-[A-Za-z0-9]+)', 'match', 'once');
    if isempty(hit)
        [~, hit] = fileparts(parent);
    end
    if isempty(hit)
        hit = sprintf('row%03d', row_index);
    end
    pid = string(hit);
end

function condition = infer_condition(file_path)
    [~, stem] = fileparts(char(file_path));
    s = upper(stem);
    if contains(s, '_EC') || contains(s, 'EYESCLOSED') || contains(s, 'CLOSED')
        condition = "EC";
    elseif contains(s, '_EO') || contains(s, 'EYESOPEN') || contains(s, 'OPEN')
        condition = "EO";
    else
        condition = "condition";
    end
end

function y = first_string(x)
    x = string(x);
    if isempty(x)
        y = "";
    else
        y = x(find(strlength(x) > 0, 1, 'first'));
        if isempty(y), y = ""; end
    end
end

function T = vertcat_nonempty(rows)
    rows = rows(~cellfun(@isempty, rows));
    if isempty(rows)
        T = table();
    else
        T = vertcat(rows{:});
    end
end

function x = mean_or_nan(v)
    v = double(v(:));
    v = v(isfinite(v));
    if isempty(v)
        x = NaN;
    else
        x = mean(v);
    end
end

function msg = append_message(msg, extra)
    extra = string(extra);
    if strlength(extra) == 0
        return;
    end
    if strlength(msg) == 0
        msg = extra;
    else
        msg = msg + " | " + extra;
    end
end

function v = vector_or_nan(S, field, n)
    v = nan(n, 1);
    if isstruct(S) && isfield(S, field)
        x = double(S.(field));
        x = x(:);
        v(1:min(n, numel(x))) = x(1:min(n, numel(x)));
    end
end

function x = scalar_field_or_nan(S, field)
    if isstruct(S) && isfield(S, field) && ~isempty(S.(field))
        value = S.(field);
        x = double(value(1));
    else
        x = NaN;
    end
end

function v = pad_vector(v, n)
    x = nan(n, 1);
    v = double(v(:));
    x(1:min(n, numel(v))) = v(1:min(n, numel(v)));
    v = x;
end

function key = clean_key(x)
    key = char(regexprep(string(x), '[^A-Za-z0-9._-]+', '_'));
    if isempty(key)
        key = 'item';
    end
end
