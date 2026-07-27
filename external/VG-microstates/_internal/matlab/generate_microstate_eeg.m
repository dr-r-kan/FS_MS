function [Sim, maps_true, pos] = generate_microstate_eeg(K_true, snr_db, duration_s, sfreq, seed, overlap_opts)
    % ...existing code...
% GENERATE_MICROSTATE_EEG: Generate synthetic EEG with REALISTIC amplitudes
%
% Microstate amplitudes: 10-50 µV (typical values from literature)
% Pink noise baseline: 5-20 µV 
% GFP range: 0.5-3 µV
% Optional temporal overlap between adjacent microstates
%
% NOTE: Generates 71-channel EEG to match artifact template montage
% Uses channel positions from template file

    if nargin < 4, sfreq = 250; end
    if nargin < 5, seed = 42; end
    if nargin < 6, overlap_opts = struct(); end
    
    overlap_defaults = struct( ...
        'prob', 0, ...                % Probability a boundary is overlapped
        'ms_range', [10 40], ...      % Min/max overlap duration in ms
        'strength', 0.5, ...          % Weight of the secondary state during overlap
        'clean_sanity_profile', false, ...
        'ecological_profile', true, ...
        'real_background_file', '', ...
        'map_jitter_fraction', 0.15, ...
        'inject_artifacts', true, ...
        'microstate_amplitude_uv', 30, ...
        'background_noise_rms_uv', 10, ...
        'mean_dur_ms', 80, ...
        'gfp_envelope_eta', 0.2, ...
        'template_pool_K', K_true, ...
        'template_indices', []);
    overlap_opts = fill_overlap_defaults(overlap_opts, overlap_defaults);
    overlap_opts = apply_clean_sanity_profile_defaults(overlap_opts);
    
    rng(seed, 'twister');
    
    % Load the canonical template maps and channel geometry from the same
    % shared loader used by the template sense-check plotting code.
    fprintf('Loading channel positions from template...\n');
    [maps_true_normalized, template_labels, pos, chanlocs, channel_labels] = ...
        load_true_templates_from_template(K_true, overlap_opts.template_pool_K, overlap_opts.template_indices);
    
    if isempty(pos)
        error('Failed to load channel positions from template file');
    end
    
    C = size(pos, 1);
    fprintf('✓ Loaded %d channels from template\n', C);
    fprintf('✓ Channel labels: %s\n', sprintf('%s ', channel_labels{:}));
    fprintf('✓ Loaded %d canonical template maps: %s\n', K_true, strjoin(template_labels, ', '));
    
    % Scale the template maps into a realistic microvolt range.
    microstate_amplitude_uv = double(overlap_opts.microstate_amplitude_uv);
    maps_true = maps_true_normalized * microstate_amplitude_uv;  % Scale to µV
    len_scale = 0.25;
    W = rbf_kernel(pos, len_scale);
    
    % Temporal parameters
    T = round(duration_s * sfreq);
    mean_dur_ms = double(overlap_opts.mean_dur_ms);
    mean_dur_samples = max(1, round((mean_dur_ms/1000)*sfreq));
    
    % Generate clean EEG (store segments for optional overlap)
    X_clean_base = zeros(C, T);
    z = zeros(1, T);  % Ground-truth dominant labels per sample
    state_weights_base = zeros(K_true, T);
    segments = struct('state', {}, 'start', {}, 'len', {}, 'template', {}, 'gfp', {});
    
    p_switch = 1 / mean_dur_samples;
    cur_state = randi(K_true);
    t = 1;
    
    while t <= T
        dwell = min(T - t + 1, geornd(p_switch) + 1);
        
        % Add spatial jitter to maps (~15% of amplitude)
        map_jitter_sd = max(0, double(overlap_opts.map_jitter_fraction)) * microstate_amplitude_uv;
        jitter = map_jitter_sd * (W * randn(C, 1));
        template = maps_true(cur_state, :)' + jitter;
        
        if overlap_opts.ecological_profile
            gfp_env = ecological_gfp_packet(dwell, sfreq);
        elseif overlap_opts.clean_sanity_profile
            gfp_env = clean_profile_state_trace(dwell, sfreq);
        else
            % GFP envelope: realistic range 0.3-0.8 (modulates 30-80% of peak)
            gfp_env = ar1_positive_realistic(dwell, 0.98, double(overlap_opts.gfp_envelope_eta));  % 0.3-0.8 range
        end
        
        seg_idx = t:(t + dwell - 1);
        segments(end+1) = struct(...
            'state', cur_state, ...
            'start', t, ...
            'len', dwell, ...
            'template', template, ...
            'gfp', gfp_env); %#ok<AGROW>
        
        z(seg_idx) = cur_state;
        state_weights_base(cur_state, seg_idx) = 1;
        
        % Pick next state
        if K_true > 1
            other_states = setdiff(1:K_true, cur_state);
            cur_state = other_states(randi(length(other_states)));
        end
        
        t = t + dwell;
    end
    
    % Reconstruct clean EEG without overlap
    for s = 1:numel(segments)
        seg = segments(s);
        seg_idx = seg.start:min(T, seg.start + seg.len - 1);
        seg_gfp = seg.gfp(1:numel(seg_idx));
        X_clean_base(:, seg_idx) = X_clean_base(:, seg_idx) + seg.template * seg_gfp;
    end
    
    rng_state = rng;  % isolate overlap randomness from downstream noise generation
    [X_clean, state_weights_true, overlap_events] = apply_microstate_overlap( ...
        X_clean_base, state_weights_base, segments, overlap_opts, sfreq, T);
    rng(rng_state);
    [~, z] = max(state_weights_true, [], 1);
    
    % Generate PINK noise with REALISTIC AMPLITUDE
    % Background EEG: 5-20 µV RMS (let's use 10 µV as baseline)
    background_noise_rms_uv = double(overlap_opts.background_noise_rms_uv);
    
    if overlap_opts.ecological_profile
        [N, background_source] = ecological_background_noise(C, T, pos, sfreq, background_noise_rms_uv, overlap_opts);
    else
        fprintf('Generating pink noise (%.1f µV RMS)...\n', background_noise_rms_uv);
        N = zeros(C, T);
        for c = 1:C
            N(c, :) = generate_pink_noise(T) * background_noise_rms_uv;
        end
        
        % Spatially correlate noise through RBF kernel
        for t = 1:T
            N(:, t) = W * (randn(C, 1) .* N(:, t) / (norm(N(:, t)) + eps));
        end
        
        % Normalize to target noise level
        N = N / (std(N(:)) + eps) * background_noise_rms_uv;
        background_source = 'synthetic_pink_rbf';
    end
    
    % Scale by SNR
    Ps = mean(X_clean(:).^2);
    Pn = mean(N(:).^2) + eps;
    
    % SNR_dB = 10 * log10(Ps / Pn) => Scale noise appropriately
    scale = sqrt((10^(-snr_db/10) * Ps) / Pn);
    X_noisy = X_clean + scale * N;
    
    % Inject real artifacts from template file
    if overlap_opts.inject_artifacts
        fprintf('Injecting real artifacts...\n');
        try
            X_noisy = inject_real_artifacts(X_noisy, sfreq);
        catch ME
            fprintf('⚠ Warning: Could not load artifacts (%s). Continuing without real artifacts.\n', ME.message);
        end
    else
        fprintf('Skipping real artifact injection for clean sanity profile.\n');
    end
    
    % Return structure with amplitude info
    Sim = struct( ...
        'X_clean', X_clean, ...
        'X_noisy', X_noisy, ...
        'clean_gfp', sqrt(mean((X_clean - mean(X_clean, 1)) .^ 2, 1)), ...
        'maps_true', maps_true, ...  % In µV
        'z_true', z, ...
        'state_weights_true', state_weights_true, ...
        'segment_state', [segments.state], ...
        'segment_start', [segments.start], ...
        'segment_len', [segments.len], ...
        'sfreq', sfreq, ...
        'pos', pos, ...
        'chanlocs', chanlocs, ...  % Channel location structure
        'channel_labels', {channel_labels}, ...  % ✅ NEW: Channel labels as cell array
        'true_template_labels', {template_labels}, ...
        'true_template_pool_K', overlap_opts.template_pool_K, ...
        'true_template_indices', overlap_opts.template_indices, ...
        'K_true', K_true, ...
        'SNR_dB', snr_db, ...
        'duration_s', duration_s, ...
        'microstate_amplitude_uv', microstate_amplitude_uv, ...
        'background_noise_rms_uv', background_noise_rms_uv, ...
        'map_jitter_fraction', overlap_opts.map_jitter_fraction, ...
        'inject_artifacts', logical(overlap_opts.inject_artifacts), ...
        'clean_sanity_profile', logical(overlap_opts.clean_sanity_profile), ...
        'ecological_profile', logical(overlap_opts.ecological_profile), ...
        'background_source', background_source, ...
        'overlap_prob', overlap_opts.prob, ...
        'overlap_ms_range', overlap_opts.ms_range, ...
        'overlap_strength', overlap_opts.strength, ...
        'overlap_events', overlap_events, ...
        'n_overlap_events', numel(overlap_events));

end

function opts = fill_overlap_defaults(opts, defaults)
    if ~isstruct(opts)
        opts = struct();
    end
    fields = fieldnames(defaults);
    for i = 1:numel(fields)
        f = fields{i};
        if ~isfield(opts, f) || isempty(opts.(f))
            opts.(f) = defaults.(f);
        end
    end
    if numel(opts.ms_range) == 1
        opts.ms_range = [opts.ms_range opts.ms_range];
    end
end

function [X_out, weights_out, events] = apply_microstate_overlap(X_base, weights_base, segments, overlap_opts, sfreq, T)
% APPLY_MICROSTATE_OVERLAP: inject partial overlap around segment boundaries
%
% Uses cross-fade style mixing so adjacent states co-activate without
% exploding amplitude. The base signal remains untouched when prob = 0.

    X_out = X_base;
    weights_out = weights_base;
    events = struct('boundary_sample', {}, 'overlap_len', {}, 'strength', {});
    
    if overlap_opts.prob <= 0 || numel(segments) < 2
        return;
    end
    
    min_len = max(1, round(overlap_opts.ms_range(1) / 1000 * sfreq));
    max_len = max(min_len, round(overlap_opts.ms_range(2) / 1000 * sfreq));
    strength = max(0, min(1, overlap_opts.strength));
    
    for s = 1:(numel(segments) - 1)
        if rand() > overlap_opts.prob
            continue;
        end
        
        seg_a = segments(s);
        seg_b = segments(s + 1);
        boundary = seg_a.start + seg_a.len - 1;
        
        overlap_len = min([max_len, seg_a.len, seg_b.len, T - boundary]);
        if overlap_len < min_len
            continue;
        end
        
        % Fade-in of next state on the tail of the current state
        pre_idx = (boundary - overlap_len + 1):boundary;
        w_forward = linspace(0, strength, numel(pre_idx));
        for i = 1:numel(pre_idx)
            idx = pre_idx(i);
            b_idx = min(i, numel(seg_b.gfp));
            bleed = seg_b.template * seg_b.gfp(b_idx);
            X_out(:, idx) = (1 - w_forward(i)) * X_out(:, idx) + w_forward(i) * bleed;
            next_weight = zeros(size(weights_out, 1), 1);
            next_weight(seg_b.state) = 1;
            weights_out(:, idx) = (1 - w_forward(i)) * weights_out(:, idx) + w_forward(i) * next_weight;
        end
        
        % Fade-out of previous state into start of next state
        post_idx = seg_b.start:min(T, seg_b.start + overlap_len - 1);
        w_backward = linspace(strength, 0, numel(post_idx));
        for i = 1:numel(post_idx)
            idx = post_idx(i);
            a_idx = max(1, seg_a.len - overlap_len + i);
            bleed = seg_a.template * seg_a.gfp(a_idx);
            X_out(:, idx) = (1 - w_backward(i)) * X_out(:, idx) + w_backward(i) * bleed;
            prev_weight = zeros(size(weights_out, 1), 1);
            prev_weight(seg_a.state) = 1;
            weights_out(:, idx) = (1 - w_backward(i)) * weights_out(:, idx) + w_backward(i) * prev_weight;
        end
        
        events(end+1) = struct(...
            'boundary_sample', boundary, ...
            'overlap_len', overlap_len, ...
            'strength', strength); %#ok<AGROW>
    end

    weights_out = max(weights_out, 0);
    weights_out = weights_out ./ max(sum(weights_out, 1), eps);
end

function opts = apply_clean_sanity_profile_defaults(opts)
    if ~isfield(opts, 'clean_sanity_profile') || ~opts.clean_sanity_profile
        return;
    end

    opts.map_jitter_fraction = 0;
    opts.inject_artifacts = false;
    opts.gfp_envelope_eta = min(double(opts.gfp_envelope_eta), 0.05);
    opts.mean_dur_ms = max(double(opts.mean_dur_ms), 150);
end

function x = clean_profile_state_trace(T, sfreq)
    pad = max(16, round(0.25 * T));
    raw = randn(1, T + 2 * pad);
    x = fft_bandpass_local(raw, sfreq, [8 20]);
    x = x((pad + 1):(pad + T));
    t = (0:(T - 1)) / sfreq;
    f0 = 12 + 4 * rand();
    x = x + 0.75 * sin(2 * pi * f0 * t + 2 * pi * rand());
    x = x - mean(x);
    s = std(x);
    if s < eps
        x = sin(2 * pi * 10 * (0:(T - 1)) / sfreq);
        s = std(x);
    end
    x = 0.6 * (x / max(s, eps));
end

function x = ecological_gfp_packet(T, sfreq)
%ECOLOGICAL_GFP_PACKET Smooth GFP bursts with valleys at state boundaries.

    if T <= 0
        x = [];
        return;
    end
    t = 1:T;
    dwell_s = T / max(sfreq, eps);
    n_peaks = max(1, 1 + poissrnd(max(0, 3 * dwell_s - 0.4)));
    x = zeros(1, T);
    lo = max(1, round(0.15 * T));
    hi = max(lo, min(T, round(0.85 * T)));
    for p = 1:n_peaks
        center = lo + rand() * max(0, hi - lo);
        width = max(1.5, (0.018 + 0.025 * rand()) * sfreq);
        amp = exp(0.25 * randn());
        x = x + amp * exp(-0.5 * ((t - center) / width) .^ 2);
    end
    taper = sin(pi * t / (T + 1)) .^ 0.8;
    alpha = 1 + 0.12 * sin(2 * pi * (8 + 4 * rand()) * (t / sfreq) + 2 * pi * rand());
    x = x .* taper .* max(alpha, 0.25);
    if max(abs(x)) > eps
        x = x / max(abs(x));
    end
    x = (0.08 + 0.95 * x) * (0.7 + 0.6 * rand());
    if rand() < 0.5
        x = -x;
    end
end

function [N, source] = ecological_background_noise(C, T, pos, sfreq, target_rms, opts)
    bg_file = '';
    if isfield(opts, 'real_background_file') && ~isempty(opts.real_background_file)
        bg_file = char(string(opts.real_background_file));
    end
    if ~isempty(bg_file) && isfile(bg_file)
        try
            N = real_background_snippet(bg_file, C, T, sfreq);
            N = N / (std(N(:)) + eps) * target_rms;
            source = bg_file;
            fprintf('Using real EEG background snippet: %s\n', bg_file);
            return;
        catch ME
            fprintf('⚠ Real background failed (%s). Falling back to ecological synthetic noise.\n', ME.message);
        end
    end

    fprintf('Generating ecological synthetic background (%.1f µV RMS)...\n', target_rms);
    latent = zeros(C, T);
    for c = 1:C
        pink = generate_pink_noise(T);
        alpha = fft_bandpass_local(randn(1, T + 256), sfreq, [8 13]);
        alpha = alpha(129:(128 + T));
        slow = fft_bandpass_local(randn(1, T + 256), sfreq, [1 4]);
        slow = slow(129:(128 + T));
        latent(c, :) = 0.75 * pink + 0.35 * alpha / (std(alpha) + eps) + 0.15 * slow / (std(slow) + eps);
    end

    D = pdist2(pos, pos);
    R = exp(-(D .^ 2) / (2 * 0.35 ^ 2)) + 0.03 * eye(C);
    R = (R + R') / 2;
    [V, S] = eig(R);
    mixer = V * diag(sqrt(max(diag(S), 0))) * V';
    N = mixer * latent;
    N = N - mean(N, 1);
    N = N / (std(N(:)) + eps) * target_rms;
    source = 'synthetic_1f_alpha_spatial';
end

function X = real_background_snippet(file_path, C, T, sfreq)
    if exist('pop_loadset', 'file') ~= 2
        error('EEGLAB pop_loadset is not on the MATLAB path.');
    end
    EEG = pop_loadset('filename', file_path);
    X = double(EEG.data);
    if ndims(X) > 2
        X = reshape(X, size(X, 1), []);
    end
    if isfield(EEG, 'srate') && ~isempty(EEG.srate) && double(EEG.srate) ~= sfreq
        X = resample_eeg_data(X, double(EEG.srate), sfreq);
    end
    if size(X, 1) ~= C
        X = interpolate_channels(X, C);
    end
    if size(X, 2) < T
        X = repmat(X, 1, ceil(T / size(X, 2)));
    end
    start_idx = randi(size(X, 2) - T + 1);
    X = X(:, start_idx:(start_idx + T - 1));
    X = X - mean(X, 1);
end

function Xf = fft_bandpass_local(X, sfreq, bp)
    X = double(X);
    F = fft(X, [], 2);
    n = size(X, 2);
    freqs = (0:(n - 1)) * (sfreq / n);
    mask = (freqs >= bp(1) & freqs <= bp(2)) | ...
        (freqs >= sfreq - bp(2) & freqs <= sfreq - bp(1));
    F(:, ~mask) = 0;
    Xf = real(ifft(F, [], 2));
end

% ======================== CHANNEL MONTAGE ========================

function [template_maps, template_labels, pos, chanlocs, channel_labels] = load_true_templates_from_template(K_true, template_pool_K, template_indices)
% LOAD_TRUE_TEMPLATES_FROM_TEMPLATE: Load canonical template maps and geometry
% through the shared MetaMaps loader.

    if nargin < 2 || isempty(template_pool_K)
        template_pool_K = K_true;
    end
    if nargin < 3
        template_indices = [];
    end

    template_maps = [];
    template_labels = {};
    pos = [];
    chanlocs = [];
    channel_labels = {};
    
    try
        util = microstate_utilities();
        cfg = util.load_config();
        template_file = char(cfg.paths.template_file);
        if ~isfile(template_file)
            template_file = find_template_file();
        end
        
        if isempty(template_file)
            fprintf('Error: Template SET file not found\n');
            return;
        end
        
        fprintf('Loading channel positions from: %s\n', template_file);
        
        if ~exist('pop_loadset', 'file')
            fprintf('Error: EEGLAB not found\n');
            return;
        end

        [template_maps, template_labels, channel_labels, chanlocs] = ...
            load_metamaps_templates(template_file, 'K', template_pool_K);
        if ~isempty(template_indices)
            template_indices = double(template_indices(:)');
            if numel(template_indices) ~= K_true || any(template_indices < 1) || any(template_indices > numel(template_labels))
                error('template_indices must contain %d valid indices from the K=%d canonical template set.', K_true, template_pool_K);
            end
            template_maps = template_maps(template_indices, :);
            template_labels = template_labels(template_indices);
        elseif size(template_maps, 1) ~= K_true
            template_maps = template_maps(1:K_true, :);
            template_labels = template_labels(1:K_true);
        end
        n_channels = size(template_maps, 2);
        if n_channels == 0
            fprintf('Error: Template has no usable loaded channels\n');
            return;
        end
        if numel(chanlocs) ~= n_channels
            fprintf('Error: Template map/channel geometry mismatch (%d maps vs %d chanlocs)\n', ...
                n_channels, numel(chanlocs));
            template_maps = [];
            template_labels = {};
            pos = [];
            chanlocs = [];
            channel_labels = {};
            return;
        end

        pos = util.positions_from_chanlocs(chanlocs, n_channels);
        valid_coords = sum(all(isfinite(pos), 2));
        if valid_coords ~= n_channels
            fprintf('Error: Template loader returned %d/%d valid coordinates\n', valid_coords, n_channels);
            template_maps = [];
            template_labels = {};
            pos = [];
            chanlocs = [];
            channel_labels = {};
            return;
        end

        fprintf('✓ Extracted %d channel labels\n', length(channel_labels));
        fprintf('✓ Loaded %d channels with valid 3D coordinates\n', valid_coords);
        
    catch ME
        fprintf('Error loading template: %s\n', ME.message);
        template_maps = [];
        template_labels = {};
        pos = [];
        chanlocs = [];
        channel_labels = {};
    end
end

function template_file = find_template_file()
% FIND_TEMPLATE_FILE: Locate the template SET file

    search_paths = {
        'E:\EEGs\SPM_MS\MetaMaps_2023_06.set', ...
        fullfile(pwd, 'MetaMaps_2023_06.set'), ...
        fullfile(pwd, '..', 'MetaMaps_2023_06.set'), ...
        fullfile(pwd, 'data', 'MetaMaps_2023_06.set'), ...
        fullfile(pwd, '..', 'data', 'MetaMaps_2023_06.set')
    };
    
    template_file = '';
    for p = 1:length(search_paths)
        if isfile(search_paths{p})
            template_file = search_paths{p};
            return;
        end
    end
end

% ======================== ARTIFACT INJECTION ========================

function X_out = inject_real_artifacts(X, sfreq_target)
% INJECT_REAL_ARTIFACTS: Load and inject real EEG artifacts from EEGLAB SET file

    artifact_file = find_artifact_template();
    
    if isempty(artifact_file)
        error('artefact_template.set not found');
    end
    
    fprintf('  Loading artifact template from: %s\n', artifact_file);
    
    if ~exist('pop_loadset', 'file')
        error('EEGLAB not found. Please add EEGLAB to path.');
    end
    
    try
        EEG = pop_loadset('filename', artifact_file);
        
        if isempty(EEG.data)
            error('No data in SET file');
        end
        
        X_artifacts = EEG.data;  % Channels x Timepoints
        sfreq_artifacts = EEG.srate;
        n_channels_artifact = size(X_artifacts, 1);
        n_channels_target = size(X, 1);
        
        fprintf('  ✓ Loaded %d channels, %.1f seconds at %.0f Hz\n', ...
            n_channels_artifact, size(X_artifacts, 2)/sfreq_artifacts, sfreq_artifacts);
        
        % Verify channel match
        if n_channels_artifact ~= n_channels_target
            fprintf('  ⚠ Channel count mismatch: artifact has %d, target has %d\n', ...
                n_channels_artifact, n_channels_target);
        end
        
    catch ME
        error('Failed to load SET file: %s', ME.message);
    end
    
    % Resample artifacts if needed
    if sfreq_artifacts ~= sfreq_target
        fprintf('  Resampling artifacts from %.0f to %.0f Hz\n', sfreq_artifacts, sfreq_target);
        X_artifacts = resample_eeg_data(X_artifacts, sfreq_artifacts, sfreq_target);
    end
    
    % Inject random artifact snippets with REALISTIC AMPLITUDE
    X_out = X;
    n_artifacts = max(1, floor(size(X, 2) / (10 * sfreq_target)));
    snippet_duration_s = 0.5;
    snippet_len = round(snippet_duration_s * sfreq_target);
    
    fprintf('  Injecting %d artifact snippets (~%.1f s each)...\n', n_artifacts, snippet_duration_s);
    
    for art_idx = 1:n_artifacts
        max_start = max(1, size(X_artifacts, 2) - snippet_len);
        art_start = randi([1, max_start]);
        art_end = min(art_start + snippet_len - 1, size(X_artifacts, 2));
        
        max_inj = max(1, size(X_out, 2) - (art_end - art_start + 1));
        inj_start = randi([1, max_inj]);
        inj_end = inj_start + (art_end - art_start);
        
        artifact_snippet = X_artifacts(:, art_start:art_end);
        
        % Only interpolate if channel counts don't match
        if size(artifact_snippet, 1) ~= size(X_out, 1)
            fprintf('    Interpolating artifact from %d to %d channels\n', ...
                size(artifact_snippet, 1), size(X_out, 1));
            artifact_snippet = interpolate_channels(artifact_snippet, size(X_out, 1));
        end
        
        % Scale artifact to be 50-100% of signal amplitude (realistic contamination)
        artifact_rms = sqrt(mean(artifact_snippet(:).^2));
        output_std = std(X_out(:));
        artifact_scale = (0.5 + 0.5*rand()) * output_std / (artifact_rms + eps);
        artifact_snippet = artifact_scale * artifact_snippet;
        
        if inj_end <= size(X_out, 2)
            fade_len = min(10, floor((inj_end - inj_start) / 10));
            fade_in = linspace(0, 1, fade_len);
            fade_out = linspace(1, 0, fade_len);
            
            X_out(:, inj_start:inj_start+fade_len-1) = ...
                X_out(:, inj_start:inj_start+fade_len-1) + ...
                artifact_snippet(:, 1:fade_len) .* fade_in;
            
            mid_start = inj_start + fade_len;
            mid_end = inj_end - fade_len;
            if mid_start <= mid_end
                X_out(:, mid_start:mid_end) = ...
                    X_out(:, mid_start:mid_end) + ...
                    artifact_snippet(:, fade_len+1:end-fade_len);
            end
            
            X_out(:, inj_end-fade_len+1:inj_end) = ...
                X_out(:, inj_end-fade_len+1:inj_end) + ...
                artifact_snippet(:, end-fade_len+1:end) .* fade_out;
        end
    end
    
    fprintf('  ✓ Artifacts injected successfully\n');
end

function artifact_file = find_artifact_template()
    search_paths = {
        fullfile(pwd, 'artefact_template.set'), ...
        fullfile(pwd, '..', 'artefact_template.set'), ...
        fullfile(pwd, 'data', 'artefact_template.set'), ...
        fullfile(pwd, '..', 'data', 'artefact_template.set'), ...
        'E:\EEGs\SPM_MS\artefact_template.set', ...
        'D:\EEGData\artefact_template.set'
    };
    
    artifact_file = '';
    for p = 1:length(search_paths)
        if isfile(search_paths{p})
            artifact_file = search_paths{p};
            return;
        end
    end
end

function X_resampled = resample_eeg_data(X, sfreq_old, sfreq_new)
    [P, Q] = rat(sfreq_new / sfreq_old);
    X_resampled = resample(X', P, Q)';
end

function X_interp = interpolate_channels(X_source, n_target)
    [n_source, T] = size(X_source);
    
    if n_source == n_target
        X_interp = X_source;
        return;
    end
    
    if n_target > n_source
        X_interp = zeros(n_target, T);
        source_idx = linspace(1, n_source, n_target);
        
        for t = 1:T
            X_interp(:, t) = interp1(1:n_source, X_source(:, t), source_idx, 'linear', 'extrap');
        end
    else
        X_interp = zeros(n_target, T);
        bin_size = n_source / n_target;
        
        for i = 1:n_target
            start_idx = round((i-1) * bin_size + 1);
            end_idx = round(i * bin_size);
            X_interp(i, :) = mean(X_source(start_idx:end_idx, :), 1);
        end
    end
end

function pink_noise = generate_pink_noise(N)
    white = randn(1, N);
    fft_white = fft(white);
    
    freqs = (0:N-1)';
    freqs(1) = 1;
    spectrum_1d = 1 ./ sqrt(freqs);
    
    if mod(N, 2) == 0
        spectrum = [spectrum_1d(1:N/2+1); flipud(spectrum_1d(2:N/2))];
    else
        spectrum = [spectrum_1d(1:(N+1)/2); flipud(spectrum_1d(2:(N+1)/2))];
    end
    
    spectrum = spectrum(1:N);
    spectrum = reshape(spectrum, size(fft_white));
    
    fft_pink = fft_white .* spectrum;
    pink_noise = real(ifft(fft_pink));
    pink_noise = pink_noise / (std(pink_noise) + eps);
end

function templates = generate_microstate_templates(K, pos, W)
    C = size(pos, 1);
    templates = zeros(K, C);
    anchors = fibonacci_sphere(K);
    for k = 1:K
        v = pos * anchors(k, :)';
        m = W * v;
        m = m - mean(m);
        templates(k, :) = m / (norm(m) + eps);
    end
end

function g = ar1_positive_realistic(T, rho, eta)
    % Generate GFP envelope with REALISTIC range: 0.3-0.8
    if T < 2
        g = 0.5 * ones(1, T);
        return;
    end
    
    g = zeros(1, T);
    x = 0;
    for t = 1:T
        x = rho * x + eta * randn;
        g(t) = log1p(exp(x));
    end
    
    g_mean = mean(g);
    g_std = std(g);
    if g_std > eps
        g = (g - g_mean) / g_std;
    else
        g = (g - g_mean);
    end
    
    g_min = min(g);
    g_max = max(g);
    if g_max > g_min
        % Map to 0.3-0.8 range (more realistic GFP)
        g = 0.3 + 0.5 * ((g - g_min) / (g_max - g_min));
    else
        g = 0.55 * ones(1, T);
    end
end

function W = rbf_kernel(pos, len_scale)
    D = pdist2(pos, pos);
    W = exp(-(D.^2) / (2 * len_scale^2));
    W = W ./ (sum(W, 2) + eps);
end

function pos = fibonacci_sphere(C)
    ga = (sqrt(5) - 1) / 2;
    i = (0:C-1)' + 0.5;
    phi = 2 * pi * mod(i * ga, 1);
    z = 1 - 2 * i / C;
    r = sqrt(max(0, 1 - z.^2));
    pos = [r.*cos(phi), r.*sin(phi), z];
end
