function test_prespecified_labels(work_dir)
%TEST_PRESPECIFIED_LABELS Unit tests of the pre-specified labelling rule and the stage-C helpers.
%   test_prespecified_labels(work_dir) uses work_dir for temporary files (must not be restricted study storage).
%   Prints ALL LABELLING TESTS PASSED or errors. Run headless:  matlab -batch "test_prespecified_labels('<dir>')"
    here = fileparts(mfilename('fullpath'));
    repo_root = fileparts(fileparts(here));
    addpath(fullfile(repo_root, 'fsms', 'matlab'));
    env = fsms_paths();
    if ~isfolder(work_dir), mkdir(work_dir); end
    tmpl_file = fullfile(env.vg_root, 'MetaMaps_2023_06.set');
    [tmaps, tlabels, tchannels, tchanlocs] = load_metamaps_templates(tmpl_file, 'K', 7);
    util = microstate_utilities();
    results = {};

    classes = {'A', 'B', 'C', 'E', 'F', 'G'};
    ch19 = {'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2'};
    ch21 = [ch19, {'Oz', 'Fpz'}];
    alias = containers.Map({'T3', 'T4', 'T5', 'T6'}, {'T7/T3', 'T8/T4', 'P7/T5', 'P8/T6'});

    % ---- 1. recovery of a known permutation, coordinate mode (19 channels, 15 label matches) --------------
    rng(7);
    [maps19, locs19] = sample_template(tmaps, tlabels, tchannels, tchanlocs, ch19, alias, util);
    [centers, truth] = shuffled_class_maps(maps19, tlabels, classes, 0.05, util);
    r = template_prespecified_v2(centers, ch19, locs19);
    results{end + 1} = check(r.accepted, 'coordinate mode: shuffled known maps accepted');
    results{end + 1} = check(strcmp(r.channel_match_mode, 'coordinates_idw_to_template'), ...
        sprintf('19 channels use coordinate IDW (got %s)', r.channel_match_mode));
    results{end + 1} = check(isequal(r.labels(:), truth(:)), ...
        sprintf('coordinate mode recovers truth %s (got %s)', strjoin(truth, ''), strjoin(r.labels, '')));
    results{end + 1} = check(all(r.assigned_abs_r >= 0.5), 'all assigned |r| >= 0.5');
    results{end + 1} = check(~any(strcmp(r.labels, 'D')), 'class D is never assigned');

    % ---- 2. label mode (21 channels, >= 16 label matches) ----------------------------------------------
    [maps21, locs21] = sample_template(tmaps, tlabels, tchannels, tchanlocs, ch21, alias, util);
    [centers21, truth21] = shuffled_class_maps(maps21, tlabels, classes, 0.05, util);
    r21 = template_prespecified_v2(centers21, ch21, locs21);
    results{end + 1} = check(strcmp(r21.channel_match_mode, 'labels'), ...
        sprintf('21 channels use label matching (got %s)', r21.channel_match_mode));
    results{end + 1} = check(r21.accepted && isequal(r21.labels(:), truth21(:)), 'label mode recovers truth');

    % ---- 3. polarity invariance ---------------------------------------------------------------------------
    flipped = centers .* [-1; 1; -1; 1; 1; -1];
    rf = template_prespecified_v2(flipped, ch19, locs19);
    results{end + 1} = check(isequal(rf.labels(:), r.labels(:)), 'labels are polarity invariant');
    results{end + 1} = check(all(rf.polarity(:) == r.polarity(:) .* [-1; 1; -1; 1; 1; -1]), 'polarity is reported');

    % ---- 4. rejection: random maps ---------------------------------------------------------------------------
    rng(3);
    random_centers = util.normalize_maps(randn(6, 19));
    rr = template_prespecified_v2(random_centers, ch19, locs19);
    results{end + 1} = check(rr.accepted && isempty(rr.reasons) && strcmp(rr.labelling_quality, 'below_threshold') && ...
        ~isempty(rr.states_below_threshold) && numel(rr.states_below_threshold) == sum(rr.assigned_abs_r < 0.5), ...
        'random maps are NOT a technical failure; they are flagged below_threshold with the weak states listed');
    results{end + 1} = check(strcmp(r.labelling_quality, 'ok') && isempty(r.states_below_threshold), ...
        'true template maps have labelling_quality ok');

    % ---- 5. rejection: first_common_channels fallback is never accepted ------------------------------------
    anonymous = arrayfun(@(i) sprintf('Ch%03d', i), 1:19, 'UniformOutput', false);
    rn = template_prespecified_v2(centers, anonymous, []);
    results{end + 1} = check(~rn.accepted && strcmp(rn.channel_match_mode, 'first_common_channels') && ...
        any(strcmp(rn.reasons, 'CHANNEL_MATCH_MODE_NOT_ALLOWED:first_common_channels')), ...
        'unlabelled channels without positions are rejected (first_common_channels)');

    % ---- 6. deterministic tie-breaking ----------------------------------------------------------------------
    tie = centers;
    tie(2, :) = tie(1, :);                      % two identical fitted states
    t1 = template_prespecified_v2(tie, ch19, locs19);
    t2 = template_prespecified_v2(tie, ch19, locs19);
    results{end + 1} = check(isequal(t1.labels, t2.labels) && isequal(t1.assignment_class_index, t2.assignment_class_index), ...
        'repeated evaluation gives the identical assignment');
    results{end + 1} = check(t1.n_tied_optimal_assignments >= 2 && ~t1.accepted && ...
        any(startsWith(t1.reasons, 'AMBIGUOUS_ASSIGNMENT')), 'identical states: tied optimum is a technical failure');
    results{end + 1} = check(t1.assignment_class_index(1) < t1.assignment_class_index(2), ...
        'ties resolve to the lexicographically smallest assignment vector');

    % ---- 7. class D excluded a priori -----------------------------------------------------------------------
    with_d = centers;
    d19 = sample_class(tmaps, tlabels, tchannels, 'D', ch19, alias, util);
    with_d(1, :) = d19;
    rd = template_prespecified_v2(with_d, ch19, locs19);
    results{end + 1} = check(~any(strcmp(rd.labels, 'D')), 'a D-like map is still never labelled D');

    % ---- 8. prespecified_global_labels: files, audit, idempotency, rejection ---------------------------------
    good_dir = fullfile(work_dir, 'fake_hierarchy_good');
    local_make_global_solution(good_dir, centers, ch19, locs19);
    out = prespecified_global_labels(good_dir);
    csv_file = fullfile(good_dir, 'global', 'global_manual_labels.csv');
    T = readtable(csv_file, 'TextType', 'string', 'Delimiter', ',');
    results{end + 1} = check(isequal(T.Properties.VariableNames, {'state_index', 'label', 'selected_utc', 'source'}), ...
        'labels CSV has the established columns');
    results{end + 1} = check(isequal(cellstr(T.label)', truth) && all(T.source == "template_prespecified_v2"), ...
        'labels CSV content and source');
    L = load(fullfile(good_dir, 'global', 'global_manual_labels.mat'));
    results{end + 1} = check(isequal(cellstr(string(L.labels(:)))', truth) && isequal(L.reference_centers, centers) && ...
        strcmp(char(string(L.label_source)), 'template_prespecified_v2'), 'labels MAT content (labels, reference_centers, label_source)');
    audit = fsms_read_json(fullfile(good_dir, 'global', 'labelling_audit.json'));
    needed = {'rule_id', 'accepted', 'labels_by_fitted_state', 'correlation_matrix_abs', 'channel_match_mode', ...
        'template_sha256', 'script_sha256', 'matlab_version', 'spm', 'strong_threshold', 'classes_excluded_a_priori'};
    results{end + 1} = check(all(isfield(audit, needed)), 'audit carries the required fields');
    results{end + 1} = check(strcmp(audit.template_sha256, fsms_sha256_file(tmpl_file)), 'audit template SHA-256 is correct');
    h1 = fsms_sha256_file(csv_file);
    audit_hash = fsms_sha256_file(fullfile(good_dir, 'global', 'labelling_audit.json'));
    again = prespecified_global_labels(good_dir);
    results{end + 1} = check(again.reused_existing && strcmp(h1, fsms_sha256_file(csv_file)) && ...
        strcmp(audit_hash, fsms_sha256_file(fullfile(good_dir, 'global', 'labelling_audit.json'))), ...
        'second call leaves label and audit files byte-identical'); %#ok<NASGU>
    results{end + 1} = check(isequal(out.labels, truth(:)), 'returned labels equal truth');

    bad_dir = fullfile(work_dir, 'fake_hierarchy_bad');
    local_make_global_solution(bad_dir, centers, anonymous, []);      % no positions, < 16 label matches
    threw = '';
    try
        prespecified_global_labels(bad_dir);
    catch ME
        threw = ME.identifier;
    end
    results{end + 1} = check(strcmp(threw, 'FSMS:LabellingRejected'), 'rejection raises FSMS:LabellingRejected');
    rej = fullfile(bad_dir, 'global', 'LABELLING_REJECTED.json');
    results{end + 1} = check(isfile(rej) && ~isfile(fullfile(bad_dir, 'global', 'global_manual_labels.csv')) && ...
        ~isfile(fullfile(bad_dir, 'global', 'global_manual_labels.mat')), ...
        'rejection writes LABELLING_REJECTED.json and NO label files');
    evidence = fsms_read_json(rej);
    results{end + 1} = check(~evidence.accepted && any(strcmp(evidence.rejection_reasons, ...
        'CHANNEL_MATCH_MODE_NOT_ALLOWED:first_common_channels')) && isfield(evidence, 'correlation_matrix_abs'), ...
        'rejection file carries the evidence (technical failure only)');

    % below-threshold solution PROCEEDS and is flagged (state names are nominal)
    weak_dir = fullfile(work_dir, 'fake_hierarchy_weak');
    local_make_global_solution(weak_dir, random_centers, ch19, locs19);
    weak = prespecified_global_labels(weak_dir);
    weak_audit = fsms_read_json(fullfile(weak_dir, 'global', 'labelling_audit.json'));
    results{end + 1} = check(isfile(fullfile(weak_dir, 'global', 'global_manual_labels.csv')) && ...
        ~isfile(fullfile(weak_dir, 'global', 'LABELLING_REJECTED.json')) && strcmp(weak.labelling_quality, 'below_threshold'), ...
        'below-threshold solution writes labels and no rejection marker');
    results{end + 1} = check(strcmp(weak_audit.labelling_quality, 'below_threshold') && weak_audit.accepted && ...
        ~isempty(weak_audit.states_below_threshold) && contains(weak_audit.nominal_label_semantics, 'nominal') && ...
        numel(weak_audit.assigned_abs_r) == 6 && any(weak_audit.assigned_abs_r < 0.5), ...
        'audit records labelling_quality=below_threshold, the states below, and the nominal-label statement');
    for bad = {'few', 'nan'}
        tdir = fullfile(work_dir, ['fake_hierarchy_' bad{1}]);
        tcenters = centers;
        if strcmp(bad{1}, 'few'), tcenters = centers(1:5, :); else, tcenters(2, 3) = NaN; end
        local_make_global_solution(tdir, tcenters, ch19, locs19);
        threw = '';
        try, prespecified_global_labels(tdir); catch ME, threw = ME.identifier; end
        marker = fsms_read_json(fullfile(tdir, 'global', 'LABELLING_REJECTED.json'));
        results{end + 1} = check(strcmp(threw, 'FSMS:LabellingRejected') && ~isfile(fullfile(tdir, 'global', 'global_manual_labels.csv')) && ...
            any(startsWith(marker.rejection_reasons, {'FEWER_STATES_THAN_CLASSES', 'NONFINITE_MAPS'})), ...
            sprintf('technical failure "%s" stops with a rejection marker and no labels', bad{1})); %#ok<AGROW>
    end

    % ---- 9. helpers --------------------------------------------------------------------------------------------
    results{end + 1} = check(strcmp(fsms_sha256_text('abc'), ...
        'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'), 'SHA-256 of "abc"');
    probe = fullfile(work_dir, 'atomic_probe.txt');
    fsms_atomic_write_text(probe, 'abc');
    results{end + 1} = check(strcmp(fsms_sha256_file(probe), fsms_sha256_text('abc')) && isempty(dir([probe '.*.part'])), ...
        'atomic text write leaves no temporary file');
    [~, ~, ~, ep] = fsms_window_specs('preict', 150);
    [~, s_ii] = fsms_window_specs('ii', 180);
    [wi_short] = fsms_window_specs('preict', 59);
    results{end + 1} = check(isequal(ep', [0 60]) && numel(s_ii) == 3 && isempty(wi_short), 'window specs (preict/ii/short)');

    manifest_dir = fullfile(work_dir, 'manifest_tests');
    if ~isfolder(manifest_dir), mkdir(manifest_dir); end
    fid = fopen(fullfile(manifest_dir, 'a.mat'), 'w'); fclose(fid);
    fid = fopen(fullfile(manifest_dir, 'b.mat'), 'w'); fclose(fid);
    collide = fullfile(manifest_dir, 'collide.csv');
    writetable(table(["A.1"; "A_1"], ["II"; "II"], ["FS"; "FS"], [string(fullfile(manifest_dir, 'a.mat')); ...
        string(fullfile(manifest_dir, 'b.mat'))], 'VariableNames', {'participant', 'condition', 'group', 'file_path'}), collide);
    results{end + 1} = check(throws('FSMS:ManifestKeyCollision', @() fsms_read_manifest(collide)), ...
        'participant names that collide after cleaning are rejected');
    dup = fullfile(manifest_dir, 'dup.csv');
    writetable(table(["P1"; "P2"], ["II"; "II"], ["FS"; "FS"], [string(fullfile(manifest_dir, 'a.mat')); ...
        string(fullfile(manifest_dir, 'A.MAT'))], 'VariableNames', {'participant', 'condition', 'group', 'file_path'}), dup);
    results{end + 1} = check(throws('FSMS:ManifestDuplicatePath', @() fsms_read_manifest(dup)), 'duplicate EEG paths are rejected');

    % ---- 10. restricted-data guard (textual only: these paths are never opened) ----------------------------------
    roots = fsms_restricted_roots();
    results{end + 1} = check(numel(roots) >= 7 && any(strcmpi(roots, 'G:\FS_microstates_study')), ...
        'restricted roots are parsed from fsms/guard.py');
    all_refused = true;
    for i = 1:numel(roots)
        all_refused = all_refused && throws('FSMS:RestrictedData', @() fsms_assert_unrestricted(fullfile(roots{i}, 'x', 'y.csv'), 't'));
        all_refused = all_refused && throws('FSMS:RestrictedData', @() fsms_assert_unrestricted(upper(strrep(roots{i}, '\', '/')), 't'));
    end
    results{end + 1} = check(all_refused, 'every restricted root (and a child path, any case/slash) is refused');
    results{end + 1} = check(~throws('FSMS:RestrictedData', @() fsms_assert_unrestricted([roots{1} '_not_a_child\x'], 't')), ...
        'a sibling folder sharing a name prefix is not refused');
    results{end + 1} = check(~throws('FSMS:RestrictedData', @() fsms_assert_unrestricted(work_dir, 't')), 'scratch folder is allowed');
    results{end + 1} = check(throws('FSMS:RestrictedData', @() fsms_read_manifest('G:\FS_microstates_study\vg_microstates\m.csv')), ...
        'the manifest reader refuses a restricted path before touching it');
    results{end + 1} = check(throws('FSMS:RestrictedData', @() run_vg_microstates('D:\EEGData\m.csv', fullfile(work_dir, 'never'))), ...
        'run_vg_microstates refuses a restricted manifest');
    setappdata(0, 'FSMS_REAL_STUDY_MODE', true);
    lifted = ~throws('FSMS:RestrictedData', @() fsms_assert_unrestricted('G:\FS_microstates_study\x', 't'));
    rmappdata(0, 'FSMS_REAL_STUDY_MODE');
    results{end + 1} = check(lifted && throws('FSMS:RestrictedData', @() fsms_assert_unrestricted('G:\FS_microstates_study\x', 't')), ...
        'the guard is lifted only while real_study_mode is set');

    failed = find(~cellfun(@(x) x.ok, results));
    for i = 1:numel(results)
        fprintf('%s  %s\n', ternary(results{i}.ok, 'PASS', 'FAIL'), results{i}.msg);
    end
    if ~isempty(failed)
        error('FSMS:LabellingTestsFailed', '%d of %d labelling tests failed.', numel(failed), numel(results));
    end
    fprintf('ALL LABELLING TESTS PASSED (%d checks)\n', numel(results));
end

function out = check(condition, msg)
    out = struct('ok', logical(condition), 'msg', msg);
end

function s = ternary(c, a, b)
    if c, s = a; else, s = b; end
end

function tf = throws(identifier, fn)
    tf = false;
    try
        fn();
    catch ME
        tf = strcmp(ME.identifier, identifier);
    end
end

function [maps, chanlocs] = sample_template(tmaps, tlabels, tchannels, tchanlocs, channels, alias, util)
    idx = zeros(1, numel(channels));
    for i = 1:numel(channels)
        key = channels{i};
        if isKey(alias, key), key = alias(key); end
        idx(i) = find(strcmp(tchannels, key), 1);
    end
    maps = util.normalize_maps(tmaps(:, idx));
    chanlocs = [];
    if ~isempty(tchanlocs)
        chanlocs = tchanlocs(idx);
        for i = 1:numel(channels)
            chanlocs(i).labels = channels{i};
        end
    end
    assert(isequal(size(maps, 1), numel(tlabels)));
end

function map = sample_class(tmaps, tlabels, tchannels, cls, channels, alias, util)
    [maps, ~] = sample_template(tmaps, tlabels, tchannels, [], channels, alias, util);
    map = maps(strcmp(tlabels, cls), :);
end

function [centers, truth] = shuffled_class_maps(maps, tlabels, classes, noise, util)
    order = randperm(6);                       % fitted state k carries class classes{order(k)}
    centers = zeros(6, size(maps, 2));
    truth = cell(1, 6);
    for k = 1:6
        cls = classes{order(k)};
        centers(k, :) = maps(strcmp(tlabels, cls), :);
        truth{k} = cls;
    end
    centers = util.normalize_maps(centers + noise * randn(size(centers)) / sqrt(size(centers, 2)));
    centers = centers .* sign(randn(6, 1));
end

function local_make_global_solution(hierarchy_dir, centers, labels, chanlocs)
    gdir = fullfile(hierarchy_dir, 'global');
    if ~isfolder(gdir), mkdir(gdir); end
    Fit = struct('centers', centers, 'K_estimated', 6, 'n_maps', 1000, 'name', 'global'); %#ok<NASGU>
    common_labels = labels(:); %#ok<NASGU>
    common_chanlocs = chanlocs; %#ok<NASGU>
    save(fullfile(gdir, 'global_solution.mat'), 'Fit', 'common_labels', 'common_chanlocs', '-v7');
end
