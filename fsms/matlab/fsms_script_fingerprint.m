function fp = fsms_script_fingerprint(repo_root)
%FSMS_SCRIPT_FINGERPRINT SHA-256 of every MATLAB source and data file that determines stage-C output.
%   Covers: all .m/.json/.set files of the vendored VG-microstates tree (including the MetaMaps
%   template and configuration), the three root-level stage scripts, and fsms/matlab/*.m.
%   Returns fp.files (struct array: path, sha256), fp.map (path -> sha256 struct) and fp.combined.
    repo_root = char(string(repo_root));
    vg_root = fullfile(repo_root, 'external', 'VG-microstates');
    paths = strings(0, 1);
    paths = [paths; local_list(vg_root, ["*.m", "*.json", "*.set"])];
    paths = [paths; local_list(fullfile(repo_root, 'fsms', 'matlab'), "*.m")];
    for name = ["run_vg_microstates.m", "recover_timestamped_transition_backfits.m", ...
            "export_directional_peak_windows.m"]
        paths(end + 1, 1) = string(fullfile(repo_root, char(name))); %#ok<AGROW>
    end
    paths = unique(paths);
    rel = strings(numel(paths), 1);
    for i = 1:numel(paths)
        rel(i) = strrep(extractAfter(paths(i), strlength(string(repo_root)) + 1), '\', '/');
    end
    [rel, order] = sort(rel);
    paths = paths(order);
    files = repmat(struct('path', '', 'sha256', ''), numel(paths), 1);
    lines = strings(numel(paths), 1);
    map = struct();
    for i = 1:numel(paths)
        if ~isfile(paths(i))
            error('FSMS:FingerprintMissing', 'Expected stage source file is missing: %s', paths(i));
        end
        digest = fsms_sha256_file(paths(i));
        files(i).path = char(rel(i));
        files(i).sha256 = digest;
        lines(i) = rel(i) + string(char(9)) + string(digest);
        map.(matlab.lang.makeValidName(char(rel(i)))) = digest;
    end
    fp = struct('files', files, 'map', map, ...
        'combined', fsms_sha256_text(strjoin(lines, newline)));
end

function out = local_list(folder, patterns)
    out = strings(0, 1);
    if ~isfolder(folder), return; end
    for pattern = string(patterns(:))'
        listing = dir(fullfile(folder, '**', char(pattern)));
        listing = listing(~[listing.isdir]);
        for i = 1:numel(listing)
            full = string(fullfile(listing(i).folder, listing(i).name));
            if ~endsWith(full, ".part")
                out(end + 1, 1) = full; %#ok<AGROW>
            end
        end
    end
end
