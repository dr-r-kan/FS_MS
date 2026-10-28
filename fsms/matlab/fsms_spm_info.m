function info = fsms_spm_info()
%FSMS_SPM_INFO Best-effort provenance of the SPM installation used (never errors).
    info = struct('dir', '', 'version', '', 'git_commit', '', 'spm_mix_sha256', '', ...
        'spm_mix_prior_sha256', '', 'spm_mix_path', '');
    try
        spm_main = which('spm');
        if ~isempty(spm_main)
            info.dir = fileparts(spm_main);
        end
        mix = which('spm_mix');
        info.spm_mix_path = mix;
        if ~isempty(mix), info.spm_mix_sha256 = fsms_sha256_file(mix); end
        mixp = which('spm_mix_prior');
        if ~isempty(mixp), info.spm_mix_prior_sha256 = fsms_sha256_file(mixp); end
    catch
    end
    try
        evalc('v = spm(''Ver'');');
        info.version = char(string(v));
    catch
    end
    try
        info.git_commit = local_git_commit(info.dir);
    catch
    end
end

function commit = local_git_commit(spm_dir)
    commit = '';
    if isempty(spm_dir), return; end
    git_dir = fullfile(spm_dir, '.git');
    head_file = fullfile(git_dir, 'HEAD');
    if ~isfile(head_file), return; end
    head = strtrim(fileread(head_file));
    if startsWith(head, 'ref:')
        ref = strtrim(extractAfter(head, 'ref:'));
        ref_file = fullfile(git_dir, strrep(ref, '/', filesep));
        if isfile(ref_file)
            commit = strtrim(fileread(ref_file));
        else
            packed = fullfile(git_dir, 'packed-refs');
            if isfile(packed)
                lines = strtrim(splitlines(string(fileread(packed))));
                hit = find(endsWith(lines, " " + string(ref)), 1);
                if ~isempty(hit)
                    parts = split(lines(hit));
                    commit = char(parts(1));
                end
            end
        end
    else
        commit = head;
    end
end
