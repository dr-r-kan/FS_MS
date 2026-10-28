function env = fsms_paths()
%FSMS_PATHS Put the repo, vendored VG-microstates and SPM mixture code on the MATLAB path.
%   Idempotent and headless. Returns a struct describing the software environment
%   (repo root, VG root, MATLAB release, SPM location/version/git commit).
%   Nothing here depends on a hard-coded drive letter or user directory.

    here = fileparts(mfilename('fullpath'));
    repo_root = fileparts(fileparts(here));
    vg_root = fullfile(repo_root, 'external', 'VG-microstates');
    if ~isfolder(vg_root)
        error('FSMS:VgMissing', 'Vendored VG-microstates not found at %s', vg_root);
    end
    addpath(repo_root);
    addpath(here);
    addpath(vg_root);
    setup_microstate_paths(vg_root);
    util = microstate_utilities();
    repo_cfg = util.load_config();
    [ok, info] = util.ensure_spm_mix('', repo_cfg.paths.spm_mixture_paths, false);
    if ~ok
        error('FSMS:SpmMissing', ['SPM spm_mix not found on the MATLAB path. Checked: %s. ' ...
            'Add SPM (with toolbox/mixture) to the path before running the study.'], ...
            strjoin(info.attempted, ', '));
    end
    if exist('spm_mix_prior', 'file') ~= 2
        error('FSMS:SpmMixPriorMissing', 'Vendored spm_mix_prior.m is not on the path.');
    end
    env = struct();
    env.repo_root = repo_root;
    env.vg_root = vg_root;
    env.matlab_dir = here;
    env.matlab_version = version;
    env.matlab_release = version('-release');
    env.spm = fsms_spm_info();
end
