function root_dir = setup_microstate_paths(root_dir)
% SETUP_MICROSTATE_PATHS Add repository internal helper folders to the path.

    if nargin < 1 || isempty(root_dir)
        root_dir = fileparts(mfilename('fullpath'));
    else
        root_dir = char(root_dir);
    end

    internal_matlab = fullfile(root_dir, '_internal', 'matlab');
    analysis_matlab = fullfile(root_dir, '_analysis', 'matlab');
    koenig_dir = fullfile(internal_matlab, 'Koenig_code');
    bsd_compat_dir = fullfile(internal_matlab, 'compat', 'spm_bsd');

    add_if_dir(internal_matlab);
    add_if_dir(analysis_matlab);
    add_if_dir(koenig_dir);
    add_if_dir(bsd_compat_dir);
end

function add_if_dir(path_to_add)
    if isfolder(path_to_add) && ~contains(path, path_to_add)
        addpath(path_to_add);
    end
end
