function fsms_assert_unrestricted(path_value, purpose)
%FSMS_ASSERT_UNRESTRICTED Refuse paths inside restricted real-study storage (MATLAB twin of fsms.guard).
%
%   fsms_assert_unrestricted(path, purpose) errors with FSMS:RestrictedData when path lies inside one of
%   the roots listed in fsms/guard.py (RESTRICTED_ROOTS, parsed from that file so there is a single source
%   of truth). The comparison is purely textual (absolute path, case-insensitive, no file-system access,
%   so the check itself never touches restricted storage). Subst/junction aliases are NOT resolved here;
%   the Python guard (os.path.realpath) is the primary defence and the orchestrator calls it first.
%
%   The check is skipped only while a real-study run is in progress: run_study_hierarchy(...,
%   'real_study_mode', true) sets it, and the orchestrator passes that option only after fsms.lock has
%   verified the registration record. Calling any stage function directly never bypasses the guard.
    if nargin < 2, purpose = 'access'; end
    if isappdata(0, 'FSMS_REAL_STUDY_MODE') && logical(getappdata(0, 'FSMS_REAL_STUDY_MODE'))
        return;
    end
    candidate = local_normalise(path_value);
    roots = fsms_restricted_roots();
    for i = 1:numel(roots)
        root = local_normalise(roots{i});
        if strcmp(candidate, root) || startsWith(candidate, [root '\'])
            error('FSMS:RestrictedData', ['Refusing %s of %s: it is inside restricted real-study storage ' ...
                '(%s) and the analysis plan is not yet preregistered/unlocked for real data.'], purpose, ...
                char(string(path_value)), roots{i});
        end
    end
end

function out = local_normalise(value)
    value = strrep(char(string(value)), '/', '\');
    out = lower(char(java.io.File(value).getAbsolutePath()));
    out = regexprep(out, '\\+$', '');
end
