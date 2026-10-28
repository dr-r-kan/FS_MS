function roots = fsms_restricted_roots()
%FSMS_RESTRICTED_ROOTS The restricted real-study roots, parsed from fsms/guard.py (RESTRICTED_ROOTS).
%   Falls back to the built-in copy below if guard.py cannot be parsed; the two are checked against each
%   other by tests/matlab/test_prespecified_labels.m.
    fallback = {'F:\FS_microstates_study', 'G:\FS_microstates_study', 'D:\EEGData', 'D:\Chalfont', ...
        'G:\Chalfont', 'E:\EEGs', 'F:\EEGs'};
    roots = fallback;
    guard_py = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'fsms', 'guard.py');
    if ~isfile(guard_py), return; end
    text = fileread(guard_py);
    block = regexp(text, 'RESTRICTED_ROOTS[^=]*=\s*\((.*?)\n\)', 'tokens', 'once');
    if isempty(block), return; end
    found = regexp(block{1}, 'r"([^"]+)"', 'tokens');
    if isempty(found), return; end
    parsed = cellfun(@(c) c{1}, found, 'UniformOutput', false);
    roots = parsed(:)';
end
