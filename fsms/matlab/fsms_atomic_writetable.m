function fsms_atomic_writetable(T, path_value, varargin)
%FSMS_ATOMIC_WRITETABLE writetable() via a temporary file and rename.
%   Extra arguments are passed to writetable (e.g. 'QuoteStrings', true).
    path_value = char(string(path_value));
    folder = fileparts(path_value);
    if ~isempty(folder) && ~isfolder(folder), mkdir(folder); end
    tmp = [path_value '.' fsms_tmp_token() '.part'];
    try
        writetable(T, tmp, 'FileType', 'text', 'Delimiter', ',', varargin{:});
    catch ME
        if isfile(tmp), delete(tmp); end
        rethrow(ME);
    end
    fsms_move_into_place(tmp, path_value);
end
