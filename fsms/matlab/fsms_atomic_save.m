function fsms_atomic_save(path_value, payload, varargin)
%FSMS_ATOMIC_SAVE save(path, '-struct', 'payload', ...) via a temporary file and rename.
%   Extra arguments are save() options such as '-v7.3'.
    path_value = char(string(path_value));
    folder = fileparts(path_value);
    if ~isempty(folder) && ~isfolder(folder), mkdir(folder); end
    tmp = [path_value '.' fsms_tmp_token() '.part'];
    try
        save(tmp, '-struct', 'payload', '-mat', varargin{:});
    catch ME
        if isfile(tmp), delete(tmp); end
        rethrow(ME);
    end
    fsms_move_into_place(tmp, path_value);
end
