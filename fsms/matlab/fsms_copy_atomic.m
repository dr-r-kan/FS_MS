function fsms_copy_atomic(source, target)
%FSMS_COPY_ATOMIC Copy a file via a temporary name so a crash never leaves a truncated target.
    source = char(string(source));
    target = char(string(target));
    folder = fileparts(target);
    if ~isempty(folder) && ~isfolder(folder), mkdir(folder); end
    tmp = [target '.' fsms_tmp_token() '.part'];
    [ok, msg] = copyfile(source, tmp, 'f');
    if ~ok
        error('FSMS:CopyFailed', 'Could not copy %s: %s', source, msg);
    end
    fsms_move_into_place(tmp, target);
end
