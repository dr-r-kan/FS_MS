function fsms_atomic_write_text(path_value, text)
%FSMS_ATOMIC_WRITE_TEXT Write UTF-8 text so readers see the old or the new complete file.
    path_value = char(string(path_value));
    folder = fileparts(path_value);
    if ~isempty(folder) && ~isfolder(folder), mkdir(folder); end
    tmp = [path_value '.' fsms_tmp_token() '.part'];
    fid = fopen(tmp, 'w');
    if fid < 0
        error('FSMS:WriteFailed', 'Cannot open %s for writing.', tmp);
    end
    try
        fwrite(fid, unicode2native(char(string(text)), 'UTF-8'), 'uint8');
        fclose(fid);
    catch ME
        fclose(fid);
        if isfile(tmp), delete(tmp); end
        rethrow(ME);
    end
    fsms_move_into_place(tmp, path_value);
end
