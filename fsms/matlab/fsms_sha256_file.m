function digest = fsms_sha256_file(path_value)
%FSMS_SHA256_FILE Lower-case hex SHA-256 of a file (streamed in 4 MiB blocks).
    path_value = char(string(path_value));
    if ~isfile(path_value)
        error('FSMS:HashMissingFile', 'Cannot hash missing file: %s', path_value);
    end
    md = java.security.MessageDigest.getInstance('SHA-256');
    fid = fopen(path_value, 'r');
    if fid < 0
        error('FSMS:HashOpenFailed', 'Cannot open file for hashing: %s', path_value);
    end
    closer = onCleanup(@() fclose(fid)); %#ok<NASGU>
    block = 4 * 1024 * 1024;
    while true
        bytes = fread(fid, block, '*uint8');
        if isempty(bytes), break; end
        md.update(bytes);
    end
    digest = fsms_hex(md.digest());
end
