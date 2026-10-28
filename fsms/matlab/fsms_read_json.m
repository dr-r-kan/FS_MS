function value = fsms_read_json(path_value)
%FSMS_READ_JSON jsondecode of a UTF-8 file.
    path_value = char(string(path_value));
    if ~isfile(path_value)
        error('FSMS:JsonMissing', 'JSON file not found: %s', path_value);
    end
    value = jsondecode(fileread(path_value));
end
