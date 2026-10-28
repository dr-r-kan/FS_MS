function fsms_write_json(path_value, value)
%FSMS_WRITE_JSON Atomically write a pretty-printed JSON file (temporary file, then rename).
    text = jsonencode(value, 'PrettyPrint', true);
    fsms_atomic_write_text(path_value, [text newline]);
end
