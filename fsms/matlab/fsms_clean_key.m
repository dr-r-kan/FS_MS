function key = fsms_clean_key(value)
%FSMS_CLEAN_KEY Same file-name key the vendored hierarchy uses for participant/condition folders.
    key = regexprep(char(string(value)), '[^A-Za-z0-9_-]', '_');
    if isempty(key), key = 'empty'; end
end
