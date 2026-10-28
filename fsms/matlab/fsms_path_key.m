function key = fsms_path_key(value)
%FSMS_PATH_KEY Case-insensitive, slash-normalised key used to match manifest rows to caches.
    key = lower(strrep(char(string(value)), '/', '\'));
end
