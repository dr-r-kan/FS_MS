function token = fsms_tmp_token()
%FSMS_TMP_TOKEN Unique suffix for temporary files (random UUID; never reused).
    token = char(java.util.UUID.randomUUID().toString());
end
