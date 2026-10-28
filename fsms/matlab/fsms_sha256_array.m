function digest = fsms_sha256_array(values)
%FSMS_SHA256_ARRAY SHA-256 of the raw bytes of a numeric array converted to double (column-major).
    md = java.security.MessageDigest.getInstance('SHA-256');
    md.update(typecast(double(values(:)), 'uint8'));
    digest = fsms_hex(md.digest());
end
