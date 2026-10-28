function digest = fsms_sha256_text(text_value)
%FSMS_SHA256_TEXT Lower-case hex SHA-256 of the UTF-8 bytes of a string.
    md = java.security.MessageDigest.getInstance('SHA-256');
    bytes = unicode2native(char(string(text_value)), 'UTF-8');
    md.update(uint8(bytes));
    digest = fsms_hex(md.digest());
end
