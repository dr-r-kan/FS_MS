function s = fsms_hex(bytes)
%FSMS_HEX Lower-case hexadecimal string of a Java/MATLAB byte vector.
    bytes = typecast(int8(bytes(:)), 'uint8');
    s = lower(reshape(dec2hex(bytes, 2)', 1, []));
end
