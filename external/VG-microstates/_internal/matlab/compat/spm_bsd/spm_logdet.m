function H = spm_logdet(C, varargin)
%SPM_LOGDET Compatibility wrapper for BSD toolboxes calling spm_logdet(A,true).

    if isempty(C)
        H = 0;
        return;
    end

    C = full(C);
    keep = diag(C) ~= 0;
    C = C(keep, keep);
    if isempty(C)
        H = 0;
        return;
    end
    if any(isnan(C(:)))
        H = NaN;
        return;
    end

    tol = 1e-16;
    if norm(C - C', inf) <= tol
        C = (C + C') / 2;
        try
            R = chol(C);
            H = 2 * sum(log(abs(diag(R))));
            return;
        catch
        end
    end

    s = svd(C);
    s = s(isfinite(s) & s > tol & s < 1 / tol);
    H = sum(log(s));
end
