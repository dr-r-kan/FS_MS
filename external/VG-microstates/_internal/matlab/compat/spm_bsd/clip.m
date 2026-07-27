function y = clip(x, lo, hi)
%CLIP Compatibility helper for SPM BSD on MATLAB releases with datafun/clip.
    if isempty(lo), lo = -Inf; else, lo = min(lo(:)); end
    if isempty(hi), hi = Inf; else, hi = max(hi(:)); end
    y = min(max(x, lo), hi);
end
