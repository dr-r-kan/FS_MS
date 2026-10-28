function [window_index, start_s, stop_s, endpoint_s] = fsms_window_specs(condition_key, duration_s)
%FSMS_WINDOW_SPECS The non-overlapping 60-s windows exported for one record.
%   condition_key is the lower-case condition: "preict" gives up to 20 windows counted back from the end
%   of the record (the ictus), endpoint_s = seconds between the window end and the ictus; "ii" gives
%   consecutive windows from the start of the record; any other exported condition ("postict") gives its
%   first minute only. Records too short for a full 60-s window yield no windows.
    condition_key = lower(string(condition_key));
    duration_s = double(duration_s);
    if condition_key == "preict"
        endpoints = 0:60:min(1140, floor((duration_s - 60) / 60) * 60);
        window_index = (1:numel(endpoints))';
        stop_s = duration_s - endpoints(:);
        start_s = stop_s - 60;
        endpoint_s = endpoints(:);
    elseif condition_key == "ii"
        starts = 0:60:(floor(duration_s / 60) * 60 - 60);
        window_index = (1:numel(starts))';
        start_s = starts(:);
        stop_s = start_s + 60;
        endpoint_s = nan(numel(starts), 1);
    elseif duration_s >= 60
        window_index = 1;
        start_s = 0;
        stop_s = 60;
        endpoint_s = NaN;
    else
        window_index = zeros(0, 1);
        start_s = zeros(0, 1);
        stop_s = zeros(0, 1);
        endpoint_s = zeros(0, 1);
    end
end
