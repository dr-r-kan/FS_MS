function [retained_mask, exclusions, peak_counts] = fsms_screen_peak_counts(manifest, screen_dir, min_peaks_per_record, min_peaks_per_participant)
%FSMS_SCREEN_PEAK_COUNTS Pre-specified exclusion of records/participants with too few GFP peaks.
%
%   [retained_mask, exclusions, peak_counts] = fsms_screen_peak_counts(manifest, screen_dir, ...
%       min_peaks_per_record, min_peaks_per_participant)
%
%   manifest   the sorted manifest table (fsms_read_manifest) a hierarchy run in screen_dir was fitted on
%   screen_dir hierarchy output folder holding gfp_peaks/<NNN>_<participant>_<condition>_gfp_peaks.mat
%
%   RULE (mechanical feasibility only; it never looks at labels, maps or outcomes)
%     1. A record with fewer than min_peaks_per_record GFP peaks is EXCLUDED
%        (reason code RECORD_TOO_FEW_GFP_PEAKS).
%     2. A participant whose remaining records hold fewer than min_peaks_per_participant GFP peaks in total
%        is EXCLUDED with all of its records (reason code PARTICIPANT_TOO_FEW_GFP_PEAKS), because its
%        participant-level K=6 fit would not be estimable.
%     3. If, after 1-2, no record remains or any group or condition holds fewer than
%        min_peaks_per_participant peaks, the stage stops with FSMS:TooFewMapsAfterExclusion
%        (nothing is silently skipped at that level).
%   Retained participants' fits are untouched by this screening: it only removes whole records or
%   participants before the global fit, and the fit mathematics are the vendored ones.
    n = height(manifest);
    peak_counts = zeros(n, 1);
    for i = 1:n
        cache_file = fullfile(screen_dir, 'gfp_peaks', sprintf('%03d_%s_%s_gfp_peaks.mat', i, ...
            fsms_clean_key(manifest.participant(i)), fsms_clean_key(manifest.condition(i))));
        if ~isfile(cache_file)
            error('FSMS:PeakCacheMissing', 'Screening needs GFP-peak cache %s', cache_file);
        end
        C = load(cache_file, 'rec');
        if ~strcmp(fsms_path_key(C.rec.file_path), fsms_path_key(manifest.file_path(i)))
            error('FSMS:PeakCacheStale', 'Peak cache %s belongs to %s, not to manifest record %s.', ...
                cache_file, C.rec.file_path, manifest.file_path(i));
        end
        peak_counts(i) = size(C.rec.maps_norm, 1);
    end
    reason = strings(n, 1);
    detail = strings(n, 1);
    threshold = zeros(n, 1);
    low = peak_counts < min_peaks_per_record;
    reason(low) = "RECORD_TOO_FEW_GFP_PEAKS";
    threshold(low) = min_peaks_per_record;
    detail(low) = "record has " + string(peak_counts(low)) + " GFP peaks, minimum " + string(min_peaks_per_record);
    participants = unique(manifest.participant, 'stable');
    for k = 1:numel(participants)
        rows = find(manifest.participant == participants(k));
        alive = rows(reason(rows) == "");
        total = sum(peak_counts(alive));
        if ~isempty(alive) && total < min_peaks_per_participant
            reason(alive) = "PARTICIPANT_TOO_FEW_GFP_PEAKS";
            threshold(alive) = min_peaks_per_participant;
            detail(alive) = "participant retains " + string(total) + " GFP peaks in total, minimum " + ...
                string(min_peaks_per_participant);
        end
    end
    retained_mask = reason == "";
    if ~any(retained_mask)
        error('FSMS:TooFewMapsAfterExclusion', 'Every record was excluded by the GFP-peak screening.');
    end
    for column = ["group", "condition"]
        levels = unique(manifest.(column)(retained_mask));
        for level = levels'
            total = sum(peak_counts(retained_mask & manifest.(column) == level));
            if total < min_peaks_per_participant
                error('FSMS:TooFewMapsAfterExclusion', ...
                    '%s "%s" holds only %d GFP peaks after exclusions (minimum %d).', column, level, total, min_peaks_per_participant);
            end
        end
    end
    excluded = find(~retained_mask);
    exclusions = table(manifest.manifest_row(excluded), manifest.participant(excluded), ...
        manifest.condition(excluded), manifest.group(excluded), manifest.file_path(excluded), ...
        reason(excluded), peak_counts(excluded), threshold(excluded), detail(excluded), ...
        'VariableNames', {'manifest_row', 'participant', 'condition', 'group', 'file_path', ...
        'reason_code', 'n_gfp_peaks', 'threshold', 'detail'});
end
