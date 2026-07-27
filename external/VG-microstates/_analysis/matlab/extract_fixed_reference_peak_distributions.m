function D = extract_fixed_reference_peak_distributions(H, semiology_manifest_csv, grouping)
% Participant-condition GFP-map distributions on one common global basis.

    if nargin < 2, semiology_manifest_csv = ''; end
    if nargin < 3, grouping = "participant_condition"; end

    required = {'manifest','peak_records','global','selected_K','common_labels'};
    for i = 1:numel(required)
        if ~isfield(H, required{i}) || isempty(H.(required{i}))
            error('HResults.%s is required for fixed-reference inference.', required{i});
        end
    end
    M = H.manifest;
    if numel(H.peak_records) ~= height(M)
        error('peak_records and manifest must contain the same number of rows.');
    end
    centers = normalise_rows(double(H.global.centers));
    K = double(H.selected_K); C = numel(H.common_labels);
    if size(centers,1) ~= K || size(centers,2) ~= C
        error('Global centres do not match selected K/common channels.');
    end

    [row_event_type, row_semiology] = row_event_metadata(M, semiology_manifest_csv);
    key = string(M.participant) + "|" + string(M.condition);
    if grouping == "event_type" || grouping == "semiology"
        key = key + "|" + row_event_type;
    end
    if grouping == "semiology"
        key = key + "|" + row_semiology;
    end
    keys = unique(key, 'stable');
    mu = nan(numel(keys),K,C); variance = mu; n_maps = zeros(numel(keys),K);
    participant = strings(numel(keys),1); group = participant; condition = participant;
    event_type = participant; semiology = participant;
    for i = 1:numel(keys)
        idx = key == keys(i);
        first = find(idx, 1);
        participant(i) = string(M.participant(first));
        condition(i) = string(M.condition(first));
        group(i) = string(M.group(find(idx,1)));
        event_values = unique(row_event_type(idx & row_event_type ~= ""));
        if numel(event_values) == 1, event_type(i) = event_values; end
        semiology_values = unique(row_semiology(idx & row_semiology ~= ""));
        if numel(semiology_values) == 1, semiology(i) = semiology_values; end
        maps = cellfun(@(r) double(r.maps_norm), H.peak_records(idx), 'UniformOutput', false);
        X = normalise_rows(vertcat(maps{:}));
        [~, labels] = max(abs(X*centers'),[],2);
        for k = 1:K
            Xi = X(labels==k,:);
            if size(Xi,1) < 2, continue; end
            polarity = sign(Xi*centers(k,:)'); polarity(polarity==0)=1;
            Xi = Xi.*polarity;
            mu(i,k,:) = mean(Xi,1,'omitnan');
            variance(i,k,:) = max(var(Xi,0,1,'omitnan'),1e-10);
            n_maps(i,k) = size(Xi,1);
        end
    end
    labels = arrayfun(@(k)sprintf('State_%d',k),1:K,'UniformOutput',false);
    if isfield(H.global,'canonical_state_labels') && numel(H.global.canonical_state_labels)>=K
        labels = cellstr(string(H.global.canonical_state_labels(1:K)));
    end
    D = struct('mu',mu,'var',variance,'participant',participant,'group',group, ...
        'condition',condition,'n_maps',n_maps,'K',K,'C',C,'state_labels',{labels}, ...
        'channel_labels',{cellstr(string(H.common_labels(:)))},'reference',"global");
    D.event_type = event_type;
    if grouping == "participant_condition"
        D.semiology = participant_semiology(participant, semiology_manifest_csv);
    else
        D.semiology = semiology;
    end
end

function [event_type, semiology] = row_event_metadata(HManifest, manifest_csv)
    event_type = strings(height(HManifest), 1); semiology = event_type;
    if isempty(char(string(manifest_csv))) || ~isfile(manifest_csv) || ...
            ~ismember('file_path', HManifest.Properties.VariableNames), return; end
    M = readtable(manifest_csv, 'TextType', 'string');
    names = lower(string(M.Properties.VariableNames));
    fcol = find(names == "file_path", 1);
    ecol = find(ismember(names, ["seizure_type", "event_seizure_type"]), 1);
    scol = find(ismember(names, ["hypo_hyper", "semiology"]), 1);
    if isempty(fcol), return; end
    source_path = lower(replace(string(M{:, fcol}), '/', '\'));
    target_path = lower(replace(string(HManifest.file_path), '/', '\'));
    [found, location] = ismember(target_path, source_path);
    if ~isempty(ecol), event_type(found) = upper(strtrim(string(M{location(found), ecol}))); end
    if ~isempty(scol), semiology(found) = lower(strtrim(string(M{location(found), scol}))); end
    event_type(event_type == "FS") = "NEAD";
    event_type(~ismember(event_type, ["NEAD", "ES"])) = "";
    semiology(~ismember(semiology, ["hypo", "hyper"])) = "";
end

function semiology = participant_semiology(participants, manifest_csv)
    semiology = strings(numel(participants), 1);
    if isempty(char(string(manifest_csv))) || ~isfile(manifest_csv), return; end
    M = readtable(manifest_csv, 'TextType', 'string');
    names = lower(string(M.Properties.VariableNames));
    pcol = find(names == "participant", 1);
    ccol = find(names == "condition", 1);
    scol = find(ismember(names, ["hypo_hyper", "semiology"]), 1);
    if isempty(pcol) || isempty(ccol) || isempty(scol), return; end
    p = strtrim(string(M{:, pcol}));
    condition = lower(strtrim(string(M{:, ccol})));
    label = lower(strtrim(string(M{:, scol})));
    valid_condition = ismember(condition, ["preict", "preictal"]);
    for i = 1:numel(participants)
        values = unique(label(p == participants(i) & valid_condition & ismember(label, ["hypo", "hyper"])));
        if numel(values) == 1, semiology(i) = values; end
    end
end

function X = normalise_rows(X)
    X=X-mean(X,2,'omitnan'); d=sqrt(sum(X.^2,2));
    d(~isfinite(d)|d<=eps)=1; X=X./d;
end
