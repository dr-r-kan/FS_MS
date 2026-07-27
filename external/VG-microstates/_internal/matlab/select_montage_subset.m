function [EEG_reduced, pos_reduced, chanlocs_reduced, labels_reduced, indices] = select_montage_subset(EEG, pos, chanlocs, labels, montage_type)
% SELECT_MONTAGE_SUBSET: Reduce full EEG to standard clinical montages
%
% INPUTS:
%   EEG          - Full EEG data (channels × timepoints) or maps (K × channels)
%   pos          - Channel positions (channels × 3)
%   chanlocs     - EEGLAB channel locations structure
%   labels       - Cell array of channel labels
%   montage_type - String: 'full', '10-20-20', or '10-20-12'
%
% OUTPUTS:
%   EEG_reduced      - Reduced EEG data
%   pos_reduced      - Reduced channel positions
%   chanlocs_reduced - Reduced channel locations
%   labels_reduced   - Reduced channel labels
%   indices          - Indices of selected channels in original montage
%
% MONTAGE TYPES:
%   'full'      - No reduction (returns input)
%   '10-20-20'  - Standard 10-20 system with 19 leads (excludes A1/A2):
%                 Fp1, Fp2, F7, F3, Fz, F4, F8, T7, C3, Cz, C4, T8,
%                 P7, P3, Pz, P4, P8, O1, O2
%                 (accepts T3/T4/T5/T6 as alternatives for T7/T8/P7/P8)
%   '10-20-12'  - Clinical 12-lead montage:
%                 Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, Fz, Pz
%
% NOTES:
%   - Channel matching is case-insensitive
%   - Handles alternative channel names (T3→T7, T4→T8, T5→P7, T6→P8)
%   - Automatically detects data format (EEG vs maps)
%   - Warns if <80% of requested leads are found
%   - Preserves spatial distribution of electrodes
%
% Example:
%   [EEG_20, pos_20, chanlocs_20, labels_20, idx] = ...
%       select_montage_subset(EEG, pos, chanlocs, labels, '10-20-20');

    % Validate inputs
    if nargin < 5
        error('select_montage_subset:InvalidInput', ...
            'Usage: select_montage_subset(EEG, pos, chanlocs, labels, montage_type)');
    end
    
    % Handle 'full' montage - no reduction
    if strcmpi(montage_type, 'full')
        EEG_reduced = EEG;
        pos_reduced = pos;
        chanlocs_reduced = chanlocs;
        labels_reduced = labels;
        indices = 1:length(labels);
        return;
    end
    
    % Define standard montages with alternative names
    switch lower(montage_type)
        case '10-20-20'
            % Standard 10-20 system with 20 leads
            % Note: Using modern naming (T7/T8/P7/P8) instead of old (T3/T4/T5/T6)
            % Excluding A1/A2 (reference electrodes, rarely in active montages)
            target_labels = {
                'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', ...
                'T7', 'C3', 'Cz', 'C4', 'T8', ...
                'P7', 'P3', 'Pz', 'P4', 'P8', ...
                'O1', 'O2'
            };
            % Alternative names for backward compatibility
            alternative_labels = {
                '', '', '', '', '', '', '', ...
                'T3', '', '', '', 'T4', ...
                'T5', '', '', '', 'T6', ...
                '', ''
            };
            
        case '10-20-12'
            % Clinical 12-lead montage
            target_labels = {
                'Fp1', 'Fp2', ...
                'F3', 'F4', 'Fz', ...
                'C3', 'C4', ...
                'P3', 'P4', 'Pz', ...
                'O1', 'O2'
            };
            alternative_labels = cell(1, length(target_labels));
            alternative_labels(:) = {''};  % No alternatives for 10-20-12
            
        otherwise
            error('select_montage_subset:UnknownMontage', ...
                'Unknown montage type: %s. Use ''full'', ''10-20-20'', or ''10-20-12''', ...
                montage_type);
    end
    
    % Match channels case-insensitively with alternative names
    n_channels = length(labels);
    n_target = length(target_labels);
    indices = zeros(n_target, 1);
    matched = false(n_target, 1);
    
    for i = 1:n_target
        % Try primary label first
        match_idx = find(strcmpi(labels, target_labels{i}), 1);
        
        % If no match and alternative exists, try alternative
        if isempty(match_idx) && ~isempty(alternative_labels{i})
            match_idx = find(strcmpi(labels, alternative_labels{i}), 1);
        end
        
        % Also try combined format like "T7/T3" or "T3/T7"
        if isempty(match_idx) && ~isempty(alternative_labels{i})
            combined1 = sprintf('%s/%s', target_labels{i}, alternative_labels{i});
            combined2 = sprintf('%s/%s', alternative_labels{i}, target_labels{i});
            match_idx = find(strcmpi(labels, combined1) | strcmpi(labels, combined2), 1);
        end
        
        if ~isempty(match_idx)
            indices(i) = match_idx;
            matched(i) = true;
        end
    end
    
    % Check match rate
    match_rate = sum(matched) / n_target;
    
    if match_rate < 0.8
        warning('select_montage_subset:LowMatchRate', ...
            'Only %.1f%% of %s montage leads found in template (%.0f/%d). Results may be unreliable.', ...
            match_rate * 100, montage_type, sum(matched), n_target);
        
        % List missing channels
        missing = target_labels(~matched);
        if ~isempty(missing)
            fprintf('Missing channels: %s\n', strjoin(missing, ', '));
        end
    end
    
    % Extract only matched channels
    indices = indices(matched);
    
    if isempty(indices)
        error('select_montage_subset:NoMatches', ...
            'No matching channels found for montage type: %s', montage_type);
    end
    
    % Reduce all outputs
    % Handle both EEG (channels × time) and maps (K × channels) formats
    % Maps have format (K × channels) where K is typically 2-12
    % EEG has format (channels × time) where channels is typically 12-71
    if size(EEG, 1) <= 20 && size(EEG, 2) > 20
        % Likely maps format: small first dimension, large second dimension
        % (K × channels) → select columns
        EEG_reduced = EEG(:, indices);
    else
        % Standard EEG format or square matrices
        % (channels × time) → select rows
        EEG_reduced = EEG(indices, :);
    end
    
    pos_reduced = pos(indices, :);
    labels_reduced = labels(indices);
    
    % Handle chanlocs structure
    if ~isempty(chanlocs)
        chanlocs_reduced = chanlocs(indices);
    else
        chanlocs_reduced = [];
    end
    
    % Report success
    fprintf('✓ Montage reduction: %d → %d channels (%s, %.1f%% match)\n', ...
        n_channels, length(indices), montage_type, match_rate * 100);
end
