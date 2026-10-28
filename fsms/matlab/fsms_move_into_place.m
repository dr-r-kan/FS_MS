function fsms_move_into_place(tmp, target)
%FSMS_MOVE_INTO_PLACE Rename a completed temporary file over its final name.
    [ok, msg] = movefile(tmp, target, 'f');
    if ~ok
        if isfile(tmp), delete(tmp); end
        error('FSMS:MoveFailed', 'Could not move %s to %s: %s', tmp, target, msg);
    end
end
