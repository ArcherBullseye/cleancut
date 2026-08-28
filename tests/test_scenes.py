from cleancut.scenes import Shot, shot_containing, snap_range_to_shots


def test_shot_containing():
    shots = [Shot(0, 10), Shot(10, 25), Shot(25, 40)]
    assert shot_containing(5, shots) == shots[0]
    assert shot_containing(10, shots) == shots[1]   # boundary belongs to next shot
    assert shot_containing(24.9, shots) == shots[1]
    assert shot_containing(100, shots) is None


def test_snap_range_to_shots_expands_outward():
    shots = [Shot(0, 10), Shot(10, 25), Shot(25, 40)]
    # Range [12, 20] sits entirely within shot 2 -> snaps to [10, 25].
    assert snap_range_to_shots(12, 20, shots) == (10, 25)


def test_snap_range_spanning_two_shots():
    shots = [Shot(0, 10), Shot(10, 25), Shot(25, 40)]
    # Range [8, 27] crosses shot 1, 2, 3 -> snaps to [0, 40].
    assert snap_range_to_shots(8, 27, shots) == (0, 40)


def test_snap_with_no_shots_is_passthrough():
    assert snap_range_to_shots(5, 10, []) == (5, 10)
