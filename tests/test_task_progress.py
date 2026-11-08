from probemap.registration.desktop_task_progress import TaskTiming, duration_text


def test_eta_waits_for_measured_progress_and_detects_a_stalled_source():
    now = [0.]
    timer = TaskTiming(clock=lambda: now[0]); timer.start()
    now[0] = 12
    assert timer.elapsed() == 12 and timer.remaining() is None
    timer.report(20)
    assert timer.remaining() == 48
    now[0] = 24; timer.report(40)
    assert timer.remaining() == 36
    now[0] = 50
    assert timer.remaining() == 62 and timer.waiting()
    timer.report(100)
    assert timer.remaining() == 0


def test_chained_stages_keep_elapsed_time_but_reset_the_progress_estimate():
    now = [0.]
    timer = TaskTiming(clock=lambda: now[0]); timer.start()
    now[0] = 60; timer.report(100)
    timer.start(continuing=True)
    assert timer.elapsed() == 60 and timer.value == 0 and timer.remaining() is None
    now[0] = 65; timer.report(50)
    assert timer.elapsed() == 65 and timer.remaining() == 5
    assert duration_text(65) == '01:05'
    assert duration_text(3665) == '1:01:05'


def test_slow_network_keeps_eta_between_sparse_integer_percent_updates():
    now = [0.]
    timer = TaskTiming(clock=lambda: now[0]); timer.start()
    now[0] = 120; timer.report(1)
    now[0] = 240; timer.report(2)
    now[0] = 300
    assert timer.remaining() == 11820  # measured 120 seconds per percent plus waiting
    assert timer.waiting()
    timer.report(2)  # bytes advanced within the same displayed percentage
    assert not timer.waiting() and timer.remaining() == 11820
    now[0] = 360; timer.report(3)
    assert timer.remaining() == 11640


def test_eta_uses_byte_progress_before_first_whole_percent():
    now = [0.]
    timer = TaskTiming(clock=lambda: now[0]); timer.start()
    now[0] = 10; timer.report(.25)
    assert timer.value == .25 and timer.remaining() == 3990
    now[0] = 20; timer.report(.5)
    assert timer.remaining() == 3980
