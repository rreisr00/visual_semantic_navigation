from semantic_navigation_core.capture_state_machine import (
    CaptureState,
    CaptureStateMachine,
)


class TestHappyPath:
    def test_full_sequence_reaches_done(self):
        sm = CaptureStateMachine()
        assert sm.start(has_image=True, has_pose=True) == CaptureState.WAITING_FEATURES
        assert sm.features_ok() == CaptureState.WAITING_STORE
        assert sm.store_ok() == CaptureState.DONE
        assert sm.succeeded
        assert sm.is_terminal


class TestStartGuards:
    def test_missing_image_fails(self):
        sm = CaptureStateMachine()
        assert sm.start(has_image=False, has_pose=True) == CaptureState.FAILED
        assert "image" in sm.reason

    def test_missing_pose_fails(self):
        sm = CaptureStateMachine()
        assert sm.start(has_image=True, has_pose=False) == CaptureState.FAILED
        assert "pose" in sm.reason


class TestPartialFailure:
    def test_features_failure_is_terminal(self):
        sm = CaptureStateMachine()
        sm.start(has_image=True, has_pose=True)
        assert sm.features_failed("CUDA OOM") == CaptureState.FAILED
        assert not sm.succeeded
        assert "CUDA OOM" in sm.reason

    def test_store_failure_after_encode(self):
        sm = CaptureStateMachine()
        sm.start(has_image=True, has_pose=True)
        sm.features_ok()
        assert sm.store_failed("db locked") == CaptureState.FAILED
        assert "db locked" in sm.reason


class TestOutOfOrderGuards:
    def test_features_ok_before_start_fails(self):
        sm = CaptureStateMachine()
        assert sm.features_ok() == CaptureState.FAILED

    def test_store_ok_before_features_fails(self):
        sm = CaptureStateMachine()
        sm.start(has_image=True, has_pose=True)
        assert sm.store_ok() == CaptureState.FAILED


class TestCancel:
    def test_cancel_mid_flight(self):
        sm = CaptureStateMachine()
        sm.start(has_image=True, has_pose=True)
        assert sm.cancel() == CaptureState.FAILED
        assert sm.reason == "cancelled"

    def test_cancel_after_done_is_noop(self):
        sm = CaptureStateMachine()
        sm.start(has_image=True, has_pose=True)
        sm.features_ok()
        sm.store_ok()
        assert sm.cancel() == CaptureState.DONE
