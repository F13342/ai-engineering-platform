from ai_platform.core.application import Application


def test_application_can_be_created() -> None:
    application = Application()

    assert application is not None
