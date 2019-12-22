import pytest


def pytest_addoption(parser):
    parser.addoption("--data-path", default="../coco/dev")


@pytest.fixture
def data_path(request):
    return request.config.getoption("--data-path")
