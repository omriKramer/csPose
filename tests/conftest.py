import pytest


def pytest_addoption(parser):
    parser.addoption("--lip-root", default="/Volumes/waic/omrik/LIP")


@pytest.fixture
def data_path(request):
    return request.config.getoption("--data-path")
