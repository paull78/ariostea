import ariostea


def test_package_exposes_version():
    assert isinstance(ariostea.__version__, str)
    assert ariostea.__version__
