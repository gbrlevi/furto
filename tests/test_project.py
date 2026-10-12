import tcc


def test_package_is_importable() -> None:
    assert tcc.__name__ == "tcc"
