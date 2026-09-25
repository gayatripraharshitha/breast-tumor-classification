from src.preprocessing import encode_target


def test_encode_target():
    target = [2, 4, 2, 4]

    encoded = encode_target(
        __import__("pandas").Series(target)
    )

    assert encoded.tolist() == [0, 1, 0, 1]