import numpy as np

from napari_opt_handler._sample_data import make_sample_data


def test_outputs():
    """
    Test the make_sample_data function.

    Parameters:
    - None

    Returns:
    - None

    Raises:
    - AssertionError: If the output array dtype or values do not match the
        expected values.
    """
    out = make_sample_data()
    assert len(out) == 4
    for i in range(4):
        assert len(out[i]) == 3
        assert isinstance(out[i][0], np.ndarray)
        assert isinstance(out[i][1], dict)
        assert isinstance(out[i][2], str)
    assert out[0][0].shape == (2048, 1536)
    assert out[1][0].shape == (2048, 1536)
    assert out[2][0].shape == (2048, 1536)
    assert out[3][0].shape == (7, 2048, 1536)
    assert out[0][2] == "Image"
    assert out[1][2] == "Image"
    assert out[2][2] == "Image"
    assert out[3][2] == "Image"
    assert out[0][1] == {"name": "bad"}
    assert out[1][1] == {"name": "dark"}
    assert out[2][1] == {"name": "bright"}
    assert out[3][1] == {"name": "OPT data"}
