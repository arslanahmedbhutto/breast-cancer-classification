from src.data import FEATURES, load_data


def test_load_data_shape_and_labels():
    X, y = load_data()
    assert X.shape == (569, 30)
    assert list(X.columns) == FEATURES
    assert y.value_counts().to_dict() == {0: 357, 1: 212}
    assert not X.isna().any().any()
