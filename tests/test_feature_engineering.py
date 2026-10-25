import pandas as pd
import pytest

from skext.feature_engineering import RareCategoryEncoder, TargetEncoder, TimeFeatureExtractor


def test_rare_category_encoder_groups_infrequent_values():
    X = pd.DataFrame({"city": ["ny", "ny", "la", "sf", "sf", "sf", "bos"]})

    transformer = RareCategoryEncoder(min_frequency=0.25, replacement="other")
    transformed = transformer.fit_transform(X)

    assert transformed["city"].tolist() == ["ny", "ny", "other", "sf", "sf", "sf", "other"]


def test_target_encoder_uses_target_mean():
    X = pd.DataFrame({"color": ["red", "red", "blue", "blue", "green"]})
    y = pd.Series([1.0, 0.0, 1.0, 1.0, 0.0])

    transformer = TargetEncoder()
    transformed = transformer.fit_transform(X, y)

    assert transformed["color"].tolist() == [0.5, 0.5, 1.0, 1.0, 0.0]


def test_time_feature_extractor_creates_datetime_features():
    X = pd.DataFrame({
        "ts": pd.to_datetime([
            "2024-01-01 12:30:00",
            "2024-02-04 18:00:00",
        ])
    })

    transformer = TimeFeatureExtractor(features=["month", "day_of_week", "hour", "is_weekend"])
    transformed = transformer.fit_transform(X)

    assert set(transformed.columns) == {"month", "day_of_week", "hour", "is_weekend"}
    assert transformed["month"].tolist() == [1, 2]
    assert transformed["hour"].tolist() == [12, 18]
    assert transformed["is_weekend"].tolist() == [False, True]


def test_rare_category_encoder_explicit_columns_and_feature_names():
    X = pd.DataFrame({
        "group": ["a", "a", "b", "c"],
        "value": [1, 2, 3, 4],
    })

    transformer = RareCategoryEncoder(columns=["group"], min_frequency=0.5, replacement="other")
    transformed = transformer.fit_transform(X)

    assert transformed["group"].tolist() == ["a", "a", "other", "other"]
    assert transformer.get_feature_names_out().tolist() == ["group"]


def test_target_encoder_handles_unknown_and_custom_columns():
    X = pd.DataFrame({
        "color": ["red", "red", "blue", "green"],
        "score": [10, 11, 12, 13],
    })
    y = pd.Series([1.0, 0.0, 1.0, 0.0])

    transformer = TargetEncoder(columns=["color"])
    transformed = transformer.fit_transform(X, y)

    assert transformed["color"].tolist() == [0.5, 0.5, 1.0, 0.0]
    assert transformer.get_feature_names_out().tolist() == ["color"]

    unknown = pd.DataFrame({"color": ["purple"], "score": [99]})
    result = transformer.transform(unknown)
    assert result["color"].iloc[0] == 0.5


def test_time_feature_extractor_supports_custom_columns_and_errors():
    X = pd.DataFrame({
        "ts": pd.to_datetime(["2024-01-01 12:30:00", "2024-01-07 18:45:00"]),
        "other": [1, 2],
    })

    transformer = TimeFeatureExtractor(features=["month", "day_of_week"], columns=["ts"])
    transformed = transformer.fit_transform(X)

    assert transformed.columns.tolist() == ["month", "day_of_week"]
    assert transformed["month"].tolist() == [1, 1]
    assert transformed["day_of_week"].tolist() == [0, 6]
    assert transformer.get_feature_names_out().tolist() == ["month", "day_of_week"]

    with pytest.raises(ValueError, match="Unsupported time features"):
        TimeFeatureExtractor(features=["not_a_real_feature"]).fit(X)

    with pytest.raises(ValueError, match="Transformer has not been fitted"):
        TimeFeatureExtractor().transform(X)

    with pytest.raises(ValueError, match="Column 'missing' is not present in X"):
        TimeFeatureExtractor(columns=["missing"]).fit(X).transform(X)


def test_rare_category_encoder_and_target_encoder_error_paths():
    X = pd.DataFrame({"city": ["ny", "la", "ny"], "value": [1, 2, 3]})
    y = pd.Series([1.0, 0.0, 1.0])

    rare = RareCategoryEncoder(columns=["city"]).fit(X)
    assert rare.transform(X).columns.tolist() == ["city", "value"]

    with pytest.raises(ValueError, match="Column 'missing' is not present in X"):
        RareCategoryEncoder(columns=["missing"]).fit(X).transform(X)

    with pytest.raises(ValueError, match="Encoder has not been fitted"):
        TargetEncoder(columns=["city"]).transform(X)

    target = TargetEncoder(columns=["city"]).fit(X, y)
    assert target.transform(X)["city"].tolist() == [1.0, 0.0, 1.0]
