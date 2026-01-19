from src.feature_eng import create_features
import pandas as pd

def test_create_features():
    df = pd.DataFrame({"a":[1,2],"b":[3,4]})
    df_feat = create_features(df)
    assert "sum_feature" in df_feat.columns