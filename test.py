import pandas as pd
import numpy as np
import tensorflow as tf

tf.random.set_seed(42)
def create_sample_dataframe():
    data = {
        'A': np.random.randint(0, 100, size=10),
        'B': np.random.rand(10),
        'C': ['foo', 'bar', 'baz', 'qux', 'quux', 'corge', 'grault', 'garply', 'waldo', 'fred']
    }
    df = pd.DataFrame(data)
    return df

def test_create_sample_dataframe():
    df = create_sample_dataframe()
    assert isinstance(df, pd.DataFrame), "Output is not a DataFrame"
    assert df.shape == (10, 3), "DataFrame shape is incorrect"
    assert all(col in df.columns for col in ['A', 'B', 'C']), "DataFrame columns are incorrect"


# test tensorflow functionality
def create_sample_tensor():
    tensor = tf.random.uniform((5, 5), minval=0, maxval=10, dtype=tf.int32)
    return tensor

def test_create_sample_tensor():
    tensor = create_sample_tensor()
    assert isinstance(tensor, tf.Tensor), "Output is not a Tensor"
    assert tensor.shape == (5, 5), "Tensor shape is incorrect"
    assert tensor.dtype == tf.int32, "Tensor dtype is incorrect"

if __name__ == "__main__":
    test_create_sample_dataframe()
    test_create_sample_tensor()
    print("All tests passed.")