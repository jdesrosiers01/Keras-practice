from keras.models import Sequential
import numpy as np

model = Sequential.from_config(
    [
        {
            'class_name': 'Dense',
            'config': {
                'W_constraint': None,
                'b_constraint': None,
                'name': 'dense_1',
                'output_dim': 30,
                'activity_regularizer': None,
                'trainable': True,
                'init': 'glorot_uniform',
                'bias': True,
                'input_dtype': 'float32',
                'input_dim': 15,
                'b_regularizer': None,
                'W_regularizer': None,
                'activation': 'linear',
                'batch_input_shape': (None, 15)
            }
        },
        {
            'class_name': 'Activation',
            'config': {
                'activation': 'tanh',
                'trainable': True,
                'name': 'activation_1'
            }
        },
        {
            'class_name': 'Dense',
            'config': {
                'W_constraint': None,
                'b_constraint': None,
                'name': 'dense_2',
                'activity_regularizer': None,
                'trainable': True,
                'init': 'glorot_uniform',
                'bias': True,
                'input_dim': 30, 'b_regularizer': None,
                'W_regularizer': None,
                'activation': 'linear',
                'output_dim': 1
            }
        },
        {
            'class_name': 'Activation',
            'config': {
                'activation': 'sigmoid',
                'trainable': True,
                'name': 'activation_2'
            }
        }
    ]
)

model.load_weights('./learnedModel')

results = model.predict_proba(np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]
))

print(results)
