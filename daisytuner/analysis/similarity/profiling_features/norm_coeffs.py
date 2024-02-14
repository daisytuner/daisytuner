# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import numpy as np

ARCHS_MEAN_CPU = np.array(
    [
        1.4018419,
        2.1090446e01,
        2.0000000,
        7.0618506e02,
        2.3261250e04,
        1.0911565e05,
        7.8630862e05,
        9.8121695e04,
        4.4188637e04,
        6.1363500e04,
        7.3410109e04,
    ]
)
ARCHS_STD_CPU = np.array(
    [
        1.0217322e00,
        1.7427402e01,
        0.0000000e00,
        3.2877734e02,
        9.1927510e03,
        5.8284828e04,
        4.9223181e05,
        6.8787961e04,
        3.2695980e04,
        4.4761520e04,
        5.3269879e04,
    ]
)
COUNTERS_MEAN_CPU = np.array(
    [
        2.2074378e08,
        2.6036586e08,
        1.1544925e10,
        2.4003282e08,
        3.6360972e06,
        3.8392535e06,
        1.0537522e08,
        3.6811998e06,
        2.8878557e01,
        4.6488846e01,
        2.0689988e03,
        3.0420179e01,
        6.1514359e04,
        6.3835113e04,
        2.4061290e06,
        6.2740844e04,
        0.0000000e00,
        0.0000000e00,
        0.0000000e00,
        0.0000000e00,
        5.2468155e06,
        5.3587640e06,
        2.3070851e08,
        5.3151875e06,
        1.0364780e02,
        2.0729559e02,
        5.9079243e03,
        1.6410901e02,
        2.8573781e05,
        3.0481338e05,
        1.0976434e07,
        2.9430103e05,
        0.0000000e00,
        0.0000000e00,
        0.0000000e00,
        0.0000000e00,
        4.4544268e07,
        5.4632684e07,
        2.5121267e09,
        4.9517792e07,
        1.1998625e05,
        1.5044244e05,
        4.5696910e06,
        1.2880972e05,
        0.0000000e00,
        1.5024505e07,
        2.0373896e07,
        6.7977831e05,
        1.1725982e06,
        6.2033250e06,
        6.5892900e07,
        1.8853186e06,
        3.1061860e06,
        3.4761835e06,
        1.0063043e08,
        3.2568405e06,
        6.3986468e07,
        7.6190824e07,
        3.2508365e09,
        6.9852400e07,
        1.5596405e07,
        1.8069276e07,
        7.4328698e08,
        1.6783624e07,
    ]
)
COUNTERS_STD_CPU = np.array(
    [
        3.90842342e09,
        4.54517504e09,
        3.23779494e11,
        4.11123558e09,
        1.27223656e08,
        1.29541096e08,
        1.77607654e09,
        1.27922992e08,
        2.35702905e03,
        2.94700854e03,
        1.64390938e05,
        2.37002783e03,
        1.48757525e06,
        1.49665662e06,
        5.48738160e07,
        1.49142750e06,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        1.41625904e08,
        1.42025040e08,
        6.91525427e09,
        1.41946480e08,
        1.19775625e04,
        2.39551250e04,
        6.82721062e05,
        1.89644746e04,
        2.08438080e07,
        2.27194400e07,
        7.80819328e08,
        2.16771660e07,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        9.13009600e08,
        1.08366950e09,
        7.82938440e10,
        9.72121664e08,
        1.68289238e06,
        2.68745450e06,
        8.38955360e07,
        1.77275962e06,
        0.00000000e00,
        1.71716624e08,
        5.90585408e08,
        6.44748050e06,
        1.76951460e07,
        5.49561760e07,
        9.20650560e08,
        2.09314280e07,
        6.64752920e07,
        6.92478000e07,
        1.51754816e09,
        6.77567440e07,
        1.09121395e09,
        1.27162074e09,
        8.80655647e10,
        1.15268915e09,
        2.68967872e08,
        3.06897984e08,
        1.98714880e10,
        2.81724992e08,
    ]
)
TARGETS_MEAN_CPU = np.array(
    [
        2.0379238e-01,
        2.3236525e01,
        7.8289051e00,
        1.7506173e-01,
        4.1312058e-03,
        1.9962501e03,
        2.5812896e03,
        1.9355975e04,
        1.9790768e04,
        3.6081062e04,
        6.7856922e04,
        6.6194891e04,
        1.2313299e05,
        1.5550522e-02,
        2.8492892e-01,
        2.2077280e-01,
        1.5039954e-01,
    ]
)
TARGETS_STD_CPU = np.array(
    [
        3.48962045e00,
        2.99652252e01,
        1.72033424e01,
        3.13294083e-01,
        2.12756582e-02,
        1.24071904e04,
        1.84296055e04,
        2.61430020e04,
        3.00859336e04,
        1.03582336e05,
        1.35255969e05,
        2.73128344e05,
        3.20384312e05,
        4.20848019e-02,
        3.02885979e-01,
        3.12015325e-01,
        1.20083630e-01,
    ]
)

##### GPU #####

ARCHS_MEAN_GPU = np.array(
    [1.0000e00, 8.3379e00, 1.8243e01, 7.8054e03, 3.2000e01, 1.8068e03, 8.1865e03]
)
ARCHS_STD_GPU = np.array(
    [0.0000e00, 6.0190e-01, 2.1310e01, 2.8105e03, 0.0000e00, 5.9780e02, 1.6574e03]
)
COUNTERS_MEAN_GPU = np.array(
    [
        1.0799e08,
        1.0799e08,
        1.0799e08,
        1.0799e08,
        3.8212e05,
        3.8212e05,
        3.8212e05,
        3.8212e05,
        1.1133e05,
        1.1133e05,
        1.1133e05,
        1.1133e05,
        3.5786e04,
        3.5786e04,
        3.5786e04,
        3.5786e04,
        0.0000e00,
        0.0000e00,
        0.0000e00,
        0.0000e00,
        0.0000e00,
        0.0000e00,
        0.0000e00,
        0.0000e00,
        0.0000e00,
        0.0000e00,
        0.0000e00,
        0.0000e00,
        4.7849e04,
        4.7849e04,
        4.7849e04,
        4.7849e04,
        2.5635e04,
        2.5635e04,
        2.5635e04,
        2.5635e04,
        1.7247e04,
        1.7247e04,
        1.7247e04,
        1.7247e04,
        3.2370e-02,
        3.2370e-02,
        3.2370e-02,
        3.2370e-02,
        4.5782e02,
        4.5782e02,
        4.5782e02,
        4.5782e02,
        2.8330e02,
        2.8330e02,
        2.8330e02,
        2.8330e02,
        1.7451e02,
        1.7451e02,
        1.7451e02,
        1.7451e02,
        9.4037e02,
        9.4037e02,
        9.4037e02,
        9.4037e02,
        7.4827e01,
        7.4827e01,
        7.4827e01,
        7.4827e01,
        3.5868e02,
        3.5868e02,
        3.5868e02,
        3.5868e02,
    ]
)
COUNTERS_STD_GPU = np.array(
    [
        1.6134e09,
        1.6134e09,
        1.6134e09,
        1.6134e09,
        3.8526e06,
        3.8526e06,
        3.8526e06,
        3.8526e06,
        8.4027e05,
        8.4027e05,
        8.4027e05,
        8.4027e05,
        3.4170e05,
        3.4170e05,
        3.4170e05,
        3.4170e05,
        0.0000e00,
        0.0000e00,
        0.0000e00,
        0.0000e00,
        0.0000e00,
        0.0000e00,
        0.0000e00,
        0.0000e00,
        0.0000e00,
        0.0000e00,
        0.0000e00,
        0.0000e00,
        3.5028e05,
        3.5028e05,
        3.5028e05,
        3.5028e05,
        1.7599e05,
        1.7599e05,
        1.7599e05,
        1.7599e05,
        9.7009e04,
        9.7009e04,
        9.7009e04,
        9.7009e04,
        1.7708e-01,
        1.7708e-01,
        1.7708e-01,
        1.7708e-01,
        4.7097e03,
        4.7097e03,
        4.7097e03,
        4.7097e03,
        3.2189e03,
        3.2189e03,
        3.2189e03,
        3.2189e03,
        1.5717e03,
        1.5717e03,
        1.5717e03,
        1.5717e03,
        3.5200e03,
        3.5200e03,
        3.5200e03,
        3.5200e03,
        1.7537e02,
        1.7537e02,
        1.7537e02,
        1.7537e02,
        6.2308e03,
        6.2308e03,
        6.2308e03,
        6.2308e03,
    ]
)
TARGETS_MEAN_GPU = np.array(
    [
        1.3554e00,
        1.3567e-01,
        6.7546e00,
        3.0748e-02,
        2.6369e-01,
        6.0462e00,
        2.5486e-02,
        0.0000e00,
        1.0464e-01,
        7.3441e03,
        4.0947e03,
        3.2447e03,
        5.0334e-01,
        4.2912e04,
        3.7284e03,
        1.0000e00,
        1.0000e00,
        1.0000e00,
        2.7722e02,
        1.1771e02,
        5.5508e02,
        1.8883e02,
        3.6617e00,
        3.3836e00,
        1.0000e00,
        1.5800e00,
        1.1779e00,
        0.0000e00,
        5.2366e01,
        1.3726e-03,
        1.1899e-02,
        1.5084e-02,
    ]
)
TARGETS_STD_GPU = np.array(
    [
        2.4327e01,
        1.3391e-01,
        2.1714e00,
        4.6834e-02,
        3.3512e-01,
        2.2191e01,
        8.3910e-02,
        0.0000e00,
        7.2604e-01,
        1.3208e04,
        8.0741e03,
        6.8893e03,
        3.1056e-01,
        1.2345e05,
        1.2135e04,
        0.0000e00,
        0.0000e00,
        0.0000e00,
        8.3265e02,
        3.8095e02,
        1.0676e03,
        4.3825e02,
        3.1932e01,
        3.0259e01,
        0.0000e00,
        2.4597e01,
        1.3805e01,
        0.0000e00,
        1.1949e02,
        2.5791e-03,
        1.8996e-02,
        2.2346e-02,
    ]
)
