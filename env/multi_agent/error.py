Failure # 1 (occurred at 2023-08-31_21-54-22)
The actor died because of an error raised in its creation task, [36mray::AIRRLTrainer.__init__()[39m (pid=5279, ip=127.0.0.1, actor_id=82cd4da7b4edf064d8b847f801000000, repr=AIRA2C)
  File "/Users/floriankockler/anaconda3/envs/py310/lib/python3.10/site-packages/ray/rllib/evaluation/worker_set.py", line 227, in _setup
    self.add_workers(
  File "/Users/floriankockler/anaconda3/envs/py310/lib/python3.10/site-packages/ray/rllib/evaluation/worker_set.py", line 593, in add_workers
    raise result.get()
  File "/Users/floriankockler/anaconda3/envs/py310/lib/python3.10/site-packages/ray/rllib/utils/actor_manager.py", line 481, in __fetch_result
    result = ray.get(r)
ray.exceptions.RayActorError: The actor died because of an error raised in its creation task, [36mray::RolloutWorker.__init__()[39m (pid=5285, ip=127.0.0.1, actor_id=36e1efa3be3a8440c5c1371801000000, repr=<ray.rllib.evaluation.rollout_worker.RolloutWorker object at 0x158a63a60>)
  File "/Users/floriankockler/anaconda3/envs/py310/lib/python3.10/site-packages/ray/rllib/utils/pre_checks/env.py", line 338, in check_multiagent_environments
    raise ValueError(error)
ValueError: The observation collected from env.reset was not contained within your env's observation space. Its possible that there was a typemismatch (for example observations of np.float32 and a space ofnp.float64 observations), or that one of the sub-observations wasout of bounds

 reset_obs: {'worker': (array([ 1.0000000e+06,  8.2849997e-01,  0.0000000e+00,  6.6008695e+06,
        8.0900002e-01,  7.5940001e-01,  7.2000003e-01,  6.8720001e-01,
        7.8160000e-01,  6.3639500e+01,  1.0000000e-03,  2.6000000e-03,
        3.2121200e-02,  3.5865200e+01,  3.8922901e+01,  1.7600000e-02,
        1.4414001e+00,  6.4513130e+06,  1.9200000e-01,  0.0000000e+00,
        2.7120000e+06,  2.0250000e-01,  1.7739999e-01,  1.4830001e-01,
        1.2500000e-01,  1.8660000e-01,  5.1070801e+01,  3.0000001e-04,
        1.1000000e-03,  1.6757365e-02,  1.3611200e+01,  3.7022499e+01,
        7.7999998e-03,  5.7950002e-01,  8.8800000e+05,  2.1572001e+00,
        0.0000000e+00,  1.4716704e+06,  2.1882999e+00,  2.0676000e+00,
        1.9876000e+00,  1.8763000e+00,  2.1280999e+00,  5.2394699e+01,
        3.1999999e-03,  1.3700000e-02,  5.8082845e-02,  1.7067101e+01,
        3.7153400e+01,  2.7100001e-02,  9.8750001e-01,  6.1828000e+05,
        2.0991001e+00,  0.0000000e+00,  2.8819525e+06,  2.1087000e+00,
        1.9918000e+00,  1.9024000e+00,  1.8837000e+00,  2.0509000e+00,
        5.6803101e+01,  2.3000001e-03,  1.5400000e-02,  4.9362659e-02,
        9.9177999e+00,  3.5919998e+01,  3.0400001e-02,  1.0458000e+00,
        1.3305640e+06,  1.5995600e+01,  0.0000000e+00,  3.3783200e+05,
        1.6270300e+01,  1.4810900e+01,  1.4907200e+01,  1.5356400e+01,
        1.5259000e+01,  5.8494099e+01,  1.0854000e+00,  8.8209999e-01,
        1.0370754e+00,  1.7062000e+00,  2.6076700e+01,  4.2010000e-01,
        8.0390000e-01,  1.8950000e+05,  1.4890000e+00,  0.0000000e+00,
        8.9391680e+06,  1.5023000e+00,  1.4270999e+00,  1.3663000e+00,
        1.3285000e+00,  1.4598000e+00,  5.3890499e+01,  2.9000000e-03,
        6.5000001e-03,  5.3075951e-02,  1.7767700e+01,  3.5822601e+01,
        2.3100000e-02,  1.0783000e+00,  3.3312000e+06,  2.3942001e+00,
        0.0000000e+00,  2.0565440e+06,  2.3724000e+00,  2.2061999e+00,
        2.0425000e+00,  1.9007000e+00,  2.2830999e+00,  6.2822701e+01,
        7.0000002e-03,  3.7300002e-02,  8.6248346e-02,  3.7857399e+01,
        3.9324501e+01,  5.5300001e-02,  1.8139000e+00,  1.7568000e+06,
        4.0239999e-01,  0.0000000e+00,  3.3790720e+06,  3.9080000e-01,
        3.8460001e-01,  3.6340001e-01,  3.2749999e-01,  3.9100000e-01,
        5.9131599e+01,  9.9999997e-05,  6.9999998e-04,  9.6539799e-03,
        1.8160801e+01,  2.6746500e+01,  2.7999999e-03,  9.2369998e-01,
        3.3664000e+06,  1.1337000e+00,  0.0000000e+00,  6.6739320e+06,
        1.1091000e+00,  1.0306000e+00,  9.7060001e-01,  9.2140001e-01,
        1.0681000e+00,  7.0823601e+01,  1.5000000e-03,  6.2000002e-03,
        4.0974230e-02,  4.8390800e+01,  5.0257999e+01,  2.5300000e-02,
        1.7750000e+00,  8.6359390e+06,  7.1780002e-01,  0.0000000e+00,
        1.2221424e+07,  7.3320001e-01,  7.2070003e-01,  6.9760001e-01,
        6.8730003e-01,  7.3839998e-01,  4.2753799e+01,  5.0000002e-04,
        1.6000000e-03,  2.2213263e-02,  2.3009100e+01,  3.1001400e+01,
       -2.4000001e-03,  1.3231000e+00,  8.9400000e+06,  4.0570000e-01,
        0.0000000e+00,  5.4134400e+05,  4.0070000e-01,  3.9219999e-01,
        3.7090001e-01,  3.5780001e-01,  3.9780000e-01,  5.6426800e+01,
        0.0000000e+00,  6.9999998e-04,  6.4477473e-03,  1.6508200e+01,
        1.6720800e+01,  3.4000000e-03,  5.4119998e-01,  2.0640000e+05,
        2.7255001e+00,  0.0000000e+00,  1.4056536e+05,  2.7816999e+00,
        2.8318000e+00,  2.8761001e+00,  2.6111000e+00,  2.8545001e+00,
        4.1289299e+01,  1.9099999e-02,  2.0600000e-02,  1.3748071e-01,
        1.2211100e+01,  1.5229400e+01, -3.9600000e-02,  4.4400001e-01,
        1.1036200e+05,  2.4990000e-01,  0.0000000e+00,  1.2128000e+06,
        2.5119999e-01,  2.4810000e-01,  2.4590001e-01,  2.4640000e-01,
        2.4710000e-01,  5.1323502e+01,  0.0000000e+00,  1.9999999e-04,
        6.7977011e-03,  1.0141300e+01,  1.6625999e+01,  2.4999999e-03,
        3.0950001e-01,  2.1408000e+06,  0.0000000e+00,  0.0000000e+00,
        1.8114200e+05,  0.0000000e+00], dtype=float32), {}), 'manager': (array([66.], dtype=float32), {})}

 env.observation_space_sample(): {'worker': array([-0.20289347,  0.11215498,  0.09123678, -0.47268885, -0.5170849 ,
       -0.03265719,  1.025571  ,  0.29880205,  0.27762076, -1.54744   ,
       -0.47113052,  1.3880175 ,  0.11046211, -0.22518797, -0.61193943,
       -0.6325778 ,  0.8941513 , -1.3140239 , -0.4416165 ,  0.8855457 ,
        1.9597298 ,  1.247972  , -1.0848751 ,  1.1544305 , -0.75664365,
        1.3746328 ,  0.19960149, -0.57200277,  1.4024386 ,  1.3484436 ,
        0.2928772 , -0.8452085 , -2.408678  , -0.8820077 ,  0.98478115,
        0.46658143, -1.9329342 ,  1.0455369 ,  0.2963286 ,  1.3098681 ,
       -0.8403922 ,  1.2858739 ,  0.13813981, -0.37492937,  0.94566995,
       -0.923609  ,  0.17142965,  0.46283364,  2.5075457 , -0.3266498 ,
       -2.0951962 ,  0.52249324, -0.69209397, -0.6390537 , -0.1911823 ,
        0.7833148 ,  1.2130321 , -0.3903246 , -0.57595086, -1.7115734 ,
       -0.35662797,  0.03563855,  0.11664727,  0.7075007 ,  0.9967628 ,
        0.6618875 , -0.31348   , -0.99213207, -0.10415779, -0.94414777,
       -1.103353  , -1.4805537 , -1.4806898 , -1.9195913 ,  0.68176794,
       -1.701665  , -1.524638  ,  0.5038316 , -0.9818233 , -0.12863109,
        1.1921574 , -0.95900047,  1.191072  ,  2.3432536 , -0.89479387,
       -0.44318518,  0.99862444, -0.41647112,  1.7736892 , -1.3457669 ,
       -1.4682574 ,  0.8403623 ,  0.36548048,  0.1995969 ,  0.1717999 ,
        0.252014  ,  0.22803083,  0.11720931,  0.45130366,  1.0384587 ,
        1.3070786 ,  0.8777872 , -1.6547159 , -1.6848826 ,  0.43917903,
       -0.12891226,  0.14131844, -1.5563654 ,  1.8166871 ,  1.2791089 ,
        0.13444622,  1.336699  ,  0.78228426, -0.5616967 , -1.2872282 ,
       -0.5978215 ,  1.9469616 ,  2.060816  ,  1.325076  ,  1.4470675 ,
       -0.50299275,  0.42577764,  0.00464835, -1.7581351 , -0.09081732,
        0.36288866, -0.51137847,  0.15255326, -0.4179586 , -0.12787189,
        2.1442575 , -0.712688  ,  0.32242692, -0.7813733 , -0.6430479 ,
        1.0392716 , -1.2413206 , -0.14977813,  0.9266303 ,  0.36401156,
        0.21835652,  1.4666615 ,  0.767971  ,  0.5893219 ,  0.32377478,
        2.0442357 ,  1.6362653 , -0.5781356 , -1.9579936 ,  1.4419864 ,
       -0.44825733, -2.4178603 ,  0.108367  ,  0.28814316, -0.02632944,
        1.3431495 ,  1.4702865 ,  0.1804034 , -0.9136138 ,  1.2336341 ,
        0.0894188 ,  1.9531444 ,  0.16007975, -0.58052456, -1.4881477 ,
        0.22176485,  1.1730281 , -0.96186006, -0.24914971,  0.9190409 ,
        0.20533632, -1.2440946 , -1.2401365 , -3.4045808 ,  0.7232453 ,
       -0.10568149,  0.15005219, -0.01681262,  0.801422  , -0.14774959,
        0.5647222 ,  2.6449168 ,  0.47841895, -0.9746622 ,  0.8087209 ,
       -0.21483089, -0.43108413, -0.27050295, -1.2235771 ,  0.6889069 ,
        1.1268495 , -1.0787399 , -1.2907518 ,  0.64510506,  0.5676752 ,
        1.2233466 , -0.5769342 , -0.27809045, -1.3952928 ,  0.28910315,
       -0.59807   , -1.0306895 ,  0.46862534,  0.98465633,  0.36538604,
       -2.016394  , -0.32455796,  0.38752025, -1.2624243 ,  0.4091282 ,
       -1.7375559 , -0.6182222 , -0.5046965 , -1.0100905 ,  0.9497864 ,
       -1.2108189 , -1.1647646 ,  0.94855094, -0.5343272 , -0.36582062,
       -0.8830123 ,  0.89345014,  0.32460436, -0.8566404 , -1.211331  ,
       -0.2098477 ], dtype=float32), 'manager': array([0.04394142], dtype=float32)}

 

During handling of the above exception, another exception occurred:

[36mray::RolloutWorker.__init__()[39m (pid=5285, ip=127.0.0.1, actor_id=36e1efa3be3a8440c5c1371801000000, repr=<ray.rllib.evaluation.rollout_worker.RolloutWorker object at 0x158a63a60>)
  File "/Users/floriankockler/anaconda3/envs/py310/lib/python3.10/site-packages/ray/rllib/evaluation/rollout_worker.py", line 404, in __init__
    check_env(self.env, self.config)
  File "/Users/floriankockler/anaconda3/envs/py310/lib/python3.10/site-packages/ray/rllib/utils/pre_checks/env.py", line 96, in check_env
    raise ValueError(
ValueError: Traceback (most recent call last):
  File "/Users/floriankockler/anaconda3/envs/py310/lib/python3.10/site-packages/ray/rllib/utils/pre_checks/env.py", line 81, in check_env
    check_multiagent_environments(env)
  File "/Users/floriankockler/anaconda3/envs/py310/lib/python3.10/site-packages/ray/rllib/utils/pre_checks/env.py", line 338, in check_multiagent_environments
    raise ValueError(error)
ValueError: The observation collected from env.reset was not contained within your env's observation space. Its possible that there was a typemismatch (for example observations of np.float32 and a space ofnp.float64 observations), or that one of the sub-observations wasout of bounds

 reset_obs: {'worker': (array([ 1.0000000e+06,  8.2849997e-01,  0.0000000e+00,  6.6008695e+06,
        8.0900002e-01,  7.5940001e-01,  7.2000003e-01,  6.8720001e-01,
        7.8160000e-01,  6.3639500e+01,  1.0000000e-03,  2.6000000e-03,
        3.2121200e-02,  3.5865200e+01,  3.8922901e+01,  1.7600000e-02,
        1.4414001e+00,  6.4513130e+06,  1.9200000e-01,  0.0000000e+00,
        2.7120000e+06,  2.0250000e-01,  1.7739999e-01,  1.4830001e-01,
        1.2500000e-01,  1.8660000e-01,  5.1070801e+01,  3.0000001e-04,
        1.1000000e-03,  1.6757365e-02,  1.3611200e+01,  3.7022499e+01,
        7.7999998e-03,  5.7950002e-01,  8.8800000e+05,  2.1572001e+00,
        0.0000000e+00,  1.4716704e+06,  2.1882999e+00,  2.0676000e+00,
        1.9876000e+00,  1.8763000e+00,  2.1280999e+00,  5.2394699e+01,
        3.1999999e-03,  1.3700000e-02,  5.8082845e-02,  1.7067101e+01,
        3.7153400e+01,  2.7100001e-02,  9.8750001e-01,  6.1828000e+05,
        2.0991001e+00,  0.0000000e+00,  2.8819525e+06,  2.1087000e+00,
        1.9918000e+00,  1.9024000e+00,  1.8837000e+00,  2.0509000e+00,
        5.6803101e+01,  2.3000001e-03,  1.5400000e-02,  4.9362659e-02,
        9.9177999e+00,  3.5919998e+01,  3.0400001e-02,  1.0458000e+00,
        1.3305640e+06,  1.5995600e+01,  0.0000000e+00,  3.3783200e+05,
        1.6270300e+01,  1.4810900e+01,  1.4907200e+01,  1.5356400e+01,
        1.5259000e+01,  5.8494099e+01,  1.0854000e+00,  8.8209999e-01,
        1.0370754e+00,  1.7062000e+00,  2.6076700e+01,  4.2010000e-01,
        8.0390000e-01,  1.8950000e+05,  1.4890000e+00,  0.0000000e+00,
        8.9391680e+06,  1.5023000e+00,  1.4270999e+00,  1.3663000e+00,
        1.3285000e+00,  1.4598000e+00,  5.3890499e+01,  2.9000000e-03,
        6.5000001e-03,  5.3075951e-02,  1.7767700e+01,  3.5822601e+01,
        2.3100000e-02,  1.0783000e+00,  3.3312000e+06,  2.3942001e+00,
        0.0000000e+00,  2.0565440e+06,  2.3724000e+00,  2.2061999e+00,
        2.0425000e+00,  1.9007000e+00,  2.2830999e+00,  6.2822701e+01,
        7.0000002e-03,  3.7300002e-02,  8.6248346e-02,  3.7857399e+01,
        3.9324501e+01,  5.5300001e-02,  1.8139000e+00,  1.7568000e+06,
        4.0239999e-01,  0.0000000e+00,  3.3790720e+06,  3.9080000e-01,
        3.8460001e-01,  3.6340001e-01,  3.2749999e-01,  3.9100000e-01,
        5.9131599e+01,  9.9999997e-05,  6.9999998e-04,  9.6539799e-03,
        1.8160801e+01,  2.6746500e+01,  2.7999999e-03,  9.2369998e-01,
        3.3664000e+06,  1.1337000e+00,  0.0000000e+00,  6.6739320e+06,
        1.1091000e+00,  1.0306000e+00,  9.7060001e-01,  9.2140001e-01,
        1.0681000e+00,  7.0823601e+01,  1.5000000e-03,  6.2000002e-03,
        4.0974230e-02,  4.8390800e+01,  5.0257999e+01,  2.5300000e-02,
        1.7750000e+00,  8.6359390e+06,  7.1780002e-01,  0.0000000e+00,
        1.2221424e+07,  7.3320001e-01,  7.2070003e-01,  6.9760001e-01,
        6.8730003e-01,  7.3839998e-01,  4.2753799e+01,  5.0000002e-04,
        1.6000000e-03,  2.2213263e-02,  2.3009100e+01,  3.1001400e+01,
       -2.4000001e-03,  1.3231000e+00,  8.9400000e+06,  4.0570000e-01,
        0.0000000e+00,  5.4134400e+05,  4.0070000e-01,  3.9219999e-01,
        3.7090001e-01,  3.5780001e-01,  3.9780000e-01,  5.6426800e+01,
        0.0000000e+00,  6.9999998e-04,  6.4477473e-03,  1.6508200e+01,
        1.6720800e+01,  3.4000000e-03,  5.4119998e-01,  2.0640000e+05,
        2.7255001e+00,  0.0000000e+00,  1.4056536e+05,  2.7816999e+00,
        2.8318000e+00,  2.8761001e+00,  2.6111000e+00,  2.8545001e+00,
        4.1289299e+01,  1.9099999e-02,  2.0600000e-02,  1.3748071e-01,
        1.2211100e+01,  1.5229400e+01, -3.9600000e-02,  4.4400001e-01,
        1.1036200e+05,  2.4990000e-01,  0.0000000e+00,  1.2128000e+06,
        2.5119999e-01,  2.4810000e-01,  2.4590001e-01,  2.4640000e-01,
        2.4710000e-01,  5.1323502e+01,  0.0000000e+00,  1.9999999e-04,
        6.7977011e-03,  1.0141300e+01,  1.6625999e+01,  2.4999999e-03,
        3.0950001e-01,  2.1408000e+06,  0.0000000e+00,  0.0000000e+00,
        1.8114200e+05,  0.0000000e+00], dtype=float32), {}), 'manager': (array([66.], dtype=float32), {})}

 env.observation_space_sample(): {'worker': array([-0.20289347,  0.11215498,  0.09123678, -0.47268885, -0.5170849 ,
       -0.03265719,  1.025571  ,  0.29880205,  0.27762076, -1.54744   ,
       -0.47113052,  1.3880175 ,  0.11046211, -0.22518797, -0.61193943,
       -0.6325778 ,  0.8941513 , -1.3140239 , -0.4416165 ,  0.8855457 ,
        1.9597298 ,  1.247972  , -1.0848751 ,  1.1544305 , -0.75664365,
        1.3746328 ,  0.19960149, -0.57200277,  1.4024386 ,  1.3484436 ,
        0.2928772 , -0.8452085 , -2.408678  , -0.8820077 ,  0.98478115,
        0.46658143, -1.9329342 ,  1.0455369 ,  0.2963286 ,  1.3098681 ,
       -0.8403922 ,  1.2858739 ,  0.13813981, -0.37492937,  0.94566995,
       -0.923609  ,  0.17142965,  0.46283364,  2.5075457 , -0.3266498 ,
       -2.0951962 ,  0.52249324, -0.69209397, -0.6390537 , -0.1911823 ,
        0.7833148 ,  1.2130321 , -0.3903246 , -0.57595086, -1.7115734 ,
       -0.35662797,  0.03563855,  0.11664727,  0.7075007 ,  0.9967628 ,
        0.6618875 , -0.31348   , -0.99213207, -0.10415779, -0.94414777,
       -1.103353  , -1.4805537 , -1.4806898 , -1.9195913 ,  0.68176794,
       -1.701665  , -1.524638  ,  0.5038316 , -0.9818233 , -0.12863109,
        1.1921574 , -0.95900047,  1.191072  ,  2.3432536 , -0.89479387,
       -0.44318518,  0.99862444, -0.41647112,  1.7736892 , -1.3457669 ,
       -1.4682574 ,  0.8403623 ,  0.36548048,  0.1995969 ,  0.1717999 ,
        0.252014  ,  0.22803083,  0.11720931,  0.45130366,  1.0384587 ,
        1.3070786 ,  0.8777872 , -1.6547159 , -1.6848826 ,  0.43917903,
       -0.12891226,  0.14131844, -1.5563654 ,  1.8166871 ,  1.2791089 ,
        0.13444622,  1.336699  ,  0.78228426, -0.5616967 , -1.2872282 ,
       -0.5978215 ,  1.9469616 ,  2.060816  ,  1.325076  ,  1.4470675 ,
       -0.50299275,  0.42577764,  0.00464835, -1.7581351 , -0.09081732,
        0.36288866, -0.51137847,  0.15255326, -0.4179586 , -0.12787189,
        2.1442575 , -0.712688  ,  0.32242692, -0.7813733 , -0.6430479 ,
        1.0392716 , -1.2413206 , -0.14977813,  0.9266303 ,  0.36401156,
        0.21835652,  1.4666615 ,  0.767971  ,  0.5893219 ,  0.32377478,
        2.0442357 ,  1.6362653 , -0.5781356 , -1.9579936 ,  1.4419864 ,
       -0.44825733, -2.4178603 ,  0.108367  ,  0.28814316, -0.02632944,
        1.3431495 ,  1.4702865 ,  0.1804034 , -0.9136138 ,  1.2336341 ,
        0.0894188 ,  1.9531444 ,  0.16007975, -0.58052456, -1.4881477 ,
        0.22176485,  1.1730281 , -0.96186006, -0.24914971,  0.9190409 ,
        0.20533632, -1.2440946 , -1.2401365 , -3.4045808 ,  0.7232453 ,
       -0.10568149,  0.15005219, -0.01681262,  0.801422  , -0.14774959,
        0.5647222 ,  2.6449168 ,  0.47841895, -0.9746622 ,  0.8087209 ,
       -0.21483089, -0.43108413, -0.27050295, -1.2235771 ,  0.6889069 ,
        1.1268495 , -1.0787399 , -1.2907518 ,  0.64510506,  0.5676752 ,
        1.2233466 , -0.5769342 , -0.27809045, -1.3952928 ,  0.28910315,
       -0.59807   , -1.0306895 ,  0.46862534,  0.98465633,  0.36538604,
       -2.016394  , -0.32455796,  0.38752025, -1.2624243 ,  0.4091282 ,
       -1.7375559 , -0.6182222 , -0.5046965 , -1.0100905 ,  0.9497864 ,
       -1.2108189 , -1.1647646 ,  0.94855094, -0.5343272 , -0.36582062,
       -0.8830123 ,  0.89345014,  0.32460436, -0.8566404 , -1.211331  ,
       -0.2098477 ], dtype=float32), 'manager': array([0.04394142], dtype=float32)}

 

The above error has been found in your environment! We've added a module for checking your custom environments. It may cause your experiment to fail if your environment is not set up correctly. You can disable this behavior via calling `config.environment(disable_env_checking=True)`. You can run the environment checking module standalone by calling ray.rllib.utils.check_env([your env]).

During handling of the above exception, another exception occurred:

[36mray::AIRRLTrainer.__init__()[39m (pid=5279, ip=127.0.0.1, actor_id=82cd4da7b4edf064d8b847f801000000, repr=AIRA2C)
  File "/Users/floriankockler/anaconda3/envs/py310/lib/python3.10/site-packages/ray/train/rl/rl_trainer.py", line 209, in __init__
    super(AIRRLTrainer, self).__init__(config=rllib_config, **kwargs)
  File "/Users/floriankockler/anaconda3/envs/py310/lib/python3.10/site-packages/ray/rllib/utils/deprecation.py", line 106, in patched_init
    return obj_init(*args, **kwargs)
  File "/Users/floriankockler/anaconda3/envs/py310/lib/python3.10/site-packages/ray/rllib/utils/deprecation.py", line 106, in patched_init
    return obj_init(*args, **kwargs)
  File "/Users/floriankockler/anaconda3/envs/py310/lib/python3.10/site-packages/ray/rllib/algorithms/algorithm.py", line 517, in __init__
    super().__init__(
  File "/Users/floriankockler/anaconda3/envs/py310/lib/python3.10/site-packages/ray/tune/trainable/trainable.py", line 169, in __init__
    self.setup(copy.deepcopy(self.config))
  File "/Users/floriankockler/anaconda3/envs/py310/lib/python3.10/site-packages/ray/rllib/algorithms/a2c/a2c.py", line 165, in setup
    super().setup(config)
  File "/Users/floriankockler/anaconda3/envs/py310/lib/python3.10/site-packages/ray/rllib/algorithms/algorithm.py", line 639, in setup
    self.workers = WorkerSet(
  File "/Users/floriankockler/anaconda3/envs/py310/lib/python3.10/site-packages/ray/rllib/evaluation/worker_set.py", line 179, in __init__
    raise e.args[0].args[2]
ValueError: Traceback (most recent call last):
  File "/Users/floriankockler/anaconda3/envs/py310/lib/python3.10/site-packages/ray/rllib/utils/pre_checks/env.py", line 81, in check_env
    check_multiagent_environments(env)
  File "/Users/floriankockler/anaconda3/envs/py310/lib/python3.10/site-packages/ray/rllib/utils/pre_checks/env.py", line 338, in check_multiagent_environments
    raise ValueError(error)
ValueError: The observation collected from env.reset was not contained within your env's observation space. Its possible that there was a typemismatch (for example observations of np.float32 and a space ofnp.float64 observations), or that one of the sub-observations wasout of bounds

 reset_obs: {'worker': (array([ 1.0000000e+06,  8.2849997e-01,  0.0000000e+00,  6.6008695e+06,
        8.0900002e-01,  7.5940001e-01,  7.2000003e-01,  6.8720001e-01,
        7.8160000e-01,  6.3639500e+01,  1.0000000e-03,  2.6000000e-03,
        3.2121200e-02,  3.5865200e+01,  3.8922901e+01,  1.7600000e-02,
        1.4414001e+00,  6.4513130e+06,  1.9200000e-01,  0.0000000e+00,
        2.7120000e+06,  2.0250000e-01,  1.7739999e-01,  1.4830001e-01,
        1.2500000e-01,  1.8660000e-01,  5.1070801e+01,  3.0000001e-04,
        1.1000000e-03,  1.6757365e-02,  1.3611200e+01,  3.7022499e+01,
        7.7999998e-03,  5.7950002e-01,  8.8800000e+05,  2.1572001e+00,
        0.0000000e+00,  1.4716704e+06,  2.1882999e+00,  2.0676000e+00,
        1.9876000e+00,  1.8763000e+00,  2.1280999e+00,  5.2394699e+01,
        3.1999999e-03,  1.3700000e-02,  5.8082845e-02,  1.7067101e+01,
        3.7153400e+01,  2.7100001e-02,  9.8750001e-01,  6.1828000e+05,
        2.0991001e+00,  0.0000000e+00,  2.8819525e+06,  2.1087000e+00,
        1.9918000e+00,  1.9024000e+00,  1.8837000e+00,  2.0509000e+00,
        5.6803101e+01,  2.3000001e-03,  1.5400000e-02,  4.9362659e-02,
        9.9177999e+00,  3.5919998e+01,  3.0400001e-02,  1.0458000e+00,
        1.3305640e+06,  1.5995600e+01,  0.0000000e+00,  3.3783200e+05,
        1.6270300e+01,  1.4810900e+01,  1.4907200e+01,  1.5356400e+01,
        1.5259000e+01,  5.8494099e+01,  1.0854000e+00,  8.8209999e-01,
        1.0370754e+00,  1.7062000e+00,  2.6076700e+01,  4.2010000e-01,
        8.0390000e-01,  1.8950000e+05,  1.4890000e+00,  0.0000000e+00,
        8.9391680e+06,  1.5023000e+00,  1.4270999e+00,  1.3663000e+00,
        1.3285000e+00,  1.4598000e+00,  5.3890499e+01,  2.9000000e-03,
        6.5000001e-03,  5.3075951e-02,  1.7767700e+01,  3.5822601e+01,
        2.3100000e-02,  1.0783000e+00,  3.3312000e+06,  2.3942001e+00,
        0.0000000e+00,  2.0565440e+06,  2.3724000e+00,  2.2061999e+00,
        2.0425000e+00,  1.9007000e+00,  2.2830999e+00,  6.2822701e+01,
        7.0000002e-03,  3.7300002e-02,  8.6248346e-02,  3.7857399e+01,
        3.9324501e+01,  5.5300001e-02,  1.8139000e+00,  1.7568000e+06,
        4.0239999e-01,  0.0000000e+00,  3.3790720e+06,  3.9080000e-01,
        3.8460001e-01,  3.6340001e-01,  3.2749999e-01,  3.9100000e-01,
        5.9131599e+01,  9.9999997e-05,  6.9999998e-04,  9.6539799e-03,
        1.8160801e+01,  2.6746500e+01,  2.7999999e-03,  9.2369998e-01,
        3.3664000e+06,  1.1337000e+00,  0.0000000e+00,  6.6739320e+06,
        1.1091000e+00,  1.0306000e+00,  9.7060001e-01,  9.2140001e-01,
        1.0681000e+00,  7.0823601e+01,  1.5000000e-03,  6.2000002e-03,
        4.0974230e-02,  4.8390800e+01,  5.0257999e+01,  2.5300000e-02,
        1.7750000e+00,  8.6359390e+06,  7.1780002e-01,  0.0000000e+00,
        1.2221424e+07,  7.3320001e-01,  7.2070003e-01,  6.9760001e-01,
        6.8730003e-01,  7.3839998e-01,  4.2753799e+01,  5.0000002e-04,
        1.6000000e-03,  2.2213263e-02,  2.3009100e+01,  3.1001400e+01,
       -2.4000001e-03,  1.3231000e+00,  8.9400000e+06,  4.0570000e-01,
        0.0000000e+00,  5.4134400e+05,  4.0070000e-01,  3.9219999e-01,
        3.7090001e-01,  3.5780001e-01,  3.9780000e-01,  5.6426800e+01,
        0.0000000e+00,  6.9999998e-04,  6.4477473e-03,  1.6508200e+01,
        1.6720800e+01,  3.4000000e-03,  5.4119998e-01,  2.0640000e+05,
        2.7255001e+00,  0.0000000e+00,  1.4056536e+05,  2.7816999e+00,
        2.8318000e+00,  2.8761001e+00,  2.6111000e+00,  2.8545001e+00,
        4.1289299e+01,  1.9099999e-02,  2.0600000e-02,  1.3748071e-01,
        1.2211100e+01,  1.5229400e+01, -3.9600000e-02,  4.4400001e-01,
        1.1036200e+05,  2.4990000e-01,  0.0000000e+00,  1.2128000e+06,
        2.5119999e-01,  2.4810000e-01,  2.4590001e-01,  2.4640000e-01,
        2.4710000e-01,  5.1323502e+01,  0.0000000e+00,  1.9999999e-04,
        6.7977011e-03,  1.0141300e+01,  1.6625999e+01,  2.4999999e-03,
        3.0950001e-01,  2.1408000e+06,  0.0000000e+00,  0.0000000e+00,
        1.8114200e+05,  0.0000000e+00], dtype=float32), {}), 'manager': (array([66.], dtype=float32), {})}

 env.observation_space_sample(): {'worker': array([-0.20289347,  0.11215498,  0.09123678, -0.47268885, -0.5170849 ,
       -0.03265719,  1.025571  ,  0.29880205,  0.27762076, -1.54744   ,
       -0.47113052,  1.3880175 ,  0.11046211, -0.22518797, -0.61193943,
       -0.6325778 ,  0.8941513 , -1.3140239 , -0.4416165 ,  0.8855457 ,
        1.9597298 ,  1.247972  , -1.0848751 ,  1.1544305 , -0.75664365,
        1.3746328 ,  0.19960149, -0.57200277,  1.4024386 ,  1.3484436 ,
        0.2928772 , -0.8452085 , -2.408678  , -0.8820077 ,  0.98478115,
        0.46658143, -1.9329342 ,  1.0455369 ,  0.2963286 ,  1.3098681 ,
       -0.8403922 ,  1.2858739 ,  0.13813981, -0.37492937,  0.94566995,
       -0.923609  ,  0.17142965,  0.46283364,  2.5075457 , -0.3266498 ,
       -2.0951962 ,  0.52249324, -0.69209397, -0.6390537 , -0.1911823 ,
        0.7833148 ,  1.2130321 , -0.3903246 , -0.57595086, -1.7115734 ,
       -0.35662797,  0.03563855,  0.11664727,  0.7075007 ,  0.9967628 ,
        0.6618875 , -0.31348   , -0.99213207, -0.10415779, -0.94414777,
       -1.103353  , -1.4805537 , -1.4806898 , -1.9195913 ,  0.68176794,
       -1.701665  , -1.524638  ,  0.5038316 , -0.9818233 , -0.12863109,
        1.1921574 , -0.95900047,  1.191072  ,  2.3432536 , -0.89479387,
       -0.44318518,  0.99862444, -0.41647112,  1.7736892 , -1.3457669 ,
       -1.4682574 ,  0.8403623 ,  0.36548048,  0.1995969 ,  0.1717999 ,
        0.252014  ,  0.22803083,  0.11720931,  0.45130366,  1.0384587 ,
        1.3070786 ,  0.8777872 , -1.6547159 , -1.6848826 ,  0.43917903,
       -0.12891226,  0.14131844, -1.5563654 ,  1.8166871 ,  1.2791089 ,
        0.13444622,  1.336699  ,  0.78228426, -0.5616967 , -1.2872282 ,
       -0.5978215 ,  1.9469616 ,  2.060816  ,  1.325076  ,  1.4470675 ,
       -0.50299275,  0.42577764,  0.00464835, -1.7581351 , -0.09081732,
        0.36288866, -0.51137847,  0.15255326, -0.4179586 , -0.12787189,
        2.1442575 , -0.712688  ,  0.32242692, -0.7813733 , -0.6430479 ,
        1.0392716 , -1.2413206 , -0.14977813,  0.9266303 ,  0.36401156,
        0.21835652,  1.4666615 ,  0.767971  ,  0.5893219 ,  0.32377478,
        2.0442357 ,  1.6362653 , -0.5781356 , -1.9579936 ,  1.4419864 ,
       -0.44825733, -2.4178603 ,  0.108367  ,  0.28814316, -0.02632944,
        1.3431495 ,  1.4702865 ,  0.1804034 , -0.9136138 ,  1.2336341 ,
        0.0894188 ,  1.9531444 ,  0.16007975, -0.58052456, -1.4881477 ,
        0.22176485,  1.1730281 , -0.96186006, -0.24914971,  0.9190409 ,
        0.20533632, -1.2440946 , -1.2401365 , -3.4045808 ,  0.7232453 ,
       -0.10568149,  0.15005219, -0.01681262,  0.801422  , -0.14774959,
        0.5647222 ,  2.6449168 ,  0.47841895, -0.9746622 ,  0.8087209 ,
       -0.21483089, -0.43108413, -0.27050295, -1.2235771 ,  0.6889069 ,
        1.1268495 , -1.0787399 , -1.2907518 ,  0.64510506,  0.5676752 ,
        1.2233466 , -0.5769342 , -0.27809045, -1.3952928 ,  0.28910315,
       -0.59807   , -1.0306895 ,  0.46862534,  0.98465633,  0.36538604,
       -2.016394  , -0.32455796,  0.38752025, -1.2624243 ,  0.4091282 ,
       -1.7375559 , -0.6182222 , -0.5046965 , -1.0100905 ,  0.9497864 ,
       -1.2108189 , -1.1647646 ,  0.94855094, -0.5343272 , -0.36582062,
       -0.8830123 ,  0.89345014,  0.32460436, -0.8566404 , -1.211331  ,
       -0.2098477 ], dtype=float32), 'manager': array([0.04394142], dtype=float32)}

 

The above error has been found in your environment! We've added a module for checking your custom environments. It may cause your experiment to fail if your environment is not set up correctly. You can disable this behavior via calling `config.environment(disable_env_checking=True)`. You can run the environment checking module standalone by calling ray.rllib.utils.check_env([your env]).
