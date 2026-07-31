import pickle

import numpy as np

from tiangong_recorder.dataset_writer import write_episode_pickle


def test_writes_existing_list_of_dicts_format_atomically(tmp_path):
    episode = [
        {
            "timestamp": 0.0,
            "qpos": {"left_arm": np.array([1.0])},
            "image": {
                "head": {
                    "color": np.zeros((2, 2, 3), dtype=np.uint8),
                }
            },
        }
    ]

    path = write_episode_pickle("teleop_log_test_1", episode, tmp_path)

    assert path.name == "teleop_log_test_1.pkl"
    assert not (tmp_path / "teleop_log_test_1.pkl.tmp").exists()
    with path.open("rb") as stream:
        loaded = pickle.load(stream)
    assert isinstance(loaded, list)
    assert loaded[0]["image"]["head"]["color"].dtype == np.uint8

