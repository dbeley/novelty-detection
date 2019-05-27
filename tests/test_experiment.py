from experiments.experiment import Experiment


def test_experiment():
    exp = Experiment("test_encoder", [2000, 300, 20])

    assert exp.size_historic == 2000
    assert exp.size_context == 300
    assert exp.size_novelty == 20
