import argparse
from multiprocessing import freeze_support

import trackeval
from trackeval.datasets import VisDroneMOT

if __name__ == "__main__":
    freeze_support()

    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_eval_config["DISPLAY_LESS_PROGRESS"] = False

    default_dataset_config = VisDroneMOT.get_default_dataset_config()
    default_dataset_config["CLASSES_TO_EVAL"] = ["pedestrian"]

    default_metrics_config = {
        "METRICS": ["HOTA", "CLEAR", "Identity"],
        "THRESHOLD": 0.5,
    }

    config = {**default_eval_config, **default_dataset_config, **default_metrics_config}

    parser = argparse.ArgumentParser()
    for setting in config.keys():
        if isinstance(config[setting], list) or config[setting] is None:
            parser.add_argument("--" + setting, nargs="+")
        else:
            parser.add_argument("--" + setting)

    args = parser.parse_args().__dict__

    for setting in args.keys():
        if args[setting] is not None:
            if isinstance(config[setting], bool):
                if args[setting] == "True":
                    x = True
                elif args[setting] == "False":
                    x = False
                else:
                    raise Exception(
                        f"Command line parameter {setting} must be True or False"
                    )
            elif isinstance(config[setting], int):
                x = int(args[setting])
            elif setting == "SEQ_INFO":
                x = dict(zip(args[setting], [None] * len(args[setting])))
            elif setting == "IGNORE_IOA_THRESHOLD":
                x = float(args[setting])
            else:
                x = args[setting]
            config[setting] = x

    eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
    dataset_config = {
        k: v for k, v in config.items() if k in default_dataset_config.keys()
    }
    metrics_config = {
        k: v for k, v in config.items() if k in default_metrics_config.keys()
    }

    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [VisDroneMOT(dataset_config)]

    metrics_list = []
    for metric in [
        trackeval.metrics.HOTA,
        trackeval.metrics.CLEAR,
        trackeval.metrics.Identity,
    ]:
        if metric.get_name() in metrics_config["METRICS"]:
            metrics_list.append(metric(metrics_config))

    if len(metrics_list) == 0:
        raise Exception("No metrics selected for evaluation")

    evaluator.evaluate(dataset_list, metrics_list)
