import argparse

import mlflow


def start_parent_run(tracking_uri: str, experiment_name: str, run_name: str) -> str:
    """Open a parent run, reusing an existing run with the same name.

    Parameters
    ----------
    tracking_uri : str
        MLflow tracking URI.
    experiment_name : str
        Experiment to hold the run, created if missing.
    run_name : str
        Name of the parent run.

    Returns
    -------
    str
        ID of the parent run, marked RUNNING.
    """
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(experiment_name)
    client = mlflow.MlflowClient(tracking_uri)
    existing = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"attributes.run_name = '{run_name}'",
        order_by=["attributes.start_time ASC"],
        max_results=1,
    )
    if existing:
        run_id = existing[0].info.run_id
        client.update_run(run_id, status="RUNNING")
        return run_id
    return client.create_run(experiment.experiment_id, run_name=run_name).info.run_id


def finish_parent_run(tracking_uri: str, run_id: str) -> None:
    """Mark a parent run FINISHED.

    Parameters
    ----------
    tracking_uri : str
        MLflow tracking URI.
    run_id : str
        ID of the parent run.
    """
    mlflow.MlflowClient(tracking_uri).set_terminated(run_id, status="FINISHED")


def main():
    parser = argparse.ArgumentParser(
        description="Open or close an MLflow parent run that groups a sweep."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser(
        "start", help="Print the ID of the parent run, reusing one with the same name."
    )
    start.add_argument("tracking_uri")
    start.add_argument("experiment_name")
    start.add_argument("run_name")

    finish = subparsers.add_parser("finish", help="Mark the parent run FINISHED.")
    finish.add_argument("tracking_uri")
    finish.add_argument("run_id")

    args = parser.parse_args()
    if args.command == "start":
        print(start_parent_run(args.tracking_uri, args.experiment_name, args.run_name))
    else:
        finish_parent_run(args.tracking_uri, args.run_id)


if __name__ == "__main__":
    main()
