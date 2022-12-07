"""This is the worker, it's the main workhorse that deals with getting requests, and spawning data processing"""
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from nataili.model_manager import ModelManager
from nataili.util import logger, quiesce_logger, set_logger_verbosity
from worker.argparser import args
from worker.bridge_data import BridgeData
from worker.job import HordeJob
from worker.stats import bridge_stats


def reload_data(this_bridge_data):
    """This is just a utility function to reload the configuration"""
    this_bridge_data.reload_data()
    this_bridge_data.check_models(model_manager)
    this_bridge_data.reload_models(model_manager)


@logger.catch(reraise=True)
def bridge(this_model_manager, this_bridge_data):
    """This is the worker, it's the main workhorse that deals with getting requests, and spawning data processing"""
    running_jobs = []
    run_count = 0
    last_config_reload = 0 # Means immediate config reload
    logger.stats("Starting new stats session")
    reload_data(this_bridge_data)
    try:
        should_stop = False
        while True:  # This is just to allow it to loop through this and handle shutdowns correctly
            with ThreadPoolExecutor(max_workers=this_bridge_data.max_threads) as executor:
                while True:
                    try:
                        if time.time() - last_config_reload > 60:
                            reload_data(this_bridge_data)
                            executor._max_workers = this_bridge_data.max_threads
                            logger.stats(f"Stats this session:\n{bridge_stats.get_pretty_stats()}")
                            try:
                                models_data = requests.get(bridge_data.horde_url + "/api/v2/status/models", timeout=10).json()
                                models_data.sort(key=lambda x: (x["eta"], x["queued"] / x["performance"]), reverse=True)
                                top_5 = [x["name"] for x in models_data[:5]]
                                logger.stats(f"Top 5 models by load: {', '.join(top_5)}")
                                dynamic_models = [x["name"] for x in models_data[:this_bridge_data.number_of_dynamic_models]]
                                logger.info("Dynamically loading new models to attack the relevant queue: {}", dynamic_models)
                                this_bridge_data.model_names = dynamic_models
                            # pylint: disable=broad-except
                            except Exception as err:
                                logger.warning("Failed to get models_req to do dynamic model loading: {}", err)

                            last_config_reload = time.time()

                        if len(this_model_manager.get_loaded_models_names()) == 0:
                            time.sleep(5)
                            logger.info("No models loaded. Waiting for the first model to be up before polling the horde")
                            continue

                        pop_count = 0
                        while len(running_jobs) < this_bridge_data.max_threads:
                            pop_count += 1
                            if pop_count > 3:  # Just to allow reload to fire
                                break
                            new_job = HordeJob(this_model_manager, this_bridge_data)
                            pop = new_job.get_job_from_server()  # This sleeps itself, so no need for extra
                            if pop is None:
                                del new_job
                                continue
                            job_model = pop.get("model", "Unknown")
                            logger.debug("Got a new job from the horde for model: {}", job_model)
                            running_jobs.append(executor.submit(new_job.start_job, pop))

                        for job in running_jobs:
                            if job.done():
                                run_count += 1
                                logger.debug("Job finished successfully (Total Completed: {})", run_count)
                                running_jobs.remove(job)
                                continue

                            if job.exception():
                                logger.error("Job failed with exception, {}", job.exception())
                                logger.exception(job.exception())
                                if job in running_jobs:  # Sometimes it's already removed
                                    running_jobs.remove(job)
                                continue

                            # check if any job has run for more than 60 seconds
                            if job.running() and job.running_for() > 180:
                                logger.warning("Restarting all jobs, as job as was running for more than 180 seconds: {}", job.running_for())
                                for inner_job in running_jobs:  # Sometimes it's already removed
                                    running_jobs.remove(inner_job)
                                    job.cancel()
                                executor.shutdown(wait=False)
                                break

                        time.sleep(0.1)  # Give the CPU a break
                    except KeyboardInterrupt:
                        should_stop = True
                        break

            if should_stop:
                break

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt detected, shutting down")

    logger.info("Shutting down bridge")
    executor.shutdown(wait=False)
    for job in running_jobs:
        job.cancel()
    logger.info("Shutting down bridge - Done")


if __name__ == "__main__":
    set_logger_verbosity(args.verbosity)
    if args.log_file:
        logger.add("koboldai_bridge_log.log", retention="7 days", level="warning")  # Automatically rotate too big file
    quiesce_logger(args.quiet)
    logger.add("logs/worker.log", retention="1 days", level=10)

    bridge_data = BridgeData()
    model_manager = ModelManager(disable_voodoo=bridge_data.disable_voodoo.active)
    model_manager.init()
    try:
        bridge(model_manager, bridge_data)
    except KeyboardInterrupt:
        logger.info("Keyboard Interrupt Received. Ending Process")
    logger.init(f"{bridge_data.worker_name} Instance", status="Stopped")
