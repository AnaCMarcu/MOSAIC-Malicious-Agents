import json
from simulation import Simulation
from utils import Utils
import logging
import pandas as pd

if __name__ == "__main__":
    with open('configs/experiment_config.json', 'r') as file:
        config = json.load(file)

    Utils.configure_logging(engine=config['engine'])
    logging.info(f"Starting simulation with {config['num_users']} users for {config['num_time_steps']} time steps using {config['engine']}...")

    sim = Simulation(config)
    sim.run(config['num_time_steps'])
    # splits = {'MF': 'data/MF-00000-of-00001-76b8ff6de79a2e48.parquet'}
    # df = pd.read_parquet("hf://datasets/Jinyan1/PolitiFact/" + splits["MF"])
    #
    # output_file = "../data/fake_news.jsonl"
    # df[["id", "description", "text", "title"]].to_json(output_file, orient="records", lines=True, force_ascii=False)


    