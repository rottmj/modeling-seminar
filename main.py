import argparse

import os, sys

import agent
import environment

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

sumoBinary = "/usr/bin/sumo"
sumoCmd = [sumoBinary, "-c", "simpleTrafficLight.sumocfg"]


def main():
    my_parser = argparse.ArgumentParser(description='Trafic flow simulation using RL')
    
    my_parser.add_argument(
        'Model', 
        metavar='model', 
        type=str, 
        help='linear or rbf model'
    )

    args = my_parser.parse_args()

    model = args.Model
    controller = agent.Agent(model)
    
    env = environment.Environment()

    reward = env.run_simulation(sumoCmd=sumoCmd, agent=controller)

    # train model
    controller.train()

if __name__ == '__main__':
    main()
