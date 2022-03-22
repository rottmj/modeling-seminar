import traci
import agent

class Environment:
    def __init__(self) -> None:
        self.time_switch = 0
        self.edges = []
        self.tl_id 
        self.nb_steps = 500
        self.a_switch = 0.5


    def get_state(self):
        '''
        Returns the state of the simulation for one traffic light at timestep t
        The state contains the following information:

        for each incoming lane:
            number of vehicles (density)
            minimum distance of the closest vehicle to the traffic light
            average speed
            accumulated waiting time of the vehicles

        binary variables:
            which direction has green (vertical or horizontal)
            if traffic light is yellow

        time until next possible light switch [0, t_switch] 
        '''
        state_list = []
        for edge in self.edges: 
            state_list.append(traci.lane.getLastStepMeanSpeed(edge))
            state_list.append(traci.lane.getLastStepVehicleNumber(edge))
            state_list.append(traci.lane.getLastStepHaltingNumber(edge))
            # state_list.append()
            # minimum distance

        state_list.append(self.time_switch)
        state_list.append(traci.trafficlight.getPhase(self.tl_id))

        # direction aus GrGr = VHVH ablesen
        state_list.append(int(traci.trafficlight.getProgram(traci.trafficlight.getIDList[0])))
        return state_list


    def get_reward():
        ''''
        Computes and returns the average speed of all vehicles at timestep t as reward.
        Average Speed is measured in m/s.

            Parameters:
                    veh_num (int): Number of vehicles
                    veh_ids (List[str]): List of vehicle IDs
                    overall_speed (int): Aggregated speed of all vehicles
            
            Returns:
                    TODO (float?): Average speed of all vehicles
        '''
        def compute_reward():
            veh_num = traci.vehicle.getIDCount()
            if veh_num == 0: return 0
            veh_ids = traci.vehicle.getIDList()
            overall_speed = sum([traci.vehicle.getSpeed(veh_id) for veh_id in veh_ids])
            return overall_speed / veh_num
        return compute_reward()


    def select_action(self, a_t):
        if a_t > self.a_switch:
            if self.time_switch == 0:
                return True
        return False


    def run_simulation(self, sumoCmd, agent: agent.Agent):
        traci.start(sumoCmd)
        step = 0
        reward = []

        # TODO vertical und horitontal ob richtig checken
        phases_vertical = (
            traci.trafficlight.Phase(duration=2.0, state='yryr', next=()),
            traci.trafficlight.Phase(duration=self.nb_steps, state='rGrG', next=())
            )
        phases_horizontal = (
            traci.trafficlight.Phase(duration=2.0, state='ryry', next=()),
            traci.trafficlight.Phase(duration=self.nb_steps, state='GrGr', next=())
            )
        
        logic_vertical = traci.trafficlight.Logic('vertical', 0, 0, phases_vertical)
        logic_horizontal = traci.trafficlight.Logic('horizontal', 0, 0, phases_horizontal)

        # set inital trafic light logic
        traci.trafficlight.setProgramLogic(self.tl_id, logic_vertical)
        current_logic = 'vertical'

        while step < self.nb_steps:  
            # get observation, compute reward, select action
            state = self.get_state()
            reward.append(self.get_reward())
            a_t = agent.compute_action()
            if self.select_action(a_t):
                # change tf logic
                if current_logic == 'horizontal':
                    traci.trafficlight.setProgramLogic(self.tl_id, 'vertical')
                    current_logic = 'vertical'
                else:
                    traci.trafficlight.setProgramLogic(self.tl_id, 'horizontal')
                    current_logic = 'horizontal'
            
            # update environment
            traci.simulationStep()
            step += 1

        traci.close()
        return sum(reward)

