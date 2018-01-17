"""
Created on May 14, 2016

a rule-based user simulator

-- user_goals_first_turn_template.revised.v1.p: all goals
-- user_goals_first_turn_template.part.movie.v1.p: moviename in goal.inform_slots
-- user_goals_first_turn_template.part.nomovie.v1.p: no moviename in goal.inform_slots

@author: xiul, t-zalipt
"""

from .usersim import UserSimulator
import argparse, json, random, copy

from deep_dialog import dialog_config



class RealUser(UserSimulator):
    """ A rule-based user simulator for testing dialog policy """
    
    def __init__(self, movie_dict=None, act_set=None, slot_set=None, start_set=None, params=None):
        """ Constructor shared by all user simulators """
        
        self.movie_dict = movie_dict
        self.act_set = act_set
        self.slot_set = slot_set
        self.start_set = start_set
        
        self.max_turn = params['max_turn']
        self.slot_err_probability = params['slot_err_probability']
        self.slot_err_mode = params['slot_err_mode']
        self.intent_err_probability = params['intent_err_probability']
        
        self.simulator_run_mode = params['simulator_run_mode']
        self.simulator_act_level = params['simulator_act_level']
        
        self.learning_phase = params['learning_phase']
    
    def initialize_episode(self):
        """ Initialize a new episode (dialog) 
        state['history_slots']: keeps all the informed_slots
        state['rest_slots']: keep all the slots (which is still in the stack yet)
        """

        self.state = {}
        self.state['turn'] = 0

        self.episode_over = False
        self.dialog_status = dialog_config.NO_OUTCOME_YET


        response_action = {}
        response_action['turn'] = self.state['turn']
        print("입력")
        response_action['nl'] = input()


        if self.simulator_act_level == 1:
            user_nlu_res = self.nlu_model.generate_dia_act(response_action['nl']) # NLU
            if user_nlu_res != None:
                #user_nlu_res['diaact'] = user_action['diaact'] # or not?
                response_action.update(user_nlu_res)

        assert (self.episode_over != 1),' but we just started'
        return response_action

        
    def next(self, system_action):
        """ Generate next User Action based on last System Action """
        
        self.state['turn'] += 2
        self.episode_over = False
        self.dialog_status = dialog_config.NO_OUTCOME_YET
        print("입력")
        response_action = {}
        response_action['turn'] = self.state['turn']
        response_action['nl'] = input()

        if (self.max_turn > 0 and self.state['turn'] > self.max_turn):
            self.dialog_status = dialog_config.FAILED_DIALOG
            self.episode_over = True
            response_action['nl'] == "감사합니다"

        if (response_action['nl'] == "감사합니다"):
            self.episode_over = True
            self.dialog_status = dialog_config.SUCCESS_DIALOG

        elif (response_action['nl'] == "실패"):
            self.episode_over = True
            self.dialog_status = dialog_config.FAILED_DIALOG


        if self.simulator_act_level == 1:
            user_nlu_res = self.nlu_model.generate_dia_act(response_action['nl']) # NLU
            if user_nlu_res != None:
                #user_nlu_res['diaact'] = user_action['diaact'] # or not?
                response_action.update(user_nlu_res)

        return response_action, self.episode_over, self.dialog_status
    



def main(params):
    user_sim = RealUser()
    user_sim.initialize_episode()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    params = vars(args)

    print ("User Simulator Parameters:")
    print (json.dumps(params, indent=2))

    main(params)
