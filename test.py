# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 16:37:26 2024

@author: William
"""

import os
import unittest
from hydra import initialize, compose

from train import main

class End2EndTestCase(unittest.TestCase):
    
    @classmethod
    def setUpClass(self) -> None:
        
        self.dir_path = os.path.join('.', 'test_outputs')

        #Create the test outputs:        
        if not os.path.exists(self.dir_path):
            os.mkdir(self.dir_path)      
                
    def run_experiment_config(self, override_list):
        
        with initialize(version_base=None, config_path="./config"):
                            
            # config is relative to a module
            cfg = compose(config_name="config", overrides=override_list)

            #Set the script to save to the local dir path
            cfg.local_run_dir = self.dir_path      
                        
            main(cfg)
        

class TestBasicTraining(End2EndTestCase):
    
    """
    Tests the Basic sft and dpo training scripts end to end. The experiment yaml
    test_sft.yaml and test_dpo.yaml files contain suitable test experiment configs
    """
        
    @classmethod
    def setUpClass(self) -> None:
        super().setUpClass()

    @classmethod
    def tearDownClass(self) -> None:
        super().tearDownClass()
        
    def test_a_basic_sft(self) -> None:
        """
        We ensure that test_basic_sft is placed first in the class so as to create
        the pretrained reference model setup in test_output dir that the other tests
        can use...
        """
        self.run_experiment_config(override_list=["+test_experiment=test_sft"])
    
    def test_basic_dpo(self) -> None:
        
        model_archive = os.path.join(self.dir_path, 'LATEST', 'policy.pt')
        
        self.run_experiment_config(override_list=["+test_experiment=test_dpo",
                                                  f"++model.archive={model_archive}"])
        

class TestDataSelection(End2EndTestCase):
    
    """
    Test the e2e training scripts that run data selection code
    """
    
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

    @classmethod
    def tearDownClass(self) -> None:
        super().tearDownClass()
        
        
    def test_a_basic_sft(self) -> None:
        """
        We ensure that test_basic_sft is placed first in the class so as to create
        the pretrained reference model setup in test_output dir that the other tests
        can use...
        """
        self.run_experiment_config(override_list=["+test_experiment=test_sft"])
        
    def test_basic_us_dpo(self) -> None:
        
        model_archive = os.path.join(self.dir_path, 'LATEST', 'policy.pt')
        self.run_experiment_config(override_list=["+test_experiment=test_dpo_us",
                                    f"++model.archive={model_archive}"])        

    def test_basic_rho_loss_dpo(self) -> None:
        
        model_archive = os.path.join(self.dir_path, 'LATEST', 'policy.pt')
        self.run_experiment_config(override_list=["+test_experiment=test_dpo_rho",
                                    f"++model.archive={model_archive}",
                                    "+model@data_selection.model=tiny-mistral",
                                    f"++data_selection.ft_state_dict_path={model_archive}",
                                    f"++data_selection.sft_state_dict_path={model_archive}"])

if __name__ == '__main__':
    
    #Ensure the tests run in the order written:
    def cmp(a, b):
        return (a > b) - (a < b) 
    unittest.TestLoader.sortTestMethodsUsing = lambda self, a, b: cmp(a, b) * -1
    
    unittest.main()
