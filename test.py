# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 16:37:26 2024

@author: William
"""

import os
import shutil
import unittest
from hydra import initialize, compose

from train import main

#Ensure the tests run in the order written
unittest.TestLoader.sortTestMethodsUsing = None

class TestBasicTraining(unittest.TestCase):
    
    @classmethod
    def setUpClass(self) -> None:
        
        self.dir_path = './test_outputs'
        self.sft_save_dir = None
        self.dpo_save_dir = None

        #Create the test outputs:        
        if not os.path.exists(self.dir_path):
            os.mkdir(self.dir_path)      
        
    @classmethod
    def tearDownClass(self) -> None:
        
        for file in os.listdir(self.sft_save_dir):
            print(file)
        
    
    def test_basic_sft(self) -> None:
    
        with initialize(version_base=None, config_path="./config"):
            
            # config is relative to a module
            cfg = compose(config_name="config", overrides=["+experiment=test_sft"])
            
            # Create the save dir path:
            self.sft_save_dir = cfg.local_run_dir                                
            
            #Run the e2e training script
            main(cfg)

    def test_basic_dpo(self) -> None:
        
        with initialize(version_base=None, config_path="./config"):
            
            # Build the sft training script archive path:
            model_archive = os.path.join(self.sft_save_dir,
                                         'LATEST', 'policy.pt')
            
            # config is relative to a module
            cfg = compose(config_name="config", overrides=["+experiment=test_sft",
                                                           f"++model.archive={model_archive}"])

            self.dpo_save_dir = cfg.local_run_dir             
                        
            main(cfg)


# class TestDataSelection(unittest.TestCase):
            
#     def test_basic_us_dpo(self) -> None:
        
#         with initialize(version_base=None, config_path="./config"):
            
#             # config is relative to a module
#             cfg = compose(config_name="config", overrides=["+experiment=test_sft"])
#             main(cfg)
            
#     def test_basic_rho_loss_dpo(self): -> None:
        
#         with initialize(version_base=None, config_path="./config"):
            
#             # config is relative to a module
#             cfg = compose(config_name="config", overrides=["+experiment=test_sft"])
#             main(cfg)

if __name__ == '__main__':
    
    unittest.main()
