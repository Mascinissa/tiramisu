import numpy as np
from os import environ
import sys, json

from hier_lstm import Model_Recursive_LSTM_v2
from json_to_tensor import *

import os
environ['MKL_THREADING_LAYER'] = 'GNU'

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning) 
warnings.filterwarnings('ignore', category=UserWarning)

model_path = '/data/scratch/mmerouani/tiramisu2/tiramisu/tutorials/tutorial_autoscheduler/model/MAPE_base_13+4+2.6_22.7.pkl'

with torch.no_grad():
        device = 'cpu'
        torch.device('cpu')

#         environ['layers'] = '600 350 200 180'
#         environ['dropouts'] = '0.225 ' * 4

        input_size = 776
        output_size = 1

#         layers_sizes = list(map(int, environ.get('layers', '300 200 120 80 30').split()))
#         drops = list(map(float, environ.get('dropouts', '0.2 0.2 0.1 0.1 0.1').split()))

        model = Model_Recursive_LSTM_v2(input_size)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.to(device)
        model.eval()

        try:
            while True:
                prog_json = input()
                sched_json = input()

                prog_json = json.loads(prog_json)
                sched_json = json.loads(sched_json)
                prog_tree, comps_repr_templates_list, loops_repr_templates_list, comps_placeholders_indices_dict, loops_placeholders_indices_dict = get_representation_template(prog_json, max_depth = 5)
                comps_tensor, loops_tensor = get_schedule_representation(prog_json, sched_json, comps_repr_templates_list, loops_repr_templates_list, comps_placeholders_indices_dict, loops_placeholders_indices_dict, max_depth=5)
                
                tree_tensor = (prog_tree, torch.cat([comps_tensor], 0), torch.cat([loops_tensor],0))

#                 tree_tensor = get_representation(prog_json, sched_json)

                speedup = model.forward(tree_tensor,)
                print(float(speedup.item()))

        except EOFError:
            exit()
        
