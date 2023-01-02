import warnings
import logging
import numpy as np
from os import environ
import sys
import json

from hier_lstm import Model_Recursive_LSTM_v2
from json_to_tensor import *

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning) 
warnings.filterwarnings('ignore', category=UserWarning)

    if pad:
        for i in range(pad_amount):
            vector = torch.zeros_like(vector)
            vectors.append(vector)
    return (first_part, vectors[0], torch.cat(vectors[1:], dim=1), third_part)

with torch.no_grad():
    device = "cpu"
    torch.device("cpu")

    environ["layers"] = "600 350 200 180"
    environ["dropouts"] = "0.05 " * 4
    logging.info("got here")
    input_size = 1056
    output_size = 1

    layers_sizes = list(map(int, environ.get("layers", "600 350 200 180").split()))
    drops = list(map(float, environ.get("dropouts", "0.05 0.05 0.05 0.05 0.05").split()))

    model = Model_Recursive_LSTM_v2(
        input_size=input_size,
        comp_embed_layer_sizes=layers_sizes,
        drops=drops,
        transformation_matrix_dimension=13,
        loops_tensor_size=33,
        expr_embed_size=100,
        train_device=device,
        bidirectional=True,
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    
    model.to(device)
    model.eval()

    with torch.no_grad():
        try:
            while True:
                
                
                prog_json = input()
                no_sched_json = input()
                sched_json = input()
                

                program_json = json.loads(prog_json)
                sched_json = json.loads(sched_json)
                no_sched_json = json.loads(no_sched_json)
                
                (
                    prog_tree,
                    comps_repr_templates_list,
                    loops_repr_templates_list,
                    comps_placeholders_indices_dict,
                    loops_placeholders_indices_dict,
                    comps_expr_tensor,
                    comps_expr_lengths,
                ) = get_representation_template(program_json, no_sched_json, MAX_DEPTH)
                comps_tensor, loops_tensor = get_schedule_representation(
                    program_json,
                    sched_json,
                    comps_repr_templates_list,
                    loops_repr_templates_list,
                    comps_placeholders_indices_dict,
                    loops_placeholders_indices_dict,
                    max_depth=5,
                )
                
                x = comps_tensor
                batch_size, num_comps, __dict__ = x.shape
                x = x.view(batch_size * num_comps, -1)
                (first_part, final_matrix, vectors, third_part) = seperate_vector(
                        x, num_matrices=5, pad=False
                    )
                x = torch.cat(
                    (
                        first_part,
                        third_part,
                        final_matrix.reshape(batch_size * num_comps, -1),
                    ),
                    dim=1,
                ).view(batch_size, num_comps, -1)
                
                tree_tensor = (prog_tree, x, vectors, loops_tensor, comps_expr_tensor, comps_expr_lengths)

                speedup = model.forward(tree_tensor)
                print(float(speedup.item()))
        except EOFError:
            exit()