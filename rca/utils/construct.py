import re
from datasets import load_dataset

trajectory_path = "neulab/agent-data-collection"
trajectory_name = "SWE-smith_5kTrajectories"
trajectory_data = load_dataset(trajectory_path, trajectory_name, split="train").to_pandas()

# regex function that captures string between <X= and >
def parse_action(response):
    function_format = r"<function=(.*?)>(.*?)</function>"
    function_matched = re.search(function_format, response, re.DOTALL)
    
    if function_matched:
        function_name = function_matched.group(1)
        function_param_string = function_matched.group(2).strip()
        
        param_format = r"<parameter=(.*?)>(.*?)</parameter>"
        param_matched = re.findall(param_format, function_param_string, re.DOTALL)
        if param_matched:
            params = {name: value.strip() for name, value in param_matched}
        else:
            params = {}
        
        return function_name, params
    else:
        return None, {}

def get_function(traj):
    for step in traj["conversations"]:
        if step["from"] == "gpt":
            response = step["value"]

# Input, system_message + state, output-> <think>...</think> + action
# parse_action(response)
# reward correct function,

## Example of model response
response = """
<function=str_replace_editor>
<parameter=command>view</parameter>
<parameter=path>/testbed/conan/tools/files/files.py</parameter>
<parameter=view_range>[432, 455]</parameter>
</function>
"""

function, params = parse_action(response)
print(f"Function: {function}")
print(f"Parameters: {params}")