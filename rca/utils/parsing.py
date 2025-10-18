import re

<<<<<<< HEAD
=======

>>>>>>> ae7c2f7 (add all)
# regex function that captures string between <X= and >
def parse_string_between_tags(response, tag="function"):
    pattern = rf"<{tag}=(.*?)>(.*?)</{tag}>"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return f"<{tag}={match.group(1)}>{match.group(2)}</{tag}>"
    return None

<<<<<<< HEAD
=======

>>>>>>> ae7c2f7 (add all)
def parse_action(response, string_only=False):
    function_format = r"<function=(.*?)>(.*?)</function>"
    try:
        function_matched = re.search(function_format, response, re.DOTALL)
<<<<<<< HEAD
    except Exception as e:
=======
    except Exception:
>>>>>>> ae7c2f7 (add all)
        return None

    if function_matched:
        if string_only:
            return f"<function={function_matched.group(1)}>{function_matched.group(2)}</function>"

        function_name = function_matched.group(1)
        function_param_string = function_matched.group(2).strip()
<<<<<<< HEAD
        
=======

>>>>>>> ae7c2f7 (add all)
        param_format = r"<parameter=(.*?)>(.*?)</parameter>"
        param_matched = re.findall(param_format, function_param_string, re.DOTALL)
        if param_matched:
            params = {name: value.strip() for name, value in param_matched}
        else:
            params = {}
<<<<<<< HEAD
        
=======

>>>>>>> ae7c2f7 (add all)
        return function_name, params
    else:
        return None

<<<<<<< HEAD
if __name__ == "__main__":

=======

if __name__ == "__main__":
>>>>>>> ae7c2f7 (add all)
    ## Example of model response
    response = """
    This is an example response with a function call.
    <function=str_replace_editor>
    <parameter=command>view</parameter>
    <parameter=path>/testbed/conan/tools/files/files.py</parameter>
    <parameter=view_range>[432, 455]</parameter>
    </function>
    """

    function, params = parse_action(response)
    print(f"Function: {function}")
<<<<<<< HEAD
    print(f"Parameters: {params}")
=======
    print(f"Parameters: {params}")
>>>>>>> ae7c2f7 (add all)
