from openai import OpenAI
client = OpenAI(api_key="sk-proj-1234567890", base_url="http://localhost:5004") # Replace with your local server address

completion = client.chat.completions.create(
    model="Qwen/Qwen3-4B",
    # model="Qwen/Qwen3-4B",
    messages=[{'content': """Here\'s the information you\'ll have:\nThe user\'s objective: This is the task you\'re trying to complete.\nThe current web page\'s accessibility tree: This is a simplified representation of the webpage,\n  providing key information.\nThe current web page\'s URL: This is the page you\'re currently navigating.\nThe open tabs: These are the tabs you have open.\nThe previous action: This is the action you just performed.\n\nThe actions you can perform fall into several categories:\n\nPage Operation Actions:\n`click [id]`: This action clicks on an element with a specific id on the webpage.\n`type [id] [content] [press_enter_after=0|1]`: Use this to type the content into the field with id.\n  By default, the "Enter" key is pressed after typing unless press_enter_after is set to 0.\n`hover [id]`: Hover over an element with id.\n`press [key_comb]`: Simulates the pressing of a key combination on the keyboard (e.g., Ctrl+v).\n`scroll [direction=down|up]`: Scroll the page up or down.\n\nTab Management Actions:\n`new_tab`: Open a new, empty browser tab.\n`tab_focus [tab_index]`: Switch the browser\'s focus to a specific tab using its index.\n`close_tab`: Close the currently active tab.\n\nURL Navigation Actions:\n`goto [url]`: Navigate to a specific URL.\n`go_back`: Navigate to the previously viewed page.\n`go_forward`: Navigate to the next page (if a previous \'go_back\' action was performed).\n\nCompletion Action:\n`stop [answer]`: Issue this action when you believe the task is complete. If the objective is to find a\ntext-based answer, provide the answer in the bracket. If you believe the task is impossible to complete,\nprovide the answer as "N/A" in the bracket.\n\nTo be successful, it is very important to follow the following rules:\n1. You should only issue an action that is valid given the current observation.\n2. You should only issue one action at a time.\n3. You should follow the examples to reason step by step and then issue the next action.\n4. All reasoning must be inside `<think></think>` tags, and there must be no output before `<think></think>`.\n5. After `<think></think>`, only the action should be generated in the correct format, enclosed in code fences.\n   For example:\n   <think>This button looks relevant to my goal. Clicking it should take me to the next step.</think>\n   ```click [1234]```\n6. Issue the stop action when you think you have achieved the objective. Don’t generate anything after stop.\n""", 'role': 'system'}],
    temperature=0,
    max_tokens=10240,
    top_p=1,
    n=1,
    extra_body={'golden_answers': ['Wilhelm Conrad Röntgen'],
        'gt': 'Wilhelm Conrad Röntgen',
        'id': 0,
        'index': 0,
        'question': 'who got the first nobel prize in physics',
        'split': 'test',
        'url': 'https://tigerai.ca/wiki/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing'}
)

print(completion.choices[0].message.content)