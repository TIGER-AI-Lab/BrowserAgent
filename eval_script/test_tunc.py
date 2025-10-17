example = """
"<|im_start|>system
Here's the information you'll have:
The user's objective: This is the task you're trying to complete.
The current web page's accessibility tree: This is a simplified representation of the webpage, providing key information.
The current web page's URL: This is the page you're currently navigating.
The open tabs: These are the tabs you have open.
The previous action: This is the action you just performed. It may be helpful to track your progress.

The actions you can perform fall into several categories:

Page Operation Actions:
`click [id]`: This action clicks on an element with a specific id on the webpage.
`type [id] [content] [press_enter_after=0|1]`: Use this to type the content into the field with id. By default, the ""Enter"" key is pressed after typing unless press_enter_after is set to 0.
`hover [id]`: Hover over an element with id.
`press [key_comb]`:  Simulates the pressing of a key combination on the keyboard (e.g., Ctrl+v).
`scroll [down|up]`: Scroll the page up or down.

Tab Management Actions:
`new_tab`: Open a new, empty browser tab.
`tab_focus [tab_index]`: Switch the browser's focus to a specific tab using its index.
`close_tab`: Close the currently active tab.

URL Navigation Actions:
`goto [url]`: Navigate to a specific URL.
`go_back`: Navigate to the previously viewed page.
`go_forward`: Navigate to the next page (if a previous 'go_back' action was performed).

Completion Action:
`stop [answer]`: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. If you believe the task is impossible to complete, provide the answer as ""N/A"" in the bracket.

To be successful, it is very important to follow the following rules:
1. You should only issue an action that is valid given the current observation.
2. You should only issue one action at a time.
3. You should follow the examples to reason step by step and then issue the next action.
4. All reasoning must be inside `<think></think>` tags, and there must be no output before `<think></think>`.
5. After `<think></think>`, only the action should be generated in the correct format, enclosed in code fences. For example:
   <think>This button looks relevant to my goal. Clicking it should take me to the next step.</think>
   ```click [1234]```
6. Issue the stop action when you think you have achieved the objective. Don’t generate anything after stop.
7. Always format actions correctly: 
```command [parameters]```
For example, if searching for ""death row inmates in the US"" in a search field with ID `21`, correctly format it as:
```type [21] [death row inmates in the US] [1]```
Avoid incorrect formats that omit brackets around parameters or numeric values.

<|im_end|>
<browser>Objective: the combination of carbon and oxygen would result in what type of bond

URL: http://tigerai.ca/wiki/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing
Observation:
[1] RootWebArea 'User:The other Kiwix guy/Landing' focused: True
	[16] textbox ""Search 'Wikipedia'"" required: False
	[10] checkbox '' checked: false
	[18] link 'Go to welcome page'
		[24] button '🏠'
	[19] link ""Go to the main page of 'Wikipedia'""
		[26] button 'Wikipedia'
	[20] link 'Go to a randomly selected page'
		[28] button '🎲'
	[78] StaticText 'Welcome to '
	[79] link 'Wikipedia'
	[80] StaticText 'The free encyclopedia.'
	[367] StaticText '6,489,052'
	[82] StaticText ' articles in '
	[365] link 'English'
	[49] heading 'Arts'
	[85] link 'Architecture'
	[87] link 'Books'
	[89] link 'Cinematography'
	[91] link 'Dance'
	[93] link 'Design'
	[95] link 'Fashion'
	[97] link 'Films'
	[99] link 'Gastronomy'
	[101] link 'Literature'
	[103] link 'Magic (illusion)'
	[105] link 'Music'
	[107] link 'Painting'
	[109] link 'Photography'
	[111] link 'Poetry'
	[113] link 'Sculpture'
	[115] link 'Theatre'
	[51] heading 'Geography'
	[118] link 'Africa'
	[120] link 'Antarctica'
	[122] link 'Arctic'
	[124] link 'Asia'
	[126] link 'Caribbean'
	[128] link 'Central America'
	[130] link 'Europe'
	[132] link 'Latin America'
	[134] link 'Mediterranean'
	[136] link 'Middle East'
	[138] link 'North America'
	[140] link 'Oceania'
	[142] link 'South America'
	[144] link 'Cartography'
	[53] heading 'History'
	[146] link 'Ancient Egypt'
	[148] link 'Ancient Greece'
	[150] link 'Ancient Near East'
	[152] link 'Ancient Rome'
	[154] link 'Archaeology'
	[156] link 'British Empire'
	[158] link 'Byzantine Empire'
	[160] link 'Colonialism'
	[162] link 'Crusades'
	[164] link 'Heraldry'
	[166] link 'History of science'
	[168] link 'Imperial China'
	[170] link 'Indian independence movement'
	[172] link 'Japan'
	[174] link 'Middle Ages'
	[176] link 'Mughal Empire'
	[178] link 'Ottoman Empire'
	[180] link 'Russian Empire'
	[182] link 'Sasanian Empire'
	[184] link 'Seljuk Empire'
	[186] link 'Soviet Union'
	[188] link 'War'
	[55] heading 'Sciences'
Parsed Previous Action:
None
</browser><|im_start|>assistant"
"""

from prompt_process import tuncate_input, tuncate_plain, direct_plain

print(tuncate_input(example))