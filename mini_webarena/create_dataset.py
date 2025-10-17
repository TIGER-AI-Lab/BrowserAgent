# Contain functions to create dataset from raw data
# create_dataset.py

import os
import csv
import argparse
import random

from datasets import Dataset

import json

TEMPLATES = {
    'qwen-instruct': {
        "system":(
        # ---- system message ----
        "<|im_start|>system\n"
        "Here's the information you'll have:\n"
        "The user's objective: This is the task you're trying to complete.\n"
        "The current web page's accessibility tree: This is a simplified representation of the webpage, providing key information.\n"
        "The current web page's URL: This is the page you're currently navigating.\n"
        "The open tabs: These are the tabs you have open.\n"
        "The previous action: This is the action you just performed. It may be helpful to track your progress.\n\n"

        "The actions you can perform fall into several categories:\n\n"

        "Page Operation Actions:\n"
        "`click [id]`: This action clicks on an element with a specific id on the webpage.\n"
        "`type [id] [content] [press_enter_after=0|1]`: Use this to type the content into the field with id. By default, the \"Enter\" key is pressed after typing unless press_enter_after is set to 0.\n"
        "`hover [id]`: Hover over an element with id.\n"
        "`press [key_comb]`:  Simulates the pressing of a key combination on the keyboard (e.g., Ctrl+v).\n"
        "`scroll [down|up]`: Scroll the page up or down.\n\n"

        "Tab Management Actions:\n"
        "`new_tab`: Open a new, empty browser tab.\n"
        "`tab_focus [tab_index]`: Switch the browser's focus to a specific tab using its index.\n"
        "`close_tab`: Close the currently active tab.\n\n"

        "URL Navigation Actions:\n"
        "`goto [url]`: Navigate to a specific URL.\n"
        "`go_back`: Navigate to the previously viewed page.\n"
        "`go_forward`: Navigate to the next page (if a previous 'go_back' action was performed).\n\n"

        "Completion Action:\n"
        "`stop [answer]`: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. If you believe the task is impossible to complete, provide the answer as \"N/A\" in the bracket.\n\n"

        "To be successful, it is very important to follow the following rules:\n"
        "1. You should only issue an action that is valid given the current observation.\n"
        "2. You should only issue one action at a time.\n"
        "3. You should follow the examples to reason step by step and then issue the next action.\n"
        "4. All reasoning must be inside `<think></think>` tags, and there must be no output before `<think></think>`.\n"
        "5. After `<think></think>`, only the action should be generated in the correct format, enclosed in code fences. For example:\n"
        "   <think>This button looks relevant to my goal. Clicking it should take me to the next step.</think>\n"
        "   ```click [1234]```\n"
        "6. Issue the stop action when you think you have achieved the objective. Don’t generate anything after stop.\n"
        """7. Always format actions correctly: 
```command [parameters]```
For example, if searching for "death row inmates in the US" in a search field with ID `21`, correctly format it as:
```type [21] [death row inmates in the US] [1]```
Avoid incorrect formats that omit brackets around parameters or numeric values.\n\n"""
"<|im_end|>\n"
        ),
        # ---- user message ----
        "user":("<|im_start|>user\n"
        "Objective: {objective}\n\n"
        "URL: {url}\n"
        "Observation:\n"
        "{observation}\n"
        "Parsed Previous Action:\n"
        "{previous_action}\n"
        "<|im_end|>\n\n"
        ),
        "assistant":"<|im_start|>assistant {pred}\n<|im_end|>",
     },
}
WIKI_LANDING = """<|im_start|>user
Objective: {objective}

URL: http://localhost:22015/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing
Observation:
[1] RootWebArea 'User:The other Kiwix guy/Landing' focused: True
        [21] textbox "Search 'Wikipedia'" required: False
        [23] link 'Go to welcome page'
                [30] button '🏠'
        [24] link "Go to the main page of 'Wikipedia'"
                [32] button 'Wikipedia'
        [25] link 'Go to a randomly selected page'
                [34] button '🎲'
        [82] StaticText 'Welcome to '
        [83] link 'Wikipedia'
        [84] StaticText 'The free encyclopedia.'
        [371] StaticText '6,489,052'
        [86] StaticText ' articles in '
        [369] link 'English'
        [53] heading 'Arts'
        [89] link 'Architecture'
        [91] link 'Books'
        [93] link 'Cinematography'
        [95] link 'Dance'
        [97] link 'Design'
        [99] link 'Fashion'
        [101] link 'Films'
        [103] link 'Gastronomy'
        [105] link 'Literature'
        [107] link 'Magic (illusion)'
        [109] link 'Music'
        [111] link 'Painting'
        [113] link 'Photography'
        [115] link 'Poetry'
        [117] link 'Sculpture'
        [119] link 'Theatre'
        [55] heading 'Geography'
        [122] link 'Africa'
        [124] link 'Antarctica'
        [126] link 'Arctic'
        [128] link 'Asia'
        [130] link 'Caribbean'
        [132] link 'Central America'
        [134] link 'Europe'
        [136] link 'Latin America'
        [138] link 'Mediterranean'
        [140] link 'Middle East'
        [142] link 'North America'
        [144] link 'Oceania'
        [146] link 'South America'
        [148] link 'Cartography'
        [57] heading 'History'
        [150] link 'Ancient Egypt'
        [152] link 'Ancient Greece'
        [154] link 'Ancient Near East'
        [156] link 'Ancient Rome'
        [158] link 'Archaeology'
        [160] link 'British Empire'
        [162] link 'Byzantine Empire'
        [164] link 'Colonialism'
        [166] link 'Crusades'
        [168] link 'Heraldry'
        [170] link 'History of science'
        [172] link 'Imperial China'
        [174] link 'Indian independence movement'
        [176] link 'Japan'
        [178] link 'Middle Ages'
        [180] link 'Mughal Empire'
        [182] link 'Ottoman Empire'
        [184] link 'Russian Empire'
        [186] link 'Sasanian Empire'
        [188] link 'Seljuk Empire'
        [190] link 'Soviet Union'
        [192] link 'War'
        [59] heading 'Sciences'
Parsed Previous Action:
None
<|im_end|>\n\n
"""
def main():
    parser = argparse.ArgumentParser(description="Generate puzzle dataset from CSV.")
    parser.add_argument("--output_dir", type=str, default="data/puzzle", help="Output directory.")
    # parser.add_argument("--train_size", type=int, default=900000, help="How many training instances to take.")
    # parser.add_argument("--test_size", type=int, default=1500, help="How many test instances to take.")
    parser.add_argument("--prefix", type=str, default='qwen-instruct', choices=['qwen-instruct'])
    # parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 可自行决定哪些模板级别要放到数据集中
    # Template_List = [None, "Simple", "Midium", "Hard", "Extreme"]
    Template_List = [None]

    # 读取 CSV
    # data_list = []
    # with open(args.input_csv, "r", encoding="utf-8") as f:
    #     reader = csv.DictReader(f)
    #     for row in reader:
    #         data_list.append(row)
    from datasets import load_dataset

    dataset = load_dataset("RUC-NLPIR/FlashRAG_datasets", "nq")
    print(dataset)

    def build_dataset(data_split, split_name):
        """
        data_split: 对应 train_data / dev_data / test_data
        split_name: 'train' / 'dev' / 'test'
        """
        instances = []
        index_offset = 0

        for row_i, row in enumerate(data_split):
            question = row["question"]
            golden_answers = row["golden_answers"]
            # 有些条目可能没有答案，这里简单处理一下
            if len(golden_answers) == 0:
                answer = "N/A"
            else:
                answer = golden_answers[0]

            # 分拆 System / User Prompt
            system_prompt = (
                "Here's the information you'll have:\n"
                "The user's objective: This is the task you're trying to complete.\n"
                "The current web page's accessibility tree: This is a simplified representation of the webpage,\n"
                "  providing key information.\n"
                "The current web page's URL: This is the page you're currently navigating.\n"
                "The open tabs: These are the tabs you have open.\n"
                "The previous action: This is the action you just performed.\n\n"
                "The actions you can perform fall into several categories:\n\n"

                "Page Operation Actions:\n"
                "`click [id]`: This action clicks on an element with a specific id on the webpage.\n"
                "`type [id] [content] [press_enter_after=0|1]`: Use this to type the content into the field with id.\n"
                "  By default, the \"Enter\" key is pressed after typing unless press_enter_after is set to 0.\n"
                "`hover [id]`: Hover over an element with id.\n"
                "`press [key_comb]`: Simulates the pressing of a key combination on the keyboard (e.g., Ctrl+v).\n"
                "`scroll [direction=down|up]`: Scroll the page up or down.\n\n"

                "Tab Management Actions:\n"
                "`new_tab`: Open a new, empty browser tab.\n"
                "`tab_focus [tab_index]`: Switch the browser's focus to a specific tab using its index.\n"
                "`close_tab`: Close the currently active tab.\n\n"

                "URL Navigation Actions:\n"
                "`goto [url]`: Navigate to a specific URL.\n"
                "`go_back`: Navigate to the previously viewed page.\n"
                "`go_forward`: Navigate to the next page (if a previous 'go_back' action was performed).\n\n"

                "Completion Action:\n"
                "`stop [answer]`: Issue this action when you believe the task is complete. If the objective is to find a\n"
                "text-based answer, provide the answer in the bracket. If you believe the task is impossible to complete,\n"
                "provide the answer as \"N/A\" in the bracket.\n\n"

                "To be successful, it is very important to follow the following rules:\n"
                "1. You should only issue an action that is valid given the current observation.\n"
                "2. You should only issue one action at a time.\n"
                "3. You should follow the examples to reason step by step and then issue the next action.\n"
                "4. All reasoning must be inside `<think></think>` tags, and there must be no output before `<think></think>`.\n"
                "5. After `<think></think>`, only the action should be generated in the correct format, enclosed in code fences.\n"
                "   For example:\n"
                "   <think>This button looks relevant to my goal. Clicking it should take me to the next step.</think>\n"
                "   ```click [1234]```\n"
                "6. Issue the stop action when you think you have achieved the objective. Don’t generate anything after stop.\n"
            )

            # 将 NQ 的问题放入 OBJECTIVE 字段
            user_prompt = (
                "OBJECTIVE: {question}\n\n"
                "URL: {url}\n"
                f"OBSERVATION:\n"
                "{observation}\n"
            )

            # 使用我们在模板里定义的 qwen-instruct 形式，来拼出整个大字符串
            # 但这里仅演示如何分拆后再拼接，若有需要可直接使用下面的 prompt_full
            # prompt_full = TEMPLATES[args.prefix].format(
            #     objective=question,
            #     url="{url}",
            #     observation="{observation}"
            # )

            # 为了兼容你的数据结构，这里把 system 和 user 都放到 "prompt" 字段中
            instance = {
                "data_source": "wiki_qa",
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "ability": "wiki",
                "reward_model": {
                    "style": "rule",
                },
                "extra_info": {
                    "split": split_name,
                    "seed": row_i + index_offset,
                    "index": row_i + index_offset,
                    "question": question,
                    "golden_answers": golden_answers,
                    "selected_answer": answer,
                }
            }

            instances.append(instance)
            index_offset += 1

        return instances

    # 将数据集划分到 train/dev/test
    train_data = dataset["train"]
    dev_data = dataset["dev"]
    test_data = dataset["test"]

    train_instances = build_dataset(train_data, "train")
    dev_instances = build_dataset(dev_data, "dev")
    test_instances = build_dataset(test_data, "test")

    # 转成HF Dataset
    train_dataset = Dataset.from_list(train_instances)
    dev_dataset = Dataset.from_list(dev_instances)
    test_dataset = Dataset.from_list(test_instances)

    # 保存为 parquet
    train_dataset.to_parquet(os.path.join(args.output_dir, "train.parquet"))
    dev_dataset.to_parquet(os.path.join(args.output_dir, "dev.parquet"))
    test_dataset.to_parquet(os.path.join(args.output_dir, "test.parquet"))

    print("Done! Train size:", len(train_dataset), "Dev size:", len(dev_dataset), "Test size:", len(test_dataset))

if __name__ == "__main__":
    main()

# Here is a example output
"""<|im_start|>system
You are an autonomous intelligent agent tasked with navigating a web browser. You will be given web-based tasks. These tasks will be accomplished through the use of specific actions you can issue.

Here's the information you'll have:
The user's objective: This is the task you're trying to complete.
The current web page's accessibility tree: This is a simplified representation of the webpage, providing key information.
The current web page's URL: This is the page you're currently navigating.
The open tabs: These are the tabs you have open.
The previous action: This is the action you just performed. It may be helpful to track your progress.

The actions you can perform fall into several categories:

Page Operation Actions:
`click [id]`: This action clicks on an element with a specific id on the webpage.
`type [id] [content] [press_enter_after=0|1]`: Use this to type the content into the field with id. By default, the "Enter" key is pressed after typing unless press_enter_after is set to 0.
`hover [id]`: Hover over an element with id.
`press [key_comb]`:  Simulates the pressing of a key combination on the keyboard (e.g., Ctrl+v).
`scroll [direction=down|up]`: Scroll the page up or down.

Tab Management Actions:
`new_tab`: Open a new, empty browser tab.
`tab_focus [tab_index]`: Switch the browser's focus to a specific tab using its index.
`close_tab`: Close the currently active tab.

URL Navigation Actions:
`goto [url]`: Navigate to a specific URL.
`go_back`: Navigate to the previously viewed page.
`go_forward`: Navigate to the next page (if a previous 'go_back' action was performed).

Completion Action:
`stop [answer]`: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. If you believe the task is impossible to complete, provide the answer as "N/A" in the bracket.

To be successful, it is very important to follow the following rules:

1. You should only issue an action that is valid given the current observation.
2. You should only issue one action at a time.
3. You should follow the examples to reason step by step and then issue the next action.
4. All reasoning must be inside `<think></think>` tags, and there must be no output before `<think></think>`.
5. After `<think></think>`, only the action should be generated in the correct format, enclosed in code fences. For example:
   `<think>This button looks relevant to my goal. Clicking it should take me to the next step.</think>`
   ```click [1234]```
6. Issue the stop action when you think you have achieved the objective. Don’t generate anything after stop.<|im_end|>
OBSERVATION:
[1744] link 'HP CB782A#ABA 640 Inkjet Fax Machine (Renewed)'
                [1749] StaticText '$279.49'
                [1757] button 'Add to Cart'
                [1760] button 'Add to Wish List'
                [1761] button 'Add to Compare'
URL: http://onestopmarket.com/office-products/office-electronics.html
OBJECTIVE: What is the price of HP Inkjet Fax Machine
PREVIOUS ACTION: None<|im_end|>
<|im_start|>assistant
<think>This page lists the information of an HP Inkjet Fax Machine, which is the product identified in the objective. Its price is $279.49. I have achieved the objective.</think>```stop [$279.49]```<|im_end|>
<|im_start|>user
OBSERVATION:
[164] textbox 'Search' focused: True required: False
[171] button 'Go'
[174] link 'Find directions between two points'
[212] heading 'Search Results'
[216] button 'Close'
URL: http://openstreetmap.org
OBJECTIVE: Show me the restaurants near CMU
PREVIOUS ACTION: None<|im_end|>
<|im_start|>assistant
<think>This page has a search box whose ID is [164]. According to the Nominatim rule of OpenStreetMap, I can search for restaurants near a location by using "restaurants near". I can submit my typing by pressing Enter afterwards.</think>```type [164] [restaurants near CMU] [1]```<|im_end|>
<|im_start|>user
OBSERVATION:
[1022] RootWebArea 'Search: who replaced Bradie Tennell in 2021 Skate America' focused: True
        [1629] StaticText '🔍'
        [1182] textbox "Search 'Wikipedia'" required: False
        [1184] link 'Go to welcome page'
                [1631] button '🏠'
        [1185] link "Go to the main page of 'Wikipedia'"
                [1633] button 'Wikipedia'
        [1186] link 'Go to a randomly selected page'
                [1635] button '🎲'
        [1035] StaticText 'Results '
        [1037] StaticText '1-25 '
        [1038] StaticText 'of '
        [1040] StaticText '30 '
        [1041] StaticText 'for '
        [1043] StaticText '"who replaced Bradie Tennell in 2021 Skate America"'
        [1077] link 'Bradie Tennell'
        [1188] StaticText '...'
        [1191] StaticText '). Bradie Tennell Tennell at the 2018 Internationaux de France Personal information Country represented \xa0United States Born (1998-01-31) January 31, 1998 Winfield, Illinois Home town Carpentersville, Illinois Residence Colorado Springs, Colorado Height 1.68\xa0m (5\xa0ft 6\xa0'
        [1194] StaticText ') Coach Tom Zakrajsek Former coach Denise Myers Jeremy Allen Yevgeny Martynov Choreographer Benoît Richaud Former choreographer Scott Brown Shanetta Folle Cindy Stuart Skating club Skokie Valley FSC Former skating club......'
        [1195] StaticText 'from Wikipedia'
        [1196] StaticText '11,157 words'
        [1081] link '2017 Skate America'
        [1204] StaticText ' The 2017 '
        [1210] StaticText ' was the sixth event of six '
        [1213] StaticText ' the 2017–18 ISU Grand Prix of Figure Skating, a senior-level international invitational competition series. It was held '
        [1216] StaticText ' Lake Placid, New York on November 24–26. Medals were awarded '
        [1221] StaticText 'Skate'
        [1224] StaticText 'America'
        [1225] StaticText ' Type: Grand Prix Date: November 24 – 26 Season: 2017–18......'
        [1226] StaticText 'from Wikipedia'
        [1227] StaticText '1,830 words'
        [1085] link 'Mariah Bell'
        [1229] StaticText '...the 2020 '
        [1231] StaticText 'Skate'
        [1234] StaticText 'America'
        [1235] StaticText ' gold medalist, 2016 '
        [1241] StaticText ' silver medalist, the 2019 Internationaux de France bronze medalist, the 2019 Rostelecom Cup bronze medalist, the 2019 CS Nebelhorn Trophy champion, and the 2016 CS U.S. International Classic silver medalist. Mariah Bell Bell at the 2019 Internationaux de France medal ceremony Personal information Full name Mariah Cheyenne Bell Country represented United States Born (1996-04-18) April 18, 1996 Tulsa, Oklahoma Home town Westminster, Colorado......'
        [1242] StaticText 'from Wikipedia'
        [1243] StaticText '5,164 words'
        [1089] link 'Karen Chen'
        [1245] StaticText '...'
        [1247] StaticText '2021'
        [1248] StaticText '). Karen Chen Chen at 2017 '
        [1250] StaticText 'Skate'
        [1251] StaticText ' Canada Personal information Country represented United States Born (1999-08-16) August 16, 1999 Fremont, California Home town Ithaca, New York Residence Colorado Springs, Colorado Height 1.55\xa0m (5\xa0ft 1\xa0'
        [1254] StaticText ') Coach Tammy Gambill Former coach Gilley Nicholson Sherri Krahne-Thomas Choreographer Drew Meekins Pasquale Camerlengo Karen Chen Former choreographer Ilona Melnichenko Massimo Scali Rohene Ward Marina Zoueva Mark Pillay Jonathan Cassar Justin Dillon......'
        [1255] StaticText 'from Wikipedia'
        [1256] StaticText '5,834 words'
        [1093] link '2021 U.S. Figure Skating Championships'
        [1260] StaticText ' U.S. Figure Skating Championships The '
        [1263] StaticText ' Toyota U.S. Figure Skating Championships were held from January 11–21, '
        [1266] StaticText ' at the Orleans Arena '
        [1269] StaticText ' Las Vegas, Nevada.[1] Medals were awarded '
        [1274] StaticText '2021'
        [1275] StaticText ' World Championships. It would also have been part of the selection criteria for the '
        [1278] StaticText ' World Junior Championships and the '
        [1281] StaticText '......'
        [1282] StaticText 'from Wikipedia'
        [1283] StaticText '4,591 words'
        [1097] link '2019–20 ISU Grand Prix of Figure Skating'
        [1285] StaticText '...'
        [1287] StaticText 'Skate'
        [1290] StaticText 'America'
        [1291] StaticText ' Las Vegas, Nevada, United States Details October 25–27 2019 '
        [1294] StaticText ' Canada Kelowna, British Columbia, Canada Details November 1–3 2019 Internationaux de France Grenoble, France Details November 8–10 2019 Cup of China Chongqing, China Details November 15–17 2019 Rostelecom Cup Moscow, Russia Details November 22–24 2019 NHK Trophy Sapporo, Japan Details December 5–8 2019–20 Grand Prix Final Turin, Italy Details Requirements Skaters were eligible to compete on the senior......'
URL: http://localhost:8888/search?content=wikipedia_en_all_maxi_2022-05&pattern=who+replaced+Bradie+Tennell+in+2021+Skate+America
OBJECTIVE: Search online to determine who replaced Bradie Tennell in the 2021 Skate America?
PREVIOUS ACTION: None<|im_end|>
<|im_start|>assistant
"""





# Here is the example used in the soup puzzle:

"""
(main_task pid=887984) You are playing a turtle-soup puzzle.
(main_task pid=887984) 
(main_task pid=887984) Puzzle statement:
(main_task pid=887984) The doors of traditional Japanese houses generally open inward, while the doors of modern apartment-style houses generally open outward. Why?
(main_task pid=887984) 
(main_task pid=887984) In each turn, you can take exactly **one** of the following actions:
(main_task pid=887984) 1) `<ask> [Your Yes/No question] </ask>`
(main_task pid=887984) 2) `<guess> [Your final guess for the puzzle solution] </guess>`
(main_task pid=887984) 
(main_task pid=887984) Please show your chain-of-thought in <think>...</think>, then output exactly one action.
(main_task pid=887984) Example:
(main_task pid=887984) <think> I will reason about the puzzle... </think>
(main_task pid=887984) <ask> Is the victim already dead when found? </ask>
(main_task pid=887984) 
(main_task pid=887984) Or, if you want to guess the solution:
(main_task pid=887984) <think> I believe the solution is that he was killed by glass shards... </think>
(main_task pid=887984) <guess> He was cut by broken glass during the break-in. </guess>
(main_task pid=887984) 
(main_task pid=887984) Remember: only ONE action per turn.
(main_task pid=887984) 
(main_task pid=887984) Always output: <think> [Your thoughts] </think> <answer> [Your next question] </answer> with no extra text. Strictly follow this format.
(main_task pid=887984) <|im_end|>
(main_task pid=887984) <|im_start|>assistant
(main_task pid=887984) <think> Traditional Japanese houses are designed for a compact living space, often with limited indoor storage. The inward-opening doors of these houses prevent items from spilling out and cluttering the interiors. In contrast, modern apartment-style houses, designed for larger spaces and more open living areas, often have outward-opening doors to allow easier movement of furniture and cleaning.</think>
(main_task pid=887984) <answer> Are traditional Japanese houses known for their limited storage space compared to modern apartment-style houses? </answer>
"""