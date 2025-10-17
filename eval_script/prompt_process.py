import re
from transformers import AutoTokenizer

# 初始化tokenizer
model_name = "Qwen/Qwen2.5-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

def _normalize_browser_tags(text: str) -> str:
    """
    把 <browser>…</browser> 块转成 user 消息：
      <browser>        -> <|im_start|>user\n
      </browser>       -> <|im_end|>
    其余内容保持不变
    """
    text = text.replace("<browser>", "<|im_start|>user\n")
    text = text.replace("</browser>", "<|im_end|>")
    return text


def tuncate_input(input_str, max_tokens=4096):
    """
    将对话字符串解析为消息列表，每条消息形如：
      <|im_start|>{role}
      {content}
      <|im_end|>

    保留第一条 system 消息和最后一条 assistant 消息，
    如果整个对话（包含折叠占位符）经过 tokenizer 编码后的 token 数超过 max_tokens，
    则从系统消息和最后消息之间的候选消息中，
    按顺序移除最前面的消息，并累计被移除部分的 token 数。

    组装时：
      - 系统消息和中间消息均使用 <|im_start|> 与 <|im_end|> 包裹；
      - 如果有折叠的部分，则在系统消息之后插入一行纯文本，
        格式为 "…[X tokens folded]…"（不使用 <|im_start|> 包裹）；
      - 最后一条消息以 <|im_start|>assistant 开始，但结尾不附加 <|im_end|>。
    """
    input_str = _normalize_browser_tags(input_str)
    # 解析消息（按照 <|im_start|> 拆分）
    raw_parts = input_str.split("<|im_start|>")
    messages = []
    for part in raw_parts:
        part = part.strip()
        if not part:
            continue
        # 消息格式：第一行为 role，其余为内容，遇到 <|im_end|> 则结束
        if "\n" in part:
            role_line, rest = part.split("\n", 1)
        else:
            role_line = part
            rest = ""
        if "<|im_end|>" in rest:
            content = rest.split("<|im_end|>")[0].strip()
        else:
            content = rest.strip()
        messages.append({"role": role_line.strip(), "content": content})

    if not messages:
        return ""

    # 保留第一条消息（若为 system）和最后一条消息（最终必须为 assistant）
    first_msg = messages[0]
    final_msg = messages[-1]

    # 中间候选消息：除第一条和最后一条之外的部分
    candidate = messages[1:-1] if len(messages) > 2 else []

    def assemble_conversation(candidate_list, folded_token_count):
        parts = []
        # 添加系统消息（第一条），始终用 <|im_start|> 与 <|im_end|> 包裹
        parts.append(f"<|im_start|>{first_msg['role']}\n{first_msg['content']}<|im_end|>")
        # 如果有折叠部分，则插入折叠文本（不使用包裹）
        if folded_token_count > 0:
            parts.append(f"…[{folded_token_count} tokens folded]…")
        # 添加候选消息
        for msg in candidate_list:
            parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
        # 最后一条消息：确保角色为 assistant，且仅使用 <|im_start|> 包裹，结尾不加 <|im_end|>
        final_role = final_msg['role']
        if final_role != "assistant":
            final_role = "assistant"
        parts.append(f"<|im_start|>{final_role}\n{final_msg['content']}")
        # 使用换行连接各部分
        return "\n".join(parts)

    # 初始组装
    folded_tokens = 0
    assembled = assemble_conversation(candidate, folded_tokens)
    total_tokens = len(tokenizer.encode(assembled))

    # 如果总 token 数超过限制，则从候选消息中依次移除最前面的消息
    candidate_copy = candidate.copy()
    while total_tokens > max_tokens and candidate_copy:
        removed_msg = candidate_copy.pop(0)
        removed_tokens = len(tokenizer.encode(removed_msg["content"]))
        folded_tokens += removed_tokens
        assembled = assemble_conversation(candidate_copy, folded_tokens)
        total_tokens = len(tokenizer.encode(assembled))

    return assembled

def tuncate_plain(input_str, max_tokens=4096):
    # 1) 从输入中提取 system 内容（如果存在）
    system_pattern = r"<\|im_start\|>system\s*(.*?)<\|im_end\|>"
    match = re.search(system_pattern, input_str, flags=re.DOTALL)
    system_text = ""
    if match:
        # 仅保留 system 的纯文本内容
        system_text = match.group(1).strip()
        # 删除原文本中的 system 段，避免重复处理
        input_str = input_str[:match.start()] + input_str[match.end():]

    # 2) 删除其他 role 标记
    remove_list = [
        "<|im_start|>system",
        "<|im_start|>user",
        "<|im_start|>assistant",
        "<|im_end|>"
    ]
    for to_remove in remove_list:
        input_str = input_str.replace(to_remove, "")

    # 编码为 tokens，计算长度
    system_tokens = tokenizer.encode(system_text)
    rest_tokens = tokenizer.encode(input_str)
    total_len = len(system_tokens) + len(rest_tokens)

    # 3) 若未超限，直接拼接
    if total_len <= max_tokens:
        if system_text:
            return system_text + "\n" + input_str
        else:
            return input_str

    # 若超出 max_tokens，先保留 system，再只截取够剩余空间的末尾部分
    remain_for_rest = max_tokens - len(system_tokens)
    if remain_for_rest <= 0:
        # 如果 system 自身就已超长，则只保留 system 的最后 max_tokens 个 token
        truncated_system_tokens = system_tokens[-max_tokens:]
        return tokenizer.decode(truncated_system_tokens)
    else:
        # 截取剩余内容尾部
        truncated_rest_tokens = rest_tokens[-remain_for_rest:]
        if system_text:
            return system_text + "\n" + tokenizer.decode(truncated_rest_tokens)
        else:
            return tokenizer.decode(truncated_rest_tokens)

def direct_plain(input_str, max_tokens=4096):
    remove_list = [
        "<|im_start|>system",
        "<|im_start|>user",
        "<|im_start|>assistant",
        "<|im_end|>"
    ]
    for to_remove in remove_list:
        input_str = input_str.replace(to_remove, "")
    tokens = tokenizer.encode(input_str)
    if len(tokens) <= max_tokens:
        return input_str
    truncated_tokens = tokens[-max_tokens:]
    return tokenizer.decode(truncated_tokens)

example_input = """<|im_start|>system
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
<|im_end|>
7. Always format actions correctly: 
```command [parameters]```
For example, if searching for ""death row inmates in the US"" in a search field with ID `21`, correctly format it as:
```type [21] [death row inmates in the US] [1]```
Avoid incorrect formats that omit brackets around parameters or numeric values.

<|im_start|>user
Objective: big little lies season 2 how many episodes

URL: http://localhost:22015/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing
Observation:
[1] RootWebArea 'User:The other Kiwix guy/Landing' focused: True
	[21] textbox ""Search 'Wikipedia'"" required: False
	[23] link 'Go to welcome page'
		[30] button '🏠'
	[24] link ""Go to the main page of 'Wikipedia'""
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
<|im_end|>

<|im_start|>assistant <think>To find out how many episodes are in season 2 of ""Big Little Lies,"" I should use the search functionality on this page. The search textbox has the ID 21. I will type the query ""Big Little Lies season 2 episodes"" into the search box and press enter to search.</think>
```type [21] [Big Little Lies season 2 episodes] [1]```
<|im_end|><|im_start|>user
Objective: big little lies season 2 how many episodes

URL: http://localhost:22015/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing
Observation:
[511] RootWebArea 'Search: Big Little Lies season 2 episodes' focused: True
	[679] textbox ""Search 'Wikipedia'"" required: False
	[681] link 'Go to welcome page'
		[1098] button '🏠'
	[682] link ""Go to the main page of 'Wikipedia'""
		[1100] button 'Wikipedia'
	[683] link 'Go to a randomly selected page'
		[1102] button '🎲'
	[524] StaticText 'Results 1-25 of 9,000 for ""Big Little Lies season 2 episodes""'
	[570] link 'Big Little Lies (TV series)'
	[685] StaticText '...Big Little Lies is an American drama television series based on the 2014 novel of the same name by Liane Moriarty. Created and written by David E. Kelley, it aired on HBO from February 19, 2017, to July 21, 2019,[2][3] encompassing 14 episodes and two seasons. Originally billed as a miniseries, Jean-Marc Vallée directed the first season, while Andrea Arnold directed the second season.[4][5] Big Little Lies Genre Drama Mystery Black comedy[1] Created by David E. Kelley Based on Big Little Lies......'
	[704] StaticText 'from Wikipedia'
	[705] StaticText '11,579 words'
	[574] link 'Sarah Sokolovic'
	[707] StaticText '...Big Little Lies Early life Sokolovic graduated in 2011 from the Yale School of Drama.[1] Her father is Dimso ""Dan"" Sokolovic, of Serbian descent who immigrated to the United States from Germany, and her mother is Donna Stowell, an American of German and English descent.[2][3][4] She has a brother Stevan and a sister Stephanie. Career Theatre work Sokolovic started her career in theatre, she performed the role Betty in the musical The Shaggs: Philosophy of the World at Playwrights Horizons in......'
	[714] StaticText 'from Wikipedia'
	[715] StaticText '723 words'
	[578] link 'Darby Camp'
	[731] StaticText '1,691 words'
	[582] link 'Little Big Girl'
	[733] StaticText '...Big Man. The last time the title was parodied was in season 11\'s ""Little Big Mom."" This article is about the Simpsons episode. For the Danish film, see Little Big Girl (film). ""Little Big Girl"" The Simpsons episode Promotional artwork for the episode, featuring Bart and Darcy. Episode no. Season\xa018 Episode 12 Directed by Raymond S. Persi Written by Don Payne Production code JABF04 Original air date February\xa011,\xa02007\xa0(2007-02-11) Guest appearance Natalie Portman as Darcy Episode features......'
	[752] StaticText 'from Wikipedia'
	[753] StaticText '1,033 words'
	[586] link 'Crystal R. Fox'
	[755] StaticText '...her big-screen debut playing Katie Bell in the Academy Award-winning comedy-drama film Driving Miss Daisy (1989) starring Morgan Freeman and Jessica Tandy. She later joined the cast of NBC police-drama series In the Heat of the Night in its third season, playing Luann Corbin. She was a regular cast member through the series finale in 1995, appearing in more than 100 episodes.[5] In the 1990s, Fox had supporting roles in films Drop Squad (1994) and Once Upon a Time...When We Were Colored (1995)......'
	[762] StaticText 'from Wikipedia'
	[763] StaticText '1,475 words'
	[590] link 'Ivy George'
	[765] StaticText '...[2][3] She then appeared in Girl Meets World and Big Little Lies.[4][5] Ivy has also appeared in quite a few films including Paranormal Activity: The Ghost Dimension, and Brimstone[6][7] In 2016, Ivy George was nominated at the 37th Young Artist Awards for the film Paranormal Activity: The Ghost Dimension[8] Filmography Film Year Title Role Notes Ref(s) 2015 Paranormal Activity: The Ghost Dimension Leila Fleege [9][10][11][12] 2015 Krampus Perchta the Cherub (voice) 2016 13 Hours: The Secret......'
Parsed Previous Action:
type [21] [Big Little Lies season 2 episodes] [1]
<|im_end|>

<|im_start|>assistant <think>There is a link titled 'Big Little Lies (TV series)' which seems relevant to the search query about the number of episodes in season 2. Clicking this link should provide more detailed information about the series, including episode counts.</think>
```click [570]```
<|im_end|><|im_start|>user
Objective: big little lies season 2 how many episodes

URL: http://localhost:22015/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing
Observation:
[1104] RootWebArea 'Big Little Lies (TV series)' focused: True
	[2246] textbox ""Search 'Wikipedia'"" required: False
	[2248] link 'Go to welcome page'
		[2254] button '🏠'
	[2249] link ""Go to the main page of 'Wikipedia'""
		[2256] button 'Wikipedia'
	[2250] link 'Go to a randomly selected page'
		[2258] button '🎲'
	[2260] heading 'Big Little Lies (TV series)' hasPopup: menu
	[2278] StaticText 'Big Little Lies is an American '
	[2280] link 'drama'
	[2281] StaticText ' television series based on the 2014 '
	[2282] link 'novel of the same name'
	[2284] link 'Liane Moriarty'
	[2285] StaticText '. Created and written by '
	[2286] link 'David E. Kelley'
	[2287] StaticText ', it aired on '
	[2288] link 'HBO'
	[2289] StaticText ' from February 19, 2017, to July 21, 2019,'
	[1115] link '[]'
	[1118] link '[]'
	[2290] StaticText ' encompassing 14 episodes and two seasons. Originally billed as a '
	[2291] link 'miniseries'
	[2293] link 'Jean-Marc Vallée'
	[2294] StaticText ' directed the first season, while '
	[2295] link 'Andrea Arnold'
	[2296] StaticText ' directed the second season.'
	[1121] link '[]'
	[1124] link '[]'
	[1126] table ''
		[5353] row ''
			[5372] columnheader 'Big Little Lies' required: False
		[5354] row ''
			[5373] gridcell '' required: False
		[1128] row ''
			[5374] rowheader 'Genre' required: False
			[1129] gridcell 'Drama Mystery Black comedy[]' required: False
				[1716] ListMarker '■ '
				[5925] link 'Drama'
				[1718] ListMarker '■ '
				[5927] link 'Mystery'
				[1719] ListMarker '■ '
				[5929] link 'Black comedy'
				[1134] link '[]'
		[5355] row ''
			[5375] rowheader 'Created by' required: False
			[5376] gridcell 'David E. Kelley' required: False
				[5936] link 'David E. Kelley'
		[5356] row ''
			[5377] rowheader 'Based on' required: False
			[5378] gridcell 'Big Little Lies by Liane Moriarty' required: False
				[5940] link 'Big Little Lies'
				[5943] link 'Liane Moriarty'
		[5357] row ''
			[5379] rowheader 'Written by' required: False
			[5380] gridcell 'David E. Kelley' required: False
		[1720] row ''
			[5381] rowheader 'Directed by' required: False
			[1721] gridcell 'Jean-Marc Vallée Andrea Arnold' required: False
				[1725] ListMarker '■ '
				[5949] link 'Jean-Marc Vallée'
				[1727] ListMarker '■ '
				[5951] link 'Andrea Arnold'
		[1728] row ''
			[5382] rowheader 'Starring' required: False
			[1729] gridcell 'Reese Witherspoon Nicole Kidman Shailene Woodley Alexander Skarsgård Adam Scott Zoë Kravitz James Tupper Jeffrey Nordling Laura Dern Kathryn Newton Iain Armitage Meryl Streep' required: False
				[1733] ListMarker '■ '
				[5954] link 'Reese Witherspoon'
				[1735] ListMarker '■ '
				[5956] link 'Nicole Kidman'
				[1737] ListMarker '■ '
				[5958] link 'Shailene Woodley'
				[1739] ListMarker '■ '
				[5960] link 'Alexander Skarsgård'
				[1741] ListMarker '■ '
				[5962] link 'Adam Scott'
				[1743] ListMarker '■ '
				[5964] link 'Zoë Kravitz'
	[2298] StaticText 'Big Little Lies stars '
	[2300] link 'Reese Witherspoon'
	[2302] link 'Nicole Kidman'
	[2304] link 'Shailene Woodley'
	[2306] link 'Laura Dern'
	[2307] StaticText ', and '
	[2308] link 'Zoë Kravitz'
	[2309] StaticText ' as five women in '
	[2310] link 'Monterey, California'
	[2311] StaticText ', who become embroiled in a murder investigation. '
	[2312] link 'Alexander Skarsgård'
	[2314] link 'Adam Scott'
	[2316] link 'James Tupper'
	[2317] StaticText ' and '
	[2318] link 'Jeffrey Nordling'
	[2319] StaticText ' also feature in supporting roles. For the second season, '
	[2320] link 'Meryl Streep'
	[2321] StaticText ' joined the main cast while '
	[2322] link 'Kathryn Newton'
	[2323] StaticText ' and '
	[2324] link 'Iain Armitage'
	[2325] StaticText ' were upgraded following their appearances in recurring capacities.'
	[2326] StaticText 'The series has received critical acclaim, particularly for its writing, directing, acting, cinematography and soundtrack. The first season received 16 '
	[2327] link 'Primetime Emmy Award'
	[2328] StaticText ' nominations and won eight, including '
	[2329] link 'Outstanding Limited Series'
	[2331] link 'directing'
	[2332] StaticText ' award for Vallée, and acting awards for Witherspoon, Kidman, Skarsgård, and Dern. The trio also won '
	[2333] link 'Golden Globe Awards'
	[2334] StaticText ' in addition to a '
	[2335] link 'Best Miniseries or Television Film'
	[2336] StaticText ' win for the series. Kidman and Skarsgård also received '
	[2337] link 'Screen Actors Guild Awards'
	[2338] StaticText ' for their performances.'
	[1814] Details ''
		[1816] DisclosureTriangle 'Cast and characters' expanded: True
			[2340] heading 'Cast and characters'
		[1819] Details ''
			[1821] DisclosureTriangle 'Main' expanded: True
				[5434] heading 'Main'
Parsed Previous Action:
click [570]
<|im_end|>

<|im_start|>assistant <think>The information about the number of episodes in season 2 might be found in the text or in a section specifically about the seasons. The text mentions that the series encompasses 14 episodes and two seasons, but it doesn't specify how many episodes are in each season. I should look for more detailed information about the seasons, possibly in a section that lists episodes or provides a breakdown of the seasons.</think>
```scroll [down]```
<|im_end|><|im_start|>user
Objective: big little lies season 2 how many episodes

URL: http://localhost:22015/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing
Observation:
[1104] RootWebArea 'Big Little Lies (TV series)' focused: True
	[2246] textbox ""Search 'Wikipedia'"" required: False
	[2248] link 'Go to welcome page'
		[2254] button '🏠'
	[2249] link ""Go to the main page of 'Wikipedia'""
		[2256] button 'Wikipedia'
	[2250] link 'Go to a randomly selected page'
		[2258] button '🎲'
	[1126] table ''
		[1728] row ''
			[5382] rowheader 'Starring' required: False
			[1729] gridcell 'Reese Witherspoon Nicole Kidman Shailene Woodley Alexander Skarsgård Adam Scott Zoë Kravitz James Tupper Jeffrey Nordling Laura Dern Kathryn Newton Iain Armitage Meryl Streep' required: False
				[1735] ListMarker '■ '
				[5956] link 'Nicole Kidman'
				[1737] ListMarker '■ '
				[5958] link 'Shailene Woodley'
				[1739] ListMarker '■ '
				[5960] link 'Alexander Skarsgård'
				[1741] ListMarker '■ '
				[5962] link 'Adam Scott'
				[1743] ListMarker '■ '
				[5964] link 'Zoë Kravitz'
				[1745] ListMarker '■ '
				[5966] link 'James Tupper'
				[1747] ListMarker '■ '
				[5968] link 'Jeffrey Nordling'
				[1749] ListMarker '■ '
				[5970] link 'Laura Dern'
				[1751] ListMarker '■ '
				[5972] link 'Kathryn Newton'
				[1753] ListMarker '■ '
				[5974] link 'Iain Armitage'
				[1755] ListMarker '■ '
				[5976] link 'Meryl Streep'
		[5358] row ''
			[5383] rowheader 'Opening theme' required: False
			[5384] gridcell '""Cold Little Heart"" by Michael Kiwanuka' required: False
				[5980] link 'Michael Kiwanuka'
		[5359] row ''
			[5385] rowheader 'Country of origin' required: False
			[5386] gridcell 'United States' required: False
		[5360] row ''
			[5387] rowheader 'Original language' required: False
			[5388] gridcell 'English' required: False
		[5361] row ''
			[5389] rowheader 'No. of seasons' required: False
				[5986] Abbr ''
			[5390] gridcell '2' required: False
		[5362] row ''
			[5391] rowheader 'No. of episodes' required: False
				[5990] Abbr ''
			[5392] gridcell '14' required: False
		[5363] row ''
			[5393] columnheader 'Production' required: False
		[1756] row ''
			[5394] rowheader 'Executive producers' required: False
			[1757] gridcell 'David E. Kelley Jean-Marc Vallée Reese Witherspoon Bruna Papandrea Nicole Kidman Per Saari Gregg Fienberg Nathan Ross Andrea Arnold Liane Moriarty' required: False
				[1761] ListMarker '■ '
				[1763] ListMarker '■ '
				[1765] ListMarker '■ '
				[5998] StaticText 'Reese Witherspoon'
				[1767] ListMarker '■ '
				[5999] link 'Bruna Papandrea'
				[1769] ListMarker '■ '
				[6001] StaticText 'Nicole Kidman'
				[1771] ListMarker '■ '
				[6002] StaticText 'Per Saari'
	[2336] StaticText ' win for the series. Kidman and Skarsgård also received '
	[2337] link 'Screen Actors Guild Awards'
	[2338] StaticText ' for their performances.'
	[1814] Details ''
		[1816] DisclosureTriangle 'Cast and characters' expanded: True
			[2340] heading 'Cast and characters'
		[1819] Details ''
			[1821] DisclosureTriangle 'Main' expanded: True
				[5434] heading 'Main'
			[1826] ListMarker '■ '
			[6061] link 'Reese Witherspoon'
			[6062] StaticText ' as Madeline Martha Mackenzie'
			[1828] ListMarker '■ '
			[6063] link 'Nicole Kidman'
			[6064] StaticText ' as Celeste Wright'
			[1830] ListMarker '■ '
			[6065] link 'Shailene Woodley'
			[6066] StaticText ' as Jane Chapman'
			[1832] ListMarker '■ '
			[6067] link 'Alexander Skarsgård'
			[1834] ListMarker '■ '
			[6069] link 'Adam Scott'
			[1836] ListMarker '■ '
			[6071] link 'Zoë Kravitz'
			[1838] ListMarker '■ '
			[6073] link 'James Tupper'
			[1840] ListMarker '■ '
			[6075] link 'Laura Dern'
			[6076] StaticText ' as Renata Klein'
			[1842] ListMarker '■ '
			[6077] link 'Jeffrey Nordling'
			[1844] ListMarker '■ '
			[6079] link 'Kathryn Newton'
			[1846] ListMarker '■ '
			[6081] link 'Iain Armitage'
			[1848] ListMarker '■ '
			[6083] link 'Meryl Streep'
		[1849] Details ''
			[1851] DisclosureTriangle 'Recurring' expanded: True
				[5436] heading 'Recurring'
			[1857] ListMarker '■ '
			[6086] link 'Darby Camp'
Parsed Previous Action:
scroll [down]
<|im_end|>

<|im_start|>assistant <think>The current page provides information about the TV series ""Big Little Lies."" The table includes a row for the number of episodes, which is listed as 14. However, this is the total number of episodes across all seasons. To find the number of episodes specifically for season 2, I need to look for more detailed information, possibly in a section that breaks down episodes by season. I will scroll further down to see if there is a section that lists episodes by season.</think>
```scroll [down]```
<|im_end|><|im_start|>user
Objective: big little lies season 2 how many episodes

URL: http://localhost:22015/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing
Observation:
[1104] RootWebArea 'Big Little Lies (TV series)' focused: True
	[2246] textbox ""Search 'Wikipedia'"" required: False
	[2248] link 'Go to welcome page'
		[2254] button '🏠'
	[2249] link ""Go to the main page of 'Wikipedia'""
		[2256] button 'Wikipedia'
	[2250] link 'Go to a randomly selected page'
		[2258] button '🎲'
	[1126] table ''
		[1756] row ''
			[5394] rowheader 'Executive producers' required: False
			[1757] gridcell 'David E. Kelley Jean-Marc Vallée Reese Witherspoon Bruna Papandrea Nicole Kidman Per Saari Gregg Fienberg Nathan Ross Andrea Arnold Liane Moriarty' required: False
				[1763] ListMarker '■ '
				[1765] ListMarker '■ '
				[1767] ListMarker '■ '
				[5999] link 'Bruna Papandrea'
				[1769] ListMarker '■ '
				[6001] StaticText 'Nicole Kidman'
				[1771] ListMarker '■ '
				[6002] StaticText 'Per Saari'
				[1773] ListMarker '■ '
				[6003] link 'Gregg Fienberg'
				[1775] ListMarker '■ '
				[6005] link 'Nathan Ross'
				[1777] ListMarker '■ '
				[6007] StaticText 'Andrea Arnold'
				[1779] ListMarker '■ '
				[6008] link 'Liane Moriarty'
		[1780] row ''
			[5395] rowheader 'Producers' required: False
			[1781] gridcell 'Barbara A. Hall David Auge' required: False
				[1785] ListMarker '■ '
				[1787] ListMarker '■ '
		[5364] row ''
			[5396] rowheader 'Production location' required: False
			[5397] gridcell 'Monterey, California' required: False
				[6014] link 'Monterey, California'
		[1788] row ''
			[5398] rowheader 'Cinematography' required: False
			[1789] gridcell 'Yves Bélanger Jim Frohna' required: False
				[1793] ListMarker '■ '
				[6017] link 'Yves Bélanger'
				[1795] ListMarker '■ '
				[6019] link 'Jim Frohna'
		[5365] row ''
			[5399] rowheader 'Camera setup' required: False
			[5400] gridcell 'Single-camera' required: False
				[6022] link 'Single-camera'
		[5366] row ''
			[5401] rowheader 'Running time' required: False
			[5402] gridcell '45–58 minutes' required: False
		[1796] row ''
			[5403] rowheader 'Production companies' required: False
			[1797] gridcell 'Hello Sunshine Blossom Films David E. Kelley Productions Crazyrose (season 2)' required: False
				[1801] ListMarker '■ '
				[6027] link 'Hello Sunshine'
				[1803] ListMarker '■ '
				[6029] link 'Blossom Films'
				[1805] ListMarker '■ '
				[6031] StaticText 'David E. Kelley Productions'
				[1807] ListMarker '■ '
				[6032] link 'Crazyrose'
				[6033] StaticText ' (season 2)'
		[1808] row ''
			[5404] rowheader 'Distributor' required: False
			[1809] gridcell 'Warner Bros. Television Distribution' required: False
				[1813] ListMarker '■ '
				[6036] link 'Warner Bros. Television Distribution'
		[5367] row ''
			[5405] columnheader 'Release' required: False
		[5368] row ''
			[5406] rowheader 'Original network' required: False
			[5407] gridcell 'HBO' required: False
				[6040] link 'HBO'
	[1814] Details ''
		[1819] Details ''
			[1846] ListMarker '■ '
			[6081] link 'Iain Armitage'
			[1848] ListMarker '■ '
			[6083] link 'Meryl Streep'
		[1849] Details ''
			[1851] DisclosureTriangle 'Recurring' expanded: True
				[5436] heading 'Recurring'
			[1857] ListMarker '■ '
			[6086] link 'Darby Camp'
			[1859] ListMarker '■ '
			[1861] ListMarker '■ '
			[1863] ListMarker '■ '
			[1865] ListMarker '■ '
			[6091] link 'Ivy George'
			[1867] ListMarker '■ '
			[6093] link 'Merrin Dungey'
			[6094] StaticText ' as Detective Adrienne Quinlan'
			[1869] ListMarker '■ '
			[6095] link 'Robin Weigert'
			[1871] ListMarker '■ '
			[6097] link 'P. J. Byrne'
			[6098] StaticText ' as Warren Nippal, the principal of Otter Bay Elementary'
			[1873] ListMarker '■ '
			[6099] link 'Santiago Cabrera'
			[6100] StaticText ' as Joseph Bachman, a theater director'
			[1875] ListMarker '■ '
			[6101] link 'Sarah Sokolovic'
			[1877] ListMarker '■ '
			[6103] link 'Kelen Coleman'
			[6104] StaticText ' as Harper Stimson'
			[1879] ListMarker '■ '
			[6105] link 'Larry Sullivan'
			[6106] StaticText ' as Oren'
			[1881] ListMarker '■ '
			[6107] link 'Gia Carides'
			[6108] StaticText ' as Melissa'
			[1883] ListMarker '■ '
			[6109] StaticText 'Larry Bates as Stu'
			[1885] ListMarker '■ '
			[6110] link 'Sarah Baker'
			[6111] StaticText ' as Thea Cunningham (season 1)'
			[1887] ListMarker '■ '
			[6112] link 'Kathreen Khavari'
			[6113] StaticText ' as Samantha (season 1)'
			[1889] ListMarker '■ '
			[6114] link 'Sarah Burns'
			[6115] StaticText ' as Gabrielle (season 1)'
Parsed Previous Action:
scroll [down]
<|im_end|>

<|im_start|>assistant <think>Since the page is about ""Big Little Lies,"" I should continue scrolling to find information about the number of episodes in season 2.</think>
```scroll [down]```
<|im_end|><|im_start|>user
Objective: big little lies season 2 how many episodes

URL: http://localhost:22015/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing
Observation:
[1104] RootWebArea 'Big Little Lies (TV series)' focused: True
	[2246] textbox ""Search 'Wikipedia'"" required: False
	[2248] link 'Go to welcome page'
		[2254] button '🏠'
	[2249] link ""Go to the main page of 'Wikipedia'""
		[2256] button 'Wikipedia'
	[2250] link 'Go to a randomly selected page'
		[2258] button '🎲'
	[1126] table ''
		[1796] row ''
			[5403] rowheader 'Production companies' required: False
			[1797] gridcell 'Hello Sunshine Blossom Films David E. Kelley Productions Crazyrose (season 2)' required: False
				[1807] ListMarker '■ '
				[6032] link 'Crazyrose'
		[1808] row ''
			[5404] rowheader 'Distributor' required: False
			[1809] gridcell 'Warner Bros. Television Distribution' required: False
				[1813] ListMarker '■ '
				[6036] link 'Warner Bros. Television Distribution'
		[5367] row ''
			[5405] columnheader 'Release' required: False
		[5368] row ''
			[5406] rowheader 'Original network' required: False
			[5407] gridcell 'HBO' required: False
				[6040] link 'HBO'
		[5369] row ''
			[5408] rowheader 'Picture format' required: False
			[5409] gridcell 'HDTV 1080i' required: False
				[6043] link 'HDTV'
				[6045] link '1080i'
		[5370] row ''
			[5410] rowheader 'Audio format' required: False
			[5411] gridcell '5.1 surround' required: False
				[6049] link '5.1 surround'
		[5371] row ''
			[5412] rowheader 'Original release' required: False
			[5413] gridcell 'February 19, 2017\xa0– July 21, 2019' required: False
				[6052] StaticText 'February 19, 2017\xa0–July 21, 2019'
	[1814] Details ''
		[1849] Details ''
			[1883] ListMarker '■ '
			[6109] StaticText 'Larry Bates as Stu'
			[1885] ListMarker '■ '
			[6110] link 'Sarah Baker'
			[6111] StaticText ' as Thea Cunningham (season 1)'
			[1887] ListMarker '■ '
			[6112] link 'Kathreen Khavari'
			[6113] StaticText ' as Samantha (season 1)'
			[1889] ListMarker '■ '
			[6114] link 'Sarah Burns'
			[6115] StaticText ' as Gabrielle (season 1)'
			[1891] ListMarker '■ '
			[6116] link 'Hong Chau'
			[6117] StaticText ' as Jackie (season 1)'
			[1893] ListMarker '■ '
			[6118] link 'David Monahan'
			[6119] StaticText ' as Bernard (season 1)'
			[1895] ListMarker '■ '
			[6120] link 'Molly Hagan'
			[1897] ListMarker '■ '
			[6122] link 'Virginia Kull'
			[1899] ListMarker '■ '
			[6124] link 'Joseph Cross'
			[1901] ListMarker '■ '
			[6126] link 'Douglas Smith'
			[6128] link 'Monterey Bay Aquarium'
			[6129] StaticText ' (season 2)'
			[1903] ListMarker '■ '
			[6130] link 'Mo McRae'
			[6131] StaticText ' as Michael Perkins, a new second-grade teacher at Otter Bay Elementary (season 2)'
			[1905] ListMarker '■ '
			[6132] link 'Crystal Fox'
			[1907] ListMarker '■ '
			[6134] link 'Martin Donovan'
			[1909] ListMarker '■ '
			[6136] link 'Poorna Jagannathan'
			[1911] ListMarker '■ '
			[6138] link ""Denis O'Hare""
			[1913] ListMarker '■ '
			[6140] link 'Becky Ann Baker'
	[1136] Details ''
		[1915] DisclosureTriangle 'Episodes' expanded: True
			[2342] heading 'Episodes'
Parsed Previous Action:
scroll [down]
<|im_end|>

<|im_start|>assistant"""

if __name__ == "__main__":
    print("tuncate_input 测试：")
    print(tuncate_input(example_input, max_tokens=4096))
    # print("\n" + "="*40 + "\n")
    # print("tuncate_plain 测试：")
    # print(tuncate_plain(example_input, max_tokens=512))
    # print("\n" + "="*40 + "\n")
    # print("direct_plain 测试：")
    # print(direct_plain(example_input, max_tokens=512))
