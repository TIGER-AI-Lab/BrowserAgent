# BrowserAgent
An agent that can interact with browser to complete tasks

### Installation


#### 1. **Clone this repository and navigate to the folder:**
```bash
git clone https://github.com/TIGER-AI-Lab/BrowserAgent.git
cd BrowserAgent
```


**Install the inference package:**
```bash
conda create -n browseragent python=3.10
conda activate browseragent
pip install -r requirement.txt
cd verl-tool
pip install -e .
pip install -e verl
pip install vllm==0.8.4
pip install flash-attn --no-build-isolation
pip install -e ".[acecoder,torl]"
pip uninstall uvloop
```


#### 2. Data preparation

Download [📊 BrowserAgent-Data](https://huggingface.co/datasets/TIGER-Lab/BrowserAgent-Data) and place it in the data folder, the final structure should look like this:

```
benchmark
-- | nq
-- | hotpot
-- | 2wiki
-- | popqa
-- | musique
-- | bamboogle
```


#### 3. **Deploy the wiki webpage:**

We adopt the WikiPedia from [WebArena](https://github.com/web-arena-x/webarena/tree/main/environment_docker#wikipedia-website).

To deploy the Wiki webpage locally, run the following Docker command and open http://localhost:22015.
```bash
docker run -d --name=wikipedia --volume=<your-path-to-downloaded-folder>/:/data -p 22015:80 ghcr.io/kiwix/kiwix-serve:3.3.0 wikipedia_en_all_maxi_2022-05.zim
```

For public deployment, see our live example at [tigerai.ca/wiki/.../Landing](https://tigerai.ca/wiki/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing) and apply the Nginx template in this repo, updating `proxy_pass` to match your Docker port.


```conf
location = /search {
    return 301 /wiki/search?$args;
}

location = /random {
    return 301 /wiki/random?$args;
}

location ~ ^/wikipedia_en_all_maxi_2022-05 {
    return 301 /wiki$request_uri;
}


location /wiki/ {
    proxy_pass http://localhost:22015/;
    proxy_buffering off;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;

    sub_filter_once off;

    sub_filter '<a href="/' '<a href="/wiki/';
    sub_filter '<link href="/' '<link href="/wiki/';
    sub_filter '<script src="/' '<script src="/wiki/';
    sub_filter '<img src="/' '<img src="/wiki/';

    sub_filter 'href="/search/' 'href="/wiki/search/';
    sub_filter 'href="/random/' 'href="/wiki/random/';
    sub_filter 'href="/wikipedia_en_all_maxi_2022-05/' 'href="/wiki/wikipedia_en_all_maxi_2022-05/';
    sub_filter 'action="/search' 'action="/wiki/search';
    sub_filter 'src="/wikipedia_en_all_maxi_2022-05/' 'src="/wiki/wikipedia_en_all_maxi_2022-05/';
```


#### 4. **SFT and RFT model:**

Download [📊 BroswerAgent-SFT](https://huggingface.co/TIGER-Lab/BrowserAgent-SFT) or [📊 BroswerAgent-RFT](https://huggingface.co/TIGER-Lab/BrowserAgent-RFT), and deploy using vllm.

```bash
Terminal 1:
conda activate browseragent
cd Browseragent
bash deploy_vllm.sh /path/to/your/model
```



#### 5. **Data generation and model evaluation:**

(1) For SFT data generation

```bash
Terminal 2:
conda activate browseragent
cd Browseragent
bash verl-tool\examples\train\wikiRL\wikiRL_server.sh

Terminal 3:
conda activate browseragent
cd Browseragent
python data_generate.py /path/to/your/output_file /path/to/your/sft_data_path
```

Then you can run the following code to convert the generated data into the ms-swift training format.

```bash
conda activate browseragent
cd Browseragent
python judge_sft.py /path/to/your/sft_data_path /path/to/your/previous_step_output_file /path/to/your/output_file
python swift_switch.py /path/to/your/previous_step_output_file /path/to/your/output_file
```


(2) For RFT data generation

```bash
Terminal 2:
conda activate browseragent
cd Browseragent
bash verl-tool\examples\train\wikiRL\wikiRL_server.sh

Terminal 3:
conda activate browseragent
cd Browseragent
python data_generate_rft.py /path/to/your/output_file /path/to/your/rft_data_path
```

Then you can run the following code to convert the generated data into the ms-swift training format.

```bash
conda activate browseragent
cd Browseragent
python judge_rft.py /path/to/your/sft_data_path /path/to/your/previous_step_output_file /path/to/your/output_file
python swift_switch.py /path/to/your/previous_step_output_file /path/to/your/output_file
```

(3) For model evaluation

```bash
Terminal 2:
conda activate browseragent
cd Browseragent
bash verl-tool\examples\train\wikiRL\wikiRL_server.sh

Terminal 3:
conda activate browseragent
cd Browseragent
python run_model.py /path/to/your/benchmark_path
```

Then you can run the following code to calculate the rule-based accuracy.

```bash
conda activate browseragent
cd Browseragent
python val_answer.py /path/to/your/benchmark_path /path/to/your/previous_step_output_file
```

Then you can run the following code to calculate the model-based accuracy.
```bash
conda activate browseragent
cd Browseragent
python val_answer_model_based.py /path/to/your/benchmark_path /path/to/your/previous_step_output_file /path/to/your/output_file
```


### Citation

If you find it useful for your research and applications, please cite related papers/blogs using this BibTeX:
```bibtex
@misc{yu2025browseragentbuildingwebagents,
      title={BrowserAgent: Building Web Agents with Human-Inspired Web Browsing Actions}, 
      author={Tao Yu and Zhengbo Zhang and Zhiheng Lyu and Junhao Gong and Hongzhu Yi and Xinming Wang and Yuxuan Zhou and Jiabing Yang and Ping Nie and Yan Huang and Wenhu Chen},
      year={2025},
      eprint={2510.10666},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.10666}, 
}
```
