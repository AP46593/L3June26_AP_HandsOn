One-time setup - 
Install Python, Ollama, UV(optional)

Pull manifest for the ollama models you would be using. 
check/update config file to setup models specific for each agent.
command - ollama pull gpt-oss:120b-cloud

Create venv to run apps - 
 uv venv .venv1 
Activate venv -
.venv1\Scripts\activate

Install all requirements in .venv1
uv pip install -r requirements.txt

------
Check and update config.py file to customise llm models, token limit, temperature etc.
also setup the following flags, per your preference.
Opik tracing; set false to disable tracing. if true, then add api key on .env file. 
Web search; set duckduckgo for free (no key), if tavily, then add api key on .env file.
Logging; set false to disable. If true, writes all chat logs and agentlogs to \Userchat\ folder

** check .env.example to build your own .env file **

------
To run the apps:

In a New terminal run command to start ollama service 
command - Ollama serve 

In another new terminal - 
run specific py chat apps like 
py pyt1.py or py pyt5.py etc. 


------
To generate graph view for PYt5.py:
open pyt5_graph.ipynb
select out .venv1 python environment on kernal

