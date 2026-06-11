import sys

sys.path.append("/user1/irlab/sagnik/lost_in_the_middle/Project/QA")

from tqdm import tqdm 
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompt_creation_qa import prompt_qa
import matplotlib.pyplot as plt
import torch


with open(r"/user1/irlab/sagnik/API_KEY","r") as f:
    TOKEN_KEY=f.read()




PATH="lost_in_the_middle/Project/QA/Data/10/nq-open-10_total_documents_gold_at_0.jsonl"


prompts=[]
for i in range(2):
    prompts.append(prompt_qa(PATH,i)[0])


def measure_attention_sink(model,prompts,tokenizer,device=torch.device("cuda")):
    
    num_layers=model.config.num_hidden_layers     # Number of layers in the model
    num_heads=model.config.num_attention_heads    # Number of attention heads per layer

    inputs=[]

    for i in tqdm(prompts):
        input=tokenizer(i,return_tensors="pt").to(device)      # return_tensors="pt" returns the output in pytorch tensor form
        inputs.append(input)

    outputs=[]

    for i in tqdm(inputs):
        output=model.generate(**i,output_attentions=True,return_dict_in_generate=True)     # **i: Unpacks the kv data stored in dictionary i and makes it ready to use as a input, return_dict_in_generate returns the output in dictionary form which is better and much structured way of outputting stuff when we are outputting stuff other than the just output tokens
        outputs.append(output)

    return [i.attentions for i in outputs],num_heads,num_layers,[i.input_ids.size() for i in inputs]






model_name="mesolitica/llama2-embedding-1b-8k"

model=AutoModelForCausalLM.from_pretrained(model_name,attn_implementation="eager",token=TOKEN_KEY).to(torch.device("cuda"))
tokenizer=AutoTokenizer.from_pretrained(model_name,token=TOKEN_KEY)

prompt_index=0
output,num_heads,num_layers,token_counts1 = measure_attention_sink(model=model,prompts=prompts,tokenizer=tokenizer)
token_counts=token_counts1[prompt_index][1]
epsilon=0.3

sum_score_layer=torch.zeros(token_counts).to(torch.device("cuda"))


for k in tqdm(range(num_layers)):
    tensor=output[prompt_index][0][k]
    attention_weights_head=torch.zeros(token_counts).to(torch.device("cuda"))


    for i in range(num_heads):
        matrix=tensor[0][i]
        importance_score=[]

        count=0
        for j in range(token_counts):
            temp=matrix[count:,j]
            mean=torch.mean(temp).item()
            importance_score.append(mean)
            count+=1
            importance_score_tensor=torch.tensor(importance_score).to(torch.device("cuda"))

        attention_weights_head+=importance_score_tensor
        
    across_head_importance_score=(1/num_heads)*attention_weights_head

    sum_score_layer+=across_head_importance_score

across_layer_importance_score=(1/num_layers)*sum_score_layer

torch.save(across_layer_importance_score,"across_layer_importance_score.pt")

### Plot

xcount=across_layer_importance_score.size()[0]

xaxis=list(range(xcount+1))[1:]

plt.figure(figsize=(8, 5))

plt.plot(
    xaxis,
    across_layer_importance_score.to(torch.device("cpu")),
    marker="o"
)

plt.xticks(xaxis)

plt.xlabel("Prompt Token Position")
plt.ylabel("Average Attention Score")
plt.title("Attention Score")

plt.grid(True)

plt.savefig("/user1/irlab/sagnik/attention_sink_analysis/Plot/attention_weights_layer.png", dpi=300, bbox_inches="tight")


print(across_layer_importance_score)

