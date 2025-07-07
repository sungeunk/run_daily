@echo off

if "%1"=="llama_v2.0_7b" (
	set hf_name="meta-llama/Llama-2-7b-chat-hf"
	set sub_model="llama2"
    set task="text-generation-with-past"
)
if "%1"=="llama_v3.0_8b" (
	set hf_name="meta-llama/Meta-Llama-3-8B-Instruct"
	set sub_model="llama3"
    set task="text-generation-with-past"
)
if "%1"=="llama_v3.1_8b" (
	set hf_name="meta-llama/Llama-3.1-8B-Instruct"
	set sub_model="llama3"
    set task="text-generation-with-past"
)
if "%1"=="llama_v3.2_1b" (
	set hf_name="meta-llama/Llama-3.2-1B-Instruct"
	set sub_model="llama3"
    set task="text-generation-with-past"
)
if "%1"=="llama_v3.2_3b" (
	set hf_name="meta-llama/Llama-3.2-3B-Instruct"
	set sub_model="llama3"
    set task="text-generation-with-past"
)
if "%1"=="ds_r1_d_llama_8b" (
	set hf_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
	set sub_model="llama3"
    set task="text-generation-with-past"
)
if "%1"=="ds_r1_d_qwen_1.5b" (
	set hf_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
	set sub_model="qwen"
    set task="text-generation-with-past"
)
if "%1"=="ds_r1_d_qwen_7b" (
	set hf_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
	set sub_model="qwen"
    set task="text-generation-with-past"
)
if "%1"=="ds_r1_d_qwen_14b" (
	set hf_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
	set sub_model="qwen"
    set task="text-generation-with-past"
)
if "%1"=="ds_r1_0528_qwen_v3.0_8b" (
	set hf_name="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
	set sub_model="qwen"
    set task="text-generation-with-past"
)
if "%1"=="mistral_v0.1_7b" (
	set hf_name="mistralai/Mistral-7B-Instruct-v0.1"
	set sub_model="mistral"
    set task="text-generation-with-past"
)
if "%1"=="mistral_v0.2_7b" (
	set hf_name="mistralai/Mistral-7B-Instruct-v0.2"
	set sub_model="mistral"
    set task="text-generation-with-past"
)    
if "%1"=="qwen_v2.0_7b" (
	set hf_name="Qwen/Qwen2-7B-Instruct"
	set sub_model="qwen"
    set task="text-generation-with-past"
)
if "%1"=="qwen_v2.5_1.5b" (
	set hf_name="Qwen/Qwen2.5-1.5B-Instruct"
	set sub_model="qwen"
    set task="text-generation-with-past"
)
if "%1"=="qwen_v2.5_7b" (
	set hf_name="Qwen/Qwen2.5-7B-Instruct"
	set sub_model="qwen"
    set task="text-generation-with-past"
)
if "%1"=="qwen_v3.0_8b" (
	set hf_name="Qwen/Qwen3-8B"
	set sub_model="qwen"
    set task="text-generation-with-past"
)
if "%1"=="qwen_v3.0_0.6b" (
	set hf_name="Qwen/Qwen3-0.6B"
	set sub_model="qwen"
    set task="text-generation-with-past"
)
if "%1"=="qwen_v3.0_30b_a3b" (
	set hf_name="Qwen/Qwen3-30B-A3B"
	set sub_model="qwen"
    set task="text-generation-with-past"
)
if "%1"=="phi_v2.0_3b" (
	set hf_name="microsoft/phi-2"
	set sub_model="phi"
    set task="text-generation-with-past"
)
if "%1"=="phi_v3.0_4b" (
	set hf_name="microsoft/Phi-3-mini-4k-instruct"
	set sub_model="phi"
    set task="text-generation-with-past"
)
if "%1"=="phi_v3.5_4b" (
	set hf_name="microsoft/Phi-3.5-mini-instruct"
	set sub_model="phi"
    set task="text-generation-with-past"
)
if "%1"=="phi_v4.0_4b" (
	set hf_name="microsoft/Phi-4-mini-instruct"
	set sub_model="phi"
    set task="text-generation-with-past"
)
if "%1"=="chatglm_v3.0_6b" (
	set hf_name="THUDM/chatglm3-6b"
	set sub_model="chatglm"
    set task="text-generation-with-past"
)
if "%1"=="minicpm_v1.0_1b" (
	set hf_name="openbmb/MiniCPM-1B-sft-bf16"
	set sub_model="minicpm"
    set task="text-generation-with-past"
)
if "%1"=="gemma_v1.0_7b" (
	set hf_name="google/gemma-7b-it"
	set sub_model="gemma"
    set task="text-generation-with-past"
)
if "%1"=="gemma_v3.0_4b" (
	set hf_name="google/gemma-3-4b-it"
	set sub_model="gemma"
    set task="image-text-to-text"
)
if "%1"=="glm_edge_4b" (
	set hf_name="THUDM/glm-edge-4b-chat"
	set sub_model="chatglm"
    set task="text-generation-with-past"
)
if "%1"=="zamba_v2.0_2.7b" (
	set hf_name="Zyphra/Zamba2-2.7B-instruct"
	set sub_model="chatglm"
    set task="text-generation-with-past"
)

