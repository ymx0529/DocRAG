import re
import base64
from openai import OpenAI

# 智谱
max_token_count = 32000
model="glm-4.5" 
tokenizer = "zai-org/GLM-4.5"
reasoning_model = False
embedding_model = "Qwen/Qwen3-Embedding-0.6B"
vl_model = "glm-4.5v"
api_key="04bd903ba5f54360bacd46f3a9f63edd.nmXLF9exPcONh607"
base_url="https://open.bigmodel.cn/api/paas/v4/"

# # 本地 Ollama 模型配置
# max_token_count = 32768
# model="qwen3:30b-a3b-instruct-2507-fp16"            # ollama 模型名称
# tokenizer = "Qwen/Qwen3-30B-A3B-Instruct-2507"      # huggingface 模型名称, 用于计算 token 数, 要和 model 匹配。
# reasoning_model = False                             # 是否是推理模型，有无</think>符号
# embedding_model = "Qwen/Qwen3-Embedding-0.6B"       # 用于获取语义向量，做相似度检索，无需与 LLM 匹配
# vl_model="qwen2.5vl:32b"                            # 多模态大模型
# api_key="ollama"                                    # 本地 ollama 服务的 api key
# base_url="http://localhost:11434/v1/"               # ollama 服务地址

class ChatModel:
    def __init__(self, 
                 model: str, 
                 reasoning_model: bool,
                 api_key: str | None, 
                 base_url: str | None
                 ):
        self.model = model
        self.reasoning_model = reasoning_model
        self.api_key = api_key or "ollama"
        self.base_url = base_url or "http://localhost:11434/v1/"
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
    
    def _encode_image_to_base64(self, image_path):
        """将图片文件转换为base64编码字符串"""
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
        ret = base64.b64encode(image_data).decode('utf-8')

        return ret

    def chat_with_llm(self, 
                      prompt="Please provide clear and accurate answers.", 
                      system_content="You are a helpful assistant.", 
                      ):
        """与 LLM 进行对话"""
        if self.reasoning_model:
            input_messages=[
                {'role': 'system', 'content': system_content}, 
                {'role': 'user', 'content': prompt + "/no_think"}
                ]
            try:
                completion = self.client.chat.completions.create(
                    messages=input_messages,
                    model=self.model
                )
                think_result = completion.choices[-1].message.content
                result = re.sub(r'<think>.*?</think>\s*', '', think_result, flags=re.DOTALL | re.IGNORECASE)
                return result
            except Exception as e:
                print(f"An error occurred in LLM: {e}")
                return ""
        else:
            input_messages=[
                {'role': 'system', 'content': system_content}, 
                {'role': 'user', 'content': prompt}
                ]
            try:
                completion = self.client.chat.completions.create(
                    messages=input_messages,
                    model=self.model
                )
                result = completion.choices[-1].message.content
                return result
            except Exception as e:
                print(f"An error occurred in LLM: {e}")
                return ""
        
    def chat_with_mullm(self, 
                        image_paths: list[str],  # 支持多个图片路径
                        prompt="Describe these images.", 
                        system_content="You are a helpful assistant."
                        ):
        """与多模态 LLM 进行对话"""
        # 构建图片内容部分
        image_contents = [{"type": "image_url", "image_url": 
                        {"url": f"data:image/jpeg;base64,{self._encode_image_to_base64(path)}"}} 
                        for path in image_paths
                        ]
        input_messages = [
            {'role': 'system', 'content': system_content}, 
            {'role': 'user', 'content': [{"type": "text", "text": prompt}] + image_contents}
            ]
        try:
            completion = self.client.chat.completions.create(
                messages=input_messages,
                model=self.model
                )
            result = completion.choices[-1].message.content
            return result
        except Exception as e:
            print(f"An error occurred in VLM: {e}")
            return ""
        

chatLLM = ChatModel(model=model,
                    reasoning_model=reasoning_model, 
                    api_key=api_key, 
                    base_url=base_url)

# 测试
response = chatLLM.chat_with_llm(prompt="你是谁？", 
                                 system_content="你是一个有帮助的助手。")

print(response)