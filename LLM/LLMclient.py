import re
import base64
import logging
from openai import OpenAI

class ChatModel:
    def __init__(self, 
                 model: str, 
                 reasoning_model: bool,
                 api_key: str | None,  
                 base_url: str | None, 
                 temperature: float=1.0
                 ):
        self.model = model
        self.reasoning_model = reasoning_model
        self.api_key = api_key or "ollama"
        self.base_url = base_url or "http://localhost:11434/v1/"
        self.temperature = temperature
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
                    model=self.model,
                    temperature=self.temperature
                )
                think_result = completion.choices[-1].message.content
                result = re.sub(r'<think>.*?</think>\s*', '', think_result, flags=re.DOTALL | re.IGNORECASE)
                return result
            except Exception as e:
                logging.error(f"An error occurred in LLM: {e}")
                return ""
        else:
            input_messages=[
                {'role': 'system', 'content': system_content}, 
                {'role': 'user', 'content': prompt}
                ]
            try:
                completion = self.client.chat.completions.create(
                    messages=input_messages,
                    model=self.model,
                    temperature=self.temperature
                )
                result = completion.choices[-1].message.content
                return result
            except Exception as e:
                logging.error(f"An error occurred in LLM: {e}")
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
                model=self.model,
                temperature=self.temperature
                )
            result = completion.choices[-1].message.content
            return result
        except Exception as e:
            logging.error(f"An error occurred in VLM: {e}")
            return ""