# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import torch
from cog import BasePredictor, Input, Path
from transformers import AutoModel, AutoTokenizer

class Predictor(BasePredictor):
    def setup(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bertjapanese = AutoModel.from_pretrained("cl-tohoku/bert-base-japanese", cache_dir="model", local_files_only=True).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese", cache_dir="tokenizer", local_files_only=True)
        

    def predict(
        self,
        input_1: str = Input(description="Sentence 1"),
        input_2: str = Input(description="Sentence 2"),
    ) -> str:
        inputs = self.tokenizer([input_1, input_2], return_tensors="pt", padding=True).to(self.device)

        outputs = self.bertjapanese(**inputs).pooler_output

        similarity = torch.nn.CosineSimilarity(dim=1)(outputs[:1], outputs[-1:]).detach().cpu().numpy()[0]
        similarity_label = f"{similarity*100:.2f}%"

        return similarity_label
