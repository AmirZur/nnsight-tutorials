from collections import OrderedDict
import torch

class MysteryTokenizer:
    def __init__(self):
        self.entity_tokens = ["Paris", "Rome", "Seattle"]
        self.property_tokens = ["language", "food", "country", "?"]
        self.output_tokens = ["France", "French", "steak", "Eiffel", "Italy", "Italian", "pizza", "Pisa", "USA", "English", "salmon", "Space Needle"]

    def tokenize(self, text, type="input"):
        if type == "output":
            return self.output_tokens.index(text)
        if isinstance(text, str):
            text = text.split()
        assert text[0] in self.entity_tokens, f"{text[0]} is not a city"
        assert text[1] in self.property_tokens, f"{text[1]} is not a property of a city"
        return [self.entity_tokens.index(text[0]), self.property_tokens.index(text[1])]
    
    def decode(self, token_ids, type="output"):
        if type == "output":
            token_list = self.output_tokens
        elif type == "entity":
            token_list = self.entity_tokens
        else: # type is property
            token_list = self.property_tokens
        if isinstance(token_ids, int):
            return token_list[token_ids] if 0 <= token_ids < len(token_list) else "<unknown>"
        return [token_list[token_id] if 0 <= token_ids < len(token_list) else "<unknown>" for token_id in token_ids]

class MysteryModel(torch.nn.Module):
    def __init__(self, n_entities, n_properties, hidden_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_entities = n_entities
        self.n_properties = n_properties
        self.hidden_dim = hidden_dim
        self.tokenizer = MysteryTokenizer()
        self.entity_embed = torch.nn.Linear(n_entities, hidden_dim, bias=False)
        self.property_embed = torch.nn.Linear(n_properties, hidden_dim, bias=False)
        self.hidden_layer = torch.nn.Sequential(OrderedDict(
            linear=torch.nn.Linear(hidden_dim * 2, hidden_dim),
            activation=torch.nn.ReLU()
        ))
        self.out_head = torch.nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, inputs):
        token_ids = self.tokenizer.tokenize(inputs)
        
        entity_embed = self.entity_embed(torch.eye(self.n_entities)[int(token_ids[0])])
        property_embed = self.property_embed(torch.eye(self.n_properties)[int(token_ids[1])])

        embedding = torch.cat([entity_embed, property_embed])

        output = self.out_head(self.hidden_layer(embedding))

        return self.tokenizer.decode(round(output.item()))

def construct_mystery_model():
    mystery_model = MysteryModel(n_entities=3, n_properties=4, hidden_dim=6)
    mystery_model.entity_embed.weight.data = torch.tensor([
        [0, 4, 6, 10, 0, 8],
        [0, 12, 14, 18, 0, 16],
        [0, 20, 22, 26, 0, 24],
    ], dtype=torch.float32).T

    mystery_model.property_embed.weight.data = torch.tensor([
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ], dtype=torch.float32).T

    mystery_model.hidden_layer.linear.weight.data = torch.cat([0.5 * torch.eye(6), 99 * torch.eye(6)], dim=1)

    mystery_model.hidden_layer.linear.bias.data = torch.tensor([
        -101, -101, -101, -101, -101, -101
    ], dtype=torch.float32)

    mystery_model.out_head.weight.data = torch.tensor([
        [1, 1, 1, 1, 1, 1]
    ], dtype=torch.float32)

    return mystery_model