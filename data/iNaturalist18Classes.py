import json

def get_class_names():
    with open('data/inatclasses.json', 'r') as f:
        body = json.load(f)
    class_names = [f"a photo of a {desc['name']}" for desc in body] # , a type of {desc['supercategory']}
    return class_names
