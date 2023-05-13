import json

def get_class_names():
    with(open('data/inatclasses.json') as f):
        body = json.load(f)
    class_names = [f"{desc.name}, a type of {desc.supercategory}" for desc in body]
    return class_names
