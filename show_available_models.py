# test_download_models
from nataili.model_manager import ModelManager

# TODO: huggingface_hub or some way to use token instead of username/password
mm = ModelManager()

filtered_models = mm.get_filtered_models(type="ckpt")
ppmodels = ""
for model_name in filtered_models:
    ppmodels += model_name
    if filtered_models[model_name].get("description"):
        ppmodels += f" : {filtered_models[model_name].get('description')}"
    ppmodels += "\n"
print(f"## Known ckpt Models ##\n{ppmodels}")
input("Press ENTER to continue")
