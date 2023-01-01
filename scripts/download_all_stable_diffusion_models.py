from nataili import ModelManager, logger

mm = ModelManager()

for model in mm.compvis.models:
    logger.info(f"Downloading {model}...")
    mm.compvis.download_model(model)
