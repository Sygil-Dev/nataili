from nataili.postprocessor import *


class codeformers(PostProcessor):
    def set_filename_append(self):
        self.filename_append = "codeformers"

    def process(self, img, img_array, **kwargs):
        output_image = self.model(img)
        return output_image
