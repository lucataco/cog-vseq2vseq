# motexture / vseq2vseq Cog model

This is an implementation of the [motexture/vseq2vseq](https://huggingface.co/motexture/vseq2vseq) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights:

    cog run script/download-weights

Then, you can run predictions:

    cog predict -i prompt="A stormtrooper surfing on the ocean"

## Example:

"A stormtrooper surfing on the ocean"

![alt text](output.mp4)