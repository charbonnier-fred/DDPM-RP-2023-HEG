# Learning to generate molecules

Repository created by Frédéric Charbonnier & Joel Clerc of the Master in Information Sciences research project "Learning to generate molecules" at the Haute école de gestion de Genève.  
More details in the [project poster](https://github.com/charbonnier-fred/DDPM-RP-2023-HEG/blob/main/Poster_Charbonnier_Clerc.pdf).

## Annotated diffusion

[Annotated diffusion notebook](https://github.com/charbonnier-fred/DDPM-RP-2023-HEG/blob/main/Annotated_diffusion.ipynb) is to experiment and implemente the Denoising Diffusion Probabilistic Models (DDPMs) initialized by [Sohl-Dickstein et al](http://arxiv.org/abs/1503.03585), proposed by [Ho. et al](http://arxiv.org/abs/2006.11239) and improved by [Nichol, Dhariwal](http://arxiv.org/abs/2102.09672). The code used comes from [Phil Wang's GitHub](https://github.com/lucidrains/denoising-diffusion-pytorch) and [Niels Rogge and Kashif Rasul's blog post](https://huggingface.co/blog/annotated-diffusion). Theoretical support by [Karagiannakos, Adaloglou](https://theaisummer.com/diffusion-models/).

## DiffGenMol

[DiffGenMol](https://github.com/charbonnier-fred/DDPM-RP-2023-HEG/tree/main/DiffGenMol) enables a DDPM to be trained simultaneously, unconditionally and conditionally, using training data sets consisting of molecules and their calculated properties. Once the model has been trained, several molecules are generated and different metrics are calculated. 

To start training:  
cd .\DiffGenMol\
python train.py
