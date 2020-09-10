VAE and SDVAE have things in common, so it was made a class VAEBase (vaeBase.py) and VAE (vae.py) and SDVAE (sdvae.py) are inherited class. 

To train this models, a class Generator (generator.py) is used. After each epoch, the class Callback is executed (callback.py) to evaluate the model in validation or test trajectories. The losses functions are in losses.py. In metrics.py are the functions to compute mADE and mFDE. The metrics IPA and <img src="https://render.githubusercontent.com/render/math?math=\sigma_{l'}"> are in the crossroad notebook results.

The autoencoder structor for crossroad patches is in autoencoder_patches.py. In that file it is its generator (PatchesDataGenerator) as well.
