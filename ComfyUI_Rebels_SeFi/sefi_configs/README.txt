Drop the official sefi_config.yaml files here, named <scale>-<family>.yaml:
  5b-base.yaml    (from SeFi-Image/SeFi-Image-5B-Base)
  5b-turbo.yaml   (from SeFi-Image/SeFi-Image-5B-turbo)
The loader reads delta_t / timestep_shift_alpha / semantic_channels from these
automatically when no yaml sits next to the model file itself.
