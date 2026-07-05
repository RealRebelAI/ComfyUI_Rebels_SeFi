Optional but recommended: drop the official VAE config.json files here for exact values:
  flux2.json   (from the SeFi repo's vae/config.json, or any FLUX.2 diffusers repo)
  flux1.json   (from any FLUX.1 diffusers repo's vae/config.json)
Config resolution order for a VAE picked as a single file:
  1) <filename>.json next to the vae file
  2) this folder: <filename-stem>.json, then flux2.json / flux1.json by detected family
  3) architecture derived from the weight shapes (defaults for non-derivable fields)
