from torchio import Compose, RandomAffine, RandomElasticDeformation, RandomNoise, RandomBlur, Pad

train_transforms_dict = {
    #RandomAffine(): 0.05,
    #RandomElasticDeformation(max_displacement=3): 0.15,
    RandomNoise(std=(0,0.1)): 0.10,
    #RandomBlur(std=(0,0.1)): 0.10,
    Pad(64):1
}
train_transform = Compose(train_transforms_dict)