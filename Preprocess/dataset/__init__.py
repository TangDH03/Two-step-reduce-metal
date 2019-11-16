from .DeepLesion import DeepLesion
def get_dataset(** dataset_opts):
    return DeepLesion(**dataset_opts["deep_lesion"])