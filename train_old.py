from results.model_visualisation.instance_bag_SCEMILA_visulaizer import get_labeled_dinobloom_encodings
import numpy as np

enc, lab = get_labeled_dinobloom_encodings()
np.savez("data/SCEMILA/dinbloomS_labeled.npz",embedding  = enc,labels = lab)