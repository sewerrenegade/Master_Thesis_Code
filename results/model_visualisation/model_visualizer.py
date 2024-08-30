import sys
sys.path.append('/home/milad/Desktop/Master_Thesis/code/Master_Thesis_Code')
from models.SCEMILA.SCEMILA_model import AMiL
from models.SCEMILA.topo_SCEMILA_model import TopoAMiL
from torchsummary import summary
from torchviz import make_dot
import torch

different_encoders = {
    "images": (1,400,3,144,144),
    "gray_images": (1,400,1,144,144),
    "fnl34": (1,400,512,5,5),
    "dino_bloom_small":(1,400,384)
    
}

if __name__ == '__main__':
    mode = "images"
    model = AMiL(class_count= 5, multicolumn=True,input_type='images',device= "cuda:0")
    summary(model, input_size=(3, 144, 144))



    input_tensor = torch.rand(different_encoders[mode]).to("cuda")
    output = model(input_tensor)
    dot = make_dot(output, params=dict(model.named_parameters()))
    dot.format = 'png' #or pdf
    dot.render("model_architecture")


# this is all bad///, both somehow dont take into considertation the forward pass logic