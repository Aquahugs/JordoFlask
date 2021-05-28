import torch

def create_model(opt):
    if opt.model == 'pix2pix':
        from .pix2pix_model import Pix2PixModel, InferenceModel
        if opt.isTrain:
            print("pix2pix")
            model = Pix2PixModel() 
        else:
            print("InferenceModel")
            model = InferenceModel()
    else:
    	from .ui_model import UIModel
    	model = UIModel()
    print("uiting")
    model.initialize(opt)
  
