from fastai.vision.all import PILImage

def predict(model, img_path):
    img = PILImage.create(img_path)
    pred_class, pred_idx, probs = model.predict(img)
    return pred_class
