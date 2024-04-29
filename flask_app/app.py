from flask import Flask, render_template, request
from flask_app.config import Config
from transformers import SamModel, SamConfig, SamProcessor
import torch
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import base64


app = Flask(__name__)
app.config.from_object(Config)

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = app.config['DEVICE']

# loads the model and processor from this link https://huggingface.co/batuhandumani/zerosam
my_wase_model = SamModel.from_pretrained(app.config['MODEL_NAME'])
processor = SamProcessor.from_pretrained(app.config['MODEL_NAME'])
colors = app.config['COLORS']
bounds = app.config['BOUNDS']
cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(bounds, cmap.N)

my_wase_model.to(device)

def predict_image(image):
    test_image = Image.open(io.BytesIO(image))
    prompt = [0, 0, test_image.width, test_image.height]

    prompt = [[prompt]]  
    inputs = processor(test_image, input_boxes=prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    my_wase_model.eval()

    with torch.no_grad():
        outputs = my_wase_model(**inputs, multimask_output=False)

    fedsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
    fedsam_seg_prob = fedsam_seg_prob.cpu().numpy().squeeze()
    fedsam_seg = (fedsam_seg_prob > 0.5).astype(np.uint8)

    return image, fedsam_seg, fedsam_seg_prob



@app.route('/')
def upload_file():
    return render_template('upload.html')


@app.route('/result', methods=['POST'])
def result():
    files = request.files.getlist('files[]')
    if files:
        predictions = []
        for file in files:
            image = file.read()
            test_image, fedsam_seg, fedsam_seg_prob = predict_image(image)
            
            test_image_b64 = base64.b64encode(test_image).decode('utf-8')

            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(fedsam_seg, cmap='gray')
            ax[0].axis('off')
            # ax[1].imshow(medsam_seg_prob, cmap='gray')
            ax[1].imshow(fedsam_seg_prob, cmap=cmap, norm=norm)
            ax[1].axis('off')
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            buf.seek(0)
            plt.close(fig)

            fedsam_seg_prob_img_b64 = base64.b64encode(buf.read()).decode('utf-8')
            predictions.append((test_image_b64, fedsam_seg_prob_img_b64))

        return render_template('result.html', predictions=predictions)




if __name__ == '__main__':
    app.run(debug=app.config['DEBUG'], host=app.config['HOST'], port=app.config['PORT'])