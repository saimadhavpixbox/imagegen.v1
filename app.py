from flask import Flask, request, jsonify,render_template,url_for
from diffusers import DiffusionPipeline, DPMSolverSinglestepScheduler
import torch
import os
from datetime import datetime
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

# Initialize the DiffusionPipeline
pipe = DiffusionPipeline.from_pretrained(
    "mann-e/Mann-E_Dreams", torch_dtype=torch.float16
).to("cuda")

# Set up the scheduler
pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_image_complete', methods=['POST'])
def generate_image_complete():
    try:
        # Extract data from the form
        prompt = request.form.get('prompt')
        width = int(request.form.get('width', 1024))
        height = int(request.form.get('height', 512))

        width = (width // 8) * 8
        height = (height // 8) * 8

        # Generate the image
        image = pipe(
            prompt=prompt,
            num_inference_steps=8,
            guidance_scale=3,
            width=width,
            height=height,
            clip_skip=1
        ).images[0]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"{timestamp}_{prompt.replace(' ', '_')}.png"
        image_path = os.path.join("static", "images", image_filename)
        image.save(image_path)

        # Get the URL for the saved image
        image_url = url_for('static', filename=f'images/{image_filename}')
        image_url="http://127.0.0.1:8080/"+image_url

        return jsonify({"message": f"{image_url}"})
    except Exception as e:
        return jsonify({"error": str(e)})
    

@app.route('/generate_image_background', methods=['POST'])
def generate_image_background():
    try:
        # Extract data from the form
        prompt = request.form.get('prompt')
        width = int(request.form.get('width', 1024))
        height = int(request.form.get('height', 512))
        width = (width // 8) * 8
        height = (height // 8) * 8

        # Generate the image
        image = pipe(
            prompt=prompt+" Create full plain background image for the poster making",
            num_inference_steps=8,
            guidance_scale=3,
            width=width,
            height=height,
            clip_skip=1
        ).images[0]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"{timestamp}_{prompt.replace(' ', '_')}.png"
        image_path = os.path.join("static", "images", image_filename)
        image.save(image_path)

        # Get the URL for the saved image
        image_url = url_for('static', filename=f'images/{image_filename}')
        image_url="http://127.0.0.1:8080/"+image_url

        return jsonify({"message": f"{image_url}"})
    except Exception as e:
        return jsonify({"error": str(e)})
    
@app.route('/generate_image_subject', methods=['POST'])
def generate_image_subject():
    try:
        # Extract data from the form
        prompt = request.form.get('prompt')
        width = int(request.form.get('width', 1024))
        height = int(request.form.get('height', 512))
        width = (width // 8) * 8
        height = (height // 8) * 8

        # Generate the image
        image = pipe(
            prompt=prompt+" create complete image with plain background and subject should be on center",
            num_inference_steps=8,
            guidance_scale=3,
            width=width,
            height=height,
            clip_skip=1
        ).images[0]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"{timestamp}_{prompt.replace(' ', '_')}.png"
        image_path = os.path.join("static", "images", image_filename)
        image.save(image_path)

        # Get the URL for the saved image
        image_url = url_for('static', filename=f'images/{image_filename}')
        image_url="http://127.0.0.1:8080/"+image_url

        return jsonify({"message": f"{image_url}"})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=False,host="0.0.0.0",port=8080)
