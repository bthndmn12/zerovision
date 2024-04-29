
# Zerovision

This is a Flask-based web application for performing image segmentation using a pre-trained model. The application allows users to upload images, performs segmentation, and displays the results interactively.

## Setup and Installation

1. Clone the repository:
```bash 
   git clone https://github.com/bthndmn12/zerovision
   cd zerovision
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask application:
   ```bash
   cd flask_app
   python app.py
   ```

4. Open a web browser and go to [http://localhost:5000](http://localhost:5000) to access the application.

## Usage

- Upon accessing the application, users will see an interface to upload images.
- Users can upload one or more images through the interface.
- After uploading, the application performs semantic segmentation on the images using a pre-trained model.
- The segmentation results are displayed to the users, showing the original image along with its segmented version.
- Segmented regions are highlighted with different colors based on the segmentation results.

## Dependencies

- Flask: 3.0.3
- transformers: 4.36.2
- torch: 2.1.2+cu121
- numpy: 1.24.3
- Pillow: 9.3.0
- matplotlib: 3.8.2

## License

[MIT License](LICENSE)