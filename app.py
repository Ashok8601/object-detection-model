import torch
import torch.nn as nn
from flask import Flask, render_template, request
from PIL import Image
import os
from torchvision import transforms

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"

os.makedirs("uploads", exist_ok=True)

# ---------------- MODEL ----------------
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ---------------- DEVICE ----------------
device = torch.device("cpu")

model = AutoEncoder()
model.load_state_dict(torch.load("light.pth", map_location=device))
model.to(device)
model.eval()

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ---------------- THRESHOLD ----------------
THRESHOLD = 0.011  # tumhara calculated value

# ---------------- PREDICT FUNCTION ----------------
def predict_image(path):
    img = Image.open(path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        recon = model(x)

    error = torch.mean((x - recon) ** 2).item()

    if error > THRESHOLD:
        return "NOT GOOD", error
    else:
        return "GOOD", error


# ---------------- ROUTES ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    score = None
    img_path = None

    if request.method == "POST":
        file = request.files["image"]
        path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(path)

        result, score = predict_image(path)
        img_path = path

    return render_template("index.html",
                           result=result,
                           score=score,
                           img_path=img_path)


if __name__ == "__main__":
    app.run(debug=True)