
import visualkeras
from tensorflow.keras.models import load_model

# Load your model from a .keras file
model = load_model('output/10_deep_wide/best_model.keras')

visualkeras.layered_view(model, legend=True, to_file="assets/model_architecture.png")