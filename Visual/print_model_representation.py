import dlsca as dlsca
from tensorflow.keras.utils import plot_model
import visualkeras
from PIL import Image, ImageFont


def main():
    mlp, model_summary_str = dlsca.Models.aisy_mlp(64, 4, 100, 'categorical_crossentropy', 'relu', 'Adam')
    cnn, model_summary_str = dlsca.Models.aisy_cnn(64, 4, 100, 'categorical_crossentropy', 'relu', 'Adam')
    font = ImageFont.truetype("./Arial.ttf", 16)
    scale = 1
    img = visualkeras.layered_view(cnn, legend=True, font=font, scale_xy=scale, scale_z=scale,)

    new_size = (img.width * 4, img.height * 4)  # Adjust the multiplier as needed
    img_resized = img.resize(new_size, Image.Resampling.LANCZOS)

    # Save the resized image
    img_resized.save("cnn_high_res.png")
    #plot_model(mlp, to_file='mlp.png', show_shapes=True, show_layer_names=True, dpi=100)


if __name__ == "__main__":
    main()