# AI-generated-chemical-materials
A PyTorch implementation of generative models to generate novel chemical materials.

To run training,
1. Download multiple_lines.zip dataset at https://drive.google.com/file/d/1QpHOx9C_nBF6hrKfws5Q6hPdKKTzYXJG/view?usp=sharing.
2. Unzip multiple_lines.zip and place the multiple_lines inside the data folder in the root directory.
3. Open multiple_lines and unzip images.zip.
4. Run setup.sh by running ./setup.sh in terminal. If you're running the training for the first time, the code will
ask for a WandB API key. Enter a valid API key. This command should run the src/models/sweep.py file which contains the configuration parameters used in our tinypaper. To see our tiny paper, see toward_learning_latent_variable.pdf.
