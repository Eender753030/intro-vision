# Intro-Vision
Still working on...

### Enviroment Setup
- Get `pixi` first:
    ```bash
    # Download and install pixi
    curl -fsSL https://pixi.sh/install.sh | bash
    # Update environment variable
    source ~/.bashrc
    # Check installation is valid
    pixi --version
    ```
- Build environment
    ```bash
    # This will download all requiremented packages
    pixi install
    ```

### Commands
You can check `tasks` in `pixi.toml` for commands detail.

- `pixi run emo_train`: Train the emotion detect model.
- `pixi run emo_test`: Test the emotion detect model.
- `pixi run emo_infer <input_path>`: Use emotion detect model to inference your input image. Need path argument.
