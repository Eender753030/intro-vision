# Intro-Vision
Still working on...

### Enviroment Setup
- Get `pixi` first:
    ```bash
    # Download pixi
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

- `pixi run train`: Train the emotion detect model
- `pixi run test`: Test the emotion detect model
