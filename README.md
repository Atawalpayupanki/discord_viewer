# Discord Data Viewer

A Streamlit application to visualize your Discord data export locally.

## Privacy & Security

*   **Local Use (Recommended):** When running locally with `run_app.bat`, **all data stays on your computer**. The Python script runs on your machine, and your files never leave your hard drive.
*   **Cloud Deployment:** If you deploy this code to Streamlit Cloud or another server, your data **will be uploaded** to that server to be processed. For strict privacy, run this application locally.

## Setup

1.  Run `run_app.bat` to install dependencies and launch the app.
2.  Upload your Discord Data ZIP file.

## Pushing to GitHub

Since you don't have the GitHub CLI installed, follow these steps to upload this code to GitHub:

1.  Go to [GitHub.com](https://github.com) and sign in.
2.  Click the **+** icon in the top right and select **New repository**.
3.  Name the repository (e.g., `discord-data-viewer`).
4.  Do **not** initialize with README, .gitignore, or License (we already have them locally).
5.  Click **Create repository**.
6.  Copy the commands under "…or push an existing repository from the command line". It will look like this:

    ```bash
    git remote add origin https://github.com/YOUR_USERNAME/discord_viewer.git
    git branch -M main
    git push -u origin main
    ```

7.  Open a terminal in this folder and paste those commands.
