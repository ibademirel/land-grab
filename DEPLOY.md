# Putting Land Grab online (Streamlit Community Cloud)

Streamlit Community Cloud is free. It runs `app.py` straight from a GitHub repository and gives you a link such as
`https://your-app-name.streamlit.app` that your friends can open on their phones.

## 1. Create an empty repository on GitHub

1. Sign in at https://github.com.
2. Click **+** (top right) → **New repository**.
3. Name it, for example `land-grab`.
4. Choose **Public** (the free Streamlit plan works with public repositories; a private one needs you to allow access).
5. Do **not** tick "Add a README", ".gitignore" or a licence. The repository must be empty.
6. Click **Create repository**.

## 2. Push the project from your Mac

Open Terminal and run these commands one after another. Replace
`YOUR-USERNAME` with your GitHub username and `land-grab` with the repository name if you chose another.

```bash
cd /Users/iremdemirel/Desktop/land-grab
git init
git add .
git status
git commit -m "Land Grab route planner"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/land-grab.git
git push -u origin main
```

`git status` shows what is about to be uploaded. You should see `app.py`, `land_grab/`, `tests/`, `requirements.txt`,
`.streamlit/config.toml` and `DEPLOY.md`. You should NOT see `.venv`, `cache`, `my_route.*` or `demo_*`.

The first push asks you to sign in. GitHub no longer accepts your account password in Terminal: use a
**personal access token** (GitHub → Settings → Developer settings → Personal access tokens → generate one with the
`repo` permission) as the password, or install the GitHub CLI and run `gh auth login`.

## 3. Create the app on Streamlit

1. Go to https://share.streamlit.io and sign in with GitHub (allow it to see your repositories).
2. Click **Create app** / **New app** and choose to deploy from a GitHub repository.
3. **Repository:** `YOUR-USERNAME/land-grab`
4. **Branch:** `main`
5. **Main file path:** `app.py`
6. Under **Advanced settings**, pick **Python 3.13** (`requirements.txt` was tested with 3.13; 3.12 should also work).
7. Optionally choose your own **App URL**, then click **Deploy**.

The first start takes a few minutes while it installs the packages. After that, send your friends the link.

## What to expect

- **The first route for a new area is slow** (up to a minute or two): the street map is downloaded from OpenStreetMap.
  Asking again for the same place and distance is instant, but the server forgets its downloads whenever it
  restarts or goes to sleep, so a place may need downloading again.
- **Distance limits.** The free server has about 1 GB of memory. The web app plans up to 10 km for a run or walk and
  15 km for a ride (see `MAX_KM` in `land_grab/web.py`); beyond that it shows a friendly message. The command line
  has no limit.
- **Sleeping.** A free app that nobody visits for a few days goes to sleep; the next visitor presses "Wake up".
- **Updating.** After changing the code, run `git add .`, `git commit -m "what changed"` and `git push`. The app
  redeploys by itself.
- **If the app crashes with "out of memory"**, lower the numbers in `MAX_KM`.
