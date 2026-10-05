# Deploying ScopeGuard to Render

The repo includes a `Dockerfile` that runs `uvicorn main:app` on the `PORT` Render provides.
The API key is set in Render's dashboard, never in the repo.

## 1. Push the repo to GitHub
Make sure the repo is public or connected to Render's GitHub integration.

## 2. Create the service
1. Go to https://dashboard.render.com and click **New +** → **Web Service**.
2. Choose **Build and deploy from a Git repository** and select the ScopeGuard repo.
3. Fill in the form:
   - **Name:** `scopeguard` (any name works)
   - **Runtime:** **Docker** (Render detects the `Dockerfile` at the repo root)
   - **Branch:** `main`
   - **Instance type:** Free is enough for a demo
4. Leave Docker command, build command and start command blank. The Dockerfile's `CMD` is used.

## 3. Add environment variables
Still on the create form (or later under **Environment** in the service's left menu), add:

| Key | Value |
|---|---|
| `LLM_API_KEY` | your Google AI Studio key (mark it as a secret) |
| `LLM_MODEL` | `gemini-2.5-flash` (or the current Gemini model name from AI Studio) |
| `LLM_BASE_URL` | `https://generativelanguage.googleapis.com/v1beta/openai` |

For Groq instead, use:
- `LLM_BASE_URL` = `https://api.groq.com/openai/v1`
- `LLM_MODEL` = `llama-3.3-70b-versatile`
- `LLM_API_KEY` = your Groq key

Do not set `PORT`. Render sets it automatically.

## 4. Deploy
Click **Create Web Service**. The first build takes a few minutes. When the log shows
`Uvicorn running`, the service is live.

## 5. Verify
Replace `YOUR-SERVICE` with your Render URL:

```
curl https://YOUR-SERVICE.onrender.com/api/health
```

You should see `"llm_configured": true`. Then open `https://YOUR-SERVICE.onrender.com` in a browser
and press **Load bakery demo**, then **Check the chat**.

## Notes
- The free tier sleeps after about 15 minutes idle. The first request after that can take ~30–60 seconds.
- Every push to `main` triggers an automatic redeploy (unless auto-deploy is turned off in settings).
- `.env` is in `.gitignore`, so your local key is never uploaded. Only the Render dashboard holds it.
- Demo results are cached in `data/cache.json`, so replayed demo messages work even if the LLM is rate limited.
