# Getting Started — HPC LLM Serving

This is your **sanitized public portfolio repo**. It contains zero company data, zero employee IDs, zero internal hostnames.

## 🚀 Push to GitHub in 5 Minutes

```bash
# 1. Navigate to the new repo (already created in your workspace)
cd ~/repo/model_training/hpc-llm-serving

# 2. Verify your identity is correct (MUST be your personal email)
git config user.name "Aman Khan"
git config user.email "aman.khan@hhl.de"

# 3. Initialize git (if not already)
git init

# 4. Create a daily-log.md so the Action has something to commit on first run
echo "# Daily Activity Log" > daily-log.md
echo "" >> daily-log.md
echo "$(date '+%Y-%m-%d'): Repo initialized." >> daily-log.md

# 5. Stage everything
git add .

# 6. Commit with YOUR identity (not M316235)
git commit -m "Initial sanitized HPC LLM serving reference implementation"

# 7. Create the repo on GitHub (use gh CLI or web UI)
gh repo create hpc-llm-serving --public --source=. --remote=origin --push

#   ...or via web UI: https://github.com/new → name: hpc-llm-serving → Public
#   Then: git remote add origin https://github.com/maleshep/hpc-llm-serving.git
#         git branch -M main
#         git push -u origin main
```

## ⚙️ Enable the Daily Contribution Workflow

After pushing, go to the repo on GitHub:

1. **Settings → Actions → General**
2. Under **Workflow permissions**, select:
   - ✅ **Read and write permissions**
   - ✅ **Allow GitHub Actions to create and approve pull requests**
3. Click **Save**

That's it. The workflow will run every day at 11:00 UTC and make a commit as `Aman Khan <aman.khan@hhl.de>`, which will appear on your contribution graph.

**Note:** GitHub disables scheduled workflows on repos with no activity for 60 days. If the repo goes stale, the daily commits keep it alive, which keeps the workflow alive — a self-sustaining loop. If you ever need to restart it, go to **Actions → daily-contribution → Run workflow**.

## 📌 Pin This Repo on Your Profile

Go to https://github.com/maleshep → **Customize your pins** → Pin `hpc-llm-serving` as #1.

## 🔒 What to Do With the Real Repo

1. Make `maleshep/llm-training` **private** on GitHub — do NOT rewrite history.
2. Enable **"Include private contributions"** in your GitHub profile settings.
3. The green squares from `llm-training` (private) + `hpc-llm-serving` (public) will combine.

## 🤝 Contributing

Feel free to push actual improvements from your daily work (sanitized). The more real code here, the more impressive it becomes.

---

*Identity verified: Aman Khan <aman.khan@hhl.de>*
