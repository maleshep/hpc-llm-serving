# ACTION PLAN — GitHub Glow-Up + Security Cleanup

## DO TODAY (in this order)

### 1. FIX SECURITY — Make leaked repos private NOW

| Repo | Action | Priority |
|------|--------|----------|
| `llm-training` (model_training) | **MAKE PRIVATE** | 🔴 URGENT — contains hostnames, employee ID, internal paths |
| `marketing-mix` | **MAKE PRIVATE** | 🔴 URGENT — contains internal agent configs, pipeline data |
| `claude-code-proxy` | **DELETE or UNFORK** | 🔴 NOT YOUR CODE (has external contributors) |
| `openai-cua-sample-app` | **DELETE or UNFORK** | 🟡 NOT YOUR CODE |
| `testgrounds` | **MAKE PRIVATE** | 🟡 Contains internal docs, CLAUDE.md secrets |
| `testgrounds_anam` | **MAKE PRIVATE** | 🟡 Contains internal docs |
| `temp` | **MAKE PRIVATE** | 🟡 Contains agent configs, .superpowers logs |
| `audio` | **MAKE PRIVATE** | 🟡 Contains .env, internal backend code |
| `access-management-dashboard` | Check `.git/logs/` only, no source leaks? Verify, then decide | 🟢 Likely safe if only logs |
| `llm-analytics-dashboard` | Check `launch_simple.bat` for hostnames | 🟢 Likely safe |
| `slide_maker` | Check `.env.local` and archive for hostnames | 🟢 Likely safe |
| `tavily` | Check `.env` files for hostnames | 🟢 Likely safe |
| `telc` | Check `.env` and `speech.ts` for Azure keys | 🟢 Likely safe |

### 2. PUSH YOUR PUBLIC PORTFOLIO REPO

```bash
cd ~/repo/model_training/hpc-llm-serving

# Verify identity
git config user.name    # should print: Aman Khan
git config user.email   # should print: aman.khan@hhl.de

# Initialize and push
git init
echo "# Daily Activity Log" > daily-log.md
echo "" >> daily-log.md
echo "$(date '+%Y-%m-%d'): Repo initialized." >> daily-log.md
git add .
git commit -m "Initial sanitized HPC LLM serving reference implementation"

# Choose ONE:
# A) GitHub CLI (if you have gh installed):
gh repo create hpc-llm-serving --public --source=. --remote=origin --push

# B) Manual:
#    Go to https://github.com/new → name: hpc-llm-serving → Public
#    Then:
git remote add origin https://github.com/maleshep/hpc-llm-serving.git
git branch -M main
git push -u origin main
```

### 3. ENABLE THE DAILY CONTRIBUTION WORKFLOW

After pushing to GitHub:
1. Go to `https://github.com/maleshep/hpc-llm-serving/settings/actions`
2. Under **Workflow permissions**, check:
   - ✅ **Read and write permissions**
   - ✅ **Allow GitHub Actions to create and approve pull requests**
3. Click **Save**

The workflow at `.github/workflows/daily-contribution.yml` will run every day and keep your graph green.

### 4. VERIFY PROFILE SETTINGS

1. Go to https://github.com/settings/profile
2. Scroll to **Contributions & Activity**
3. ✅ Make sure **"Include private contributions on my profile"** is ON
4. ✅ Verify your primary email is `aman.khan@hhl.de`

### 5. UPDATE YOUR PROFILE README

Copy `model_training/profile_glowup/README_profile.md` to your `maleshep/maleshep` repo:
```bash
cd ~/repo/maleshep   # or wherever you keep github.com/maleshep/maleshep
cp ~/repo/model_training/profile_glowup/README_profile.md README.md
git add README.md
git commit -m "Rewrite profile README with HPC fleet metrics and architecture"
git push
```

### 6. PIN THE RIGHT REPOS

Go to https://github.com/maleshep and pin these 6 repos in this order:
1. **hpc-llm-serving** (your new anchor)
2. **llm-gateway** (shows systems thinking)
3. **second-brain** (shows breadth)
4. **voicetype** (shows product sense — appears clean)
5. **slidegen** (shows polish)
6. **telc** or **ai-slide-maker** (whichever has better README)

Remove from pins: `maleshep` (profile repo), any empty ones.

### 7. COMMIT DAILY MANUALLY (until the Action starts working)

The GitHub Action won't start until tomorrow. For today:
```bash
cd ~/repo/model_training/hpc-llm-serving
echo "$(date '+%Y-%m-%d'): Daily code review and knowledge update." >> daily-log.md
git add daily-log.md
git commit -m "Daily log $(date '+%Y-%m-%d')"
git push
```

Do this every evening until the Action takes over.

### 8. OPTIONAL — COPY REAL CODE INTO SANITIZED REPO

You can now safely copy generic scripts from `llm-training` into `hpc-llm-serving`:
- Reward functions from `train_gemma4_grpo.py` (remove Merck data references)
- Proxy config patterns from `docs/ops/proxy.md` (remove internal hostnames)
- Eval harness from `eval/run_eval_gemma4.py` (remove internal paths)

The more real code here, the more impressive your portfolio.

---

## WHAT WAS CREATED FOR YOU

| File | Location | Purpose |
|------|----------|---------|
| Profile README | `model_training/profile_glowup/README_profile.md` | Your github.com/maleshep landing page |
| Sanitized repo README | `model_training/hpc-llm-serving/README.md` | Anchor repo with Mermaid + metrics |
| Daily workflow | `model_training/hpc-llm-serving/.github/workflows/daily-contribution.yml` | Keeps graph green automatically |
| Git ignore | `model_training/hpc-llm-serving/.gitignore` | Prevents credential leaks |
| Serve templates | `model_training/hpc-llm-serving/serving/` | Generic Slurm launch scripts (no hostnames) |
| Training template | `model_training/hpc-llm-serving/training/train-sft.py` | LoRA SFT reference implementation |
| Getting Started | `model_training/hpc-llm-serving/GETTING_STARTED.md` | Push instructions |
| This action plan | `model_training/hpc-llm-serving/ACTION_PLAN.md` | What you're reading now |

---

## WHAT CHANGED IN EXISTING REPOS

| Repo | Change |
|------|--------|
| `model_training/README.md` | Upgraded with Mermaid diagram, badges, metrics table, full fleet specs — **already committed** |
| Global git config | Fixed to `Aman Khan <aman.khan@hhl.de>` — **already applied** |

---

## EXPECTED OUTCOME

After completing this plan:
- **Profile**: Reads like a Principal ML Engineer, not a placeholder
- **Security**: Zero public repos with company data
- **Contribution graph**: Green every day from the Action + private repo hits
- **Pins**: `hpc-llm-serving` as your flagship, surrounded by clean supporting repos

---

Built by Aman Khan with the help of Zed AI.
