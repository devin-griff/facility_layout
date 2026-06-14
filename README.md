# Facility Layout

Plant facility layout via GDP — minimize facility size + pipe costs

**Live demo:** https://facility-layout.griffith-pse.com  
**Home:** https://griffith-pse.com

## Run locally

    pip install -r requirements.txt
    streamlit run app.py

## Deployment

Auto-deploys to Fly.io on every push to `main` via
`.github/workflows/deploy.yml`. The `Dockerfile` builds a Python 3.12 image
and installs everything from `requirements.txt`; `fly.toml` configures
auto-stop machines. Custom domain wired through Cloudflare DNS.

- **Machine**: `shared-cpu-1x` · 1 GB RAM · single region (`ord`) · `min_machines_running=0` (auto-stops on idle).
- **Cost ceiling**: ~$3.89/mo if traffic kept the VM awake 24/7. Realistic on idle-heavy demo traffic: well under $1/mo per app. Bandwidth is effectively free under Fly's 100 GB/mo egress allowance.

## Files

- `app.py` — Streamlit UI and computation
- `requirements.txt` — Python deps
- `favicon.png` — Griffith PSE blackletter G favicon
- `Dockerfile`, `fly.toml`, `.dockerignore` — Fly.io production image config
- `.streamlit/config.toml` — Streamlit defaults baked into the image
- `.github/workflows/deploy.yml` — auto-deploy pipeline

## References

[1] L. G. Papageorgiou and G. E. Rotstein, "Continuous-Domain Mathematical
Models for Optimal Process Plant Layout," *Industrial & Engineering Chemistry
Research*, vol. 37, no. 9, pp. 3631–3639, 1998.
[ACS](https://pubs.acs.org/doi/10.1021/ie980146v)

[2] J. Westerlund and L. G. Papageorgiou, "Improved Performance in Process
Plant Layout Problems Using Symmetry-Breaking Constraints," *Proc. FOCAPD 2004
(Foundations of Computer-Aided Process Design)*, 2004.
[PDF](https://skoge.folk.ntnu.no/prost/proceedings/focapd_2004/pdffiles/papers/075_46.pdf)

[3] N. W. Sawaya and I. E. Grossmann, "A Cutting Plane Method for Solving Linear
Generalized Disjunctive Programming Problems," *Computers & Chemical
Engineering*, vol. 29, no. 9, pp. 1891–1913, 2005.
[ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0098135405000992)

[4] M. L. Bynum, G. A. Hackebeil, W. E. Hart, C. D. Laird, B. L. Nicholson,
J. D. Siirola, J.-P. Watson, and D. L. Woodruff, *Pyomo — Optimization Modeling
in Python*, 3rd ed. Cham: Springer, 2021.
[Springer](https://link.springer.com/book/10.1007/978-3-030-68928-5)
