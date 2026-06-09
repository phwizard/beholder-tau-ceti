# Zenodo setup (one-time, ~5 minutes)

Zenodo is a free, CERN-operated, open-access research data repository. Connecting it to GitHub gives every release a citable DOI.

## Steps

1. **Create a Zenodo account** at https://zenodo.org/login. You can sign in with your GitHub account — no separate password needed.

2. **Authorise GitHub integration**: go to https://zenodo.org/account/settings/github/. Click "Sync now" if needed.

3. **Enable Zenodo for this repository**: on the GitHub repositories list, flip the toggle for `phwizard/beholder-tau-ceti` to **ON**.

4. **Create a GitHub release** on the public repo:
   ```bash
   gh release create v1.0.0 --title "RNAAS submission v1.0.0" --notes "Initial submission to RNAAS"
   ```
   Zenodo will detect the release and mint a DOI within ~30 seconds.

5. **Find the DOI** at the Zenodo project page; it looks like `10.5281/zenodo.xxxxxxx`. Copy that DOI into the RNAAS submission's "Data Availability" or "Supplementary Materials" field.

## Optional but recommended

- Add a "Cite this repository" button to the GitHub repo: GitHub now reads `CITATION.cff` files. Once you have the DOI, add one to make citation copy-paste easy for readers.
- `.zenodo.json` (already created in this repo) controls the metadata Zenodo uses. Update author affiliations, ORCIDs, and keywords there before the release.

## What I (Claude) cannot do for you

- Creating the Zenodo account (it's tied to your identity).
- Authorising the GitHub integration (you must click the consent button).
- Pressing the "release" button if you'd rather review the repo state first.

I can prepare `.zenodo.json`, `CITATION.cff`, and the GitHub release notes — those are all done. The remaining action is yours to take.
