# Prerequisites only Charles can run

**Done 2026-11-08** (steps 1 to 4; step 5 partly; step 6 uploaded). Kept as the record of what was granted and why.

Paste each command as **one line**. zsh doesn't treat `#` as a comment in an interactive shell, and a backslash followed by trailing spaces from a copy doesn't continue the line. On 2026-11-08 both of these silently split commands in two.

The agent's gcloud settings deny `gcloud iam`, `gcloud projects` and firewall commands. It never widens a role itself. Run these once, from your own gcloud config, then tick them off in `status/deploy.md`. Context: `docs/deploy-wiring-plan.md`, section 7.

```bash
P=sportlight-514020; PN=251387512246; B=sportlight-514020-si-private
CI=si-web-ci@$P.iam.gserviceaccount.com
```

## 1. Let the engine repo's Actions use si-web-ci

The provider's attribute condition has to let `Asyncharles/SportlightEngine` in. Until it does, `google-github-actions/auth` fails with "The given credential is rejected by the attribute condition", even though the binding below exists. It was `assertion.repository=='Asyncharles/sportlight'`; on 2026-11-08 it became:

```bash
C="assertion.repository in ['Asyncharles/sportlight', 'Asyncharles/SportlightEngine']"
gcloud iam workload-identity-pools providers update-oidc github-sportlight --workload-identity-pool=github --location=global --project=$P --attribute-condition="$C"
gcloud iam service-accounts add-iam-policy-binding $CI --role=roles/iam.workloadIdentityUser --member="principalSet://iam.googleapis.com/projects/$PN/locations/global/workloadIdentityPools/github/attribute.repository/Asyncharles/SportlightEngine"
```

## 2. Let si-web-ci build the engine image with Cloud Build (as si-build)

The CUDA image is too big for a GitHub runner, so `image.yml` runs `gcloud builds submit` as `si-build`.

```bash
gcloud projects add-iam-policy-binding $P --member=serviceAccount:$CI --role=roles/cloudbuild.builds.editor --condition=None
gcloud projects add-iam-policy-binding $P --member=serviceAccount:$CI --role=roles/serviceusage.serviceUsageConsumer --condition=None
gcloud iam service-accounts add-iam-policy-binding si-build@$P.iam.gserviceaccount.com \
  --member=serviceAccount:$CI --role=roles/iam.serviceAccountUser
gcloud storage buckets add-iam-policy-binding gs://${P}_cloudbuild --member=serviceAccount:$CI --role=roles/storage.objectAdmin
# gcloud builds submit also reads the staging bucket itself (storage.buckets.get), which objectAdmin
# lacks; without this it fails "forbidden from accessing the bucket ... serviceusage.services.use".
gcloud storage buckets add-iam-policy-binding gs://${P}_cloudbuild --member=serviceAccount:$CI --role=roles/storage.legacyBucketReader
```

Deploys also need `compute.viewer`, `iap.tunnelResourceAccessor` and `compute.osAdminLogin`. si-web-ci already deploys si-web, so it should have them. Check with:

```bash
gcloud projects get-iam-policy $P --flatten=bindings --filter="bindings.members:$CI" --format='value(bindings.role)'
```

## 3. The GPU VM's own identity, si-gpu-sa

```bash
gcloud iam service-accounts create si-gpu-sa --display-name="Sportlight GPU VM"
G=si-gpu-sa@$P.iam.gserviceaccount.com; sleep 30
for r in artifactregistry.reader logging.logWriter monitoring.metricWriter; do
  gcloud projects add-iam-policy-binding $P --member=serviceAccount:$G --role=roles/$r --condition=None
done
gcloud secrets add-iam-policy-binding rabbitmq-app-password --member=serviceAccount:$G --role=roles/secretmanager.secretAccessor
# OS Login SSH into a VM that runs as si-gpu-sa needs serviceAccountUser on it (CI deploys, the agent)
for who in serviceAccount:$CI serviceAccount:si-deployer@$P.iam.gserviceaccount.com user:contact.charlestrolle@gmail.com; do
  gcloud iam service-accounts add-iam-policy-binding $G --member=$who --role=roles/iam.serviceAccountUser
done
```

After this, the agent runs `infra/gcp/bucket.sh`. That gives si-gpu-sa read access to everything, plus write access to `reports/`, `tapes/` and `events/`, which is the bucket access the brief allows.

## 4. Confirm the firewall

The GPU VM is tagged `si`. These rules must exist: `si-mq-internal` (tcp:5672 from tag `si` to tag `si-mq`) and `si-iap-ssh` (tcp:22 from 35.235.240.0/20 to tag `si`). No new port is opened.

```bash
gcloud compute firewall-rules list \
  --format="table(name,sourceRanges.list(),sourceTags.list(),targetTags.list(),allowed[].map().firewall_rule().list())"
```

## 5. GitHub (the agent can do this with `gh` once you say so)

On `Asyncharles/SportlightEngine`:

- Repository variables:
  - `GCP_PROJECT=sportlight-514020`
  - `GCP_REGION=europe-west4`
  - `GCP_ZONE=europe-west4-a`
  - `GCP_WIF_PROVIDER=projects/251387512246/locations/global/workloadIdentityPools/github/providers/github-sportlight`
  - `GCP_CI_SA=si-web-ci@sportlight-514020.iam.gserviceaccount.com`
  - `GCP_BUILD_SA=si-build@sportlight-514020.iam.gserviceaccount.com`
  - `AR_REPO=europe-west4-docker.pkg.dev/sportlight-514020/si`
- Environments:
  - `dev`, restricted to `master`;
  - `demo`, restricted to `master`. Required reviewers aren't available for private repos on this GitHub plan (the API refuses them, HTTP 422), so the gate is that only people with write access can dispatch `deploy-demo`.
- Secret `SPORTLIGHT_REPO_TOKEN`: a fine-grained token with read-only Contents on `Asyncharles/sportlight`, for the end-to-end CI job. **Only Charles can create the token.**

## 6. Data

- Upload `sportlight-data.zip` to `gs://$B` at the bucket-plan paths. Done 2026-11-08; the 2018 `raw/football/` files are approved by the ML lead.
- Then run `make data-verify PACKAGE=<path to the zip>`.
- Send the team's Google account emails. The agent then runs `TEAM_MEMBERS=a@x,b@y infra/gcp/bucket.sh`.
